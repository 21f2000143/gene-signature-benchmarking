# The code combines repeated cross-validation with inner cross-validation, but it does NOT actually perform bootstrap confidence intervals. 
# Fixed version: haven't included in the current results
"""Within-cohort nested-CV benchmark: c-index per (cohort x gene set x model).

Arms: gene-only, clinical-only, gene+clinical.
Metrics: Harrell c-index, Uno's c, integrated Brier score.

Uncertainty is reported at TWO distinct levels — do not conflate them:

  1. Repeated-CV fold spread (`fold_cindex_lo/hi`):
     the 2.5th/97.5th percentile of the per-fold c-index values produced by
     N_REPEATS x N_SPLITS outer folds. This reflects fold-to-fold variability
     (which patients land in which fold, small-sample noise) but is NOT a
     bootstrap CI, and outer folds are not independent draws, so these bounds
     can be mis-calibrated.

  2. True bootstrap CI (`boot_cindex_lo/hi/mean`):
     out-of-fold risk predictions are pooled across all outer folds and
     repeats for a given (cohort, gene_set, arm, model), then B_BOOT
     bootstrap resamples (with replacement, patient-level) are drawn from
     that pooled set and the c-index is recomputed on each resample. The
     2.5th/97.5th percentiles of that resampled statistic distribution are
     the bootstrap CI. This is the standard nonparametric bootstrap and is
     what the original docstring claimed but the original code did not
     implement.
"""
from __future__ import annotations
import os, sys, json, time, warnings, itertools
# warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            integrated_brier_score)
from sksurv.util import Surv

HARM = os.environ.get("HARM", "./harmonised")
OUT  = os.environ.get("OUT", "./bench2")
os.makedirs(OUT, exist_ok=True)
SEED = 42
N_SPLITS = 5
N_REPEATS = int(os.environ.get("N_REPEATS", "4"))
NJOBS = int(os.environ.get("NJOBS", "40"))
B_BOOT = int(os.environ.get("B_BOOT", "1000"))          # bootstrap resamples per group
MIN_BOOT_EVENTS = int(os.environ.get("MIN_BOOT_EVENTS", "3"))  # min events per resample to keep it

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

with open(os.environ.get("GS", "results/gene_sets.json")) as _f:
    GENE_SETS = json.load(_f)
CLIN_NUM = ["age", "size", "node_num"]
CLIN_CAT = ["grade", "er", "pr", "stage_c"]

# ── models ────────────────────────────────────────────────────────────────
def model_grid(n_feat: int):
    """(name, estimator, param_grid) — grids kept small; alpha paths dominate."""
    return [
      ("CoxPH-ridge", CoxPHSurvivalAnalysis(ties="efron"),
        {"est__alpha": [0.01, 0.1, 1.0, 10.0]}),
      # alphas below is a fallback only — eval_fold overrides it per outer-train
      # fold with a data-scaled path via _coxnet_alpha_grid().
      ("Coxnet", CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=100000),
        {"est__alphas": [[a] for a in (1.0, 0.5, 0.1, 0.05, 0.01)]}),
      ("RSF", RandomSurvivalForest(n_estimators=300, min_samples_leaf=15,
                                   max_features="sqrt", random_state=SEED, n_jobs=1),
        {"est__min_samples_leaf": [10, 25]}),
      ("GBSA", GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.1,
                                                max_depth=2, subsample=0.8, random_state=SEED),
        {"est__learning_rate": [0.05, 0.1]}),
      # tol loosened from 1e-5: at 1e-5, scipy's newton-cg line search often can't
      # certify convergence past float64 roundoff on this problem scale even when
      # the fit has effectively converged, which is what produced the
      # "precision loss" ConvergenceWarning.
      ("FastSVM", FastSurvivalSVM(max_iter=200, tol=1e-4, random_state=SEED),
        {"est__alpha": [0.5, 1.0, 4.0]}),
    ]

def build_X(expr, surv, genes, use_genes, use_clin):
    blocks, num_cols, cat_cols = [], [], []
    if use_genes:
        g = [x for x in genes if x in expr.columns]
        if not g: return None, None, None
        blocks.append(expr[g]); num_cols += g
    if use_clin:
        cl = surv.copy()
        cl["node_num"] = pd.to_numeric(cl["node"], errors="coerce")
        cl["stage_c"]  = cl["stage"].astype(str)
        nn = [c for c in CLIN_NUM if c in cl.columns and pd.to_numeric(cl[c], errors="coerce").notna().sum() > 0.3*len(cl)]
        cc = [c for c in CLIN_CAT if c in cl.columns and cl[c].astype(str).nunique() > 1 and cl[c].notna().sum() > 0.3*len(cl)]
        if not nn and not cc: return None, None, None
        sub = cl[nn+cc].copy()
        for c in nn: sub[c] = pd.to_numeric(sub[c], errors="coerce")
        for c in cc: sub[c] = sub[c].astype(str)
        blocks.append(sub); num_cols += nn; cat_cols += cc
    if not blocks: return None, None, None
    X = pd.concat(blocks, axis=1)
    return X, num_cols, cat_cols

def make_pre(num_cols, cat_cols):
    tf = []
    if num_cols:
        tf.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler())]), num_cols))
    if cat_cols:
        tf.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                    ("oh", OneHotEncoder(handle_unknown="ignore",
                                                         min_frequency=0.05, sparse_output=False))]), cat_cols))
    return ColumnTransformer(tf, remainder="drop")

from contextlib import contextmanager

@contextmanager
def _suppress_known_divergence_warnings():
    """Silence warnings that are the expected byproduct of an
    already-handled diverging/under-fit model (extreme alpha grid points,
    near-zero training events in a fold): Coxnet non-convergence or
    numerical blow-up, the resulting non-finite CV scores, and the
    downstream risk-set/baseline-hazard overflow those blown-up
    coefficients cause in sksurv's Cox fit/predict/survival-function code.
    error_score=np.nan and the surrounding try/except already turn these
    into a skipped grid point or a NaN metric — this just stops them from
    also being printed N_REPEATS x N_SPLITS x n_grid times per run. Any
    warning type not listed here still surfaces normally.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FitFailedWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                 message="overflow encountered in exp")
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                 message="overflow encountered in power")
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                 message="divide by zero encountered in divide")
        warnings.filterwarnings("ignore", category=UserWarning,
                                 message="One or more of the test scores are non-finite")
        warnings.filterwarnings("ignore", category=UserWarning,
                                 message="all coefficients are zero.*")
        yield

def _coxnet_alpha_grid(Xtr_pre, y_tr, l1_ratio, max_iter, n_alphas=5):
    """Derive a per-fold Coxnet alpha path instead of reusing one fixed list
    across every cohort. A fixed alpha is not the same regularization
    strength on an 88-sample cohort (GSE20711) as on a 3069-sample one
    (SCANB_GSE96058) — too small for the former and it diverges; safe for
    the latter. Fitting alphas=None lets sksurv compute alpha_max (the
    smallest alpha that zeroes every coefficient) from this fold's own data
    and derive n_alphas points down to alpha_min_ratio='auto' * alpha_max —
    scaled to the fold, so the weak end of the path is never smaller than
    what's numerically safe for it.
    """
    path = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio="auto",
                                  n_alphas=n_alphas, max_iter=max_iter)
    with _suppress_known_divergence_warnings():
        path.fit(Xtr_pre, y_tr)
    return [[a] for a in path.alphas_]

def eval_fold(X, y, tr, te, name, est, grid, num_cols, cat_cols, times):
    """Fit on the outer-train fold (with inner CV for hyperparameter selection),
    score on the outer-test fold, and also return the raw out-of-fold risk
    predictions (+ their event/time) so the caller can pool them across folds
    and repeats for a downstream bootstrap CI."""
    pipe = Pipeline([("pre", make_pre(num_cols, cat_cols)), ("est", est)])
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    ev_tr = y["event"][tr].astype(int)
    if name == "Coxnet":
        try:
            Xtr_pre = pipe.named_steps["pre"].fit_transform(X.iloc[tr])
            grid = {"est__alphas": _coxnet_alpha_grid(Xtr_pre, y[tr], est.l1_ratio, est.max_iter)}
        except Exception:
            pass  # keep the fixed grid passed in as a fallback for this fold
    try:
        with _suppress_known_divergence_warnings():
            gs = GridSearchCV(pipe, grid, cv=list(inner.split(X.iloc[tr], ev_tr)),
                              n_jobs=1, error_score=np.nan, refit=True)
            gs.fit(X.iloc[tr], y[tr])
            best = gs.best_estimator_
            risk = best.predict(X.iloc[te])
    except Exception as e:
        return dict(model=name, cindex=np.nan, uno_c=np.nan, ibs=np.nan, err=str(e)[:120],
                    oof_idx=None, oof_risk=None)
    ev_te, t_te = y["event"][te], y["time"][te]
    if ev_te.sum() < 3:
        return dict(model=name, cindex=np.nan, uno_c=np.nan, ibs=np.nan, err="too few test events",
                    oof_idx=None, oof_risk=None)
    ci = concordance_index_censored(ev_te, t_te, risk)[0]
    uno = np.nan
    try:
        tau = min(t_te[ev_te].max(), y["time"][tr][y["event"][tr]].max())
        uno = concordance_index_ipcw(y[tr], y[te], risk, tau=tau)[0]
    except Exception:
        pass
    ibs = np.nan
    try:
        lo = max(t_te.min(), y["time"][tr].min()) + 1e-3
        hi = min(t_te.max(), y["time"][tr].max()) - 1e-3
        tt = np.array([t for t in times if lo < t < hi])
        if len(tt) >= 2 and hasattr(best, "predict_survival_function"):
            with _suppress_known_divergence_warnings():
                sf = best.predict_survival_function(X.iloc[te])
                P = np.vstack([[fn(t) for t in tt] for fn in sf])
            ibs = integrated_brier_score(y[tr], y[te], P, tt)
    except Exception:
        pass
    # oof_idx/oof_risk carry the outer-test-fold positions and their predicted
    # risk scores back to the caller, purely for pooled bootstrap CI use below.
    return dict(model=name, cindex=ci, uno_c=uno, ibs=ibs, err="",
                oof_idx=np.asarray(te), oof_risk=np.asarray(risk))

def bootstrap_cindex_ci(event, time, risk, B=B_BOOT, seed=SEED, min_events=MIN_BOOT_EVENTS):
    """True nonparametric bootstrap CI for Harrell's c-index.

    Resamples (event, time, risk) triples WITH REPLACEMENT B times, recomputes
    the concordance index on each resample, and returns the 2.5th/97.5th
    percentiles of that distribution plus its mean. This is what actually
    estimates sampling uncertainty of the c-index statistic (unlike the
    fold-to-fold percentile spread, which measures something different).
    """
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)
    n = len(event)
    if n < 10 or event.sum() < min_events:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    scores = np.empty(B)
    scores[:] = np.nan
    kept = 0
    for b in range(B):
        idx = rng.integers(0, n, n)
        e_b, t_b, r_b = event[idx], time[idx], risk[idx]
        if e_b.sum() < min_events:
            continue
        try:
            scores[kept] = concordance_index_censored(e_b, t_b, r_b)[0]
            kept += 1
        except Exception:
            continue
    if kept < max(50, B // 10):   # require enough successful resamples to trust the CI
        return np.nan, np.nan, np.nan, kept
    valid = scores[:kept]
    lo, hi = np.nanpercentile(valid, [2.5, 97.5])
    return lo, hi, float(np.nanmean(valid)), kept

def bootstrap_group(key, sub):
    """Run bootstrap_cindex_ci for one (cohort,gene_set,arm,model) group's
    pooled out-of-fold predictions."""
    lo, hi, mean, kept = bootstrap_cindex_ci(sub["event"].values, sub["time"].values, sub["risk"].values)
    return dict(cohort=key[0], gene_set=key[1], arm=key[2], model=key[3],
                n_oof=len(sub), boot_kept=kept,
                boot_cindex_mean=mean, boot_cindex_lo=lo, boot_cindex_hi=hi)

def main():
    cohorts = sorted({f.split("_surv.parquet")[0] for f in os.listdir(HARM) if f.endswith("_surv.parquet")})
    log(f"cohorts: {cohorts}")
    rows = []
    oof_rows = []   # pooled out-of-fold predictions for the bootstrap step

    for coh in cohorts:
        surv = pd.read_parquet(f"{HARM}/{coh}_surv.parquet").set_index("sample")
        expr = pd.read_parquet(f"{HARM}/{coh}_expr.parquet")
        expr = expr.loc[surv.index]
        y = Surv.from_arrays(surv["event"].astype(bool).values, surv["time_months"].values)
        endpoint = surv["endpoint"].iloc[0]
        ev_times = surv.loc[surv["event"] == 1, "time_months"]
        times = np.percentile(ev_times, [20, 40, 60, 80]) if len(ev_times) > 10 else []
        log(f"--- {coh}: n={len(surv)} events={int(surv['event'].sum())} genes={expr.shape[1]}")

        arms = [("gene", True, False), ("clinical", False, True), ("gene+clinical", True, True)]
        tasks = []
        for gs_name, gs in GENE_SETS.items():
            for arm, ug, uc in arms:
                if arm == "clinical" and gs_name != "Novel5":   # clinical arm is gene-set independent
                    continue
                X, nc, cc = build_X(expr, surv, gs["genes"], ug, uc)
                if X is None: continue
                avail = sum(1 for g in gs["genes"] if g in expr.columns)
                for rep in range(N_REPEATS):
                    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED + rep)
                    for k, (tr, te) in enumerate(skf.split(X, surv["event"].astype(int))):
                        for mname, est, grid in model_grid(X.shape[1]):
                            tasks.append((gs_name, arm, avail, rep, k, X, y, tr, te, mname, est, grid, nc, cc, times))
        log(f"    {len(tasks)} fold-tasks")
        res = Parallel(n_jobs=NJOBS, verbose=0, batch_size=4)(
            delayed(eval_fold)(t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14]) for t in tasks)

        for t, r in zip(tasks, res):
            gs_name, arm, avail, rep, k = t[0], t[1], t[2], t[3], t[4]
            mname = t[9]
            oof_idx, oof_risk = r.pop("oof_idx"), r.pop("oof_risk")
            rows.append(dict(cohort=coh, endpoint=endpoint, gene_set=gs_name, arm=arm,
                             n_genes_avail=avail, n_genes_set=len(GENE_SETS[gs_name]["genes"]),
                             rep=rep, fold=k, n=len(surv), events=int(surv["event"].sum()), **r))
            if oof_idx is not None:
                ev = y["event"][oof_idx].astype(int)
                tm = y["time"][oof_idx]
                for i in range(len(oof_idx)):
                    oof_rows.append((coh, gs_name, arm, mname, rep, int(oof_idx[i]),
                                     int(ev[i]), float(tm[i]), float(oof_risk[i])))
        pd.DataFrame(rows).to_csv(f"{OUT}/within_cohort_folds.csv", index=False)
        log(f"    wrote {len(rows)} rows")

    df = pd.DataFrame(rows)

    # ── repeated-CV fold-spread summary (kept from the original script, but
    #    renamed fold_cindex_* to make clear this is NOT a bootstrap CI) ────
    agg = (df.groupby(["cohort","endpoint","gene_set","arm","model"], dropna=False)
             .agg(n=("n","first"), events=("events","first"),
                  n_genes_avail=("n_genes_avail","first"), n_genes_set=("n_genes_set","first"),
                  cindex_mean=("cindex","mean"), cindex_sd=("cindex","std"),
                  fold_cindex_lo=("cindex", lambda s: np.nanpercentile(s,2.5) if s.notna().sum()>2 else np.nan),
                  fold_cindex_hi=("cindex", lambda s: np.nanpercentile(s,97.5) if s.notna().sum()>2 else np.nan),
                  uno_mean=("uno_c","mean"), ibs_mean=("ibs","mean"),
                  n_folds=("cindex","count")).reset_index())

    # ── true bootstrap CI on pooled out-of-fold predictions ─────────────────
    oof = pd.DataFrame(oof_rows, columns=["cohort","gene_set","arm","model","rep",
                                          "sample_idx","event","time","risk"])
    oof.to_csv(f"{OUT}/within_cohort_oof_predictions.csv", index=False)
    log(f"pooled {len(oof)} out-of-fold predictions across all groups")

    groups = list(oof.groupby(["cohort","gene_set","arm","model"]))
    log(f"running bootstrap ({B_BOOT} resamples) for {len(groups)} groups")
    boot_results = Parallel(n_jobs=NJOBS, verbose=0)(
        delayed(bootstrap_group)(key, sub) for key, sub in groups)
    boot_df = pd.DataFrame(boot_results)
    boot_df.to_csv(f"{OUT}/within_cohort_bootstrap_ci.csv", index=False)

    agg = agg.merge(boot_df, on=["cohort","gene_set","arm","model"], how="left")
    agg.to_csv(f"{OUT}/within_cohort_summary.csv", index=False)
    log("done")
    print(agg[agg.arm=="gene"].sort_values(["cohort","cindex_mean"], ascending=[True,False]).to_string(index=False))

if __name__ == "__main__":
    main()
