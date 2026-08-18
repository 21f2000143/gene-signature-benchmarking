import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import traceback
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os

from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

warnings.filterwarnings("ignore")

DATA = os.environ.get("BC_BENCH", "/mnt/kedargouri/sachin/projects/paper2/harmonised")
SEED = 20260725
N_BOOT = 2000
PARTS = "parts"
os.makedirs(PARTS, exist_ok=True)

OS_COHORTS = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203",
              "GSE20711", "GSE58812"]
SEC_COHORTS = ["GSE6532", "GSE11121", "GSE21653"]

with open(os.path.join(DATA, "gene_sets.json")) as fh:
    _RAW = json.load(fh)
GENE_SETS = {k: list(v["genes"]) if isinstance(v, dict) else list(v)
             for k, v in _RAW.items()}
GS_FAMILY = {k: (v.get("family", "") if isinstance(v, dict) else "")
             for k, v in _RAW.items()}
ALL_GENES = sorted({g for v in GENE_SETS.values() for g in v})
print({k: len(v) for k, v in GENE_SETS.items()}, flush=True)
print(f"union of gene-set genes: {len(ALL_GENES)}", flush=True)

SURV_KEEP = ["time_months", "event", "endpoint", "cohort", "platform"]


def load_cohort(c):
    surv = pd.read_parquet(os.path.join(DATA, f"{c}_surv.parquet"))
    if "sample" in surv.columns:
        surv = surv.set_index("sample")
    surv = surv[[k for k in SURV_KEEP if k in surv.columns]]
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(os.path.join(DATA, f"{c}_expr.parquet")).schema.names)
    cols = [g for g in ALL_GENES if g in have]
    expr = pd.read_parquet(os.path.join(DATA, f"{c}_expr.parquet"), columns=cols)
    common = surv.index.intersection(expr.index)
    surv, expr = surv.loc[common], expr.loc[common]
    ok = (surv["time_months"].astype(float) > 0) & surv["time_months"].notna() \
         & surv["event"].notna()
    return surv.loc[ok.values], expr.loc[ok.values].astype(np.float32)


COH = {}
for c in OS_COHORTS + SEC_COHORTS:
    COH[c] = load_cohort(c)
    s, e = COH[c]
    print(f"loaded {c}: n={len(s)} events={int(s['event'].sum())} "
          f"endpoint={s['endpoint'].iloc[0]} panel_genes={e.shape[1]}", flush=True)


def y_of(surv):
    return Surv.from_arrays(event=surv["event"].astype(float).astype(bool).values,
                            time=surv["time_months"].astype(float).values)


def make_models(n_feat):
    m = {}
    m["CoxPH_ridge"] = ([(f"alpha={a}", lambda a=a: CoxPHSurvivalAnalysis(alpha=a, n_iter=200))
                         for a in [0.01, 0.1, 1.0, 10.0, 100.0]], 5)
    m["Coxnet"] = ([(f"alpha={a}", lambda a=a: CoxnetSurvivalAnalysis(
                        l1_ratio=0.9, alphas=[a], fit_baseline_model=False,
                        max_iter=100000, tol=1e-6))
                    for a in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]], 5)
    m["RSF"] = ([("leaf=50,frac=0.5", lambda: RandomSurvivalForest(
                    n_estimators=100, min_samples_leaf=50, max_features="sqrt",
                    max_samples=0.5, n_jobs=1, random_state=SEED, low_memory=True)),
                 ("leaf=100,frac=0.4", lambda: RandomSurvivalForest(
                    n_estimators=100, min_samples_leaf=100, max_features="sqrt",
                    max_samples=0.4, n_jobs=1, random_state=SEED, low_memory=True))], 3)
    m["GBSA"] = ([(f"n={n},lr={lr},d={d}", lambda n=n, lr=lr, d=d:
                    GradientBoostingSurvivalAnalysis(
                        n_estimators=n, learning_rate=lr, max_depth=d,
                        subsample=0.8, random_state=SEED))
                  for (n, lr, d) in [(100, 0.1, 2), (150, 0.05, 2)]], 3)
    return m


def risk(est, X):
    return np.asarray(est.predict(X), dtype=float).ravel()


def comparable_matrix(t, e):
    t = np.asarray(t, float)
    e = np.asarray(e, bool)
    later = t[None, :] > t[:, None]
    same_cens = (t[None, :] == t[:, None]) & (~e[None, :])
    A = (e[:, None] & (later | same_cens))
    return A.astype(np.float32)


def conc_matrix(A, s):
    d = s[:, None] - s[None, :]
    W = np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0)).astype(np.float32)
    return A * W


def boot_ci_fast(A, N, n_boot=N_BOOT, seed=SEED, block=200):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for start in range(0, n_boot, block):
        b = min(block, n_boot - start)
        Wm = rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float32)
        den = np.einsum("bi,bi->b", Wm @ A, Wm)
        num = np.einsum("bi,bi->b", Wm @ N, Wm)
        good = den > 0
        vals.append(num[good] / den[good])
    v = np.concatenate(vals)
    if v.size < 50:
        return np.nan, np.nan, int(v.size)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi), int(v.size)


def uno(y_train, surv_test, score):
    try:
        ev = surv_test["event"].astype(float).astype(bool).values
        tt = surv_test["time_months"].astype(float).values
        if ev.sum() < 2:
            return np.nan, "too few events"
        y_te = Surv.from_arrays(event=ev, time=tt)
        tau = min(float(y_train["time"].max()), float(np.quantile(tt[ev], 0.95)))
        keep = tt < tau
        if keep.sum() < 10 or ev[keep].sum() < 2:
            return np.nan, "too few events below tau"
        return float(concordance_index_ipcw(y_train, y_te[keep], score[keep],
                                            tau=tau)[0]), ""
    except Exception as e:
        return np.nan, f"uno:{type(e).__name__}: {e}"


def cohort_folds(cohort_labels, n_splits):
    uniq, counts = np.unique(cohort_labels, return_counts=True)
    k = min(n_splits, len(uniq))
    assign, load = {}, np.zeros(k)
    for i in np.argsort(-counts):
        j = int(np.argmin(load))
        assign[uniq[i]] = j
        load[j] += counts[i]
    fold_of = np.array([assign[c] for c in cohort_labels])
    folds = []
    for j in range(k):
        te = np.where(fold_of == j)[0]
        tr = np.where(fold_of != j)[0]
        if len(te) >= 10 and len(tr) >= 30:
            folds.append((tr, te))
    return folds


def run_cell(held_out, gs_name, train_cohorts, tag):
    rows, score_rows = [], []
    genes_req = GENE_SETS[gs_name]
    surv_te, expr_te = COH[held_out]

    avail = set(genes_req) & set(expr_te.columns)
    for c in train_cohorts:
        avail &= set(COH[c][1].columns)
    genes = [g for g in genes_req if g in avail]
    n_genes = len(genes)

    base = dict(held_out_cohort=held_out, gene_set=gs_name,
                gene_set_family=GS_FAMILY.get(gs_name, ""),
                n_genes_requested=len(genes_req), n_genes_used=n_genes,
                genes_used="|".join(genes),
                n_test=len(surv_te), events_test=int(surv_te["event"].sum()),
                test_endpoint=surv_te["endpoint"].iloc[0],
                train_cohorts="|".join(train_cohorts),
                train_endpoints="|".join(sorted({COH[c][0]["endpoint"].iloc[0]
                                                 for c in train_cohorts})),
                track=tag)

    if n_genes == 0:
        for mdl in ["CoxPH_ridge", "Coxnet", "RSF", "GBSA"]:
            rows.append({**base, "model": mdl, "n_train": 0, "events_train": 0,
                         "cindex": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "uno_c": np.nan,
                         "error": "no gene of this set present in both the training pool and the held-out cohort"})
        return rows, score_rows

    Xtr = pd.concat([COH[c][1][genes] for c in train_cohorts], axis=0)
    Str = pd.concat([COH[c][0] for c in train_cohorts], axis=0)
    Xtr = Xtr.loc[Str.index].astype(np.float64).fillna(0.0)
    ytr = y_of(Str)
    Xte = expr_te[genes].astype(np.float64).fillna(0.0)
    base.update(n_train=len(Str), events_train=int(Str["event"].sum()))

    ev_te = surv_te["event"].astype(float).astype(bool).values
    tt_te = surv_te["time_months"].astype(float).values
    A = comparable_matrix(tt_te, ev_te)
    coh_lab = Str["cohort"].values
    Xtr_v, Xte_v = Xtr.values, Xte.values

    for mdl, (grid, nsplit) in make_models(n_genes).items():
        try:
            folds = cohort_folds(coh_lab, nsplit)
            best_lab, best_sc, cv_note = grid[0][0], np.nan, ""
            if len(grid) > 1 and folds:
                bs = -np.inf
                for lab, fac in grid:
                    sc = []
                    for tr_i, te_i in folds:
                        try:
                            est = fac().fit(Xtr_v[tr_i], ytr[tr_i])
                            sc.append(concordance_index_censored(
                                ytr["event"][te_i], ytr["time"][te_i],
                                risk(est, Xtr_v[te_i]))[0])
                        except Exception:
                            sc.append(np.nan)
                    ms = np.nanmean(sc) if not np.all(np.isnan(sc)) else -np.inf
                    if ms > bs:
                        bs, best_lab = ms, lab
                best_sc = bs
                cv_note = f"cohort_grouped_cv n_folds={len(folds)} inner_c={bs:.4f}"
            est = dict(grid)[best_lab]().fit(Xtr_v, ytr)
            s_te = risk(est, Xte_v)
            if np.unique(s_te).size < 2:
                raise ValueError("degenerate (constant) risk score on held-out cohort")

            c_h = float(concordance_index_censored(ev_te, tt_te, s_te)[0])
            N = conc_matrix(A, s_te)
            den = float(A.sum())
            c_fast = float(N.sum() / den) if den > 0 else np.nan
            lo, hi, nb = boot_ci_fast(A, N)
            u, u_err = uno(ytr, surv_te, s_te)
            rows.append({**base, "model": mdl, "best_param": best_lab,
                         "cindex": c_h, "ci_lo": lo, "ci_hi": hi, "uno_c": u,
                         "n_boot_ok": nb, "n_folds_inner": len(folds),
                         "inner_cv": cv_note,
                         "c_fast_minus_sksurv": c_fast - c_h,
                         "error": u_err})
            if gs_name == "Novel5":
                for smp, tm, evn, sv in zip(surv_te.index, tt_te, ev_te, s_te):
                    score_rows.append(dict(sample=smp, cohort=held_out, model=mdl,
                                           gene_set=gs_name, track=tag,
                                           time_months=float(tm), event=int(evn),
                                           risk_score=float(sv)))
        except Exception as e:
            rows.append({**base, "model": mdl, "best_param": "",
                         "cindex": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "uno_c": np.nan, "n_boot_ok": 0,
                         "error": f"{type(e).__name__}: {e} | {traceback.format_exc(limit=1)}"})

    with open(os.path.join(PARTS, f"{tag}__{held_out}__{gs_name}.json"), "w") as fh:
        json.dump({"rows": rows, "scores": score_rows}, fh)
    return rows, score_rows


TASKS = []
for h in OS_COHORTS:
    for gs in GENE_SETS:
        TASKS.append((h, gs, [c for c in OS_COHORTS if c != h], "loco_os"))
for h in SEC_COHORTS:
    for gs in GENE_SETS:
        TASKS.append((h, gs, [c for c in SEC_COHORTS if c != h], "loco_secondary"))
for h in SEC_COHORTS:
    for gs in GENE_SETS:
        TASKS.append((h, gs, list(OS_COHORTS), "transfer"))
print(f"n_tasks={len(TASKS)}", flush=True)

res = Parallel(n_jobs=54, verbose=10, backend="loky")(
    delayed(run_cell)(h, gs, tc, tag) for (h, gs, tc, tag) in TASKS)

rows_all, score_all = [], []
for rows, sc in res:
    rows_all.extend(rows)
    score_all.extend(sc)

df = pd.DataFrame(rows_all)
COLS = ["held_out_cohort", "gene_set", "gene_set_family", "model", "n_test",
        "events_test", "n_genes_requested", "n_genes_used", "cindex", "ci_lo",
        "ci_hi", "uno_c", "n_train", "events_train", "best_param",
        "n_folds_inner", "inner_cv", "n_boot_ok", "c_fast_minus_sksurv",
        "test_endpoint", "train_cohorts", "train_endpoints", "genes_used", "error"]
df = df.reindex(columns=[c for c in COLS if c in df.columns]
                + [c for c in df.columns if c not in COLS])
os.makedirs("setup", exist_ok=True)
for tag, fn in [("loco_os", "loco_os.csv"), ("loco_secondary", "loco_secondary.csv"),
                ("transfer", "cross_endpoint_transfer.csv")]:
    df[df.track == tag].drop(columns=["track"]).to_csv(os.path.join("setup", fn), index=False)