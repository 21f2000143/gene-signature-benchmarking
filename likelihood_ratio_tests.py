import os, json, time
import numpy as np
import pandas as pd
import re
from scipy import stats
from scipy.optimize import minimize
from joblib import Parallel, delayed
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

BASE = "/mnt/kedargouri/sachin/projects/paper2/harmonised"
COH = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203"]
SEED = 20260725
ALPHA = 10.0
B = int(os.environ.get("B", 2000))
NJOBS = int(os.environ.get("NJOBS", 16))
CLIN_CANDIDATES = ["age", "grade", "size", "node"]

gs_raw = json.load(open(f"{BASE}/gene_sets.json"))
GS = {k: list(v["genes"]) for k, v in gs_raw.items()}
PANELS = list(GS.keys())

surv, EMAT, cols_of, EV, TM, YC = {}, {}, {}, {}, {}, {}
raw = {}
for c in COH:
    s = pd.read_parquet(f"{BASE}/{c}_surv.parquet")
    if "sample" in s.columns:
        s = s.set_index("sample")
    e = pd.read_parquet(f"{BASE}/{c}_expr.parquet").loc[s.index]
    surv[c], raw[c] = s, e
    EV[c] = s["event"].to_numpy(bool)
    TM[c] = s["time_months"].to_numpy(float)
    YC[c] = Surv.from_arrays(EV[c], TM[c])
universe = set.intersection(*[set(raw[c].columns) for c in COH])

# ---- leakage-free per-patient risk scores: LOCO ridge Cox --------------------
avail = {p: [g for g in GS[p] if g in universe] for p in PANELS}
RISK = {}          # RISK[cohort] -> DataFrame (n x panels), z-scored
for h in COH:
    tr = [c for c in COH if c != h]
    out = {}
    for p in PANELS:
        g = avail[p]
        X = np.vstack([raw[c][g].to_numpy(np.float64) for c in tr])
        y = np.concatenate([YC[c] for c in tr])
        m = CoxPHSurvivalAnalysis(alpha=ALPHA, n_iter=200, tol=1e-7).fit(X, y)
        r = m.predict(raw[h][g].to_numpy(np.float64))
        out[p] = (r - r.mean()) / r.std()
    RISK[h] = pd.DataFrame(out, index=surv[h].index)
    cs = {p: concordance_index_censored(EV[h], TM[h], RISK[h][p].to_numpy())[0] for p in PANELS}
    print(h, {p: round(cs[p], 4) for p in PANELS}, flush=True)
del raw

# ---- Cox partial likelihood (Breslow) for LRTs -------------------------------
def cox_loglik_fit(X, ev, tm):
    o = np.argsort(tm, kind="mergesort")
    X, ev, tm = X[o], ev[o], tm[o]
    start = np.searchsorted(tm, tm, side="left")
    ei = np.flatnonzero(ev)
    n, p = X.shape
    if p == 0:
        rc = np.cumsum(np.ones(n)[::-1])[::-1]
        return np.zeros(0), float(-np.log(rc[start[ei]]).sum())

    def nll(b):
        eta = X @ b
        eta -= eta.max()
        w = np.exp(eta)
        rc = np.cumsum(w[::-1])[::-1]
        rcx = np.cumsum((w[:, None] * X)[::-1], axis=0)[::-1]
        s = rc[start[ei]]
        ll = (eta[ei] - np.log(s)).sum()
        g = (X[ei] - rcx[start[ei]] / s[:, None]).sum(0)
        return -ll, -g

    r = minimize(nll, np.zeros(p), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=500, ftol=1e-12, gtol=1e-10))
    return r.x, float(-r.fun)

# validate against sksurv on a real cohort
_h = "METABRIC"
_X = RISK[_h][["Novel5"]].to_numpy()
_b, _ll = cox_loglik_fit(_X, EV[_h], TM[_h])
_sk = CoxPHSurvivalAnalysis(alpha=1e-8, n_iter=500, tol=1e-9).fit(_X, YC[_h]).coef_
print("LRT_ENGINE_CHECK coef_mine", np.round(_b, 6), "coef_sksurv", np.round(_sk, 6),
      "absdiff", float(np.abs(_b - _sk).max()), flush=True)
assert np.abs(_b - _sk).max() < 1e-3, "Cox partial-likelihood engine disagrees with sksurv"

def encode_clin(col, name):
    """Numeric encoding of a clinical column. Several cohorts store grade/node as text
    ('G2', 'NodePositive', 'N1a'); plain to_numeric would blank the whole column."""
    num = pd.to_numeric(col.astype("object"), errors="coerce")
    if num.notna().mean() >= 0.5:
        return pd.Series(np.asarray(num, dtype=float), index=col.index)

    # plain-Python strings: pandas 3 arrow dtypes turn None back into float NaN on .map
    vals = [None if (v is None or (isinstance(v, float) and v != v)) else str(v).strip()
            for v in col.tolist()]

    def grade_ord(v):
        return float(v[1]) if (isinstance(v, str) and re.fullmatch(r"G[123]", v)) else np.nan

    def node_ord(v):
        if not isinstance(v, str):
            return np.nan
        if v == "NodeNegative":
            return 0.0
        if v == "NodePositive":
            return 1.0
        if v == "SubMicroMet":
            return 1.0
        if v == "1to3":
            return 2.0
        if v == "4toX":
            return 3.0
        m = re.match(r"N([0-3])", v)          # TCGA AJCC N stage: N0/N0 (i-)/N1a/N2/N3b
        return float(m.group(1)) if m else np.nan

    fn = {"grade": grade_ord, "node": node_ord}.get(name)
    enc = [fn(v) for v in vals] if fn else [np.nan] * len(vals)
    return pd.Series(np.asarray(enc, dtype=float), index=col.index)

lr_rows = []
clin_used = {}
for h in COH:
    s = surv[h]
    D = pd.DataFrame({k: encode_clin(s[k], k) for k in CLIN_CANDIDATES if k in s.columns})
    D = D.loc[:, D.notna().mean() >= 0.5]     # drop covariates absent in this cohort
    use = list(D.columns)
    clin_used[h] = use
    keep = D.notna().all(1).to_numpy() & np.isfinite(TM[h]) & (TM[h] > 0)
    print("clin", h, use, "n_complete", int(keep.sum()), "events", int(EV[h][keep].sum()), flush=True)
    assert keep.sum() > 50 and EV[h][keep].sum() > 5, f"{h}: clinical model has too few usable rows"
    Xc = D.to_numpy(float)[keep]
    assert Xc.shape[1] == len(use) and Xc.shape[0] == keep.sum()
    Xc = (Xc - Xc.mean(0)) / np.where(Xc.std(0) > 0, Xc.std(0), 1)
    ev, tm = EV[h][keep], TM[h][keep]
    _, ll0 = cox_loglik_fit(Xc, ev, tm)
    for p in PANELS:
        r = RISK[h][p].to_numpy()[keep]
        r = (r - r.mean()) / r.std()
        Xf = np.column_stack([Xc, r])
        bf, ll1 = cox_loglik_fit(Xf, ev, tm)
        chi2 = 2 * (ll1 - ll0)
        lr_rows.append(dict(cohort=h, model_compared=f"clinical[{'+'.join(use)}] vs clinical+{p}",
                            signature=p, clinical_covariates="+".join(use),
                            n_used=int(keep.sum()), n_events_used=int(ev.sum()),
                            n_genes_available=len(avail[p]),
                            loglik_clinical=ll0, loglik_full=ll1,
                            lr_chi2=float(chi2), df=1,
                            p=float(stats.chi2.sf(max(chi2, 0.0), 1)),
                            beta_signature=float(bf[-1])))
    # signature alone vs null model, for reference
    _, llnull = cox_loglik_fit(np.zeros((keep.sum(), 0)), ev, tm)
    for p in PANELS:
        r = RISK[h][p].to_numpy()[keep]
        r = ((r - r.mean()) / r.std())[:, None]
        _, lls = cox_loglik_fit(r, ev, tm)
        chi2 = 2 * (lls - llnull)
        lr_rows.append(dict(cohort=h, model_compared=f"null vs {p} alone", signature=p,
                            clinical_covariates="", n_used=int(keep.sum()),
                            n_events_used=int(ev.sum()), n_genes_available=len(avail[p]),
                            loglik_clinical=llnull, loglik_full=lls,
                            lr_chi2=float(chi2), df=1,
                            p=float(stats.chi2.sf(max(chi2, 0.0), 1)), beta_signature=np.nan))
lr = pd.DataFrame(lr_rows)
os.makedirs("setup", exist_ok=True)
os.makedirs("setup", exist_ok=True)
lr.to_csv(os.path.join("setup", "likelihood_ratio_tests.csv"), index=False)
print("PARTB_DONE", flush=True)