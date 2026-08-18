import os, json, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from joblib import Parallel, delayed
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import re

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

# ---- paired bootstrap -------------------------------------------------------
def boot_one(cohort, seed):
    rng = np.random.default_rng(seed)
    n = len(EV[cohort])
    R = RISK[cohort].to_numpy()
    idx = rng.integers(0, n, n)
    ev, tm = EV[cohort][idx], TM[cohort][idx]
    if ev.sum() < 2:
        return np.full(R.shape[1], np.nan)
    out = np.empty(R.shape[1])
    for j in range(R.shape[1]):
        try:
            out[j] = concordance_index_censored(ev, tm, R[idx, j])[0]
        except Exception:
            out[j] = np.nan
    return out

rows = []
for ci, h in enumerate(COH):
    t = time.time()
    res = Parallel(n_jobs=NJOBS, batch_size=8)(
        delayed(boot_one)(h, SEED + 1000 * ci + b) for b in range(B))
    Cb = np.vstack(res)                        # B x n_panels
    print("boot", h, "sec", round(time.time() - t, 1),
          "nan_resamples", int(np.isnan(Cb).any(1).sum()), flush=True)
    inov = PANELS.index("Novel5")
    for j, p in enumerate(PANELS):
        if p == "Novel5":
            continue
        d = Cb[:, inov] - Cb[:, j]
        d = d[~np.isnan(d)]
        nle, nge = int((d <= 0).sum()), int((d >= 0).sum())
        pv = min(1.0, 2 * min(nle + 1, nge + 1) / (len(d) + 1))
        rows.append(dict(cohort=h, comparator=p,
                         n_genes_novel5=len(avail["Novel5"]), n_genes_comparator=len(avail[p]),
                         c_novel5=float(concordance_index_censored(
                             EV[h], TM[h], RISK[h]["Novel5"].to_numpy())[0]),
                         c_comparator=float(concordance_index_censored(
                             EV[h], TM[h], RISK[h][p].to_numpy())[0]),
                         n_boot=len(d), mean_diff=float(d.mean()),
                         ci_lo=float(np.percentile(d, 2.5)),
                         ci_hi=float(np.percentile(d, 97.5)),
                         p=pv))
pb = pd.DataFrame(rows)

def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    m = len(p)
    prev = 1.0
    for r, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * m / (m - r))
        q[i] = prev
    return q

pb["q"] = np.nan
for h in COH:
    msk = pb.cohort == h
    pb.loc[msk, "q"] = bh(pb.loc[msk, "p"].to_numpy())
pb["significant_q05"] = pb["q"] < 0.05
os.makedirs("setup", exist_ok=True)
os.makedirs("setup", exist_ok=True)
pb.to_csv(os.path.join("setup", "paired_bootstrap.csv"), index=False)