import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")

D = os.environ.get("BC_BENCH", "/mnt/kedargouri/sachin/projects/paper2/harmonised")
SEED, N_BOOT = 20260725, 2000
OS_COHORTS = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203", "GSE20711", "GSE58812"]
_R = json.load(open(f"{D}/gene_sets.json"))
GS = {k: list(v["genes"]) if isinstance(v, dict) else list(v) for k, v in _R.items()}
ALL = sorted({g for v in GS.values() for g in v})

def load(c):
    s = pd.read_parquet(f"{D}/{c}_surv.parquet")
    if "sample" in s.columns: s = s.set_index("sample")
    s = s[["time_months", "event", "endpoint", "cohort"]]
    have = set(pq.ParquetFile(f"{D}/{c}_expr.parquet").schema.names)
    e = pd.read_parquet(f"{D}/{c}_expr.parquet", columns=[g for g in ALL if g in have])
    i = s.index.intersection(e.index); s, e = s.loc[i], e.loc[i]
    ok = (s["time_months"].astype(float) > 0) & s["time_months"].notna() & s["event"].notna()
    return s.loc[ok.values], e.loc[ok.values].astype(np.float32)

COH = {c: load(c) for c in OS_COHORTS}

def yof(s):
    return Surv.from_arrays(event=s["event"].astype(float).astype(bool).values,
                            time=s["time_months"].astype(float).values)

def comparable(t, e):
    t = np.asarray(t, float); e = np.asarray(e, bool)
    return (e[:, None] & ((t[None, :] > t[:, None]) |
            ((t[None, :] == t[:, None]) & (~e[None, :])))).astype(np.float32)

def conc(A, s):
    d = s[:, None] - s[None, :]
    return A * np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0)).astype(np.float32)

def folds_of(lab, k=5):
    u, ct = np.unique(lab, return_counts=True)
    k = min(k, len(u)); a, load = {}, np.zeros(k)
    for i in np.argsort(-ct):
        j = int(np.argmin(load)); a[u[i]] = j; load[j] += ct[i]
    fo = np.array([a[x] for x in lab]); out = []
    for j in range(k):
        te = np.where(fo == j)[0]; tr = np.where(fo != j)[0]
        if len(te) >= 10 and len(tr) >= 30: out.append((tr, te))
    return out

def score_for(held, gsname):
    """LOCO risk score for one gene set on one held-out cohort."""
    tc = [c for c in OS_COHORTS if c != held]
    ste, ete = COH[held]
    av = set(GS[gsname]) & set(ete.columns)
    for c in tc: av &= set(COH[c][1].columns)
    genes = [g for g in GS[gsname] if g in av]
    if not genes: return None, 0
    Xtr = pd.concat([COH[c][1][genes] for c in tc]); Str = pd.concat([COH[c][0] for c in tc])
    Xtr = Xtr.loc[Str.index].astype(float).fillna(0.0); ytr = yof(Str)
    fl = folds_of(Str["cohort"].values, 5)
    best, bs = 1.0, -np.inf
    for al in [0.01, 0.1, 1.0, 10.0, 100.0]:
        sc = []
        for tr_i, te_i in fl:
            try:
                m = CoxPHSurvivalAnalysis(alpha=al, n_iter=200).fit(Xtr.values[tr_i], ytr[tr_i])
                sc.append(concordance_index_censored(ytr["event"][te_i], ytr["time"][te_i],
                          m.predict(Xtr.values[te_i]))[0])
            except Exception: sc.append(np.nan)
        ms = np.nanmean(sc) if sc and not np.all(np.isnan(sc)) else -np.inf
        if ms > bs: bs, best = ms, al
    m = CoxPHSurvivalAnalysis(alpha=best, n_iter=200).fit(Xtr.values, ytr)
    return np.asarray(m.predict(ete[genes].astype(float).fillna(0.0).values), float).ravel(), len(genes)

def per_cohort(held):
    ste, _ = COH[held]
    ev = ste["event"].astype(float).astype(bool).values
    tt = ste["time_months"].astype(float).values
    A = comparable(tt, ev); den_A = A.sum()
    S, NG = {}, {}
    for g in GS:
        s, n = score_for(held, g); S[g], NG[g] = s, n
    rng = np.random.default_rng(SEED)
    Wm = rng.multinomial(len(tt), np.full(len(tt), 1.0/len(tt)), size=N_BOOT).astype(np.float32)
    dA = np.einsum("bi,bi->b", Wm @ A, Wm)
    good = dA > 0
    cb = {}
    for g, s in S.items():
        if s is None: continue
        Ng = conc(A, s)
        cb[g] = np.einsum("bi,bi->b", Wm @ Ng, Wm)[good] / dA[good]
    rows = []
    c_point = {g: float(conc(A, s).sum()/den_A) for g, s in S.items() if s is not None}
    for g in cb:
        if g == "Novel5": continue
        d = cb["Novel5"] - cb[g]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(held_out_cohort=held, n_test=len(tt), events_test=int(ev.sum()),
            comparator=g, n_genes_novel5=NG["Novel5"], n_genes_comparator=NG[g],
            cindex_novel5=c_point["Novel5"], cindex_comparator=c_point[g],
            delta_cindex=c_point["Novel5"]-c_point[g],
            delta_ci_lo=float(lo), delta_ci_hi=float(hi),
            p_novel5_better=float((d > 0).mean()),
            favours=("Novel5" if lo > 0 else ("comparator" if hi < 0 else "inconclusive"))))
    return rows

res = Parallel(n_jobs=6, verbose=5)(delayed(per_cohort)(h) for h in OS_COHORTS)
df = pd.DataFrame([r for rs in res for r in rs])
os.makedirs("setup", exist_ok=True)
df.to_csv(os.path.join("setup", "loco_paired_novel5_vs_comparators.csv"), index=False)
print(df.round(4).to_string(index=False), flush=True)
print("\n=== wins/losses per comparator (CoxPH ridge, paired) ===", flush=True)
print(df.groupby("comparator").favours.value_counts().unstack(fill_value=0).to_string(), flush=True)
print("DONE", flush=True)