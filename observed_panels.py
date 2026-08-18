import os, json, time
import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

BASE = "/mnt/kedargouri/sachin/projects/paper2/harmonised"
COH = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203"]
SEED = 20260725
ALPHA = 10.0
N_RANDOM_5 = int(os.environ.get("N_RANDOM_5", 10000))
N_SIZE_MATCHED = int(os.environ.get("N_SIZE_MATCHED", 500))
NJOBS = int(os.environ.get("NJOBS", 64))

gs_raw = json.load(open(f"{BASE}/gene_sets.json"))
GS = {k: list(v["genes"]) for k, v in gs_raw.items()}

surv, expr_cols = {}, {}
raw = {}
for c in COH:
    s = pd.read_parquet(f"{BASE}/{c}_surv.parquet")
    if "sample" in s.columns:
        s = s.set_index("sample")
    e = pd.read_parquet(f"{BASE}/{c}_expr.parquet").loc[s.index]
    surv[c], raw[c] = s, e
    expr_cols[c] = set(e.columns)

universe = set.intersection(*[expr_cols[c] for c in COH])
UNIV = np.array(sorted(universe))
GIDX = {g: i for i, g in enumerate(UNIV)}
EMAT, YC, EV, TM = {}, {}, {}, {}
for c in COH:
    EMAT[c] = np.ascontiguousarray(raw[c][UNIV].to_numpy(np.float64))
    EV[c] = surv[c]["event"].to_numpy(bool)
    TM[c] = surv[c]["time_months"].to_numpy(float)
    YC[c] = Surv.from_arrays(EV[c], TM[c])
del raw
print("universe_4cohort", len(UNIV), flush=True)

def score_panel(genes, alpha=ALPHA):
    """Mean held-out Harrell c across the 4 cohorts; ridge Cox trained on other 3."""
    cols = np.array([GIDX[g] for g in genes], int)
    per, errs = {}, []
    for h in COH:
        try:
            tr = [c for c in COH if c != h]
            X = np.vstack([EMAT[c][:, cols] for c in tr])
            y = np.concatenate([YC[c] for c in tr])
            m = CoxPHSurvivalAnalysis(alpha=alpha, n_iter=200, tol=1e-7).fit(X, y)
            r = m.predict(EMAT[h][:, cols])
            per[h] = float(concordance_index_censored(EV[h], TM[h], r)[0])
        except Exception as ex:
            per[h] = np.nan
            errs.append(f"{h}:{type(ex).__name__}:{str(ex)[:120]}")
    vals = np.array([per[c] for c in COH], float)
    mean = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else np.nan
    return mean, per, ";".join(errs)

# ---- observed panels -------------------------------------------------------
obs_rows = []
for name, genes in GS.items():
    avail = [g for g in genes if g in universe]
    sc, per, err = score_panel(avail)
    row = dict(panel=name, family=gs_raw[name].get("family", ""),
               n_genes_nominal=len(genes), n_genes_available=len(avail),
               loco_mean_cindex=sc, **{f"c_{c}": per[c] for c in COH}, error=err)
    for a in (1.0, 100.0):
        row[f"loco_mean_alpha{int(a)}"] = score_panel(avail, alpha=a)[0]
    obs_rows.append(row)
    print("observed", name, len(avail), round(sc, 5) if sc == sc else "nan", flush=True)
obs = pd.DataFrame(obs_rows)
os.makedirs("setup", exist_ok=True)
obs.to_csv(os.path.join("setup", "observed_panels.csv"), index=False)