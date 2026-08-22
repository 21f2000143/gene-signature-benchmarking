"""
rsf_mtry_control.py -- reviewer control: is the RSF reordering (Novel-5 falls to 6th
of 9 gene sets) a feature-count / mtry artifact, or a genuine learner-driven
reordering that survives when every gene set gets the same absolute number of
candidate features per split?

Design: identical to learner_grid.py's RSF cell (same data loading via
nested_core.load_all/common_genes, same LOCO folds over the 6 OS cohorts, same
RSF hyperparameters -- 300 trees, min_samples_leaf=15, bootstrap, max_samples=0.5,
random_state=0 -- and the same per-fold feature set: signature genes present in
all 6 cohorts of that fold), except max_features is fixed to a constant integer
(FIXED_MF, default 8 = round(sqrt(58)), the number of candidates GGI -- the
largest gene set at 58 genes -- already receives under the default "sqrt" rule)
for every gene set, not scaled by that gene set's own size. If the ranking still
reorders under this fixed-mtry control, feature count is not the explanation; if
it reverts toward the ridge ranking, it is consistent with the mechanism.

Usage: python3 rsf_mtry_control.py [fixed_max_features]
Output: rsf_mtry_control.csv (gene_set, held_out_cohort, cindex),
        rsf_mtry_control_summary.json (mean per gene set, Spearman vs ridge)
"""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nested_core as nc

OS6 = nc.OS6
FIXED_MF = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SEED = 0
GENE_SETS = ["Novel5", "Anchor4", "Novel5_plus_Anchor4", "PAM50", "OncotypeDX21",
             "GGI", "MammaPrint70", "BuffaHypoxia", "CNetCox6"]

with open("results/gene_sets.json") as f:
    gs_raw = json.load(f)

print("loading cohorts", flush=True)
store = nc.load_all(OS6, verbose=True)

# Build every cell's (Xtr, ttr, etr, Xte, tte, ete) once, matching learner_grid.py's
# per-fold feature-set rule: signature genes present in all 6 cohorts of that fold.
cells = []
for held in OS6:
    train = [c for c in OS6 if c != held]
    genes_all = nc.common_genes(store, train + [held])
    for gsname in GENE_SETS:
        sig_genes = [g for g in gs_raw[gsname]["genes"] if g in genes_all]
        if len(sig_genes) < 2:
            continue
        Xtr_parts, ttr_parts, etr_parts = [], [], []
        for coh in train:
            X, t, ev, _surv = store[coh]
            Xtr_parts.append(X[sig_genes].to_numpy(dtype=np.float64))
            ttr_parts.append(t)
            etr_parts.append(ev)
        Xtr = np.vstack(Xtr_parts)
        ttr = np.concatenate(ttr_parts)
        etr = np.concatenate(etr_parts)
        Xte_full, tte, ete, _surv_te = store[held]
        Xte = Xte_full[sig_genes].to_numpy(dtype=np.float64)
        cells.append((held, gsname, Xtr, ttr, etr, Xte, tte, ete, len(sig_genes)))

print("total cells: %d" % len(cells), flush=True)


def run_cell(args):
    held, gsname, Xtr, ttr, etr, Xte, tte, ete, k = args
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    t0 = time.time()
    mf = min(FIXED_MF, k)  # cannot exceed the gene set's own size
    m = RandomSurvivalForest(n_estimators=300, min_samples_leaf=15,
                              max_features=mf, bootstrap=True, max_samples=0.5,
                              n_jobs=1, random_state=SEED, low_memory=True)
    m.fit(Xtr, Surv.from_arrays(etr.astype(bool), ttr))
    risk = m.predict(Xte)
    c = nc.cindex(risk, tte, ete.astype(bool))
    return dict(held_out_cohort=held, gene_set=gsname, cindex=float(c),
                n_genes_used=k, max_features_used=mf,
                fit_seconds=round(time.time() - t0, 1))


rows = []
t_start = time.time()
with ProcessPoolExecutor(max_workers=48) as ex:
    futs = {ex.submit(run_cell, c): c for c in cells}
    for i, fut in enumerate(as_completed(futs)):
        r = fut.result()
        rows.append(r)
        print("  [%d/%d] %-20s %-10s c=%.4f (%.1fs)" %
              (i + 1, len(cells), r["gene_set"], r["held_out_cohort"],
               r["cindex"], r["fit_seconds"]), flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/rsf_mtry_control.csv", index=False)

summ = df.groupby("gene_set").cindex.mean().sort_values(ascending=False)
ridge_ms = pd.read_csv("results/metrics_summary.csv").set_index("gene_set")["harrell_mean"]
ridge_rank = ridge_ms.loc[summ.index].rank(ascending=False)
mtry_rank = summ.rank(ascending=False)
rho, p = spearmanr(ridge_rank, mtry_rank)

out = {
    "fixed_max_features": FIXED_MF,
    "n_cells": len(cells),
    "mean_cindex_by_geneset": summ.to_dict(),
    "ridge_mean_cindex_by_geneset": ridge_ms.loc[summ.index].to_dict(),
    "spearman_rho_vs_ridge_ranking": float(rho),
    "spearman_p": float(p),
    "elapsed_min": (time.time() - t_start) / 60,
}
json.dump(out, open("results/rsf_mtry_control_summary.json", "w"), indent=2)
print(json.dumps(out, indent=2), flush=True)
