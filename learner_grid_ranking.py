import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OS6 = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203", "GSE20711", "GSE58812"]
GENE_SETS = ["Novel5", "Anchor4", "Novel5_plus_Anchor4", "PAM50", "OncotypeDX21",
             "GGI", "MammaPrint70", "BuffaHypoxia", "CNetCox6", "Clinical"]
LEARNERS = ["CoxPH_ridge", "Coxnet", "RSF", "GBSA"]
PRIMARY = "CoxPH_ridge"

shard_files = {
    "TCGA": "learner_grid_TCGA.csv",
    "METABRIC": "learner_grid_METABRIC.csv",
    "SCANB_GSE96058": "learner_grid_SCANB_GSE96058.csv",
    "SCANB_GSE202203": "learner_grid_SCANB_GSE202203.csv",
    "GSE20711": "learner_grid_GSE20711.csv",
    "GSE58812": "learner_grid_GSE58812.csv",
}

frames = []
for coh in OS6:
    p = shard_files[coh]
    d = pd.read_csv(p)
    if len(d):
        frames.append(d)

full = pd.concat(frames, ignore_index=True)
full = full[full["cindex"].notna()]
full = full.drop_duplicates(subset=["held_out_cohort", "gene_set", "learner"],
                            keep="first")
full = full.sort_values(["learner", "gene_set", "held_out_cohort"]).reset_index(drop=True)
full.to_csv("learner_grid_full.csv", index=False)

# ------------------------------------------------------------------ completeness audit
expected = {(l, c, g) for l in LEARNERS for c in OS6 for g in GENE_SETS}
present = set(zip(full["learner"], full["held_out_cohort"], full["gene_set"]))
missing = sorted(expected - present)
complete = len(missing) == 0

# ------------------------------------------------------------------------- mean c-index
piv = full.pivot_table(index="gene_set", columns="learner", values="cindex",
                       aggfunc="mean").reindex(index=GENE_SETS, columns=LEARNERS)
ncoh = full.pivot_table(index="gene_set", columns="learner", values="cindex",
                        aggfunc="count").reindex(index=GENE_SETS, columns=LEARNERS)

# best-of-four: per-cell max over learners, then mean over cohorts. Only cells where all
# four learners ran contribute, otherwise the "best of four" would be a best-of-fewer.
cell = full.pivot_table(index=["gene_set", "held_out_cohort"], columns="learner",
                        values="cindex", aggfunc="mean")
cell_complete = cell.dropna(subset=[l for l in LEARNERS if l in cell.columns])
bo4 = cell_complete.max(axis=1).groupby("gene_set").mean().reindex(GENE_SETS)
bo4_n = cell_complete.max(axis=1).groupby("gene_set").size().reindex(GENE_SETS)
piv["BestOfFour"] = bo4
ncoh["BestOfFour"] = bo4_n

rank_tbl = piv.rank(ascending=False, method="dense")

summary_rows = []
for g in GENE_SETS:
    for col in piv.columns:
        v = piv.loc[g, col]
        summary_rows.append({
            "gene_set": g, "learner": col,
            "mean_cindex": (round(float(v), 6) if pd.notna(v) else None),
            "rank": (int(rank_tbl.loc[g, col]) if pd.notna(rank_tbl.loc[g, col]) else None),
            "n_cohorts": (int(ncoh.loc[g, col]) if pd.notna(ncoh.loc[g, col]) else 0),
            "is_prespecified_primary": bool(col == PRIMARY),
            "is_selection_on_evaluation_data": bool(col == "BestOfFour"),
        })
pd.DataFrame(summary_rows).to_csv("learner_grid_summary.csv", index=False)

# ----------------------------------------------------------------------------- Spearman
spearman = {}
for col in [c for c in piv.columns if c != PRIMARY]:
    both = piv[[PRIMARY, col]].dropna()
    if len(both) >= 3:
        rho, p = spearmanr(both[PRIMARY].values, both[col].values)
        tau_pairs = int(len(both))
    else:
        rho, p, tau_pairs = float("nan"), float("nan"), int(len(both))
    spearman[col] = {"spearman_rho": (round(float(rho), 6) if np.isfinite(rho) else None),
                     "p_value": (round(float(p), 8) if np.isfinite(p) else None),
                     "n_gene_sets": tau_pairs,
                     "identical_ordering": bool(np.isfinite(rho) and abs(rho - 1.0) < 1e-12)}

rho_others = [spearman[c]["spearman_rho"] for c in spearman
              if c != "BestOfFour" and spearman[c]["spearman_rho"] is not None]
# The ranking is declared learner-dependent only if some learner reorders the gene sets.
depends = bool(len(rho_others) and min(rho_others) < 0.95)

rank_shift = {}
for col in [c for c in piv.columns if c != PRIMARY]:
    shifts = {}
    for g in GENE_SETS:
        a, b = rank_tbl.loc[g, PRIMARY], rank_tbl.loc[g, col]
        if pd.notna(a) and pd.notna(b) and int(a) != int(b):
            shifts[g] = {"rank_under_ridge": int(a), "rank_under_%s" % col: int(b)}
    rank_shift[col] = shifts

novel5_means = {col: (round(float(piv.loc["Novel5", col]), 6)
                      if pd.notna(piv.loc["Novel5", col]) else None)
                for col in piv.columns}

opt = (piv["BestOfFour"] - piv[PRIMARY]).dropna()
optimism = {
    "definition": "mean over gene sets of (best-of-four mean c) minus (ridge mean c); "
                  "the inflation produced by selecting the learner on the evaluation data",
    "mean_over_gene_sets": (round(float(opt.mean()), 6) if len(opt) else None),
    "max_over_gene_sets": (round(float(opt.max()), 6) if len(opt) else None),
    "per_gene_set": {g: round(float(opt[g]), 6) for g in opt.index},
}

top = {col: (piv[col].idxmax() if piv[col].notna().any() else None) for col in piv.columns}

ranking = {
    "review_item": "5 (best-of-four learner selection is tuning on the evaluation data)",
    "primary_prespecified_learner": PRIMARY,
    "grid_complete": complete,
    "n_cells_expected": len(expected),
    "n_cells_present": len(present),
    "missing_cells": [{"learner": l, "held_out_cohort": c, "gene_set": g}
                      for (l, c, g) in missing],
    "gene_sets_affected_by_missing_cells": sorted({g for (_, _, g) in missing}),
    "cohorts_affected_by_missing_cells": sorted({c for (_, c, _) in missing}),
    "ranking_depends_on_learner": depends,
    "spearman_vs_ridge": spearman,
    "mean_c_table": {col: {g: (round(float(piv.loc[g, col]), 6)
                               if pd.notna(piv.loc[g, col]) else None)
                           for g in GENE_SETS} for col in piv.columns},
    "rank_table": {col: {g: (int(rank_tbl.loc[g, col])
                             if pd.notna(rank_tbl.loc[g, col]) else None)
                         for g in GENE_SETS} for col in piv.columns},
    "rank_shift_vs_ridge": rank_shift,
    "top_gene_set_by_learner": top,
    "novel5_mean_c_per_learner": novel5_means,
    "best_of_four_optimism": optimism,
}
with open("learner_grid_ranking.json", "w") as f:
    json.dump(ranking, f, indent=1)

print("cells: %d/%d present%s" % (len(present), len(expected),
                                  "" if complete else "  INCOMPLETE"), flush=True)
if missing:
    print("missing:", missing[:20], flush=True)
print("\n=== mean LOCO c-index over held-out cohorts ===", flush=True)
print(piv.round(4).to_string(), flush=True)
print("\n=== rank (1 = best) ===", flush=True)
print(rank_tbl.astype("Int64").to_string(), flush=True)
print("\n=== Spearman of gene-set ordering vs ridge ===", flush=True)
for col, v in spearman.items():
    print("  %-12s rho=%s p=%s (n=%d)" % (col, v["spearman_rho"], v["p_value"],
                                          v["n_gene_sets"]), flush=True)
print("\nranking_depends_on_learner:", depends, flush=True)
print("Novel5 mean c:", novel5_means, flush=True)
print("best-of-four optimism (mean over gene sets):",
      optimism["mean_over_gene_sets"], flush=True)
print("\nwrote learner_grid_full.csv, learner_grid_summary.csv, learner_grid_ranking.json",
      flush=True)