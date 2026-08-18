import os
import sys
import json
import glob
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

piv = full.pivot_table(index="gene_set", columns="learner", values="cindex",
                       aggfunc="mean").reindex(index=GENE_SETS, columns=LEARNERS)
ncoh = full.pivot_table(index="gene_set", columns="learner", values="cindex",
                        aggfunc="count").reindex(index=GENE_SETS, columns=LEARNERS)

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