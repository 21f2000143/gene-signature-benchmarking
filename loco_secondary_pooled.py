import pandas as pd
import numpy as np
import os

D = "hpc/"
sec = pd.read_csv(D+"loco_secondary.csv")

ssum = sec.groupby(["gene_set","model"]).apply(lambda g: pd.Series({
    "weighted_mean_cindex": np.average(g.cindex, weights=g.n_test),
    "unweighted_mean_cindex": g.cindex.mean(), "n_cohorts": len(g),
    "min_n_genes_used": int(g.n_genes_used.min()),
    "endpoints": "|".join(sorted(g.test_endpoint.unique()))}), include_groups=False).reset_index()
os.makedirs("setup", exist_ok=True)
ssum.round(4).to_csv(os.path.join("setup", "loco_secondary_pooled.csv"), index=False)