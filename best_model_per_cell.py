import pandas as pd
import numpy as np

pd.set_option("display.width", 250)

folds = pd.read_csv("results/within_cohort_folds.csv")

rng = np.random.default_rng(42)
def boot_ci(s):
    v = s.dropna().values
    if len(v) < 3: return (np.nan, np.nan)
    bm = rng.choice(v, size=(2000, len(v)), replace=True).mean(axis=1)
    return (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))

g = folds.groupby(["cohort","endpoint","gene_set","arm","model"], dropna=False)
summary = g.agg(n=("n","first"), events=("events","first"),
                n_genes_avail=("n_genes_avail","first"), n_genes_set=("n_genes_set","first"),
                cindex_mean=("cindex","mean"), cindex_sd=("cindex","std"),
                uno_mean=("uno_c","mean"), uno_sd=("uno_c","std"),
                ibs_mean=("ibs","mean"), n_folds=("cindex","count"),
                n_folds_attempted=("cindex","size")).reset_index()
ci = g["cindex"].apply(boot_ci)
summary[["cindex_lo","cindex_hi"]] = pd.DataFrame(ci.tolist(), index=ci.index).values
summary = summary[["cohort","endpoint","gene_set","arm","model","n","events","n_genes_avail",
                   "n_genes_set","cindex_mean","cindex_sd","cindex_lo","cindex_hi",
                   "uno_mean","uno_sd","ibs_mean","n_folds","n_folds_attempted"]].round(4)

best = (summary.sort_values("cindex_mean", ascending=False)
        .groupby(["cohort","endpoint","gene_set","arm"], as_index=False).first()
        .sort_values(["endpoint","cohort","gene_set","arm"]))
best.to_csv("best_model_per_cell.csv", index=False)