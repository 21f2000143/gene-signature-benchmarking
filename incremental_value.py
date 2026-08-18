import pandas as pd
import numpy as np

pd.set_option("display.width", 250)

folds = pd.read_csv("within_cohort_folds.csv")

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

bb = best.set_index(["cohort","gene_set","arm"])
clin = best[best.arm=="clinical"].set_index("cohort")
rows=[]
for (coh, gs), _ in best[best.arm=="gene+clinical"].groupby(["cohort","gene_set"]):
    gc = bb.loc[(coh, gs, "gene+clinical")]
    go = bb.loc[(coh, gs, "gene")] if (coh, gs, "gene") in bb.index else None
    cl = clin.loc[coh] if coh in clin.index else None
    rows.append(dict(cohort=coh, endpoint=gc.endpoint, gene_set=gs, n=gc.n, events=gc.events,
        n_genes_avail=gc.n_genes_avail, n_genes_set=gc.n_genes_set,
        cindex_gene=None if go is None else go.cindex_mean, model_gene=None if go is None else go.model,
        cindex_clinical=None if cl is None else cl.cindex_mean, model_clinical=None if cl is None else cl.model,
        cindex_gene_clinical=gc.cindex_mean, model_gene_clinical=gc.model,
        delta_vs_clinical=None if cl is None else round(gc.cindex_mean-cl.cindex_mean,4),
        delta_vs_gene_only=None if go is None else round(gc.cindex_mean-go.cindex_mean,4)))
incr = pd.DataFrame(rows).sort_values(["endpoint","cohort","delta_vs_clinical"], ascending=[True,True,False])
incr.to_csv("incremental_value.csv", index=False)