"""
run_reconstruct_check.py -- does the reconstructed forward search recover the
published five-gene panel when run on the cohorts the discovery report used?

The discovery report (ICMR Objective 3) states the search used TCGA, METABRIC and
SCAN-B with leave-one-cohort-out among them (GSE20685, its external cohort, is not
part of the present paper). If our reconstruction reproduces FLT3/CLIC6/SUSD3/ZIC2/
P4HA2 -- or most of it -- from those cohorts, then the nested re-run in
run_nested_selection.py is measuring the same procedure the panel came from.

Writes: reconstruct_check.csv (per-step trace), reconstruct_top10.csv
"""
import json
import time

import numpy as np
import pandas as pd

import nested_core as nc

# discovery-era cohorts present in this paper. SCAN-B appears here as two GEO
# releases of the same programme; the discovery report treats SCAN-B as one cohort.
DISCOVERY = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203"]

t0 = time.time()
store = nc.load_all(DISCOVERY)
genes = nc.common_genes(store, DISCOVERY)
print("candidate universe (genes common to discovery cohorts): %d" % len(genes), flush=True)

sel, info = nc.forward_search(store, DISCOVERY, genes, n_steps=5,
                              anchor=nc.ANCHOR4, alpha=100.0, verbose=True)

print("\nreconstructed panel:", sel, flush=True)
print("published panel    :", nc.NOVEL5, flush=True)
overlap = sorted(set(sel) & set(nc.NOVEL5))
print("overlap: %d/5 %s" % (len(overlap), overlap), flush=True)
print("elapsed %.1f min" % ((time.time() - t0) / 60), flush=True)

pd.DataFrame([{"inner_held_out": r["inner_held_out"], "step": r["step"],
               "chosen": r["chosen"], "heldout_c": r["score"],
               "pool_size": r["pool_size"], "n_candidates": r["n_candidates"]}
              for r in info["trace"]]).to_csv("reconstruct_check.csv", index=False)

rows = []
for r in info["trace"]:
    for rank, (g, s) in enumerate(r["top10"], 1):
        rows.append({"inner_held_out": r["inner_held_out"], "step": r["step"],
                     "rank": rank, "gene": g, "heldout_c": s})
pd.DataFrame(rows).to_csv("reconstruct_top10.csv", index=False)
pd.DataFrame(info["consensus"]).to_csv("reconstruct_consensus.csv", index=False)

json.dump({"reconstructed": sel, "published": nc.NOVEL5, "overlap": overlap,
           "per_fold_panels": {h: p for h, p in zip(info["inner_held_out"],
                                                    info["per_fold_panels"])},
           "pool_sizes": info["pool_sizes"],
           "n_candidates": len(genes), "discovery_cohorts": DISCOVERY,
           "elapsed_min": (time.time() - t0) / 60},
          open("reconstruct_summary.json", "w"), indent=2)
