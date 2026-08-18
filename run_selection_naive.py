"""
run_selection_naive.py -- reviewer item 1, third leg: the SELECTION-NAIVE estimate.

The discovery report states the panel was selected using TCGA, METABRIC and SCAN-B,
with GSE20685 as its external cohort. GSE20685 is not in the present paper, and none
of the five GEO cohorts analysed here took part in the selection. The paper's cohorts
therefore partition into

  selection-involved : TCGA, METABRIC, SCANB_GSE96058, SCANB_GSE202203
  selection-naive    : GSE20711, GSE58812 (overall survival)
                       GSE6532, GSE11121, GSE21653 (secondary endpoints)

Evaluating the PUBLISHED panel on the selection-naive cohorts gives an estimate that
required no re-selection and carries no selection optimism: those cohorts were invisible
to the search. To keep the training side clean as well, the model is trained only on the
other selection-naive cohorts wherever that is possible; the version trained on the
selection-involved cohorts is also reported, since training-side exposure inflates
nothing about the panel's identity.

Writes: selection_naive.csv, selection_naive_summary.json
"""
import json

import numpy as np
import pandas as pd

import nested_core as nc

ALPHA = 100.0
INVOLVED = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203"]
NAIVE_OS = ["GSE20711", "GSE58812"]
NAIVE_SEC = ["GSE6532", "GSE11121", "GSE21653"]

store = nc.load_all(INVOLVED + NAIVE_OS + NAIVE_SEC)

rows = []
for coh in NAIVE_OS + NAIVE_SEC:
    endpoint = "OS" if coh in NAIVE_OS else "DMFS/DFS"
    X, t, ev, _ = store[coh]
    genes_here = set(X.columns)

    for panel_name, panel in [("Novel5", nc.NOVEL5), ("Anchor4", nc.ANCHOR4)]:
        p = [g for g in panel if g in genes_here]
        if len(p) < len(panel):
            print("  %s: %s missing %s" % (coh, panel_name,
                                           set(panel) - genes_here), flush=True)
        if not p:
            continue

        # (a) trained on the selection-involved cohorts (training-side exposure only)
        tr_inv = [z for z in INVOLVED if set(p) <= set(store[z][0].columns)]
        c_inv, _ = nc.evaluate_panel(store, tr_inv, coh, p, alpha=ALPHA)

        # (b) trained on the OTHER selection-naive cohorts of the same endpoint --
        #     fully insulated from the selection cohorts on both sides
        peers = [z for z in (NAIVE_OS if endpoint == "OS" else NAIVE_SEC)
                 if z != coh and set(p) <= set(store[z][0].columns)]
        c_naive = np.nan
        if peers:
            c_naive, _ = nc.evaluate_panel(store, peers, coh, p, alpha=ALPHA)

        rows.append({"cohort": coh, "endpoint": endpoint, "panel": panel_name,
                     "n_genes_available": len(p), "n": len(t), "events": int(ev.sum()),
                     "cindex_trained_on_selection_cohorts": c_inv,
                     "train_cohorts_a": "|".join(tr_inv),
                     "cindex_trained_on_naive_peers": c_naive,
                     "train_cohorts_b": "|".join(peers)})
        print("%-12s %-9s %-8s n=%4d ev=%4d | C(train=involved)=%.4f | C(train=naive peers)=%s"
              % (coh, endpoint, panel_name, len(t), int(ev.sum()), c_inv,
                 "%.4f" % c_naive if np.isfinite(c_naive) else "n/a"), flush=True)

df = pd.DataFrame(rows)
df.to_csv("selection_naive.csv", index=False)

n5 = df[df.panel.eq("Novel5")]
os_only = n5[n5.endpoint.eq("OS")]
summary = {
    "selection_involved_cohorts": INVOLVED,
    "selection_naive_os_cohorts": NAIVE_OS,
    "selection_naive_secondary_cohorts": NAIVE_SEC,
    "novel5_mean_c_naive_os_trained_involved":
        float(os_only.cindex_trained_on_selection_cohorts.mean()),
    "novel5_mean_c_naive_all_trained_involved":
        float(n5.cindex_trained_on_selection_cohorts.mean()),
    "novel5_mean_c_naive_os_trained_naive_peers":
        float(os_only.cindex_trained_on_naive_peers.mean()),
    "anchor4_mean_c_naive_os":
        float(df[df.panel.eq("Anchor4") & df.endpoint.eq("OS")]
              .cindex_trained_on_selection_cohorts.mean()),
    "total_events_naive_os": int(os_only.events.sum()),
    "alpha": ALPHA,
}
json.dump(summary, open("selection_naive_summary.json", "w"), indent=2)
print("\n" + json.dumps(summary, indent=2), flush=True)
