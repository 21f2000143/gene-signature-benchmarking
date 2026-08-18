"""
run_nested_selection.py -- reviewer item 1.1: FULLY NESTED re-selection.

For each of the six OS cohorts in turn:
    * hold that cohort out entirely
    * run the COMPLETE forward search using only the remaining five cohorts
      (the inner LOCO scoring loop runs among those five and never touches
       the held-out cohort)
    * evaluate the panel that run produces on the held-out cohort, with the
      pre-specified ridge Cox
This yields an unbiased estimate of the *procedure*, which is the only number that
supports a claim about a selected panel. Also reported: how much each fold's panel
overlaps the published five, and -- for reference on the same folds -- the c-index of
the published panel itself (which is the biased number, since the published panel saw
these cohorts during its own selection).

Writes: nested_selection_folds.csv, nested_selection_traces.csv,
        nested_selection_summary.json
"""
import json
import time

import numpy as np
import pandas as pd

import nested_core as nc

ALPHA = 100.0
N_STEPS = 5
OUTER = nc.OS6

t0 = time.time()
store = nc.load_all(OUTER)
genes_all = nc.common_genes(store, OUTER)
print("genes common to all six OS cohorts: %d" % len(genes_all), flush=True)

rows, traces, traces_consensus = [], [], []
for held in OUTER:
    inner = [c for c in OUTER if c != held]
    print("\n=== outer fold: held-out %s | search cohorts %s ===" % (held, ",".join(inner)),
          flush=True)
    # candidate universe restricted to genes measurable in the search cohorts AND the
    # held-out cohort, so the selected panel is always evaluable (no coverage advantage)
    genes = nc.common_genes(store, OUTER)
    sel, info = nc.forward_search(store, inner, genes, n_steps=N_STEPS,
                                  anchor=nc.ANCHOR4, alpha=ALPHA, verbose=True)

    Xte, tte, ete, _ = store[held]
    c_nested, _ = nc.evaluate_panel(store, inner, held, sel, alpha=ALPHA)
    # for reference on the same fold: the PUBLISHED panel (biased -- it saw this cohort)
    c_published, _ = nc.evaluate_panel(store, inner, held, nc.NOVEL5, alpha=ALPHA)
    # anchor-only, same fold, as the mechanistic negative control
    c_anchor, _ = nc.evaluate_panel(store, inner, held, nc.ANCHOR4, alpha=ALPHA)
    trace = info["trace"]

    ov = sorted(set(sel) & set(nc.NOVEL5))
    print("  panel: %s" % ",".join(sel), flush=True)
    print("  nested C=%.4f | published-panel C=%.4f | anchor C=%.4f | overlap %d/5 %s"
          % (c_nested, c_published, c_anchor, len(ov), ov), flush=True)

    rows.append({"held_out_cohort": held, "n_test": len(tte), "events_test": int(ete.sum()),
                 "search_cohorts": "|".join(inner),
                 "selected_panel": "|".join(sel),
                 "cindex_nested": c_nested,
                 "cindex_published_panel": c_published,
                 "cindex_anchor_only": c_anchor,
                 "overlap_with_published": len(ov),
                 "overlap_genes": "|".join(ov),
                 "per_fold_panels": " ; ".join(
                     "%s:%s" % (h, ",".join(p)) for h, p in
                     zip(info["inner_held_out"], info["per_fold_panels"])),
                 "pool_size_median": float(np.median(info["pool_sizes"])),
                 "n_candidates": len(genes), "alpha": ALPHA, "n_steps": N_STEPS})
    for r in trace:
        traces.append({"held_out_cohort": held, "inner_held_out": r["inner_held_out"],
                       "step": r["step"], "chosen": r["chosen"],
                       "inner_heldout_c": r["score"], "pool_size": r["pool_size"],
                       "n_candidates": r["n_candidates"],
                       "top10": ";".join("%s:%.4f" % (g, s) for g, s in r["top10"])})
    for rec in info["consensus"]:
        rec["held_out_cohort"] = held
    traces_consensus.extend(info["consensus"])

folds = pd.DataFrame(rows)
folds.to_csv("nested_selection_folds.csv", index=False)
pd.DataFrame(traces).to_csv("nested_selection_traces.csv", index=False)
pd.DataFrame(traces_consensus).to_csv("nested_selection_consensus.csv", index=False)

# gene-level stability across folds
from collections import Counter
cnt = Counter()
for r in rows:
    cnt.update(r["selected_panel"].split("|"))
stab = pd.DataFrame([{"gene": g, "n_folds_selected": n, "frac_folds": n / len(rows),
                      "in_published_panel": g in nc.NOVEL5}
                     for g, n in cnt.most_common()])
stab.to_csv("nested_selection_stability.csv", index=False)

summary = {
    "mean_cindex_nested": float(folds.cindex_nested.mean()),
    "sd_cindex_nested": float(folds.cindex_nested.std()),
    "mean_cindex_published_panel_same_folds": float(folds.cindex_published_panel.mean()),
    "mean_cindex_anchor_only": float(folds.cindex_anchor_only.mean()),
    "optimism_published_minus_nested": float(folds.cindex_published_panel.mean()
                                             - folds.cindex_nested.mean()),
    "mean_overlap_with_published": float(folds.overlap_with_published.mean()),
    "n_distinct_genes_selected": int(len(cnt)),
    "genes_selected_in_all_folds": [g for g, n in cnt.items() if n == len(rows)],
    "n_candidates": int(len(genes_all)),
    "alpha": ALPHA, "n_steps": N_STEPS, "outer_cohorts": OUTER,
    "elapsed_min": (time.time() - t0) / 60,
}
json.dump(summary, open("nested_selection_summary.json", "w"), indent=2)
print("\n" + json.dumps(summary, indent=2), flush=True)
