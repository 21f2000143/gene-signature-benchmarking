"""
run_fixedform_scores.py -- REVIEW ITEM 7, the part the penalty table cannot reach.

run_comparator_penalty.py showed the HARMONISATION penalty is ~0.  The
reviewer's complaint has a second limb: we RE-FIT the comparators as bare gene
lists instead of applying them as the fixed algorithms they were published as.
We do not hold the published centroids or coefficients, so the published
algorithms cannot be reproduced.  What CAN be done, and is done here, is to
drop the re-fitting step: score every signature by a fixed, coefficient-free
rule that requires no training data, and see whether the comparators do better
without a Cox fit than with one.

If a comparator scores HIGHER under a fixed rule than under our ridge Cox, then
re-fitting was actively harmful and the benchmark understated it.  If it scores
lower or the same, re-fitting was not the problem.

FIXED RULES IMPLEMENTED (all on within-cohort z-scored expression, no training)
  mean_z          unweighted mean z-score across the signature's available
                  genes.  The coefficient-free analogue of a linear score.
  metagene_count  number of member genes above their within-cohort median,
                  divided by the number available.  This is the PUBLISHED form
                  of the Buffa hypoxia metagene, so for that signature this row
                  is the actual published algorithm, not an approximation.
  pc1             first principal component of the signature's gene block,
                  sign-oriented so that it correlates positively with mean_z.
                  A coefficient-free summary that, unlike mean_z, does not
                  assume every gene points the same way.
Note that mean_z and metagene_count assume all member genes are risk-increasing.
That is false for several signatures (PAM50 contains ESR1 and BCL2, protective,
alongside proliferation genes; MammaPrint contains both poor- and
good-prognosis genes).  Their published forms handle direction through
centroids or a template correlation, which we do not have.  This is stated as a
limitation of these rows rather than silently absorbed: for such signatures a
low mean_z c-index is expected and is NOT evidence about the published assay.
pc1 is the direction-agnostic row and is the fairest of the three.

For every rule, c-index is computed per cohort and the |c - 0.5| orientation is
NOT flipped -- a c-index below 0.5 is reported as-is, since flipping would use
the test outcome and inflate performance.

Comparison target: the LOCO ridge-Cox c-index from
comparator_coverage_penalty.csv (feature_definition == "harmonised"), the
number reported in the manuscript.

Genes: the per-cohort maximal available subset (every member gene present in
that cohort), since no training pool is needed and therefore no intersection is
forced.  This is the fairest possible coverage for the fixed rules.

OUTPUT comparator_fixedform.csv, comparator_fixedform_summary.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.getcwd())
import nested_core as nc
from nested_core import OS6, cindex

GENE_SETS = json.load(open("results/gene_sets.json"))
SETS = ["Novel5", "Anchor4", "Novel5_plus_Anchor4", "CNetCox6", "PAM50",
        "OncotypeDX21", "GGI", "MammaPrint70", "BuffaHypoxia"]
NBOOT = 1000


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


log("loading OS cohorts")
store = nc.load_all(OS6)

rows = []
for gsn in SETS:
    G = GENE_SETS[gsn]["genes"]
    for coh in OS6:
        X, t, ev, _ = store[coh]
        have = [g for g in G if g in X.columns]
        if len(have) < 2:
            continue
        B = X[have].values                      # already within-cohort z-scored
        scores = {}
        scores["mean_z"] = B.mean(axis=1)
        med = np.median(B, axis=0)
        scores["metagene_count"] = (B > med).sum(axis=1) / B.shape[1]
        Bc = B - B.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
            pc = U[:, 0] * S[0]
            if np.corrcoef(pc, scores["mean_z"])[0, 1] < 0:
                pc = -pc
            scores["pc1"] = pc
        except Exception:
            scores["pc1"] = np.full(B.shape[0], np.nan)
        for rule, sc in scores.items():
            if not np.all(np.isfinite(sc)):
                continue
            c = float(cindex(sc, t, ev))
            rng = np.random.default_rng(5)
            vals = []
            for _ in range(NBOOT):
                i = rng.integers(0, len(t), len(t))
                if ev[i].sum() >= 2:
                    vals.append(cindex(sc[i], t[i], ev[i]))
            rows.append(dict(gene_set=gsn, cohort=coh, rule=rule,
                             n_genes_published=len(G), n_genes_used=len(have),
                             cindex=c,
                             ci_lo=float(np.percentile(vals, 2.5)),
                             ci_hi=float(np.percentile(vals, 97.5)),
                             n=int(len(t)), events=int(ev.sum()),
                             training_required=False))
    log("fixed-form done %s" % gsn)

ff = pd.DataFrame(rows)
ff.to_csv("comparator_fixedform.csv", index=False)
log("wrote comparator_fixedform.csv rows=%d" % len(ff))

summ = {"note": "fixed rules need no training; compared against the LOCO "
                "ridge-Cox c-index under the harmonised feature set"}
piv = ff.pivot_table(index="gene_set", columns="rule", values="cindex")
if os.path.exists("comparator_coverage_penalty.csv"):
    pen = pd.read_csv("comparator_coverage_penalty.csv")
    ref = pen[pen.feature_definition == "harmonised"].groupby("gene_set").cindex.mean()
    piv["ridge_cox_LOCO"] = ref
    for r in ["mean_z", "metagene_count", "pc1"]:
        if r in piv:
            piv["gain_" + r] = piv[r] - piv["ridge_cox_LOCO"]
summ["mean_cindex_by_rule"] = {k: {kk: (None if pd.isna(v) else round(float(v), 4))
                                   for kk, v in row.items()}
                               for k, row in piv.iterrows()}
gaincols = [c for c in piv.columns if c.startswith("gain_")]
if gaincols:
    summ["any_comparator_improved_by_dropping_the_fit"] = bool(
        (piv.loc[[s for s in SETS if s not in ("Novel5", "Anchor4",
                                               "Novel5_plus_Anchor4")], gaincols] > 0.01).any().any())
json.dump(summ, open("comparator_fixedform_summary.json", "w"), indent=1, default=float)
print(piv.round(4).to_string(), flush=True)
