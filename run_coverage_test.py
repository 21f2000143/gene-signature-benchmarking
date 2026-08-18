"""
run_coverage_test.py -- REVIEW ITEM 8 (test, do not assert, the coverage story).

The manuscript ASSERTS that Novel5's weaker secondary-endpoint (DMFS/DFS)
performance is caused by platform gene coverage: GSE6532 and GSE11121 are
13,039-gene arrays carrying only FLT3 and P4HA2 of the panel, while GSE21653
(21,355 genes) carries all five.  That confounds coverage with cohort and with
endpoint.  This script separates them.

FACTS ESTABLISHED BY PROBE (probe_cohorts.json), re-verified here:
  GSE6532  n=380 ev=96  13039 genes  panel genes present: FLT3, P4HA2
  GSE11121 n=200 ev=46  13039 genes  panel genes present: FLT3, P4HA2
  GSE21653 n=248 ev=79  21355 genes  panel genes present: all five
  all six OS cohorts carry all five.

THE THREE COMPARISONS
---------------------
A. BETWEEN-COHORT (what the manuscript currently leans on).  Novel5 LOCO
   c-index on each secondary cohort using whatever panel genes that cohort
   carries.  Confounded: cohort, endpoint definition and coverage all differ.

B. WITHIN-COHORT COVERAGE CONTRAST -- the decisive test.  Hold the cohort, the
   endpoint, the samples, the learner and the LOCO design FIXED and vary ONLY
   the gene set:
       full5  = FLT3, CLIC6, SUSD3, ZIC2, P4HA2
       sub2   = FLT3, P4HA2                       (the U133A-measurable subset)
   evaluated on GSE21653 (secondary endpoint) and on all six OS cohorts.
   Delta = c(full5) - c(sub2) is the causal effect of losing CLIC6, SUSD3 and
   ZIC2, with coverage the only thing that changed.
   If Delta is small, coverage CANNOT explain a large secondary-endpoint drop
   and the manuscript's assertion is unsupported.

C. QUANTITATIVE ADEQUACY.  Compare the observed secondary-endpoint drop
       drop_observed = mean c(Novel5, six OS cohorts) - mean c(Novel5, GSE6532
                       and GSE11121)
   against the coverage effect measured in B
       drop_attributable_to_coverage = mean over the six OS cohorts of
                       [ c(full5) - c(sub2) ]
   The share = drop_attributable_to_coverage / drop_observed is the fraction of
   the secondary-endpoint loss that gene coverage can account for.  A share
   near 1 supports the manuscript; a share near 0 refutes it.

Additional context columns: the same full5-vs-sub2 contrast for the 9-gene
anchored panel, and Novel5 restricted to the genes common to the U133A arrays
across ALL cohorts, so the reader can see the effect is not specific to one
gene pair.  Anchor4 (the near-uninformative scaffold) is included as a floor.

DESIGN DETAILS
  Learner: ridge Cox alpha=100, Breslow ties (nested_core.fit_ridge_cox),
  every gene z-scored within cohort.  c-index: Harrell's.
  OS cohorts: LOCO -- train on the pooled other five OS cohorts, test once on
  the held-out cohort.
  Secondary cohorts: LOCO within the three secondary cohorts, matching the
  published loco_secondary.csv protocol (train on the other two, test on the
  held-out one).  Where a training cohort lacks a gene it contributes 0 for it
  (its within-cohort z-scale mean); this is stated per row in
  `train_imputation_used`.
  Also reported: a TRANSFER variant for the secondary cohorts (train on the
  pooled six OS cohorts, test on the secondary cohort), which removes the tiny
  secondary training pool as an alternative explanation.

Bootstrap: 2000 resamples of the test cohort, percentile 2.5/97.5.
Paired test: for each cohort, the same 2000 bootstrap resamples are used for
full5 and sub2, giving a paired delta distribution and a two-sided empirical
p-value for delta = 0.

OUTPUTS
  coverage_test.csv            -- one row per (panel_variant, cohort, design)
  coverage_test_summary.json   -- the deltas, the share, and the verdict
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import nested_core as nc
from nested_core import OS6, SEC3, NOVEL5, ANCHOR4, fit_ridge_cox, cindex

ALL9 = OS6 + SEC3
ALPHA = 100.0
NBOOT = 2000
SUB2 = ["FLT3", "P4HA2"]
MISSING3 = ["CLIC6", "SUSD3", "ZIC2"]

VARIANTS = {
    "Novel5_full5": NOVEL5,
    "Novel5_sub2_FLT3_P4HA2": SUB2,
    "Novel5_missing3_CLIC6_SUSD3_ZIC2": MISSING3,
    "Novel9_full": NOVEL5 + ANCHOR4,
    "Novel9_sub2plusAnchor": SUB2 + ANCHOR4,
    "Anchor4_only": ANCHOR4,
}


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


log("loading all nine cohorts")
store = nc.load_all(ALL9)
GC = {c: set(store[c][0].columns) for c in ALL9}

for c in ALL9:
    log("%-18s genes=%6d panel_present=%s" % (
        c, len(GC[c]), ",".join(g for g in NOVEL5 if g in GC[c])))


def build(cohorts, genes):
    Xs, ts, es = [], [], []
    imputed = False
    for c in cohorts:
        X, t, ev, _ = store[c]
        M = np.zeros((X.shape[0], len(genes)))
        have = [g for g in genes if g in X.columns]
        if len(have) < len(genes):
            imputed = True
        if have:
            M[:, [genes.index(g) for g in have]] = X[have].values
        Xs.append(M); ts.append(t); es.append(ev)
    return (np.vstack(Xs), np.concatenate(ts),
            np.concatenate(es).astype(np.int32), imputed)


def eval_design(test_cohort, train_cohorts, genes, seed=0):
    Xtr, ttr, etr, imp_tr = build(train_cohorts, genes)
    Xte, tte, ete, imp_te = build([test_cohort], genes)
    b = fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)
    risk = Xte @ b
    return dict(risk=risk, t=tte, ev=ete, cindex=float(cindex(risk, tte, ete)),
                n_train=int(Xtr.shape[0]), events_train=int(etr.sum()),
                n_test=int(Xte.shape[0]), events_test=int(ete.sum()),
                train_imputation_used=bool(imp_tr),
                n_genes_present_in_test=int(sum(g in GC[test_cohort] for g in genes)))


def boot_pair(rA, rB, t, ev, seed=0):
    """Paired bootstrap of c(A)-c(B) on identical resamples."""
    rng = np.random.default_rng(seed)
    n = len(t)
    a, b, d = [], [], []
    for _ in range(NBOOT):
        i = rng.integers(0, n, n)
        if ev[i].sum() < 2:
            continue
        ca = cindex(rA[i], t[i], ev[i])
        cb = cindex(rB[i], t[i], ev[i])
        a.append(ca); b.append(cb); d.append(ca - cb)
    a, b, d = np.array(a), np.array(b), np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(ciA=(float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))),
                ciB=(float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))),
                delta_mean=float(d.mean()),
                delta_ci=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
                p_paired=float(min(1.0, p)), n_boot_ok=int(len(d)))


rows = []
cache = {}
for vname, genes in VARIANTS.items():
    # --- OS cohorts, LOCO within OS6
    for held in OS6:
        tr = [c for c in OS6 if c != held]
        r = eval_design(held, tr, genes)
        cache[(vname, held, "LOCO_OS6")] = r
        rows.append(dict(panel_variant=vname, genes="|".join(genes),
                         n_genes_nominal=len(genes), cohort=held, endpoint="OS",
                         design="LOCO_OS6", train_cohorts="|".join(tr),
                         **{k: v for k, v in r.items() if k not in ("risk", "t", "ev")}))
    # --- secondary cohorts, LOCO within SEC3 (published protocol)
    for held in SEC3:
        tr = [c for c in SEC3 if c != held]
        r = eval_design(held, tr, genes)
        cache[(vname, held, "LOCO_SEC3")] = r
        rows.append(dict(panel_variant=vname, genes="|".join(genes),
                         n_genes_nominal=len(genes), cohort=held,
                         endpoint="DMFS/DFS", design="LOCO_SEC3",
                         train_cohorts="|".join(tr),
                         **{k: v for k, v in r.items() if k not in ("risk", "t", "ev")}))
        # --- transfer: train on all six OS cohorts, test on the secondary cohort
        r2 = eval_design(held, OS6, genes)
        cache[(vname, held, "TRANSFER_OS6_to_SEC")] = r2
        rows.append(dict(panel_variant=vname, genes="|".join(genes),
                         n_genes_nominal=len(genes), cohort=held,
                         endpoint="DMFS/DFS", design="TRANSFER_OS6_to_SEC",
                         train_cohorts="|".join(OS6),
                         **{k: v for k, v in r2.items() if k not in ("risk", "t", "ev")}))
    log("variant %s done" % vname)

df = pd.DataFrame(rows)

# ---- paired full5 vs sub2, same cohort / same resamples
pairs = []
for design, cohorts in (("LOCO_OS6", OS6), ("LOCO_SEC3", SEC3),
                        ("TRANSFER_OS6_to_SEC", SEC3)):
    for coh in cohorts:
        A = cache[("Novel5_full5", coh, design)]
        B = cache[("Novel5_sub2_FLT3_P4HA2", coh, design)]
        bp = boot_pair(A["risk"], B["risk"], A["t"], A["ev"], seed=17)
        pairs.append(dict(cohort=coh, design=design,
                          endpoint="OS" if coh in OS6 else "DMFS/DFS",
                          n_panel_genes_present=int(sum(g in GC[coh] for g in NOVEL5)),
                          c_full5=A["cindex"], c_sub2=B["cindex"],
                          delta_full5_minus_sub2=A["cindex"] - B["cindex"],
                          delta_boot_mean=bp["delta_mean"],
                          delta_ci_lo=bp["delta_ci"][0], delta_ci_hi=bp["delta_ci"][1],
                          p_paired=bp["p_paired"], n_boot_ok=bp["n_boot_ok"],
                          events_test=A["events_test"], n_test=A["n_test"]))
pairdf = pd.DataFrame(pairs)
df.to_csv("coverage_test.csv", index=False)
pairdf.to_csv("coverage_test_paired.csv", index=False)
log("wrote coverage_test.csv and coverage_test_paired.csv")

# ---------------------------------------------------------------- verdict
def mc(v, design, cohorts):
    s = df[(df.panel_variant == v) & (df.design == design) & (df.cohort.isin(cohorts))]
    return float(s.cindex.mean())


LOWCOV = ["GSE6532", "GSE11121"]
c_os_full = mc("Novel5_full5", "LOCO_OS6", OS6)
c_sec_low = mc("Novel5_full5", "LOCO_SEC3", LOWCOV)          # uses only FLT3,P4HA2 there
c_sec_full = mc("Novel5_full5", "LOCO_SEC3", ["GSE21653"])
drop_observed = c_os_full - c_sec_low
cov_effect_os = float(pairdf[(pairdf.design == "LOCO_OS6")].delta_full5_minus_sub2.mean())
cov_effect_21653 = float(pairdf[(pairdf.design == "LOCO_SEC3") &
                                (pairdf.cohort == "GSE21653")].delta_full5_minus_sub2.iloc[0])
share = cov_effect_os / drop_observed if abs(drop_observed) > 1e-9 else float("nan")

# does the coverage effect explain the drop?  Criterion stated explicitly:
#   coverage is accepted as THE explanation only if the within-cohort effect of
#   dropping the three genes accounts for >= 50% of the observed drop AND is
#   itself statistically distinguishable from zero in a majority of cohorts.
sig = int((pairdf[pairdf.design == "LOCO_OS6"].p_paired < 0.05).sum())
explains = bool(share >= 0.5 and sig >= 4)

summary = {
    "definitions": {
        "drop_observed": "mean c(Novel5, LOCO over six OS cohorts) - mean c(Novel5, LOCO_SEC3 on GSE6532 and GSE11121)",
        "coverage_effect_within_cohort": "mean over the six OS cohorts of c(FLT3,CLIC6,SUSD3,ZIC2,P4HA2) - c(FLT3,P4HA2), identical cohort/design/learner",
        "share": "coverage_effect_within_cohort / drop_observed",
        "verdict_criterion": "coverage accepted as the explanation only if share >= 0.50 AND the paired full5-vs-sub2 delta is significant (p<0.05) in at least 4 of the 6 OS cohorts",
    },
    "coverage_facts": {c: {"n_genes": len(GC[c]),
                           "panel_genes_present": [g for g in NOVEL5 if g in GC[c]],
                           "panel_genes_missing": [g for g in NOVEL5 if g not in GC[c]]}
                       for c in ALL9},
    "c_novel5_OS6_mean": round(c_os_full, 4),
    "c_novel5_GSE21653_all5_present": round(c_sec_full, 4),
    "c_novel5_low_coverage_mean_GSE6532_GSE11121": round(c_sec_low, 4),
    "drop_observed": round(drop_observed, 4),
    "coverage_effect_within_OS6_mean": round(cov_effect_os, 4),
    "coverage_effect_within_GSE21653": round(cov_effect_21653, 4),
    "share_of_drop_explained_by_coverage": round(share, 4),
    "n_OS_cohorts_with_significant_full5_vs_sub2": sig,
    "per_cohort_paired": pairdf.to_dict("records"),
    "coverage_explains_secondary_drop": explains,
}
json.dump(summary, open("coverage_test_summary.json", "w"), indent=1, default=float)
log("wrote coverage_test_summary.json")
print(pairdf.round(4).to_string(), flush=True)
print("\ndrop_observed=%.4f  coverage_effect=%.4f  share=%.3f  explains=%s"
      % (drop_observed, cov_effect_os, share, explains), flush=True)
