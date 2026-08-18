"""
run_comparator_penalty.py -- REVIEW ITEM 7 (comparator fairness under harmonisation).

The reviewer objects that PAM50 / OncotypeDX21 / GGI / MammaPrint70 lose to
clinicopathology in our hands, and suspects we disadvantaged them by (a)
intersecting their gene lists down to a common harmonised feature space and
(b) re-fitting them as bare gene lists with one fixed learner.  This script
quantifies (a) and probes the learner-penalty part of (b).  It does NOT and
cannot re-create the published algorithms (see comparator_implementation_caveats.csv).

EXACT DEFINITIONS
-----------------
Learner (pre-specified): ridge-penalised Cox partial likelihood, Breslow ties,
Newton-Raphson, alpha = 100 (nested_core.fit_ridge_cox).  Every gene is
z-scored WITHIN its cohort before use (nested_core.load_cohort).

Discrimination: Harrell's c-index on the held-out cohort (nested_core.cindex).

Design: leave-one-cohort-out (LOCO) over the six OS cohorts.  For held-out
cohort H, the model is fitted once on the row-concatenation of the other five
cohorts (each already within-cohort z-scored) and evaluated once on H.

Three feature-set definitions per comparator gene set G:

  (i)  "harmonised"  -- G_h = G  intersect  genes(c) for EVERY c in the six OS
       cohorts.  This is the feature set the published benchmark actually used;
       it is identical in every fold.  Reproduces loco_os.csv n_genes_used.

  (ii) "maximal_feasible" -- G_m(H) = G intersect genes(H) intersect
       [ intersect over the five TRAINING cohorts of genes(c) ].
       The common-intersection constraint is dropped for the four cohorts that
       are neither the test cohort nor... (all five training cohorts still must
       carry the gene, because the model has to be fitted on them).  In
       practice this relaxes the constraint imposed by whichever cohort is held
       out.  No imputation.  This is the PRIMARY definition of "maximal".

  (iii) "maximal_imputed" -- G_i(H) = G intersect genes(H).  Every gene present
       in the test cohort is used; a training cohort that lacks a gene
       contributes the value 0 for it, which is that cohort's mean on the
       within-cohort z-scale (mean imputation).  This is the most generous
       possible reading of "use every gene present in that cohort".

HARMONISATION PENALTY = c_index(ii) - c_index(i)   [also reported for (iii)].
A positive penalty means harmonisation cost the comparator discrimination.

SUPPLEMENTARY (plan step 1: is the loss about transfer or about the lists?)
  * alpha sweep: LOCO c-index at alpha in {1, 10, 100, 1000, 10000} for each
    comparator under the harmonised feature set, to test whether alpha = 100
    over- or under-penalises the large (46-58 gene) signatures relative to the
    5-9 gene panels.  If the large signatures peak at a different alpha, the
    single pre-specified alpha is part of the story.
  * within-cohort discrimination: 5-fold CV repeated 4 times (seed 0..3) inside
    each cohort separately, same learner, same alpha grid point 100.  This
    separates "the gene list carries no prognostic signal here" (low
    within-cohort c) from "the model does not transfer across platforms"
    (decent within-cohort c, low LOCO c).

Bootstrap CI: 1000 resamples of the test cohort's subjects, percentile 2.5/97.5.

OUTPUTS
  comparator_coverage_penalty.csv   -- one row per (gene_set, held_out_cohort,
                                       feature_definition)
  comparator_alpha_sweep.csv        -- one row per (gene_set, cohort, alpha)
  comparator_within_cohort.csv      -- one row per (gene_set, cohort)
  comparator_penalty_summary.json   -- means and the overtakes-Clinical check
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import nested_core as nc
from nested_core import OS6, fit_ridge_cox, cindex

ALPHA = 100.0
NBOOT = 1000
GENE_SETS = json.load(open("results/gene_sets.json"))
COMPARATORS = ["CNetCox6", "PAM50", "OncotypeDX21", "GGI", "MammaPrint70", "BuffaHypoxia"]
OURS = ["Novel5", "Anchor4", "Novel5_plus_Anchor4"]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def boot_ci(risk, t, ev, n_boot=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(t)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if ev[idx].sum() < 2:
            continue
        vals.append(cindex(risk[idx], t[idx], ev[idx]))
    if len(vals) < 50:
        return np.nan, np.nan, len(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


def build(store, cohorts, genes):
    """Row-stack cohorts over `genes`; genes absent in a cohort get 0 (= that
    cohort's within-cohort z-scale mean)."""
    Xs, ts, es = [], [], []
    for c in cohorts:
        X, t, ev, _ = store[c]
        M = np.zeros((X.shape[0], len(genes)))
        have = [g for g in genes if g in X.columns]
        if have:
            M[:, [genes.index(g) for g in have]] = X[have].values
        Xs.append(M); ts.append(t); es.append(ev)
    return np.vstack(Xs), np.concatenate(ts), np.concatenate(es).astype(np.int32)


def loco_eval(store, held, genes, alpha=ALPHA, seed=0):
    tr = [c for c in OS6 if c != held]
    Xtr, ttr, etr = build(store, tr, genes)
    Xte, tte, ete = build(store, [held], genes)
    beta = fit_ridge_cox(Xtr, ttr, etr, alpha=alpha)
    risk = Xte @ beta
    c = cindex(risk, tte, ete)
    lo, hi, nb = boot_ci(risk, tte, ete, seed=seed)
    return dict(cindex=float(c), ci_lo=lo, ci_hi=hi, n_boot_ok=nb,
                n_train=int(Xtr.shape[0]), events_train=int(etr.sum()),
                n_test=int(Xte.shape[0]), events_test=int(ete.sum()))


def within_cohort(store, coh, genes, alpha=ALPHA, n_splits=5, n_repeats=4):
    """Repeated stratified-by-event K-fold inside one cohort."""
    X, t, ev, _ = store[coh]
    have = [g for g in genes if g in X.columns]
    if len(have) == 0:
        return None
    M = X[have].values
    n = len(t)
    cs = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(1000 + rep)
        order = rng.permutation(n)
        folds = np.array_split(order, n_splits)
        for f in folds:
            mask = np.ones(n, bool); mask[f] = False
            if ev[mask].sum() < 5 or ev[f].sum() < 2:
                continue
            b = fit_ridge_cox(M[mask], t[mask], ev[mask], alpha=alpha)
            cs.append(cindex(M[f] @ b, t[f], ev[f]))
    if not cs:
        return None
    return dict(n_genes_used=len(have), cindex_mean=float(np.mean(cs)),
                cindex_sd=float(np.std(cs)), n_folds=len(cs))


# ------------------------------------------------------------------ run
log("loading OS cohorts")
store = nc.load_all(OS6)
gene_cols = {c: set(store[c][0].columns) for c in OS6}

rows = []
for gsn in OURS + COMPARATORS:
    G = GENE_SETS[gsn]["genes"]
    inter6 = sorted(set(G) & set.intersection(*[gene_cols[c] for c in OS6]))
    for held in OS6:
        tr = [c for c in OS6 if c != held]
        present_here = sorted(set(G) & gene_cols[held])
        feas = sorted(set(present_here) & set.intersection(*[gene_cols[c] for c in tr]))
        defs = [("harmonised", inter6, False),
                ("maximal_feasible", feas, False),
                ("maximal_imputed", present_here, True)]
        for name, genes, imputed in defs:
            if len(genes) == 0:
                continue
            r = loco_eval(store, held, list(genes), alpha=ALPHA)
            r.update(gene_set=gsn, family=GENE_SETS[gsn].get("family", ""),
                     held_out_cohort=held, feature_definition=name,
                     n_genes_published=len(G),
                     n_genes_present_in_cohort=len(present_here),
                     n_genes_used=len(genes),
                     frac_lost_vs_published=round(1 - len(genes) / len(G), 4),
                     frac_lost_by_harmonisation=round(
                         (len(present_here) - len(genes)) / len(G), 4),
                     train_imputation=imputed, alpha=ALPHA,
                     genes_used="|".join(genes))
            rows.append(r)
        log("%-20s %-18s harm=%d feas=%d present=%d"
            % (gsn, held, len(inter6), len(feas), len(present_here)))

pen = pd.DataFrame(rows)
pen.to_csv("comparator_coverage_penalty.csv", index=False)
log("wrote comparator_coverage_penalty.csv  rows=%d" % len(pen))

# ---- alpha sweep (harmonised feature set only)
sweep = []
for gsn in OURS + COMPARATORS:
    G = GENE_SETS[gsn]["genes"]
    inter6 = sorted(set(G) & set.intersection(*[gene_cols[c] for c in OS6]))
    for a in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        for held in OS6:
            tr = [c for c in OS6 if c != held]
            Xtr, ttr, etr = build(store, tr, inter6)
            Xte, tte, ete = build(store, [held], inter6)
            b = fit_ridge_cox(Xtr, ttr, etr, alpha=a)
            sweep.append(dict(gene_set=gsn, alpha=a, held_out_cohort=held,
                              n_genes_used=len(inter6),
                              cindex=float(cindex(Xte @ b, tte, ete))))
    log("alpha sweep done %s" % gsn)
sw = pd.DataFrame(sweep)
sw.to_csv("comparator_alpha_sweep.csv", index=False)
log("wrote comparator_alpha_sweep.csv")

# ---- within-cohort
wc = []
for gsn in OURS + COMPARATORS:
    G = GENE_SETS[gsn]["genes"]
    for coh in OS6:
        r = within_cohort(store, coh, G)
        if r:
            r.update(gene_set=gsn, cohort=coh, alpha=ALPHA,
                     n_genes_published=len(G))
            wc.append(r)
w = pd.DataFrame(wc)
w.to_csv("comparator_within_cohort.csv", index=False)
log("wrote comparator_within_cohort.csv")

# ---- summary
summ = {"alpha": ALPHA, "n_boot": NBOOT, "definition_note":
        "penalty = maximal_feasible - harmonised, mean over the six OS cohorts"}
p = pen.pivot_table(index="gene_set", columns="feature_definition", values="cindex")
p["penalty_feasible"] = p["maximal_feasible"] - p["harmonised"]
p["penalty_imputed"] = p["maximal_imputed"] - p["harmonised"]
summ["mean_loco_cindex"] = {k: {kk: (None if pd.isna(vv) else round(float(vv), 4))
                                for kk, vv in v.items()} for k, v in p.iterrows()}
g = pen.groupby(["gene_set", "feature_definition"]).agg(
    mean_n_genes_used=("n_genes_used", "mean"),
    mean_frac_lost=("frac_lost_vs_published", "mean")).reset_index()
summ["coverage"] = g.to_dict("records")
sw_best = sw.groupby(["gene_set", "alpha"])["cindex"].mean().reset_index()
summ["alpha_sweep_best"] = {k: {"best_alpha": float(v.loc[v.cindex.idxmax(), "alpha"]),
                                "best_mean_c": round(float(v.cindex.max()), 4),
                                "c_at_alpha100": round(float(
                                    v.loc[v.alpha == 100.0, "cindex"].iloc[0]), 4)}
                            for k, v in sw_best.groupby("gene_set")}
json.dump(summ, open("comparator_penalty_summary.json", "w"), indent=1, default=float)
log("wrote comparator_penalty_summary.json")
print(p.round(4).to_string(), flush=True)
