"""
run_clinical_supplement.py -- REVIEW ITEM 9, continued.

run_clinical_arm.py established that the strict intersection of usable
harmonised covariates over ALL SIX OS cohorts is EMPTY: GSE58812 (TNBC, n=107)
carries age alone, and GSE20711 (n=88) has age recorded as an unlabelled 0/1
dichotomy rather than years, so no single covariate is usable in all six.  A
"Clinical" arm that means the same thing in every OS cohort therefore does not
exist.  This script reports the next-best consistent definitions rather than
leaving the arm undefined, and it separates the cohorts by grade availability
as the revision plan requires.

DEFINITIONS ADDED (learner and metric identical to run_clinical_arm.py:
ridge Cox alpha=100, within-cohort z-scored covariates, Harrell's c-index,
LOCO with the training pool stated per row)

  clinical_age_only            age_years alone; LOCO over the five OS cohorts
                               where age is on the native years scale
                               (all but GSE20711).  The single most widely
                               available covariate -- the floor for "clinical".
  clinical_common4_with_grade  grade3|node_pos|er_pos, the intersection over the
                               four OS cohorts that carry grade (METABRIC,
                               SCANB_GSE96058, SCANB_GSE202203, GSE20711);
                               LOCO within those four.  This is the largest
                               internally consistent clinical arm available.
  clinical_common4_full        age_years|grade3|node_pos|size_gt20|er_pos|pr_pos,
                               the intersection over the four richly annotated
                               cohorts (METABRIC, both SCANB, and -- for the
                               secondary endpoint -- GSE6532/GSE21653);
                               LOCO within METABRIC + both SCANB only, since
                               TCGA lacks grade and size.
  clinical_no_grade_cohorts    the two OS cohorts with no grade at all (TCGA,
                               GSE58812) reported separately, each on its own
                               available covariates, so the manuscript can state
                               plainly that these two cohorts' "Clinical"
                               numbers are not comparable to the other four.

Also computed, on the SAME folds, for the four grade-bearing cohorts:
  gene:Novel5                  the panel alone
  clinical_common4_with_grade  the consistent clinical arm
  combined_Novel5_plus_clinical  the panel and that clinical arm together
so incremental value can be read off a like-for-like comparison.  Delta
(combined - clinical) with a paired bootstrap over the held-out cohort is the
panel's added value over clinicopathology when clinicopathology means one
fixed thing.

OUTPUT clinical_arm_supplement.csv, clinical_arm_supplement_summary.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.getcwd())
import nested_core as nc
from nested_core import OS6, SEC3, fit_ridge_cox, cindex
from clin_harmonise import harmonise_clin, usable, zscore, HARM_COVS   # reuse, do not re-derive

# guarantee the reused definitions have not drifted from the ones that produced
# clinical_arm_audit.csv / clinical_arm_recomputed.csv
_a = open(os.path.join(HERE, "run_clinical_arm.py")).read()
_b = open(os.path.join(HERE, "clin_harmonise.py")).read()
_blk = _a[_a.index("def _num(s):"):_a.index("def boot_ci(")].rstrip()
assert _blk in _b, "clin_harmonise.py has drifted from run_clinical_arm.py"
print("covariate-harmonisation drift check OK (%d chars shared)" % len(_blk), flush=True)

ALPHA = 100.0
NBOOT = 2000
ALL9 = OS6 + SEC3
GENE_SETS = json.load(open("results/gene_sets.json"))
NOVEL5 = GENE_SETS["Novel5"]["genes"]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


log("loading cohorts")
store = nc.load_all(ALL9)
H, AVAIL = {}, {}
for coh in ALL9:
    X, t, ev, s = store[coh]
    H[coh] = harmonise_clin(coh, s)
    AVAIL[coh] = {c: usable(H[coh][c], len(H[coh])) for c in HARM_COVS}

GRADE_COHORTS = [c for c in OS6 if AVAIL[c]["grade3"]]
NOGRADE_COHORTS = [c for c in OS6 if not AVAIL[c]["grade3"]]
AGE_COHORTS = [c for c in OS6 if AVAIL[c]["age_years"]]
RICH = ["METABRIC", "SCANB_GSE96058", "SCANB_GSE202203"]
log("grade cohorts: %s | no-grade: %s | age cohorts: %s"
    % (GRADE_COHORTS, NOGRADE_COHORTS, AGE_COHORTS))

COMMON4_GRADE = [c for c in HARM_COVS if all(AVAIL[k][c] for k in GRADE_COHORTS)]
COMMON_RICH = [c for c in HARM_COVS if all(AVAIL[k][c] for k in RICH)]
log("COMMON4_GRADE=%s  COMMON_RICH=%s" % (COMMON4_GRADE, COMMON_RICH))


def clin_mat(coh, covs):
    return np.column_stack([zscore(H[coh][c].values) for c in covs])


def gene_mat(coh, genes):
    X = store[coh][0]
    return X[[g for g in genes if g in X.columns]].values


def stack(cohorts, fn):
    Xs, ts, es = [], [], []
    for c in cohorts:
        Xs.append(fn(c)); ts.append(store[c][1]); es.append(store[c][2])
    return np.vstack(Xs), np.concatenate(ts), np.concatenate(es).astype(np.int32)


def run(held, pool, fn, label, covlabel):
    tr = [c for c in pool if c != held]
    Xtr, ttr, etr = stack(tr, fn)
    Xte = fn(held); tte = store[held][1]; ete = np.asarray(store[held][2], np.int32)
    b = fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)
    risk = Xte @ b
    rng = np.random.default_rng(11)
    vals = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(tte), len(tte))
        if ete[i].sum() >= 2:
            vals.append(cindex(risk[i], tte[i], ete[i]))
    return dict(arm=label, held_out_cohort=held, covariates=covlabel,
                n_features=int(Xtr.shape[1]), cindex=float(cindex(risk, tte, ete)),
                ci_lo=float(np.percentile(vals, 2.5)), ci_hi=float(np.percentile(vals, 97.5)),
                n_train=int(Xtr.shape[0]), events_train=int(etr.sum()),
                n_test=int(len(tte)), events_test=int(ete.sum()),
                train_pool="|".join(tr), alpha=ALPHA), risk, tte, ete


rows, risks = [], {}
for held in AGE_COHORTS:
    r, rk, tt, ee = run(held, AGE_COHORTS, lambda c: clin_mat(c, ["age_years"]),
                        "clinical_age_only", "age_years")
    rows.append(r)
for held in GRADE_COHORTS:
    for label, covs in [("clinical_common4_with_grade", COMMON4_GRADE)]:
        r, rk, tt, ee = run(held, GRADE_COHORTS, lambda c: clin_mat(c, covs),
                            label, "|".join(covs))
        rows.append(r); risks[(label, held)] = (rk, tt, ee)
    r, rk, tt, ee = run(held, GRADE_COHORTS, lambda c: gene_mat(c, NOVEL5),
                        "gene_Novel5_on_grade_cohorts", "|".join(NOVEL5))
    rows.append(r); risks[("gene", held)] = (rk, tt, ee)
    r, rk, tt, ee = run(held, GRADE_COHORTS,
                        lambda c: np.hstack([gene_mat(c, NOVEL5), clin_mat(c, COMMON4_GRADE)]),
                        "combined_Novel5_plus_clinical",
                        "|".join(NOVEL5) + "||" + "|".join(COMMON4_GRADE))
    rows.append(r); risks[("comb", held)] = (rk, tt, ee)
for held in RICH:
    r, _, _, _ = run(held, RICH, lambda c: clin_mat(c, COMMON_RICH),
                     "clinical_common_rich3", "|".join(COMMON_RICH))
    rows.append(r)
for held in NOGRADE_COHORTS:
    covs = [c for c in HARM_COVS if AVAIL[held][c]]
    r, _, _, _ = run(held, OS6, lambda c: clin_mat(c, covs),
                     "clinical_no_grade_cohort_own_covars", "|".join(covs))
    r["note"] = "cohort has NO grade; arm not comparable to the grade-bearing cohorts"
    rows.append(r)

# paired incremental value: combined vs clinical, and gene vs clinical
inc = []
for held in GRADE_COHORTS:
    rC, tt, ee = risks[("clinical_common4_with_grade", held)]
    rG, _, _ = risks[("gene", held)]
    rB, _, _ = risks[("comb", held)]
    rng = np.random.default_rng(23)
    dBC, dGC = [], []
    for _ in range(NBOOT):
        i = rng.integers(0, len(tt), len(tt))
        if ee[i].sum() < 2:
            continue
        cC = cindex(rC[i], tt[i], ee[i])
        dBC.append(cindex(rB[i], tt[i], ee[i]) - cC)
        dGC.append(cindex(rG[i], tt[i], ee[i]) - cC)
    dBC, dGC = np.array(dBC), np.array(dGC)
    inc.append(dict(held_out_cohort=held,
                    c_clinical=cindex(rC, tt, ee), c_gene=cindex(rG, tt, ee),
                    c_combined=cindex(rB, tt, ee),
                    delta_combined_minus_clinical=float(cindex(rB, tt, ee) - cindex(rC, tt, ee)),
                    d_cc_lo=float(np.percentile(dBC, 2.5)), d_cc_hi=float(np.percentile(dBC, 97.5)),
                    p_combined_vs_clinical=float(min(1.0, 2 * min((dBC <= 0).mean(), (dBC >= 0).mean()))),
                    delta_gene_minus_clinical=float(cindex(rG, tt, ee) - cindex(rC, tt, ee)),
                    d_gc_lo=float(np.percentile(dGC, 2.5)), d_gc_hi=float(np.percentile(dGC, 97.5)),
                    p_gene_vs_clinical=float(min(1.0, 2 * min((dGC <= 0).mean(), (dGC >= 0).mean()))),
                    events_test=int(ee.sum())))

df = pd.DataFrame(rows)
df.to_csv("clinical_arm_supplement.csv", index=False)
pd.DataFrame(inc).to_csv("clinical_arm_incremental.csv", index=False)
log("wrote clinical_arm_supplement.csv and clinical_arm_incremental.csv")

summ = {
    "grade_available_cohorts_OS": GRADE_COHORTS,
    "grade_unavailable_cohorts_OS": NOGRADE_COHORTS,
    "age_in_years_cohorts_OS": AGE_COHORTS,
    "common_covars_over_grade_cohorts": COMMON4_GRADE,
    "common_covars_over_rich3": COMMON_RICH,
    "mean_cindex_by_arm": {k: round(float(v), 4)
                           for k, v in df.groupby("arm").cindex.mean().items()},
    "incremental": inc,
}
json.dump(summ, open("clinical_arm_supplement_summary.json", "w"), indent=1, default=float)
print(df[["arm", "held_out_cohort", "n_features", "cindex"]].round(4).to_string(index=False), flush=True)
print(pd.DataFrame(inc).round(4).to_string(index=False), flush=True)
