"""
verify_manuscript_numbers.py
============================

Checks the numbers asserted in the manuscript prose against the result files
they are supposed to come from.

Motivation: the clinicopathology mean appeared in the prose as 0.707 while the
tables printed a mean of 0.695, because two scripts had computed the clinical
arm under different covariate rules and the prose quoted the wrong one. Reading
prose against CSVs by eye does not scale over 40 pages, so the checks that
matter are asserted here instead.

Each check names the claim, the source of truth, and the tolerance. A failure
prints the claim, the prose value and the computed value.

Run from the repo root (expects results/ and paper/).
"""
import os
import re
import sys
import json
import numpy as np
import pandas as pd

RES = "results"
PAPER = "paper"
ORDER = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203",
         "GSE20711", "GSE58812"]

FAIL, OK = [], []


def rd(n):
    return pd.read_csv(os.path.join(RES, n))


def js(n):
    return json.load(open(os.path.join(RES, n)))


def prose():
    """All hand-written LaTeX: the section files plus the main file, which holds
    the figure captions. Captions quote numbers too and drifted once already."""
    txt = ""
    for f in sorted(os.listdir(PAPER)):
        if re.match(r"0\d_.*\.tex$", f) or f == "cmpb_article.tex":
            txt += open(os.path.join(PAPER, f)).read()
    return txt


def check(claim, prose_val, truth, tol=6e-4):
    """prose_val is what the manuscript says; truth is recomputed."""
    good = abs(float(prose_val) - float(truth)) <= tol
    (OK if good else FAIL).append(
        "%-58s prose=%.4f source=%.4f" % (claim, float(prose_val), float(truth)))
    return good


def present(claim, needle, txt):
    good = needle in txt
    (OK if good else FAIL).append("%-58s %s %r" % (
        claim, "found" if good else "MISSING", needle))
    return good


T = prose()

# ---- clinical arm: the defect that motivated this script -------------------
rec = rd("clinical_arm_reconciled.csv").set_index("held_out_cohort")
cm = rec.clinical_harrell.reindex(ORDER)
nm = rec.novel5_harrell.reindex(ORDER)
ev = rec.events_test.reindex(ORDER)

check("clinical mean concordance", 0.695, cm.mean())
check("clinical event-weighted mean", 0.686, np.average(cm, weights=ev))
check("novel5 mean concordance", 0.661, nm.mean())
check("clinical uno mean", 0.683, rec.clinical_uno.reindex(ORDER).mean())
check("novel5 uno mean", 0.622, rec.novel5_uno.reindex(ORDER).mean())
check("smallest clinical margin (TCGA)", 0.002, -rec.loc["TCGA", "delta_harrell"])
check("largest clinical margin (GSE202203)", 0.086,
      -rec.loc["SCANB_GSE202203", "delta_harrell"])
check("GSE20711 audited clinical c", 0.653, rec.loc["GSE20711", "clinical_harrell"])
check("GSE20711 uno reversal margin", 0.022, rec.loc["GSE20711", "delta_uno"])

# the unaudited value must NOT appear as a clinical mean anywhere in the prose
for bad in ["0.707", "0.697"]:
    if re.search(r"concordance of %s|mean concordance %s|\(%s versus" % (bad, bad, bad), T):
        FAIL.append("%-58s stale unaudited clinical mean %s still in prose"
                    % ("no stale clinical mean", bad))
    else:
        OK.append("%-58s absent from prose" % ("stale clinical mean %s" % bad))

# ---- gene-set means -------------------------------------------------------
ms = rd("metrics_summary.csv").set_index("gene_set")
for gs, val in [("Novel5", 0.661), ("BuffaHypoxia", 0.646), ("MammaPrint70", 0.631),
                ("PAM50", 0.626), ("OncotypeDX21", 0.602), ("GGI", 0.591)]:
    if gs in ms.index:
        check("%s mean Harrell" % gs, val, ms.loc[gs, "harrell_mean"], tol=1e-3)

# ---- tables and prose must agree on every clinical cell -------------------
t2 = open(os.path.join(PAPER, "tab2_loco.tex")).read()
t4 = open(os.path.join(PAPER, "tab4_clinical.tex")).read()
for co in ORDER:
    v = "%.3f" % rec.loc[co, "clinical_harrell"]
    if v not in t2 or v not in t4:
        FAIL.append("%-58s %s not in both tables" % ("clinical cell %s" % co, v))
    else:
        OK.append("%-58s %s in tables 2 and 4" % ("clinical cell %s" % co, v))

# ---- cohort totals -------------------------------------------------------
# Two totals are legitimately in play and must not be conflated: all nine
# harmonised cohorts, and the six carrying overall survival that the benchmark
# actually evaluates. Both must appear, so a future edit cannot silently swap
# one for the other.
t1 = rd("table1_cohorts.csv")
present("nine-cohort patient total", format(int(t1.n.sum()), ","), T)
present("six-cohort (OS) patient total",
        format(int(t1[t1.cohort.isin(ORDER)].n.sum()), ","), T)
if "events" in t1:
    present("nine-cohort event total", format(int(t1.events.sum()), ","), T)
    present("six-cohort event total",
            format(int(t1[t1.cohort.isin(ORDER)].events.sum()), ","), T)

# ---- nested selection / permutation ------------------------------------
if os.path.exists(os.path.join(RES, "nested_selection_summary.json")):
    ns = js("nested_selection_summary.json")
    for k, claim in [("mean_overlap_with_published", "mean fold-panel overlap"),
                     ("unbiased_mean_cindex", "unbiased nested mean c-index")]:
        if k in ns:
            m = re.search(r"%.2f" % ns[k], T) or re.search(r"%.3f" % ns[k], T)
            OK.append("%-58s %.4f %s" % (claim, ns[k],
                                         "quoted" if m else "(not quoted verbatim)"))

# ---- learner grid ---------------------------------------------------------
# The manuscript once claimed the gene-set ranking was learner-invariant and
# that the ridge ordering equalled the best-of-four ordering. Both are false on
# the complete grid. These checks make the corrected claims falsifiable and
# forbid the old ones from returning.
if os.path.exists(os.path.join(RES, "learner_grid_full.csv")):
    g = rd("learner_grid_full.csv")
    check("learner grid cell count", len(g), 240, 0)
    B = g.pivot_table(index="gene_set", columns="learner", values="cindex",
                      aggfunc="mean")
    bo4 = (g.groupby(["gene_set", "held_out_cohort"]).cindex.max()
            .groupby("gene_set").mean())
    arms = [x for x in B.index if x != "Clinical"]
    opt = float(np.mean([bo4[x] - B.loc[x, "CoxPH_ridge"] for x in arms]))
    present("best-of-four optimism in prose", "0.031", T)
    check("best-of-four optimism recomputed", 0.031, opt, 5e-4)
    # Rank correlations must be recomputed over the NINE gene sets the table
    # displays. learner_grid_ranking.json computed them over all ten arms
    # including Clinical, which the table drops; Clinical is first under every
    # learner, so its inclusion inflates apparent stability (RSF 0.806 vs
    # 0.733). The prose says "the nine gene sets", so nine is the right base.
    from scipy.stats import spearmanr
    _ref = B.loc[arms, "CoxPH_ridge"].rank(ascending=False)
    _rho = {m: float(spearmanr(_ref, B.loc[arms, m].rank(ascending=False))[0])
            for m in B.columns if m != "CoxPH_ridge"}
    # bo4 is the per-CELL maximum averaged over cohorts, which is the quantity
    # the table prints; taking the max of learner MEANS instead gives 0.950 and
    # understates the reordering.
    _rho["BestOfFour"] = float(spearmanr(_ref, bo4[arms].rank(ascending=False))[0])
    check("RSF Spearman rho vs ridge (9 arms)", 0.733, _rho["RSF"], 5e-4)
    check("ridge vs best-of-four rho (9 arms)", 0.933, _rho["BestOfFour"], 5e-4)
    # Ranks are stated over the nine gene sets, not the ten-arm grid.
    _rr = B.loc[arms, "CoxPH_ridge"].rank(ascending=False)
    check("panel rank under ridge (9 gene sets)", 1, _rr["Novel5"], 0)
    check("panel rank under RSF (9 gene sets)", 6,
          B.loc[arms, "RSF"].rank(ascending=False)["Novel5"], 0)
    check("panel rank under best-of-four (9 gene sets)", 3,
          bo4[arms].rank(ascending=False)["Novel5"], 0)
    for m_ in ("Coxnet", "GBSA"):
        check("%s rho vs ridge (9 arms)" % m_, 1.000, _rho[m_], 5e-4)
    if re.search(r"\\rho = 0\.806|\\rho=0\.806", T):
        FAIL.append("%-58s ten-arm rho present in prose" % "stale RSF rho 0.806")
    else:
        OK.append("%-58s absent from prose" % "stale ten-arm RSF rho (0.806)")
    for nm_, val in [("BuffaHypoxia", 0.663), ("MammaPrint70", 0.653),
                     ("PAM50", 0.650), ("OncotypeDX21", 0.647),
                     ("Novel5", 0.643)]:
        check("RSF mean c, %s" % nm_, val, B.loc[nm_, "RSF"], 5e-4)
    # The four signatures the forest puts above the panel must be exactly the
    # ones the prose names, and the panel must really be below them.
    rsf = B["RSF"].drop("Clinical").sort_values(ascending=False)
    above = [x for x in rsf.index if rsf[x] > rsf["Novel5"]]
    expect = {"BuffaHypoxia", "MammaPrint70", "PAM50", "OncotypeDX21",
              "Novel5_plus_Anchor4"}
    if set(above) != expect:
        FAIL.append("%-58s RSF arms above Novel5 = %s, prose names %s"
                    % ("RSF reordering", sorted(above), sorted(expect)))
    else:
        OK.append("%-58s %s" % ("RSF places these above the panel",
                                ", ".join(sorted(above))))
    # Scoped to the panel: "top-ranked under each of the four learners" is a
    # true statement about the CLINICAL arm and must stay allowed.
    for bad, why in [(r"of any gene set tested, under all four learners",
                      "claims panel best under all learners"),
                     (r"ordering of gene sets is essentially preserved",
                      "claims ordering preserved under RSF")]:
        if re.search(bad, T):
            FAIL.append("%-58s superseded claim present: %s" % (why, bad))
        else:
            OK.append("%-58s absent from prose" % why)

# ---- null table cells and the 20k-draw caption statistics ------------------
# The nulls table carries two distinct designs whose numbers are easy to cross:
# the own-size size-matched nulls (500 draws each) in the body, and a separate
# 20,000-draw five-gene null quoted in the caption. Pin both, and pin the
# expressed-universe mean, which was once transcribed as 0.5555 (true 0.5552).
if os.path.exists(os.path.join(RES, "null_summary.csv")):
    NS = rd("null_summary.csv")
    # Distinct from the size-matched k=5 null (0.5519): this row samples the
    # expressed-gene universe over 10,000 draws and means 0.5552. The two were
    # once conflated, which is how 0.5555 got written.
    ex = NS[NS.null_set.eq("random5_expressed")].iloc[0]
    check("expressed-universe null mean", 0.5552, float(ex.null_mean), 5e-4)
    if "0.5555" in T:
        FAIL.append("%-58s superseded transcription 0.5555 is back"
                    % "expressed-universe null mean")
    else:
        OK.append("%-58s absent from prose" % "stale null mean 0.5555")
    # Own-size null means and percentiles as printed in the table body.
    for lbl, k, nmean, pct in [("Novel5", 5, 0.552, 100.0),
                               ("Anchor4", 4, 0.548, 42.8),
                               ("GGI", 58, 0.618, 29.2),
                               ("PAM50", 46, 0.610, 99.2)]:
        row = NS[NS.null_set.eq("sizematched_all_k%d" % k)
                 & NS.observed_panel.eq(lbl)]
        if row.empty:
            FAIL.append("%-58s no size-matched row" % ("null: %s" % lbl))
            continue
        check("null mean, %s (k=%d)" % (lbl, k), nmean,
              float(row.null_mean.iloc[0]), 5e-4)
        check("null pctile, %s (k=%d)" % (lbl, k), pct,
              float(row.observed_percentile.iloc[0]), 0.05)

if os.path.exists(os.path.join(RES, "null_20k_summary.json")):
    J = json.load(open(os.path.join(RES, "null_20k_summary.json")))
    _o, _m, _s = (J["observed_novel5_mean_loco_c"], J["null_mean"], J["null_sd"])
    check("20k null: observed panel c", 0.6612, _o, 5e-5)
    check("20k null: null mean", 0.5430, _m, 5e-5)
    check("20k null: z statistic", 4.97, (_o - _m) / _s, 5e-3)
    check("20k null: draws at or above observed", 0,
          J["n_draws_ge_observed"], 0)
    check("20k null: draw count", 20000, J["n_draws"], 0)
    # This phrase lives in a GENERATED table caption, which prose() excludes by
    # design (it only reads hand-written files). Read tab5 directly.
    _t5 = os.path.join(PAPER, "tab5_nulls.tex")
    if os.path.exists(_t5):
        present("20k null quoted in nulls caption", "null mean of 0.5430",
                open(_t5).read())

# ---- participant flow arithmetic must close ---------------------------------
if os.path.exists(os.path.join(RES, "participant_flow_upstream.csv")):
    U = rd("participant_flow_upstream.csv")
    src = int(U.n_source_records_expr_and_clinical.sum())
    exc = int(U.n_excluded_upstream_harmonisation.sum())
    check("flow: source records", 10139, src, 0)
    check("flow: upstream exclusions", 70, exc, 0)
    check("flow: harmonised remainder", 10069, src - exc, 0)
    check("flow: stored equals remainder", src - exc,
          int(U.n_stored_harmonised_matrix.sum()), 0)
    present("flow: source total in prose", "10,139 source records", T)
    present("flow: exclusions in prose", "70 were excluded", T)

# ---- KM stratification: the quoted HR range and its exception --------------
# The prose quotes 1.75-4.78 for the held-out cohorts. That range is over the
# cohorts whose stratification is SIGNIFICANT: GSE20711 has a lower HR (1.615)
# but p = 0.47 on 25 events, and the prose names it as the exception. Verify
# both the range and that the exception is stated, so the range can never be
# read as covering all six.
if os.path.exists(os.path.join(RES, "km_stratification.csv")):
    KM = rd("km_stratification.csv")
    hh = KM[~KM.cohort.eq("POOLED")]
    sig = hh[hh.logrank_p < 0.05]
    check("KM HR range low (significant cohorts)", 1.75,
          sig.hr_high_vs_low.min(), 5e-3)
    check("KM HR range high (significant cohorts)", 4.78,
          sig.hr_high_vs_low.max(), 5e-3)
    ns = hh[hh.logrank_p >= 0.05].cohort.tolist()
    if ns != ["GSE20711"]:
        FAIL.append("%-58s expected only GSE20711, got %s"
                    % ("KM non-significant cohorts", ns))
    else:
        OK.append("%-58s GSE20711 only (named as the exception)"
                  % "KM non-significant cohorts")
    present("KM exception named in prose", "only GSE20711", T)
    # GSE20711's HR is below the quoted floor, so the exception must be stated.
    if float(hh[hh.cohort.eq("GSE20711")].hr_high_vs_low.iloc[0]) >= 1.75:
        FAIL.append("%-58s no longer below the quoted floor" % "GSE20711 HR")
    else:
        OK.append("%-58s 1.615, below quoted floor: exception required"
                  % "GSE20711 HR")

# ---- forest figure: gene counts and the size/margin claim ------------------
# Panel c once asserted "the margin is smallest against the largest signatures".
# It is not: over the eight comparators the size/margin association is weak and
# non-significant, and the smallest margin of all is against the 9-gene
# Novel5+anchor superset. The title must carry the statistic, not the assertion.
if os.path.exists(os.path.join(RES, "loco_paired_novel5_vs_comparators.csv")):
    LPv = rd("loco_paired_novel5_vs_comparators.csv")
    # Counts must be constant within a comparator, or a single y-label lies.
    nun = LPv.groupby("comparator").n_genes_comparator.nunique()
    if int(nun.max()) != 1:
        FAIL.append("%-58s varies across cohorts: %s"
                    % ("forest comparator gene counts",
                       nun[nun > 1].index.tolist()))
    else:
        OK.append("%-58s constant across all six cohorts"
                  % "forest comparator gene counts")
    sgv = LPv.groupby("comparator").agg(k=("n_genes_comparator", "first"),
                                        d=("delta_cindex", "mean"))
    from scipy.stats import spearmanr as _sr
    _r, _p = _sr(sgv.k, sgv.d)
    if _p < 0.05:
        FAIL.append("%-58s p=%.3f now significant; retitle panel c"
                    % ("size/margin association", _p))
    else:
        OK.append("%-58s rho=%.2f p=%.2f (weak, as panel c states)"
                  % ("size/margin association", _r, _p))
    # The figure title is rendered from these values; the script comment also
    # quotes the four-cohort bootstrap figure for contrast. Pin both so neither
    # the rendered title nor the comment can drift from its source.
    check("size/margin rho (six-cohort, plotted)", -0.40, _r, 5e-3)
    if os.path.exists(os.path.join(RES, "paired_bootstrap.csv")):
        PBv = rd("paired_bootstrap.csv")
        g4v = PBv.groupby("comparator").agg(k=("n_genes_comparator", "first"),
                                            d=("mean_diff", "mean"))
        _r4, _p4 = _sr(g4v.k, g4v.d)
        check("size/margin rho (four-cohort bootstrap)", -0.33, _r4, 5e-3)
        if sorted(g4v.index) != sorted(sgv.index):
            FAIL.append("%-58s comparator sets differ between the two files"
                        % "size/margin comparator sets")
        else:
            OK.append("%-58s same eight comparators in both files"
                      % "size/margin comparator sets")
    # The smallest margin must be against the superset, not a large signature.
    check("smallest-margin comparator is the 9-gene superset", 9,
          sgv.k[sgv.d.idxmin()], 0)

# ---- anchor control honesty ----------------------------------------------
# Anchor4 was described as a "plausible but uninformative" negative control.
# It is neither uninformative (significant alone in all four large cohorts) nor
# independent (it is the scaffold the original search forced in). Both facts
# must be disclosed and the old wording must not return.
if os.path.exists(os.path.join(RES, "likelihood_ratio_tests.csv")):
    L = rd("likelihood_ratio_tests.csv")
    alone = L[L.model_compared.eq("null vs Anchor4 alone")]
    nsig = int((alone.p < 0.05).sum())
    if nsig != len(alone):
        FAIL.append("%-58s expected all %d significant, got %d"
                    % ("Anchor4 significant alone", len(alone), nsig))
    else:
        OK.append("%-58s %d/%d cohorts, p max %.4f"
                  % ("Anchor4 significant alone (must be disclosed)",
                     nsig, len(alone), alone.p.max()))
    present("Anchor4 scaffold disclosure", "scaffold that the original", T)
    present("Anchor4 not-uninformative disclosure", "It is not\nuninformative", T)
    if re.search(r"plausible but uninformative", T):
        FAIL.append("%-58s superseded wording present" % "Anchor4 'uninformative'")
    else:
        OK.append("%-58s absent from prose" % "Anchor4 'uninformative' claim")
    present("Anchor4 alone p-range", "$p$ from %.3f to %.4f"
            % (alone.p.max(), alone.p.min()), T)
    # The adjusted result: non-significant in 3 of 4, TCGA the exception.
    adj = L[L.model_compared.str.endswith("vs clinical+Anchor4")]
    nns = int((adj.p >= 0.05).sum())
    check("Anchor4 adjusted non-significant cohorts", 3, nns, 0)

print("=" * 78)
print("PASS (%d)" % len(OK))
for l in OK:
    print("  " + l)
print()
if FAIL:
    print("FAIL (%d)" % len(FAIL))
    for l in FAIL:
        print("  " + l)
    sys.exit(1)
print("all manuscript numbers agree with their source files")
