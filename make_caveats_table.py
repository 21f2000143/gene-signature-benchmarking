"""
make_caveats_table.py -- REVIEW ITEM 7(b): explicit limitation table.

This is a FACTUAL table about method form, not new computation.  It states, for
each comparator, the form in which the signature was published, what this
benchmark actually implemented, and therefore what the comparison does and does
not license.

The distinction the reviewer is pressing on: every clinical-assay comparator was
published as a FIXED algorithm -- fixed genes, fixed coefficients or centroids,
a specified normalisation, and in several cases a specified assay platform and a
fixed risk-group cut-point.  This benchmark re-fits each comparator's GENE LIST
with a single common learner (ridge Cox, alpha=100) on within-cohort z-scored
expression.  That is a test of the gene lists as feature sets under a common
learner.  It is NOT a test of the published assays, and no statement about the
clinical validity, regulatory standing or published performance of PAM50,
Oncotype DX, MammaPrint or the Genomic Grade Index follows from it.

The per-cohort gene availability that qualifies each row is computed in
run_comparator_penalty.py and reported in comparator_coverage_penalty.csv;
the counts quoted in the n_genes_available_range column are read from that file
when it is present, so this table never states a coverage number by hand.

Writes comparator_implementation_caveats.csv with the columns required by the
review response: signature, n_genes_published, published_form,
our_implementation, comparison_licenses, comparison_does_not_license
(plus supporting columns).
"""
import os, json
import pandas as pd

GENE_SETS = json.load(open("results/gene_sets.json"))

COMMON_IMPL = ("Gene list only. Every gene z-scored within each cohort; features "
               "row-stacked across the pooled training cohorts; single "
               "pre-specified learner (ridge-penalised Cox, Breslow ties, "
               "alpha=100); leave-one-cohort-out, evaluated once on the held-out "
               "cohort by Harrell's c-index. No published coefficients, "
               "centroids, reference-gene normalisation, platform-specific "
               "preprocessing or risk-group cut-point was used.")

ROWS = [
 dict(signature="PAM50",
      reference="Parker et al., J Clin Oncol 2009",
      published_form=(
        "Subtype centroid classifier. Fifty genes; each sample is assigned to "
        "Luminal A / Luminal B / HER2-enriched / Basal-like / Normal-like by "
        "nearest-centroid (Spearman) correlation to five published centroids, "
        "after median-centring against a defined ER-balanced reference set. The "
        "published prognostic output is the subtype call plus a risk-of-relapse "
        "score (ROR-S / ROR-P) that combines centroid correlations with "
        "proliferation and, optionally, tumour size. It is a classifier, not a "
        "fitted Cox linear predictor."),
      comparison_licenses=(
        "That the 50 PAM50 genes, as a feature set under our common learner and "
        "our harmonisation, discriminate to the reported c-index in these "
        "cohorts, and that they can be ranked against other gene lists under "
        "identical conditions."),
      comparison_does_not_license=(
        "Any claim about PAM50 as an assay, about intrinsic-subtype calls, about "
        "ROR-S/ROR-P, or about the Prosigna product. Our procedure discards the "
        "centroids and the reference-set median-centring that make PAM50 what it "
        "is; a poor result here is evidence about the gene list under our "
        "learner, not about the published classifier."),
      key_element_discarded="five subtype centroids; ER-balanced reference-set median centring; ROR score formula"),
 dict(signature="OncotypeDX21",
      reference="Paik et al., N Engl J Med 2004",
      published_form=(
        "Fixed linear risk-index formula on 21 genes: 16 cancer-related genes "
        "normalised against 5 reference genes (ACTB, GAPDH, GUS, RPLPO, TFRC) "
        "measured by RT-PCR, combined by published fixed coefficients into a "
        "grouped Recurrence Score on 0-100 with fixed low/intermediate/high "
        "cut-points. Validated for ER-positive, node-negative, tamoxifen-treated "
        "disease."),
      comparison_licenses=(
        "That the 21 genes as a feature set under our learner discriminate to "
        "the reported c-index in unselected cohorts."),
      comparison_does_not_license=(
        "Any claim about the Recurrence Score. We do not apply the published "
        "coefficients, do not perform reference-gene normalisation (the five "
        "reference genes enter as ordinary features), do not restrict to the "
        "ER-positive node-negative tamoxifen-treated population in which the "
        "score was validated, and do not use the RT-PCR platform. The score's "
        "published purpose is chemotherapy-benefit prediction, which we do not "
        "evaluate at all."),
      key_element_discarded="published coefficients; 5-reference-gene normalisation; population restriction; RS cut-points"),
 dict(signature="GGI",
      reference="Sotiriou et al., J Natl Cancer Inst 2006",
      published_form=(
        "Fixed linear score (a scaled difference between the mean expression of "
        "genes up-regulated in grade 3 and those up-regulated in grade 1), "
        "derived on Affymetrix U133A and applied with a fixed cut-point to "
        "reclassify histological grade 2 tumours. Proliferation-dominated by "
        "construction."),
      comparison_licenses=(
        "That the GGI gene list under our learner discriminates to the reported "
        "c-index, and comparison against other lists on equal terms."),
      comparison_does_not_license=(
        "Any claim about the published GGI score or about its intended use, "
        "which is grade reclassification rather than direct survival "
        "prediction. Our version re-fits free coefficients per fold instead of "
        "the published up-minus-down contrast, and we evaluate on RNA-seq as "
        "well as array cohorts."),
      key_element_discarded="fixed up-minus-down contrast; U133A provenance; grade-2 reclassification cut-point"),
 dict(signature="MammaPrint70",
      reference="van 't Veer et al., Nature 2002; van de Vijver et al., NEJM 2002",
      published_form=(
        "Correlation-to-template classifier. Seventy genes; a sample's Pearson "
        "correlation to the published good-prognosis template is thresholded at "
        "a fixed cut-point into a binary good/poor prognosis call. Developed on "
        "Agilent two-colour arrays against a pooled reference, in patients under "
        "55 with node-negative disease; the commercial assay (MammaPrint) is "
        "run on a specified platform with a specified normalisation."),
      comparison_licenses=(
        "That the 70-gene list as a feature set under our learner discriminates "
        "to the reported c-index."),
      comparison_does_not_license=(
        "Any claim about MammaPrint, its binary risk call, or the MINDACT "
        "evidence. We discard the template correlation and the cut-point, use "
        "single-channel data rather than two-colour ratios against a pooled "
        "reference, and do not restrict to the age/node population of "
        "derivation. This signature also suffers the largest symbol-mapping "
        "loss of any comparator (legacy clone-based identifiers), so its "
        "harmonised feature set is the furthest from the published list."),
      key_element_discarded="good-prognosis template correlation; binary cut-point; two-colour ratio normalisation; population restriction"),
 dict(signature="BuffaHypoxia",
      reference="Buffa et al., Br J Cancer 2010",
      published_form=(
        "Metagene score: the signature is summarised as the number of member "
        "genes expressed above their median (or the median-centred mean), i.e. "
        "an unweighted rank/threshold summary, not a fitted survival model. "
        "Intended as a hypoxia-activity readout."),
      comparison_licenses=(
        "That the hypoxia metagene's genes under our learner discriminate to the "
        "reported c-index."),
      comparison_does_not_license=(
        "Any claim about the published hypoxia metagene score as constructed by "
        "its authors, which is unweighted and threshold-based rather than "
        "Cox-fitted. Note this signature shares P4HA2 with the novel panel, so "
        "it is not statistically independent of it."),
      key_element_discarded="unweighted above-median metagene summarisation"),
 dict(signature="CNetCox6",
      reference="network-regularised Cox risk score (primary published comparator)",
      published_form=(
        "Fixed linear prognostic risk score: published gene set with published "
        "Cox coefficients estimated under a network-regularised penalty on the "
        "authors' training data."),
      comparison_licenses=(
        "That the six genes as a feature set, refitted by our learner, "
        "discriminate to the reported c-index in these cohorts."),
      comparison_does_not_license=(
        "Any claim about the published risk score with its own coefficients and "
        "network penalty, which we replace with a plain ridge penalty refitted "
        "per fold."),
      key_element_discarded="published Cox coefficients; network regularisation structure"),
 dict(signature="Novel5",
      reference="this study",
      published_form=(
        "Not a published fixed algorithm. A five-gene list discovered in this "
        "work by leave-one-cohort-out forward selection anchored on the "
        "EEF1A2-IQGAP1-IQGAP2-FRG1 complex."),
      comparison_licenses=(
        "Comparison of this gene list against other gene lists under one common "
        "learner, which is the only comparison this benchmark supports."),
      comparison_does_not_license=(
        "A claim of superiority over the published PAM50, Oncotype DX, "
        "MammaPrint or GGI assays. IMPORTANT ASYMMETRY: the novel panel is "
        "evaluated in exactly the form in which it was created -- a bare gene "
        "list fed to ridge Cox -- whereas every comparator is evaluated in a "
        "form its authors did not publish. The benchmark is therefore "
        "structurally favourable to the novel panel, and this must be stated in "
        "the manuscript's limitations."),
      key_element_discarded="none (this is the panel's native form)"),
 dict(signature="Anchor4",
      reference="this study (negative control)",
      published_form=(
        "Not a signature. The four-gene mechanistic scaffold used during "
        "discovery, evaluated alone as a near-uninformative floor."),
      comparison_licenses="Use as a lower reference point for c-index scale.",
      comparison_does_not_license="Any prognostic claim.",
      key_element_discarded="not applicable"),
]

df = pd.DataFrame(ROWS)
df["n_genes_published"] = [
    len(GENE_SETS[s]["genes"]) if s in GENE_SETS else
    (len(GENE_SETS["Novel5_plus_Anchor4"]["genes"]) if s == "Novel9" else None)
    for s in df.signature]
df["our_implementation"] = COMMON_IMPL
df["comparison_type"] = "gene list re-fitted with a common learner"

# attach the measured coverage, read from the penalty table (never hand-typed)
if os.path.exists("comparator_coverage_penalty.csv"):
    pen = pd.read_csv("comparator_coverage_penalty.csv")
    h = pen[pen.feature_definition == "harmonised"].groupby("gene_set").agg(
        n_genes_harmonised=("n_genes_used", "max")).reset_index()
    m = pen[pen.feature_definition == "maximal_feasible"].groupby("gene_set").agg(
        n_genes_max_lo=("n_genes_used", "min"),
        n_genes_max_hi=("n_genes_used", "max")).reset_index()
    cov = h.merge(m, on="gene_set", how="outer").rename(columns={"gene_set": "signature"})
    df = df.merge(cov, on="signature", how="left")
    df["n_genes_available_range"] = [
        ("%d harmonised; %d-%d per-cohort maximal" % (a, b, c))
        if pd.notna(a) else "not evaluated"
        for a, b, c in zip(df.get("n_genes_harmonised"), df.get("n_genes_max_lo"),
                           df.get("n_genes_max_hi"))]
else:
    df["n_genes_available_range"] = "see comparator_coverage_penalty.csv"

cols = ["signature", "n_genes_published", "published_form", "our_implementation",
        "comparison_licenses", "comparison_does_not_license", "reference",
        "key_element_discarded", "n_genes_available_range", "comparison_type"]
cols += [c for c in df.columns if c not in cols]
df[cols].to_csv("comparator_implementation_caveats.csv", index=False)
print("WROTE comparator_implementation_caveats.csv rows=%d" % len(df))
print(df[["signature", "n_genes_published", "n_genes_available_range"]].to_string(index=False))
