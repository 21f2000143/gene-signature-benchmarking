"""
participant_flow.py -- TRIPOD participant flow accounting (review items 10 and 14).

REVIEW ITEM ADDRESSED
  Items 10/14: TRIPOD (Transparent Reporting of a multivariable prediction model for
  Individual Prognosis Or Diagnosis) requires an explicit account of participant flow:
  how many samples each source cohort contained, how many were excluded and for what
  reason, and how many entered the analysis. The submitted manuscript reports only the
  final analysed n per cohort, with no exclusion accounting.

EXACT DEFINITIONS USED
  The exclusion rule is not invented here: it is the rule actually applied by
  nested_core.load_cohort(), replicated line-for-line. In that function:

      e = read_parquet(<COHORT>_expr.parquet)          # samples x genes
      s = read_parquet(<COHORT>_surv.parquet)
      if "sample" in s.columns:
          s = s.set_index("sample")
          common = [i for i in e.index if i in s.index]   # expr order preserved
          e, s = e.loc[common], s.loc[common]
      keep = isfinite(s.time_months) & isfinite(s.event) & (s.time_months > 0)
      e, s = e.loc[keep], s.loc[keep]

  Hence the columns of participant_flow.csv are defined as:

  n_source_matrix
      = e.shape[0] before any filtering: the number of sample rows in the source
        harmonised expression matrix for that cohort.
  n_with_survival
      = len(common): sample rows of the expression matrix for which a survival record
        exists in the survival table (the linkage step). Samples in the survival table
        with no expression row are NOT counted here; they are reported separately in
        participant_flow_summary.json as n_surv_records_not_in_expr.
  n_excluded_nonpositive_or_missing_time
      = n_with_survival - keep.sum(): linked samples failing
        isfinite(time_months) & isfinite(event) & (time_months > 0).
        The sub-reasons (missing time, missing event, time <= 0) are reported in the
        JSON; they are not mutually exclusive, so they need not sum to this column.
  n_analysed
      = keep.sum(): samples entering every analysis in this paper.
        By construction n_analysed = n_with_survival - n_excluded_nonpositive_or_missing_time.
  events
      = sum(event == 1) among the n_analysed samples.
  median_followup_months
      = median of the OBSERVED time_months over the n_analysed samples (a plain
        descriptive median of recorded follow-up, not a reverse Kaplan-Meier estimate).
        The reverse-KM median potential follow-up (Schemper-Smith: Kaplan-Meier on the
        censoring indicator 1-event) is additionally reported in the JSON as
        median_followup_reverse_km_months, because the two differ materially in
        cohorts with high event rates and reviewers may expect the latter.
  platform_gene_count
      = e.shape[1]: number of gene columns in the source harmonised matrix for that
        cohort (the effective measured feature space of that platform after
        harmonisation). The platform label from the survival table is in the JSON.
  endpoint
      = the unique value(s) of the 'endpoint' column of the survival table.

DIAGNOSTIC
  The task brief states the SCANB_GSE96058 copy on this host is not z-scored. Every
  analysis in this track therefore z-scores each gene within each cohort via the
  nested_core rule. To document the actual state of each stored matrix, this script
  also records, over a random sample of 500 gene columns per cohort, the median column
  mean and median column SD of the STORED (unstandardised) values. z-scoring is
  idempotent, so applying it to an already-standardised matrix is harmless; this
  diagnostic simply records which matrices needed it.

IMPORTANT FINDING -- THE FLOW HAS TWO STAGES, NOT ONE
  Running the accounting above returns ZERO exclusions for all nine cohorts: the stored
  harmonised matrices already contain only samples that satisfy the rule. The exclusions
  therefore happened UPSTREAM, when the matrices were built, and are invisible to
  load_cohort. Reporting only the load_cohort stage would tell a reviewer that nothing
  was ever excluded, which is false.

  The upstream stage is recoverable from qc_summary.csv, which the harmonisation step
  wrote alongside the matrices. Its 'overlap' column records the number of samples with
  BOTH an expression profile and a clinical record before the survival-validity filter,
  and its 'n' column the number retained. The difference is the upstream exclusion:

      TCGA 1100->1086, METABRIC 1980->1979, SCANB_GSE202203 2913->2912,
      GSE20711 90->88, GSE6532 414->380, GSE21653 266->248,
      SCANB_GSE96058 / GSE58812 / GSE11121 unchanged.

  This script therefore reports BOTH stages. participant_flow.csv keeps exactly the nine
  requested columns and describes the ANALYSIS-STAGE account, in which n_source_matrix is
  the stored harmonised matrix. participant_flow_upstream.csv and the JSON carry the
  harmonisation stage so the flow diagram can show the true funnel:

      source records with expression + clinical  (qc 'overlap')
        -> minus upstream survival-validity exclusions
        -> stored harmonised matrix  ( = n_source_matrix here)
        -> minus analysis-stage exclusions (zero, verified)
        -> n_analysed

  The upstream exclusion reason is not itemised in qc_summary.csv. It is reported as
  'not itemised upstream; consistent with the same finite-and-positive-follow-up rule'
  rather than guessed at.

FOLLOW-UP DEFINITION MISMATCH WITH THE EXISTING MANUSCRIPT TABLE
  qc_summary.csv also carries a median_fu_months column whose values differ from the
  median observed time computed here (e.g. METABRIC 157.9 vs 116.5; TCGA 25.2 vs 28.2).
  Three different quantities are in play, so all three are reported in the JSON:
  median observed time over all analysed samples (the CSV column), median observed time
  among CENSORED samples only, and the reverse-KM median potential follow-up. The
  manuscript should state which it uses; they are not interchangeable.

OUTPUTS
  participant_flow.csv           one row per cohort, the nine requested columns
  participant_flow_upstream.csv  harmonisation-stage funnel per cohort
  participant_flow_summary.json  totals overall and per endpoint, sub-reason counts,
                                 platform labels, scale diagnostics, follow-up variants,
                                 consistency checks
"""
import os
import json
import numpy as np
import pandas as pd

DATA = os.path.expanduser("harmonised")
OS6 = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203", "GSE20711", "GSE58812"]
SEC3 = ["GSE6532", "GSE11121", "GSE21653"]
ALL9 = OS6 + SEC3
RNG = np.random.default_rng(0)


def reverse_km_median(time, event):
    """Median potential follow-up: Kaplan-Meier on the reversed censoring indicator.

    Treats censoring as the 'event' (delta = 1 - event) and returns the time at which
    the reverse-KM survival curve first drops to or below 0.5. Returns NaN if the
    curve never reaches 0.5.
    """
    t = np.asarray(time, float)
    d = 1 - np.asarray(event, int)          # censoring is the event of interest
    order = np.argsort(t, kind="mergesort")
    t, d = t[order], d[order]
    n = len(t)
    at_risk = n
    surv = 1.0
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        n_ev = int(d[i:j].sum())
        if n_ev > 0 and at_risk > 0:
            surv *= (1.0 - n_ev / at_risk)
            if surv <= 0.5:
                return float(t[i])
        at_risk -= (j - i)
        i = j
    return float("nan")


rows = []
detail = {}

for coh in ALL9:
    ep = os.path.join(DATA, coh + "_expr.parquet")
    sp = os.path.join(DATA, coh + "_surv.parquet")

    # --- source matrix shape, read from the parquet schema where possible ----------
    e = pd.read_parquet(ep)
    s = pd.read_parquet(sp)
    n_source_matrix = int(e.shape[0])
    platform_gene_count = int(e.shape[1])
    n_surv_records = int(s.shape[0])

    # --- linkage, exactly as nested_core.load_cohort ------------------------------
    if "sample" in s.columns:
        s_idx = s.set_index("sample")
        common = [i for i in e.index if i in s_idx.index]
        s_lnk = s_idx.loc[common]
        n_surv_not_in_expr = int(len(set(s_idx.index) - set(e.index)))
    else:
        common = list(e.index.intersection(s.index))
        s_lnk = s.loc[common]
        n_surv_not_in_expr = int(len(set(s.index) - set(e.index)))
    n_with_survival = int(len(common))
    n_expr_no_surv = int(n_source_matrix - n_with_survival)

    tm = pd.to_numeric(s_lnk["time_months"], errors="coerce").values.astype(float)
    evv = pd.to_numeric(s_lnk["event"], errors="coerce").values.astype(float)

    finite_t = np.isfinite(tm)
    finite_e = np.isfinite(evv)
    keep = finite_t & finite_e & (tm > 0)
    n_analysed = int(keep.sum())
    n_excluded = int(n_with_survival - n_analysed)

    # sub-reasons (overlapping, for the JSON only)
    sub = {
        "missing_or_nonfinite_time": int((~finite_t).sum()),
        "missing_or_nonfinite_event": int((~finite_e).sum()),
        "time_le_zero_among_finite_time": int((finite_t & (tm <= 0)).sum()),
        "time_exactly_zero": int((finite_t & (tm == 0)).sum()),
        "time_negative": int((finite_t & (tm < 0)).sum()),
    }

    t_a = tm[keep]
    e_a = evv[keep].astype(int)
    events = int(e_a.sum())
    median_fu = float(np.median(t_a)) if n_analysed else float("nan")
    rkm = reverse_km_median(t_a, e_a) if n_analysed else float("nan")
    cens_t = t_a[e_a == 0]
    median_fu_censored = float(np.median(cens_t)) if len(cens_t) else float("nan")

    endpoints = sorted(set(map(str, s_lnk["endpoint"].dropna().unique()))) \
        if "endpoint" in s_lnk.columns else []
    endpoint = "/".join(endpoints) if endpoints else "UNSPECIFIED"
    platforms = sorted(set(map(str, s_lnk["platform"].dropna().unique()))) \
        if "platform" in s_lnk.columns else []

    # --- stored-scale diagnostic over a random column sample ----------------------
    ncol = e.shape[1]
    pick = RNG.choice(ncol, size=int(min(500, ncol)), replace=False)
    sub_mat = e.iloc[:, np.sort(pick)].to_numpy(dtype=float, copy=False)
    diag = {
        "n_cols_sampled": int(sub_mat.shape[1]),
        "median_col_mean": float(np.nanmedian(np.nanmean(sub_mat, axis=0))),
        "median_col_sd": float(np.nanmedian(np.nanstd(sub_mat, axis=0))),
        "max_abs_col_mean": float(np.nanmax(np.abs(np.nanmean(sub_mat, axis=0)))),
        "nan_fraction": float(np.isnan(sub_mat).mean()),
    }
    # A median-based verdict is not sufficient: a matrix can be standardised in the bulk
    # of its columns yet carry a minority of unstandardised ones. The verdict below is
    # therefore driven by the WORST column, and both statistics are retained.
    col_means = np.nanmean(sub_mat, axis=0)
    col_sds = np.nanstd(sub_mat, axis=0)
    diag["max_abs_col_mean"] = float(np.nanmax(np.abs(col_means)))
    diag["max_col_sd"] = float(np.nanmax(col_sds))
    diag["n_cols_offscale"] = int(np.sum((np.abs(col_means) > 1e-3) |
                                         (np.abs(col_sds - 1.0) > 1e-2)))
    diag["frac_cols_offscale"] = round(float(diag["n_cols_offscale"]
                                             / max(sub_mat.shape[1], 1)), 4)
    diag["bulk_appears_standardised"] = bool(abs(diag["median_col_mean"]) < 1e-6
                                             and abs(diag["median_col_sd"] - 1.0) < 1e-3)
    diag["fully_standardised_all_sampled_cols"] = bool(diag["n_cols_offscale"] == 0)
    diag["requires_zscoring"] = bool(not diag["fully_standardised_all_sampled_cols"])

    rows.append({
        "cohort": coh,
        "endpoint": endpoint,
        "n_source_matrix": n_source_matrix,
        "n_with_survival": n_with_survival,
        "n_excluded_nonpositive_or_missing_time": n_excluded,
        "n_analysed": n_analysed,
        "events": events,
        "median_followup_months": round(median_fu, 3),
        "platform_gene_count": platform_gene_count,
    })
    detail[coh] = {
        "primary_or_secondary": "primary_OS" if coh in OS6 else "secondary_DMFS_DFS",
        "endpoint_values": endpoints,
        "platform_labels": platforms,
        "n_surv_records_total": n_surv_records,
        "n_surv_records_not_in_expr": n_surv_not_in_expr,
        "n_expr_rows_without_survival_record": n_expr_no_surv,
        "exclusion_subreasons_overlapping": sub,
        "events": events,
        "censored": int(n_analysed - events),
        "event_rate": round(events / n_analysed, 4) if n_analysed else None,
        "median_followup_months_observed": round(median_fu, 3),
        "median_followup_months_censored_only": (round(median_fu_censored, 3)
                                                 if np.isfinite(median_fu_censored)
                                                 else None),
        "median_followup_reverse_km_months": (round(rkm, 3) if np.isfinite(rkm) else None),
        "followup_min_months": round(float(t_a.min()), 3) if n_analysed else None,
        "followup_max_months": round(float(t_a.max()), 3) if n_analysed else None,
        "stored_scale_diagnostic": diag,
        "consistency_ok": bool(n_analysed == n_with_survival - n_excluded),
    }
    print("%-18s endpoint=%-6s source=%5d linked=%5d excl=%3d analysed=%5d events=%5d "
          "medFU=%7.1f genes=%6d offscale_cols=%d"
          % (coh, endpoint, n_source_matrix, n_with_survival, n_excluded, n_analysed,
             events, median_fu, platform_gene_count, diag["n_cols_offscale"]),
          flush=True)

flow_lookup = {r_["cohort"]: r_ for r_ in rows}

# ------------------------------------------------------- upstream harmonisation stage
# qc_summary.csv was written by the harmonisation step that BUILT these matrices. Its
# 'overlap' column is the number of source records having both an expression profile and
# a clinical record; its 'n' column is the number retained in the stored matrix. The
# difference is the upstream exclusion, which load_cohort cannot see.
up_rows = []
qc_path = os.path.join(DATA, "qc_summary.csv")
qc_available = os.path.exists(qc_path)
if qc_available:
    qc = pd.read_csv(qc_path).set_index("cohort")
    for coh in ALL9:
        analysed = int(flow_lookup[coh]["n_analysed"])
        stored = int(flow_lookup[coh]["n_source_matrix"])
        if coh in qc.index:
            ov = int(qc.loc[coh, "overlap"])
            qc_n = int(qc.loc[coh, "n"])
            up_excl = int(ov - stored)
            up_rows.append({
                "cohort": coh,
                "endpoint": flow_lookup[coh]["endpoint"],
                "n_source_records_expr_and_clinical": ov,
                "n_excluded_upstream_harmonisation": up_excl,
                "n_stored_harmonised_matrix": stored,
                "n_excluded_analysis_stage":
                    int(flow_lookup[coh]["n_excluded_nonpositive_or_missing_time"]),
                "n_analysed": analysed,
                "qc_summary_n_matches_stored": bool(qc_n == stored),
                "qc_summary_events": int(qc.loc[coh, "events"]),
                "events_recomputed": int(flow_lookup[coh]["events"]),
                "events_match": bool(int(qc.loc[coh, "events"])
                                     == int(flow_lookup[coh]["events"])),
                "qc_summary_median_fu_months": float(qc.loc[coh, "median_fu_months"]),
                "median_fu_months_recomputed": float(
                    flow_lookup[coh]["median_followup_months"]),
                "platform_label": str(qc.loc[coh, "platform"]),
                "source_time_unit": str(qc.loc[coh, "time_unit"]),
                "upstream_exclusion_reason":
                    "not itemised upstream; consistent with the same "
                    "finite-and-positive-follow-up rule",
            })
    up = pd.DataFrame(up_rows)
    up.to_csv("participant_flow_upstream.csv", index=False)
    print("\nupstream harmonisation stage (from qc_summary.csv):", flush=True)
    for r_ in up_rows:
        print("  %-18s overlap=%5d -> stored=%5d (upstream excl %3d) -> analysed=%5d"
              % (r_["cohort"], r_["n_source_records_expr_and_clinical"],
                 r_["n_stored_harmonised_matrix"],
                 r_["n_excluded_upstream_harmonisation"], r_["n_analysed"]), flush=True)
else:
    up = pd.DataFrame()
    print("\nqc_summary.csv not found; upstream stage not reconstructed", flush=True)

flow = pd.DataFrame(rows, columns=["cohort", "endpoint", "n_source_matrix",
                                   "n_with_survival",
                                   "n_excluded_nonpositive_or_missing_time",
                                   "n_analysed", "events", "median_followup_months",
                                   "platform_gene_count"])
flow.to_csv("participant_flow.csv", index=False)

os_mask = flow["cohort"].isin(OS6)
sec_mask = flow["cohort"].isin(SEC3)
by_ep = {}
for ep_name, grp in flow.groupby("endpoint"):
    by_ep[ep_name] = {"cohorts": list(grp["cohort"]),
                      "n_analysed": int(grp["n_analysed"].sum()),
                      "events": int(grp["events"].sum()),
                      "n_source_matrix": int(grp["n_source_matrix"].sum()),
                      "n_excluded": int(grp["n_excluded_nonpositive_or_missing_time"].sum())}

summary = {
    "definitions": {
        "exclusion_rule": "isfinite(time_months) & isfinite(event) & (time_months > 0), "
                          "applied after linking expression rows to survival records; "
                          "replicated from nested_core.load_cohort",
        "n_with_survival": "expression-matrix rows with a matching survival record",
        "n_analysed": "n_with_survival minus exclusions; identity checked per cohort",
        "median_followup_months": "median observed time_months among analysed samples",
        "median_followup_reverse_km_months": "reverse Kaplan-Meier median potential "
                                             "follow-up (censoring as the event)",
        "platform_gene_count": "gene columns in the harmonised source matrix",
    },
    "totals_all_nine_cohorts": {
        "n_cohorts": int(len(flow)),
        "n_source_matrix": int(flow["n_source_matrix"].sum()),
        "n_with_survival": int(flow["n_with_survival"].sum()),
        "n_excluded_nonpositive_or_missing_time":
            int(flow["n_excluded_nonpositive_or_missing_time"].sum()),
        "n_analysed": int(flow["n_analysed"].sum()),
        "events": int(flow["events"].sum()),
    },
    "totals_primary_OS_six_cohorts": {
        "cohorts": OS6,
        "n_source_matrix": int(flow.loc[os_mask, "n_source_matrix"].sum()),
        "n_with_survival": int(flow.loc[os_mask, "n_with_survival"].sum()),
        "n_excluded_nonpositive_or_missing_time":
            int(flow.loc[os_mask, "n_excluded_nonpositive_or_missing_time"].sum()),
        "n_analysed": int(flow.loc[os_mask, "n_analysed"].sum()),
        "events": int(flow.loc[os_mask, "events"].sum()),
    },
    "totals_secondary_three_cohorts": {
        "cohorts": SEC3,
        "n_source_matrix": int(flow.loc[sec_mask, "n_source_matrix"].sum()),
        "n_with_survival": int(flow.loc[sec_mask, "n_with_survival"].sum()),
        "n_excluded_nonpositive_or_missing_time":
            int(flow.loc[sec_mask, "n_excluded_nonpositive_or_missing_time"].sum()),
        "n_analysed": int(flow.loc[sec_mask, "n_analysed"].sum()),
        "events": int(flow.loc[sec_mask, "events"].sum()),
    },
    "totals_by_endpoint_label": by_ep,
    "per_cohort": detail,
    "consistency_all_ok": bool(all(d["consistency_ok"] for d in detail.values())),
}

if len(up):
    summary["upstream_harmonisation_stage"] = {
        "source": "qc_summary.csv written by the harmonisation step",
        "note": "load_cohort applies zero further exclusions; every exclusion in this "
                "study occurred at the harmonisation stage, so a TRIPOD flow diagram "
                "built only from the analysis stage would show no exclusions at all",
        "totals": {
            "n_source_records_expr_and_clinical":
                int(up["n_source_records_expr_and_clinical"].sum()),
            "n_excluded_upstream_harmonisation":
                int(up["n_excluded_upstream_harmonisation"].sum()),
            "n_stored_harmonised_matrix": int(up["n_stored_harmonised_matrix"].sum()),
            "n_excluded_analysis_stage": int(up["n_excluded_analysis_stage"].sum()),
            "n_analysed": int(up["n_analysed"].sum()),
        },
        "totals_primary_OS_six": {
            "n_source_records_expr_and_clinical": int(
                up.loc[up["cohort"].isin(OS6), "n_source_records_expr_and_clinical"].sum()),
            "n_excluded_upstream_harmonisation": int(
                up.loc[up["cohort"].isin(OS6), "n_excluded_upstream_harmonisation"].sum()),
            "n_analysed": int(up.loc[up["cohort"].isin(OS6), "n_analysed"].sum()),
        },
        "totals_secondary_three": {
            "n_source_records_expr_and_clinical": int(
                up.loc[up["cohort"].isin(SEC3), "n_source_records_expr_and_clinical"].sum()),
            "n_excluded_upstream_harmonisation": int(
                up.loc[up["cohort"].isin(SEC3), "n_excluded_upstream_harmonisation"].sum()),
            "n_analysed": int(up.loc[up["cohort"].isin(SEC3), "n_analysed"].sum()),
        },
        "per_cohort": up_rows,
        "events_agree_with_qc_summary": bool(up["events_match"].all()),
        "stored_n_agrees_with_qc_summary": bool(up["qc_summary_n_matches_stored"].all()),
        "median_followup_disagrees_with_qc_summary": [
            r_["cohort"] for r_ in up_rows
            if abs(r_["qc_summary_median_fu_months"]
                   - r_["median_fu_months_recomputed"]) > 0.6],
    }

summary["followup_definition_variants"] = {
    "note": "three distinct quantities; the manuscript must state which it reports",
    "per_cohort": {coh: {
        "median_observed_all_analysed": detail[coh]["median_followup_months_observed"],
        "median_observed_censored_only": detail[coh]["median_followup_months_censored_only"],
        "reverse_km_median_potential": detail[coh]["median_followup_reverse_km_months"],
    } for coh in ALL9},
}

summary["zscoring_requirement"] = {
    "note": "every analysis z-scores each gene within each cohort via "
            "nested_core.load_cohort; this records the stored state of each matrix",
    "per_cohort": {coh: {
        "bulk_appears_standardised":
            detail[coh]["stored_scale_diagnostic"]["bulk_appears_standardised"],
        "fully_standardised_all_sampled_cols":
            detail[coh]["stored_scale_diagnostic"]["fully_standardised_all_sampled_cols"],
        "n_cols_offscale": detail[coh]["stored_scale_diagnostic"]["n_cols_offscale"],
        "frac_cols_offscale": detail[coh]["stored_scale_diagnostic"]["frac_cols_offscale"],
        "max_abs_col_mean": detail[coh]["stored_scale_diagnostic"]["max_abs_col_mean"],
    } for coh in ALL9},
}
with open("participant_flow_summary.json", "w") as f:
    json.dump(summary, f, indent=1)

print("\nTOTALS all9 : analysed=%d events=%d (source=%d, excluded=%d)"
      % (summary["totals_all_nine_cohorts"]["n_analysed"],
         summary["totals_all_nine_cohorts"]["events"],
         summary["totals_all_nine_cohorts"]["n_source_matrix"],
         summary["totals_all_nine_cohorts"]["n_excluded_nonpositive_or_missing_time"]),
      flush=True)
print("TOTALS OS6  : analysed=%d events=%d"
      % (summary["totals_primary_OS_six_cohorts"]["n_analysed"],
         summary["totals_primary_OS_six_cohorts"]["events"]), flush=True)
print("TOTALS SEC3 : analysed=%d events=%d"
      % (summary["totals_secondary_three_cohorts"]["n_analysed"],
         summary["totals_secondary_three_cohorts"]["events"]), flush=True)
print("consistency_all_ok:", summary["consistency_all_ok"], flush=True)
print("wrote participant_flow.csv, participant_flow_summary.json", flush=True)
