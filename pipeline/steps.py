"""Declarative registry of pipeline steps, transcribed from README.md's
"Run order" section (stages 1-6). Stage 0 (harmonise.py) is intentionally
excluded — it is a separate, one-time raw-data preparation step run outside
this orchestrator (see CLAUDE.md / README.md prerequisites).

Each Step describes exactly one `python3 <script>.py [args]` invocation as
README documents it: same script, same arguments, same environment
variables, same working directory (the repo root for every step -- see the
`extra_copy` note on the learner_grid_* steps below for how their
shard_out/ output is produced without changing their working directory).
Nothing about *how* any script computes its numbers is touched here.

category:
    "onetime" - stage 2/3 scripts. These are the expensive, numerically-heavy
        core analyses. Several of them write plain relative filenames (e.g.
        "hr_per_cohort.csv") which land at the repo root rather than
        results/, while some *downstream* stage-3 scripts then read the same
        file back from results/ (e.g. pooled_hr_stratified.py reads
        "results/loco_risk_novel5.csv"). This was already a manual-copy step
        in the original workflow; the runner makes it explicit by mirroring
        every declared output into results/ after a onetime step completes.
        onetime steps are cached in ./harmonised/<step-name>/ and, on a
        later run, are skipped (outputs restored from cache) unless the user
        asks to regenerate them.
    "always" - stage 4/5/6 scripts (figures, tables, verification). These
        already read/write results/, media/, paper/ directly (RES="results"
        convention) and are cheap to re-run, so they are treated as "the
        main analysis" that always re-executes on every `main.py` run and is
        snapshotted into results/<run-timestamp>/.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    name: str
    stage: int
    category: str  # "onetime" | "always"
    script: str | None  # relative to repo root; None if the script does not
                         # exist in this checkout (documented but unavailable)
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = "."  # relative to repo root
    # Outputs as *written* by the script (relative to `cwd`). For onetime
    # steps these get mirrored into results/<basename> after the run.
    outputs: list[str] = field(default_factory=list)
    # Extra directories (relative to repo root) to also copy each declared
    # output's basename into, after the step runs or is restored from cache.
    extra_copy: list[str] = field(default_factory=list)
    note: str = ""


OS6_COHORTS = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203",
               "GSE20711", "GSE58812"]

STEPS: list[Step] = [
    # ---------------------------------------------------------------
    # Stage 1 - independent core analyses (order-free amongst themselves,
    # kept in README order for readability).
    # ---------------------------------------------------------------
# Scripts deliberately NOT registered as steps (checked 2026-08-22):
#   loco_paired_noel5_vs_comparators.py -- now registered, see stage 3 below.
#   clin_harmonise.py, nested_core.py    -- shared library modules, not standalone scripts.
#   ibc_notebook                         -- not a script in this checkout.
#   ph_tests.csv                         -- written by metrics_uno_auc_ph.py (registered);
#                                          there is no separate ph_tests.py to run.
#   loco_os.py, loco_secondary.py, cross_endpoint_transfer.py -- near-identical copies
#                                          of loco_os_pooled.py's monolithic sweep; only
#                                          loco_os_pooled is registered (see stage 2).
#   null_random_panels.py, observed_panels.py -- redundant subsets of null_summary.py's
#                                          computation; only null_summary is registered.
#   time_dependent_auc.py                -- duplicates metrics_uno_auc_ph.py's tAUC
#                                          computation under a different, unregistered
#                                          output filename (results/time_dependent_auc.csv,
#                                          vs the registered time_dependent_auc_revised.csv).
#                                          That plain-named CSV in results/ is a stale,
#                                          pre-supplied file with no current generator.
#   km_stratification.py                 -- BROKEN against current data: expects a
#                                          `risk_group_tertile` column in
#                                          results/loco_risk_scores.csv that the current
#                                          hr_pooled.py output does not have.
#   timevarying_hr_windows.py            -- runs without raising, but every hazard ratio
#                                          comes out NaN (the per-window Cox fit silently
#                                          fails); needs investigation before it is wired in.
#   permutation_search_pooled.py         -- reads permutation_search_s0..s3.csv, sharded
#                                          outputs of some earlier multi-seed batch run
#                                          that are not present anywhere in this checkout.
#   calibration_quintiles.py             -- `import host` / `host.artifact_path(...)` is
#                                          an interactive-session helper, not a real
#                                          package; not portable outside whatever tool
#                                          originally generated setup/calibration_quintiles.csv.
    Step("run_nested_selection", 2, "onetime", "run_nested_selection.py",
         outputs=["nested_selection_folds.csv", "nested_selection_traces.csv",
                   "nested_selection_consensus.csv", "nested_selection_stability.csv",
                   "nested_selection_summary.json"]),
    Step("run_reconstruct_check", 2, "onetime", "run_reconstruct_check.py",
         outputs=["reconstruct_check.csv", "reconstruct_top10.csv",
                   "reconstruct_consensus.csv", "reconstruct_summary.json"]),
    Step("run_permutation_search", 2, "onetime", "run_permutation_search.py",
         outputs=["permutation_search.csv", "permutation_search_summary.json"],
         note="README allows [N_REP] [SEED0] [tag] positional args; run with defaults"),
    Step("null_random_panels_20k", 2, "onetime", "null_random_panels_20k.py",
         outputs=["null_random_panels_20k.csv", "null_20k_summary.json"]),
    Step("run_selection_naive", 2, "onetime", "run_selection_naive.py",
         outputs=["selection_naive.csv", "selection_naive_summary.json"]),
    Step("metrics_uno_auc_ph", 2, "onetime", "metrics_uno_auc_ph.py",
         outputs=["metrics_harrell_uno.csv", "time_dependent_auc_revised.csv",
                   "metrics_summary.csv", "loco_risk_novel5.csv", "ph_tests.csv",
                   "run_notes.json"],
         extra_copy=["."],
         note="extra_copy mirrors metrics_harrell_uno.csv back to repo root too, "
              "since uno_event_weighted.py (stage 3) reads it as a bare relative path"),
    Step("hr_pooled", 2, "onetime", "hr_pooled.py",
         outputs=["hr_per_cohort.csv", "loco_risk_scores.csv", "hr_pooled_methods.csv",
                   "heterogeneity.json", "hr_loco_sensitivity.csv", "resolution_floor.json"],
         note="resolution_floor.json was previously omitted from this Step's outputs, "
              "so it was never mirrored to results/ or cached -- make_fig_null_resolution.py "
              "and make_fig_paired_forest.py read a stale, pre-supplied copy instead of a "
              "fresh one; added here so it flows through the normal pipeline"),
    Step("run_comparator_penalty", 2, "onetime", "run_comparator_penalty.py",
         outputs=["comparator_coverage_penalty.csv", "comparator_alpha_sweep.csv",
                   "comparator_within_cohort.csv", "comparator_penalty_summary.json"]),
    Step("run_coverage_test", 2, "onetime", "run_coverage_test.py",
         outputs=["coverage_test.csv", "coverage_test_paired.csv", "coverage_test_summary.json"]),
    Step("run_sample_normalization_c6", 2, "onetime", "run_sample_normalization_c6.py",
         outputs=["sample_normalisation_c6.csv"]),
    Step("run_clinical_arm", 2, "onetime", "run_clinical_arm.py",
         outputs=["clinical_arm_audit.csv", "clinical_arm_recomputed.csv",
                   "clinical_arm_summary.json"]),
    Step("censoring_sensitivity", 2, "onetime", "censoring_sensitivity.py",
         outputs=["metrics_censoring_sensitivity.csv", "tauc_censoring_sensitivity.csv",
                   "censoring_sensitivity_notes.json"]),
    Step("participant_flow", 2, "onetime", "participant_flow.py",
         outputs=["participant_flow_upstream.csv", "participant_flow.csv",
                   "participant_flow_summary.json"],
         note="needs harmonised/qc_summary.csv from stage 0"),
    Step("benchmark_within", 2, "onetime", "benchmark_within.py",
         env={"OUT": "./bench"},
         outputs=["bench/within_cohort_folds.csv", "bench/within_cohort_summary.csv"]),

    # learner_grid is sharded by held-out cohort; each shard is its own
    # cacheable step. learner_grid.py's `nc.load_all()` call resolves
    # nested_core.DATA ("harmonised") relative to cwd (not __file__), so it
    # must run with cwd="." (repo root) or it can't find the parquet inputs.
    # Its bare relative output `learner_grid_<COHORT>.csv` therefore lands
    # at the repo root too; extra_copy mirrors it into shard_out/ afterward
    # so combine_grid.py's default `shard_dir="shard_out"` still finds it --
    # a post-hoc copy only, learner_grid.py's own code is untouched.
    *[Step(f"learner_grid_{coh}", 2, "onetime", "learner_grid.py",
           env={"SHARD": coh},
           outputs=[f"learner_grid_{coh}.csv"], extra_copy=["shard_out"])
      for coh in OS6_COHORTS],
    Step("combine_grid", 2, "onetime", "combine_grid.py",
         outputs=["learner_grid_full.csv", "learner_grid_summary.csv",
                   "learner_grid_ranking.json"],
         note="reads shard_out/learner_grid_<COHORT>.csv, mirrored there by the "
              "6 learner_grid_* steps above via extra_copy"),

    Step("run_clinical_supplement", 2, "onetime", "run_clinical_supplement.py",
         outputs=["clinical_arm_supplement.csv", "clinical_arm_incremental.csv",
                   "clinical_arm_supplement_summary.json"],
         note="asserts its reused covariate-harmonisation block matches run_clinical_arm.py; "
              "must run after run_clinical_arm"),
    Step("reconcile_clinical_arm", 2, "onetime", "reconcile_clinical_arm.py",
         outputs=["clinical_arm_reconciled.csv", "clinical_arm_reconciled.json"],
         note="unifies metrics_uno_auc_ph.py vs run_clinical_arm.py; run after both"),
    Step("run_incremental_value_c1", 2, "onetime", "run_incremental_value_c1.py",
         outputs=["incremental_lr_dca_c1.csv", "incremental_dca_curves_c1.csv"]),
    Step("run_incremental_value_c1_common", 2, "onetime", "run_incremental_value_c1_common.py",
         outputs=["incremental_lr_dca_c1_common.csv", "incremental_dca_curves_c1_common.csv"],
         note="common-covariate-specification counterpart to run_incremental_value_c1; "
              "was never registered, so its two CSVs were never (re)produced by main.py"),

    Step("table1_cohorts", 2, "onetime", "table1_cohorts.py",
         outputs=["setup/table1_cohorts.csv"], extra_copy=["."],
         note="extra_copy mirrors it to repo root too, since uno_event_weighted.py "
              "(stage 3) reads it as a bare relative path"),
    Step("core_hours_estimate", 2, "onetime", "core_hours_estimate.py",
         outputs=["core_hours_estimate.csv"]),
    Step("likelihood_ratio_tests", 2, "onetime", "likelihood_ratio_tests.py",
         outputs=["setup/likelihood_ratio_tests.csv"], extra_copy=["."],
         note="extra_copy mirrors it to repo root too, since lr_tests_with_q.py "
              "(stage 3) reads it as a bare relative path"),
    Step("solver_validation", 2, "onetime", "solver_validation.py",
         outputs=["setup/solver_validation.csv"],
         note="reads from ~/.claude-science-scratch/bc_bench (a copy of the harmonised "
              "parquet files outside this repo's own harmonised/ convention); confirmed "
              "present and working in this checkout, but this path is environment-specific "
              "and may not exist elsewhere"),
    Step("rsf_mtry_control", 2, "onetime", "rsf_mtry_control.py",
         outputs=["results/rsf_mtry_control.csv", "results/rsf_mtry_control_summary.json"],
         note="writes directly under results/ already; expensive (ProcessPoolExecutor "
              "grid over RSF hyperparameters) -- was never registered, so it never reran, "
              "and the checked-in CSV was from a prior manual invocation"),
    Step("loco_os_pooled", 2, "onetime", "loco_os_pooled.py",
         outputs=["setup/loco_os.csv", "setup/loco_secondary.csv",
                   "setup/cross_endpoint_transfer.csv", "setup/loco_os_pooled.csv"],
         note="loco_os.py, loco_secondary.py and cross_endpoint_transfer.py are "
              "near-byte-identical copies of the same monolithic LOCO/transfer sweep "
              "(Parallel n_jobs=54, expensive); this is the superset variant (also "
              "computes the pooled summary) and is registered in their place so the "
              "same computation does not run three times over"),
    Step("null_summary", 2, "onetime", "null_summary.py",
         outputs=["setup/gene_universe_filter.csv", "setup/observed_panels.csv",
                   "setup/null_random_panels.csv", "setup/null_summary.csv"],
         note="null_random_panels.py and observed_panels.py are redundant subsets of "
              "this script's computation and are registered in their place; fixed a "
              "`NameError: name 'fn' is not defined` (a stray dead statement) that made "
              "this script crash before it could write any output. Expensive by default "
              "(N_SIZE_MATCHED=500 random draws per gene set per cohort)"),
    Step("gene_clinical_loco", 2, "onetime", "run_gene_clinical_loco.py",
         outputs=["results/gene_clinical_arm_loco.csv", "results/gene_clinical_arm_summary.json"],
         note="combined gene+clinical LOCO arm for every gene set (Table "
              "gene_clinical in cmpb_revised/manuscript.tex); self-contained, loads "
              "cohorts and harmonises clinical covariates itself, same audited rule as "
              "reconcile_clinical_arm.py"),

    # ---------------------------------------------------------------
    # Stage 2 - analyses that depend on stage-2 outputs.
    # ---------------------------------------------------------------
    Step("hr_weights_forest", 3, "onetime", "hr_weights_forest.py",
         outputs=["hr_cohort_weights.csv"],
         note="needs hr_per_cohort.csv, hr_pooled_methods.csv, heterogeneity.json (hr_pooled)"),
    Step("validate_hr", 3, "onetime", "validate_hr.py",
         outputs=["validation_hr.json"],
         note="needs loco_risk_scores.csv (hr_pooled)"),
    Step("validate_hr_local", 3, "onetime", "validate_hr_local.py",
         args=["results/loco_risk_scores.csv", "results/validation_hr_local.json"],
         outputs=["results/validation_hr_local.json"],
         note="needs results/loco_risk_scores.csv (mirrored copy of hr_pooled's output)"),
    Step("pooled_hr_stratified", 3, "onetime", "pooled_hr_stratified.py",
         outputs=["pooled_hr_stratified.csv", "results/pooled_hr_summary.json"],
         note="needs results/loco_risk_novel5.csv (metrics_uno_auc_ph)"),
    Step("run_c11_er_prolif_singlegene", 3, "onetime", "run_c11_er_prolif_singlegene.py",
         outputs=["er_stratified_concordance_c11.csv", "proliferation_confound_c11.csv",
                   "joint_hypoxia_model_c11.csv", "proliferation_pooled_summary_c11.json",
                   "mki67_single_gene_arm_c11.csv"],
         note="needs results/loco_risk_novel5.csv (metrics_uno_auc_ph)"),
    Step("make_caveats_table", 3, "onetime", "make_caveats_table.py",
         outputs=["comparator_implementation_caveats.csv"],
         note="needs comparator_coverage_penalty.csv (run_comparator_penalty)"),
    Step("run_fixedform_scores", 3, "onetime", "run_fixedform_scores.py",
         outputs=["comparator_fixedform.csv", "comparator_fixedform_summary.json"]),
    Step("best_model_per_cell", 3, "onetime", "best_model_per_cell.py",
         outputs=["best_model_per_cell.csv"],
         note="needs results/within_cohort_folds.csv (benchmark_within)"),
    Step("incremental_value", 3, "onetime", "incremental_value.py",
         outputs=["incremental_value.csv"],
         note="needs results/within_cohort_folds.csv (benchmark_within); the "
              "gene+clinical-vs-clinical-vs-gene-only within-cohort comparison, "
              "across every gene set"),
    Step("loco_secondary_pooled", 3, "onetime", "loco_secondary_pooled.py",
         outputs=["setup/loco_secondary_pooled.csv"],
         note="needs setup/loco_secondary.csv (loco_os_pooled); fixed a hardcoded "
              "input path (D='hpc/', a directory that does not exist in this "
              "checkout) to D='setup/', where loco_os_pooled.py actually writes it"),
    Step("loco_paired_novel5_vs_comparators", 3, "onetime",
         "loco_paired_noel5_vs_comparators.py",
         outputs=["setup/loco_paired_novel5_vs_comparators.csv"],
         note="script filename keeps the repo's own 'noel5' typo; expensive "
              "(LOCO refit per comparator pair) -- allow a long timeout"),
    Step("uno_event_weighted", 3, "onetime", "uno_event_weighted.py",
         outputs=["uno_event_weighted.csv"],
         note="needs table1_cohorts.csv and metrics_harrell_uno.csv at the repo root "
              "(bare relative paths); both are mirrored there via extra_copy on the "
              "table1_cohorts and metrics_uno_auc_ph steps"),

    # ---------------------------------------------------------------
    # Stage 3 - figures. Already read/write results/ and media/ directly.
    # ---------------------------------------------------------------
    Step("make_fig_pipeline", 4, "always", "make_fig_pipeline.py",
         note="schematic only, no data dependency"),
    Step("make_fig_participant_flow", 4, "always", "make_fig_participant_flow.py",
         note="needs participant_flow.csv, participant_flow_upstream.csv"),
    Step("make_fig_cindex_dual", 4, "always", "make_fig_cindex_dual.py",
         note="needs metrics_summary.csv, metrics_harrell_uno.csv, clinical_arm_reconciled.csv"),
    Step("make_fig_auc_calibration", 4, "always", "make_fig_auc_calibration.py",
         note="needs time_dependent_auc_revised.csv, tauc_censoring_sensitivity.csv, "
              "calibration_quintiles.csv (pre-supplied)"),
    Step("make_fig_nested_selection", 4, "always", "make_fig_nested_selection.py",
         note="needs nested_selection_folds.csv, _stability.csv, "
              "permutation_search_pooled.csv (pre-supplied), permutation_search_summary.json, "
              "selection_naive.csv"),
    Step("make_fig_null_resolution", 4, "always", "make_fig_null_resolution.py",
         note="needs null_random_panels_20k.csv, null_20k_summary.json, "
              "null_summary.csv (pre-supplied), resolution_floor.json (pre-supplied)"),
    Step("make_fig_paired_forest", 4, "always", "make_fig_paired_forest.py",
         note="needs loco_paired_novel5_vs_comparators.csv, resolution_floor.json (pre-supplied); "
              "writes results/paired_sign_test_by_comparator.csv"),
    Step("make_fig_km_hazard", 4, "always", "make_fig_km_hazard.py",
         note="needs loco_risk_scores.csv (hr_pooled), km_stratification.csv, ph_tests.csv "
              "(pre-supplied, then overwritten), pooled_hr_stratified.csv, pooled_hr_summary.json"),

    # ---------------------------------------------------------------
    # Stage 4 - tables. Must run after stage 4 (needs
    # paired_sign_test_by_comparator.csv written by make_fig_paired_forest).
    # ---------------------------------------------------------------
    Step("make_tables", 5, "always", "make_tables.py",
         note="writes paper/tab1_cohorts.tex .. tab9_samplenorm.tex"),

    # ---------------------------------------------------------------
    # Stage 5 - verification. Run last.
    # ---------------------------------------------------------------
    Step("verify_manuscript_numbers", 6, "always", "verify_manuscript_numbers.py"),
]


def get_step(name: str) -> Step:
    for s in STEPS:
        if s.name == name:
            return s
    raise KeyError(f"unknown step: {name}")


def stages() -> list[int]:
    return sorted({s.stage for s in STEPS})
