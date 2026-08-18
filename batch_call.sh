# python3 run_permutation_search.py
# python3 run_nested_selection.py
# python3 run_reconstruct_check.py
# python3 null_random_panels_20k.py
# python3 run_selection_naive.py
# python3 metrics_uno_auc_ph.py
# python3 hr_pooled.py
# python3 run_comparator_penalty.py
# python3 run_coverage_test.py
# python3 run_sample_normalization_c6.py
# python3 run_clinical_arm.py
# python3 censoring_sensitivity.py
# python3 participant_flow.py
# OUT=./bench python3 benchmark_within.py

# python3 run_clinical_supplement.py 
# python3 reconcile_clinical_arm.py
# python3 run_incremental_value_c1.py


# python3 hr_weights_forest.py
# python3 validate_hr.py
# python3 validate_hr_local.py results/loco_risk_scores.csv results/validation_hr_local.json
# python3 pooled_hr_stratified.py
# python3 run_c11_er_prolif_singlegene.py
# python3 make_caveats_table.py
# python3 run_fixedform_scores.py

# python3 make_fig_pipeline.py
# python3 make_fig_participant_flow.py
# python3 make_fig_cindex_dual.py
# python3 make_fig_auc_calibration.py
# python3 make_fig_nested_selection.py
# python3 make_fig_null_resolution.py
# python3 make_fig_paired_forest.py
# python3 make_fig_km_hazard.py

python table1_cohorts.py
python loco_os.py
python loco_os_pooled.py
python loco_secondary.py
python loco_secondary_pooled.py
python cross_endpoint_transfer.py
python loco_paired_novel5_vs_comparators.py
python null_random_panels.py
python null_summary.py
python observed_panels.py
python paired_bootstrap.py
python permutation_search_pooled.py
python likelihood_ratio_tests.py
python lr_tests_with_q.py
python time_dependent_auc.py
python uno_event_weighted.py
python calibration_quintiles.py
python solver_validation.py
python validation_hr_local.py
python core_hours_estimate.py
python incremental_value.py
python best_model_per_cell.py
python learner_grid_full.py
python learner_grid_summary.py
python learner_grid_ranking.py
python loco_risk_scores.py