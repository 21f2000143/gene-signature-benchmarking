import pandas as pd

# Real measured CPU (user+sys) times from remote timing jobs
# Nested re-selection: 1 outer fold, forward search over 5 cohorts
fold_user = 25*60+41.900
fold_sys  = 1*60+7.066
fold_cpu_sec = fold_user + fold_sys
n_outer_folds = 6
nested_total_cpu_hr = fold_cpu_sec * n_outer_folds / 3600

# Permutation search: 1 replicate (full forward search + honest held-out eval)
perm_user = 38*60+2.926
perm_sys  = 1*60+8.957
perm_cpu_sec = perm_user + perm_sys
n_replicates = 91  # from permutation_search_summary.json
perm_total_cpu_hr = perm_cpu_sec * n_replicates / 3600

# Null 20k: 200-draw sample, combined build+score CPU (user+sys) from `time`
null_user = 1*60+4.137
null_sys  = 53.814
null_cpu_200 = null_user + null_sys
n_draws_small = 200
n_draws_full = 20000
null_total_cpu_hr = null_cpu_200 * (n_draws_full / n_draws_small) / 3600
# (this folds the one-time ~18s build cost into the per-draw rate at 200 draws --
#  a conservative/slight overestimate since build is amortized, not re-paid per draw)

# Primary benchmark: 10 gene sets x 6 cohorts x 4 learners = 240 cells (matches learner_grid_ranking.json n_cells_expected)
n_gene_sets = 10
n_cohorts = 6
per_learner_fits = n_gene_sets * n_cohorts  # 60
cpu_ridge = 0.269
cpu_coxnet = 0.021
cpu_rsf = 487.743
cpu_gbsa = 60.097
primary_cpu_sec = per_learner_fits * (cpu_ridge + cpu_coxnet + cpu_rsf + cpu_gbsa)
primary_total_cpu_hr = primary_cpu_sec / 3600

audit_total_cpu_hr = nested_total_cpu_hr + perm_total_cpu_hr + null_total_cpu_hr

rows = [
    {"stage": "Primary benchmark (10 gene sets x 6 cohorts x 4 learners, 240 fits)",
     "measured_from": "per-learner CPU-seconds/fit x 60 fits/learner (ridge, coxnet, RSF, GBSA)",
     "core_hours": round(primary_total_cpu_hr, 2)},
    {"stage": "Audit 1: fully nested re-selection (6 outer folds)",
     "measured_from": "1 outer-fold forward search, CPU-s x 6 folds",
     "core_hours": round(nested_total_cpu_hr, 2)},
    {"stage": "Audit 2: permutation calibration of the search (91 replicates)",
     "measured_from": "1 replicate (search + honest held-out eval), CPU-s x 91",
     "core_hours": round(perm_total_cpu_hr, 2)},
    {"stage": "Audit 3: size-matched random-panel null (20,000 draws)",
     "measured_from": "200-draw sample (build+score CPU-s) linearly scaled to 20,000",
     "core_hours": round(null_total_cpu_hr, 2)},
    {"stage": "Audit stages total (1+2+3)",
     "measured_from": "sum of above three audit stages",
     "core_hours": round(audit_total_cpu_hr, 2)},
]
df = pd.DataFrame(rows)
df.to_csv("core_hours_estimate.csv", index=False)