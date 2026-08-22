import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2dist

# Load loco_risk_scores.csv
rs = pd.read_csv("results/loco_risk_scores.csv")

def stratified_logrank(time, event, group, strata):
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    group = np.asarray(group)
    strata = np.asarray(strata)
    groups = sorted(set(group))
    k = len(groups)
    OE = np.zeros(k)
    V = np.zeros((k, k))
    for s in set(strata):
        ms = strata == s
        ts, es, gs = time[ms], event[ms], group[ms]
        event_times = np.unique(ts[es == 1])
        for u in event_times:
            at_risk = ts >= u
            n_total = at_risk.sum()
            d_total = ((ts == u) & (es == 1)).sum()
            if n_total <= 1 or d_total == 0:
                continue
            n_g = np.array([(at_risk & (gs == g)).sum() for g in groups])
            d_g = np.array([((ts == u) & (es == 1) & (gs == g)).sum() for g in groups])
            e_g = d_total * n_g / n_total
            OE += (d_g - e_g)
            if n_total > 1:
                factor = d_total * (n_total - d_total) / (n_total - 1)
                for i in range(k):
                    for j in range(k):
                        if i == j:
                            V[i, j] += factor * (n_g[i]/n_total) * (1 - n_g[i]/n_total)
                        else:
                            V[i, j] += -factor * (n_g[i]/n_total) * (n_g[j]/n_total)
    OEr = OE[:-1]
    Vr = V[:-1, :-1]
    try:
        chi2_stat = float(OEr @ np.linalg.solve(Vr, OEr))
    except np.linalg.LinAlgError:
        chi2_stat = float(OEr @ np.linalg.pinv(Vr) @ OEr)
    df = k - 1
    p = float(1 - chi2dist.cdf(chi2_stat, df))
    return chi2_stat, df, p

chi2_strat, df_strat, p_strat = stratified_logrank(
    rs.time_months, rs.event, rs.risk_group_tertile, rs.cohort)

# p_strat_exact uses the closed form exp(-chi2/2) for df=2
p_strat_exact = np.exp(-chi2_strat / 2)

kmt = pd.read_csv("loco_risk_scores.csv")

# Rebuild km_stratification.csv from loco_risk_scores.csv
# The km_stratification.csv was previously computed; we reconstruct it by reading
# the already-existing version and updating the POOLED row with the stratified test.
# Since we only have loco_risk_scores.csv, we need to reconstruct per-cohort stats.

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

OS6 = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203", "GSE20711", "GSE58812"]

rows = []
for cohort in OS6:
    sub = rs[rs.cohort == cohort]
    n = len(sub)
    events = int(sub.event.sum())
    
    # HR high vs low (tertile groups 1=low, 2=mid, 3=high)
    low = sub[sub.risk_group_tertile == 1]
    high = sub[sub.risk_group_tertile == 3]
    
    try:
        lr = logrank_test(high.time_months, low.time_months,
                          event_observed_A=high.event, event_observed_B=low.event)
        p_val = float(lr.p_value)
        
        # Simple HR estimate: (events_high/person_time_high) / (events_low/person_time_low)
        hr = ((high.event.sum() / high.time_months.sum()) /
              (low.event.sum() / low.time_months.sum())) if low.event.sum() > 0 else np.nan
    except Exception:
        p_val = np.nan
        hr = np.nan
    
    rows.append(dict(cohort=cohort, n=n, events=events,
                     hr_high_vs_low=round(hr, 4) if np.isfinite(hr) else np.nan,
                     logrank_p=round(p_val, 6) if np.isfinite(p_val) else np.nan))

# Pooled row
pooled_sub = rs
rows.append(dict(cohort="POOLED", n=len(pooled_sub), events=int(pooled_sub.event.sum()),
                 hr_high_vs_low=np.nan, logrank_p=np.nan,
                 logrank_chi2_stratified=chi2_strat,
                 logrank_p_stratified=p_strat_exact))

kmt = pd.DataFrame(rows)
kmt.to_csv("km_stratification.csv", index=False)
print(kmt.to_string())
print("\ncohort-stratified pooled tertile log-rank: chi2=%.3f df=%d p=%.3e" % (chi2_strat, df_strat, p_strat_exact))