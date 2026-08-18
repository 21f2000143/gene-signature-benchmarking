import pandas as pd
from scipy.stats import false_discovery_control

lr = pd.read_csv('likelihood_ratio_tests.csv')

rec = []
for (coh, arm), g in lr.groupby(['cohort','arm']):
    q = false_discovery_control(g.p.values, method='bh')
    for (_, r), qq in zip(g.iterrows(), q):
        rec.append({**r.to_dict(), 'q_bh_within_cohort_arm': qq})
lrq = pd.DataFrame(rec)
lrq.to_csv("lr_tests_with_q.csv", index=False)