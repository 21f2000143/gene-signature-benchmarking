import pandas as pd
import numpy as np
import json
import host
import os

calj = json.load(open(host.artifact_path("f37d0af9-7bbf-4ed5-834b-61b4f1c5e462")))

C4 = ['TCGA', 'METABRIC', 'SCANB_GSE96058', 'SCANB_GSE202203']

cr = pd.DataFrame([dict(cohort=c, landmark_months=calj['landmark_months'][c], **r)
                   for c in C4 for r in calj['quintiles'][c]]).round(4)

mono = {}
for coh in C4:
    y = [r['obs_surv'] for r in calj['quintiles'][coh]]
    d = np.diff(y)
    from scipy.stats import spearmanr
    rho, pv = spearmanr([r['quintile'] for r in calj['quintiles'][coh]], y)
    mono[coh] = dict(strict=bool((d < 0).all()), n_reversals=int((d > 0).sum()),
                     q1_q5=round(y[0]-y[-1],4), rho=round(float(rho),3), p=float(pv),
                     reversal_sizes=[round(float(x),4) for x in d[d>0]])

cr2 = cr.merge(pd.DataFrame([{"cohort":k, "strictly_monotone":v['strict'], "n_reversals":v['n_reversals'],
                               "spearman_rho":v['rho'], "spearman_p":round(v['p'],5),
                               "q1_minus_q5_surv":v['q1_q5']} for k,v in mono.items()]), on="cohort")
os.makedirs("setup", exist_ok=True)
cr2.to_csv(os.path.join("setup", "calibration_quintiles.csv"), index=False)