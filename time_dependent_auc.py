import json
import warnings

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '.')


import nested_core as nc
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.nonparametric import CensoringDistributionEstimator

warnings.filterwarnings("ignore")

ALPHA = 100.0
OS6 = nc.OS6
HORIZONS = [36.0, 60.0, 120.0] # 2, 4, 6 years in months


def surv_y(t, ev):
	return np.array([(bool(e), float(x)) for e, x in zip(ev, t)],
					dtype=[("event", bool), ("time", float)])


def main():
	gene_sets = json.load(open("harmonised/gene_sets.json"))
	store = nc.load_all(OS6)

	auc_rows = []

	for held in OS6:
		train = [c for c in OS6 if c != held]
		Xte_all, tte, ete, _ = store[held]
		ttr = np.concatenate([store[c][1] for c in train])
		etr = np.concatenate([store[c][2] for c in train])
		y_tr, y_te = surv_y(ttr, etr), surv_y(tte, ete)

		# --- tau: largest event time in held-out cohort, capped by train censoring support
		cens = CensoringDistributionEstimator().fit(y_tr)
		max_ev_te = float(tte[ete == 1].max())
		gx, gy = cens.unique_time_, cens.prob_
		pos = gx[gy > 1e-8]
		cap = float(pos.max()) if len(pos) else float(ttr.max())
		tau_candidates = [min(max_ev_te, cap * (1 - 1e-9))]
		for q in (0.99, 0.95, 0.90):
			tau_candidates.append(float(np.quantile(tte[ete == 1], q)))

		arm = "Novel5"
		nominal = gene_sets[arm]["genes"]
		avail = [g for g in nominal
				 if g in Xte_all.columns and all(g in store[c][0].columns for c in train)]
		if not avail:
			continue
		Xtr = np.vstack([store[c][0][avail].values for c in train])
		Xte = Xte_all[avail].values

		beta = nc.fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)
		risk = Xte @ beta

		tau_used = np.nan
		for tc in tau_candidates:
			try:
				from sksurv.metrics import concordance_index_ipcw
				concordance_index_ipcw(y_tr, y_te, risk, tau=tc)[0]
				tau_used = float(tc)
				break
			except Exception:
				pass

		for H in HORIZONS:
			n_at_risk = int((tte >= H).sum())
			ev_by = int(((ete == 1) & (tte <= H)).sum())
			auc, reason = np.nan, ""
			if H >= float(tte.max()):
				reason = ("horizon exceeds maximum follow-up in cohort "
						  f"({tte.max():.1f} months)")
			elif H >= cap:
				reason = ("horizon exceeds training censoring support "
						  f"({cap:.1f} months)")
			elif ev_by < 5:
				reason = f"only {ev_by} events by horizon"
			else:
				try:
					auc = float(cumulative_dynamic_auc(y_tr, y_te, risk, [H])[0][0])
				except Exception as e:
					reason = repr(e)
			auc_rows.append(dict(held_out_cohort=held, horizon_years=H / 12.0,
								 horizon_months=H, auc=auc,
								 n_at_risk=n_at_risk, events_by_horizon=ev_by,
								 n_test=int(len(tte)),
								 max_followup_months=float(tte.max()),
								 note=reason))
			print("    tAUC %2.0fy: %s  (%s)" % (H / 12, auc, reason), flush=True)

	auc_df = pd.DataFrame(auc_rows)
	auc_df.to_csv("results/time_dependent_auc_revised.csv", index=False)
	print("DONE", flush=True)


if __name__ == "__main__":
	main()