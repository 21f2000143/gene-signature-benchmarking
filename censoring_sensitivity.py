"""
censoring_sensitivity.py
========================
Follow-up to metrics_uno_auc_ph.py (review items 2 and 3).

WHY THIS EXISTS. In the pre-specified analysis the IPCW censoring model for Uno's C,
the integrated Brier score and the time-dependent AUC is fitted on the POOLED TRAINING
cohorts. For five of the six LOCO folds this is unproblematic. For the METABRIC fold it
is not: the five training cohorts have much shorter follow-up than METABRIC (the pooled
training censoring Kaplan-Meier reaches zero at 281 months while METABRIC events run to
355 months), so the inverse-probability weights diverge. The consequence in the primary
run was that METABRIC's time-dependent AUC could not be evaluated at any horizon
("censoring survival function is zero at one or more time points"), its integrated
Brier score was 13.2 (an impossible value for a score bounded in [0,1] under correct
weights), and its Uno C fell to 0.49 while Harrell's C was 0.61. Those are artefacts of
a mis-specified censoring model, not properties of the gene panels, and reporting them
without qualification would be as misleading as the claim the review objects to.

THREE CENSORING SPECIFICATIONS ARE COMPARED (same folds, same fitted risk scores):
  (A) train_maxtau   -- pre-specified. Censoring KM fitted on the pooled training
                        cohorts; tau = min(largest event time in the held-out cohort,
                        largest time with positive training censoring probability).
                        Reproduced here from the primary run for direct comparison.
  (B) train_safetau  -- censoring KM still fitted on the pooled training cohorts, but
                        tau = min(largest event time in the held-out cohort,
                                  max{t : Ghat_train(t) >= 0.10}).
                        Restricting to where at least 10% of the training set is still
                        uncensored keeps every IPCW weight below 10.
  (C) test_own       -- censoring KM fitted on the HELD-OUT cohort itself, which is the
                        distribution actually generating that cohort's censoring;
                        tau = min(largest event time in the held-out cohort,
                                  max{t : Ghat_test(t) >= 0.10}).
                        The risk score is still purely out-of-fold: only the nuisance
                        censoring weights use test-cohort follow-up, never the outcome-
                        model coefficients.

Learner, gene sets, z-scoring and LOCO design are exactly as in metrics_uno_auc_ph.py
(ridge Cox, Breslow ties, alpha = 100; within-cohort z-scoring; six OS cohorts).

Time-dependent AUC is recomputed at 3, 5 and 10 years for Novel5 under (B) and (C) so
that the METABRIC fold, missing from the primary table, is filled in and so the honest
5-year statement can be checked for robustness to the censoring specification.

OUTPUTS
  metrics_censoring_sensitivity.csv   held_out_cohort, gene_set, spec, tau_months,
                                      uno_c, ibs, max_ipcw_weight
  tauc_censoring_sensitivity.csv      held_out_cohort, horizon_years, spec, auc, ...
  censoring_sensitivity_notes.json    tau per fold per spec, weight diagnostics
"""
import json
import warnings

import numpy as np
import pandas as pd

import nested_core as nc
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (concordance_index_ipcw, cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.nonparametric import CensoringDistributionEstimator

warnings.filterwarnings("ignore")

ALPHA = 100.0
OS6 = nc.OS6
HORIZONS = [36.0, 60.0, 120.0]
GMIN = 0.10


def surv_y(t, ev):
    return np.array([(bool(e), float(x)) for e, x in zip(ev, t)],
                    dtype=[("event", bool), ("time", float)])


def km_cens(y):
    cde = CensoringDistributionEstimator().fit(y)
    return cde.unique_time_, cde.prob_


def tau_at(gx, gy, thresh, cap):
    ok = gx[gy >= thresh]
    lim = float(ok.max()) if len(ok) else float(gx.min())
    return float(min(lim, cap))


def main():
    gene_sets = json.load(open("results/gene_sets.json"))
    store = nc.load_all(OS6)
    arms = list(gene_sets.keys())

    rows, auc_rows, notes = [], [], {}
    for held in OS6:
        train = [c for c in OS6 if c != held]
        Xte_all, tte, ete, _ = store[held]
        ttr = np.concatenate([store[c][1] for c in train])
        etr = np.concatenate([store[c][2] for c in train])
        y_tr, y_te = surv_y(ttr, etr), surv_y(tte, ete)

        gxtr, gytr = km_cens(y_tr)
        gxte, gyte = km_cens(y_te)
        max_ev_te = float(tte[ete == 1].max())
        pos = gxtr[gytr > 1e-8]
        cap_tr = float(pos.max()) if len(pos) else float(ttr.max())

        specs = {
            "A_train_maxtau": dict(cens=y_tr, tau=min(max_ev_te, cap_tr * (1 - 1e-9))),
            "B_train_safetau": dict(cens=y_tr, tau=tau_at(gxtr, gytr, GMIN, max_ev_te)),
            "C_test_own": dict(cens=y_te, tau=tau_at(gxte, gyte, GMIN, max_ev_te)),
        }
        notes[held] = {k: dict(tau=v["tau"]) for k, v in specs.items()}
        notes[held]["max_event_time_test"] = max_ev_te
        notes[held]["train_cens_support_max"] = cap_tr
        notes[held]["train_cens_G_at_60mo"] = float(
            gytr[np.searchsorted(gxtr, 60.0, side="right") - 1]) if gxtr.min() <= 60 else 1.0
        notes[held]["test_cens_G_at_60mo"] = float(
            gyte[np.searchsorted(gxte, 60.0, side="right") - 1]) if gxte.min() <= 60 else 1.0
        # largest IPCW weight actually applied under each spec
        for k, v in specs.items():
            gx, gy = km_cens(v["cens"])
            m = gx <= v["tau"]
            g = gy[m]
            g = g[g > 0]
            notes[held][k]["max_ipcw_weight"] = float(1.0 / g.min()) if len(g) else np.inf

        for arm in arms:
            nominal = gene_sets[arm]["genes"]
            avail = [g for g in nominal
                     if g in Xte_all.columns and all(g in store[c][0].columns for c in train)]
            if not avail:
                continue
            Xtr = np.vstack([store[c][0][avail].values for c in train])
            Xte = Xte_all[avail].values
            beta = nc.fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)
            risk = Xte @ beta

            m = CoxPHSurvivalAnalysis(alpha=ALPHA, ties="breslow", n_iter=200)
            m.fit(Xtr, y_tr)
            sf = m.predict_survival_function(Xte)

            for spec, cfg in specs.items():
                tau = cfg["tau"]
                try:
                    uno = float(concordance_index_ipcw(cfg["cens"], y_te, risk, tau=tau)[0])
                except Exception as e:
                    uno = np.nan
                    notes.setdefault("errors", []).append(f"uno {held}/{arm}/{spec}: {e!r}")
                try:
                    lo = float(np.quantile(tte[ete == 1], 0.05))
                    grid = np.linspace(max(lo, float(tte.min()) + 1e-6), tau * 0.999, 25)
                    grid = grid[(grid > tte.min()) & (grid < tte.max())]
                    P = np.asarray([[fn(g) for g in grid] for fn in sf])
                    ibs = float(integrated_brier_score(cfg["cens"], y_te, P, grid))
                except Exception as e:
                    ibs = np.nan
                    notes.setdefault("errors", []).append(f"ibs {held}/{arm}/{spec}: {e!r}")
                rows.append(dict(held_out_cohort=held, gene_set=arm, spec=spec,
                                 tau_months=tau, uno_c=uno, ibs=ibs,
                                 n_genes_used=len(avail),
                                 max_ipcw_weight=notes[held][spec]["max_ipcw_weight"]))
                print("  %-16s %-20s %-16s tau=%7.1f Uno=%.4f IBS=%.4f"
                      % (held, arm, spec, tau, uno, ibs), flush=True)

            if arm == "Novel5":
                for spec, cfg in specs.items():
                    for H in HORIZONS:
                        auc, reason = np.nan, ""
                        if H >= float(tte.max()):
                            reason = f"horizon exceeds cohort follow-up ({tte.max():.1f} mo)"
                        elif H > cfg["tau"]:
                            reason = (f"horizon exceeds tau for this spec "
                                      f"({cfg['tau']:.1f} mo)")
                        elif int(((ete == 1) & (tte <= H)).sum()) < 5:
                            reason = "fewer than 5 events by horizon"
                        else:
                            try:
                                auc = float(cumulative_dynamic_auc(cfg["cens"], y_te,
                                                                   risk, [H])[0][0])
                            except Exception as e:
                                reason = repr(e)
                        auc_rows.append(dict(held_out_cohort=held, spec=spec,
                                             horizon_years=H / 12.0, horizon_months=H,
                                             auc=auc,
                                             n_at_risk=int((tte >= H).sum()),
                                             events_by_horizon=int(((ete == 1) & (tte <= H)).sum()),
                                             tau_months=cfg["tau"], note=reason))
                        print("    tAUC %-16s %2.0fy: %s (%s)" % (spec, H / 12, auc, reason),
                              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("metrics_censoring_sensitivity.csv", index=False)
    ad = pd.DataFrame(auc_rows)
    ad.to_csv("tauc_censoring_sensitivity.csv", index=False)

    print("\n--- mean Uno C by gene set and specification ---", flush=True)
    piv = df.pivot_table(index="gene_set", columns="spec", values="uno_c", aggfunc="mean")
    print(piv.round(4).to_string(), flush=True)
    print("\n--- mean IBS by gene set and specification ---", flush=True)
    print(df.pivot_table(index="gene_set", columns="spec", values="ibs",
                         aggfunc="mean").round(4).to_string(), flush=True)
    print("\n--- Novel5 5-year AUC by specification ---", flush=True)
    print(ad[ad.horizon_years == 5.0].round(4).to_string(index=False), flush=True)

    json.dump(notes, open("censoring_sensitivity_notes.json", "w"), indent=1, default=str)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
