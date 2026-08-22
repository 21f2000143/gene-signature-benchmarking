"""
metrics_uno_auc_ph.py
=====================
Revision analyses for CMPB submission, censoring-robust-metrics track.

Review item 3  (a): the manuscript pre-specifies Uno's censoring-robust C but reports
                    only Harrell's C. Here BOTH are computed for every LOCO fold x gene
                    set under the pre-specified learner (ridge Cox, alpha=100).
Review item 2  (b): the manuscript claims time-dependent AUC "consistently above 0.70
                    at 5 years". Here cumulative/dynamic AUC is recomputed at 3, 5 and
                    10 years on held-out LOCO risk scores.
New (PH)       (c): whether a single hazard ratio is even meaningful is tested by a
                    Grambsch-Therneau scaled-Schoenfeld-residual test.

EXACT DEFINITIONS
-----------------
Design. Leave-one-cohort-out (LOCO) over the six overall-survival cohorts
  TCGA, METABRIC, SCANB_GSE96058, SCANB_GSE202203, GSE20711, GSE58812.
For held-out cohort h: train on the pooled remaining five, evaluate once on h.
Every gene is z-scored WITHIN each cohort before pooling (nested_core.load_cohort);
this matters because the SCANB_GSE96058 source matrix is not z-scored.

Learner. Ridge-penalised Cox partial likelihood, Breslow ties, alpha = 100
  (nested_core.fit_ridge_cox — the pre-specified implementation). A scikit-survival
  CoxPHSurvivalAnalysis(alpha=100, ties='breslow') is ALSO fitted on the identical
  training matrix, solely to obtain a Breslow baseline hazard for the integrated
  Brier score; the maximum absolute difference in the two risk scores' concordance
  is recorded per fold as column `c_agreement_absdiff` (sanity check, not a result).

Harrell's C. sksurv.metrics.concordance_index_censored on the held-out cohort,
  using the held-out linear predictor. (Cross-checked against nested_core.cindex.)

Uno's C. sksurv.metrics.concordance_index_ipcw(survival_train, survival_test,
  estimate, tau). The IPCW censoring model is the Kaplan-Meier estimate of the
  CENSORING distribution fitted on the TRAINING data (this is what passing
  survival_train does). tau is chosen per fold as
      tau = min( largest EVENT time in the held-out cohort ,
                 largest time at which the training censoring KM is still > 0 )
  and, if that value still yields a non-finite IPCW weight, is reduced to the
  0.99 / 0.95 / 0.90 quantile of held-out event times (first value that evaluates).
  The tau actually used is reported per fold in column tau_months.

Integrated Brier score. sksurv.metrics.integrated_brier_score over a grid of 25
  equally spaced times between the 5th percentile of held-out event times and tau,
  using Breslow survival curves from the sksurv fit. Lower is better.

Time-dependent AUC. sksurv.metrics.cumulative_dynamic_auc(survival_train,
  survival_test, estimate, times) evaluated at 36, 60 and 120 months on the held-out
  LOCO risk score. A horizon is reported as NaN (with a reason) when it exceeds the
  follow-up actually available in that cohort or in the training censoring support.
  n_at_risk = #{held-out subjects with time >= horizon};
  events_by_horizon = #{held-out subjects with event=1 and time <= horizon}.

Proportional hazards. For each cohort the covariate is that cohort's OUT-OF-FOLD
  (LOCO) Novel5 linear predictor, z-scored within the cohort; for "POOLED" the six
  cohorts' out-of-fold, within-cohort z-scored predictors are concatenated. An
  unpenalised Cox model (Breslow ties) is fitted on that single covariate and the
  Grambsch-Therneau global test is applied to the scaled Schoenfeld residuals with
  the rank transform of time:
      T = [ sum_k (g_k - gbar) * s_k ]^2 / ( (V/d) * sum_k (g_k - gbar)^2 ) ~ chi2_1
  where s_k are the (unscaled) Schoenfeld residuals at the d event times, V the
  observed Cox information, and g_k the rank of the k-th event time.
  lifelines is NOT installed on this host, so this is an own implementation of the
  standard test; it is validated on the pooled data against a second, independent
  test (Pearson correlation of scaled Schoenfeld residuals with rank-time, t-test)
  reported as method 'schoenfeld_corr_ttest'.

Clinical arm. There is no clinical variable set common to all six cohorts, so the arm
  is defined by an explicit availability rule rather than a fixed list. Candidate
  covariates: age (years), grade_high (G3 vs G1/G2), size (mm), node_pos, er_pos,
  harmonised from heterogeneous encodings. A covariate is "available" in a cohort if
  >=80% non-missing with non-zero variance there. In a given fold a covariate is used
  if it is available in the held-out cohort AND in >=3 of the 5 training cohorts;
  training cohorts lacking it contribute the cohort mean (0 after within-cohort
  z-scoring). Missing values are cohort-median imputed, then z-scored within cohort.
  The covariates actually used are listed per fold in column features_used.
  CAVEAT recorded in the output: GSE20711 supplies age/size/node only as 0/1
  dichotomies, and GSE58812 is all-TNBC so er_pos has no variance there.

OUTPUTS
  metrics_harrell_uno.csv        one row per (held_out_cohort, gene_set)
  metrics_summary.csv            mean/SD of both C metrics per gene set + ranks
  time_dependent_auc_revised.csv Novel5 tAUC at 3/5/10 years per held-out cohort
  ph_tests.csv                   PH tests per cohort and pooled
  loco_risk_novel5.csv           out-of-fold Novel5 risk scores (feeds other tracks)
  run_notes.json                 tau derivation, gene availability, caveats
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import nested_core as nc
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            cumulative_dynamic_auc, integrated_brier_score)
from sksurv.nonparametric import CensoringDistributionEstimator

warnings.filterwarnings("ignore")

ALPHA = 100.0
OS6 = nc.OS6
HORIZONS = [36.0, 60.0, 120.0]
NOTES = {}


def surv_y(t, ev):
    return np.array([(bool(e), float(x)) for e, x in zip(ev, t)],
                    dtype=[("event", bool), ("time", float)])


# ----------------------------------------------------------------- clinical harmonisation
def _num(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def harmonise_clinical(s, coh):
    n = len(s)
    out = pd.DataFrame(index=s.index)

    # age -------------------------------------------------------------------
    out["age"] = pd.to_numeric(s["age"], errors="coerce") if "age" in s else np.nan

    # grade_high (G3 vs G1/G2) ----------------------------------------------
    if "grade" in s:
        g = s["grade"].astype(str).str.upper().str.replace("G", "", regex=False)
        g = pd.to_numeric(g, errors="coerce")
        out["grade_high"] = (g >= 3).astype(float).where(g.notna())
    else:
        out["grade_high"] = np.nan

    # size (mm) --------------------------------------------------------------
    out["size"] = pd.to_numeric(s["size"], errors="coerce") if "size" in s else np.nan

    # node positivity --------------------------------------------------------
    if "node" in s:
        raw = s["node"].astype(str)
        num = pd.to_numeric(raw, errors="coerce")
        npos = pd.Series(np.nan, index=s.index)
        # numeric encodings: METABRIC node count, GSE20711 0/1
        npos[num.notna()] = (num[num.notna()] > 0).astype(float)
        # string encodings
        strm = raw[num.isna()].str.upper()
        neg = strm.str.startswith("N0") | strm.str.contains("NODENEGATIVE")
        nx = strm.str.strip().isin(["NX", "NAN", "NONE", ""])
        val = pd.Series(1.0, index=strm.index)
        val[neg] = 0.0
        val[nx] = np.nan
        npos[strm.index] = val
        out["node_pos"] = npos
    else:
        out["node_pos"] = np.nan

    # ER positivity ----------------------------------------------------------
    if "er" in s:
        raw = s["er"].astype(str).str.upper().str.strip()
        num = pd.to_numeric(raw, errors="coerce")
        erp = pd.Series(np.nan, index=s.index)
        erp[num.notna()] = (num[num.notna()] > 0).astype(float)
        strm = raw[num.isna()]
        val = pd.Series(np.nan, index=strm.index)
        val[strm.str.startswith("POS")] = 1.0
        val[strm.str.startswith("NEG")] = 0.0
        erp[strm.index] = val
        out["er_pos"] = erp
    else:
        out["er_pos"] = np.nan
    return out


CLIN_VARS = ["age", "grade_high", "size", "node_pos", "er_pos"]


def clinical_availability(clin):
    """var available in cohort if >=80% non-missing and SD>0."""
    av = {}
    for v in CLIN_VARS:
        col = clin[v]
        av[v] = bool(col.notna().mean() >= 0.8 and np.nanstd(col.values.astype(float)) > 1e-9)
    return av


def clinical_matrix(clin, varlist, available):
    """cohort-median impute -> within-cohort z-score; unavailable vars -> 0."""
    M = np.zeros((len(clin), len(varlist)))
    for j, v in enumerate(varlist):
        if not available.get(v, False):
            continue
        x = clin[v].values.astype(float)
        med = np.nanmedian(x)
        x = np.where(np.isfinite(x), x, med)
        sd = np.nanstd(x)
        M[:, j] = (x - np.nanmean(x)) / (sd if sd > 1e-9 else 1.0)
    return M


# ------------------------------------------------------------------------------ Schoenfeld
def cox_newton_unpenalised(x, t, ev, max_iter=100, tol=1e-9):
    """Single-covariate Cox, Breslow ties. Returns beta, information V."""
    o = np.argsort(-t, kind="mergesort")
    x, t, ev = x[o], t[o], ev[o]
    beta = 0.0
    for _ in range(max_iter):
        w = np.exp(beta * x)
        s0 = s1 = s2 = 0.0
        grad = 0.0
        hess = 0.0
        i = 0
        n = len(x)
        while i < n:
            j = i
            tt = t[i]
            while j < n and t[j] == tt:
                s0 += w[j]
                s1 += w[j] * x[j]
                s2 += w[j] * x[j] * x[j]
                j += 1
            d = int(ev[i:j].sum())
            if d > 0:
                grad += x[i:j][ev[i:j] == 1].sum() - d * (s1 / s0)
                hess -= d * (s2 / s0 - (s1 / s0) ** 2)
            i = j
        V = -hess
        if V <= 0:
            break
        step = grad / V
        beta += step
        if abs(step) < tol:
            break
    return beta, V


def schoenfeld(x, t, ev, beta):
    """Schoenfeld residuals at each event (Breslow). Returns (times, resid, Vk)."""
    o = np.argsort(-t, kind="mergesort")
    x, t, ev = x[o], t[o], ev[o]
    w = np.exp(beta * x)
    s0 = s1 = s2 = 0.0
    times, res, vks = [], [], []
    i, n = 0, len(x)
    while i < n:
        j = i
        tt = t[i]
        while j < n and t[j] == tt:
            s0 += w[j]
            s1 += w[j] * x[j]
            s2 += w[j] * x[j] * x[j]
            j += 1
        xbar = s1 / s0
        vk = s2 / s0 - xbar ** 2
        for k in range(i, j):
            if ev[k] == 1:
                times.append(tt)
                res.append(x[k] - xbar)
                vks.append(vk)
        i = j
    return np.array(times), np.array(res), np.array(vks)


def ph_test(x, t, ev, label):
    x = np.asarray(x, float)
    x = (x - x.mean()) / (x.std() if x.std() > 1e-12 else 1.0)
    beta, V = cox_newton_unpenalised(x, np.asarray(t, float), np.asarray(ev, int))
    tt, s, vk = schoenfeld(x, np.asarray(t, float), np.asarray(ev, int), beta)
    d = len(s)
    if d < 5 or V <= 0:
        return dict(cohort=label, beta=beta, n_events=d, test_stat=np.nan, df=1,
                    p_value=np.nan, method="insufficient events", ph_violated_05=None,
                    corr_scaled_rank=np.nan, p_corr=np.nan)
    g = stats.rankdata(tt)
    gc = g - g.mean()
    num = float((gc * s).sum()) ** 2
    den = (V / d) * float((gc ** 2).sum())
    T = num / den
    p = float(stats.chi2.sf(T, 1))
    # independent cross-check: correlation of SCALED Schoenfeld residuals with rank time
    s_scaled = beta + s * d / V
    r, p_r = stats.pearsonr(s_scaled, g)
    return dict(cohort=label, beta=float(beta), n_events=int(d), test_stat=float(T),
                df=1, p_value=p,
                method="Grambsch-Therneau scaled Schoenfeld, rank(time) transform, "
                       "chi2 1 df (own implementation; lifelines not installed)",
                ph_violated_05=bool(p < 0.05),
                corr_scaled_rank=float(r), p_corr=float(p_r))


# =============================================================================== main
def main():
    gene_sets = json.load(open("results/gene_sets.json"))
    store = nc.load_all(OS6)

    clin_store, clin_avail = {}, {}
    for coh in OS6:
        _, _, _, s = store[coh]
        clin_store[coh] = harmonise_clinical(s, coh)
        clin_avail[coh] = clinical_availability(clin_store[coh])
    NOTES["clinical_availability"] = clin_avail
    print("clinical availability:", json.dumps(clin_avail), flush=True)

    arms = list(gene_sets.keys()) + ["Clinical"]
    rows, auc_rows, tau_notes, avail_notes = [], [], {}, {}
    loco_risk = {}

    for held in OS6:
        train = [c for c in OS6 if c != held]
        Xte_all, tte, ete, _ = store[held]
        ytr_parts = [(store[c][1], store[c][2]) for c in train]
        ttr = np.concatenate([a for a, _ in ytr_parts])
        etr = np.concatenate([b for _, b in ytr_parts])
        y_tr, y_te = surv_y(ttr, etr), surv_y(tte, ete)

        # --- tau: largest event time in held-out cohort, capped by train censoring support
        # cap uses a 0.05 floor on the censoring-KM survival estimate, not 1e-8: IPCW
        # weights are 1/G(t), so a near-zero-but-nonzero G(t) still yields exploding
        # per-subject weights and an unstable integrated Brier score near that boundary.
        cens = CensoringDistributionEstimator().fit(y_tr)
        max_ev_te = float(tte[ete == 1].max())
        gx, gy = cens.unique_time_, cens.prob_
        pos = gx[gy > 0.05]
        cap = float(pos.max()) if len(pos) else float(ttr.max())
        tau_candidates = [min(max_ev_te, cap * (1 - 1e-9))]
        for q in (0.99, 0.95, 0.90):
            tau_candidates.append(float(np.quantile(tte[ete == 1], q)))
        tau_notes[held] = dict(max_event_time_test=max_ev_te,
                               train_censoring_support_max=cap,
                               candidates=tau_candidates)

        for arm in arms:
            if arm == "Clinical":
                use = [v for v in CLIN_VARS
                       if clin_avail[held][v] and sum(clin_avail[c][v] for c in train) >= 3]
                Xtr = np.vstack([clinical_matrix(clin_store[c], use, clin_avail[c])
                                 for c in train])
                Xte = clinical_matrix(clin_store[held], use, clin_avail[held])
                n_used, n_nom, feats = len(use), len(CLIN_VARS), ";".join(use)
            else:
                nominal = gene_sets[arm]["genes"]
                avail = [g for g in nominal
                         if g in Xte_all.columns and all(g in store[c][0].columns for c in train)]
                if len(avail) == 0:
                    continue
                Xtr = np.vstack([store[c][0][avail].values for c in train])
                Xte = Xte_all[avail].values
                n_used, n_nom, feats = len(avail), len(nominal), ";".join(avail)
                avail_notes.setdefault(arm, {})[held] = len(avail)

            beta = nc.fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)
            risk = Xte @ beta

            hc_sk = concordance_index_censored(ete.astype(bool), tte, risk)[0]
            hc_jit = nc.cindex(risk, tte, ete)

            # sksurv fit for the Breslow baseline (IBS only)
            ibs = np.nan
            c_agree = np.nan
            try:
                m = CoxPHSurvivalAnalysis(alpha=ALPHA, ties="breslow", n_iter=200)
                m.fit(Xtr, y_tr)
                r_sk = m.predict(Xte)
                c_agree = abs(concordance_index_censored(ete.astype(bool), tte, r_sk)[0] - hc_sk)
            except Exception as e:
                m = None
                NOTES.setdefault("sksurv_fit_errors", []).append(f"{held}/{arm}: {e!r}")

            # Uno's C with fold-specific tau
            uno, tau_used, uno_err = np.nan, np.nan, None
            for tc in tau_candidates:
                try:
                    uno = float(concordance_index_ipcw(y_tr, y_te, risk, tau=tc)[0])
                    tau_used = float(tc)
                    break
                except Exception as e:
                    uno_err = repr(e)
            if not np.isfinite(uno):
                NOTES.setdefault("uno_errors", []).append(f"{held}/{arm}: {uno_err}")

            # integrated Brier score up to tau
            if m is not None and np.isfinite(tau_used):
                try:
                    lo = float(np.quantile(tte[ete == 1], 0.05))
                    grid = np.linspace(max(lo, float(tte.min()) + 1e-6), tau_used * 0.999, 25)
                    grid = grid[(grid > tte.min()) & (grid < tte.max())]
                    sf = m.predict_survival_function(Xte)
                    P = np.asarray([[fn(g) for g in grid] for fn in sf])
                    ibs = float(integrated_brier_score(y_tr, y_te, P, grid))
                except Exception as e:
                    NOTES.setdefault("ibs_errors", []).append(f"{held}/{arm}: {e!r}")

            rows.append(dict(held_out_cohort=held, gene_set=arm,
                             n_genes_used=n_used, n_genes_nominal=n_nom,
                             harrell_c=float(hc_sk), uno_c=uno,
                             ibs=ibs, tau_months=tau_used,
                             n_test=int(len(tte)), events_test=int(ete.sum()),
                             n_train=int(len(ttr)), events_train=int(etr.sum()),
                             harrell_c_jit=float(hc_jit),
                             c_agreement_absdiff=c_agree,
                             features_used=feats))
            print("  %-16s %-20s nG=%3d Harrell=%.4f Uno=%.4f tau=%.1f IBS=%.4f"
                  % (held, arm, n_used, hc_sk, uno, tau_used, ibs), flush=True)

            # ---------------- Novel5: tAUC + keep out-of-fold risk
            if arm == "Novel5":
                z = (risk - risk.mean()) / (risk.std() if risk.std() > 1e-12 else 1.0)
                loco_risk[held] = pd.DataFrame(
                    dict(cohort=held, sample=Xte_all.index, risk=risk, risk_z=z,
                         time_months=tte, event=ete))
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

    met = pd.DataFrame(rows)
    met.to_csv("metrics_harrell_uno.csv", index=False)

    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv("time_dependent_auc_revised.csv", index=False)

    g = met.groupby("gene_set")
    summ = pd.DataFrame({
        "n_folds": g.size(),
        "harrell_mean": g["harrell_c"].mean(), "harrell_sd": g["harrell_c"].std(),
        "uno_mean": g["uno_c"].mean(), "uno_sd": g["uno_c"].std(),
        "ibs_mean": g["ibs"].mean(), "ibs_sd": g["ibs"].std(),
        "mean_n_genes_used": g["n_genes_used"].mean(),
    }).reset_index()
    summ["rank_harrell"] = summ["harrell_mean"].rank(ascending=False).astype(int)
    summ["rank_uno"] = summ["uno_mean"].rank(ascending=False).astype(int)
    summ["rank_delta"] = summ["rank_harrell"] - summ["rank_uno"]
    summ = summ.sort_values("rank_harrell")
    summ.to_csv("metrics_summary.csv", index=False)
    print(summ.to_string(index=False), flush=True)

    sp = stats.spearmanr(summ["harrell_mean"], summ["uno_mean"])
    NOTES["ranking"] = dict(
        changed=bool((summ["rank_delta"] != 0).any()),
        n_sets_moved=int((summ["rank_delta"] != 0).sum()),
        spearman_rho=float(sp.statistic), spearman_p=float(sp.pvalue),
        top_harrell=summ.sort_values("rank_harrell")["gene_set"].tolist(),
        top_uno=summ.sort_values("rank_uno")["gene_set"].tolist())

    # ------------------------------------------------------------------ PH tests
    risk_df = pd.concat(loco_risk.values(), ignore_index=True)
    risk_df.to_csv("loco_risk_novel5.csv", index=False)
    ph_rows = [ph_test(risk_df["risk_z"].values, risk_df["time_months"].values,
                       risk_df["event"].values, "POOLED")]
    for coh in OS6:
        d = loco_risk[coh]
        ph_rows.append(ph_test(d["risk_z"].values, d["time_months"].values,
                               d["event"].values, coh))
    ph = pd.DataFrame(ph_rows)
    ph["note"] = np.where(ph["p_value"] < 0.05,
                          "PH VIOLATED at 0.05 - a single HR summarises a time-varying effect",
                          "no evidence against PH at 0.05")
    ph.to_csv("ph_tests.csv", index=False)
    print(ph.to_string(index=False), flush=True)

    NOTES["tau_per_fold"] = tau_notes
    NOTES["genes_available_per_set_per_cohort"] = avail_notes
    NOTES["caveats"] = [
        "GSE20711 encodes age, size and node as 0/1 dichotomies, not continuous values; "
        "the Clinical arm therefore uses a coarser age variable in that cohort.",
        "GSE58812 is an all-TNBC series: er_pos has zero variance and grade/size/node are "
        "absent, so the Clinical arm there reduces to age alone.",
        "lifelines and statsmodels are not installed on the compute host; the PH test is "
        "an own implementation of the Grambsch-Therneau test, cross-checked against a "
        "Pearson correlation of scaled Schoenfeld residuals with rank-time.",
    ]
    json.dump(NOTES, open("run_notes.json", "w"), indent=1, default=str)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
