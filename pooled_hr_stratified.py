"""
pooled_hr_stratified.py
=======================
Review item: the manuscript reports a single pooled risk-group hazard ratio computed by
concatenating all cohorts and ignoring cohort membership. That estimate is (i) confounded
by between-cohort differences in baseline hazard and censoring, and (ii) dominated by the
cohort contributing most events. This script recomputes the HR with COHORT AS A STRATUM
and quantifies the dominance explicitly.

EXACT DEFINITIONS
-----------------
Input: out-of-fold (LOCO) Novel5 risk scores, one per subject, from
metrics_uno_auc_ph.py (loco_risk_novel5.csv). Using out-of-fold scores means no
subject's own outcome contributed to the model that scored it.

Risk group: within EACH cohort separately, subjects are split at that cohort's median
out-of-fold risk into high (1) vs low (0). A within-cohort split is required because the
risk score's location and scale differ between cohorts (different platforms, different
training folds); a pooled global cut would confound risk with cohort.

Model A (unstratified pooled, = the manuscript's specification):
    Cox PH, Breslow ties, single covariate high-risk, all six OS cohorts concatenated,
    one common baseline hazard.
Model B (stratified, = the recommended specification):
    Cox PH, Breslow ties, single covariate high-risk, with a SEPARATE baseline hazard
    per cohort. The partial log-likelihood is the sum of per-cohort partial
    log-likelihoods; risk sets never cross cohort boundaries.
Both are fitted by Newton-Raphson to convergence (|step| < 1e-9); the variance is the
inverse observed information, and the CI is the Wald interval exp(beta +/- 1.96 SE).
Model C: the same stratified model additionally adjusted for the within-cohort z-scored
    continuous risk score, reported as HR per 1 SD, so the result does not depend on the
    median dichotomisation.
Per-cohort HRs (unstratified, one cohort at a time) are also reported.

Dominance: for each cohort, share of the total event count and share of the total
sample size; and a leave-one-cohort-out refit of Model B to show how much the pooled
stratified HR moves when each cohort is dropped.

OUTPUTS
  pooled_hr_stratified.csv   one row per model / per-cohort / LOCO-drop estimate
  pooled_hr_summary.json     headline stratified vs unstratified HR, dominance table
"""
import json

import numpy as np
import pandas as pd
from scipy import stats


def cox_strat(x, t, ev, strata, max_iter=200, tol=1e-10):
    """Cox PH, Breslow ties, one covariate, separate baseline hazard per stratum.

    Returns (beta, information V, loglik, n_events).
    Pass a constant `strata` for the unstratified model.
    """
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    ev = np.asarray(ev, int)
    strata = np.asarray(strata)
    groups = []
    for s in pd.unique(strata):
        m = strata == s
        o = np.argsort(-t[m], kind="mergesort")     # descending time
        groups.append((x[m][o], t[m][o], ev[m][o]))
    beta = 0.0
    ll = np.nan
    V = np.nan
    for _ in range(max_iter):
        grad = 0.0
        hess = 0.0
        ll = 0.0
        for xg, tg, eg in groups:
            w = np.exp(beta * xg)
            s0 = s1 = s2 = 0.0
            i, n = 0, len(xg)
            while i < n:
                j = i
                tt = tg[i]
                while j < n and tg[j] == tt:
                    s0 += w[j]; s1 += w[j] * xg[j]; s2 += w[j] * xg[j] * xg[j]
                    j += 1
                d = int(eg[i:j].sum())
                if d > 0:
                    sx = xg[i:j][eg[i:j] == 1].sum()
                    ll += beta * sx - d * np.log(s0)
                    grad += sx - d * (s1 / s0)
                    hess -= d * (s2 / s0 - (s1 / s0) ** 2)
                i = j
        V = -hess
        if V <= 0:
            break
        step = grad / V
        beta += step
        if abs(step) < tol:
            break
    return beta, V, ll, int(np.sum(ev))


def row(label, model, x, t, ev, strata, unit):
    b, V, ll, d = cox_strat(x, t, ev, strata)
    se = 1.0 / np.sqrt(V) if V > 0 else np.nan
    z = b / se if np.isfinite(se) else np.nan
    return dict(label=label, model=model, unit=unit,
                n=len(t), events=int(np.sum(ev)),
                beta=float(b), se=float(se),
                hr=float(np.exp(b)),
                hr_lo=float(np.exp(b - 1.96 * se)), hr_hi=float(np.exp(b + 1.96 * se)),
                z=float(z), p_value=float(2 * stats.norm.sf(abs(z))),
                loglik=float(ll))


def main():
    d = pd.read_csv("results/loco_risk_novel5.csv")
    d["high"] = 0.0
    for coh, g in d.groupby("cohort"):
        med = g["risk"].median()
        d.loc[g.index, "high"] = (g["risk"] > med).astype(float)
        d.loc[g.index, "risk_z"] = (g["risk"] - g["risk"].mean()) / g["risk"].std()

    rows = []
    rows.append(row("POOLED (unstratified, manuscript specification)", "unstratified",
                    d["high"], d["time_months"], d["event"],
                    np.zeros(len(d)), "high vs low (within-cohort median)"))
    rows.append(row("POOLED (cohort-stratified, recommended)", "stratified_by_cohort",
                    d["high"], d["time_months"], d["event"],
                    d["cohort"].values, "high vs low (within-cohort median)"))
    rows.append(row("POOLED (cohort-stratified, continuous)", "stratified_by_cohort",
                    d["risk_z"], d["time_months"], d["event"],
                    d["cohort"].values, "per 1 SD of within-cohort risk score"))

    for coh, g in d.groupby("cohort"):
        rows.append(row(coh, "single_cohort", g["high"], g["time_months"], g["event"],
                        np.zeros(len(g)), "high vs low (within-cohort median)"))

    for coh in d["cohort"].unique():
        g = d[d["cohort"] != coh]
        rows.append(row("drop " + coh, "stratified_leave_one_cohort_out",
                        g["high"], g["time_months"], g["event"], g["cohort"].values,
                        "high vs low (within-cohort median)"))

    out = pd.DataFrame(rows)
    out.to_csv("pooled_hr_stratified.csv", index=False)
    print(out.to_string(index=False), flush=True)

    tot_ev = int(d["event"].sum())
    dom = (d.groupby("cohort")
             .agg(n=("event", "size"), events=("event", "sum"))
             .assign(event_share=lambda x: x["events"] / tot_ev,
                     sample_share=lambda x: x["n"] / len(d))
             .sort_values("events", ascending=False))
    print(dom.to_string(), flush=True)

    uns = out.iloc[0]
    stt = out.iloc[1]
    summary = dict(
        total_n=int(len(d)), total_events=tot_ev,
        unstratified_hr=uns["hr"], unstratified_ci=[uns["hr_lo"], uns["hr_hi"]],
        unstratified_p=uns["p_value"],
        stratified_hr=stt["hr"], stratified_ci=[stt["hr_lo"], stt["hr_hi"]],
        stratified_p=stt["p_value"],
        stratified_continuous_hr_per_sd=out.iloc[2]["hr"],
        stratified_continuous_ci=[out.iloc[2]["hr_lo"], out.iloc[2]["hr_hi"]],
        pct_change_hr_stratifying=float(100 * (stt["hr"] - uns["hr"]) / uns["hr"]),
        dominance=dom.reset_index().to_dict("records"),
        top_cohort_by_events=dom.index[0],
        top_cohort_event_share=float(dom["event_share"].iloc[0]),
        loco_drop_hr_range=[float(out[out["model"] == "stratified_leave_one_cohort_out"]["hr"].min()),
                            float(out[out["model"] == "stratified_leave_one_cohort_out"]["hr"].max())],
        pooling_specification=(
            "Cox proportional hazards, Breslow ties, cohort entered as a stratum "
            "(separate baseline hazard per cohort, risk sets confined within cohort); "
            "exposure = high vs low out-of-fold Novel5 LOCO risk split at the "
            "within-cohort median; Wald 95% CI from the inverse observed information."),
    )
    json.dump(summary, open("results/pooled_hr_summary.json", "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
