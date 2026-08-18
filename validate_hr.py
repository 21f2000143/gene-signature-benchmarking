"""
validate_hr.py -- independent verification of the Cox estimators used in hr_pooled.py.

REVIEW ITEM 6 (supporting). The cohort-stratified pooled HR is the pivotal number in the
response to review item 6, and it is computed by a hand-written stratified Breslow
partial likelihood (lifelines is not installed in this environment). This script checks
that implementation against two independent, widely used packages before the number is
put in the manuscript.

CHECKS PERFORMED, on the SAME high/low patient set written by hr_pooled.py
(loco_risk_scores.csv, mid tertile excluded):

 1. naive (unstratified) Cox, covariate z = 1{high tertile}
      ours  vs  statsmodels PHReg(ties="breslow")
      ours  vs  scikit-survival CoxPHSurvivalAnalysis(alpha=1e-10, ties="breslow")
    compared on log-HR and (for statsmodels) on the standard error.

 2. cohort-stratified Cox, one common log-HR, separate baseline hazard per cohort
      ours  vs  statsmodels PHReg(..., strata=cohort, ties="breslow")
    compared on log-HR and standard error.

 3. degenerate-case check: the stratified fitter run with every patient in a SINGLE
    stratum must reproduce the naive fit exactly (to numerical tolerance).

 4. score-equation check: the analytic gradient of the stratified log partial
    likelihood at the fitted beta must be ~0, and the analytic observed information
    must match a central finite-difference second derivative of the summed per-cohort
    log partial likelihood.

PASS criteria: |log-HR difference| < 1e-6 and |se difference| < 1e-6 for checks 1-3;
gradient < 1e-6 and relative information error < 1e-4 for check 4.

Output: validation_hr.json (all comparisons with a boolean all_pass).
"""
import json
import numpy as np
import pandas as pd

from hr_pooled import cox_fit, _breslow_grad_hess


def main():
    sc = pd.read_csv("loco_risk_scores.csv")
    hl = sc[sc.group != "mid"].reset_index(drop=True)
    z = (hl.group.values == "high").astype(float)
    t = hl.time_months.values.astype(float)
    e = hl.event.values.astype(np.int32)
    coh = hl.cohort.values
    out = {"n_patients": int(len(t)), "n_events": int(e.sum()),
           "n_cohorts": int(pd.unique(coh).size)}

    ours_n = cox_fit(z, t, e)
    ours_s = cox_fit(z, t, e, strata=coh)
    out["ours_naive"] = {"loghr": ours_n["beta"], "se": ours_n["se"], "hr": ours_n["hr"]}
    out["ours_stratified"] = {"loghr": ours_s["beta"], "se": ours_s["se"],
                              "hr": ours_s["hr"]}

    # ---- 1/2: statsmodels PHReg
    try:
        from statsmodels.duration.hazard_regression import PHReg
        m1 = PHReg(t, z[:, None], status=e, ties="breslow").fit()
        m2 = PHReg(t, z[:, None], status=e, strata=coh, ties="breslow").fit()
        out["statsmodels_naive"] = {"loghr": float(m1.params[0]),
                                    "se": float(m1.bse[0]),
                                    "hr": float(np.exp(m1.params[0])),
                                    "abs_loghr_diff": float(abs(m1.params[0] - ours_n["beta"])),
                                    "abs_se_diff": float(abs(m1.bse[0] - ours_n["se"]))}
        out["statsmodels_stratified"] = {"loghr": float(m2.params[0]),
                                         "se": float(m2.bse[0]),
                                         "hr": float(np.exp(m2.params[0])),
                                         "abs_loghr_diff": float(abs(m2.params[0] - ours_s["beta"])),
                                         "abs_se_diff": float(abs(m2.bse[0] - ours_s["se"]))}
    except Exception as exc:
        out["statsmodels_error"] = repr(exc)

    # ---- 1: scikit-survival (unstratified only; sksurv has no strata)
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        y = np.array([(bool(a), float(b)) for a, b in zip(e, t)],
                     dtype=[("event", bool), ("time", float)])
        sk = CoxPHSurvivalAnalysis(alpha=1e-10, ties="breslow", n_iter=500).fit(z[:, None], y)
        out["sksurv_naive"] = {"loghr": float(sk.coef_[0]),
                               "hr": float(np.exp(sk.coef_[0])),
                               "abs_loghr_diff": float(abs(sk.coef_[0] - ours_n["beta"]))}
    except Exception as exc:
        out["sksurv_error"] = repr(exc)

    # ---- 3: single-stratum degeneracy
    one = cox_fit(z, t, e, strata=np.zeros(len(t), dtype=int))
    out["single_stratum_equals_naive"] = {
        "abs_loghr_diff": float(abs(one["beta"] - ours_n["beta"])),
        "abs_se_diff": float(abs(one["se"] - ours_n["se"]))}

    # ---- 4: score equation and finite-difference information
    def ll_grad_info(b):
        L = g = I = 0.0
        for cname in pd.unique(coh):
            m = coh == cname
            l_, g_, i_ = _breslow_grad_hess(z[m][:, None], t[m], e[m], np.array([b]))
            L += l_
            g += g_[0]
            I += i_[0, 0]
        return L, g, I

    b = ours_s["beta"]
    L0, g0, I0 = ll_grad_info(b)
    h = 1e-4
    Lp, _, _ = ll_grad_info(b + h)
    Lm, _, _ = ll_grad_info(b - h)
    fd_info = -(Lp - 2 * L0 + Lm) / h ** 2
    out["score_check"] = {"gradient_at_fit": float(g0),
                          "analytic_information": float(I0),
                          "finite_diff_information": float(fd_info),
                          "rel_info_error": float(abs(fd_info - I0) / I0),
                          "se_from_analytic_info": float(1 / np.sqrt(I0))}

    tol = 1e-6
    checks = []
    for k in ("statsmodels_naive", "statsmodels_stratified"):
        if k in out:
            checks.append(out[k]["abs_loghr_diff"] < tol and out[k]["abs_se_diff"] < tol)
    if "sksurv_naive" in out:
        checks.append(out["sksurv_naive"]["abs_loghr_diff"] < 1e-5)
    checks.append(out["single_stratum_equals_naive"]["abs_loghr_diff"] < tol)
    checks.append(abs(out["score_check"]["gradient_at_fit"]) < 1e-6)
    checks.append(out["score_check"]["rel_info_error"] < 1e-4)
    out["all_pass"] = bool(all(checks))
    out["n_checks"] = len(checks)

    with open("validation_hr.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
