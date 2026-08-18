"""
validate_hr_local.py -- lifelines / statsmodels cross-check of the stratified pooled HR.

REVIEW ITEM 6 (supporting). The remote analysis environment has neither lifelines nor
statsmodels, so validate_hr.py could only cross-check the UNSTRATIFIED fit (against
scikit-survival, which has no strata option). The cohort-stratified pooled HR is the
pivotal number in the response to review item 6, so this script re-runs both fits
locally against two independent implementations of the stratified Cox model:

  lifelines.CoxPHFitter(strata=["cohort"])   -- stratified Efron/Breslow partial lik.
  statsmodels PHReg(strata=cohort, ties="breslow")

against our stratified Breslow partial likelihood (hr_pooled.cox_fit(..., strata=...)),
which maximises l(beta) = sum_c l_c(beta) with a separate baseline hazard per cohort.

Note on ties: our implementation and statsmodels use BRESLOW; lifelines uses EFRON and
is therefore expected to differ slightly (the cohorts contain tied event times), so the
lifelines comparison is reported with a looser tolerance and the statsmodels comparison
is the exact one. Both are reported verbatim.

Input : loco_risk_scores.csv (held-out LOCO risk scores + tertile group, from hr_pooled.py)
Output: validation_hr_local.json
"""
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from hr_pooled import cox_fit                                   # noqa: E402


def main(scores_csv, out_json):
    sc = pd.read_csv(scores_csv)
    hl = sc[sc.group != "mid"].reset_index(drop=True)
    z = (hl.group.values == "high").astype(float)
    t = hl.time_months.values.astype(float)
    e = hl.event.values.astype(np.int32)
    coh = hl.cohort.values

    ours_n = cox_fit(z, t, e)
    ours_s = cox_fit(z, t, e, strata=coh)
    out = {"n_patients": int(len(t)), "n_events": int(e.sum()),
           "ours_naive_hr": ours_n["hr"], "ours_naive_loghr": ours_n["beta"],
           "ours_naive_se": ours_n["se"],
           "ours_stratified_hr": ours_s["hr"], "ours_stratified_loghr": ours_s["beta"],
           "ours_stratified_se": ours_s["se"]}

    df = pd.DataFrame({"T": t, "E": e, "z": z, "cohort": coh})

    from lifelines import CoxPHFitter
    c1 = CoxPHFitter().fit(df[["T", "E", "z"]], "T", "E")
    c2 = CoxPHFitter().fit(df, "T", "E", strata=["cohort"])
    out["lifelines_naive"] = {
        "hr": float(np.exp(c1.params_["z"])), "loghr": float(c1.params_["z"]),
        "se": float(c1.standard_errors_["z"]), "ties": "efron",
        "abs_loghr_diff": float(abs(c1.params_["z"] - ours_n["beta"]))}
    out["lifelines_stratified"] = {
        "hr": float(np.exp(c2.params_["z"])), "loghr": float(c2.params_["z"]),
        "se": float(c2.standard_errors_["z"]), "ties": "efron",
        "abs_loghr_diff": float(abs(c2.params_["z"] - ours_s["beta"]))}

    from statsmodels.duration.hazard_regression import PHReg
    m1 = PHReg(t, z[:, None], status=e, ties="breslow").fit()
    m2 = PHReg(t, z[:, None], status=e, strata=coh, ties="breslow").fit()
    out["statsmodels_naive"] = {
        "hr": float(np.exp(m1.params[0])), "loghr": float(m1.params[0]),
        "se": float(m1.bse[0]), "ties": "breslow",
        "abs_loghr_diff": float(abs(m1.params[0] - ours_n["beta"])),
        "abs_se_diff": float(abs(m1.bse[0] - ours_n["se"]))}
    out["statsmodels_stratified"] = {
        "hr": float(np.exp(m2.params[0])), "loghr": float(m2.params[0]),
        "se": float(m2.bse[0]), "ties": "breslow",
        "abs_loghr_diff": float(abs(m2.params[0] - ours_s["beta"])),
        "abs_se_diff": float(abs(m2.bse[0] - ours_s["se"]))}

    out["breslow_exact_match"] = bool(
        out["statsmodels_naive"]["abs_loghr_diff"] < 1e-6
        and out["statsmodels_stratified"]["abs_loghr_diff"] < 1e-6
        and out["statsmodels_stratified"]["abs_se_diff"] < 1e-6)
    out["efron_close_match"] = bool(
        out["lifelines_stratified"]["abs_loghr_diff"] < 0.02)
    out["all_pass"] = bool(out["breslow_exact_match"] and out["efron_close_match"])
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
