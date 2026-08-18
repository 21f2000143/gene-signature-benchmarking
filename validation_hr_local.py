import json
import sys
import numpy as np
import pandas as pd
import os
# Inline the necessary functions from hr_pooled.py

def _breslow_grad_hess(X, t, ev, beta):
    o = np.argsort(-t, kind="mergesort")
    X, t, ev = X[o], t[o], ev[o]
    n, p = X.shape
    eta = X @ beta
    m = eta.max() if n else 0.0
    w = np.exp(eta - m)
    s0 = 0.0
    s1 = np.zeros(p)
    s2 = np.zeros((p, p))
    ll = 0.0
    grad = np.zeros(p)
    info = np.zeros((p, p))
    i = 0
    while i < n:
        j = i
        tt = t[i]
        while j < n and t[j] == tt:
            s0 += w[j]
            s1 += w[j] * X[j]
            s2 += w[j] * np.outer(X[j], X[j])
            j += 1
        d = 0
        for k in range(i, j):
            if ev[k] == 1:
                d += 1
                ll += eta[k] - m
                grad += X[k]
        if d > 0:
            ll -= d * np.log(s0)
            mu = s1 / s0
            grad -= d * mu
            info += d * (s2 / s0 - np.outer(mu, mu))
        i = j
    return ll, grad, info


def cox_fit(X, t, ev, strata=None, max_iter=100, tol=1e-9):
    from scipy import stats
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    t = np.asarray(t, dtype=np.float64)
    ev = np.asarray(ev, dtype=np.int32)
    p = X.shape[1]
    if strata is None:
        groups = [np.ones(len(t), dtype=bool)]
    else:
        strata = np.asarray(strata)
        groups = [strata == s for s in pd.unique(strata)]
    beta = np.zeros(p)
    it = 0
    for it in range(1, max_iter + 1):
        grad = np.zeros(p)
        info = np.zeros((p, p))
        for g in groups:
            if g.sum() == 0 or ev[g].sum() == 0:
                continue
            _, gr, inf = _breslow_grad_hess(X[g], t[g], ev[g], beta)
            grad += gr
            info += inf
        step = np.linalg.solve(info + 1e-12 * np.eye(p), grad)
        mx = np.abs(step).max()
        if mx > 5.0:
            step *= 5.0 / mx
        beta = beta + step
        if mx < tol:
            break
    cov = np.linalg.inv(info + 1e-12 * np.eye(p))
    se = np.sqrt(np.diag(cov))
    z = beta / se
    return {"beta": float(beta[0]), "se": float(se[0]),
            "hr": float(np.exp(beta[0])),
            "ci_lo": float(np.exp(beta[0] - 1.96 * se[0])),
            "ci_hi": float(np.exp(beta[0] + 1.96 * se[0])),
            "z": float(z[0]), "p": float(2 * stats.norm.sf(abs(z[0]))),
            "n": int(len(t)), "events": int(ev.sum()), "iterations": it}


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
    os.makedirs("setup", exist_ok=True)
    scores_csv = sys.argv[1] if len(sys.argv) > 1 else "setup/sloco_risk_scores.csv"
    out_json = sys.argv[2] if len(sys.argv) > 2 else "setup/validation_hr_local.json"
    main(scores_csv, out_json)