"""
hr_pooled.py -- pooled high-vs-low tertile hazard ratio for NOVEL5, done properly.

REVIEW ITEM 6. The manuscript reports a pooled high-vs-low tertile HR of 2.40 obtained
by pooling patients from all six OS cohorts into ONE unstratified Cox model. METABRIC
supplies 1143/2099 events yet has the LOWEST cohort-specific HR (1.746), so the pooled
estimate is (i) dominated by one cohort and (ii) inflated by between-cohort
heterogeneity of the baseline hazard (patients in a high-baseline-risk cohort are
compared against patients in a low-baseline-risk cohort, which an unstratified model
reads as covariate effect).

This script recomputes the risk groups from HELD-OUT LOCO risk scores and reports three
pooled estimators plus formal heterogeneity statistics.

--------------------------------------------------------------------------------
EXACT DEFINITIONS USED
--------------------------------------------------------------------------------
Panel            : NOVEL5 = FLT3, CLIC6, SUSD3, ZIC2, P4HA2.
Learner          : ridge Cox, alpha = 100 (pre-specified), Breslow ties.
Expression       : every gene z-scored WITHIN each cohort (nested_core.load_cohort);
                   required because SCANB_GSE96058 is not supplied z-scored.
Risk score       : leave-one-cohort-out. For held-out cohort h, beta is fit on the
                   pooled patients of the other five OS cohorts and the risk score of
                   every patient in h is x_i' beta. No patient contributes to the model
                   that scores it.
Risk groups      : tertiles of the held-out risk score formed WITHIN each held-out
                   cohort (cut points = the 33.33rd and 66.67th percentiles of that
                   cohort's own held-out scores), matching the paper's procedure.
                   "High" = top tertile, "Low" = bottom tertile; the middle tertile is
                   discarded for the high-vs-low contrast.
Cohort HR        : Cox proportional hazards regression of overall survival on a single
                   binary covariate z = 1{high tertile}, fit separately within each
                   cohort by Newton-Raphson on the Breslow partial likelihood.
                   HR = exp(beta); 95% CI = exp(beta +/- 1.96*se), se from the inverse
                   observed information (standard Wald CI on the log scale).

POOLED ESTIMATORS (all on the same high/low patients):
 1. naive_pooled       -- one Cox model on all pooled high/low patients with the single
                          covariate z and a SINGLE common baseline hazard. This is what
                          the manuscript currently reports. Risk sets mix cohorts.
 2. cohort_stratified  -- one Cox model, single common log-HR beta, but a SEPARATE
                          baseline hazard per cohort. Implemented as the stratified
                          Breslow partial likelihood
                              l(beta) = sum_{cohorts c} l_c(beta),
                          where l_c is the ordinary Breslow log partial likelihood
                          computed using only cohort-c risk sets. Gradient and observed
                          information are the sums of the per-cohort gradients and
                          informations; beta is obtained by Newton-Raphson on that sum
                          and se from the inverse of the summed information. Patients
                          are therefore only ever compared with patients from their own
                          cohort. Cross-checked against lifelines
                          CoxPHFitter(strata="cohort") when lifelines is importable.
 3. random_effects     -- DerSimonian-Laird random-effects meta-analysis of the six
                          cohort-specific log-HRs y_i with within-cohort variances v_i:
                            w_i    = 1/v_i
                            Q      = sum w_i (y_i - ybar_FE)^2 ,  df = k-1
                            tau^2  = max(0, (Q - df) / (sum w - sum w^2 / sum w))
                            w*_i   = 1/(v_i + tau^2)
                            pooled = sum w*_i y_i / sum w*_i ,  se = sqrt(1/sum w*_i)
                            I^2    = max(0, (Q - df)/Q) * 100
                          Q is referred to chi-square on k-1 df for its p-value.
                          The fixed-effect (inverse-variance) pooled estimate is also
                          computed and reported in heterogeneity.json for reference.

PART (b): leave-one-cohort-out sensitivity of the random-effects pooled HR -- the DL
meta-analysis is refit six times, each time omitting one cohort, to quantify how much
of the pooled figure any single cohort (in particular METABRIC) carries.

PART (c): the six-cohort resolution floor -- see resolution_floor() below; the minimum
attainable two-sided p-value is enumerated EXACTLY from the null distributions of the
sign test and the Wilcoxon signed-rank test at n = 6 pairs.

OUTPUTS: hr_per_cohort.csv, hr_pooled_methods.csv, heterogeneity.json,
         hr_loco_sensitivity.csv, resolution_floor.json, loco_risk_scores.csv
"""
import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats

# nested_core pulls in numba (needed for the ridge-Cox/c-index kernels used by main()).
# The Cox/meta-analysis functions below are pure numpy, so allow this module to be
# imported for validation on machines without numba; main() will fail loudly if the
# import did not succeed.
try:
    from nested_core import load_all, fit_ridge_cox, cindex, NOVEL5, OS6
    _NESTED_CORE_ERR = None
except ImportError as _exc:                                      # pragma: no cover
    _NESTED_CORE_ERR = _exc
    NOVEL5 = ["FLT3", "CLIC6", "SUSD3", "ZIC2", "P4HA2"]
    OS6 = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203",
           "GSE20711", "GSE58812"]

ALPHA = 100.0


# ------------------------------------------------------------------ Cox with se
def _breslow_grad_hess(X, t, ev, beta):
    """Log partial likelihood, gradient and observed information for ONE stratum.

    Breslow handling of ties. Returns (ll, grad, info) where info = -Hessian.
    """
    o = np.argsort(-t, kind="mergesort")
    X, t, ev = X[o], t[o], ev[o]
    n, p = X.shape
    eta = X @ beta
    m = eta.max() if n else 0.0
    w = np.exp(eta - m)          # scale-invariant for the partial likelihood ratios
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
    """Unpenalised Cox by Newton-Raphson; strata=None gives the ordinary model.

    With `strata` (array of cohort labels) this maximises the STRATIFIED partial
    likelihood sum_c l_c(beta): one common beta, a separate baseline hazard per
    stratum. Returns dict with beta, se, HR, CI, z, p, n, events, iterations.
    """
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


# ------------------------------------------------------- DerSimonian-Laird
def dersimonian_laird(y, v):
    """Random-effects meta-analysis of log-HRs y with variances v."""
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    k = len(y)
    w = 1.0 / v
    fe = float((w * y).sum() / w.sum())
    fe_se = float(np.sqrt(1.0 / w.sum()))
    Q = float((w * (y - fe) ** 2).sum())
    df = k - 1
    Qp = float(stats.chi2.sf(Q, df)) if df > 0 else float("nan")
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = float(max(0.0, (Q - df) / C)) if C > 0 else 0.0
    I2 = float(max(0.0, (Q - df) / Q) * 100.0) if Q > 0 else 0.0
    ws = 1.0 / (v + tau2)
    mu = float((ws * y).sum() / ws.sum())
    se = float(np.sqrt(1.0 / ws.sum()))
    return {"k": k, "loghr": mu, "se": se,
            "hr": float(np.exp(mu)),
            "ci_lo": float(np.exp(mu - 1.96 * se)),
            "ci_hi": float(np.exp(mu + 1.96 * se)),
            "p": float(2 * stats.norm.sf(abs(mu / se))),
            "Q": Q, "Q_df": df, "Q_p": Qp, "I2_percent": I2, "tau2": tau2,
            "tau": float(np.sqrt(tau2)),
            "fe_loghr": fe, "fe_hr": float(np.exp(fe)),
            "fe_ci_lo": float(np.exp(fe - 1.96 * fe_se)),
            "fe_ci_hi": float(np.exp(fe + 1.96 * fe_se)),
            "pred_int_lo": float(np.exp(mu - stats.t.ppf(0.975, max(df - 1, 1))
                                        * np.sqrt(se ** 2 + tau2))) if df > 1 else float("nan"),
            "pred_int_hi": float(np.exp(mu + stats.t.ppf(0.975, max(df - 1, 1))
                                        * np.sqrt(se ** 2 + tau2))) if df > 1 else float("nan")}


# --------------------------------------------------- (c) resolution floor at n=6
def resolution_floor(n=6):
    """EXACT minimum attainable two-sided p-value for cohort-level paired tests.

    Sign test (exact binomial, n non-tied pairs, H0: p=0.5): the most extreme outcome
    is all n differences of the same sign; two-sided exact p = 2 * (1/2)^n.

    Wilcoxon signed-rank (exact null, no ties, n pairs): the null distribution of W+ is
    generated by all 2^n sign assignments of ranks 1..n. The most extreme statistic
    (W+ = 0 or W+ = n(n+1)/2) has null probability 1/2^n in each tail, so the smallest
    attainable two-sided exact p = 2/2^n.

    Both are enumerated below rather than asserted.
    """
    # sign test: enumerate the exact binomial two-sided p for every possible count
    ps = []
    for k in range(n + 1):
        ps.append(float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue))
    min_sign = float(min(ps))

    # wilcoxon signed-rank: full enumeration of the exact null of W+
    ranks = np.arange(1, n + 1)
    dist = {}
    for signs in itertools.product([0, 1], repeat=n):
        wplus = int((ranks * np.array(signs)).sum())
        dist[wplus] = dist.get(wplus, 0) + 1
    total = 2 ** n
    maxw = n * (n + 1) // 2
    two_sided = {}
    for w in range(maxw + 1):
        lo = sum(c for k, c in dist.items() if k <= w) / total
        hi = sum(c for k, c in dist.items() if k >= w) / total
        two_sided[w] = min(1.0, 2 * min(lo, hi))
    min_wilcoxon = float(min(two_sided.values()))
    # sanity: scipy's exact wilcoxon on a maximally-extreme sample must agree
    x = np.arange(1.0, n + 1.0)
    try:
        scipy_w = float(stats.wilcoxon(x, np.zeros(n), method="exact").pvalue)
    except TypeError:
        scipy_w = float(stats.wilcoxon(x, np.zeros(n), mode="exact").pvalue)
    scipy_sign = float(stats.binomtest(n, n, 0.5, alternative="two-sided").pvalue)
    return {
        "n_pairs": n,
        "sign_test_min_two_sided_p": min_sign,
        "sign_test_min_p_exact_fraction": "2/2^%d = 2/%d" % (n, 2 ** n),
        "sign_test_scipy_check": scipy_sign,
        "wilcoxon_signed_rank_min_two_sided_p": min_wilcoxon,
        "wilcoxon_min_p_exact_fraction": "2/2^%d = 2/%d" % (n, 2 ** n),
        "wilcoxon_scipy_check": scipy_w,
        "wilcoxon_null_support_size": int(total),
        "wilcoxon_Wplus_null_counts": {str(k): int(v) for k, v in sorted(dist.items())},
        "sign_test_two_sided_p_by_count": {str(k): ps[k] for k in range(n + 1)},
        "wilcoxon_two_sided_p_by_Wplus": {str(k): v for k, v in sorted(two_sided.items())},
        "min_p_below_0.05": bool(min_sign < 0.05),
        "min_p_below_0.01": bool(min_sign < 0.01),
        "min_p_survives_bonferroni_2_tests": bool(min_sign < 0.05 / 2),
        "min_p_survives_bonferroni_5_tests": bool(min_sign < 0.05 / 5),
        "min_p_survives_bonferroni_9_tests": bool(min_sign < 0.05 / 9),
    }


# ------------------------------------------------------------------------- main
def main():
    if _NESTED_CORE_ERR is not None:
        raise RuntimeError("nested_core unavailable, cannot run the LOCO analysis: %r"
                           % (_NESTED_CORE_ERR,))
    store = load_all(OS6)
    panel = [g for g in NOVEL5 if all(g in store[c][0].columns for c in OS6)]
    missing = [g for g in NOVEL5 if g not in panel]
    print("panel used: %s   missing: %s" % (panel, missing), flush=True)

    rows, score_rows = [], []
    for held in OS6:
        train = [c for c in OS6 if c != held]
        Xtr = np.vstack([store[c][0][panel].values for c in train])
        ttr = np.concatenate([store[c][1] for c in train])
        etr = np.concatenate([store[c][2] for c in train])
        beta = fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)

        Xte, tte, ete, _ = store[held]
        risk = Xte[panel].values @ beta
        c_held = cindex(risk, tte, ete)

        q1, q2 = np.quantile(risk, [1.0 / 3.0, 2.0 / 3.0])
        grp = np.where(risk > q2, "high", np.where(risk <= q1, "low", "mid"))
        sel = grp != "mid"
        z = (grp[sel] == "high").astype(float)
        fit = cox_fit(z, tte[sel], ete[sel])

        for i in range(len(risk)):
            score_rows.append({"cohort": held, "sample": str(Xte.index[i]),
                               "risk": float(risk[i]), "group": grp[i],
                               "time_months": float(tte[i]), "event": int(ete[i])})
        rows.append({
            "cohort": held, "n_total": int(len(risk)),
            "events_total": int(ete.sum()), "cindex_heldout": float(c_held),
            "n_high": int((grp == "high").sum()), "n_low": int((grp == "low").sum()),
            "events_high": int(ete[grp == "high"].sum()),
            "events_low": int(ete[grp == "low"].sum()),
            "events_highlow": fit["events"], "n_highlow": fit["n"],
            "hr": fit["hr"], "ci_lo": fit["ci_lo"], "ci_hi": fit["ci_hi"],
            "log_hr": fit["beta"], "se_log_hr": fit["se"], "p": fit["p"],
            "cut_lo": float(q1), "cut_hi": float(q2)})
        print("%-18s n=%4d ev=%4d  C=%.3f  HR=%.3f (%.3f-%.3f) ev(H/L)=%d/%d"
              % (held, len(risk), int(ete.sum()), c_held, fit["hr"],
                 fit["ci_lo"], fit["ci_hi"], rows[-1]["events_high"],
                 rows[-1]["events_low"]), flush=True)

    per = pd.DataFrame(rows)
    per.to_csv("hr_per_cohort.csv", index=False)
    pd.DataFrame(score_rows).to_csv("loco_risk_scores.csv", index=False)

    # ---- pooled high/low patient set
    sc = pd.DataFrame(score_rows)
    hl = sc[sc.group != "mid"].reset_index(drop=True)
    zz = (hl.group.values == "high").astype(float)
    tt = hl.time_months.values.astype(float)
    ee = hl.event.values.astype(np.int32)

    naive = cox_fit(zz, tt, ee)
    strat = cox_fit(zz, tt, ee, strata=hl.cohort.values)
    dl = dersimonian_laird(per.log_hr.values, per.se_log_hr.values ** 2)

    # cross-check the stratified fit against lifelines if available
    lifelines_check = None
    try:
        from lifelines import CoxPHFitter
        df = pd.DataFrame({"T": tt, "E": ee, "z": zz, "cohort": hl.cohort.values})
        cph = CoxPHFitter()
        cph.fit(df, duration_col="T", event_col="E", strata=["cohort"])
        lifelines_check = {"hr": float(np.exp(cph.params_["z"])),
                           "se": float(cph.standard_errors_["z"]),
                           "abs_loghr_diff_vs_ours":
                               float(abs(cph.params_["z"] - strat["beta"]))}
        cph2 = CoxPHFitter()
        cph2.fit(df[["T", "E", "z"]], duration_col="T", event_col="E")
        lifelines_check["naive_hr"] = float(np.exp(cph2.params_["z"]))
        lifelines_check["abs_loghr_diff_naive"] = float(
            abs(cph2.params_["z"] - naive["beta"]))
        print("lifelines cross-check:", lifelines_check, flush=True)
    except Exception as exc:                                     # pragma: no cover
        lifelines_check = {"error": repr(exc)}
        print("lifelines unavailable:", exc, flush=True)

    meth = pd.DataFrame([
        {"method": "naive_pooled", "hr": naive["hr"], "ci_lo": naive["ci_lo"],
         "ci_hi": naive["ci_hi"], "log_hr": naive["beta"], "se_log_hr": naive["se"],
         "p": naive["p"], "n_patients": naive["n"], "n_events": naive["events"],
         "n_cohorts": len(OS6),
         "method_note": "single Cox model on pooled high/low patients, ONE common "
                        "baseline hazard, covariate z=1{high tertile}; risk sets mix "
                        "cohorts. This is the estimator the manuscript currently "
                        "reports (2.40)."},
        {"method": "cohort_stratified", "hr": strat["hr"], "ci_lo": strat["ci_lo"],
         "ci_hi": strat["ci_hi"], "log_hr": strat["beta"], "se_log_hr": strat["se"],
         "p": strat["p"], "n_patients": strat["n"], "n_events": strat["events"],
         "n_cohorts": len(OS6),
         "method_note": "single common log-HR from the stratified Breslow partial "
                        "likelihood l(b)=sum_c l_c(b), separate baseline hazard per "
                        "cohort, Newton-Raphson on the summed gradient/information; "
                        "Wald CI from the inverse summed information. Patients are "
                        "compared only within their own cohort."},
        {"method": "random_effects_DL", "hr": dl["hr"], "ci_lo": dl["ci_lo"],
         "ci_hi": dl["ci_hi"], "log_hr": dl["loghr"], "se_log_hr": dl["se"],
         "p": dl["p"], "n_patients": naive["n"], "n_events": naive["events"],
         "n_cohorts": dl["k"],
         "method_note": "DerSimonian-Laird random-effects meta-analysis of the six "
                        "cohort-specific log-HRs with inverse-variance weights "
                        "1/(v_i+tau^2); each cohort contributes as one unit rather "
                        "than in proportion to its event count."}])
    meth.to_csv("hr_pooled_methods.csv", index=False)

    het = dict(dl)
    het.update({
        "naive_pooled_hr": naive["hr"], "naive_pooled_ci":
            [naive["ci_lo"], naive["ci_hi"]],
        "cohort_stratified_hr": strat["hr"],
        "cohort_stratified_ci": [strat["ci_lo"], strat["ci_hi"]],
        "per_cohort_hr": {r["cohort"]: r["hr"] for r in rows},
        "per_cohort_events_highlow": {r["cohort"]: r["events_highlow"] for r in rows},
        "events_share_of_largest_cohort": float(
            per.events_highlow.max() / per.events_highlow.sum()),
        "largest_cohort_by_events": str(per.loc[per.events_highlow.idxmax(), "cohort"]),
        "lowest_hr_cohort": str(per.loc[per.hr.idxmin(), "cohort"]),
        "lowest_hr": float(per.hr.min()), "highest_hr": float(per.hr.max()),
        "highest_hr_cohort": str(per.loc[per.hr.idxmax(), "cohort"]),
        "shrink_naive_to_stratified_pct": float(
            100 * (naive["hr"] - strat["hr"]) / naive["hr"]),
        "shrink_naive_to_random_effects_pct": float(
            100 * (naive["hr"] - dl["hr"]) / naive["hr"]),
        "lifelines_cross_check": lifelines_check,
        "panel": panel, "alpha": ALPHA, "cohorts": OS6})
    with open("heterogeneity.json", "w") as fh:
        json.dump(het, fh, indent=2, default=float)

    # -------------------------------------------------- (b) LOCO sensitivity of DL
    sens = []
    full = dl
    for drop in OS6:
        sub = per[per.cohort != drop]
        d = dersimonian_laird(sub.log_hr.values, sub.se_log_hr.values ** 2)
        keep = hl.cohort.values != drop
        zk = (hl.group.values[keep] == "high").astype(float)
        tk = hl.time_months.values[keep].astype(float)
        ek = hl.event.values[keep].astype(np.int32)
        n2 = cox_fit(zk, tk, ek)
        s2 = cox_fit(zk, tk, ek, strata=hl.cohort.values[keep])
        sens.append({
            "cohort_dropped": drop, "k_remaining": d["k"],
            "re_hr": d["hr"], "re_ci_lo": d["ci_lo"], "re_ci_hi": d["ci_hi"],
            "re_p": d["p"], "I2_percent": d["I2_percent"], "tau2": d["tau2"],
            "Q": d["Q"], "Q_p": d["Q_p"],
            "naive_pooled_hr": n2["hr"], "stratified_hr": s2["hr"],
            "delta_re_hr_vs_full": d["hr"] - full["hr"],
            "pct_change_re_hr_vs_full": 100 * (d["hr"] - full["hr"]) / full["hr"],
            "events_removed": int(per.loc[per.cohort == drop, "events_highlow"].iloc[0])})
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv("hr_loco_sensitivity.csv", index=False)

    # -------------------------------------------------------- (c) resolution floor
    rf = resolution_floor(6)
    rf["context"] = ("Applies to any cohort-level paired test over the six OS cohorts "
                     "(e.g. NOVEL5 vs a comparator signature compared as six paired "
                     "per-cohort c-index differences).")
    rf["manuscript_sentence"] = (
        "With six cohorts, a paired cohort-level test has a hard resolution floor: the "
        "smallest two-sided p-value attainable is 2/2^6 = 0.03125 for both the exact "
        "sign test and the exact Wilcoxon signed-rank test, reached only when all six "
        "differences favour the same signature. No six-cohort comparison of this kind "
        "can therefore yield p < 0.03125, cannot reach p < 0.01, and does not survive "
        "Bonferroni correction for more than one such test; the cohort-level tests "
        "reported here should be read as descriptive consistency checks rather than as "
        "confirmatory evidence.")
    with open("resolution_floor.json", "w") as fh:
        json.dump(rf, fh, indent=2, default=float)

    print("\n--- pooled methods ---")
    print(meth[["method", "hr", "ci_lo", "ci_hi", "n_events"]].to_string(index=False))
    print("\nQ=%.3f (df=%d, p=%.4g)  I2=%.1f%%  tau2=%.4f"
          % (dl["Q"], dl["Q_df"], dl["Q_p"], dl["I2_percent"], dl["tau2"]))
    print("\n--- LOCO sensitivity (random-effects) ---")
    print(sens_df[["cohort_dropped", "re_hr", "re_ci_lo", "re_ci_hi",
                   "I2_percent", "events_removed"]].to_string(index=False))
    print("\nresolution floor n=6: sign %.6f  wilcoxon %.6f"
          % (rf["sign_test_min_two_sided_p"],
             rf["wilcoxon_signed_rank_min_two_sided_p"]))


if __name__ == "__main__":
    main()
