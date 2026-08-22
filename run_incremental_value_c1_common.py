
"""
run_incremental_value_c1_common.py -- common-covariate-specification counterpart to
run_incremental_value_c1.py, added to close a gap flagged in peer review: the
original incremental-value test (LR test + honest cross-validated delta-c +
decision-curve analysis) was run only under the richer per-cohort-available
clinical specification, under which clinicopathology already beats the panel
outright (Table 2/4). The specification under which the panel actually leads
(0.661 vs 0.603, Table 4's common-covariate row) is the one under which a reader
most needs to know whether the panel adds value beyond the (weaker) clinical
baseline, and that test had not been run. This script is run_incremental_value_c1.py
verbatim except that the clinical covariate set is fixed to the two covariates
common to five of the six cohorts (node_pos, er_pos -- Table 4's
clinical_common_5_excl_GSE58812 specification) instead of being selected per
cohort from the richer six-covariate set. Same three cohorts, same design
(in-sample nested LRT, 5-fold cross-validated delta-c, decision-curve analysis
at 60 months), same alpha, same seeds.

Writes: incremental_lr_dca_c1_common.csv, incremental_dca_curves_c1_common.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nested_core as nc
from nested_core import fit_ridge_cox, cindex, sort_desc

NOVEL5 = ["FLT3", "CLIC6", "SUSD3", "ZIC2", "P4HA2"]
TARGET_COHORTS = ["METABRIC", "SCANB_GSE96058", "SCANB_GSE202203"]
HARM = ["node_pos", "er_pos"]  # common-covariate specification (Table 4)
ALPHA_NEAR_MLE = 1.0
HORIZON_MONTHS = 60.0
PT_GRID = np.round(np.arange(0.01, 0.51, 0.01), 4)


def log(m):
    print(m, flush=True)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def harmonise_clin(coh, s):
    out = pd.DataFrame(index=s.index)
    col = {c.lower(): s[c] for c in s.columns}

    out["age_years"] = np.nan
    if "age" in col:
        a = _num(col["age"])
        if a.notna().any() and not set(a.dropna().unique()) <= {0.0, 1.0}:
            out["age_years"] = a

    out["grade3"] = np.nan
    if "grade" in col:
        g = col["grade"].astype(str).str.strip().str.upper().str.replace("G", "", regex=False)
        gg = _num(g)
        out["grade3"] = gg.where(gg.isin([1, 2, 3]))

    out["node_pos"] = np.nan
    if "node" in col:
        raw = col["node"]
        rs = raw.astype(str).str.strip()
        num = _num(raw)
        if num.notna().sum() > 0.5 * len(s):
            out["node_pos"] = (num > 0).astype(float).where(num.notna())
        else:
            neg = rs.str.upper().str.match(r"^N0") | rs.isin(
                ["NodeNegative", "Node negative", "negative", "0"])
            pos = rs.isin(["NodePositive", "SubMicroMet", "1to3", "4toX"]) | \
                rs.str.upper().str.match(r"^N[1-3]")
            out["node_pos"] = pd.Series(np.where(pos, 1.0, np.where(neg, 0.0, np.nan)),
                                        index=s.index)

    out["size_gt20"] = np.nan
    if "size" in col:
        raw = col["size"]
        sv = raw.astype(str).str.upper().str.strip()
        if sv.str.contains("PT[0-9]", regex=True).any():
            pt = _num(sv.str.extract(r"PT([0-9])")[0])
            out["size_gt20"] = (pt >= 2).astype(float).where(pt.notna())
        else:
            num = _num(raw)
            if num.notna().any() and set(num.dropna().unique()) <= {0.0, 1.0}:
                out["size_gt20"] = np.nan
            elif num.notna().any() and np.nanmedian(num) < 15:
                out["size_gt20"] = (num * 10 > 20).astype(float).where(num.notna())
            else:
                out["size_gt20"] = (num > 20).astype(float).where(num.notna())

    for k, oc in (("er", "er_pos"), ("pr", "pr_pos")):
        out[oc] = np.nan
        if k in col:
            v = col[k].astype(str).str.upper().str.strip()
            pos = v.str.contains("POS") | v.isin(["1", "1.0", "TRUE", "YES"])
            neg = v.str.contains("NEG") | v.isin(["0", "0.0", "FALSE", "NO"])
            out[oc] = np.where(pos, 1.0, np.where(neg, 0.0, np.nan))
    return out


def usable(series, n):
    v = series.dropna()
    return bool(len(v) > 0.3 * n and v.nunique() >= 2)


def zscore_fit(v):
    v = np.asarray(v, float)
    m = np.nanmean(v)
    s = np.nanstd(v)
    s = s if s > 0 else 1.0
    return m, s


def zscore_apply(v, m, s):
    v = np.asarray(v, float)
    return (np.nan_to_num(v, nan=m) - m) / s


def partial_loglik(X, t, ev, beta):
    """Breslow-tie partial log-likelihood at a fixed beta. Data are sorted
    DESCENDING by time so the risk set (all subjects with time >= current time)
    is prefix-cumulative as we scan forward -- same convention as
    nested_core._cox_ridge_newton."""
    Xs, ts, evs = sort_desc(np.ascontiguousarray(X, float), np.asarray(t, float), np.asarray(ev, int))
    eta = Xs @ beta
    m = eta.max() if len(eta) else 0.0
    if m > 30.0:
        eta = eta - (m - 30.0)
    w = np.exp(eta)
    n = len(ts)
    ll = 0.0
    s0 = 0.0
    i = 0
    while i < n:
        j = i
        tt = ts[i]
        while j < n and ts[j] == tt:
            j += 1
        s0 += w[i:j].sum()  # cumulative risk-set size through this time block
        d = int(evs[i:j].sum())
        if d > 0:
            ll += eta[i:j][evs[i:j] == 1].sum()
            ll -= d * np.log(s0)
        i = j
    return float(ll)


def breslow_baseline_cumhaz(X, t, ev, beta, horizon):
    """Breslow estimator of the cumulative baseline hazard at `horizon`, given a
    fitted linear predictor (data need not be pre-sorted)."""
    t = np.asarray(t, float); ev = np.asarray(ev, int)
    o = np.argsort(t)  # ascending for cumulative hazard build-up
    ts = t[o]; evs = ev[o]
    eta = (X[o] @ beta)
    w = np.exp(np.clip(eta, -30, 30))
    H0 = 0.0
    times_sorted = ts
    # risk set at time ts[k] = all with time >= ts[k] (ascending order, so suffix sum)
    # precompute suffix sums of w
    suffix = np.cumsum(w[::-1])[::-1]  # suffix[k] = sum_{j>=k} w[j]
    k = 0
    n = len(ts)
    while k < n and ts[k] <= horizon:
        j = k
        tt = ts[k]
        while j < n and ts[j] == tt:
            j += 1
        d = int(evs[k:j].sum())
        if d > 0:
            H0 += d / suffix[k]
        k = j
    return float(H0)


def predict_risk_at_horizon(Xtr, ttr, etr, beta, Xte, horizon):
    H0 = breslow_baseline_cumhaz(Xtr, ttr, etr, beta, horizon)
    eta_te = Xte @ beta
    S = np.exp(-H0 * np.exp(np.clip(eta_te, -30, 30)))
    return 1.0 - S  # predicted probability of the event by `horizon`


def net_benefit(y_true, p_pred, pt_grid):
    n = len(y_true)
    nb = np.full(len(pt_grid), np.nan)
    for i, pt in enumerate(pt_grid):
        pos = p_pred >= pt
        tp = np.sum(pos & (y_true == 1))
        fp = np.sum(pos & (y_true == 0))
        nb[i] = tp / n - fp / n * (pt / (1 - pt))
    return nb


def stratified_kfold_idx(ev, k, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ev))
    folds = np.full(len(ev), -1)
    for lab in (0, 1):
        ii = idx[ev == lab]
        rng.shuffle(ii)
        parts = np.array_split(ii, k)
        for fi, part in enumerate(parts):
            folds[part] = fi
    return folds


def boot_ci_paired(diffs, seed=0, B=2000):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n < 2:
        return (np.nan, np.nan)
    bm = rng.choice(diffs, size=(B, n), replace=True).mean(axis=1)
    return (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))


store = nc.load_all(nc.OS6, verbose=True)
H, AVAIL = {}, {}
for coh in TARGET_COHORTS:
    s = nc.load_cohort(coh)[3]
    h = harmonise_clin(coh, s)
    H[coh] = h
    AVAIL[coh] = {cc: usable(h[cc], len(h)) for cc in HARM}
    log("%-18s usable clinical covariates: %s" % (coh, ",".join(cc for cc in HARM if AVAIL[coh][cc])))

rows = []
curve_rows = []
K = 5

for coh in TARGET_COHORTS:
    covs = [cc for cc in HARM if AVAIL[coh][cc]]
    h = H[coh]
    X_all, t_all, ev_all, _ = store[coh]
    Xg = X_all[NOVEL5].values
    t = np.asarray(t_all, float)
    ev = np.asarray(ev_all, int)
    n = len(t)

    # ---- z-score clinical + gene columns on the FULL cohort for the in-sample LR test
    Zc_full = []
    for cc in covs:
        m, s = zscore_fit(h[cc].values)
        Zc_full.append(zscore_apply(h[cc].values, m, s))
    Zc_full = np.column_stack(Zc_full) if Zc_full else np.zeros((n, 0))
    Zg_full = np.column_stack([zscore_apply(Xg[:, k_], *zscore_fit(Xg[:, k_])) for k_ in range(Xg.shape[1])])

    X_reduced_full = Zc_full
    X_full_full = np.column_stack([Zc_full, Zg_full])

    beta_reduced = fit_ridge_cox(X_reduced_full, t, ev, alpha=ALPHA_NEAR_MLE)
    beta_full = fit_ridge_cox(X_full_full, t, ev, alpha=ALPHA_NEAR_MLE)

    ll_reduced = partial_loglik(X_reduced_full, t, ev, beta_reduced)
    ll_full = partial_loglik(X_full_full, t, ev, beta_full)
    lr_stat = 2.0 * (ll_full - ll_reduced)
    df = Zg_full.shape[1]
    p_lr = float(stats.chi2.sf(lr_stat, df)) if lr_stat >= 0 else 1.0

    # ---- honest delta-c and DCA via 5-fold within-cohort stratified CV
    folds = stratified_kfold_idx(ev, K, seed=13)
    diffs = []
    c_reduced_folds, c_full_folds = [], []
    all_p_reduced, all_p_full, all_y, all_t_test = [], [], [], []
    for kf in range(K):
        te = folds == kf
        tr = ~te
        if ev[te].sum() < 3 or ev[tr].sum() < 5:
            continue
        # re-standardise on the TRAIN fold only (no leakage)
        Zc_tr_cols, Zc_te_cols = [], []
        for cc in covs:
            m, s = zscore_fit(h[cc].values[tr])
            Zc_tr_cols.append(zscore_apply(h[cc].values[tr], m, s))
            Zc_te_cols.append(zscore_apply(h[cc].values[te], m, s))
        Zc_tr = np.column_stack(Zc_tr_cols) if Zc_tr_cols else np.zeros((tr.sum(), 0))
        Zc_te = np.column_stack(Zc_te_cols) if Zc_te_cols else np.zeros((te.sum(), 0))

        Zg_tr_cols, Zg_te_cols = [], []
        for k_ in range(Xg.shape[1]):
            m, s = zscore_fit(Xg[tr, k_])
            Zg_tr_cols.append(zscore_apply(Xg[tr, k_], m, s))
            Zg_te_cols.append(zscore_apply(Xg[te, k_], m, s))
        Zg_tr = np.column_stack(Zg_tr_cols); Zg_te = np.column_stack(Zg_te_cols)

        Xr_tr, Xr_te = Zc_tr, Zc_te
        Xf_tr, Xf_te = np.column_stack([Zc_tr, Zg_tr]), np.column_stack([Zc_te, Zg_te])

        br = fit_ridge_cox(Xr_tr, t[tr], ev[tr], alpha=ALPHA_NEAR_MLE)
        bf = fit_ridge_cox(Xf_tr, t[tr], ev[tr], alpha=ALPHA_NEAR_MLE)

        risk_r_te = Xr_te @ br
        risk_f_te = Xf_te @ bf
        cr = float(cindex(risk_r_te, t[te], ev[te]))
        cf = float(cindex(risk_f_te, t[te], ev[te]))
        c_reduced_folds.append(cr); c_full_folds.append(cf)
        diffs.append(cf - cr)

        # DCA: predicted 5-year event probability via Breslow baseline on the SAME train fold
        p_r = predict_risk_at_horizon(Xr_tr, t[tr], ev[tr], br, Xr_te, HORIZON_MONTHS)
        p_f = predict_risk_at_horizon(Xf_tr, t[tr], ev[tr], bf, Xf_te, HORIZON_MONTHS)
        # label: observed event status by horizon among those with follow-up >= horizon
        # or an event before horizon; drop censored-before-horizon (standard DCA convention:
        # use those with event before horizon as cases, and those known event-free through
        # horizon as controls; censored-before-horizon subjects are excluded)
        y = np.where((t[te] <= HORIZON_MONTHS) & (ev[te] == 1), 1,
             np.where(t[te] > HORIZON_MONTHS, 0, -1))
        keep = y != -1
        all_p_reduced.append(p_r[keep]); all_p_full.append(p_f[keep])
        all_y.append(y[keep]); all_t_test.append(t[te][keep])

    diffs = np.array(diffs)
    lo, hi = boot_ci_paired(diffs, seed=13)
    mean_delta_c = float(diffs.mean()) if len(diffs) else np.nan

    y_all = np.concatenate(all_y) if all_y else np.array([])
    pr_all = np.concatenate(all_p_reduced) if all_p_reduced else np.array([])
    pf_all = np.concatenate(all_p_full) if all_p_full else np.array([])

    nb_reduced = net_benefit(y_all, pr_all, PT_GRID) if len(y_all) else np.full(len(PT_GRID), np.nan)
    nb_full = net_benefit(y_all, pf_all, PT_GRID) if len(y_all) else np.full(len(PT_GRID), np.nan)
    nb_all = np.array([y_all.mean() - (1 - y_all.mean()) * (pt / (1 - pt)) for pt in PT_GRID]) if len(y_all) else np.full(len(PT_GRID), np.nan)  # "treat all"
    nb_none = np.zeros(len(PT_GRID))

    for pt_, nr_, nf_, na_, nn_ in zip(PT_GRID, nb_reduced, nb_full, nb_all, nb_none):
        curve_rows.append(dict(cohort=coh, pt=pt_, nb_clinical=nr_, nb_clinical_plus_novel5=nf_,
                                nb_treat_all=na_, nb_treat_none=nn_))

    # summary net-benefit gain at two representative thresholds (5% and 10% predicted risk)
    def nb_at(pt_grid, nb_arr, pt_want):
        idx_ = int(np.argmin(np.abs(pt_grid - pt_want)))
        return float(nb_arr[idx_])

    rows.append(dict(
        cohort=coh, n=n, events=int(ev.sum()), n_clinical_covariates=len(covs),
        covariates="|".join(covs), n_genes_added=Zg_full.shape[1],
        ll_reduced=ll_reduced, ll_full=ll_full, lr_statistic=lr_stat, lr_df=df, lr_p_value=p_lr,
        cv_folds_used=len(diffs), c_clinical_cv_mean=float(np.mean(c_reduced_folds)) if c_reduced_folds else np.nan,
        c_clinical_plus_novel5_cv_mean=float(np.mean(c_full_folds)) if c_full_folds else np.nan,
        delta_c_mean=mean_delta_c, delta_c_ci_lo=lo, delta_c_ci_hi=hi,
        n_dca_subjects=int(len(y_all)), dca_events=int(y_all.sum()) if len(y_all) else 0,
        net_benefit_gain_at_pt05=nb_at(PT_GRID, nb_full, 0.05) - nb_at(PT_GRID, nb_reduced, 0.05),
        net_benefit_gain_at_pt10=nb_at(PT_GRID, nb_full, 0.10) - nb_at(PT_GRID, nb_reduced, 0.10),
        horizon_months=HORIZON_MONTHS,
    ))
    log("%-18s LR=%.3f df=%d p=%.4g | delta-c=%+.4f [%.4f, %.4f] | NB gain@5%%=%.4f @10%%=%.4f"
        % (coh, lr_stat, df, p_lr, mean_delta_c, lo, hi,
           rows[-1]["net_benefit_gain_at_pt05"], rows[-1]["net_benefit_gain_at_pt10"]))

out = pd.DataFrame(rows)
out.to_csv("incremental_lr_dca_c1_common.csv", index=False)
curves = pd.DataFrame(curve_rows)
curves.to_csv("incremental_dca_curves_c1_common.csv", index=False)
log("\nWrote incremental_lr_dca_c1_common.csv and incremental_dca_curves_c1_common.csv")
