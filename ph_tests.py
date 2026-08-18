import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, '.')

# Copy nested_core.py to working directory
import shutil
shutil.copy('{{artifact:c6bba115-2c0c-45f1-9e72-ccfdb928df0c}}', 'nested_core.py')

import nested_core as nc

warnings.filterwarnings("ignore")

ALPHA = 100.0
OS6 = nc.OS6


def surv_y(t, ev):
    return np.array([(bool(e), float(x)) for e, x in zip(ev, t)],
                    dtype=[("event", bool), ("time", float)])


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


# Load gene_sets to get arms list (needed to identify Novel5)
gene_sets = json.load(open("gene_sets.json"))
store = nc.load_all(OS6)

loco_risk = {}

for held in OS6:
    train = [c for c in OS6 if c != held]
    Xte_all, tte, ete, _ = store[held]
    ttr = np.concatenate([store[c][1] for c in train])
    etr = np.concatenate([store[c][2] for c in train])

    nominal = gene_sets["Novel5"]["genes"]
    avail = [g for g in nominal
             if g in Xte_all.columns and all(g in store[c][0].columns for c in train)]
    Xtr = np.vstack([store[c][0][avail].values for c in train])
    Xte = Xte_all[avail].values

    beta = nc.fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)
    risk = Xte @ beta

    z = (risk - risk.mean()) / (risk.std() if risk.std() > 1e-12 else 1.0)
    loco_risk[held] = pd.DataFrame(
        dict(cohort=held, sample=Xte_all.index, risk=risk, risk_z=z,
             time_months=tte, event=ete))

risk_df = pd.concat(loco_risk.values(), ignore_index=True)

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

os.makedirs("setup", exist_ok=True)
ph.to_csv(os.path.join("setup", "ph_tests.csv"), index=False)