import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats

try:
    from nested_core import load_all, fit_ridge_cox, cindex, NOVEL5, OS6
    _NESTED_CORE_ERR = None
except ImportError as _exc:
    _NESTED_CORE_ERR = _exc
    NOVEL5 = ["FLT3", "CLIC6", "SUSD3", "ZIC2", "P4HA2"]
    OS6 = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203",
           "GSE20711", "GSE58812"]

import shutil
shutil.copy("nested_core.py", "nested_core.py")

from nested_core import load_all, fit_ridge_cox, cindex, NOVEL5, OS6

ALPHA = 100.0


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


store = load_all(OS6)
panel = [g for g in NOVEL5 if all(g in store[c][0].columns for c in OS6)]

score_rows = []
for held in OS6:
    train = [c for c in OS6 if c != held]
    Xtr = np.vstack([store[c][0][panel].values for c in train])
    ttr = np.concatenate([store[c][1] for c in train])
    etr = np.concatenate([store[c][2] for c in train])
    beta = fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)

    Xte, tte, ete, _ = store[held]
    risk = Xte[panel].values @ beta

    q1, q2 = np.quantile(risk, [1.0 / 3.0, 2.0 / 3.0])
    grp = np.where(risk > q2, "high", np.where(risk <= q1, "low", "mid"))

    for i in range(len(risk)):
        score_rows.append({"cohort": held, "sample": str(Xte.index[i]),
                           "risk": float(risk[i]), "group": grp[i],
                           "time_months": float(tte[i]), "event": int(ete[i])})

pd.DataFrame(score_rows).to_csv("loco_risk_scores.csv", index=False)