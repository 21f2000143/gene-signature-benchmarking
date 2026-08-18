import os
import numpy as np
import pandas as pd
from numba import njit, prange

ANCHOR4 = ["EEF1A2", "IQGAP1", "IQGAP2", "FRG1"]
NOVEL5 = ["FLT3", "CLIC6", "SUSD3", "ZIC2", "P4HA2"]
OS6 = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203", "GSE20711", "GSE58812"]
SEC3 = ["GSE6532", "GSE11121", "GSE21653"]

DATA = os.path.expanduser("~/.claude-science-scratch/bc_bench")


def load_cohort(coh, data_dir=DATA):
    e = pd.read_parquet(os.path.join(data_dir, coh + "_expr.parquet"))
    s = pd.read_parquet(os.path.join(data_dir, coh + "_surv.parquet"))
    if "sample" in s.columns:
        s = s.set_index("sample")
        common = [i for i in e.index if i in s.index]
        e, s = e.loc[common], s.loc[common]
    keep = np.isfinite(s["time_months"].values) & np.isfinite(s["event"].values) \
        & (s["time_months"].values > 0)
    e, s = e.loc[keep], s.loc[keep]
    X = e.values.astype(np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
    X = (X - mu) / sd
    X[~np.isfinite(X)] = 0.0
    return (pd.DataFrame(X, index=e.index, columns=e.columns),
            s["time_months"].values.astype(np.float64),
            s["event"].values.astype(np.int32),
            s)


def load_all(cohorts=OS6, data_dir=DATA, verbose=True):
    out = {}
    for coh in cohorts:
        out[coh] = load_cohort(coh, data_dir)
        if verbose:
            X, t, ev, _ = out[coh]
            print("loaded %-18s n=%5d genes=%6d events=%5d"
                  % (coh, X.shape[0], X.shape[1], int(ev.sum())), flush=True)
    return out


def common_genes(store, cohorts):
    g = None
    for coh in cohorts:
        s = set(store[coh][0].columns)
        g = s if g is None else (g & s)
    return sorted(g)


@njit(cache=True, fastmath=True)
def _cox_ridge_newton(X, t_order_idx, event, alpha, max_iter, tol):
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        m = eta.max()
        if m > 30.0:
            eta = eta - (m - 30.0)
        w = np.exp(eta)

        s0 = 0.0
        s1 = np.zeros(p)
        s2 = np.zeros((p, p))
        grad = np.zeros(p)
        hess = np.zeros((p, p))
        ll = 0.0
        i = 0
        while i < n:
            j = i
            tt = t_order_idx[i]
            while j < n and t_order_idx[j] == tt:
                wj = w[j]
                s0 += wj
                for a in range(p):
                    xa = X[j, a]
                    s1[a] += wj * xa
                    for b in range(p):
                        s2[a, b] += wj * xa * X[j, b]
                j += 1
            d = 0
            for k in range(i, j):
                if event[k] == 1:
                    d += 1
                    ll += eta[k]
                    for a in range(p):
                        grad[a] += X[k, a]
            if d > 0:
                ll -= d * np.log(s0)
                for a in range(p):
                    ma = s1[a] / s0
                    grad[a] -= d * ma
                    for b in range(p):
                        hess[a, b] -= d * (s2[a, b] / s0 - ma * (s1[b] / s0))
            i = j

        for a in range(p):
            grad[a] -= alpha * beta[a]
            hess[a, a] -= alpha

        A = -hess
        for a in range(p):
            A[a, a] += 1e-10
        try:
            step = np.linalg.solve(A, grad)
        except Exception:
            break
        mx = np.abs(step).max()
        if mx > 5.0:
            step = step * (5.0 / mx)
        beta = beta + step
        if mx < tol:
            break
    return beta


@njit(cache=True, fastmath=True)
def _cindex(risk, time, event):
    n = risk.shape[0]
    conc = 0.0
    perm = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        ti = time[i]
        ri = risk[i]
        for j in range(n):
            if i == j:
                continue
            tj = time[j]
            if tj > ti or (tj == ti and event[j] == 0):
                perm += 1.0
                rj = risk[j]
                if ri > rj:
                    conc += 1.0
                elif ri == rj:
                    conc += 0.5
    if perm == 0.0:
        return 0.5
    return conc / perm


def sort_desc(X, t, ev):
    o = np.argsort(-t, kind="mergesort")
    return (np.ascontiguousarray(X[o]), np.ascontiguousarray(t[o]),
            np.ascontiguousarray(ev[o].astype(np.int32)))


def fit_ridge_cox(X, t, ev, alpha=100.0, max_iter=50, tol=1e-6):
    Xs, ts, evs = sort_desc(np.ascontiguousarray(X, dtype=np.float64), t, ev)
    return _cox_ridge_newton(Xs, ts, evs, float(alpha), max_iter, tol)


def cindex(risk, t, ev):
    return _cindex(np.ascontiguousarray(risk, dtype=np.float64),
                   np.ascontiguousarray(t, dtype=np.float64),
                   np.ascontiguousarray(ev, dtype=np.int32))


def validate_against_sksurv(store, cohorts=("TCGA", "METABRIC"), n_panels=6, seed=0):
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    rng = np.random.default_rng(seed)
    genes = common_genes(store, list(cohorts))
    tr, te = cohorts[0], cohorts[1]
    Xtr_all, ttr, etr, _ = store[tr]
    Xte_all, tte, ete, _ = store[te]
    rows = []
    panels = [NOVEL5, ANCHOR4, NOVEL5 + ANCHOR4] + \
             [list(rng.choice(genes, size=k, replace=False)) for k in (5, 20, 50)][:n_panels - 3]
    for panel in panels:
        panel = [g for g in panel if g in genes]
        A, B = Xtr_all[panel].values, Xte_all[panel].values
        for alpha in (10.0, 100.0):
            b_mine = fit_ridge_cox(A, ttr, etr, alpha=alpha)
            c_mine = cindex(B @ b_mine, tte, ete)
            y = np.array([(bool(e), float(t)) for e, t in zip(etr, ttr)],
                         dtype=[("event", bool), ("time", float)])
            m = CoxPHSurvivalAnalysis(alpha=alpha, ties="breslow", n_iter=200)
            m.fit(A, y)
            c_sk = concordance_index_censored(ete.astype(bool), tte, m.predict(B))[0]
            rows.append({"panel_size": len(panel), "alpha": alpha,
                         "c_jit": c_mine, "c_sksurv": c_sk,
                         "abs_diff_c": abs(c_mine - c_sk),
                         "max_abs_beta_diff": float(np.abs(b_mine - m.coef_).max()),
                         "corr_beta": float(np.corrcoef(b_mine, m.coef_)[0, 1])
                         if len(panel) > 1 else 1.0})
    return pd.DataFrame(rows)


st = load_all(['TCGA', 'METABRIC'])
df = validate_against_sksurv(st)
os.makedirs("setup", exist_ok=True)
df.to_csv(os.path.join("setup", "solver_validation.csv"), index=False)
pd.set_option('display.width', 200)
print(df.to_string(index=False))
print('MAX |dC| =', df.abs_diff_c.max())