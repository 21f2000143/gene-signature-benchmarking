"""
nested_core.py -- core numerics for fully nested re-selection of the five-gene panel.

Addresses reviewer item 1: The published panel was chosen by a forward search whose scoring criterion
(leave-one-cohort-out C-index) is the same quantity later used to evaluate the panel. This module re-
implements that search within each outer LOCO fold and allows the search to be repeated with permuted labels.

Numerics:
  * ridge-penalised Cox partial likelihood (Breslow ties), Newton-Raphson, numba-JIT
  * Harrell's concordance, numba-JIT
Both are validated against scikit-survival in validate_against_sksurv().


  1. the four anchor-complex genes are forced in as scaffold;
  2. CANDIDATE POOL, computed inside each fold -- every gene is ranked by its
     univariate association with survival in the training cohorts only, and just the
     top ~120 genes that are concordantly prognostic across those cohorts are kept;
  3. greedy forward addition within that filtered pool, each candidate scored by
     held-out c-index on the fold's test cohort;
  4. CROSS-FOLD CONSENSUS -- the folds' trajectories are combined, so a gene earns its
     place by being picked early across independent folds rather than by winning a
     single averaged path.
Steps 2 and 4 matter: the pre-filter is what makes the search a ~120-candidate problem
rather than a 16,000-candidate one, and the consensus is what makes the resulting panel
a fold-stable object. Both are reproduced here.

"""
import os
import numpy as np
import pandas as pd
from numba import njit, prange

ANCHOR4 = ["EEF1A2", "IQGAP1", "IQGAP2", "FRG1"]
NOVEL5 = ["FLT3", "CLIC6", "SUSD3", "ZIC2", "P4HA2"]
OS6 = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203", "GSE20711", "GSE58812"]
SEC3 = ["GSE6532", "GSE11121", "GSE21653"]

DATA = os.path.expanduser("harmonised")


# ----------------------------------------------------------------------------- data
def load_cohort(coh, data_dir=DATA):
    """Load one cohort; z-score every gene within the cohort (the paper's protocol)."""
    e = pd.read_parquet(os.path.join(data_dir, coh + "_expr.parquet"))
    s = pd.read_parquet(os.path.join(data_dir, coh + "_surv.parquet"))
    # align on sample order
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
    """Return dict cohort -> (expr DataFrame z-scored, time, event, surv table)."""
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


# ------------------------------------------------------------------- Cox numerics
@njit(cache=True, fastmath=True)
def _cox_ridge_newton(X, t_order_idx, event, alpha, max_iter, tol):
    """Ridge Cox by Newton-Raphson, Breslow ties.

    X must already be ordered by DESCENDING time so risk sets are prefix-cumulative.
    Returns beta (p,).
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        # guard the exponential
        m = eta.max()
        if m > 30.0:
            eta = eta - (m - 30.0)
        w = np.exp(eta)

        # running (cumulative over descending time) risk-set aggregates
        s0 = 0.0
        s1 = np.zeros(p)
        s2 = np.zeros((p, p))
        grad = np.zeros(p)
        hess = np.zeros((p, p))
        ll = 0.0
        i = 0
        while i < n:
            # accumulate all subjects sharing this time into the risk set
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
            # deaths at this time (Breslow)
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

        # ridge penalty
        for a in range(p):
            grad[a] -= alpha * beta[a]
            hess[a, a] -= alpha

        # Newton step: solve (-H) step = grad
        A = -hess
        for a in range(p):
            A[a, a] += 1e-10
        try:
            step = np.linalg.solve(A, grad)
        except Exception:
            break
        # damped update
        mx = np.abs(step).max()
        if mx > 5.0:
            step = step * (5.0 / mx)
        beta = beta + step
        if mx < tol:
            break
    return beta


@njit(cache=True, fastmath=True)
def _cindex(risk, time, event):
    """Harrell's concordance. Ties in risk score contribute 0.5."""
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
    """Order rows by descending time; return contiguous arrays for the JIT fitter."""
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


# --------------------------------------------------------- prepared fold matrices
class FoldData:
    """Pre-extracted, pre-sorted matrices for one inner train/test cohort split.

    Holds the FULL gene matrix once so the forward search only has to slice columns.
    """

    def __init__(self, store, train_cohorts, test_cohort, genes, perm_seed=None):
        self.genes = list(genes)
        gi = {g: k for k, g in enumerate(self.genes)}
        self.gi = gi

        Xtr, ttr, etr = [], [], []
        for coh in train_cohorts:
            X, t, ev, _ = store[coh]
            Xtr.append(X[self.genes].values)
            t2, e2 = t.copy(), ev.copy()
            if perm_seed is not None:
                rng = np.random.default_rng(perm_seed + abs(hash(coh)) % 100000)
                o = rng.permutation(len(t2))
                t2, e2 = t2[o], e2[o]
            ttr.append(t2)
            etr.append(e2)
        Xtr = np.vstack(Xtr)
        ttr = np.concatenate(ttr)
        etr = np.concatenate(etr)
        o = np.argsort(-ttr, kind="mergesort")
        self.Xtr = np.ascontiguousarray(Xtr[o])
        self.ttr = np.ascontiguousarray(ttr[o])
        self.etr = np.ascontiguousarray(etr[o].astype(np.int32))

        X, t, ev, _ = store[test_cohort]
        t2, e2 = t.copy(), ev.copy()
        if perm_seed is not None:
            rng = np.random.default_rng(perm_seed + 7919 + abs(hash(test_cohort)) % 100000)
            o2 = rng.permutation(len(t2))
            t2, e2 = t2[o2], e2[o2]
        self.Xte = np.ascontiguousarray(X[self.genes].values)
        self.tte = np.ascontiguousarray(t2)
        self.ete = np.ascontiguousarray(e2.astype(np.int32))
        self.test_cohort = test_cohort
        self.train_cohorts = list(train_cohorts)


@njit(cache=True, fastmath=True, parallel=True)
def _score_candidates(Xtr, ttr, etr, Xte, tte, ete, base_cols, cand_cols,
                      alpha, max_iter, tol):
    """For each candidate column, fit on train / score on test with base+candidate."""
    ncand = cand_cols.shape[0]
    nb = base_cols.shape[0]
    out = np.empty(ncand)
    for q in prange(ncand):
        cc = cand_cols[q]
        ntr = Xtr.shape[0]
        nte = Xte.shape[0]
        Atr = np.empty((ntr, nb + 1))
        Ate = np.empty((nte, nb + 1))
        for a in range(nb):
            bc = base_cols[a]
            for i in range(ntr):
                Atr[i, a] = Xtr[i, bc]
            for i in range(nte):
                Ate[i, a] = Xte[i, bc]
        for i in range(ntr):
            Atr[i, nb] = Xtr[i, cc]
        for i in range(nte):
            Ate[i, nb] = Xte[i, cc]
        beta = _cox_ridge_newton(Atr, ttr, etr, alpha, max_iter, tol)
        risk = Ate @ beta
        out[q] = _cindex(risk, tte, ete)
    return out


def score_candidates(fold, base_genes, cand_genes, alpha=100.0,
                     max_iter=25, tol=1e-5):
    base_cols = np.array([fold.gi[g] for g in base_genes], dtype=np.int64)
    cand_cols = np.array([fold.gi[g] for g in cand_genes], dtype=np.int64)
    return _score_candidates(fold.Xtr, fold.ttr, fold.etr,
                             fold.Xte, fold.tte, fold.ete,
                             base_cols, cand_cols, float(alpha), max_iter, tol)


# ------------------------------------------------- step 2: univariate candidate pool
@njit(cache=True, fastmath=True, parallel=True)
def _univariate_scores(X, t, ev):
    """Per-gene univariate concordance against survival. Returns (p,) of C values."""
    p = X.shape[1]
    out = np.empty(p)
    for j in prange(p):
        out[j] = _cindex(np.ascontiguousarray(X[:, j]), t, ev)
    return out


def candidate_pool(store, train_cohorts, genes, top_k=120, perm_seed=None,
                   verbose=False):
    """Discovery report step 2: rank genes by univariate association with survival in
    the TRAINING cohorts only, keep the top_k that are concordantly prognostic across
    them.

    "Concordantly prognostic" is read as: the gene's univariate concordance points the
    same direction (all above 0.5 or all below 0.5) in every training cohort. Genes are
    then ranked by mean |C - 0.5| and the strongest top_k retained.
    """
    per = []
    for coh in train_cohorts:
        X, t, ev, _ = store[coh]
        t2, e2 = t.copy(), ev.copy()
        if perm_seed is not None:
            rng = np.random.default_rng(perm_seed + abs(hash(coh)) % 100000)
            o = rng.permutation(len(t2))
            t2, e2 = t2[o], e2[o]
        A = np.ascontiguousarray(X[genes].values, dtype=np.float64)
        o = np.argsort(-t2, kind="mergesort")
        per.append(_univariate_scores(np.ascontiguousarray(A[o]),
                                     np.ascontiguousarray(t2[o]),
                                     np.ascontiguousarray(e2[o].astype(np.int32))))
    M = np.vstack(per)                      # (n_train_cohorts, n_genes)
    sign_up = (M > 0.5).all(axis=0)
    sign_dn = (M < 0.5).all(axis=0)
    concordant = sign_up | sign_dn
    strength = np.abs(M - 0.5).mean(axis=0)
    strength[~concordant] = -1.0            # discordant genes are ineligible
    order = np.argsort(-strength)[:top_k]
    pool = [genes[k] for k in order if strength[k] > 0]
    if verbose:
        print("    pool: %d concordant of %d scanned -> kept %d"
              % (int(concordant.sum()), len(genes), len(pool)), flush=True)
    return pool


# ------------------------------------------------------------------ forward search
def _greedy_one_fold(fold, pool, n_steps, anchor, alpha):
    """Greedy forward addition within one fold, scored on that fold's test cohort."""
    selected, trace = [], []
    for step in range(n_steps):
        base = anchor + selected
        cands = [g for g in pool if g not in base]
        if not cands:
            break
        sc = score_candidates(fold, base, cands, alpha=alpha)
        order = np.argsort(-sc)
        best = cands[order[0]]
        selected.append(best)
        trace.append({"step": step + 1, "chosen": best, "score": float(sc[order[0]]),
                      "top10": [(cands[k], float(sc[k])) for k in order[:10]],
                      "n_candidates": len(cands)})
    return selected, trace


def forward_search(store, search_cohorts, genes, n_steps=5, anchor=ANCHOR4,
                   alpha=100.0, perm_seed=None, verbose=True, log=None,
                   top_k=120, n_final=5):
    """
    1. anchor forced in as scaffold
    2. per-fold univariate candidate pool (top_k concordant genes, training cohorts only)
    3. greedy forward addition inside each fold, scored on that fold's held-out cohort
    4. cross-fold consensus: rank genes by how often and how early the folds picked them

    Returns (consensus panel of n_final genes, trace records). The anchor is scaffold and
    is not part of the returned panel.
    """
    anchor = [g for g in anchor if g in genes]
    inner_folds = []
    for held in search_cohorts:
        tr = [c for c in search_cohorts if c != held]
        pool = candidate_pool(store, tr, genes, top_k=top_k, perm_seed=perm_seed,
                              verbose=verbose)
        inner_folds.append((held, tr, FoldData(store, tr, held, genes,
                                               perm_seed=perm_seed), pool))
    if verbose:
        print("  inner folds: %d | universe %d | pool/fold %s | anchor %s"
              % (len(inner_folds), len(genes),
                 ",".join(str(len(f[3])) for f in inner_folds), ",".join(anchor)),
              flush=True)

    # ---- step 3: independent greedy trajectory per fold
    per_fold = []
    trace = []
    for held, tr, fold, pool in inner_folds:
        sel, tr_rec = _greedy_one_fold(fold, pool, n_steps, anchor, alpha)
        per_fold.append(sel)
        for r in tr_rec:
            r["inner_held_out"] = held
            r["pool_size"] = len(pool)
        trace.extend(tr_rec)
        if verbose:
            print("    fold(-%s): %s" % (held, ",".join(sel)), flush=True)
        if log is not None:
            log.write("fold %s picks %s\n" % (held, ",".join(sel)))
            log.flush()

    # ---- step 4: cross-fold consensus. A gene earns its place by being picked, and
    # picked EARLY, across independent folds. Score = sum over folds of (n_steps - rank),
    # so an early pick in several folds beats a late pick in one.
    score = {}
    picks = {}
    for sel in per_fold:
        for rank, g in enumerate(sel):
            score[g] = score.get(g, 0.0) + (n_steps - rank)
            picks[g] = picks.get(g, 0) + 1
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], -picks[kv[0]], kv[0]))
    panel = [g for g, _ in ranked[:n_final]]
    consensus = [{"gene": g, "consensus_score": s, "n_folds_picked": picks[g],
                  "n_folds": len(per_fold)} for g, s in ranked[:25]]
    if verbose:
        print("  consensus panel: %s" % ",".join(panel), flush=True)
        print("  (top: %s)" % "; ".join("%s s=%.0f in %d/%d folds"
                                        % (r["gene"], r["consensus_score"],
                                           r["n_folds_picked"], r["n_folds"])
                                        for r in consensus[:8]), flush=True)
    return panel, {"trace": trace, "per_fold_panels": per_fold,
                   "consensus": consensus,
                   "pool_sizes": [len(f[3]) for f in inner_folds],
                   "inner_held_out": [f[0] for f in inner_folds]}


def evaluate_panel(store, train_cohorts, test_cohort, panel, alpha=100.0):
    """Fit ridge Cox on pooled training cohorts, score on the test cohort."""
    Xtr = np.vstack([store[c][0][panel].values for c in train_cohorts])
    ttr = np.concatenate([store[c][1] for c in train_cohorts])
    etr = np.concatenate([store[c][2] for c in train_cohorts])
    b = fit_ridge_cox(Xtr, ttr, etr, alpha=alpha)
    X, t, ev, _ = store[test_cohort]
    return cindex(X[panel].values @ b, t, ev), b


# ---------------------------------------------------------------------- validation
def validate_against_sksurv(store, cohorts=("TCGA", "METABRIC"), n_panels=6, seed=0):
    """Compare JIT ridge-Cox + concordance against scikit-survival on real panels."""
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
