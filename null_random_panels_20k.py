"""
null_random_panels_20k.py
=========================
Review item 12: the manuscript reports the random-panel permutation p as "p < 0.001",
which is that test's resolution floor with only 1000 draws -- it is not a measured
value, it merely means "0 of 1000". This script re-runs the same null with 20000
draws so the empirical p has resolution to ~5e-5.

EXACT DEFINITION
----------------
Null hypothesis: a random five-gene panel performs as well as NOVEL5 under the
pre-specified pipeline.

Draw: sample 5 distinct genes uniformly WITHOUT replacement from the genes present in
ALL SIX overall-survival cohorts (TCGA, METABRIC, SCANB_GSE96058, SCANB_GSE202203,
GSE20711, GSE58812). Draws are independent; the sampling universe is identical for
every draw and includes the NOVEL5 genes themselves.

Score of a draw: mean leave-one-cohort-out concordance. For each of the six held-out
cohorts, ridge Cox (Breslow ties, alpha = 100, nested_core.fit_ridge_cox -- the
pre-specified learner) is fitted on the pooled other five cohorts and Harrell's C
(nested_core.cindex) is computed once on the held-out cohort; the six values are
averaged. Every gene is z-scored WITHIN cohort before pooling (matters because the
SCANB_GSE96058 matrix is not z-scored at source).

Observed value: the identical quantity for NOVEL5 = FLT3, CLIC6, SUSD3, ZIC2, P4HA2,
computed by the same code path in the same run.

Empirical p (one-sided, right tail, add-one / Davison-Hinkley convention):
        p = ( #{draws with mean_loco_c >= observed} + 1 ) / ( N + 1 )
with N = 20000, so the smallest attainable value is 1/20001 = 4.9995e-5.

OUTPUTS
  null_random_panels_20k.csv   one row per draw: draw_id, panel, mean_loco_c
  null_20k_summary.json        observed value, exact p, null mean/SD/quantiles,
                               per-fold observed C, sampling universe size
"""
import gc
import json
import time

import numpy as np
import pandas as pd
from numba import njit, prange

import nested_core as nc
from nested_core import _cox_ridge_newton, _cindex

ALPHA = 100.0
NDRAW = 20000
PANEL_SIZE = 5
SEED = 20240601
MAX_ITER = 25
TOL = 1e-5


@njit(cache=True, fastmath=True, parallel=True)
def score_panels(Xall, panels, train_rows, train_off, ttr_all, etr_all,
                 test_rows, test_off, tte_all, ete_all, alpha, max_iter, tol):
    """Mean LOCO Harrell C for each panel (rows of `panels` are column indices)."""
    ndraw, p = panels.shape
    nf = train_off.shape[0] - 1
    out = np.empty(ndraw)
    per_fold = np.empty((ndraw, nf))
    for d in prange(ndraw):
        acc = 0.0
        for f in range(nf):
            a0, a1 = train_off[f], train_off[f + 1]
            ntr = a1 - a0
            Atr = np.empty((ntr, p))
            for i in range(ntr):
                r = train_rows[a0 + i]
                for j in range(p):
                    Atr[i, j] = Xall[r, panels[d, j]]
            beta = _cox_ridge_newton(Atr, ttr_all[a0:a1], etr_all[a0:a1],
                                     alpha, max_iter, tol)
            b0, b1 = test_off[f], test_off[f + 1]
            nte = b1 - b0
            risk = np.empty(nte)
            for i in range(nte):
                r = test_rows[b0 + i]
                s = 0.0
                for j in range(p):
                    s += Xall[r, panels[d, j]] * beta[j]
                risk[i] = s
            c = _cindex(risk, tte_all[b0:b1], ete_all[b0:b1])
            per_fold[d, f] = c
            acc += c
        out[d] = acc / nf
    return out, per_fold


def build():
    store = nc.load_all(nc.OS6)
    genes = nc.common_genes(store, nc.OS6)
    print("common genes across the six OS cohorts: %d" % len(genes), flush=True)

    blocks, times, events, coh_slice = [], [], [], {}
    pos = 0
    for coh in nc.OS6:
        X, t, ev, _ = store[coh]
        A = X[genes].values.astype(np.float32)
        blocks.append(A)
        times.append(t)
        events.append(ev)
        coh_slice[coh] = (pos, pos + A.shape[0])
        pos += A.shape[0]
    Xall = np.ascontiguousarray(np.vstack(blocks))
    tall = np.concatenate(times).astype(np.float64)
    eall = np.concatenate(events).astype(np.int32)
    del blocks, store
    gc.collect()
    print("Xall %s (%.2f GB)" % (Xall.shape, Xall.nbytes / 1e9), flush=True)

    tr_rows, tr_t, tr_e, tr_off = [], [], [], [0]
    te_rows, te_t, te_e, te_off = [], [], [], [0]
    for held in nc.OS6:
        idx = np.concatenate([np.arange(*coh_slice[c]) for c in nc.OS6 if c != held])
        o = np.argsort(-tall[idx], kind="mergesort")   # descending time for the fitter
        idx = idx[o]
        tr_rows.append(idx); tr_t.append(tall[idx]); tr_e.append(eall[idx])
        tr_off.append(tr_off[-1] + len(idx))
        j = np.arange(*coh_slice[held])
        te_rows.append(j); te_t.append(tall[j]); te_e.append(eall[j])
        te_off.append(te_off[-1] + len(j))

    pack = dict(
        Xall=Xall,
        train_rows=np.concatenate(tr_rows).astype(np.int64),
        train_off=np.array(tr_off, dtype=np.int64),
        ttr_all=np.ascontiguousarray(np.concatenate(tr_t)),
        etr_all=np.ascontiguousarray(np.concatenate(tr_e).astype(np.int32)),
        test_rows=np.concatenate(te_rows).astype(np.int64),
        test_off=np.array(te_off, dtype=np.int64),
        tte_all=np.ascontiguousarray(np.concatenate(te_t)),
        ete_all=np.ascontiguousarray(np.concatenate(te_e).astype(np.int32)),
    )
    return genes, pack


def run(panels, pack):
    return score_panels(pack["Xall"], panels, pack["train_rows"], pack["train_off"],
                        pack["ttr_all"], pack["etr_all"], pack["test_rows"],
                        pack["test_off"], pack["tte_all"], pack["ete_all"],
                        ALPHA, MAX_ITER, TOL)


def main():
    genes, pack = build()
    gi = {g: i for i, g in enumerate(genes)}

    missing = [g for g in nc.NOVEL5 if g not in gi]
    if missing:
        raise SystemExit("NOVEL5 genes absent from the common set: %s" % missing)
    obs_panel = np.array([[gi[g] for g in nc.NOVEL5]], dtype=np.int64)

    t0 = time.time()
    obs, obs_fold = run(obs_panel, pack)
    print("observed NOVEL5 mean LOCO C = %.6f  (compile+run %.1fs)"
          % (obs[0], time.time() - t0), flush=True)
    print("  per fold: %s" % dict(zip(nc.OS6, np.round(obs_fold[0], 4))), flush=True)

    rng = np.random.default_rng(SEED)
    G = len(genes)
    panels = np.empty((NDRAW, PANEL_SIZE), dtype=np.int64)
    for d in range(NDRAW):
        panels[d] = rng.choice(G, size=PANEL_SIZE, replace=False)

    t0 = time.time()
    warm, _ = run(panels[:200], pack)
    dt = time.time() - t0
    print("200 draws in %.1fs -> projected %.1f min for %d"
          % (dt, dt / 200 * NDRAW / 60, NDRAW), flush=True)

    scores = np.empty(NDRAW)
    scores[:200] = warm
    CH = 2000
    for s in range(200, NDRAW, CH):
        e = min(s + CH, NDRAW)
        sc, _ = run(panels[s:e], pack)
        scores[s:e] = sc
        np.save("null_scores_partial.npy", scores[:e])
        print("  draws %6d..%6d done  (elapsed %.1f min, running max %.4f)"
              % (s, e, (time.time() - t0) / 60, scores[:e].max()), flush=True)

    df = pd.DataFrame(dict(
        draw_id=np.arange(NDRAW),
        panel=[";".join(genes[i] for i in panels[d]) for d in range(NDRAW)],
        mean_loco_c=scores))
    df.to_csv("null_random_panels_20k.csv", index=False)

    observed = float(obs[0])
    n_ge = int((scores >= observed).sum())
    p_emp = (n_ge + 1) / (NDRAW + 1)
    summary = dict(
        review_item=12,
        n_draws=NDRAW, panel_size=PANEL_SIZE, seed=SEED, alpha=ALPHA,
        sampling_universe_n_genes=len(genes),
        cohorts=nc.OS6,
        observed_novel5_mean_loco_c=observed,
        observed_per_fold=dict(zip(nc.OS6, [float(x) for x in obs_fold[0]])),
        n_draws_ge_observed=n_ge,
        p_empirical_addone=p_emp,
        p_formula="(#draws >= observed + 1)/(N + 1)",
        resolution_floor=1.0 / (NDRAW + 1),
        null_mean=float(scores.mean()), null_sd=float(scores.std(ddof=1)),
        null_min=float(scores.min()), null_max=float(scores.max()),
        null_quantiles={str(q): float(np.quantile(scores, q))
                        for q in (0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999)},
        observed_z_vs_null=float((observed - scores.mean()) / scores.std(ddof=1)),
        observed_percentile=float((scores < observed).mean() * 100),
        best_random_panel=df.loc[int(np.argmax(scores)), "panel"],
        prior_report="p < 0.001 with 1000 draws, i.e. 0/1000 -- the floor of that design",
    )
    json.dump(summary, open("null_20k_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
