
"""
run_sample_normalization_c6.py -- reviewer item C6: does the panel survive a
DEPLOYABLE, sample-level normalisation rather than the cohort-level z-scoring
used everywhere else in this paper?

Every arm in this paper is fitted on gene expression standardised PER GENE,
ACROSS the cohort (nested_core.load_cohort): mu/sd are computed from all
samples in the cohort, which is unavailable to a single incoming patient
specimen. Two normalisations that ARE available to a single sample are tested
here, applied to the panel's five genes only, using the ten housekeeping genes
already present in every cohort's matrix as the reference set:

  (a) housekeeping-ratio: for each sample, subtract the mean of the ten
      housekeeping genes' (already log-like, per-gene standardised) values
      from each of the five panel genes' values for that same sample -- the
      standard ddCt-style referencing scheme for a targeted qPCR panel.
  (b) within-sample rank: replace each of the five panel genes' values, within
      that sample, with its rank (1..5) among the five -- the most aggressive
      deployable option, discarding all magnitude information.

Both are then z-scored per gene ACROSS THE TRAINING COHORTS ONLY (never using
the held-out cohort's distribution) so the fold design matches every other LOCO
fold in this paper; the held-out cohort's samples are transformed with the
training-derived mu/sd. Ridge Cox (alpha=100, the pre-specified learner) is
refit under each normalisation with the same LOCO design as Table 2, and
concordance is compared against the panel's own standard-normalisation arm.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nested_core as nc
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator

HK = ["ACTB", "GAPDH", "B2M", "RPLP0", "TBP", "HPRT1", "PPIA", "PGK1", "POLR2A", "YWHAZ"]
NOVEL5 = nc.NOVEL5
OS6 = nc.OS6


def surv_y(t, ev):
    return np.array([(bool(e), float(x)) for e, x in zip(ev, t)], dtype=[("event", bool), ("time", float)])


def tau_for_fold(ttr, etr, tte, ete):
    y_tr = surv_y(ttr, etr)
    cens = CensoringDistributionEstimator().fit(y_tr)
    max_ev_te = float(tte[ete == 1].max()) if ete.sum() > 0 else float(tte.max())
    gx, gy = cens.unique_time_, cens.prob_
    pos = gx[gy > 1e-8]
    cap = float(pos.max()) if len(pos) else float(ttr.max())
    candidates = [min(max_ev_te, cap * (1 - 1e-9))]
    ev_times = tte[ete == 1]
    for q in (0.99, 0.95, 0.90):
        if len(ev_times):
            candidates.append(float(np.quantile(ev_times, q)))
    return candidates


def safe_uno(y_tr, y_te, risk, candidates):
    for tc in candidates:
        try:
            v = float(concordance_index_ipcw(y_tr, y_te, risk, tau=tc)[0])
            if np.isfinite(v):
                return v, float(tc)
        except Exception:
            continue
    return np.nan, np.nan


# ---- load the ALREADY cohort-z-scored matrices (this is the paper's normal path) ----
store = nc.load_all(OS6, verbose=True)


def make_housekeeping_ratio(Xdf):
    """Row-wise: subtract each sample's own housekeeping-gene mean from each panel gene."""
    hk_avail = [g for g in HK if g in Xdf.columns]
    hk_mean = Xdf[hk_avail].mean(axis=1)
    out = Xdf[NOVEL5].sub(hk_mean, axis=0)
    return out, hk_avail


def make_within_sample_rank(Xdf):
    """Row-wise rank (1..5) of the five panel genes within each sample."""
    return Xdf[NOVEL5].rank(axis=1, method="average")


rows = []
for held in OS6:
    train = [c for c in OS6 if c != held]
    Xte_all, tte, ete, _ = store[held]
    ytr_parts = [(store[c][1], store[c][2]) for c in train]
    ttr = np.concatenate([a for a, _ in ytr_parts])
    etr = np.concatenate([b for _, b in ytr_parts])
    y_tr = surv_y(ttr, etr)
    tau_candidates = tau_for_fold(ttr, etr, tte, ete)

    for design in ("housekeeping_ratio", "within_sample_rank"):
        tr_frames = []
        for coh in train:
            Xdf, _, _, _ = store[coh]
            if design == "housekeeping_ratio":
                feat, hk_avail = make_housekeeping_ratio(Xdf)
            else:
                feat = make_within_sample_rank(Xdf)
            tr_frames.append(feat)
        Xtr_raw = pd.concat(tr_frames, axis=0)

        if design == "housekeeping_ratio":
            Xte_raw, hk_avail = make_housekeeping_ratio(Xte_all)
        else:
            Xte_raw = make_within_sample_rank(Xte_all)

        # z-score the derived feature per gene, using TRAIN cohorts only (never touch held-out
        # distribution), then apply the same mu/sd to the held-out cohort -- this is the
        # deployable design: a new sample is referenced against a fixed, pre-computed scale.
        mu = Xtr_raw.mean(axis=0)
        sd = Xtr_raw.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Xtr = ((Xtr_raw - mu) / sd).values
        Xte = ((Xte_raw - mu) / sd).values
        Xtr = np.nan_to_num(Xtr)
        Xte = np.nan_to_num(Xte)

        beta = nc.fit_ridge_cox(Xtr, ttr, etr, alpha=100.0)
        risk = Xte @ beta
        hc = float(concordance_index_censored(ete.astype(bool), tte, risk)[0])
        y_te = surv_y(tte, ete)
        uc, tau_used = safe_uno(y_tr, y_te, risk, tau_candidates)

        rows.append(dict(held_out_cohort=held, normalisation=design,
                          harrell_c=hc, uno_c=uc, tau_months=tau_used,
                          n_train=int(len(ttr)), n_test=int(len(tte))))
        print("  %-16s %-20s Harrell=%.4f Uno=%.4f" % (held, design, hc, uc), flush=True)

out = pd.DataFrame(rows)
out.to_csv("sample_normalisation_c6.csv", index=False)
print(out.to_string(index=False), flush=True)
summary = out.groupby("normalisation")[["harrell_c", "uno_c"]].mean().round(4)
print(summary, flush=True)
print("DONE", flush=True)
