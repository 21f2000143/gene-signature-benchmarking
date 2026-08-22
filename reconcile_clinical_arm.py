"""
reconcile_clinical_arm.py
=========================

WHY THIS SCRIPT EXISTS
----------------------
Two earlier scripts each computed a leave-one-cohort-out clinicopathology arm,
and they disagreed:

  metrics_uno_auc_ph.py -> metrics_harrell_uno.csv  (feeds Table 2)
  run_clinical_arm.py   -> clinical_arm_recomputed.csv (feeds Table 4)

The Novel-5 gene columns agree between them to <6e-4, so the fold machinery is
consistent. The CLINICAL columns differ by up to 0.058, and the disagreement is
not numerical -- it is a covariate-definition difference:

  GSE20711 supplies age and tumour size as UNLABELLED 0/1 dichotomies. The
  clinical audit (run_clinical_arm.py) therefore refuses to construct
  age_years and size_gt20 for that cohort, leaving grade3/node_pos/er_pos.
  metrics_uno_auc_ph.py instead passes the raw 0/1 columns through as if they
  were continuous age in years and size in millimetres. That inflates the
  GSE20711 clinical c-index from 0.653 to 0.711.

The audited definition is the correct one: a 0/1 column whose labelling is
unknown cannot be interpreted as years or millimetres, and the manuscript
already states in the Table 4 audit that these two variables are unusable
there. Reporting a clinical arm that quietly uses them contradicts our own
audit and, because the clinical arm is the comparator the gene panel LOSES to,
it inflates the very number the paper's central negative claim rests on.

So this script establishes ONE clinical arm, computed under the audited
covariate rule, carrying BOTH concordance measures (Harrell and Uno's
censoring-robust C at the fold-specific tau) so that Table 2 and Table 4 can
both read it and cannot drift apart again.

Uno's C needs the IPCW censoring model, so tau is taken as in
metrics_uno_auc_ph.py: the largest event time in the held-out cohort that keeps
the IPCW weights finite.

OUTPUT
------
  clinical_arm_reconciled.csv   one row per held-out cohort: audited covariates,
                                Harrell c, Uno c, tau, n/events, bootstrap CI
  clinical_arm_reconciled.json  the two-source discrepancy, quantified
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nested_core as nc
from nested_core import OS6, fit_ridge_cox, cindex

from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

ALPHA = 100.0
NOVEL5 = ["FLT3", "CLIC6", "SUSD3", "ZIC2", "P4HA2"]
# audited harmonised covariates, exactly as run_clinical_arm.py defines them
HARM = ["age_years", "grade3", "node_pos", "size_gt20", "er_pos", "pr_pos"]


def log(m):
    print(m, flush=True)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def harmonise_clin(coh, s):
    """Audited harmonisation. Mirrors run_clinical_arm.py; the two GSE20711
    refusals (age, size) are the whole point of this script."""
    out = pd.DataFrame(index=s.index)
    col = {c.lower(): s[c] for c in s.columns}

    out["age_years"] = np.nan
    if "age" in col:
        a = _num(col["age"])
        # an unlabelled 0/1 column is NOT age in years
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
        np_ = pd.Series(np.nan, index=s.index)
        if num.notna().sum() > 0.5 * len(s):
            np_ = (num > 0).astype(float).where(num.notna())
        else:
            neg = rs.str.upper().str.match(r"^N0") | rs.isin(
                ["NodeNegative", "Node negative", "negative", "0"])
            pos = rs.isin(["NodePositive", "SubMicroMet", "1to3", "4toX"]) | \
                rs.str.upper().str.match(r"^N[1-3]")
            np_ = pd.Series(np.where(pos, 1.0, np.where(neg, 0.0, np.nan)),
                            index=s.index)
        out["node_pos"] = np_

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
                out["size_gt20"] = np.nan          # unlabelled dichotomy (GSE20711)
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


def zscore(v):
    """Within-cohort z-score, NaN imputed to the cohort mean.

    A training cohort may not be able to construct a covariate that the
    held-out cohort does have (GSE20711 has no usable age_years, but TCGA
    does), leaving an all-NaN column. Such a column carries no information,
    so it is set to the pooled centre, 0 -- which is what z-scoring plus
    mean-imputation degenerates to. Returning NaN instead propagates into the
    Newton step and divides by zero.
    """
    v = np.asarray(v, float)
    if not np.isfinite(v).any():
        return np.zeros_like(v)
    m = np.nanmean(v)
    s = np.nanstd(v)
    return (np.nan_to_num(v, nan=m) - m) / (s if s > 0 else 1.0)


def boot_ci(risk, t, ev, seed=0, B=400):
    rng = np.random.default_rng(seed)
    n = len(t)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if np.sum(ev[i]) < 3:
            continue
        vals.append(cindex(risk[i], t[i], ev[i]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) \
        if vals else (np.nan, np.nan)


def uno_c(t_tr, e_tr, t_te, e_te, risk):
    """Uno's C at the largest tau keeping IPCW weights finite (as in
    metrics_uno_auc_ph.py)."""
    y_tr = Surv.from_arrays(event=e_tr.astype(bool), time=t_tr)
    y_te = Surv.from_arrays(event=e_te.astype(bool), time=t_te)
    ev_times = np.sort(t_te[e_te.astype(bool)])
    for q in (1.0, 0.99, 0.975, 0.95, 0.9):
        if len(ev_times) == 0:
            break
        tau = float(np.quantile(ev_times, q))
        if tau >= t_tr.max():
            continue
        try:
            v = float(concordance_index_ipcw(y_tr, y_te, risk, tau=tau)[0])
            if np.isfinite(v):
                return v, tau
        except Exception:
            continue
    return np.nan, np.nan


# ------------------------------------------------------------------ load
store = nc.load_all(OS6, verbose=True)
H, AVAIL = {}, {}
for coh in OS6:
    s = nc.load_cohort(coh)[3]
    h = harmonise_clin(coh, s)
    H[coh] = h
    AVAIL[coh] = {c: usable(h[c], len(h)) for c in HARM}
    log("%-18s usable: %s" % (coh, ",".join(c for c in HARM if AVAIL[coh][c])))


def clin_matrix(coh, covs):
    h = H[coh]
    cols = [zscore(h[c].values) for c in covs]
    return (np.column_stack(cols) if cols else None)


rows = []
for held in OS6:
    covs = [c for c in HARM if AVAIL[held][c]]
    tr = [c for c in OS6 if c != held]

    Xs, ts, es = [], [], []
    for c in tr:
        M = clin_matrix(c, covs)
        if M is None:
            continue
        _, t, ev, _ = store[c]
        Xs.append(M); ts.append(t); es.append(ev)
    Xtr = np.vstack(Xs)
    ttr = np.concatenate(ts); etr = np.concatenate(es).astype(np.int32)

    Mte = clin_matrix(held, covs)
    _, tte, ete, _ = store[held]
    ete = np.asarray(ete, np.int32)

    b = fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)
    risk = Mte @ b
    hc = float(cindex(risk, tte, ete))
    uc, tau = uno_c(ttr, etr, tte, ete, risk)
    lo, hi = boot_ci(risk, tte, ete)

    # same folds, gene panel, for the paired delta
    Xg = [store[c][0][NOVEL5].values for c in tr]
    bg = fit_ridge_cox(np.vstack(Xg), ttr, etr, alpha=ALPHA)
    rg = store[held][0][NOVEL5].values @ bg
    hg = float(cindex(rg, tte, ete))
    ug, _ = uno_c(ttr, etr, tte, ete, rg)

    rows.append(dict(held_out_cohort=held, covariates="|".join(covs),
                     n_covariates=len(covs), clinical_harrell=hc,
                     clinical_uno=uc, tau_months=tau,
                     clinical_ci_lo=lo, clinical_ci_hi=hi,
                     novel5_harrell=hg, novel5_uno=ug,
                     delta_harrell=hg - hc, delta_uno=ug - uc,
                     n_test=int(len(tte)), events_test=int(ete.sum()),
                     n_train=int(Xtr.shape[0]), events_train=int(etr.sum())))
    log("  %-18s covs=%d clinHarrell=%.4f clinUno=%.4f novel=%.4f d=%+.4f"
        % (held, len(covs), hc, uc, hg, hg - hc))

rec = pd.DataFrame(rows)
rec.to_csv("clinical_arm_reconciled.csv", index=False)

# quantify the discrepancy against the two superseded sources, for the record
disc = {}
for name, f, col, key in [
        ("metrics_harrell_uno.csv", "metrics_harrell_uno.csv", "harrell_c", "gene_set"),
        ("clinical_arm_recomputed.csv", "clinical_arm_recomputed.csv", "cindex", "arm")]:
    if os.path.exists(f):
        d = pd.read_csv(f)
        sel = d[d[key].isin(["Clinical", "clinical_per_cohort_available"])]
        s = sel.set_index("held_out_cohort")[col].reindex(OS6)
        delta = (rec.set_index("held_out_cohort").clinical_harrell.reindex(OS6) - s)
        disc[name] = {"per_cohort": {k: round(float(v), 4) for k, v in delta.items()},
                      "max_abs": round(float(delta.abs().max()), 4)}

summary = dict(
    n_cohorts=len(rec),
    clinical_harrell_mean=round(float(rec.clinical_harrell.mean()), 4),
    clinical_uno_mean=round(float(rec.clinical_uno.mean()), 4),
    novel5_harrell_mean=round(float(rec.novel5_harrell.mean()), 4),
    novel5_uno_mean=round(float(rec.novel5_uno.mean()), 4),
    delta_harrell_min=round(float(rec.delta_harrell.min()), 4),
    delta_harrell_max=round(float(rec.delta_harrell.max()), 4),
    clinical_wins_harrell=int((rec.delta_harrell < 0).sum()),
    clinical_wins_uno=int((rec.delta_uno < 0).sum()),
    covariates_by_cohort={r.held_out_cohort: r.covariates for r in rec.itertuples()},
    discrepancy_vs_superseded_sources=disc,
    resolution=("audited covariate rule adopted: GSE20711 age and size are "
                "unlabelled 0/1 dichotomies and are excluded, so the clinical "
                "arm there uses grade3/node_pos/er_pos only"))
json.dump(summary, open("clinical_arm_reconciled.json", "w"), indent=2)
log(json.dumps(summary, indent=2))
