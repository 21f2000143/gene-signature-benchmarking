"""
run_gene_clinical_loco.py -- adds the GENE+CLINICAL combined arm to the PRIMARY
LOCO validation (the six OS cohorts), for every gene set in results/gene_sets.json,
not just Novel5.

WHY THIS SCRIPT EXISTS
----------------------
The manuscript's main tables treat "gene" (Table 2 / metrics_harrell_uno.csv) and
"clinical" (Table 4 / clinical_arm_reconciled.csv) as two separate arms and never
report a combined model in the primary LOCO design. A combined gene+clinical arm
was already computed elsewhere (benchmark_within.py -> incremental_value.csv,
run_incremental_value_c1.py), but only under a WITHIN-cohort CV design (not LOCO)
and, for the honest LR-test/DCA version, restricted to Novel5 in three cohorts.
This script closes that gap for the PRIMARY LOCO track: for every gene set, fit a
combined ridge Cox model on [gene features + audited clinical covariates] pooled
over the five training OS cohorts and evaluate once on the held-out cohort.

This deliberately reuses the AUDITED clinical covariate harmonisation and the
per-cohort-available covariate rule from reconcile_clinical_arm.py /
run_clinical_arm.py (age_years, grade3, node_pos, size_gt20, er_pos, pr_pos;
usable when >30% non-missing and >=2 distinct values), NOT the cruder
harmonisation in metrics_uno_auc_ph.py -- that file's "Clinical" row is the
SUPERSEDED, uncorrected clinical arm (reconcile_clinical_arm.py's docstring
documents a GSE20711 discrepancy of 0.711 vs the audited 0.653 caused by
treating unlabelled 0/1 age/size columns as continuous). Reusing the audited
rule here keeps the new combined arm on the same footing as Table 4 /
clinical_arm_reconciled.csv, which Table 2 is cross-checked against.

Learner: ridge Cox, alpha=100 (nested_core.fit_ridge_cox), Breslow ties -- the
same pre-specified learner as clinical_arm_reconciled.csv and the "gene arms for
reference" rows of run_clinical_arm.py, so all three arms (gene, clinical,
gene+clinical) are on a common footing, LOCO-fold by LOCO-fold. Harrell's C uses
nested_core.cindex, the same implementation reconcile_clinical_arm.py uses for
its own Novel5 recompute (documented there to agree with the sksurv-based
Table 2 numbers to <6e-4). Uno's C uses the same fold-specific-tau IPCW routine
as reconcile_clinical_arm.py.

OUTPUT
------
  results/gene_clinical_arm_loco.csv     one row per (held_out_cohort, gene_set):
    gene-only, clinical-only (audited, per-cohort-available) and gene+clinical
    Harrell/Uno C, and both deltas.
  results/gene_clinical_arm_summary.json mean-over-cohorts summary per gene set.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nested_core as nc
from nested_core import OS6, fit_ridge_cox, cindex
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

ALPHA = 100.0
RESULTS = "results"
HARM = ["age_years", "grade3", "node_pos", "size_gt20", "er_pos", "pr_pos"]


def log(m):
    print(m, flush=True)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def harmonise_clin(coh, s):
    """Audited harmonisation -- verbatim from reconcile_clinical_arm.py, kept
    identical so this arm is on the same footing as clinical_arm_reconciled.csv."""
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


def zscore(v):
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


def main():
    gene_sets = json.load(open(os.path.join(RESULTS, "gene_sets.json")))
    gs_names = list(gene_sets.keys())

    store = nc.load_all(OS6, verbose=True)
    H, AVAIL = {}, {}
    for coh in OS6:
        s = nc.load_cohort(coh)[3]
        h = harmonise_clin(coh, s)
        H[coh] = h
        AVAIL[coh] = {c: usable(h[c], len(h)) for c in HARM}
        log("%-18s clinical usable: %s" % (coh, ",".join(c for c in HARM if AVAIL[coh][c])))

    def clin_matrix(coh, covs):
        h = H[coh]
        cols = [zscore(h[c].values) for c in covs]
        return (np.column_stack(cols) if cols else None)

    rows = []
    for held in OS6:
        covs = [c for c in HARM if AVAIL[held][c]]
        tr = [c for c in OS6 if c != held]

        Xs_c, ts, es = [], [], []
        for c in tr:
            M = clin_matrix(c, covs)
            if M is None:
                continue
            _, t, ev, _ = store[c]
            Xs_c.append(M); ts.append(t); es.append(ev)
        Xtr_clin = np.vstack(Xs_c)
        ttr = np.concatenate(ts); etr = np.concatenate(es).astype(np.int32)
        Xte_clin = clin_matrix(held, covs)
        _, tte, ete, _ = store[held]
        ete = np.asarray(ete, np.int32)

        # clinical-only reference (audited, per-cohort-available -- same design
        # as clinical_arm_reconciled.csv)
        b_c = fit_ridge_cox(Xtr_clin, ttr, etr, alpha=ALPHA)
        risk_c = Xte_clin @ b_c
        hc_c = float(cindex(risk_c, tte, ete))
        uc_c, tau_c = uno_c(ttr, etr, tte, ete, risk_c)

        for gs in gs_names:
            nominal = gene_sets[gs]["genes"] if isinstance(gene_sets[gs], dict) else gene_sets[gs]
            Xte_all = store[held][0]
            avail = [g for g in nominal
                     if g in Xte_all.columns and all(g in store[c][0].columns for c in tr)]
            if len(avail) == 0:
                continue

            Xtr_gene = np.vstack([store[c][0][avail].values for c in tr])
            Xte_gene = Xte_all[avail].values

            # gene-only reference, identical alpha/learner as the combined arm
            b_g = fit_ridge_cox(Xtr_gene, ttr, etr, alpha=ALPHA)
            risk_g = Xte_gene @ b_g
            hc_g = float(cindex(risk_g, tte, ete))
            uc_g, _ = uno_c(ttr, etr, tte, ete, risk_g)

            # combined gene+clinical
            Xtr = np.column_stack([Xtr_gene, Xtr_clin])
            Xte = np.column_stack([Xte_gene, Xte_clin])
            b = fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)
            risk = Xte @ b
            hc = float(cindex(risk, tte, ete))
            uc, tau_used = uno_c(ttr, etr, tte, ete, risk)
            lo, hi = boot_ci(risk, tte, ete)

            rows.append(dict(
                held_out_cohort=held, gene_set=gs,
                covariates="|".join(covs), n_clinical_used=len(covs),
                n_genes_used=len(avail), n_genes_nominal=len(nominal),
                harrell_c_gene=hc_g, uno_c_gene=uc_g,
                harrell_c_clinical=hc_c, uno_c_clinical=uc_c,
                harrell_c_gene_clinical=hc, uno_c_gene_clinical=uc,
                ci_lo_gene_clinical=lo, ci_hi_gene_clinical=hi,
                tau_months=tau_used,
                delta_harrell_vs_gene=round(hc - hc_g, 4),
                delta_harrell_vs_clinical=round(hc - hc_c, 4),
                delta_uno_vs_gene=round(uc - uc_g, 4) if np.isfinite(uc) and np.isfinite(uc_g) else None,
                delta_uno_vs_clinical=round(uc - uc_c, 4) if np.isfinite(uc) and np.isfinite(uc_c) else None,
                n_test=int(len(tte)), events_test=int(ete.sum()),
                n_train=int(Xtr.shape[0]), events_train=int(etr.sum()),
            ))
        log("held-out %-18s covs=%d clinHarrell=%.4f done" % (held, len(covs), hc_c))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "gene_clinical_arm_loco.csv"), index=False)
    log("wrote results/gene_clinical_arm_loco.csv rows=%d" % len(out))

    summ_rows = []
    for gs, d in out.groupby("gene_set"):
        summ_rows.append(dict(
            gene_set=gs, n_cohorts=len(d),
            harrell_c_gene_mean=round(float(d.harrell_c_gene.mean()), 4),
            harrell_c_clinical_mean=round(float(d.harrell_c_clinical.mean()), 4),
            harrell_c_gene_clinical_mean=round(float(d.harrell_c_gene_clinical.mean()), 4),
            uno_c_gene_mean=round(float(d.uno_c_gene.mean()), 4),
            uno_c_clinical_mean=round(float(d.uno_c_clinical.mean()), 4),
            uno_c_gene_clinical_mean=round(float(d.uno_c_gene_clinical.mean()), 4),
            mean_delta_harrell_vs_gene=round(float(d.delta_harrell_vs_gene.mean()), 4),
            mean_delta_harrell_vs_clinical=round(float(d.delta_harrell_vs_clinical.mean()), 4),
            cohorts_where_combined_beats_both=int(((d.delta_harrell_vs_gene > 0) & (d.delta_harrell_vs_clinical > 0)).sum()),
        ))
    summary = {"design": "LOCO over OS6, ridge Cox alpha=100, audited "
                          "per-cohort-available clinical covariates "
                          "(reconcile_clinical_arm.py rule); gene-only and "
                          "clinical-only recomputed identically alongside the "
                          "combined arm for an exact paired comparison",
               "per_gene_set": summ_rows}
    json.dump(summary, open(os.path.join(RESULTS, "gene_clinical_arm_summary.json"), "w"),
              indent=1, default=float)
    log("wrote results/gene_clinical_arm_summary.json")
    print(pd.DataFrame(summ_rows).sort_values("mean_delta_harrell_vs_clinical", ascending=False)
          .to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
