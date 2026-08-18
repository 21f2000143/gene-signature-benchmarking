"""
run_clinical_arm.py -- REVIEW ITEM 9 (the clinicopathology arm is uneven).

The reviewer notes that "Clinical" does not mean the same thing in every
cohort, because different cohorts carry different covariates.  This script
(1) audits exactly what is available and how it is encoded, and (2) recomputes
the Clinical arm under two definitions so the arm can be read consistently.

WHY A HARMONISED ENCODING IS NEEDED
-----------------------------------
The raw *_surv.parquet columns are NOT on a common scale across cohorts, so a
LOCO clinical model cannot simply row-stack them.  Observed encodings:
  age   : years (TCGA, METABRIC, both SCANB, GSE58812, GSE6532, GSE21653)
          BUT a 0/1 dichotomy in GSE20711, and absent in GSE11121
  grade : 1/2/3 (METABRIC, GSE20711, GSE6532, GSE11121, GSE21653)
          G1/G2/G3 strings (both SCANB); absent in TCGA and GSE58812
  node  : TNM strings 'N0','N1a',... (TCGA); positive-node COUNT (METABRIC);
          'NodeNegative'/'NodePositive' (SCANB_GSE96058);
          'NodeNegative'/'SubMicroMet'/'1to3'/'4toX' (SCANB_GSE202203);
          0/1 (GSE20711, GSE6532, GSE21653); constant 0 (GSE11121); absent GSE58812
  size  : millimetres (METABRIC, both SCANB); centimetres (GSE6532, GSE11121);
          pT1/pT2/pT3 (GSE21653); 0/1 (GSE20711); absent TCGA, GSE58812
  er/pr : 'Positive'/'Negative'/'Indeterminate' (TCGA); 'Positive'/'Negative'
          (METABRIC); 0/1 (SCANB, GSE20711, GSE6532, GSE21653);
          constant 0 in GSE58812 (a TNBC-only cohort -> no contrast)
  stage : roman 'Stage IIA' etc (TCGA); 0-4 numeric (METABRIC).
          *** In SCANB_GSE202203 the column named 'stage' actually contains
          molecular subtype labels ('ERpHER2n','TNBC','ERnHER2p','ERpHER2p',
          'Other').  It is NOT stage and is EXCLUDED from every clinical model
          here.  This is reported as an audit finding. ***

Harmonised covariates constructed (each then z-scored within cohort so pooled
LOCO training is on a common scale, exactly as the gene features are):
  age_years  numeric years; NOT constructed for GSE20711 (dichotomised, so not
             the same variable) or GSE11121 (absent)
  grade3     ordinal 1/2/3
  node_pos   binary, node-positive = 1 (TNM N1+ / count>0 / 'NodePositive' /
             any of SubMicroMet,1to3,4toX / raw 1)
  size_gt20  binary, invasive size > 20 mm  (cm columns x10; pT1 -> 0,
             pT2/pT3 -> 1).  Not constructed for GSE20711 (unlabelled 0/1).
  er_pos     binary ('Positive'/1 -> 1; 'Indeterminate' -> missing)
  pr_pos     binary, same rule
A harmonised covariate counts as USABLE in a cohort when it is non-missing in
>30% of subjects AND has >=2 distinct values (the >30% rule reproduces the
original benchmark's build_X rule verbatim).  A constant covariate (er in
GSE58812, node in GSE11121) is therefore unusable, correctly.

THE TWO CLINICAL DEFINITIONS (LOCO over the six OS cohorts)
  (i)  "per_cohort_available" -- as the paper did: for held-out cohort H use
       every harmonised covariate usable in H.  Training cohorts that lack one
       of those covariates contribute 0 for it (their within-cohort
       standardised mean) plus a binary missing-indicator column, so the pooled
       fit is well defined without discarding cohorts.  The arm therefore means
       something DIFFERENT in each fold -- which is precisely the reviewer's
       complaint, here made explicit.
  (ii) "common_all6" -- restricted to the harmonised covariates usable in ALL
       six OS cohorts, so the arm means the same thing everywhere.
  (iii) "common_5_excl_GSE58812" -- sensitivity: GSE58812 is a 107-sample TNBC
       series carrying age only, and it alone collapses the common set.  This
       variant reports the common set over the other five OS cohorts.

Learner is the pre-specified ridge Cox alpha=100 (nested_core), c-index is
Harrell's, identical to the gene arms, so the numbers are directly comparable
to comparator_coverage_penalty.csv.

For reference the gene arms (Novel5 and each comparator, harmonised feature
set) are recomputed here under the identical call so "does Clinical beat the
signatures" can be answered from one table.

OUTPUTS
  clinical_arm_audit.csv       -- cohort x covariate: raw column present,
                                  encoding, n_missing, usable flags (raw rule
                                  and harmonised rule)
  clinical_arm_recomputed.csv  -- LOCO c-index per held-out cohort per arm
  clinical_arm_summary.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import nested_core as nc
from nested_core import OS6, SEC3, fit_ridge_cox, cindex

ALL9 = OS6 + SEC3
ALPHA = 100.0
NBOOT = 1000
GENE_SETS = json.load(open("results/gene_sets.json"))
HARM_COVS = ["age_years", "grade3", "node_pos", "size_gt20", "er_pos", "pr_pos"]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def harmonise_clin(coh, s):
    """Return DataFrame of harmonised covariates (NaN where not constructible)."""
    out = pd.DataFrame(index=s.index)
    col = {c: s[c] for c in s.columns}

    # age -----------------------------------------------------------------
    if "age" in col:
        a = _num(col["age"])
        # a 0/1 column is a dichotomy, not years -> refuse to call it age_years
        vals = set(np.unique(a.dropna().values)) if a.notna().any() else set()
        if a.notna().any() and (np.nanmax(a.values) > 15) and not vals <= {0.0, 1.0}:
            out["age_years"] = a
        else:
            out["age_years"] = np.nan
    else:
        out["age_years"] = np.nan

    # grade ---------------------------------------------------------------
    if "grade" in col:
        g = col["grade"].astype(str).str.strip().str.upper()
        g = g.str.replace("^G", "", regex=True)
        gg = pd.to_numeric(g, errors="coerce")
        out["grade3"] = gg.where(gg.isin([1, 2, 3]))
    else:
        out["grade3"] = np.nan

    # node ----------------------------------------------------------------
    if "node" in col:
        raw = col["node"]
        rs = raw.astype(str).str.strip()
        num = _num(raw)
        np_ = pd.Series(np.nan, index=s.index)
        if num.notna().sum() > 0.5 * len(s):
            # numeric: either a count (METABRIC) or already 0/1
            np_ = (num > 0).astype(float).where(num.notna())
        else:
            neg = rs.str.upper().str.match(r"^N0") | rs.isin(
                ["NodeNegative", "Node negative", "negative", "0"])
            pos = rs.isin(["NodePositive", "SubMicroMet", "1to3", "4toX"]) | \
                rs.str.upper().str.match(r"^N[1-3]")
            np_ = pd.Series(np.where(pos, 1.0, np.where(neg, 0.0, np.nan)),
                            index=s.index)
        out["node_pos"] = np_
    else:
        out["node_pos"] = np.nan

    # size ----------------------------------------------------------------
    out["size_gt20"] = np.nan
    if "size" in col:
        raw = col["size"]
        rs = raw.astype(str).str.strip().str.upper()
        num = _num(raw)
        if rs.str.match(r"^PT[0-4]").any():
            pt = pd.to_numeric(rs.str.extract(r"^PT([0-4])")[0], errors="coerce")
            out["size_gt20"] = (pt >= 2).astype(float).where(pt.notna())
        elif num.notna().sum() > 0.5 * len(s):
            v = set(np.unique(num.dropna().values))
            if v <= {0.0, 1.0}:
                out["size_gt20"] = np.nan          # unlabelled dichotomy (GSE20711)
            elif np.nanmedian(num.values) < 12:    # centimetres
                out["size_gt20"] = (num * 10 > 20).astype(float).where(num.notna())
            else:                                   # millimetres
                out["size_gt20"] = (num > 20).astype(float).where(num.notna())

    # er / pr --------------------------------------------------------------
    for k, oc in (("er", "er_pos"), ("pr", "pr_pos")):
        out[oc] = np.nan
        if k in col:
            raw = col[k]
            rs = raw.astype(str).str.strip().str.lower()
            num = _num(raw)
            if num.notna().sum() > 0.5 * len(s):
                v = set(np.unique(num.dropna().values))
                if v <= {0.0, 1.0}:
                    out[oc] = num
            else:
                out[oc] = pd.Series(
                    np.where(rs.isin(["positive", "pos", "1"]), 1.0,
                             np.where(rs.isin(["negative", "neg", "0"]), 0.0, np.nan)),
                    index=s.index)
    return out


def usable(series, n):
    v = series.dropna()
    return bool(len(v) > 0.3 * n and v.nunique() >= 2)


def zscore(v):
    v = np.asarray(v, float)
    m = np.nanmean(v)
    sd = np.nanstd(v)
    if not np.isfinite(sd) or sd < 1e-9:
        sd = 1.0
    z = (v - m) / sd
    z[~np.isfinite(z)] = 0.0            # NaN -> cohort mean on the z-scale
    return z


def boot_ci(risk, t, ev, seed=0):
    rng = np.random.default_rng(seed)
    n = len(t); vals = []
    for _ in range(NBOOT):
        i = rng.integers(0, n, n)
        if ev[i].sum() < 2:
            continue
        vals.append(cindex(risk[i], t[i], ev[i]))
    if len(vals) < 50:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------- load
log("loading all nine cohorts")
store = nc.load_all(ALL9)
H = {}
AVAIL = {}
for coh in ALL9:
    X, t, ev, s = store[coh]
    h = harmonise_clin(coh, s)
    H[coh] = h
    AVAIL[coh] = {c: usable(h[c], len(h)) for c in HARM_COVS}
    log("%-18s harmonised-usable: %s" % (coh, ",".join(c for c in HARM_COVS if AVAIL[coh][c])))

# --------------------------------------------------------------- (c) audit
RAWCOLS = ["age", "grade", "size", "stage", "node", "er", "pr", "subtype"]
arows = []
for coh in ALL9:
    X, t, ev, s = store[coh]
    n = len(s)
    for c in RAWCOLS:
        present = c in s.columns
        if present:
            v = s[c]
            as_str = v.astype(str).str.strip()
            blank = as_str.isin(["", "nan", "None", "NA", "NaN"])
            nn = int((v.notna() & ~blank).sum())
            ex = sorted(as_str[v.notna() & ~blank].unique())[:6]
            nd = int(as_str[v.notna() & ~blank].nunique())
        else:
            nn, ex, nd = 0, [], 0
        note = ""
        if coh == "SCANB_GSE202203" and c == "stage":
            note = ("column named 'stage' contains MOLECULAR SUBTYPE labels, "
                    "not stage; excluded from all clinical models")
        if coh == "GSE20711" and c in ("age", "size"):
            note = "encoded as an unlabelled 0/1 dichotomy, not on the native scale"
        if coh == "GSE58812" and c in ("er", "pr"):
            note = "constant (TNBC-only series) -> no contrast, unusable"
        if coh == "GSE11121" and c == "node":
            note = "constant 0 (node-negative-only series) -> no contrast, unusable"
        arows.append(dict(
            cohort=coh, endpoint=("OS" if coh in OS6 else "DMFS/DFS"),
            n_samples=n, raw_column=c, raw_column_present=present,
            n_nonmissing=nn, n_missing=n - nn,
            frac_nonmissing=round(nn / n, 4), n_distinct=nd,
            example_values="|".join(map(str, ex)),
            usable_paper_rule=bool(present and nn > 0.3 * n and nd >= 2),
            note=note))
    for c in HARM_COVS:
        v = H[coh][c]
        nn = int(v.notna().sum())
        arows.append(dict(
            cohort=coh, endpoint=("OS" if coh in OS6 else "DMFS/DFS"),
            n_samples=n, raw_column="HARMONISED:" + c,
            raw_column_present=bool(nn > 0), n_nonmissing=nn, n_missing=n - nn,
            frac_nonmissing=round(nn / n, 4), n_distinct=int(v.dropna().nunique()),
            example_values="|".join(map(str, sorted(v.dropna().unique())[:6])),
            usable_paper_rule=AVAIL[coh][c], note="harmonised covariate"))
audit = pd.DataFrame(arows)
audit.to_csv("clinical_arm_audit.csv", index=False)
log("wrote clinical_arm_audit.csv rows=%d" % len(audit))

COMMON6 = [c for c in HARM_COVS if all(AVAIL[k][c] for k in OS6)]
OS5 = [c for c in OS6 if c != "GSE58812"]
COMMON5 = [c for c in HARM_COVS if all(AVAIL[k][c] for k in OS5)]
log("COMMON over all six OS cohorts: %s" % COMMON6)
log("COMMON over five (excl GSE58812): %s" % COMMON5)


def clin_matrix(coh, covs, with_indicator):
    """z-scored harmonised covariates for one cohort; optional missing flags."""
    h = H[coh]
    cols, names = [], []
    for c in covs:
        cols.append(zscore(h[c].values)); names.append(c)
        if with_indicator:
            cols.append((~h[c].notna().values).astype(float) if not AVAIL[coh][c]
                        else (~h[c].notna().values).astype(float))
            names.append(c + "_missing")
    if not cols:
        return None, []
    return np.column_stack(cols), names


def loco_clinical(held, covs, with_indicator, seed=0, pool=OS6):
    tr = [c for c in pool if c != held]
    Xs, ts, es = [], [], []
    for c in tr:
        M, nm = clin_matrix(c, covs, with_indicator)
        if M is None:
            continue
        X, t, ev, _ = store[c]
        Xs.append(M); ts.append(t); es.append(ev)
    Xtr = np.vstack(Xs); ttr = np.concatenate(ts); etr = np.concatenate(es).astype(np.int32)
    Mte, nm = clin_matrix(held, covs, with_indicator)
    X, tte, ete, _ = store[held]
    b = fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)
    risk = Mte @ b
    c = cindex(risk, tte, ete)
    lo, hi = boot_ci(risk, tte, np.asarray(ete, np.int32), seed=seed)
    return dict(cindex=float(c), ci_lo=lo, ci_hi=hi, n_features=Xtr.shape[1],
                n_train=int(Xtr.shape[0]), events_train=int(etr.sum()),
                n_test=int(len(tte)), events_test=int(np.sum(ete)))


def loco_genes(held, genes, pool=OS6):
    tr = [c for c in pool if c != held]
    Xs, ts, es = [], [], []
    for c in tr:
        X, t, ev, _ = store[c]
        Xs.append(X[genes].values); ts.append(t); es.append(ev)
    Xtr = np.vstack(Xs)
    X, tte, ete, _ = store[held]
    b = fit_ridge_cox(Xtr, np.concatenate(ts), np.concatenate(es).astype(np.int32),
                      alpha=ALPHA)
    risk = X[genes].values @ b
    lo, hi = boot_ci(risk, tte, np.asarray(ete, np.int32))
    return dict(cindex=float(cindex(risk, tte, ete)), ci_lo=lo, ci_hi=hi,
                n_features=len(genes), n_train=int(Xtr.shape[0]),
                events_train=int(np.concatenate(es).sum()),
                n_test=int(len(tte)), events_test=int(np.sum(ete)))


rows = []
for held in OS6:
    percoh = [c for c in HARM_COVS if AVAIL[held][c]]
    variants = [
        ("clinical_per_cohort_available", percoh, True, OS6),
        ("clinical_common_all6", COMMON6, False, OS6),
    ]
    if held in OS5:
        variants.append(("clinical_common_5_excl_GSE58812", COMMON5, False, OS5))
    for name, covs, ind, pool in variants:
        if not covs:
            rows.append(dict(arm=name, held_out_cohort=held, covariates="",
                             n_covariates=0, cindex=np.nan,
                             note="no usable covariate in this cohort"))
            continue
        r = loco_clinical(held, covs, ind, pool=pool)
        r.update(arm=name, held_out_cohort=held, covariates="|".join(covs),
                 n_covariates=len(covs), train_pool="|".join([c for c in pool if c != held]),
                 alpha=ALPHA, note="")
        rows.append(r)
    # gene arms for reference, harmonised feature space
    gc = {c: set(store[c][0].columns) for c in OS6}
    for gsn in ["Novel5", "PAM50", "OncotypeDX21", "GGI", "MammaPrint70",
                "BuffaHypoxia", "CNetCox6", "Anchor4"]:
        genes = sorted(set(GENE_SETS[gsn]["genes"]) & set.intersection(*[gc[c] for c in OS6]))
        r = loco_genes(held, genes)
        r.update(arm="gene:" + gsn, held_out_cohort=held,
                 covariates="|".join(genes), n_covariates=len(genes),
                 train_pool="|".join([c for c in OS6 if c != held]), alpha=ALPHA,
                 note="harmonised gene intersection, for comparison")
        rows.append(r)
    log("held-out %s done" % held)

rec = pd.DataFrame(rows)
rec.to_csv("clinical_arm_recomputed.csv", index=False)
log("wrote clinical_arm_recomputed.csv rows=%d" % len(rec))

m = rec.groupby("arm")["cindex"].mean().sort_values(ascending=False)
summ = {
    "harmonised_covariates_considered": HARM_COVS,
    "usable_by_cohort": {k: [c for c in HARM_COVS if AVAIL[k][c]] for k in ALL9},
    "common_all6_OS": COMMON6,
    "common_5_excl_GSE58812": COMMON5,
    "common_all9": [c for c in HARM_COVS if all(AVAIL[k][c] for k in ALL9)],
    "mean_loco_cindex_by_arm": {k: round(float(v), 4) for k, v in m.items()},
    "per_cohort_covariate_sets": {h: [c for c in HARM_COVS if AVAIL[h][c]] for h in OS6},
    "audit_flags": [
        "SCANB_GSE202203 'stage' column holds molecular subtype labels, not stage",
        "GSE20711 age and size are unlabelled 0/1 dichotomies",
        "GSE58812 carries age only among OS cohorts (er/pr constant, TNBC series)",
        "GSE11121 node is constant 0 and age is absent",
        "TCGA has no grade and no size column",
    ],
}
json.dump(summ, open("clinical_arm_summary.json", "w"), indent=1, default=float)
log("wrote clinical_arm_summary.json")
print(rec.pivot_table(index="arm", columns="held_out_cohort", values="cindex").round(4).to_string(), flush=True)
