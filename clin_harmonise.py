"""
clin_harmonise.py -- covariate harmonisation helpers for review item 9.

The function bodies below are VERBATIM the ones used in run_clinical_arm.py,
which produced clinical_arm_audit.csv and clinical_arm_recomputed.csv.  They
are factored out here so run_clinical_supplement.py can reuse the identical
definitions without re-executing that script.  run_clinical_supplement.py
asserts at start-up that this copy is byte-identical to the block in
run_clinical_arm.py, so the two analyses can never drift apart.

See run_clinical_arm.py's module docstring for the full statement of how each
covariate is harmonised and why.
"""
import numpy as np
import pandas as pd

HARM_COVS = ["age_years", "grade3", "node_pos", "size_gt20", "er_pos", "pr_pos"]


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
