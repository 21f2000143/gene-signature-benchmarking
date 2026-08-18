
"""Harmonise breast-cancer survival cohorts into a common schema.

Writes, per cohort, into $OUT:
  <cohort>_surv.parquet  : sample, time_months, event(0/1), endpoint, clinical covariates
  <cohort>_expr.parquet  : samples x genes, within-cohort per-gene z-scored float32
  qc_summary.csv         : one row per cohort with n, events, follow-up, join diagnostics
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd


DATA = "/mnt/kedargouri/sachin/projects/oncogenic-signaling-pathways/dataset"
OUT = os.environ.get("OUT", "./harmonised")
os.makedirs(OUT, exist_ok=True)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ── cohort registry ────────────────────────────────────────────────────────
# expr: path; clin: path; sample_col; time_col; event_col; endpoint; scale:
#   'zscored'  already z-scored vs diploid  -> re-z-score (idempotent-ish, harmless)
#   'log2'     already log2                 -> z-score only
#   'linear'   TPM/RSEM/intensity           -> log2(x+1) then z-score
COHORTS = {
 "TCGA": dict(
   expr="tcga_cbio_hiseq/corrected_data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.tsv",
   clin="tcga_cbio_hiseq/brca_tcga_clinical_data.tsv",
   sample_col="Sample ID", time_col="Overall Survival (Months)",
   event_col="Overall Survival Status", endpoint="OS", scale="zscored", platform="RNA-seq (RSEM)",
   cov=dict(age="Diagnosis Age", grade=None, size="Longest Dimension",
            node="Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code",
            stage="Neoplasm Disease Stage American Joint Committee on Cancer Code",
            er="ER Status By IHC", pr="PR status by ihc", subtype=None)),
 "METABRIC": dict(
   expr="metabric/corrected_data_mrna_illumina_microarray_zscores_ref_diploid_samples.tsv",
   clin="metabric/brca_metabric_clinical_data.tsv",
   sample_col="Sample ID", time_col="Overall Survival (Months)",
   event_col="Overall Survival Status", endpoint="OS", scale="zscored", platform="Microarray (Illumina)",
   cov=dict(age="Age at Diagnosis", grade="Neoplasm Histologic Grade", size="Tumor Size",
            node="Lymph nodes examined positive", stage="Tumor Stage",
            er="ER Status", pr="PR Status", subtype="Pam50 + Claudin-low subtype")),
 "SCANB_GSE96058": dict(
   expr="sweden/cohort1/corrected_GSE96058_gene_expression_3273_samples_and_136_replicates_transformed_original.tsv",
   clin="sweden/cohort1/GSE81540_parsed_clinical_features.tsv",
   sample_col="Sample_title", time_col="overall survival days",
   event_col="overall survival event", endpoint="OS", scale="log2", platform="RNA-seq (HiSeq/NextSeq)",
   cov=dict(age="age at diagnosis", grade="nhg", size="tumor size",
            node="lymph node status", stage=None,
            er="er status", pr="pgr status", subtype=None)),
 "SCANB_GSE202203": dict(
   expr="sweden/cohort2/corrected_GSE202203_TPM_Raw_gene_3207.tsv",
   clin="sweden/cohort2/GSE202203-GPL11154_clinical_features.tsv",
   sample_col="Sample_title", time_col="overall survival days",
   event_col="overall survival event", endpoint="OS", scale="linear", platform="RNA-seq (TPM)",
   cov=dict(age="age at diagnosis", grade="nhg", size="tumor size",
            node="lymph node status", stage="clinical groups",
            er="er status", pr="pgr status", subtype="pam50 subtype")),
 # ── secondary: DMFS / DFS endpoints ──
 "GSE6532": dict(
   expr="GEO/GSE6532/corrected_merged_gene_expression.tsv",
   clin="GEO/GSE6532/GSE6532_parsed_clinical_features.tsv",
   sample_col="Sample_title", time_col="t.dmfs", event_col="e.dmfs",
   endpoint="DMFS", scale="log2", platform="Affymetrix U133A/B",
   cov=dict(age="age", grade="grade", size="size", node="node", stage=None,
            er="er", pr="pgr", subtype=None)),
 "GSE11121": dict(
   expr="GEO/GSE11121/corrected_HG_U133A_gene_expression.tsv",
   clin="GEO/GSE11121/GSE11121_parsed_clinical_features.tsv",
   sample_col="Sample_title", time_col="t.dmfs", event_col="e.dmfs",
   endpoint="DMFS", scale="log2", platform="Affymetrix U133A",
   cov=dict(age=None, grade="grade", size="size_in_cm", node="node", stage=None,
            er=None, pr=None, subtype=None)),
 "GSE21653": dict(
   expr="GEO/GSE21653/corrected_HG_U133_Plus_2_gene_expression.tsv",
   clin="GEO/GSE21653/GSE21653_parsed_clinical_features.tsv",
   sample_col="Sample_title", time_col="dfs time (months)", event_col="dfs evt",
   endpoint="DFS", scale="log2", platform="Affymetrix U133 Plus 2.0",
   cov=dict(age="age at diagnosis", grade="sbr grade", size="pt", node="pn", stage=None,
            er="er ihc", pr="pr ihc", subtype="molecular subtype")),
 "GSE20711": dict(
   expr="GEO/GSE20711/corrected_HG_U133_Plus_2_gene_expression.tsv",
   clin="GEO/GSE20711/GSE20711_parsed_clinical_features.tsv",
   sample_col="Sample_title", time_col="t.os", event_col="e.os",
   endpoint="OS", scale="log2", platform="Affymetrix U133 Plus 2.0",
   cov=dict(age="age (bin)", grade="grade", size="size (bin)", node="node", stage=None,
            er="er status", pr=None, subtype="subtypeihc")),
 "GSE58812": dict(
   expr="GEO/GSE58812/corrected_HG_U133_Plus_2_gene_expression.tsv",
   clin="GEO/GSE58812/GSE58812_parsed_clinical_features.tsv",
   sample_col="Sample_title", time_col="os (days)", event_col="death",
   endpoint="OS", scale="log2", platform="Affymetrix U133 Plus 2.0",
   cov=dict(age="age at diag", grade=None, size=None, node=None, stage=None,
            er="er-ihc", pr="pr-ihc", subtype=None)),
}

# ── event / time normalisation ─────────────────────────────────────────────
POS = {"1","1.0","deceased","dead","death","yes","y","true","event","recurred",
       "recurred/progressed","progressed","relapse","relapsed","distant metastasis","dm"}
NEG = {"0","0.0","living","alive","no","n","false","censored","no event",
       "diseasefree","disease free","disease-free","norecurrence","no recurrence","0:living"}

def norm_event(s: pd.Series) -> pd.Series:
    def one(v):
        if pd.isna(v): return np.nan
        t = str(v).strip().lower()
        if ":" in t:                       # cBioPortal "1:DECEASED"
            head = t.split(":")[0].strip()
            if head in {"0","1"}: return float(head)
        if t in POS: return 1.0
        if t in NEG: return 0.0
        try:
            f = float(t)
            if f in (0.0, 1.0): return f
        except ValueError:
            pass
        return np.nan
    return s.map(one)

def parse_time(raw: pd.Series) -> tuple[pd.Series, str | None]:
    """Extract a numeric follow-up time, honouring a per-value unit suffix.

    Some parsed GEO clinical tables store times as strings like "8.28 y" or
    "241 d". Pull the leading float out and, if a unit letter is attached to
    the values themselves, return it so it overrides column-name inference.
    """
    s = raw.astype(str).str.strip()
    num = pd.to_numeric(s.str.extract(r"^\s*(-?\d+(?:\.\d+)?)", expand=False), errors="coerce")
    suf = s.str.extract(r"^\s*-?\d+(?:\.\d+)?\s*([A-Za-z]+)", expand=False).str.lower()
    tag = suf.dropna()
    unit = None
    if len(tag) and (tag.value_counts(normalize=True).iloc[0] > 0.8):
        u = tag.value_counts().index[0]
        if u.startswith("y"):            unit = "years"
        elif u.startswith("d"):          unit = "days"
        elif u.startswith("mo") or u == "m": unit = "months"
    return num, unit


def to_months(t: pd.Series, col: str, unit_hint: str | None = None) -> tuple[pd.Series, str]:
    """Convert follow-up time to months. Value suffix > column name > magnitude."""
    if unit_hint == "days":   return t / 30.4375, "days(suffix)"
    if unit_hint == "months": return t.astype(float), "months(suffix)"
    if unit_hint == "years":  return t * 12.0, "years(suffix)"
    lc = col.lower()
    if "day" in lc:      return t / 30.4375, "days"
    if "month" in lc:    return t.astype(float), "months"
    if "year" in lc:     return t * 12.0, "years"
    vals = pd.to_numeric(t, errors="coerce").dropna().values.astype(float)
    if len(vals) == 0: return t.astype(float), "unparseable"
    m = float(np.median(vals))                        # unlabelled t.os / t.dmfs
    if m > 400:  return t / 30.4375, "days(inferred)"
    if m > 30:   return t.astype(float), "months(inferred)"
    return t * 12.0, "years(inferred)"

# ── sample-ID reconciliation: pick the transform pair with best overlap ────
def _tcga_pat(x):  return "-".join(str(x).split("-")[:3])
def _tcga_samp(x): return "-".join(str(x).split("-")[:4])
TRANSFORMS = {
  "identity":  lambda x: str(x).strip(),
  "upper":     lambda x: str(x).strip().upper(),
  "dot2dash":  lambda x: str(x).strip().replace(".", "-").upper(),
  "dash2dot":  lambda x: str(x).strip().replace("-", ".").upper(),
  "tcga_samp": lambda x: _tcga_samp(str(x).strip().replace(".", "-").upper()),
  "tcga_pat":  lambda x: _tcga_pat(str(x).strip().replace(".", "-").upper()),
  "nospace":   lambda x: str(x).strip().replace(" ", "").upper(),
}

def best_join(expr_cols, clin_ids):
    best = (0, "identity", "identity")
    for en, ef in TRANSFORMS.items():
        e = pd.Index([ef(c) for c in expr_cols])
        for cn, cf in TRANSFORMS.items():
            k = pd.Index([cf(c) for c in clin_ids])
            n = len(set(e) & set(k))
            if n > best[0]: best = (n, en, cn)
    return best

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Per-gene z-score across samples. df is genes x samples."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=1)
    sd = sd.where((sd > 0) & sd.notna(), np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


if __name__ == "__main__":
    qc = []
    for name, cfg in COHORTS.items():
        log(f"=== {name}")
        ep = os.path.join(DATA, cfg["expr"]); cp = os.path.join(DATA, cfg["clin"])
        if not (os.path.exists(ep) and os.path.exists(cp)):
            log(f"  SKIP missing files"); continue

        expr = pd.read_csv(ep, sep="\t", index_col="hugo_symbol", low_memory=False)
        expr = expr[~expr.index.duplicated(keep="first")]
        expr = expr.apply(pd.to_numeric, errors="coerce")
        expr = expr.dropna(axis=1, how="all")
        log(f"  expr {expr.shape[0]} genes x {expr.shape[1]} samples")

        clin = pd.read_csv(cp, sep=cfg.get("sep","\t"), low_memory=False)
        sc = cfg["sample_col"]
        if sc not in clin.columns:
            log(f"  SKIP sample_col {sc!r} absent; have {list(clin.columns)[:8]}"); continue

        n_ov, et, ct = best_join(expr.columns, clin[sc])
        log(f"  join: expr[{et}] x clin[{ct}] -> {n_ov} overlapping samples")
        if n_ov < 20:
            log(f"  SKIP overlap too small"); continue
        expr.columns = [TRANSFORMS[et](x) for x in expr.columns]
        clin = clin.assign(_key=[TRANSFORMS[ct](x) for x in clin[sc]])
        # collapse replicates on both sides
        expr = expr.T.groupby(level=0).mean().T
        clin = clin.drop_duplicates(subset="_key", keep="first").set_index("_key")

        # survival
        t_raw, unit_hint = parse_time(clin[cfg["time_col"]])
        tm, unit = to_months(t_raw, cfg["time_col"], cfg.get("time_unit") or unit_hint)
        ev = norm_event(clin[cfg["event_col"]])
        surv = pd.DataFrame({"time_months": tm, "event": ev})
        # clinical covariates, renamed to a common vocabulary
        for k, src in cfg["cov"].items():
            surv[k] = clin[src] if (src and src in clin.columns) else np.nan
        surv["endpoint"] = cfg["endpoint"]; surv["cohort"] = name; surv["platform"] = cfg["platform"]

        keep = surv.index.intersection(expr.columns)
        surv = surv.loc[keep]
        surv = surv[surv["time_months"].notna() & surv["event"].notna() & (surv["time_months"] > 0)]
        keep = surv.index
        log(f"  usable: {len(keep)} samples, {int(surv['event'].sum())} events, unit={unit}")
        if len(keep) < 20:
            log("  SKIP too few usable"); continue

        e = expr[keep]
        if cfg["scale"] == "linear":
            e = np.log2(e.clip(lower=0) + 1.0)
        e = zscore(e)
        e = e.dropna(axis=0, how="any")
        log(f"  expr after z-score/dropna: {e.shape[0]} genes")

        surv.reset_index(names="sample").to_parquet(f"{OUT}/{name}_surv.parquet", index=False)
        e.T.astype(np.float32).to_parquet(f"{OUT}/{name}_expr.parquet")

        fu = surv.loc[surv["event"] == 0, "time_months"]
        qc.append(dict(cohort=name, platform=cfg["platform"], endpoint=cfg["endpoint"],
                    n=len(keep), events=int(surv["event"].sum()),
                    event_rate=round(float(surv["event"].mean()), 4),
                    median_fu_months=round(float(np.nanmedian(fu)) if len(fu) else np.nan, 1),
                    max_fu_months=round(float(surv["time_months"].max()), 1),
                    n_genes=int(e.shape[0]), time_unit=unit,
                    join=f"expr[{et}]xclin[{ct}]", overlap=n_ov))

    pd.DataFrame(qc).to_csv(f"{OUT}/qc_summary.csv", index=False)
    log("QC SUMMARY")
    print(pd.DataFrame(qc).to_string(index=False))
