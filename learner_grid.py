"""
learner_grid.py -- full learner x gene-set x held-out-cohort LOCO grid (review item 5).

REVIEW ITEM ADDRESSED
  Item 5: the submitted manuscript reports, for each cohort x gene-set cell, the BEST
  c-index over four learners. Selecting the learner by the held-out c-index is tuning on
  the evaluation data and inflates every cell. The agreed fix is to report a single
  PRE-SPECIFIED learner (ridge Cox, alpha=100) as the primary analysis and to publish the
  complete learner grid as a sensitivity analysis, together with an explicit test of
  whether the gene-set ranking is learner-dependent. This script produces that grid.

DESIGN (identical for every cell; nothing is chosen using held-out data)
  Outer design      leave-one-cohort-out (LOCO) over the six overall-survival cohorts.
                    Train on the pooled remaining five cohorts, evaluate ONCE on the
                    held-out cohort. No held-out sample influences any fitting,
                    hyperparameter or feature decision.
  Standardisation   every gene is z-scored WITHIN each cohort before pooling, via
                    nested_core.load_cohort (required: the SCANB_GSE96058 copy on this
                    host is stored unstandardised).
  Feature set       for each fold, the panel genes present in ALL cohorts of that fold
                    (five training + one held-out), after legacy-symbol aliasing.
                    Reported as n_genes_used.
  Metric            Harrell's C, computed for every learner by the single numba
                    implementation nested_core.cindex (ties in risk score = 0.5), so the
                    four learners are compared on an identical metric. Higher predicted
                    value = higher risk for all four learners.

LEARNER GRID
  CoxPH_ridge   PRE-SPECIFIED PRIMARY. Ridge-penalised Cox partial likelihood, Breslow
                ties, alpha = 100 fixed a priori (nested_core.fit_ridge_cox). No tuning.
  Coxnet        sksurv.linear_model.CoxnetSurvivalAnalysis, elastic net, l1_ratio = 0.5.
                The penalty path (n_alphas = 30, alpha_min_ratio = 0.001) is derived from
                the TRAINING cohorts only. The penalty is selected by INNER
                leave-one-cohort-out cross-validation ACROSS THE FIVE TRAINING COHORTS
                ONLY: for each candidate alpha, the mean inner-held-out-cohort C over the
                five inner folds is computed, and the alpha maximising it is refitted on
                all five training cohorts. The outer held-out cohort is never touched
                during selection.
  RSF           sksurv.ensemble.RandomSurvivalForest, 300 trees, min_samples_leaf = 15,
                max_features = "sqrt", bootstrap, random_state = 0, n_jobs <= 8.
                Fixed a priori; not tuned.
  GBSA          sksurv.ensemble.GradientBoostingSurvivalAnalysis, Cox partial-likelihood
                loss, 300 stages, learning_rate = 0.05, max_depth = 3, subsample = 0.8,
                random_state = 0. Fixed a priori; not tuned.
  Only Coxnet has a tuned hyperparameter, and it is tuned strictly inside the training
  cohorts. The other three are fixed before seeing any data.

GENE SETS (10)
  Novel5, Anchor4, Novel5_plus_Anchor4, PAM50, OncotypeDX21, GGI, MammaPrint70,
  BuffaHypoxia, CNetCox6 -- memberships from gene_sets.json.
  Clinical -- NOT a gene set and NOT in gene_sets.json; constructed here from the
  clinical columns of the survival tables as the standard prognostic covariate model.
  Its construction is fully specified in build_clinical() below and is deliberately
  conservative, because covariate availability is very uneven across these cohorts.

RANKING QUESTION ANSWERED
  For each learner, gene sets are ranked by mean LOCO C over the six cohorts (rank 1 =
  highest mean C). Spearman rho is reported between the ridge ranking and each other
  learner's ranking, and between the ridge ranking and the "best-of-four" ranking, where
  best-of-four takes the per-cell maximum over the four learners (the flawed procedure
  criticised in item 5) before averaging over cohorts.

OUTPUTS
  learner_grid_full.csv     held_out_cohort, gene_set, learner, cindex, n_genes_used,
                            hyperparams_note (+ n_train, n_test, events_test, fit_seconds)
  learner_grid_summary.csv  mean C per gene_set x learner over the six cohorts, with the
                            rank of each gene set under each learner, plus best-of-four
  learner_grid_ranking.json Spearman correlations, per-learner rankings, Novel5 means,
                            clinical-covariate audit, gene-mapping audit, failures
  Rows are appended to learner_grid_full.csv as they complete so partial results survive
  an interrupted run.
"""
import os
import sys
import json
import time
import warnings

# Must precede the numpy/sksurv imports. Cell-level parallelism means each worker's BLAS
# must be single-threaded, otherwise N_WORKERS x BLAS_THREADS oversubscribes the shared
# host. Total CPU concurrency is then exactly N_WORKERS.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nested_core as nc

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

OS6 = nc.OS6
ALPHA_RIDGE = 100.0
N_JOBS = int(os.environ.get("N_JOBS", "8"))
# Parallelism lives at the CELL level (see the worker pool in main): each cell runs
# single-threaded so that total concurrency equals the pool size and stays within the
# shared host's thread budget. RSF is the only learner exposing an inner n_jobs.
RSF_N_JOBS = int(os.environ.get("RSF_N_JOBS", "1"))
RSF_MAX_SAMPLES = float(os.environ.get("RSF_MAX_SAMPLES", "0.5"))
# Sensitivity check (referee item #11): sqrt(p) rounds to 2 features for the
# 4-5-gene panels, which may be why RSF reorders the small gene sets below the
# large ones. RSF_MAX_FEATURES lets that one hyperparameter be swept
# (e.g. to "1.0" / None, i.e. every feature considered at every split) without
# touching anything else in the fit; default reproduces the primary grid exactly.
_rmf = os.environ.get("RSF_MAX_FEATURES", "sqrt")
RSF_MAX_FEATURES = None if _rmf in ("None", "none", "1.0") else _rmf
SEED = 0

# ---------------------------------------------------------------- legacy symbol aliases
# Applied only when the primary symbol is absent from a cohort matrix. Aliases that would
# duplicate a symbol already in the same gene set are still allowed here but de-duplicated
# after mapping, so a set can never gain a column twice.
ALIASES = {
    # MammaPrint 70 legacy symbols
    "AYTL2": ["LPCAT2"], "C16orf61": ["CMC2", "DC12"], "C20orf46": ["TMEM74B"],
    "C9orf30": ["MSANTD3"], "GPR126": ["ADGRG6"], "HRASLS": ["PLAAT1"],
    "JHDM1D": ["KDM7A"], "LGP2": ["DHX58"], "PECI": ["ECI2"], "OXCT": ["OXCT1"],
    "QSCN6L1": ["QSOX2"], "ZNF533": ["ZNF385B"], "CFFM4": ["MS4A7"],
    "PALM2": ["PALM2AKAP2", "PALM2-AKAP2"], "EBF4": ["EBF4"],
    "AA555029_RC": [],           # EST with no current gene symbol; unmappable by design
    # PAM50 legacy symbols
    "CDCA1": ["NUF2"], "KNTC2": ["NDC80"], "ORC6L": ["ORC6"],
    # Oncotype DX
    "CTSV": ["CTSL2"],
    # GGI
    "H2AFZ": ["H2AZ1"],
    # Buffa hypoxia
    "HIG2": ["HILPDA"],
}

# --------------------------------------------------------------------- clinical model
CLIN_VARS = ["age", "grade", "size_gt20mm", "node_positive"]
CLIN_MIN_NONMISSING = 0.50   # a covariate counts as "available" in a cohort at >=50%


def build_clinical(surv, cohort):
    """Harmonise clinical covariates for one cohort into a numeric DataFrame.

    Returns (DataFrame with columns CLIN_VARS, audit dict).

    Harmonisation rules, chosen to be the largest common denominator across cohorts
    whose clinical fields are encoded very differently:

      age             taken as recorded. NOTE: in GSE20711 this field is stored
                      dichotomised (values {0,1}); it is used as recorded and flagged in
                      the audit as 'age_is_binary'.
      grade           histological grade on the 1/2/3 ordinal scale. String codes
                      'G1','G2','G3' are mapped to 1,2,3; numeric 1/2/3 taken as is.
      size_gt20mm     binary indicator of tumour size above 20 mm (the conventional
                      pT1/pT2 boundary), because size is recorded in mm in some cohorts,
                      cm in others, as pT categories in GSE21653 and already dichotomised
                      in GSE20711. Rule per cohort:
                        - strings beginning 'pT': pT1 -> 0, pT2/pT3/pT4 -> 1
                        - numeric taking only {0,1}: used as an existing indicator
                        - numeric with max <= 10: interpreted as cm, threshold > 2.0
                        - otherwise: interpreted as mm, threshold > 20
      node_positive   binary nodal involvement. Rule per cohort:
                        - TNM strings: 'NX' -> missing, codes beginning 'N0' -> 0,
                          all other 'N...' -> 1
                        - 'NodeNegative' -> 0; 'NodePositive','1to3','4toX',
                          'SubMicroMet' -> 1
                        - numeric: positive-node COUNT or 0/1 indicator, > 0 -> 1

      'stage' IS DELIBERATELY EXCLUDED from the clinical model. The field is not
      comparable across cohorts: TCGA stores AJCC strings ('Stage IIB','Stage X'),
      METABRIC a numeric stage, SCANB_GSE202203 stores a RECEPTOR SUBTYPE label
      ('ERpHER2n','TNBC') rather than a stage, and it is absent in SCANB_GSE96058,
      GSE20711 and GSE58812. Pooling these as one variable would be a coding error.

    Missing values are imputed to the within-cohort median of the covariate, then every
    covariate is z-scored within the cohort -- the same within-cohort standardisation the
    protocol applies to genes. A covariate absent (or below CLIN_MIN_NONMISSING) in a
    cohort becomes an all-zero column for that cohort: it contributes no information from
    that cohort to the pooled fit, rather than deleting the cohort or the covariate.
    """
    n = len(surv)
    out = pd.DataFrame(index=surv.index)
    audit = {"cohort": cohort, "available": {}, "notes": []}

    # ---- age
    if "age" in surv.columns:
        a = pd.to_numeric(surv["age"], errors="coerce")
    else:
        a = pd.Series(np.nan, index=surv.index)
    if a.notna().sum() > 0 and set(np.unique(a.dropna().values)) <= {0.0, 1.0}:
        audit["notes"].append("age_is_binary")
    out["age"] = a

    # ---- grade
    if "grade" in surv.columns:
        g = surv["grade"]
        if not pd.api.types.is_numeric_dtype(g):
            g = (g.astype(str).str.strip().str.upper()
                 .str.replace("^G", "", regex=True))
            g = pd.to_numeric(g, errors="coerce")
        else:
            g = pd.to_numeric(g, errors="coerce")
        g = g.where(g.isin([1, 2, 3]))
    else:
        g = pd.Series(np.nan, index=surv.index)
    out["grade"] = g.astype(float)

    # ---- size -> >20mm indicator
    sz = pd.Series(np.nan, index=surv.index, dtype=float)
    if "size" in surv.columns:
        raw = surv["size"]
        if not pd.api.types.is_numeric_dtype(raw):
            st = raw.astype(str).str.strip().str.upper()
            if st.str.startswith("PT").any():
                def _sz(v):
                    if not isinstance(v, str):
                        return np.nan
                    if v in ("NAN", "NONE", "", "<NA>"):
                        return np.nan
                    if v == "PT1":
                        return 0.0
                    return 1.0 if v.startswith("PT") else np.nan
                sz = st.map(_sz)
                audit["notes"].append("size_from_pT_category")
            else:
                sz = pd.to_numeric(raw, errors="coerce")
        else:
            num = pd.to_numeric(raw, errors="coerce")
            vals = set(np.unique(num.dropna().values)) if num.notna().sum() else set()
            if vals and vals <= {0.0, 1.0}:
                sz = num
                audit["notes"].append("size_already_binary_indicator")
            elif num.notna().sum() and float(np.nanmax(num.values)) <= 10.0:
                sz = (num > 2.0).astype(float).where(num.notna())
                audit["notes"].append("size_interpreted_as_cm_threshold_2cm")
            elif num.notna().sum():
                sz = (num > 20.0).astype(float).where(num.notna())
                audit["notes"].append("size_interpreted_as_mm_threshold_20mm")
    out["size_gt20mm"] = sz.astype(float)

    # ---- node -> positive indicator
    nd = pd.Series(np.nan, index=surv.index, dtype=float)
    if "node" in surv.columns:
        raw = surv["node"]
        if not pd.api.types.is_numeric_dtype(raw):
            # Arrow-backed string columns keep nulls as float NaN even after astype(str),
            # so the parser coerces every value itself rather than trusting the dtype.
            st = raw.astype(str).str.strip()
            def _node(v):
                if not isinstance(v, str):
                    if v is None or (isinstance(v, float) and not np.isfinite(v)):
                        return np.nan
                    v = str(v)
                u = v.upper()
                if u in ("NX", "NAN", "NONE", "", "<NA>"):
                    return np.nan
                if u.startswith("N0"):
                    return 0.0
                if u.startswith("N"):
                    if u.startswith("NODENEG"):
                        return 0.0
                    if u.startswith("NODEPOS"):
                        return 1.0
                    return 1.0
                if u in ("1TO3", "4TOX", "SUBMICROMET"):
                    return 1.0
                return np.nan
            nd = st.map(_node)
            audit["notes"].append("node_from_string_codes")
        else:
            num = pd.to_numeric(raw, errors="coerce")
            nd = (num > 0).astype(float).where(num.notna())
            audit["notes"].append("node_from_numeric_count_or_indicator")
    out["node_positive"] = nd.astype(float)

    # ---- availability, imputation, within-cohort z-scoring
    Z = pd.DataFrame(index=surv.index)
    for v in CLIN_VARS:
        col = out[v].astype(float)
        frac = float(col.notna().mean())
        avail = bool(frac >= CLIN_MIN_NONMISSING)
        audit["available"][v] = {"frac_nonmissing": round(frac, 4), "available": avail}
        if not avail:
            Z[v] = 0.0
            continue
        med = float(np.nanmedian(col.values))
        filled = col.fillna(med).values.astype(float)
        sd = float(np.std(filled))
        if not np.isfinite(sd) or sd < 1e-9:
            Z[v] = 0.0
            audit["available"][v]["zero_variance"] = True
        else:
            Z[v] = (filled - float(np.mean(filled))) / sd
    return Z, audit


# ------------------------------------------------------------------------- utilities
def to_sksurv_y(t, ev):
    return np.array([(bool(e), float(x)) for e, x in zip(ev, t)],
                    dtype=[("event", bool), ("time", float)])


def map_panel(panel, available):
    """Map a gene list onto the symbols available in every cohort of the fold."""
    avail = set(available)
    used, mapping, unmapped = [], {}, []
    for g in panel:
        if g in avail:
            if g not in used:
                used.append(g)
            mapping[g] = g
            continue
        hit = None
        for alt in ALIASES.get(g, []):
            if alt in avail:
                hit = alt
                break
        if hit is not None:
            mapping[g] = hit
            if hit not in used:
                used.append(hit)
        else:
            unmapped.append(g)
    return used, mapping, unmapped


def fit_predict(learner, Xtr, ttr, etr, Xte, train_blocks):
    """Fit one learner on the pooled training cohorts, return (risk on test, note)."""
    p = Xtr.shape[1]
    if learner == "CoxPH_ridge":
        beta = nc.fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA_RIDGE)
        return Xte @ beta, "ridge Cox, alpha=100 fixed a priori (pre-specified primary)"

    if learner == "Coxnet":
        ytr = to_sksurv_y(ttr, etr)
        base = CoxnetSurvivalAnalysis(l1_ratio=0.5, n_alphas=30,
                                      alpha_min_ratio=0.001, normalize=False,
                                      max_iter=100000, tol=1e-7, fit_baseline_model=False)
        base.fit(Xtr, ytr)
        grid = np.asarray(base.alphas_, dtype=float)
        # inner LOCO across TRAINING cohorts only
        scores = np.full((len(train_blocks), len(grid)), np.nan)
        for k, (name, idx) in enumerate(train_blocks):
            in_te = idx
            in_tr = np.setdiff1d(np.arange(Xtr.shape[0]), idx, assume_unique=False)
            if len(in_tr) < 20 or int(etr[in_te].sum()) < 3:
                continue
            try:
                m = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=grid, normalize=False,
                                           max_iter=100000, tol=1e-7,
                                           fit_baseline_model=False)
                m.fit(Xtr[in_tr], to_sksurv_y(ttr[in_tr], etr[in_tr]))
            except Exception:
                continue
            fitted = np.asarray(m.alphas_, dtype=float)
            for a_i, a in enumerate(grid):
                if not np.any(np.isclose(fitted, a)):
                    continue
                try:
                    r = m.predict(Xtr[in_te], alpha=float(a))
                except Exception:
                    continue
                scores[k, a_i] = nc.cindex(np.asarray(r, float),
                                           ttr[in_te], etr[in_te].astype(np.int32))
        mean_inner = np.nanmean(scores, axis=0)
        if np.all(np.isnan(mean_inner)):
            best_a = float(grid[len(grid) // 2])
            note = ("elastic net l1_ratio=0.5; inner CV failed, fell back to median "
                    "path alpha=%.5g" % best_a)
        else:
            best_a = float(grid[int(np.nanargmax(mean_inner))])
            note = ("elastic net l1_ratio=0.5; alpha=%.5g selected by inner LOCO CV over "
                    "the %d training cohorts (mean inner C=%.4f); path n_alphas=30, "
                    "alpha_min_ratio=1e-3"
                    % (best_a, len(train_blocks), float(np.nanmax(mean_inner))))
        final = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=grid, normalize=False,
                                       max_iter=100000, tol=1e-7,
                                       fit_baseline_model=False)
        final.fit(Xtr, ytr)
        return np.asarray(final.predict(Xte, alpha=best_a), float), note

    if learner == "RSF":
        # max_samples=0.5 draws each tree on half the pooled training rows. This is a
        # PRE-SPECIFIED cost-control setting applied uniformly to every RSF cell, not a
        # tuned one: it is fixed before any cell is run, identical across all cohorts and
        # gene sets, and never chosen using held-out data. It is disclosed in the
        # hyperparams_note of every RSF row.
        m = RandomSurvivalForest(n_estimators=300, min_samples_leaf=15,
                                 max_features=RSF_MAX_FEATURES, bootstrap=True,
                                 max_samples=RSF_MAX_SAMPLES,
                                 n_jobs=RSF_N_JOBS, random_state=SEED,
                                 low_memory=True)
        m.fit(Xtr, to_sksurv_y(ttr, etr))
        return np.asarray(m.predict(Xte), float), \
            ("random survival forest, 300 trees, min_samples_leaf=15, "
             "max_features=%s, max_samples=%.2f (per-tree subsample; pre-specified "
             "cost control, uniform across all cells, not tuned), seed=0"
             % (RSF_MAX_FEATURES, RSF_MAX_SAMPLES))

    if learner == "GBSA":
        m = GradientBoostingSurvivalAnalysis(loss="coxph", n_estimators=300,
                                             learning_rate=0.05, max_depth=3,
                                             subsample=0.8, random_state=SEED)
        m.fit(Xtr, to_sksurv_y(ttr, etr))
        return np.asarray(m.predict(Xte), float), \
            ("gradient boosted Cox, 300 stages, lr=0.05, max_depth=3, subsample=0.8, "
             "seed=0; fixed a priori")

    raise ValueError(learner)


# ------------------------------------------------------------------------------- main
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results/gene_sets.json")) as f:
        gs_raw = json.load(f)

    print("loading cohorts (z-scored within cohort by nested_core.load_cohort)",
          flush=True)
    store = nc.load_all(OS6, verbose=True)

    clinical, clin_audit = {}, {}
    for coh in OS6:
        Z, aud = build_clinical(store[coh][3], coh)
        clinical[coh] = Z
        clin_audit[coh] = aud
        print("  clinical %-18s available=%s" %
              (coh, [v for v in CLIN_VARS if aud["available"][v]["available"]]),
              flush=True)

    GENE_SETS = ["Novel5", "Anchor4", "Novel5_plus_Anchor4", "PAM50", "OncotypeDX21",
                 "GGI", "MammaPrint70", "BuffaHypoxia", "CNetCox6", "Clinical"]
    LEARNERS = ["CoxPH_ridge", "Coxnet", "GBSA", "RSF"]

    # SMOKE=1 exercises every learner and both gene-set code paths on two held-out
    # cohorts (one large, one small) to verify the API surface and time the run.
    # It never produces reported numbers; the full grid is the reported analysis.
    smoke = os.environ.get("SMOKE", "0") == "1"
    HELD_LIST = list(OS6)
    if smoke:
        GENE_SETS = ["Novel5", "MammaPrint70", "Clinical"]
        HELD_LIST = ["GSE20711", "METABRIC"]
        print("SMOKE MODE: subset only, results not for reporting", flush=True)
    # LEARNERS env var restricts the grid to named learners; used to validate the
    # parallel dispatch cheaply on the fast learners before committing to the full run.
    if os.environ.get("LEARNERS"):
        LEARNERS = [x for x in os.environ["LEARNERS"].split(",") if x]
        print("LEARNER SUBSET: %s" % LEARNERS, flush=True)
    if os.environ.get("GENE_SET_SUBSET") and not smoke:
        GENE_SETS = [x for x in os.environ["GENE_SET_SUBSET"].split(",") if x]
        print("GENE SET SUBSET: %s" % GENE_SETS, flush=True)

    # Cell-level parallelism. Coxnet and GBSA are single-threaded in sksurv, and GBSA
    # costs ~250 s per cell, so running the 60 cells of a learner pass serially would take
    # hours. Cells are independent by construction (each is a separate LOCO fold x gene
    # set fit), so they are dispatched across a bounded worker pool with every inner
    # thread pinned to 1. Total concurrency stays at N_WORKERS <= 8 as required by the
    # shared host. Results are bit-identical to the serial run: every learner has a fixed
    # random_state and no cell depends on another.
    n_workers = int(os.environ.get("N_WORKERS", "6"))

    # SHARDING. The grid is sharded by held-out cohort: one job per cohort, each computing
    # all gene sets x all learners for that cohort only, writing its own output file. Six
    # shards coexist on the host with a small per-shard thread budget. Each cell's row is
    # appended the moment it completes, so a shard killed by a wall-clock limit still
    # yields every cell it finished. Shards are concatenated afterwards.
    # Non-default RSF_MAX_FEATURES writes to a separate file so a sensitivity sweep
    # can never silently overwrite the canonical (sqrt) grid the paper reports.
    tag = "" if _rmf == "sqrt" else ("_mtry%s" % _rmf.replace(".", "p"))
    shard = os.environ.get("SHARD", "")
    if shard:
        HELD_LIST = [shard]
        out_path = "learner_grid_%s%s.csv" % (shard, tag)
        print("SHARD: held-out cohort %s -> %s" % (shard, out_path), flush=True)
    else:
        out_path = "learner_grid_full%s.csv" % tag
    cols = ["held_out_cohort", "gene_set", "learner", "cindex", "n_genes_used",
            "hyperparams_note", "n_train", "n_test", "events_test", "fit_seconds"]

    # RESUME. Random survival forest costs ~500 s per cell, so the full grid exceeds a
    # single job's wall clock. If a partial learner_grid_full.csv is supplied as an input,
    # the cells it already contains are skipped and the new cells are appended, so the
    # final CSV is one coherent table produced by one version of this script. Every cell
    # is deterministic (fixed seeds, no cell depends on another), so a resumed grid is
    # identical to one computed in a single pass.
    prior = pd.DataFrame(columns=cols)
    resume = os.environ.get("RESUME", "1") == "1"
    if resume and os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        prior = prior[prior["cindex"].notna()]
    done_keys = set(zip(prior["learner"], prior["held_out_cohort"], prior["gene_set"]))
    if len(done_keys):
        print("RESUME: %d cells already present, will be kept and skipped"
              % len(done_keys), flush=True)
    prior.to_csv(out_path, index=False)

    mapping_audit, failures = {}, []
    t_start = time.time()

    # ---- prepare every cell once; all four learners reuse the identical matrices ----
    cells = {}
    for held in HELD_LIST:
        train = [c for c in OS6 if c != held]
        genes_all = nc.common_genes(store, train + [held])
        for gsname in GENE_SETS:
            if gsname == "Clinical":
                used = list(CLIN_VARS)
                Xtr_parts, blocks, pos = [], [], 0
                for coh in train:
                    Zc = clinical[coh].values.astype(np.float64)
                    Xtr_parts.append(Zc)
                    blocks.append((coh, np.arange(pos, pos + Zc.shape[0])))
                    pos += Zc.shape[0]
                Xtr = np.vstack(Xtr_parts)
                Xte = clinical[held].values.astype(np.float64)
                avail_here = [v for v in CLIN_VARS
                              if clin_audit[held]["available"][v]["available"]]
                extra = ("clinical covariates %s; available in held-out cohort: %s; "
                         "cohorts lacking a covariate contribute an all-zero column"
                         % ("+".join(CLIN_VARS), "+".join(avail_here) or "none"))
                n_used = len(used)
            else:
                panel = gs_raw[gsname]["genes"]
                used, mp, unmapped = map_panel(panel, genes_all)
                mapping_audit["%s|%s" % (gsname, held)] = {
                    "n_requested": len(panel), "n_used": len(used),
                    "unmapped": unmapped,
                    "aliased": {k: v for k, v in mp.items() if k != v}}
                if len(used) == 0:
                    failures.append({"held_out_cohort": held, "gene_set": gsname,
                                     "learner": "ALL", "reason": "no genes present"})
                    continue
                Xtr_parts, blocks, pos = [], [], 0
                for coh in train:
                    Xc = store[coh][0][used].values.astype(np.float64)
                    Xtr_parts.append(Xc)
                    blocks.append((coh, np.arange(pos, pos + Xc.shape[0])))
                    pos += Xc.shape[0]
                Xtr = np.vstack(Xtr_parts)
                Xte = store[held][0][used].values.astype(np.float64)
                extra = "%d/%d panel symbols matched" % (len(used), len(panel))
                n_used = len(used)

            cells[(held, gsname)] = {
                "Xtr": Xtr, "Xte": Xte, "blocks": blocks, "extra": extra,
                "n_used": n_used,
                "ttr": np.concatenate([store[c][1] for c in train]),
                "etr": np.concatenate([store[c][2] for c in train]).astype(np.int32),
                "tte": store[held][1], "ete": store[held][2].astype(np.int32)}

    print("prepared %d (cohort, gene set) cells; dispatching %d learners over %d workers"
          % (len(cells), len(LEARNERS), n_workers), flush=True)

    # ---------------------------------------------------------- parallel cell execution
    def run_cell(task):
        learner, held, gsname = task
        cell = cells[(held, gsname)]
        t0 = time.time()
        try:
            risk, note = fit_predict(learner, cell["Xtr"], cell["ttr"], cell["etr"],
                                     cell["Xte"], cell["blocks"])
            ci = float(nc.cindex(np.asarray(risk, float), cell["tte"], cell["ete"]))
        except Exception as e:
            return {"ok": False, "learner": learner, "held_out_cohort": held,
                    "gene_set": gsname, "reason": "%s: %s" % (type(e).__name__, e)}
        return {"ok": True, "row": {
            "held_out_cohort": held, "gene_set": gsname, "learner": learner,
            "cindex": round(ci, 6), "n_genes_used": cell["n_used"],
            "hyperparams_note": note + " | " + cell["extra"],
            "n_train": int(cell["Xtr"].shape[0]), "n_test": int(cell["Xte"].shape[0]),
            "events_test": int(cell["ete"].sum()),
            "fit_seconds": round(time.time() - t0, 2)}}

    # heaviest learners first so the tail of the pool is short
    order = [lr for lr in ["RSF", "GBSA", "Coxnet", "CoxPH_ridge"] if lr in LEARNERS]
    tasks = [(lr, held, gs) for lr in order
             for held in HELD_LIST for gs in GENE_SETS
             if (held, gs) in cells and (lr, held, gs) not in done_keys]
    print("%d cells to compute (%d already present)" % (len(tasks), len(done_keys)),
          flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(run_cell, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            if not res["ok"]:
                failures.append({k: res[k] for k in
                                 ("held_out_cohort", "gene_set", "learner", "reason")})
                print("  [%3d/%3d] FAIL %-12s %-20s %-16s %s"
                      % (done, len(tasks), res["learner"], res["gene_set"],
                         res["held_out_cohort"], res["reason"]), flush=True)
                continue
            row = res["row"]
            pd.DataFrame([row], columns=cols).to_csv(out_path, mode="a", header=False,
                                                     index=False)
            print("  [%3d/%3d] %-12s %-20s %-16s C=%.4f p=%-3d %6.1fs  (%.1f min elapsed)"
                  % (done, len(tasks), row["learner"], row["gene_set"],
                     row["held_out_cohort"], row["cindex"], row["n_genes_used"],
                     row["fit_seconds"], (time.time() - t_start) / 60.0), flush=True)

    # -------------------------------------------------------------------- summarise
    if shard:
        # A single shard holds one held-out cohort; the mean-over-cohorts summary and the
        # ranking comparison are only meaningful on the concatenated grid. The shard writes
        # its cells and its audits, and stops here.
        with open("shard_audit_%s.json" % shard, "w") as f:
            json.dump({"shard": shard,
                       "clinical_covariate_audit": clin_audit,
                       "gene_mapping_audit": mapping_audit,
                       "failures": failures,
                       "n_cells_written": int(len(pd.read_csv(out_path))),
                       "runtime_minutes": round((time.time() - t_start) / 60.0, 2)},
                      f, indent=1)
        print("shard %s complete: %d cells, %.1f min"
              % (shard, len(pd.read_csv(out_path)), (time.time() - t_start) / 60.0),
              flush=True)
        return

    full = pd.read_csv(out_path)
    piv = full.pivot_table(index="gene_set", columns="learner", values="cindex",
                           aggfunc="mean")
    piv = piv.reindex(index=GENE_SETS)
    percell_best = (full.groupby(["gene_set", "held_out_cohort"])["cindex"]
                    .max().groupby("gene_set").mean())
    piv["BestOfFour"] = percell_best.reindex(piv.index)

    n_cells = full.pivot_table(index="gene_set", columns="learner", values="cindex",
                               aggfunc="count").reindex(index=GENE_SETS)

    summary = piv.copy()
    summary.columns = ["mean_c_" + c for c in summary.columns]
    for c in list(piv.columns):
        summary["rank_" + c] = piv[c].rank(ascending=False, method="min").astype("Int64")
    for c in [c for c in n_cells.columns]:
        summary["n_cohorts_" + c] = n_cells[c]
    summary = summary.reset_index()
    summary.to_csv("learner_grid_summary.csv", index=False)

    ridge = piv["CoxPH_ridge"]
    spearman = {}
    for c in piv.columns:
        if c == "CoxPH_ridge":
            continue
        ok = ridge.notna() & piv[c].notna()
        rho, pv = spearmanr(ridge[ok].values, piv[c][ok].values)
        spearman[c] = {"spearman_rho": round(float(rho), 4),
                       "p_value": float(pv), "n_gene_sets": int(ok.sum())}

    rank_tbl = pd.DataFrame({c: piv[c].rank(ascending=False, method="min")
                             for c in piv.columns})
    top_by_learner = {c: rank_tbl[c].idxmin() for c in piv.columns}
    rank_changes = {}
    for c in piv.columns:
        if c == "CoxPH_ridge":
            continue
        d = (rank_tbl[c] - rank_tbl["CoxPH_ridge"]).abs()
        rank_changes[c] = {"max_abs_rank_shift": int(d.max()),
                           "n_sets_with_rank_shift": int((d > 0).sum()),
                           "identical_top_set": bool(top_by_learner[c] ==
                                                     top_by_learner["CoxPH_ridge"])}

    all_rho = [spearman[c]["spearman_rho"] for c in spearman]
    depends = bool(min(all_rho) < 0.90 or
                   any(not rank_changes[c]["identical_top_set"] for c in rank_changes))

    novel5_means = {c: (round(float(piv.loc["Novel5", c]), 4)
                        if pd.notna(piv.loc["Novel5", c]) else None)
                    for c in piv.columns}

    ranking = {
        "spearman_vs_ridge": spearman,
        "ranking_depends_on_learner": depends,
        "criterion": "declared learner-dependent if any Spearman rho vs ridge < 0.90 or "
                     "any learner disagrees with ridge on the top-ranked gene set",
        "rank_table": {c: {g: (int(rank_tbl.loc[g, c])
                               if pd.notna(rank_tbl.loc[g, c]) else None)
                           for g in rank_tbl.index} for c in rank_tbl.columns},
        "mean_c_table": {c: {g: (round(float(piv.loc[g, c]), 4)
                                 if pd.notna(piv.loc[g, c]) else None)
                             for g in piv.index} for c in piv.columns},
        "top_gene_set_by_learner": top_by_learner,
        "rank_shift_vs_ridge": rank_changes,
        "novel5_mean_c_per_learner": novel5_means,
        "clinical_covariate_audit": clin_audit,
        "gene_mapping_audit": mapping_audit,
        "failures": failures,
        "n_rows_written": int(len(full)),
        "runtime_minutes": round((time.time() - t_start) / 60.0, 2),
    }
    with open("learner_grid_ranking.json", "w") as f:
        json.dump(ranking, f, indent=1)

    print("\n=== mean LOCO c-index over 6 OS cohorts ===", flush=True)
    print(piv.round(4).to_string(), flush=True)
    print("\n=== Spearman vs ridge ranking ===", flush=True)
    for c, v in spearman.items():
        print("  %-12s rho=%+.4f p=%.4g" % (c, v["spearman_rho"], v["p_value"]),
              flush=True)
    print("\nranking_depends_on_learner:", depends, flush=True)
    print("Novel5 mean C:", novel5_means, flush=True)
    print("failures:", len(failures), flush=True)


if __name__ == "__main__":
    main()
