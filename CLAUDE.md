# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is the analysis/manuscript pipeline for a breast-cancer prognostic gene-signature
study (the "Novel5" five-gene panel, validated against comparator signatures across 9
public cohorts). It is not a software package: there is no build system, no test suite,
and no package manifest. It is a flat directory of Python analysis scripts, their CSV/JSON
outputs (`results/`), figure-generating scripts, figures (`media/`), and the LaTeX
manuscript sources for two target journals (`paper/`).

All numerics were originally computed on a remote host (`ssh:lr`, scikit-survival 0.28.0,
scikit-learn 1.9.0); `seed = 20260725` is used throughout for reproducibility. Some
scripts note they must be run there rather than locally.

## Data dependency

Harmonised per-cohort parquet files are expected at
`harmonised/<COHORT>_expr.parquet` and `<COHORT>_surv.parquet`
(see `DATA` in `nested_core.py` / `probe_cohorts.py`). These are produced by
`harmonise.py`, whose raw inputs live outside this repo
(`/mnt/kedargouri/sachin/projects/oncogenic-signaling-pathways/dataset`, set via `DATA`
there) and are written to `$OUT` (default `./harmonised`). Without this scratch
directory populated, none of the modeling/analysis scripts can run.

## Running scripts

There is no build/lint/test tooling — everything is `python3 <script>.py`, run from the
repo root, in write-results-then-consume order. There is no single entry point.

- To regenerate a results CSV: run the corresponding `run_*.py` / analysis script (e.g.
  `python3 run_nested_selection.py`) — each writes into `results/`.
- To regenerate LaTeX tables from `results/`: `python3 make_tables.py` (writes
  `paper/tabN_*.tex`; every value is read from `results/`, nothing is hand-typed).
- To regenerate figures: run the relevant `make_fig_*.py` script (writes into `media/`).
- To check the manuscript prose against the result files it cites:
  `python3 verify_manuscript_numbers.py` (run from repo root; expects `results/` and
  `paper/` to exist). This exists because prose and tables have previously drifted apart
  when two scripts computed the "same" quantity under different rules — treat any
  failure here as a real bug, not a rounding issue.
- To compile a manuscript PDF (from `paper/`):
  `pdflatex <article> && bibtex <article> && pdflatex <article> && pdflatex <article>`
  (two target journals: `cmpb_article.tex` and `bmc_article.tex`, both `\input`-ing the
  shared `0N_*.tex` section files, `tab*.tex` tables, and figures in `media/`).

## Core architecture

**`nested_core.py`** is the shared numerical core imported by most analysis scripts. It
implements, from scratch (numba-JIT, validated against scikit-survival in
`validate_against_sksurv()`):
- ridge-penalised Cox partial likelihood (Breslow ties) via Newton-Raphson
- Harrell's concordance index
- `load_cohort` / `load_all`: reads a cohort's harmonised parquet pair, z-scores
  expression **within the cohort** (never across cohorts — this is a hard invariant of
  the whole pipeline, see below), and aligns expression/survival on sample ID
- the fully-nested re-selection protocol for the five-gene panel: per-fold candidate
  pool (top ~120 genes by univariate association in training cohorts only) → greedy
  forward addition scored by held-out c-index → cross-fold consensus. This exists
  specifically to answer a circularity concern (the published panel was originally
  selected using the same LOCO c-index later used to evaluate it).

Gene sets (`ANCHOR4`, `NOVEL5`, and comparator panels in `results/gene_sets.json`:
PAM50, MammaPrint70, OncotypeDX21, GGI, BuffaHypoxia, CNetCox6, Anchor4,
Novel5_plus_Anchor4) and the two cohort groups `OS6` (overall-survival: TCGA, METABRIC,
SCANB_GSE96058, SCANB_GSE202203, GSE20711, GSE58812) and `SEC3` (DMFS/DFS: GSE6532,
GSE11121, GSE21653) are defined once in `nested_core.py` and reused everywhere —
`ORDER`/cohort lists in other scripts should match this canonical order.

**`harmonise.py`** builds the harmonised parquet inputs from raw per-cohort
expression/clinical files (a `COHORTS` registry dict keyed by cohort name, each entry
declaring source paths, sample/time/event column names, endpoint, and a `scale` mode —
`zscored`/`log2`/`linear` — controlling how raw expression is transformed before the
per-cohort z-score). `clin_harmonise.py` / `reconcile_clinical_arm.py` handle the
heterogeneously-coded clinical covariates separately (see below).

**Evaluation design** (see `LOCO_METHODS.md` for full detail — read it before touching
any LOCO/validation script, it documents non-obvious methodological decisions and their
rationale):
- PRIMARY validation is leave-one-cohort-out (LOCO) across the 6 OS cohorts: train
  pooled on the other 5, test on the held-out one, with **cohort-grouped** inner CV for
  hyperparameter selection (whole cohorts kept together in folds) so the inner loop
  mimics cross-cohort transfer rather than within-cohort fit.
- SECONDARY is the same LOCO logic restricted to the 3 DMFS/DFS cohorts — reported
  separately, never pooled with OS results.
- TRANSFER (train on all 6 OS, test on each DMFS/DFS cohort) is endpoint transfer, not
  validation, and is explicitly not comparable to the primary table.
- Metrics: Harrell's C is primary; Uno's IPCW C is secondary and diverges from Harrell's
  when a held-out cohort's censoring pattern differs sharply from the training pool
  (notably METABRIC) — don't treat Uno's C as a drop-in substitute there.
- Comparator gene panels are evaluated GENE-ONLY (no clinical covariates mixed in),
  because clinical covariates are coded inconsistently across the 9 cohorts and
  including them would confound the comparison. `n_genes_used` vs `n_genes_requested`
  in output tables records per-cohort gene coverage gaps (several comparator panels are
  missing genes in specific cohorts/platforms — see the "Coverage caveats" section of
  `LOCO_METHODS.md` before quoting any coverage-sensitive number, e.g. Novel5 collapses
  to a 2-gene panel throughout the SECONDARY track).

**Naming conventions across scripts**:
- `run_*.py` — analysis pipelines that write to `results/`.
- `make_fig_*.py` — figure generation, reads `results/`, writes `media/`.
- `make_tables.py` — the single script that emits all `paper/tabN_*.tex` tables.
- `validate_*.py` / `probe_*.py` — one-off checks/reconnaissance, not part of the main
  pipeline; several write their own `*.json` audit trail into `results/` documenting
  what they found (e.g. `probe_env.json`, `probe_cohorts.json`).
- Scripts’ module docstrings routinely state which numbered reviewer/review item they
  address and why — read the docstring before modifying a script, since the rationale
  for a specific numeric choice (thresholds, regularization ranges, tolerance bands) is
  usually only recorded there or in `LOCO_METHODS.md`, not in the code itself.

## Invariants to preserve when editing analysis code

- Expression is z-scored **within each cohort**, never pooled/rescaled across cohorts —
  this is load-bearing for the no-leakage argument throughout the manuscript.
- Hyperparameter selection inside LOCO must stay cohort-grouped (never sample-level CV)
  or it silently reintroduces cross-cohort leakage.
- `results/*.csv` and `results/*.json` are the single source of truth for every number
  in `paper/`; `make_tables.py` and `verify_manuscript_numbers.py` exist to keep it that
  way — if you change a results file's schema, update both.
- Two manuscript targets (`cmpb_article.tex`, `bmc_article.tex`) share the same
  `paper/0N_*.tex` section files and tables; a wording/numeric fix normally needs to
  stay consistent across both compiled articles.
