
# LOCO external validation — breast-cancer prognostic gene signatures

All numerics computed on the remote host `ssh:lr` (scikit-survival 0.28.0,
scikit-learn 1.9.0). seed = 20260725 everywhere.

## Design

PRIMARY (loco_os.csv): the 6 overall-survival cohorts. For each held-out cohort H,
models are trained on the POOLED remaining 5 OS cohorts and evaluated on H. No
information from H enters training or hyperparameter selection.

SECONDARY (loco_secondary.csv): the same LOCO logic inside the 3 DMFS/DFS cohorts
(GSE6532, GSE11121 = DMFS; GSE21653 = DFS). Reported separately; never pooled with OS.

TRANSFER (cross_endpoint_transfer.csv): trained on all 6 OS cohorts, tested on each
DMFS/DFS cohort. This is endpoint TRANSFER, not validation — the training and test
endpoints differ, so it is not comparable to the primary table.

Yes. The easiest way to understand this is to imagine you have **9 cohorts**, divided into two endpoint groups:

* **6 OS cohorts** → Overall Survival
* **3 DMFS/DFS cohorts** → different survival endpoints

The key idea is **which cohorts are used for training and which cohort is kept completely unseen for testing**.

---

## 1. PRIMARY: `loco_os.csv`

**LOCO = Leave-One-Cohort-Out**

Suppose your 6 OS cohorts are:

| Cohort | Endpoint |
| ------ | -------- |
| A      | OS       |
| B      | OS       |
| C      | OS       |
| D      | OS       |
| E      | OS       |
| F      | OS       |

You perform **6 separate experiments**.

### Experiment 1 — Hold out A

```text
Training:
B + C + D + E + F
          ↓
       ML model
          ↓
       Test on A
```

A is completely unseen during training.

### Experiment 2 — Hold out B

```text
Training:
A + C + D + E + F
          ↓
       ML model
          ↓
       Test on B
```

And so on:

| Test cohort | Training cohorts |
| ----------- | ---------------- |
| A           | B,C,D,E,F        |
| B           | A,C,D,E,F        |
| C           | A,B,D,E,F        |
| D           | A,B,C,E,F        |
| E           | A,B,C,D,F        |
| F           | A,B,C,D,E        |

So the important rule is:

> **Every time, one entire cohort is treated as completely unseen test data.**

This is stronger than randomly splitting patients into train/test because patients from the same cohort can have similar technical and biological characteristics.

### Why "pooled"?

Suppose:

* A = 200 patients
* B = 300
* C = 150
* D = 250
* E = 400
* F = 200

When testing on A, you don't train six separate models.

You combine:

```text
B (300)
C (150)
D (250)
E (400)
F (200)
----------------
Total = 1,300 patients
```

and train **one model** on those 1,300 patients.

Then test it on the 200 patients in A.

---

# 2. SECONDARY: `loco_secondary.csv`

Now you have three different cohorts:

| Cohort   | Endpoint |
| -------- | -------- |
| GSE6532  | DMFS     |
| GSE11121 | DMFS     |
| GSE21653 | DFS      |

You do the **same LOCO procedure**, but only among these three cohorts.

### Hold out GSE6532

```text
Training:
GSE11121 + GSE21653
          ↓
       Model
          ↓
Test: GSE6532
```

### Hold out GSE11121

```text
Training:
GSE6532 + GSE21653
          ↓
       Model
          ↓
Test: GSE11121
```

### Hold out GSE21653

```text
Training:
GSE6532 + GSE11121
          ↓
       Model
          ↓
Test: GSE21653
```

So this is **another independent LOCO analysis**.

Importantly:

> **These 3 cohorts are NOT mixed with the 6 OS cohorts.**

Why?

Because **OS, DMFS, and DFS are different endpoints**.

---

# 3. TRANSFER: `cross_endpoint_transfer.csv`

This is the part that is easiest to confuse with validation.

Here you do something different.

You take **ALL 6 OS cohorts**:

```text
A + B + C + D + E + F
          ↓
       Train model
```

Then you test that model on the three DMFS/DFS cohorts:

```text
                 ┌──→ GSE6532 (DMFS)
6 OS cohorts ────┼──→ GSE11121 (DMFS)
                 └──→ GSE21653 (DFS)
```

For example:

```text
TRAINING
A + B + C + D + E + F
        ↓
   OS prediction model
        ↓
────────────────────────
        ↓
TEST
GSE6532 (DMFS)
```

Notice the fundamental difference:

### Primary

```text
OS → OS
```

Train on 5 OS cohorts → test on the 6th OS cohort.

### Transfer

```text
OS → DMFS/DFS
```

Train on **all OS cohorts** → test on a **different endpoint**.

---

# Why isn't TRANSFER considered validation?

Suppose your model was designed to predict **Overall Survival (OS)**.

You train it:

```text
OS data
   ↓
OS model
```

Then you test it on DMFS:

```text
OS model
   ↓
DMFS data
```

You're asking:

> "Does a model learned from OS data have predictive ability when applied to a different survival endpoint?"

That's an interesting biological/generalization experiment.

But it is **not the same question** as:

> "Can my OS model predict OS in a completely unseen cohort?"

The second question is what the **primary LOCO analysis** answers.

---

# The entire design in one picture

Think of your study as three separate boxes:

```text
                 ┌─────────────────────────────┐
                 │       PRIMARY: OS           │
                 │                             │
                 │  6 OS cohorts               │
                 │                             │
                 │  A B C D E F                │
                 │                             │
                 │  Leave one out              │
                 │                             │
                 │  B+C+D+E+F → A              │
                 │  A+C+D+E+F → B              │
                 │  A+B+D+E+F → C              │
                 │  ...                        │
                 └─────────────────────────────┘


                 ┌─────────────────────────────┐
                 │    SECONDARY: DMFS/DFS      │
                 │                             │
                 │  GSE6532                    │
                 │  GSE11121                   │
                 │  GSE21653                   │
                 │                             │
                 │  Leave one out              │
                 │                             │
                 │  2 cohorts → 1 cohort       │
                 └─────────────────────────────┘


                 ┌─────────────────────────────┐
                 │       TRANSFER              │
                 │                             │
                 │  6 OS cohorts               │
                 │       ↓                     │
                 │    Train model              │
                 │       ↓                     │
                 │  ┌────┴─────┐               │
                 │  ↓          ↓               │
                 │ DMFS       DFS              │
                 │                             │
                 │ GSE6532    GSE21653         │
                 │ GSE11121                    │
                 └─────────────────────────────┘
```

## The simplest way to remember it

| Analysis           | Training           | Testing           | Question                                               |
| ------------------ | ------------------ | ----------------- | ------------------------------------------------------ |
| **PRIMARY LOCO**   | 5 OS cohorts       | 1 OS cohort       | Does the model generalize to a new OS cohort?          |
| **SECONDARY LOCO** | 2 DMFS/DFS cohorts | 1 DMFS/DFS cohort | Does it generalize within the secondary endpoints?     |
| **TRANSFER**       | All 6 OS cohorts   | DMFS/DFS cohorts  | Does an OS-trained model transfer to another endpoint? |

### One critical point

For **PRIMARY**, if GSE96058 is the held-out cohort:

```text
GSE96058
   ❌ training
   ❌ feature selection
   ❌ hyperparameter tuning
   ❌ threshold selection
   ❌ anything that learns from data

          ↓

       FINAL TEST
```

Everything learned by the model must come from the **other 5 cohorts**.

That is what makes the primary result a genuine **external cohort-level validation/generalization test**.


## Scaling / leakage

Expression was already z-scored WITHIN each cohort upstream; no scaler is fitted
across cohorts, and no pooled statistics are computed. Features are restricted to
genes present in EVERY training cohort AND the held-out cohort; `n_genes_used` and
`genes_used` record this per row, alongside `n_genes_requested`. No survival-derived
variable enters the feature matrix.

GENE-ONLY MODELS: clinical covariates (age, grade, size, node, stage, er, pr,
subtype) are heterogeneously coded across these 9 cohorts, so LOCO is restricted to
expression features. Harmonising them would confound the comparison; that restriction
applies to every row of every table here.

## Models and hyperparameter selection

CoxPH ridge (alpha in 0.01-100), Coxnet (l1_ratio 0.9, alpha 0.01-0.5),
RandomSurvivalForest, GradientBoostingSurvivalAnalysis. Hyperparameters are chosen by
COHORT-GROUPED inner cross-validation on the training pool only: whole cohorts are
kept together in folds (5 folds for linear models, 3 for tree ensembles), so the inner
CV mimics the outer cross-cohort transfer rather than rewarding within-cohort fit.
`best_param`, `n_folds_inner` and the inner-CV c-index are recorded per row.

Tree-ensemble grids were deliberately set in a cheap, more strongly regularised range
(RSF: 100 trees, min_samples_leaf 50/100, max_samples 0.5/0.4; GBSA: 100-150 stages,
depth 2). A timing probe on the 9,153-sample pooled training pool measured 484 s for a
single min_samples_leaf=15 / 150-tree RSF fit versus 81 s at leaf=50 and 42 s at
leaf=100; the larger leaves are also the appropriate regularisation for 5-69 features.

## Metrics

`cindex` = Harrell's C (sksurv concordance_index_censored) on the held-out cohort —
the primary metric. `uno_c` = Uno's IPCW C with tau = min(max training time, 95th
percentile of held-out event times); censoring is estimated from the training pool, so
where the held-out cohort's censoring pattern differs sharply from the pool (notably
METABRIC, 1143/1979 events) Uno's C and Harrell's C diverge and Uno's should not be
read as a like-for-like substitute.

`ci_lo`/`ci_hi` = 2.5th/97.5th percentile of a 2000-resample nonparametric bootstrap
resampling the HELD-OUT cohort's patients. It is computed in closed form as
w'Nw / w'Aw over precomputed comparable-pair (A) and concordant-pair (N) matrices,
which is algebraically identical to recomputing the concordance per resample; the
column `c_fast_minus_sksurv` records agreement with sksurv at unit weights
(max |difference| = 3.0e-08 across all rows).

## Pooled summaries

loco_os_pooled.csv gives, per gene set x model, both the sample-size-weighted mean
c-index across the 6 held-out cohorts and the UNWEIGHTED mean, plus SD, min and max.
The two means differ materially: weighting favours signatures that do well on the
large SCANB cohorts, the unweighted mean gives the two ~100-patient cohorts equal
voice. Both are reported; neither alone should be quoted.

## Paired comparison (loco_paired_novel5_vs_comparators.csv)

Overlapping marginal CIs are not a test of difference, so this table bootstraps the
DIFFERENCE in c-index (Novel5 minus comparator) using the SAME resample of the same
held-out patients for both gene sets, under a common model (CoxPH ridge). `favours` is
'Novel5' / 'comparator' only when the 95% CI of the difference excludes zero. These
are 8 comparisons x 6 cohorts with no multiplicity adjustment.

## Coverage caveats (read before quoting any number)

- CNetCox6 is evaluated on 5 of its 6 genes in the OS track: MYC is absent from the
  TCGA matrix, and features must be common to the pool and the held-out cohort.
- Novel5 collapses to 2 of 5 genes (FLT3, P4HA2) throughout the SECONDARY panel:
  GSE6532 and GSE11121 (old U133A) lack CLIC6, SUSD3 and ZIC2, so even for held-out
  GSE21653 — which has all five — the training pool supplies only two. The secondary
  Novel5 numbers are therefore a 2-gene panel and are NOT a test of the 5-gene panel.
- PAM50 46/50, MammaPrint70 48/69, GGI 58/59, BuffaHypoxia 48/52 in the OS track.
- Anchor4 is a deliberate NEGATIVE CONTROL, expected near-random.

## Files

loco_os.csv, loco_os_pooled.csv, loco_secondary.csv, loco_secondary_pooled.csv,
cross_endpoint_transfer.csv, loco_paired_novel5_vs_comparators.csv, loco_risk_scores.csv.
loco_risk_scores.csv holds the per-patient held-out risk score for Novel5 under RSF
(highest weighted mean) for all 9,241 patients in the 6 OS cohorts, with within-cohort
percentile and tertile group — risk scores are NOT calibrated across cohorts, so any
Kaplan-Meier stratification must use the within-cohort columns.
