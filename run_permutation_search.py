"""
run_permutation_search.py -- reviewer item 1.2: permutation calibration of the SEARCH.

The published random-panel null drew panels at random; it therefore calibrates
"is a random five-gene panel this good?" and not "how good does this forward search
look when there is nothing to find?". The latter is what is needed, because a maximum
over ~16,000 candidates x 5 forward steps is optimistically biased even under the null.

Design: permute survival (time, event) pairs WITHIN each cohort -- destroying any
gene-survival association while preserving each cohort's censoring distribution,
sample size and expression correlation structure -- then run the IDENTICAL forward
search, and record
  (a) the inner-LOCO score the search reports for its chosen panel (the number the
      discovery procedure would have published), and
  (b) the honest held-out c-index of that panel on a cohort excluded from its search.
(a) measures search optimism directly. (b) should sit at 0.5.

Usage: python run_permutation_search.py <n_replicates> <seed0> [shard_tag]
       Replicate seeds are seed0 + i*101, so disjoint seed0 values give disjoint
       replicates and shards can be run in parallel and concatenated.
Writes: permutation_search[_<tag>].csv, permutation_search_summary[_<tag>].json
        Rows are appended to the CSV as each replicate finishes, so a shard that is
        killed by a wall-clock limit still yields every replicate it completed.

That's completely understandable. This paragraph is written for researchers who are familiar with survival analysis, feature selection, and statistical validation. Let's unpack it piece by piece in plain English.

## What problem is this trying to solve?

Imagine you have a dataset with **16,000 genes**, and you're trying to find the **best combination of 5 genes** that predicts patient survival.

Your search algorithm tries thousands (or millions) of possibilities and finally says:

> "I found these 5 genes. They give a score of 0.72."

The question is:

> **Is 0.72 really impressive? Or could a search algorithm find something this good just by luck?**

This script tries to answer that question.

---

## The problem with the old test

The comments say:

> The published random-panel null drew panels at random.

This means they previously did something like:

1. Pick 5 random genes.
2. Compute their prediction score.
3. Repeat many times.

Then compare:

```
Random panels:
0.52
0.55
0.57
0.54
```

vs

```
Our chosen panel:
0.72
```

That only answers:

> "Is our selected panel better than a random panel?"

But that's **not the right comparison.**

---

## Why is it wrong?

Suppose I ask you:

> Pick the tallest person from a city of one million people.

Of course you'll find someone 8 feet tall.

Now compare that person against a random citizen.

That comparison is unfair because you searched through **everyone**.

The same thing happens here.

The algorithm searched through

* ~16,000 genes
* many combinations
* 5 forward-selection steps

Eventually it found something that looks great.

Even if there is **no real biological signal**, a huge search can still discover something that looks impressive purely by chance.

This is called **search optimism** or **selection bias**.

---

## What should we compare against instead?

Instead of asking

> "How good is a random panel?"

we ask

> "How good would this SAME SEARCH algorithm perform if there were absolutely no relationship between genes and survival?"

That's a much better question.

---

# How do they create "no relationship"?

They **permute** the survival data.

Original data:

| Patient | Gene Expression | Survival |
| ------- | --------------- | -------- |
| A       | ...             | 2 years  |
| B       | ...             | 5 years  |
| C       | ...             | 1 year   |

After permutation:

| Patient | Gene Expression | Survival |
| ------- | --------------- | -------- |
| A       | ...             | 5 years  |
| B       | ...             | 1 year   |
| C       | ...             | 2 years  |

The survival labels are shuffled.

Now:

* gene expression stays exactly the same
* survival times stay exactly the same
* but **which patient gets which survival time becomes random**

So any true association disappears.

---

## What does "within each cohort" mean?

Suppose there are three hospitals.

Hospital A

Hospital B

Hospital C

Instead of mixing patients across hospitals, they shuffle survival only **inside each hospital**.

That preserves:

* number of patients
* censoring pattern
* cohort differences

while removing gene-survival relationships.

---

# Then what?

Now they rerun the **exact same search algorithm**.

Not a simpler version.

Not random genes.

Exactly the same discovery pipeline.

If the original search did

```
16,000 genes
↓

pick best gene

↓

pick second best

↓

pick third

↓

pick fourth

↓

pick fifth
```

they repeat all of that on the shuffled data.

---

# Why repeat many times?

One shuffle might accidentally produce a score of

```
0.60
```

Another

```
0.63
```

Another

```
0.58
```

So they repeat many permutations to estimate what scores the search algorithm achieves **when there is no real signal**.

---

# What are they recording?

They record two numbers.

## (a) Inner-LOCO score

The comments say:

> the discovery procedure would have published

Meaning:

This is the score the search algorithm reports for the panel it selected.

Example:

Search finds genes

```
Gene17
Gene304
Gene8
Gene1200
Gene91
```

and reports

```
C-index = 0.73
```

That is recorded.

---

## Why record this?

Because even on completely random data, the search might still report

```
0.70
```

just because it searched so many possibilities.

This measures the amount of **optimism introduced by searching**.

---

## (b) Honest held-out c-index

Now they take the discovered genes and evaluate them on data that the search **never saw**.

If there is truly no signal, performance should collapse.

Ideally

```
C-index ≈ 0.50
```

because random guessing gives about 0.5.

If the held-out score stays around 0.5, that's evidence that any high score from the search on permuted data was just overfitting.

---

# What is forward search?

Suppose there are

```
16,000 genes.
```

The algorithm works like this:

Step 1

```
Try every gene.

Choose the best one.
```

Step 2

```
Keep Gene A.

Try every remaining gene.

Choose the one that improves the score the most.
```

Step 3

```
Keep Genes A and B.

Try every remaining gene.

Choose the best third gene.
```

Repeat until there are 5 genes.

That's **forward search** (or forward selection).

---

# What are these output files?

### CSV

```
permutation_search.csv
```

Stores one row per permutation, for example:

| Replicate | Search Score | Held-out Score |
| --------- | ------------ | -------------- |
| 1         | 0.71         | 0.50           |
| 2         | 0.69         | 0.48           |
| 3         | 0.72         | 0.51           |

---

### JSON

```
permutation_search_summary.json
```

Stores summary statistics such as:

* mean search score
* mean held-out score
* standard deviation
* number of replicates

---

# Why append rows after every replicate?

Imagine you're running 1,000 permutations and each takes 10 minutes.

After 8 hours, the cluster kills your job.

If the script only wrote results at the end, you'd lose everything.

Instead it writes after every completed permutation:

```
Replicate 1 → saved

Replicate 2 → saved

Replicate 3 → saved

...
```

So if it stops at replicate 463, you still have results for those 463 runs.

---

## In one sentence

The purpose of `run_permutation_search.py` is to answer this question:

> **If there were actually no relationship between genes and patient survival, how often would the same gene-selection algorithm still find an apparently "good" 5-gene panel purely by chance?**

That lets researchers distinguish a genuinely predictive gene panel from one that only looks good because the algorithm searched through a huge number of possibilities.
        
"""
import json
import sys
import time

import numpy as np
import pandas as pd

import nested_core as nc

N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
TAG = ("_" + sys.argv[3]) if len(sys.argv) > 3 else ""
CSV = "permutation_search%s.csv" % TAG
ALPHA = 100.0
N_STEPS = 5

# search over five cohorts, hold one out for the honest read -- mirrors the nested design
HELD = "SCANB_GSE202203"
SEARCH = [c for c in nc.OS6 if c != HELD]

t0 = time.time()
store = nc.load_all(nc.OS6)
genes = nc.common_genes(store, nc.OS6)
print("universe %d | search cohorts %s | honest read-out %s"
      % (len(genes), ",".join(SEARCH), HELD), flush=True)

def reported_score(info):
    """The score the search would report for its chosen panel: the mean, over folds,
    of the final-step held-out c-index each fold's own trajectory reached."""
    last = {}
    for r in info["trace"]:
        h = r["inner_held_out"]
        if h not in last or r["step"] > last[h]["step"]:
            last[h] = r
    return float(np.mean([v["score"] for v in last.values()]))


# ---- observed (unpermuted) reference on the same design, for direct comparison
sel_obs, info_obs = nc.forward_search(store, SEARCH, genes, n_steps=N_STEPS,
                                      anchor=nc.ANCHOR4, alpha=ALPHA, verbose=True)
obs_inner = reported_score(info_obs)
obs_held, _ = nc.evaluate_panel(store, SEARCH, HELD, sel_obs, alpha=ALPHA)
print("OBSERVED: panel %s | inner search score %.4f | held-out %.4f"
      % (",".join(sel_obs), obs_inner, obs_held), flush=True)

rows = []
for r in range(N_REP):
    seed = SEED0 + r * 101
    # perm_seed permutes labels inside the fold builder AND inside the univariate
    # pre-filter, so the candidate pool is drawn under the null too
    sel, info = nc.forward_search(store, SEARCH, genes, n_steps=N_STEPS,
                                  anchor=nc.ANCHOR4, alpha=ALPHA,
                                  perm_seed=seed, verbose=False)
    inner = reported_score(info)
    # honest read-out on the held-out cohort with UNPERMUTED labels: does the panel
    # the search found on noise carry any real signal? (should be ~0.5)
    held_real, _ = nc.evaluate_panel(store, SEARCH, HELD, sel, alpha=ALPHA)
    rows.append({"replicate": r, "seed": seed, "panel": "|".join(sel),
                 "inner_loco_c_permuted": inner,
                 "heldout_c_real_labels": held_real,
                 "pool_size_median": float(np.median(info["pool_sizes"]))})
    print("  rep %3d: inner search score(perm) %.4f | held-out(real labels) %.4f | %s"
          % (r, inner, held_real, ",".join(sel)), flush=True)
    # append-as-you-go so a wall-clock kill still leaves usable output
    pd.DataFrame(rows[-1:]).to_csv(CSV, mode="a", header=(r == 0), index=False)

df = pd.DataFrame(rows)

ip = df.inner_loco_c_permuted.values
summary = {
    "n_replicates": N_REP,
    "observed_panel": sel_obs,
    "observed_inner_loco_c": float(obs_inner),
    "observed_heldout_c": float(obs_held),
    "perm_inner_loco_mean": float(ip.mean()),
    "perm_inner_loco_sd": float(ip.std(ddof=1)),
    "perm_inner_loco_max": float(ip.max()),
    "perm_inner_loco_q95": float(np.quantile(ip, 0.95)),
    "perm_inner_loco_q99": float(np.quantile(ip, 0.99)),
    "search_optimism_under_null": float(ip.mean() - 0.5),
    "observed_minus_perm_mean": float(obs_inner - ip.mean()),
    "p_perm_inner": float(((ip >= obs_inner).sum() + 1) / (N_REP + 1)),
    "perm_heldout_real_mean": float(df.heldout_c_real_labels.mean()),
    "perm_heldout_real_max": float(df.heldout_c_real_labels.max()),
    "p_perm_heldout": float(((df.heldout_c_real_labels.values >= obs_held).sum() + 1)
                            / (N_REP + 1)),
    "held_out_cohort": HELD, "search_cohorts": SEARCH,
    "n_candidates": int(len(genes)), "alpha": ALPHA, "n_steps": N_STEPS,
    "elapsed_min": (time.time() - t0) / 60,
}
json.dump(summary, open("permutation_search_summary%s.json" % TAG, "w"), indent=2)
print("\n" + json.dumps(summary, indent=2), flush=True)
