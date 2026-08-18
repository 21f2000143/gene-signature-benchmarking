import glob, pandas as pd, numpy as np, json, re

parts = []
for f in [
    "permutation_search_s0.csv",
    "permutation_search_s1.csv",
    "permutation_search_s2.csv",
    "permutation_search_s3.csv",
]:
    try:
        d = pd.read_csv(f)
        d = d[pd.to_numeric(d.inner_loco_c_permuted, errors="coerce").notna()]
        d["inner_loco_c_permuted"] = d.inner_loco_c_permuted.astype(float)
        d["heldout_c_real_labels"] = d.heldout_c_real_labels.astype(float)
        d["src"] = f.split("/")[-1]
        parts.append(d)
        print(f.split("/")[-1], len(d))
    except Exception as e:
        print("skip", f, e)
perm = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["seed"])
print("\npooled unique replicates:", len(perm))
print(perm[["inner_loco_c_permuted","heldout_c_real_labels"]].describe().loc[["mean","std","min","max"]].round(4).to_string())

allperm = perm.reset_index(drop=True)
allperm.to_csv("permutation_search_pooled.csv", index=False)