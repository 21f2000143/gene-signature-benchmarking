import os
import numpy as np
import pandas as pd

OS6 = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203", "GSE20711", "GSE58812"]
GENE_SETS = ["Novel5", "Anchor4", "Novel5_plus_Anchor4", "PAM50", "OncotypeDX21",
             "GGI", "MammaPrint70", "BuffaHypoxia", "CNetCox6", "Clinical"]
LEARNERS = ["CoxPH_ridge", "Coxnet", "RSF", "GBSA"]
PRIMARY = "CoxPH_ridge"

shard_files = {
    "TCGA": "{{artifact:326a2642-13f3-415f-9697-3831f7202d1d}}",
    "METABRIC": "{{artifact:d97b601e-043d-4e49-8444-ad6b37b1adf6}}",
    "SCANB_GSE96058": "{{artifact:193fd705-00ff-4ef3-960d-56ba8cdb1180}}",
    "SCANB_GSE202203": "{{artifact:36c84047-8ce7-477c-a07b-0bd94c1aa0fc}}",
    "GSE20711": "{{artifact:56756542-eba4-4ffa-8641-28efbe18c926}}",
    "GSE58812": "{{artifact:96e4841c-3fcf-4949-8cb9-14a599858083}}",
}

frames = []
for coh in OS6:
    p = shard_files[coh]
    d = pd.read_csv(p)
    if len(d):
        frames.append(d)

full = pd.concat(frames, ignore_index=True)
full = full[full["cindex"].notna()]
full = full.drop_duplicates(subset=["held_out_cohort", "gene_set", "learner"],
                            keep="first")
full = full.sort_values(["learner", "gene_set", "held_out_cohort"]).reset_index(drop=True)
os.makedirs("setup", exist_ok=True)
full.to_csv(os.path.join("setup", "learner_grid_full.csv"), index=False)