
"""
run_c11_er_prolif_singlegene.py -- reviewer item C11 (three sub-analyses):

(i)  LOCO concordance within ER-positive and ER-negative strata, for Novel-5 and
     the two strongest published comparators (Buffa hypoxia, MammaPrint-70).
(ii) Correlate the Novel-5 LOCO risk score with a proliferation metagene (mean
     z-score of 11 canonical proliferation genes) and with MKI67 alone; then fit
     Novel-5 jointly with the Buffa hypoxia metagene score (as a second covariate
     in the same ridge Cox model) to see whether Novel-5 retains independent
     concordance once hypoxia is held fixed.
(iii) Single-gene proliferation arm: add MKI67 alone as a benchmark arm, LOCO,
     same design as every other arm in metrics_harrell_uno.csv.

Reuses the LOCO design, ridge-Cox learner (alpha=100) and per-fold tau exactly as
metrics_uno_auc_ph.py so all outputs are numerically comparable to Table 2.
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nested_core as nc
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator

warnings.filterwarnings("ignore")

ALPHA = 100.0
OS6 = nc.OS6
NOVEL5 = nc.NOVEL5
PROLIF = ["BIRC5", "CCNB1", "CDC20", "CDK1", "CENPF", "KIF2C", "MKI67", "PTTG1", "RRM2", "TYMS", "UBE2C"]
BUFFA = ["ACOT7","ADM","AK4","ALDOA","ANGPTL4","ANKRD37","BNIP3","CA9","CDKN3","DDIT4","EGLN3","ENO1",
         "ESRP1","FUT11","GBE1","GPI","HIG2","HK2","INSIG2","JMJD6","KCTD11","KDM3A","LDHA","LOX","MAFF",
         "MIF","MRPS17","MTFP1","NARF","NDRG1","P4HA1","P4HA2","P4HB","PDK1","PFKP","PGAM1","PGK1","PLOD1",
         "PLOD2","PPP1R3B","PYGL","SDC4","SEC61G","SERPINE1","SIAH2","SLC2A1","TMEM45A","TPI1","TUBB6","UBC",
         "VEGFA","WSB1"]
MAMMAPRINT70_SET = json.load(open("results/gene_sets.json"))["MammaPrint70"]["genes"]


def surv_y(t, ev):
    return np.array([(bool(e), float(x)) for e, x in zip(ev, t)], dtype=[("event", bool), ("time", float)])


def zscore_col(v):
    v = np.asarray(v, float)
    m, s = np.nanmean(v), np.nanstd(v)
    s = s if s > 1e-12 else 1.0
    return (v - m) / s


def er_status_col(clin):
    """Harmonised ER positivity as 0/1/NaN, matching metrics_uno_auc_ph.harmonise_clinical."""
    if "er" not in clin.columns:
        return pd.Series(np.nan, index=clin.index)
    raw = clin["er"].astype(str).str.upper().str.strip()
    num = pd.to_numeric(raw, errors="coerce")
    erp = pd.Series(np.nan, index=clin.index)
    erp[num.notna()] = (num[num.notna()] > 0).astype(float)
    strm = raw[num.isna()]
    val = pd.Series(np.nan, index=strm.index)
    val[strm.str.startswith("POS")] = 1.0
    val[strm.str.startswith("NEG")] = 0.0
    erp[strm.index] = val
    return erp


def tau_for_fold(ttr, etr, tte, ete):
    y_tr = surv_y(ttr, etr)
    cens = CensoringDistributionEstimator().fit(y_tr)
    max_ev_te = float(tte[ete == 1].max()) if ete.sum() > 0 else float(tte.max())
    gx, gy = cens.unique_time_, cens.prob_
    pos = gx[gy > 1e-8]
    cap = float(pos.max()) if len(pos) else float(ttr.max())
    candidates = [min(max_ev_te, cap * (1 - 1e-9))]
    ev_times = tte[ete == 1]
    for q in (0.99, 0.95, 0.90):
        if len(ev_times):
            candidates.append(float(np.quantile(ev_times, q)))
    return candidates


def safe_uno(y_tr, y_te, risk, candidates):
    for tc in candidates:
        try:
            v = float(concordance_index_ipcw(y_tr, y_te, risk, tau=tc)[0])
            if np.isfinite(v):
                return v, float(tc)
        except Exception:
            continue
    return np.nan, np.nan


store = nc.load_all(OS6, verbose=True)

# harmonised per-sample ER status, and check gene availability
er_by_cohort = {}
for coh in OS6:
    _, _, _, s = store[coh]
    er_by_cohort[coh] = er_status_col(s)
    n_pos = int((er_by_cohort[coh] == 1).sum())
    n_neg = int((er_by_cohort[coh] == 0).sum())
    print(coh, "ER+:", n_pos, "ER-:", n_neg, "missing:", int(er_by_cohort[coh].isna().sum()), flush=True)

gene_sets_for_er = {
    "Novel5": NOVEL5,
    "BuffaHypoxia": BUFFA,
    "MammaPrint70": MAMMAPRINT70_SET,
}

# ================================================================ (i) ER-stratified concordance
er_rows = []
for held in OS6:
    train = [c for c in OS6 if c != held]
    Xte_all, tte, ete, s_te = store[held]
    er_te = er_by_cohort[held].values
    ytr_parts = [(store[c][1], store[c][2]) for c in train]
    ttr = np.concatenate([a for a, _ in ytr_parts])
    etr = np.concatenate([b for _, b in ytr_parts])
    y_tr = surv_y(ttr, etr)
    tau_candidates = tau_for_fold(ttr, etr, tte, ete)

    for name, genes in gene_sets_for_er.items():
        avail = [g for g in genes if g in Xte_all.columns and all(g in store[c][0].columns for c in train)]
        if not avail:
            continue
        Xtr = np.vstack([store[c][0][avail].values for c in train])
        Xte = Xte_all[avail].values
        beta = nc.fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)
        risk_te = Xte @ beta

        for stratum, mask_val in (("ER-positive", 1.0), ("ER-negative", 0.0)):
            mask = er_te == mask_val
            n_s = int(mask.sum())
            ev_s = int(ete[mask].sum()) if n_s else 0
            if n_s < 10 or ev_s < 5:
                er_rows.append(dict(held_out_cohort=held, gene_set=name, stratum=stratum,
                                     n=n_s, events=ev_s, harrell_c=np.nan, uno_c=np.nan,
                                     note="insufficient n/events for stratum"))
                continue
            hc = float(concordance_index_censored(ete[mask].astype(bool), tte[mask], risk_te[mask])[0])
            y_te_s = surv_y(tte[mask], ete[mask])
            uc, tau_used = safe_uno(y_tr, y_te_s, risk_te[mask], tau_candidates)
            er_rows.append(dict(held_out_cohort=held, gene_set=name, stratum=stratum,
                                 n=n_s, events=ev_s, harrell_c=hc, uno_c=uc, tau_months=tau_used,
                                 note=""))
        print("  %-16s %-14s done" % (held, name), flush=True)

er_df = pd.DataFrame(er_rows)
er_df.to_csv("er_stratified_concordance_c11.csv", index=False)

er_summary = (er_df.dropna(subset=["harrell_c"])
              .groupby(["gene_set", "stratum"])[["harrell_c", "uno_c"]]
              .agg(["mean", "count"]))
print(er_summary, flush=True)

# ================================================================ (ii) proliferation confound + joint hypoxia
loco_novel5 = pd.read_csv("results/loco_risk_novel5.csv")  # cohort,sample,risk,risk_z,time_months,event

prolif_rows = []
joint_rows = []
persample_rows = []
for held in OS6:
    train = [c for c in OS6 if c != held]
    Xte_all, tte, ete, s_te = store[held]
    ytr_parts = [(store[c][1], store[c][2]) for c in train]
    ttr = np.concatenate([a for a, _ in ytr_parts])
    etr = np.concatenate([b for _, b in ytr_parts])

    # --- Novel5 LOCO risk for this held-out cohort (already computed, from loco_risk_novel5.csv)
    sub = loco_novel5[loco_novel5.cohort == held].set_index("sample")
    # align to Xte_all's row order
    novel5_risk = sub.reindex(Xte_all.index)["risk_z"].values

    # --- proliferation metagene score (mean z-scored expression of the 11 genes), MKI67 alone
    prolif_avail = [g for g in PROLIF if g in Xte_all.columns]
    prolif_score = np.mean(np.column_stack([zscore_col(Xte_all[g].values) for g in prolif_avail]), axis=1)
    mki67_score = zscore_col(Xte_all["MKI67"].values) if "MKI67" in Xte_all.columns else np.full(len(Xte_all), np.nan)

    ok = np.isfinite(novel5_risk) & np.isfinite(prolif_score)
    r_prolif, p_prolif = stats.pearsonr(novel5_risk[ok], prolif_score[ok]) if ok.sum() > 5 else (np.nan, np.nan)
    ok2 = np.isfinite(novel5_risk) & np.isfinite(mki67_score)
    r_mki67, p_mki67 = stats.pearsonr(novel5_risk[ok2], mki67_score[ok2]) if ok2.sum() > 5 else (np.nan, np.nan)

    prolif_rows.append(dict(held_out_cohort=held, n=int(ok.sum()),
                             n_prolif_genes_avail=len(prolif_avail),
                             corr_novel5_vs_proliferation_metagene=r_prolif, p_value_prolif=p_prolif,
                             corr_novel5_vs_MKI67=r_mki67, p_value_mki67=p_mki67))
    persample_rows.append(pd.DataFrame(dict(cohort=held, sample=Xte_all.index,
                                             novel5_risk_z=novel5_risk,
                                             prolif_score=prolif_score, mki67_score=mki67_score)))

    # --- joint model: Novel5 gene panel + Buffa hypoxia score, fitted together
    buffa_avail = [g for g in BUFFA if g in Xte_all.columns and all(g in store[c][0].columns for c in train)]
    novel5_avail = [g for g in NOVEL5 if g in Xte_all.columns and all(g in store[c][0].columns for c in train)]
    if buffa_avail and novel5_avail:
        # training Buffa score per cohort: z-scored within cohort, mean over genes
        buffa_tr = np.concatenate([
            np.mean(np.column_stack([zscore_col(store[c][0][g].values) for g in buffa_avail]), axis=1)
            for c in train
        ])
        buffa_te = np.mean(np.column_stack([zscore_col(Xte_all[g].values) for g in buffa_avail]), axis=1)

        Xg_tr = np.vstack([store[c][0][novel5_avail].values for c in train])
        Xg_te = Xte_all[novel5_avail].values

        # Novel5-only (for reference)
        beta_novel5_only = nc.fit_ridge_cox(Xg_tr, ttr, etr, alpha=ALPHA)
        risk_novel5_only = Xg_te @ beta_novel5_only
        hc_novel5_only = float(concordance_index_censored(ete.astype(bool), tte, risk_novel5_only)[0])

        # joint: Novel5 genes (z-scored per nested_core convention, already within-cohort z-scored
        # at load) plus the Buffa score as one extra column
        X_joint_tr = np.column_stack([Xg_tr, buffa_tr])
        X_joint_te = np.column_stack([Xg_te, buffa_te])
        beta_joint = nc.fit_ridge_cox(X_joint_tr, ttr, etr, alpha=ALPHA)
        risk_joint = X_joint_te @ beta_joint
        hc_joint = float(concordance_index_censored(ete.astype(bool), tte, risk_joint)[0])

        # Buffa-only (for reference)
        beta_buffa_only = nc.fit_ridge_cox(buffa_tr.reshape(-1, 1), ttr, etr, alpha=ALPHA)
        risk_buffa_only = (buffa_te.reshape(-1, 1)) @ beta_buffa_only
        hc_buffa_only = float(concordance_index_censored(ete.astype(bool), tte, risk_buffa_only)[0])

        # Novel-5's own coefficient block in the joint fit, vs its coefficients alone --
        # report the ratio of ||beta_novel5 in joint|| to ||beta_novel5 alone|| as a shrinkage check
        beta_novel5_in_joint = beta_joint[:len(novel5_avail)]
        beta_buffa_in_joint = beta_joint[-1]

        joint_rows.append(dict(held_out_cohort=held,
                                harrell_novel5_only=hc_novel5_only,
                                harrell_buffa_only=hc_buffa_only,
                                harrell_joint_novel5_plus_buffa=hc_joint,
                                delta_joint_vs_novel5_only=hc_joint - hc_novel5_only,
                                delta_joint_vs_buffa_only=hc_joint - hc_buffa_only,
                                buffa_coef_in_joint=float(beta_buffa_in_joint),
                                novel5_coef_norm_alone=float(np.linalg.norm(beta_novel5_only)),
                                novel5_coef_norm_in_joint=float(np.linalg.norm(beta_novel5_in_joint))))
    print("  %-16s proliferation+joint done" % held, flush=True)

prolif_df = pd.DataFrame(prolif_rows)
prolif_df.to_csv("proliferation_confound_c11.csv", index=False)
joint_df = pd.DataFrame(joint_rows)
joint_df.to_csv("joint_hypoxia_model_c11.csv", index=False)

print(prolif_df.to_string(index=False), flush=True)
print(joint_df.to_string(index=False), flush=True)

persample_df = pd.concat(persample_rows, ignore_index=True)
ok_p = persample_df["novel5_risk_z"].notna() & persample_df["prolif_score"].notna()
r_pooled_prolif, p_pooled_prolif = stats.pearsonr(
    persample_df.loc[ok_p, "novel5_risk_z"], persample_df.loc[ok_p, "prolif_score"])
ok_m = persample_df["novel5_risk_z"].notna() & persample_df["mki67_score"].notna()
r_pooled_mki67, p_pooled_mki67 = stats.pearsonr(
    persample_df.loc[ok_m, "novel5_risk_z"], persample_df.loc[ok_m, "mki67_score"])
pooled_summary = dict(n_pooled=int(ok_p.sum()),
                      pooled_corr_novel5_vs_proliferation_metagene=float(r_pooled_prolif),
                      pooled_p_prolif=float(p_pooled_prolif),
                      pooled_corr_novel5_vs_MKI67=float(r_pooled_mki67),
                      pooled_p_mki67=float(p_pooled_mki67),
                      mean_delta_joint_vs_novel5_only=float(joint_df["delta_joint_vs_novel5_only"].mean()),
                      mean_delta_joint_vs_buffa_only=float(joint_df["delta_joint_vs_buffa_only"].mean()))
json.dump(pooled_summary, open("proliferation_pooled_summary_c11.json", "w"), indent=1)
print("POOLED:", pooled_summary, flush=True)

# ================================================================ (iii) single-gene proliferation arm (MKI67)
mki67_rows = []
for held in OS6:
    train = [c for c in OS6 if c != held]
    Xte_all, tte, ete, _ = store[held]
    ytr_parts = [(store[c][1], store[c][2]) for c in train]
    ttr = np.concatenate([a for a, _ in ytr_parts])
    etr = np.concatenate([b for _, b in ytr_parts])
    y_tr = surv_y(ttr, etr)
    tau_candidates = tau_for_fold(ttr, etr, tte, ete)

    if "MKI67" not in Xte_all.columns or any("MKI67" not in store[c][0].columns for c in train):
        continue
    Xtr = np.vstack([store[c][0][["MKI67"]].values for c in train])
    Xte = Xte_all[["MKI67"]].values
    beta = nc.fit_ridge_cox(Xtr, ttr, etr, alpha=ALPHA)
    risk = (Xte @ beta).ravel()
    hc = float(concordance_index_censored(ete.astype(bool), tte, risk)[0])
    y_te = surv_y(tte, ete)
    uc, tau_used = safe_uno(y_tr, y_te, risk, tau_candidates)
    mki67_rows.append(dict(held_out_cohort=held, gene_set="MKI67_single_gene",
                            n_genes_used=1, harrell_c=hc, uno_c=uc, tau_months=tau_used,
                            n_test=int(len(tte)), events_test=int(ete.sum()),
                            n_train=int(len(ttr)), events_train=int(etr.sum())))
    print("  %-16s MKI67 single-gene: Harrell=%.4f Uno=%.4f" % (held, hc, uc), flush=True)

mki67_df = pd.DataFrame(mki67_rows)
mki67_df.to_csv("mki67_single_gene_arm_c11.csv", index=False)
print(mki67_df.to_string(index=False), flush=True)
print("mean Harrell:", mki67_df.harrell_c.mean(), "mean Uno:", mki67_df.uno_c.mean(), flush=True)

print("DONE", flush=True)
