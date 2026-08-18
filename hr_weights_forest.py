"""
hr_weights_forest.py -- how much each cohort actually drives the pooled HR, plus the
forest plot for review item 6.

REVIEW ITEM 6. The reviewer's objection is that the pooled HR is "dominated by one
cohort". This script makes that quantitative by writing out, for every cohort, the
weight it receives under each of the three pooled estimators, and renders the forest
plot that should accompany the revised text.

WEIGHT DEFINITIONS (all reported in hr_cohort_weights.csv):
  events_share        -- cohort's share of the 1406 high/low events. This is the
                         informal sense in which the naive pooled Cox is "dominated":
                         an unstratified Cox score statistic is driven by event counts.
  naive_info_share    -- exact influence in the naive pooled Cox: the cohort's
                         contribution to the total observed Fisher information for the
                         single log-HR, as a share. Computed by evaluating the Breslow
                         information of the FULL pooled model restricted to that
                         cohort's event times (i.e. the cohort's additive term in the
                         pooled information sum), at the fitted pooled beta.
  strat_info_share    -- same quantity for the cohort-stratified model, where each
                         cohort contributes its own within-cohort information term.
  fe_weight           -- fixed-effect meta-analysis weight  w_i / sum w,  w_i = 1/v_i.
  re_weight           -- DerSimonian-Laird random-effects weight
                         w*_i / sum w*,  w*_i = 1/(v_i + tau^2). As tau^2 grows the
                         weights flatten towards 1/k, which is why the random-effects
                         estimate stops being a METABRIC estimate.

FIGURE (hr_forest.png): per-cohort high-vs-low HR with 95% CI (marker area
proportional to the random-effects weight), the three pooled estimates, and the
random-effects 95% prediction interval for a new cohort.

Inputs : hr_per_cohort.csv, hr_pooled_methods.csv, heterogeneity.json,
         loco_risk_scores.csv   (all written by hr_pooled.py)
Outputs: hr_cohort_weights.csv, hr_forest.png
"""
import json
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from hr_pooled import _breslow_grad_hess, cox_fit                      # noqa: E402


def main(indir, outdir="."):
    per = pd.read_csv("%s/hr_per_cohort.csv" % indir)
    meth = pd.read_csv("%s/hr_pooled_methods.csv" % indir)
    het = json.load(open("%s/heterogeneity.json" % indir))
    sc = pd.read_csv("%s/loco_risk_scores.csv" % indir)

    hl = sc[sc.group != "mid"].reset_index(drop=True)
    z = (hl.group.values == "high").astype(float)
    t = hl.time_months.values.astype(float)
    e = hl.event.values.astype(np.int32)
    coh = hl.cohort.values

    b_naive = float(meth.loc[meth.method == "naive_pooled", "log_hr"].iloc[0])
    b_strat = float(meth.loc[meth.method == "cohort_stratified", "log_hr"].iloc[0])

    # --- information contributed by each cohort
    # stratified: the model's information IS the sum of within-cohort informations
    strat_info = {}
    for cname in per.cohort:
        m = coh == cname
        _, _, I = _breslow_grad_hess(z[m][:, None], t[m], e[m], np.array([b_strat]))
        strat_info[cname] = float(I[0, 0])
    # naive: the pooled information is a sum over event times; attribute each event
    # time's term to the cohort of the patient(s) dying at it, using POOLED risk sets.
    o = np.argsort(-t, kind="mergesort")
    zs, ts, es, cs = z[o], t[o], e[o], coh[o]
    w = np.exp(b_naive * zs)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    naive_info = {c: 0.0 for c in per.cohort}
    i = 0
    n = len(ts)
    while i < n:
        j = i
        while j < n and ts[j] == ts[i]:
            s0 += w[j]
            s1 += w[j] * zs[j]
            s2 += w[j] * zs[j] ** 2
            j += 1
        mu = s1 / s0
        term = s2 / s0 - mu ** 2
        for k in range(i, j):
            if es[k] == 1:
                naive_info[cs[k]] += term
        i = j

    v = per.se_log_hr.values ** 2
    tau2 = float(het["tau2"])
    wfe = 1.0 / v
    wre = 1.0 / (v + tau2)
    tot_naive = sum(naive_info.values())
    tot_strat = sum(strat_info.values())

    wt = pd.DataFrame({
        "cohort": per.cohort,
        "n_highlow": per.n_highlow, "events_highlow": per.events_highlow,
        "hr": per.hr, "ci_lo": per.ci_lo, "ci_hi": per.ci_hi,
        "log_hr": per.log_hr, "se_log_hr": per.se_log_hr,
        "events_share": per.events_highlow / per.events_highlow.sum(),
        "naive_info_share": [naive_info[c] / tot_naive for c in per.cohort],
        "strat_info_share": [strat_info[c] / tot_strat for c in per.cohort],
        "fe_weight": wfe / wfe.sum(),
        "re_weight": wre / wre.sum()})
    wt.to_csv("%s/hr_cohort_weights.csv" % outdir, index=False)

    # ------------------------------------------------------------------ forest plot
    try:
        apply_figure_style()                                            # noqa: F821
    except Exception:
        mpl.rcParams.update({"font.size": 8, "axes.spines.top": False,
                             "axes.spines.right": False, "figure.dpi": 150})

    order = per.sort_values("hr").reset_index(drop=True)
    wmap = dict(zip(wt.cohort, wt.re_weight))
    FOCAL = "#1b3a6b"
    POOL = {"naive_pooled": "#b0762a", "cohort_stratified": "#7a7f87",
            "random_effects_DL": "#a4243b"}
    NAME = {"naive_pooled": "Naive pooled\n(as published)",
            "cohort_stratified": "Cohort-stratified",
            "random_effects_DL": "Random-effects\n(DerSimonian–Laird)"}

    k = len(order)
    ypos = list(range(k))[::-1]
    fig, ax = plt.subplots(figsize=(6.9, 4.5))

    for yi, r in zip(ypos, order.itertuples()):
        ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=FOCAL, lw=1.1, zorder=2)
        ax.scatter([r.hr], [yi], s=40 + 900 * wmap[r.cohort], color=FOCAL,
                   zorder=3, edgecolor="white", linewidth=0.6)
        ax.text(20.5, yi, "%d / %d" % (r.events_highlow, r.n_highlow),
                va="center", ha="right", fontsize=6, color="0.25")
        ax.text(30.0, yi, "%.0f%%" % (100 * wmap[r.cohort]), va="center",
                ha="right", fontsize=6, color="0.25")

    ybase = -1.4
    prow = []
    for mname in ["naive_pooled", "cohort_stratified", "random_effects_DL"]:
        r = meth[meth.method == mname].iloc[0]
        prow.append((mname, float(r.hr), float(r.ci_lo), float(r.ci_hi)))
    for idx, (mname, hr, lo, hi) in enumerate(prow):
        yi = ybase - idx * 1.05
        ax.plot([lo, hi], [yi, yi], color=POOL[mname], lw=2.0, zorder=2)
        ax.scatter([hr], [yi], marker="D", s=42, color=POOL[mname], zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.text(hr, yi + 0.42, "%.2f" % hr, ha="center", va="bottom",
                fontsize=7, color=POOL[mname], fontweight="bold")

    # prediction interval for a new cohort
    pi_lo, pi_hi = float(het["pred_int_lo"]), float(het["pred_int_hi"])
    ypi = ybase - 3 * 1.05
    ax.plot([pi_lo, pi_hi], [ypi, ypi], color=POOL["random_effects_DL"], lw=1.2,
            ls=(0, (3, 2)), zorder=2)
    ax.scatter([pi_lo, pi_hi], [ypi, ypi], marker="|", s=60,
               color=POOL["random_effects_DL"], zorder=3)

    ax.axvline(1.0, color="0.55", lw=0.8, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.55, 34)
    ax.set_xticks([0.7, 1, 2, 3, 5, 8, 15])
    ax.set_xticklabels(["0.7", "1", "2", "3", "5", "8", "15"])
    ax.set_ylim(ypi - 0.9, k - 0.3)

    # ypos[i] is the row on which order.iloc[i] was drawn, so labels must follow
    # `order` directly -- reversing here would mis-pair every cohort with its HR.
    labels = list(order.cohort)
    ax.set_yticks(ypos + [ybase - i * 1.05 for i in range(3)] + [ypi])
    ax.set_yticklabels(labels + [NAME[m] for m, _, _, _ in prow] +
                       ["95% prediction interval\nfor a new cohort"], fontsize=7)
    for tick, lab in zip(ax.get_yticklabels(), labels + [m for m, _, _, _ in prow] + ["pi"]):
        if lab in POOL:
            tick.set_color(POOL[lab])
    ax.set_xlabel("Hazard ratio, high vs low tertile of held-out risk (log scale)")
    ax.text(20.5, k - 0.55, "events / n", fontsize=6, ha="right", color="0.25")
    ax.text(30.0, k - 0.55, "RE weight", fontsize=6, ha="right", color="0.25")
    ax.set_title("Pooling across six cohorts inflates the apparent precision of the "
                 "five-gene panel's\nhigh-risk hazard ratio (I$^2$ = %.0f%%)"
                 % het["I2_percent"], loc="left", fontsize=8.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()

    # --- verification: each cohort tick label must sit on the row carrying its own HR
    drawn = {yi: r.cohort for yi, r in zip(ypos, order.itertuples())}
    for ytick, lab in zip(ypos, labels):
        assert drawn[ytick] == lab, "row %s labelled %s but carries %s" % (
            ytick, lab, drawn[ytick])
    # --- verification: no overlapping visible text boxes
    r_ = fig.canvas.get_renderer()
    tx = [(o.get_text(), o.get_window_extent(r_)) for o in fig.findobj(mpl.text.Text)
          if o.get_text().strip() and o.get_visible()]
    ov = [(a[0], b[0]) for i, a in enumerate(tx) for b in tx[i + 1:]
          if a[1].overlaps(b[1])]
    print("text overlaps:", ov)

    fig.savefig("%s/hr_forest.png" % outdir, dpi=300, bbox_inches="tight")

    print(wt.to_string(index=False))
    return fig, wt


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
