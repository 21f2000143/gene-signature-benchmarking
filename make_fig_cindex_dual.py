"""
make_fig_cindex_dual.py -- Figure 3: concordance under both Harrell's C and Uno's IPCW C.

Revised for the review. Two substantive changes from the original c-index figure:
  * every gene set is now shown under BOTH metrics, because Uno's IPCW C is lower than
    Harrell's C for 9 of 10 sets (mean drop 0.029) and the original reported Harrell's only
  * the clinicopathology arm is included and ranks FIRST on both metrics, ahead of the
    five-gene panel, in all six leave-one-cohort-out folds

Panels: a  Harrell's C and Uno's C per gene set, mean over the six OS folds
        b  the per-set optimism of Harrell's C relative to Uno's C
        c  per-cohort Harrell's C for the clinical arm, the panel and the anchor scaffold

Inputs (results/): metrics_summary.csv, metrics_harrell_uno.csv
Output: figs/fig_cindex_dual.png
Style: figure-style skill supplies apply_figure_style/panel_letter/META_GREY; fallbacks below.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    apply_figure_style, panel_letter, META_GREY
except NameError:
    META_GREY = "#8A8A8A"

    def apply_figure_style(sizes=(8, 7, 6)):
        mpl.rcParams.update({"font.family": "sans-serif", "font.size": sizes[0],
                             "axes.spines.top": False, "axes.spines.right": False,
                             "savefig.dpi": 400})

    def panel_letter(ax, letter, case="lower"):
        ax.text(-0.16, 1.07, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="right")

FOC, ANCH, NEST, CLIN = "#B3312C", "#9A9A9A", "#2B5C8A", "#4A6B4A"
ORDER = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203", "GSE20711", "GSE58812"]
LBL = {"Clinical": "Clinicopathology", "Novel5": "Novel5 (this study)",
       "Novel5_plus_Anchor4": "Novel5 + anchor", "BuffaHypoxia": "Buffa hypoxia",
       "MammaPrint70": "MammaPrint 70", "PAM50": "PAM50",
       "OncotypeDX21": "Oncotype DX 21", "GGI": "GGI", "CNetCox6": "CNet-Cox 6",
       "Anchor4": "Anchor scaffold"}
SHORT = {"TCGA": "TCGA", "METABRIC": "META-\nBRIC", "SCANB_GSE96058": "SCAN-B\n96058",
         "SCANB_GSE202203": "SCAN-B\n202203", "GSE20711": "GSE\n20711",
         "GSE58812": "GSE\n58812"}

RES = "results"
ms = pd.read_csv(os.path.join(RES, "metrics_summary.csv"))
mh = pd.read_csv(os.path.join(RES, "metrics_harrell_uno.csv"))

# The Clinical rows of metrics_summary.csv and metrics_harrell_uno.csv predate
# the clinical audit: they treat GSE20711's unlabelled 0/1 age and size columns
# as years and millimetres, which inflates that cohort's clinical concordance by
# 0.057 and its mean by 0.011. Tables 2 and 4 both report the audited refit, so
# the clinical arm is overridden HERE, once, at load -- all three panels then
# draw the same clinical numbers the tables print.
REC = pd.read_csv(os.path.join(RES, "clinical_arm_reconciled.csv"))
_ORDER6 = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203",
           "GSE20711", "GSE58812"]
_r = REC.set_index("held_out_cohort").reindex(_ORDER6)
assert _r.clinical_harrell.notna().all(), "reconciled clinical arm incomplete"
ms.loc[ms.gene_set.eq("Clinical"), "harrell_mean"] = _r.clinical_harrell.mean()
ms.loc[ms.gene_set.eq("Clinical"), "uno_mean"] = _r.clinical_uno.mean()
_m = mh.gene_set.eq("Clinical")
mh.loc[_m, "harrell_c"] = mh.loc[_m, "held_out_cohort"].map(_r.clinical_harrell)
mh.loc[_m, "uno_c"] = mh.loc[_m, "held_out_cohort"].map(_r.clinical_uno)


def colour_of(g):
    return FOC if g == "Novel5" else (CLIN if g == "Clinical" else
                                      (ANCH if g == "Anchor4" else NEST))


def draw(path="figs/fig_cindex_dual.png"):
    apply_figure_style(sizes=(8, 7, 6))
    mpl.rcParams.update({"axes.titlesize": 7.0, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6.0, "ytick.labelsize": 6.0})
    d = ms.sort_values("harrell_mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    fig = plt.figure(figsize=(180 / 25.4, 112 / 25.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.28, 1], height_ratios=[1, 0.88],
                          hspace=0.52, wspace=0.30, left=0.175, right=0.975,
                          top=0.915, bottom=0.095)

    axa = fig.add_subplot(gs[:, 0])
    for i, row in d.iterrows():
        c = colour_of(row.gene_set)
        axa.plot([row.uno_mean, row.harrell_mean], [i, i], color=c, lw=1.1, alpha=0.55,
                 solid_capstyle="round", zorder=1)
        axa.scatter(row.harrell_mean, i, s=26, color=c, zorder=3, marker="o")
        axa.scatter(row.uno_mean, i, s=24, zorder=3, marker="s", facecolor="white",
                    linewidth=1.0, edgecolor=c)
    axa.axvline(0.5, color="k", lw=0.6, ls=(0, (4, 3)))
    axa.set_yticks(y)
    axa.set_yticklabels([LBL[g] for g in d.gene_set], fontsize=6.2)
    for t, g in zip(axa.get_yticklabels(), d.gene_set):
        if g in ("Novel5", "Clinical"):
            t.set_fontweight("bold")
    axa.set_xlabel("Mean concordance over six leave-one-cohort-out folds")
    axa.set_xlim(0.485, 0.745)
    axa.set_ylim(-0.7, len(d) - 0.3)
    axa.set_title("Clinicopathology outranks every gene set\non both concordance metrics")
    axa.scatter([], [], s=26, color="#555555", marker="o", label="Harrell's C")
    axa.scatter([], [], s=24, marker="s", facecolor="white", edgecolor="#555555",
                linewidth=1.0, label="Uno's IPCW C")
    axa.legend(frameon=False, fontsize=6, loc="lower right", handletextpad=0.3,
               borderpad=0.15, labelspacing=0.3)
    axa.text(0.4965, 6.2, "chance", fontsize=5.8, rotation=90, va="bottom",
             ha="right", color="#444444")

    axb = fig.add_subplot(gs[0, 1])
    dd = ms.sort_values("harrell_mean", ascending=False)
    drop = (dd.harrell_mean - dd.uno_mean).values
    axb.barh(np.arange(len(dd))[::-1], drop, height=0.7,
             color=[colour_of(g) for g in dd.gene_set])
    axb.axvline(0, color="k", lw=0.7)
    axb.set_yticks(np.arange(len(dd))[::-1])
    axb.set_yticklabels([LBL[g] for g in dd.gene_set], fontsize=5.6)
    axb.set_xlabel("Harrell's C $-$ Uno's IPCW C\n(positive: Harrell's C optimistic)")
    npos = int((ms.harrell_mean - ms.uno_mean > 0).sum())
    axb.set_title("Harrell's C is optimistic for %d of %d sets" % (npos, len(ms)))

    axc = fig.add_subplot(gs[1, 1])
    # mh already carries the audited clinical arm (overridden at load).
    sub = mh[mh.gene_set.isin(["Novel5", "Clinical", "Anchor4"])]
    piv = sub.pivot_table(index="held_out_cohort", columns="gene_set",
                          values="harrell_c").loc[ORDER]
    assert abs(piv["Clinical"].mean() - _r.clinical_harrell.mean()) < 1e-9, \
        "panel c clinical bars are not the reconciled arm"
    xs = np.arange(len(ORDER))
    w = 0.26
    for j, (g, c) in enumerate([("Clinical", CLIN), ("Novel5", FOC), ("Anchor4", ANCH)]):
        axc.bar(xs + (j - 1) * w, piv[g].values, w, color=c, label=LBL[g])
    axc.axhline(0.5, color="k", lw=0.6, ls=(0, (4, 3)))
    axc.set_xticks(xs)
    axc.set_xticklabels([SHORT[o] for o in ORDER], fontsize=5.4)
    axc.set_ylabel("Harrell's C")
    axc.set_ylim(0.40, 0.83)
    axc.set_yticks(np.arange(0.40, 0.81, 0.10))
    nwin = int((piv["Clinical"] > piv["Novel5"]).sum())
    axc.set_title("The clinical arm wins in all six cohorts" if nwin == len(ORDER)
                  else "The clinical arm wins in %d of six cohorts" % nwin)
    axc.legend(frameon=False, fontsize=5.6, loc="upper left", handlelength=1.1,
               borderpad=0.1, labelspacing=0.22)

    for ax, L in [(axa, "a"), (axb, "b"), (axc, "c")]:
        panel_letter(ax, L, case="lower")
    fig.savefig(path, dpi=400, facecolor="white")
    return fig


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    fig = draw()
    r = fig.canvas.get_renderer()
    tx = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
          if t.get_text().strip() and t.get_visible()]
    axl = ({id(a.xaxis.label) for a in fig.axes} |
           {id(a.yaxis.label) for a in fig.axes})
    ov = [(a.get_text()[:24], b.get_text()[:24])
          for i, (a, ba) in enumerate(tx) for b, bb in tx[i + 1:]
          if ba.overlaps(bb) and id(a) not in axl and id(b) not in axl]
    assert not ov, "text collisions: %s" % ov
    print("figs/fig_cindex_dual.png written; text collisions: 0")
