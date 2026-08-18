"""
make_fig_null_resolution.py -- Figure 2: random-panel nulls and the resolution floor.

Revised for the review. Three corrections to the original null analysis:
  * the random-panel null is now 20,000 draws, not 1,000; the original reported p < 0.001,
    which was simply the floor of a 1,000-draw design (0/1000). The 20,000-draw p is 5e-05,
    again the floor of that design -- so the honest statement is "below the design floor",
    not a specific small p-value.
  * every published comparator is now compared against a null matched to ITS OWN gene count,
    because larger sets score higher under random draws; only 4 of 9 sets clear the 95th
    percentile of their size-matched null.
  * the hard resolution floor of any six-cohort paired test is shown explicitly: no such
    test can reach p < 0.03125, so cohort-level tests are descriptive, not confirmatory.

Panels: a  20,000-draw null of random five-gene panels, with the panel's observed score
        b  percentile of each gene set within its size-matched null
        c  attainable exact two-sided p against number of cohorts favouring one signature
        d  per-fold observed concordance against the null median-to-99.9th band

Inputs (results/): null_random_panels_20k.csv, null_20k_summary.json, null_summary.csv,
  resolution_floor.json
Output: figs/fig_null_resolution.png
Style: figure-style skill supplies apply_figure_style/panel_letter; fallbacks defined below.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    apply_figure_style, panel_letter
except NameError:
    def apply_figure_style(sizes=(8, 7, 6)):
        mpl.rcParams.update({"font.family": "sans-serif", "font.size": sizes[0],
                             "axes.spines.top": False, "axes.spines.right": False,
                             "savefig.dpi": 400})

    def panel_letter(ax, letter, case="lower"):
        ax.text(-0.16, 1.07, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="right")

FOC, ANCH, NEUT = "#B3312C", "#8A8A8A", "#2B5C8A"
ORDER = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203", "GSE20711", "GSE58812"]
SHORT = {"TCGA": "TCGA", "METABRIC": "META-\nBRIC", "SCANB_GSE96058": "SCAN-B\n96058",
         "SCANB_GSE202203": "SCAN-B\n202203", "GSE20711": "GSE\n20711",
         "GSE58812": "GSE\n58812"}
LBL = {"Clinical": "Clinicopathology", "Novel5": "Novel5 (this study)",
       "Novel5_plus_Anchor4": "Novel5 + anchor", "BuffaHypoxia": "Buffa hypoxia",
       "MammaPrint70": "MammaPrint 70", "PAM50": "PAM50",
       "OncotypeDX21": "Oncotype DX 21", "GGI": "GGI", "CNetCox6": "CNet-Cox 6",
       "Anchor4": "Anchor scaffold"}

RES = "results"
nd = pd.read_csv(os.path.join(RES, "null_random_panels_20k.csv"))
n20 = json.load(open(os.path.join(RES, "null_20k_summary.json")))
nul = pd.read_csv(os.path.join(RES, "null_summary.csv"))
rf = json.load(open(os.path.join(RES, "resolution_floor.json")))

sm = nul[nul.null_set.str.startswith("sizematched")
         & ~nul.observed_panel.str.contains("reference")].sort_values(
    "observed_percentile")
ST = {int(k): v for k, v in rf["sign_test_two_sided_p_by_count"].items()}
Q999 = n20["null_quantiles"]["0.999"]
OBS = n20["observed_novel5_mean_loco_c"]


def draw(path="figs/fig_null_resolution.png"):
    apply_figure_style(sizes=(8, 7, 6))
    mpl.rcParams.update({"axes.titlesize": 7.0, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6.0, "ytick.labelsize": 6.0})
    fig = plt.figure(figsize=(180 / 25.4, 100 / 25.4))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.30, left=0.10, right=0.975,
                          top=0.905, bottom=0.115)

    axa = fig.add_subplot(gs[0, 0])
    n, _, _ = axa.hist(nd.mean_loco_c, bins=70, color="#C9D4E0", edgecolor="none")
    ymax = n.max()
    axa.axvline(OBS, color=FOC, lw=1.3)
    axa.axvline(Q999, color="#666666", lw=0.8, ls=(0, (3, 2)))
    axa.set_ylim(0, ymax * 1.30)
    axa.text(OBS, ymax * 1.28, "Novel5\n%.3f" % OBS, fontsize=5.6, color=FOC,
             ha="center", va="top", linespacing=1.25)
    axa.text(Q999 - 0.006, ymax * 1.02, "null 99.9th\n%.3f" % Q999, fontsize=5.2,
             color="#555555", ha="right", va="top", linespacing=1.25)
    axa.set_xlim(0.44, 0.69)
    axa.set_xlabel("Mean leave-one-cohort-out Harrell's C")
    axa.set_ylabel("Random 5-gene panels")
    axa.set_title("0 of %s random panels reach the panel's score\n"
                  "($z$ = %.2f, $p$ = %.0e, the design floor)"
                  % (f"{n20['n_draws']:,}", n20["observed_z_vs_null"],
                     n20["p_empirical_addone"]))

    axb = fig.add_subplot(gs[0, 1])
    yy = np.arange(len(sm))
    cols = [FOC if p in ("Novel5", "Novel5_plus_Anchor4")
            else (ANCH if p == "Anchor4" else NEUT) for p in sm.observed_panel]
    axb.barh(yy, sm.observed_percentile, color=cols, height=0.66)
    axb.axvline(95, color="#666666", lw=0.8, ls=(0, (3, 2)))
    axb.set_yticks(yy)
    axb.set_yticklabels([LBL.get(p, p) for p in sm.observed_panel], fontsize=5.6)
    axb.set_xlim(0, 116)
    axb.set_ylim(-0.7, len(sm) - 0.3)
    axb.text(96.5, -0.52, "95th", fontsize=5.2, color="#555555",
             ha="left", va="bottom")
    axb.set_xlabel("Percentile within size-matched random null")
    nabove = int((sm.observed_percentile >= 95).sum())
    axb.set_title("%d of %d sets clear the 95th percentile of a\n"
                  "null matched to their own gene count" % (nabove, len(sm)))

    axc = fig.add_subplot(gs[1, 0])
    ks = sorted(ST)
    axc.plot(ks, [ST[k] for k in ks], "-o", ms=4.0, lw=1.0, color=NEUT)
    axc.axhline(0.05, color=FOC, lw=0.8, ls=(0, (3, 2)))
    axc.scatter([0, 6], [ST[0], ST[6]], s=46, facecolor="white", edgecolor=FOC,
                zorder=4, linewidth=1.1)
    axc.set_xticks(ks)
    axc.set_xlabel("Cohorts favouring one signature (of 6)")
    axc.set_ylabel("Exact two-sided $p$")
    axc.set_ylim(-0.06, 1.16)
    axc.text(3.0, 0.085, "0.05", fontsize=5.4, color=FOC, ha="center", va="bottom")
    axc.text(3.0, 1.12, "best attainable $p$ = %.5f, at either extreme" % ST[6],
             fontsize=5.2, color="#555555", ha="center", va="top")
    axc.set_title("With six cohorts no paired test can\nreach $p$ < %.5f"
                  % rf["sign_test_min_two_sided_p"])

    axd = fig.add_subplot(gs[1, 1])
    pf = n20["observed_per_fold"]
    oo = [o for o in ORDER if o in pf]
    xs = np.arange(len(oo))
    axd.axhspan(n20["null_quantiles"]["0.5"], Q999, color="#E4EAF1", zorder=0)
    axd.axhline(n20["null_mean"], color="#888888", lw=0.8, ls=(0, (3, 2)), zorder=1)
    axd.scatter(xs, [pf[o] for o in oo], s=30, color=FOC, zorder=3)
    axd.set_xticks(xs)
    axd.set_xticklabels([SHORT[o] for o in oo], fontsize=5.4)
    axd.set_ylabel("Harrell's C")
    axd.set_ylim(0.52, 0.735)
    axd.set_xlim(-0.6, len(oo) - 0.4)
    axd.text(0.985, 0.955, "band: null median to 99.9th percentile",
             transform=axd.transAxes, fontsize=5.0, color="#666666", ha="right",
             va="top")
    nab = sum(1 for o in oo if pf[o] > Q999)
    axd.set_title("Per-fold score exceeds the null 99.9th\npercentile in %d of %d cohorts"
                  % (nab, len(oo)))

    for ax, L in [(axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")]:
        panel_letter(ax, L, case="lower")
    fig.savefig(path, dpi=400, facecolor="white")
    return fig


def verify(fig):
    """Text must not collide with other text, nor with any drawn patch."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tx = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
          if t.get_text().strip() and t.get_visible()]
    axl = ({id(a.xaxis.label) for a in fig.axes} |
           {id(a.yaxis.label) for a in fig.axes})
    tt = [(a.get_text()[:26], b.get_text()[:26])
          for i, (a, ba) in enumerate(tx) for b, bb in tx[i + 1:]
          if ba.overlaps(bb) and id(a) not in axl and id(b) not in axl]
    tp = []
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_text().strip() or not t.get_visible():
                continue
            tb = t.get_window_extent(r)
            for p in ax.patches:
                if p.get_window_extent(r).overlaps(tb):
                    tp.append((t.get_text()[:24], type(p).__name__))
    return tt, tp


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    fig = draw()
    tt, tp = verify(fig)
    assert not tt, "text-text collisions: %s" % tt
    assert not tp, "text-patch collisions: %s" % tp
    print("figs/fig_null_resolution.png written; text-text 0, text-patch 0")
