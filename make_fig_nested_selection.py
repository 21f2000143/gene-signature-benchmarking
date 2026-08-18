"""
make_fig_nested_selection.py -- Figure 6: the nested re-selection result.

Reviewer item 1 (selection circularity). Four panels:
  a  per-fold nested vs published-panel vs anchor-scaffold concordance
  b  how often each gene is re-selected across the six outer folds
  c  permutation null of the search's self-reported score
  d  the two overall-survival cohorts that never took part in discovery

Inputs (results/): nested_selection_folds.csv, nested_selection_stability.csv,
  permutation_search_pooled.csv, permutation_search_summary.json, selection_naive.csv
Output: figs/fig_nested_selection.png

Style: the `figure-style` skill supplies apply_figure_style/panel_letter/META_GREY.
Standalone fallbacks are defined below so the script runs without the skill loaded.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:                      # provided by the figure-style skill kernel plugin
    apply_figure_style, panel_letter, META_GREY
except NameError:
    META_GREY = "#8A8A8A"

    def apply_figure_style(sizes=(8, 7, 6)):
        base, ann, tick = sizes
        mpl.rcParams.update({
            "font.family": "sans-serif", "font.size": base,
            "axes.titlesize": base, "axes.labelsize": base,
            "axes.titlelocation": "left", "axes.spines.top": False,
            "axes.spines.right": False, "legend.fontsize": ann,
            "xtick.labelsize": tick, "ytick.labelsize": tick,
            "axes.linewidth": 0.7, "xtick.major.width": 0.7,
            "ytick.major.width": 0.7, "savefig.dpi": 400,
        })

    def panel_letter(ax, letter, case="lower"):
        s = letter.lower() if case == "lower" else letter.upper()
        ax.text(-0.16, 1.10, s, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left")

RES = "results"
apply_figure_style(sizes=(8, 7, 6))

FOC, NEST, ANCH, NULLC = "#B3312C", "#2B5C8A", "#9A9A9A", "#C8C8C8"

nf = pd.read_csv(os.path.join(RES, "nested_selection_folds.csv"))
st = pd.read_csv(os.path.join(RES, "nested_selection_stability.csv"))
perm = pd.read_csv(os.path.join(RES, "permutation_search_pooled.csv"))
ps = json.load(open(os.path.join(RES, "permutation_search_summary.json")))
sn = pd.read_csv(os.path.join(RES, "selection_naive.csv"))

order = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203",
         "GSE20711", "GSE58812"]
short = {"TCGA": "TCGA", "METABRIC": "METABRIC",
         "SCANB_GSE96058": "SCAN-B\n96058", "SCANB_GSE202203": "SCAN-B\n202203",
         "GSE20711": "GSE20711", "GSE58812": "GSE58812"}
nfi = nf.set_index("held_out_cohort").loc[order]
inner = perm.inner_loco_c_permuted.values

fig = plt.figure(figsize=(180 / 25.4, 142 / 25.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.46, wspace=0.30,
                      left=0.085, right=0.975, top=0.915, bottom=0.105)
x = np.arange(len(order))
w = 0.27

# ---- (a) nested vs published vs anchor, per outer fold ----
axa = fig.add_subplot(gs[0, 0])
axa.bar(x - w, nfi.cindex_published_panel, w, color=FOC, label="Published panel")
axa.bar(x, nfi.cindex_nested, w, color=NEST, label="Nested re-selection")
axa.bar(x + w, nfi.cindex_anchor_only, w, color=ANCH, label="Anchor scaffold only")
axa.axhline(0.5, color="k", lw=0.6, ls=(0, (4, 3)))
axa.text(len(order) - 0.40, 0.506, "chance", fontsize=6, ha="right", va="bottom")
axa.set_xticks(x)
axa.set_xticklabels([short[o] for o in order], fontsize=5.6)
axa.set_ylabel("Concordance index")
axa.set_ylim(0.40, 0.84)
axa.set_yticks(np.arange(0.40, 0.76, 0.05))
axa.margins(x=0.05)
axa.set_title("Re-selecting the panel inside every fold\ncosts only 0.012 concordance")
axa.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.1,
           borderpad=0.1, labelspacing=0.25)
axa.text(0.985, 0.955, "higher = better", transform=axa.transAxes, fontsize=6,
         color=META_GREY, ha="right", va="top")

# ---- (b) selection stability across folds ----
axb = fig.add_subplot(gs[0, 1])
stv = st.sort_values(["n_folds_selected", "gene"], ascending=[True, False])
cols = [FOC if p else NEST for p in stv.in_published_panel]
axb.barh(np.arange(len(stv)), stv.n_folds_selected, color=cols, height=0.72)
axb.set_yticks(np.arange(len(stv)))
axb.set_yticklabels(["$\\it{%s}$" % g for g in stv.gene], fontsize=6)
axb.set_xlabel("Outer folds selecting the gene (of 6)")
axb.set_xlim(0, 6.4)
axb.set_xticks(range(0, 7))
axb.set_title("No gene is selected in all six folds")
axb.legend(handles=[Patch(color=FOC, label="In published panel"),
                    Patch(color=NEST, label="Not in published panel")],
           frameon=False, fontsize=6, loc="lower right", handlelength=1.1,
           borderpad=0.1)

# ---- (c) permutation null of the search statistic ----
axc = fig.add_subplot(gs[1, 0])
axc.hist(inner, bins=18, color=NULLC, edgecolor="white", lw=0.4)
axc.axvline(0.5, color="k", lw=0.6, ls=(0, (4, 3)))
axc.axvline(float(ps["observed_inner_search_score"]), color=FOC, lw=1.6)
ymax = axc.get_ylim()[1]
axc.set_ylim(0, ymax * 1.20)
axc.annotate("observed %.3f" % ps["observed_inner_search_score"],
             xy=(ps["observed_inner_search_score"], ymax * 0.98),
             xytext=(ps["observed_inner_search_score"] - 0.006, ymax * 1.15),
             fontsize=6, color=FOC, ha="right", va="top")
axc.text(0.556, ymax * 1.17,
         "null mean %.3f\n(+%.2f above chance\non permuted labels)"
         % (inner.mean(), inner.mean() - 0.5),
         fontsize=6, color="#333333", ha="left", va="top")
axc.text(0.503, ymax * 0.04, "chance", fontsize=6, ha="left", va="bottom")
axc.set_xlabel("Score the search reports for its own panel")
axc.set_ylabel("Permutation replicates")
axc.set_xlim(0.49, 0.725)
axc.set_title("The search statistic is inflated;\nthe panel still beats every null")

# ---- (d) selection-naive overall-survival cohorts ----
axd = fig.add_subplot(gs[1, 1])
naive_os = ["GSE20711", "GSE58812"]
sn_os = sn[sn.cohort.isin(naive_os)]
xs = np.arange(2)
w2 = 0.2
for j, (pan, col) in enumerate([("Novel5", FOC), ("Anchor4", ANCH)]):
    lab = "Panel" if pan == "Novel5" else "Anchor"
    d = sn_os[sn_os.panel.eq(pan)].set_index("cohort").loc[naive_os]
    axd.bar(xs + (j * 2 - 1) * w2 * 1.05 - w2 * 0.55,
            d.cindex_trained_on_selection_cohorts, w2, color=col,
            label="%s, trained on discovery cohorts" % lab)
    axd.bar(xs + (j * 2 - 1) * w2 * 1.05 + w2 * 0.55,
            d.cindex_trained_on_naive_peers, w2, color=col, alpha=0.45,
            hatch="///", edgecolor="white", lw=0.3,
            label="%s, trained on naive peers" % lab)
axd.axhline(0.5, color="k", lw=0.6, ls=(0, (4, 3)))
axd.set_xticks(xs)
axd.set_xticklabels(["GSE20711\n(25 events)", "GSE58812\n(29 events)"], fontsize=6)
axd.set_ylabel("Concordance index")
axd.set_ylim(0.38, 0.86)
axd.set_yticks(np.arange(0.40, 0.76, 0.05))
axd.set_title("On cohorts never seen by the search,\nthe panel holds at 0.65")
axd.legend(frameon=False, fontsize=5.6, loc="upper left", handlelength=1.1,
           borderpad=0.1, labelspacing=0.22)

for ax, L in [(axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")]:
    panel_letter(ax, L, case="lower")

os.makedirs("figs", exist_ok=True)
fig.savefig("figs/fig_nested_selection.png", dpi=400, facecolor="white")

# --- render-then-verify: no text-text collisions outside axis labels ---
r = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible()]
axis_labels = {id(ax.xaxis.label) for ax in fig.axes} | \
              {id(ax.yaxis.label) for ax in fig.axes}
ov = [(a.get_text()[:24], b.get_text()[:24])
      for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
      if ba.overlaps(bb) and id(a) not in axis_labels and id(b) not in axis_labels]
assert not ov, "text collisions: %s" % ov
print("figs/fig_nested_selection.png written; text collisions: 0")
