"""
make_fig_pipeline.py -- Figure 1: the analysis pipeline as a versioned software workflow.

Addresses the review's request that the pipeline be presented as reusable software rather
than a one-off analysis. Nine stages: cohort assembly, harmonisation, endpoint split, gene
sets, leave-one-cohort-out evaluation, selection-bias control (added in revision),
inference, sensitivity, reporting.

n = 10,069 is the total ANALYSED sample count, verified against the result files:
9,241 overall-survival (nested_selection_folds.csv n_test summed over the six folds)
+ 828 secondary-endpoint (selection_naive.csv, three DMFS/DFS cohorts).

Output: figs/fig_pipeline.png
Style: figure-style skill supplies apply_figure_style/META_GREY; fallbacks defined below.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:
    apply_figure_style, META_GREY
except NameError:
    META_GREY = "#8A8A8A"

    def apply_figure_style(sizes=(8, 7, 6)):
        mpl.rcParams.update({"font.family": "sans-serif", "font.size": sizes[0],
                             "savefig.dpi": 600})

apply_figure_style(sizes=(8, 7, 6))

STAGE, STAGE_E = "#EDF1F6", "#2B5C8A"
DATA, DATA_E = "#F4EEE8", "#8A6D3B"
OUT, OUT_E = "#F7E9E8", "#B3312C"


def draw_pipeline(path="figs/fig_pipeline.png"):
    fig = plt.figure(figsize=(180/25.4, 92/25.4))
    ax  = fig.add_axes([0,0,1,1]); ax.set_xlim(0,180); ax.set_ylim(0,92); ax.axis("off")
    def box(x,y,w,h,title,body,fc,ec,fs=6.0,tfs=6.4):
        ax.add_patch(FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0,rounding_size=1.6",
                                    linewidth=0.8, edgecolor=ec, facecolor=fc))
        ax.text(x+w/2, y+h-2.3, title, ha="center", va="top", fontsize=tfs,
                fontweight="bold", color=ec)
        if body: ax.text(x+w/2, y+h-6.4, body, ha="center", va="top", fontsize=fs,
                         color="#222222", linespacing=1.35)
    def arrow(x0,y0,x1,y1):
        ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1), arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.8, color="#555555", shrinkA=0.5, shrinkB=0.5))
    box(3, 66, 40, 23, "1  Cohort assembly",
        "9 public cohorts\nTCGA, METABRIC,\n2 $\\times$ SCAN-B, 5 $\\times$ GEO\nn = 10,069 analysed", DATA, DATA_E)
    box(49, 66, 40, 23, "2  Harmonisation",
        "HGNC symbol mapping\nlog$_2$ transform\nper-cohort gene-wise\nz-scoring", DATA, DATA_E)
    box(95, 66, 40, 23, "3  Endpoint split",
        "Overall survival: 6\nDMFS/DFS: 3\nheld-out evaluation\nnever mixes endpoints", DATA, DATA_E)
    box(141, 66, 36, 23, "4  Gene sets",
        "Novel5 + anchor\n6 published\nsignatures\n+ clinical arm", DATA, DATA_E)
    box(3, 30, 88, 29, "5  Leave-one-cohort-out evaluation",
        "Pre-specified learner: ridge Cox, $\\alpha$ = 100\n\n"
        "For each held-out cohort, fit on the pooled remaining\n"
        "cohorts of the same endpoint, then score that cohort once.\n"
        "Harrell's C and Uno's IPCW C.  No gene, hyperparameter\n"
        "or threshold is chosen using the held-out cohort.", STAGE, STAGE_E, fs=5.9)
    box(97, 30, 80, 29, "6  Selection-bias control",
        "Nested re-selection: the five-step search\nre-run inside every outer fold\n"
        "Permutation calibration: 91 replicates,\nlabels shuffled within cohort\n"
        "Selection-naive cohort subset", STAGE, STAGE_E, fs=5.9)
    box(3, 6, 55, 17, "7  Inference",
        "Cohort-stratified Cox\npooled HR, 3 specifications\nPH tests, paired bootstrap", OUT, OUT_E)
    box(64, 6, 55, 17, "8  Sensitivity",
        "Full four-learner grid\ngene-dropout subsets\ncoefficient-free scoring", OUT, OUT_E)
    box(125, 6, 52, 17, "9  Reporting",
        "TRIPOD flow accounting\nversioned scripts\nfixed seeds, recorded env", OUT, OUT_E)
    for x0,x1 in [(43,49),(89,95),(135,141)]: arrow(x0, 77.5, x1, 77.5)
    arrow(23, 66, 23, 59); arrow(159, 66, 159, 59)
    arrow(91, 44.5, 97, 44.5)
    arrow(30, 30, 30, 23); arrow(91.5, 30, 91.5, 23); arrow(151, 30, 151, 23)
    ax.text(90, 1.8, "Every stage is a versioned script; all inputs are public.",
            ha="center", va="bottom", fontsize=5.8, color=META_GREY, style="italic")
    fig.savefig(path, dpi=1000, facecolor="white")
    return fig, ax


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    draw_pipeline()
    print("figs/fig_pipeline.png written")
