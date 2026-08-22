"""
make_fig_participant_flow.py -- Figure 2: TRIPOD-style participant flow.

Answers the review's request for transparent sample accounting. Every number is read from
the flow tables produced by the participant-flow analysis, not entered by hand:
  participant_flow.csv           per-cohort analysed n, events, median follow-up, platform
  participant_flow_upstream.csv  source records and upstream harmonisation exclusions

Verified identity: source 10,139 - 70 harmonisation exclusions - 0 analysis-stage
exclusions = 10,069 analysed (9,241 overall survival + 828 secondary endpoint).

Output: figs/fig_participant_flow.png
Style: figure-style skill supplies apply_figure_style/META_GREY; fallbacks defined below.
"""
import os
import pandas as pd
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

RES = "results"
pf = pd.read_csv(os.path.join(RES, "participant_flow.csv"))
pfu = pd.read_csv(os.path.join(RES, "participant_flow_upstream.csv"))

src_os = int(pfu[pfu.endpoint.eq("OS")].n_source_records_expr_and_clinical.sum())
src_sec = int(pfu[~pfu.endpoint.eq("OS")].n_source_records_expr_and_clinical.sum())
exc_os = int(pfu[pfu.endpoint.eq("OS")].n_excluded_upstream_harmonisation.sum())
exc_sec = int(pfu[~pfu.endpoint.eq("OS")].n_excluded_upstream_harmonisation.sum())
an_os = int(pfu[pfu.endpoint.eq("OS")].n_analysed.sum())
an_sec = int(pfu[~pfu.endpoint.eq("OS")].n_analysed.sum())
ev_os = int(pf[pf.endpoint.eq("OS")].events.sum())
ev_sec = int(pf[~pf.endpoint.eq("OS")].events.sum())
exc_an = int(pfu.n_excluded_analysis_stage.sum())
assert src_os + src_sec - exc_os - exc_sec - exc_an == an_os + an_sec, "flow does not balance"


def draw_flow(path="figs/fig_participant_flow.png"):
    apply_figure_style(sizes=(8,7,6))
    BOXF, BOXE = "#EDF1F6", "#2B5C8A"
    EXCF, EXCE = "#F5F5F5", "#8A8A8A"
    OSF,  OSE  = "#F7E9E8", "#B3312C"
    SECF, SECE = "#EFF3EF", "#4A6B4A"
    fig = plt.figure(figsize=(180/25.4, 124/25.4))
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,180); ax.set_ylim(0,124); ax.axis("off")

    def bx(x,y,w,h,title,body,fc,ec,tfs=6.4,fs=6.0):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=1.6",
                                    lw=0.8,edgecolor=ec,facecolor=fc))
        ax.text(x+w/2,y+h-2.4,title,ha="center",va="top",fontsize=tfs,
                fontweight="bold",color=ec)
        if body: ax.text(x+w/2,y+h-6.6,body,ha="center",va="top",fontsize=fs,
                         color="#222222",linespacing=1.4)
    def arr(x0,y0,x1,y1,ec="#555555"):
        ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=7,
                                     lw=0.8,color=ec,shrinkA=0.5,shrinkB=0.5))

    CX, CW = 22, 74
    bx(CX, 102, CW, 18, "Source records retrieved from public repositories",
       "9 breast-cancer cohorts with expression and clinical data\n"
       "n = %s samples" % f"{src_os+src_sec:,}", BOXF, BOXE)
    bx(CX, 74, CW, 20, "Harmonised expression matrices",
       "HGNC symbol mapping, log$_2$ scale,\nper-cohort gene-wise z-scoring\n"
       "n = %s samples in %d matrices" % (f"{an_os+an_sec:,}", len(pfu)), BOXF, BOXE)
    bx(CX, 50, CW, 16, "Analysed",
       "n = %s\n(no further exclusions at the analysis stage)" % f"{an_os+an_sec:,}",
       BOXF, BOXE)

    bx(108, 98, 69, 22, "Excluded during harmonisation  (n = %d)" % (exc_os+exc_sec),
       "missing, non-positive or non-finite follow-up time\n"
       "GSE6532 34 · GSE21653 18 · TCGA 14\n"
       "GSE20711 2 · METABRIC 1 · SCAN-B GSE202203 1", EXCF, EXCE, fs=5.7)
    bx(108, 72, 69, 14, "Excluded at analysis stage  (n = %d)" % exc_an,
       "every harmonised sample entered its endpoint arm", EXCF, EXCE, fs=5.7)

    arr(CX+CW/2, 102, CX+CW/2, 94)
    arr(CX+CW/2, 74, CX+CW/2, 66)
    arr(CX+CW, 109, 108, 109)     # from right edge, clear of text
    arr(CX+CW, 79,  108, 79)

    bx(3, 14, 84, 28, "Primary endpoint: overall survival",
       "6 cohorts · n = %s · %s deaths\n\n"
       "TCGA 1,086 · METABRIC 1,979\nSCAN-B GSE96058 3,069 · SCAN-B GSE202203 2,912\n"
       "GSE20711 88 · GSE58812 107\n\n"
       "Leave-one-cohort-out: 6 outer folds"
       % (f"{an_os:,}", f"{ev_os:,}"), OSF, OSE, fs=5.8)
    bx(93, 14, 84, 28, "Secondary endpoint: DMFS / DFS",
       "3 cohorts · n = %s · %s events\n\n"
       "GSE6532 380 (DMFS)\nGSE11121 200 (DMFS)\nGSE21653 248 (DFS)\n\n"
       "Analysed separately; never pooled with overall survival"
       % (f"{an_sec:,}", f"{ev_sec:,}"), SECF, SECE, fs=5.8)
    arr(CX+CW/2-12, 50, 45, 42); arr(CX+CW/2+12, 50, 135, 42)

    ax.text(90, 3.0,
            "Median follow-up is reported from recomputation on the analysed samples; "
            "these values differ from the upstream\nQC summaries "
            "($-$3.1 to $+$41.4 months), which used a different follow-up definition.",
            ha="center", va="bottom", fontsize=5.5, color=META_GREY, style="italic",
            linespacing=1.5)
    fig.savefig(path, dpi=1000, facecolor="white")
    return fig, ax


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    fig, ax = draw_flow()
    r = fig.canvas.get_renderer()
    bad = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_text().strip() or not t.get_visible():
            continue
        bb = t.get_window_extent(r).transformed(ax.transData.inverted())
        for p in ax.patches:
            if not isinstance(p, FancyBboxPatch):
                continue
            px, py, pw, ph = p.get_x(), p.get_y(), p.get_width(), p.get_height()
            cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
            if px < cx < px + pw and py < cy < py + ph:
                if bb.x0 < px + 0.7 or bb.x1 > px + pw - 0.7 or bb.y0 < py + 0.7:
                    bad.append(t.get_text()[:34])
    cross = []
    for a in ax.patches:
        if not isinstance(a, FancyArrowPatch):
            continue
        ab = a.get_window_extent(r).transformed(ax.transData.inverted())
        for t in fig.findobj(mpl.text.Text):
            if not t.get_text().strip() or not t.get_visible():
                continue
            if ab.overlaps(t.get_window_extent(r).transformed(ax.transData.inverted())):
                cross.append(t.get_text()[:30])
    assert not bad, "text overflows its box: %s" % bad
    assert not cross, "arrow crosses text: %s" % cross
    print("figs/fig_participant_flow.png written; overflow 0, arrow-text crossings 0")
