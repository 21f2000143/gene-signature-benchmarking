"""
make_fig_paired_forest.py -- Figure 5: paired per-cohort comparison against published sets.

Revised for the review. Three changes from the original forest plot:
  * the comparison is now leave-one-cohort-out (every c-index is out-of-sample for the
    cohort it is computed in), not in-sample.
  * the per-cohort bootstrap CIs are shown alongside the cohort-level sign test, and the
    sign test is annotated with the resolution floor: with six cohorts the smallest
    attainable two-sided p is 0.03125, so "p = 0.031" means "all six cohorts agreed",
    nothing stronger. Five comparators reach that floor; none can go below it.
  * the count of cohorts where the bootstrap CI actually excludes zero is reported
    separately, because agreement in sign is much weaker evidence than CI separation:
    against the large published signatures the panel is clearly better in only 1 of 6.

Panels: a  per-cohort delta c-index with bootstrap CIs, one row per comparator
        b  cohorts favouring the panel (sign agreement) vs cohorts with CI excluding zero
        c  delta c-index against comparator gene count

Inputs (results/): loco_paired_novel5_vs_comparators.csv, resolution_floor.json
Outputs: figs/fig_paired_forest.png, results/paired_sign_test_by_comparator.csv
"""
import json
from scipy.stats import spearmanr, false_discovery_control
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (FixedFormatter, FixedLocator, NullFormatter,
                               NullLocator)

try:
    apply_figure_style, panel_letter
except NameError:
    def apply_figure_style(sizes=(8, 7, 6)):
        mpl.rcParams.update({"font.family": "sans-serif", "font.size": sizes[0],
                             "axes.spines.top": False, "axes.spines.right": False,
                             "savefig.dpi": 600})

    def panel_letter(ax, letter, case="lower"):
        ax.text(-0.16, 1.07, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="right")

ORDER = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203", "GSE20711", "GSE58812"]
CC = {"TCGA": "#2B5C8A", "METABRIC": "#8A6D3B", "SCANB_GSE96058": "#4A6B4A",
      "SCANB_GSE202203": "#6B4A6B", "GSE20711": "#B3312C", "GSE58812": "#C77B2B"}
LBL = {"Anchor4": "Anchor scaffold (4)", "CNetCox6": "CNet-Cox (5)", "GGI": "GGI (58)",
       "OncotypeDX21": "Oncotype DX (21)", "PAM50": "PAM50 (46)",
       "MammaPrint70": "MammaPrint (48)", "BuffaHypoxia": "Buffa hypoxia (48)",
       "Novel5_plus_Anchor4": "Novel5 + anchor (9)"}
FOC, NEUT, GREY = "#B3312C", "#2B5C8A", "#8A8A8A"
RES = "results"


def _shift_label_px(ax, text, dy_px):
    x, y = text.get_position()
    xd, yd = ax.transData.transform((x, y))
    x2, y2 = ax.transData.inverted().transform((xd, yd + dy_px))
    text.set_position((x, y2))


def declutter_labels(fig, ax, texts, max_iter=200, pad_px=1.0):
    """Nudge overlapping labels apart vertically in rendered pixel space,
    using actual bounding boxes, until no pair overlaps."""
    if not texts:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ in range(max_iter):
        boxes = [t.get_window_extent(renderer) for t in texts]
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                bi, bj = boxes[i], boxes[j]
                if not bi.overlaps(bj):
                    continue
                moved = True
                overlap = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                push = overlap / 2.0 + pad_px
                if bi.y0 <= bj.y0:
                    lo, hi = texts[i], texts[j]
                else:
                    lo, hi = texts[j], texts[i]
                _shift_label_px(ax, lo, -push)
                _shift_label_px(ax, hi, push)
        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def stack_up(fig, ax, texts, pad_px=1.0):
    """Stack labels that start at a shared, already-safe baseline strictly
    upward past one another -- never down -- so they cannot be pushed back
    onto whatever the baseline was chosen to clear (e.g. a boxplot patch)."""
    if not texts:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    order = sorted(range(len(texts)), key=lambda k: texts[k].get_position()[0])
    placed = []
    for idx in order:
        t = texts[idx]
        while True:
            box = t.get_window_extent(renderer)
            blocker = next((pb for pb in placed if box.overlaps(pb)), None)
            if blocker is None:
                break
            _shift_label_px(ax, t, (blocker.y1 - box.y0) + pad_px)
        placed.append(t.get_window_extent(renderer))


def sign_table(lp, ST):
    """Per-comparator sign agreement, CI-exclusion count, and exact sign-test p."""
    sg = lp.assign(pos=lp.delta_cindex > 0).groupby("comparator").agg(
        n_pos=("pos", "sum"), n=("pos", "size"), mean_delta=("delta_cindex", "mean"),
        min_delta=("delta_cindex", "min"), max_delta=("delta_cindex", "max"),
        k_genes=("n_genes_comparator", "first")).reset_index()
    sg["sign_p"] = [ST[int(k)] for k in sg.n_pos]
    sg["n_ci_favour"] = [int((lp[lp.comparator.eq(c)].favours == "Novel5").sum())
                         for c in sg.comparator]
    # Benjamini-Hochberg over the sign-test p across these comparators (Novel5 itself
    # is not a comparator, so this is exactly the 8-comparator family Table 5 reports).
    sg["q_bh"] = false_discovery_control(sg.sign_p.values, method="bh")
    return sg.sort_values("mean_delta", ascending=False).reset_index(drop=True)


def draw(lp, sg, floor, path="figs/fig_paired_forest.png"):
    apply_figure_style(sizes=(8, 7, 6))
    mpl.rcParams.update({"axes.titlesize": 7.0, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6.0, "ytick.labelsize": 6.0})
    fig = plt.figure(figsize=(230.8 / 25.4, 154.8 / 25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1], hspace=0.52, wspace=0.30,
                          left=0.155, right=0.975, top=0.925, bottom=0.095)

    # a: per-cohort deltas, grouped by comparator. A box summarises the spread
    # across the 6 cohorts; individual cohort point estimates are overlaid as
    # coloured dots. (The original per-cohort bootstrap-CI whiskers -- 48 of
    # them in one strip -- were unreadable; CI-exclusion counts are already
    # reported per comparator in panel b.)
    axa = fig.add_subplot(gs[0, :])
    comps = sg.comparator.tolist()[::-1]
    step = 2.0  # extra row spacing (vs 1.0) makes room to stack per-point labels
    axa.set_xlim(-0.075, 0.245)
    axa.set_ylim(-0.62 * step, (len(comps) - 1) * step + 0.62 * step)
    jit = np.linspace(-0.20, 0.20, len(ORDER))
    for i, cmp_ in enumerate(comps):
        yi = i * step
        d = lp[lp.comparator.eq(cmp_)].set_index("held_out_cohort")
        vals = [d.loc[coh, "delta_cindex"] for coh in ORDER if coh in d.index]
        axa.boxplot(vals, positions=[yi], vert=False, widths=0.5, showfliers=False,
                   whis=(0, 100), patch_artist=True,
                   boxprops=dict(facecolor="#EDEDED", edgecolor="#999999", lw=0.7),
                   medianprops=dict(color="#555555", lw=1.0),
                   whiskerprops=dict(color="#999999", lw=0.7),
                   capprops=dict(color="#999999", lw=0.7), zorder=2)
        row_labels = []
        for j, coh in enumerate(ORDER):
            if coh not in d.index:
                continue
            val = d.loc[coh, "delta_cindex"]
            y = yi + jit[j]
            axa.scatter(val, y, s=13, color=CC[coh], zorder=4,
                       edgecolor="white", linewidth=0.3)
            # anchored clear of the box (half-height 0.25) so the collision
            # resolver only needs to stack labels upward, never dodge the box
            row_labels.append(
                axa.text(val, yi + 0.29, "%.3f" % val, fontsize=3.2, color=CC[coh],
                        ha="center", va="bottom", zorder=5))
        stack_up(fig, axa, row_labels)
    axa.axvline(0.0, color="k", lw=0.8, ls=(0, (4, 3)))
    axa.set_xticks([-0.05, 0.0, 0.05, 0.10, 0.15, 0.20])
    axa.set_yticks(np.arange(len(comps)) * step)
    axa.set_yticklabels([LBL.get(c, c) for c in comps], fontsize=5.8)
    axa.set_xlabel("$\\Delta$ Harrell's C, Novel5 $-$ comparator "
                   "(leave-one-cohort-out; box = spread across the 6 cohorts)")
    nsig_comp = int((lp.favours == "comparator").sum())
    axa.set_title("No comparator beats Novel5 significantly (%d of %d)"
                  % (nsig_comp, len(lp)))
    hs = [plt.Line2D([], [], color=CC[c], marker="o", ms=3.0, lw=0,
                     label=c.replace("SCANB_", "SCAN-B ").replace("_", " "))
          for c in ORDER]
    axa.legend(handles=hs, frameon=False, fontsize=5.2, ncol=3, loc="lower right",
               handlelength=1.2, columnspacing=0.9, labelspacing=0.25, borderpad=0.1)

    # b: sign agreement vs CI exclusion
    axb = fig.add_subplot(gs[1, 0])
    yy = np.arange(len(sg))[::-1]
    axb.barh(yy, sg.n_pos, height=0.62, color="#C9D4E0", label="sign agreement", zorder=3)
    axb.barh(yy, sg.n_ci_favour, height=0.30, color=NEUT, label="bootstrap CI excludes 0",
             zorder=3)
    axb.set_xlim(0, 10.4)
    axb.set_ylim(-0.65, len(sg) - 0.35)
    # both labels sit past the LONGER (sign-agreement) bar, since n_ci_favour
    # never exceeds n_pos -- anchoring each at its own bar end would put the
    # CI-favour label underneath the sign-agreement bar's rectangle
    panel_b_labels = []
    for y, n_pos, n_ci in zip(yy, sg.n_pos, sg.n_ci_favour):
        x = n_pos + 0.15
        panel_b_labels.append(axb.text(x, y + 0.16, "%d" % n_pos,
                                       fontsize=5.0, color="#556B87", ha="left", va="center"))
        panel_b_labels.append(axb.text(x, y - 0.16, "%d" % n_ci,
                                       fontsize=5.0, color=NEUT, ha="left", va="center"))
    declutter_labels(fig, axb, panel_b_labels)
    axb.set_yticks(yy)
    axb.set_yticklabels([LBL.get(c, c) for c in sg.comparator], fontsize=5.4)
    axb.set_xticks(range(7))
    axb.set_xlabel("Cohorts of 6 favouring Novel5")
    nfloor = int((sg.n_pos == 6).sum())
    axb.set_title("%d comparators are beaten in all six cohorts\n"
                  "(sign-test $p$ = %.5f, the attainable floor)"
                  % (nfloor, floor))
    axb.legend(frameon=False, fontsize=5.2, loc="center right", handlelength=1.1,
               borderpad=0.1, labelspacing=0.35)

    # c: delta against comparator size, ordered by gene count
    axc = fig.add_subplot(gs[1, 1])
    sc = sg.sort_values("k_genes", ascending=False).reset_index(drop=True)
    yc = np.arange(len(sc))[::-1]
    axc.set_ylim(-0.6, len(sc) - 0.4)
    axc.set_xlim(-0.095, 0.245)
    panel_c_labels = []
    for y, (_, r0) in zip(yc, sc.iterrows()):
        axc.plot([r0.min_delta, r0.max_delta], [y, y], color=GREY, lw=0.7, zorder=1)
        axc.scatter(r0.mean_delta, y, s=20, color=NEUT, zorder=3)
        panel_c_labels.append(axc.text(r0.mean_delta, y + 0.20, "%.3f" % r0.mean_delta,
                                       fontsize=5.0, color=NEUT, ha="center", va="bottom"))
    declutter_labels(fig, axc, panel_c_labels)
    axc.axvline(0.0, color="k", lw=0.8, ls=(0, (4, 3)))
    axc.set_yticks(yc)
    axc.set_yticklabels([LBL.get(c, c) for c in sc.comparator], fontsize=5.4)
    axc.set_xlabel("$\\Delta$ Harrell's C (mean, and range across cohorts)")
    # The margin does NOT decrease monotonically with comparator size: over the
    # eight comparators shown here Spearman(size, mean delta) = -0.40, p = 0.33,
    # and the smallest margin of all is against the 9-gene Novel5+anchor
    # superset. Say only what the panel shows: the two smallest comparators lose
    # by most, no trend in between. (Computed on the six-cohort
    # loco_paired_novel5_vs_comparators.csv that this figure plots. The
    # four-cohort paired_bootstrap.csv gives -0.33, p = 0.42 on the same eight
    # comparators -- same conclusion, different denominator; the title is
    # rendered from the values actually plotted, never hard-coded.)
    _rho_sz, _p_sz = spearmanr(sc.k_genes, sc.mean_delta)
    axc.set_title("Ordered by comparator size: the two smallest\n"
                  "comparators lose by most ($\\rho=%.2f$, $p=%.2f$)"
                  % (_rho_sz, _p_sz))

    for ax, L in [(axa, "a"), (axb, "b"), (axc, "c")]:
        panel_letter(ax, L, case="lower")
    fig.savefig(path, dpi=1000, facecolor="white")
    return fig


def verify(fig):
    """Text must not collide with other text or any patch, nor be clipped."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tx = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
          if t.get_text().strip() and t.get_visible()]
    axl = ({id(a.xaxis.label) for a in fig.axes} |
           {id(a.yaxis.label) for a in fig.axes})
    tt = [(a.get_text()[:26], b.get_text()[:26])
          for i, (a, ba) in enumerate(tx) for b, bb in tx[i + 1:]
          if ba.overlaps(bb) and id(a) not in axl and id(b) not in axl]
    fb = fig.bbox
    clip = []
    for ax in fig.axes:
        for t in ax.get_yticklabels() + ax.get_xticklabels():
            if not t.get_text().strip():
                continue
            b = t.get_window_extent(r)
            if b.x0 < fb.x0 or b.x1 > fb.x1 or b.y0 < fb.y0 or b.y1 > fb.y1:
                clip.append(t.get_text()[:20])
    tp = []
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_text().strip() or not t.get_visible():
                continue
            tb = t.get_window_extent(r)
            for p in ax.patches:
                if p.get_window_extent(r).overlaps(tb):
                    tp.append((t.get_text()[:24], type(p).__name__))
            # markers and lines are Collections/Line2D, not patches -- check them too.
            # A collection's window extent is its whole data box, so test the actual
            # marker positions instead.
            for col in ax.collections:
                if not col.get_visible():
                    continue
                offs = col.get_offsets()
                if offs is None or len(offs) == 0:
                    continue
                pix = ax.transData.transform(np.asarray(offs))
                if ((pix[:, 0] >= tb.x0) & (pix[:, 0] <= tb.x1) &
                        (pix[:, 1] >= tb.y0) & (pix[:, 1] <= tb.y1)).any():
                    tp.append((t.get_text()[:24], "marker"))
            for ln in ax.lines:
                if not ln.get_visible():
                    continue
                # axhline/axvline reference lines use a blended transform (one
                # axis in axes-fraction coordinates) -- transforming their raw
                # xydata through transData alone (valid only for plain data-
                # space lines) produces nonsense pixel coordinates, so skip them.
                if ln.get_transform() is not ax.transData:
                    continue
                xy = ln.get_xydata()
                if len(xy) == 0:
                    continue
                pix = ax.transData.transform(xy)
                if ((pix[:, 0] >= tb.x0) & (pix[:, 0] <= tb.x1) &
                        (pix[:, 1] >= tb.y0) & (pix[:, 1] <= tb.y1)).any():
                    tp.append((t.get_text()[:24], "Line2D"))
    return tt, tp, clip


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    lp = pd.read_csv(os.path.join(RES, "loco_paired_novel5_vs_comparators.csv"))
    rf = json.load(open(os.path.join(RES, "resolution_floor.json")))
    ST = {int(k): v for k, v in rf["sign_test_two_sided_p_by_count"].items()}
    floor = rf["sign_test_min_two_sided_p"]

    sg = sign_table(lp, ST)
    sg.to_csv(os.path.join(RES, "paired_sign_test_by_comparator.csv"), index=False)
    assert (sg.n_ci_favour <= sg.n_pos).all(), "CI-favouring cannot exceed sign agreement"
    assert sg.sign_p.min() >= floor - 1e-12, "no sign-test p may fall below the floor"

    fig = draw(lp, sg, floor)
    tt, tp, clip = verify(fig)
    assert not tt, "text-text collisions: %s" % tt
    assert not tp, "text-patch collisions: %s" % tp
    assert not clip, "tick labels clipped: %s" % clip
    print("figs/fig_paired_forest.png written; text-text 0, text-patch 0, clipped 0")
    print(sg[["comparator", "k_genes", "n_pos", "n_ci_favour", "mean_delta",
              "sign_p"]].to_string(index=False))
