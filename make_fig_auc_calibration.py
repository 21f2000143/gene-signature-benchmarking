"""
make_fig_auc_calibration.py -- Figure 4: time-dependent AUC and calibration.

Revised for the review. The original claimed a five-year AUC of 0.70; recomputation shows
that value is not reached in ANY cohort where it is estimable (range 0.661-0.696), and that
METABRIC yields no AUC at all under the pre-specified censoring specification because its
censoring survival function reaches zero. Both facts are now shown rather than smoothed over.

Panels: a  AUC against horizon per cohort
        b  five-year AUC per cohort; METABRIC marked not estimable
        c  observed five-year survival by risk quintile (calibration)
        d  five-year AUC under three censoring-time specifications

Note on encoding: panels b and d use dots, not bars, because their y-axes are truncated;
bar length on a non-zero baseline would misrepresent the values.

Inputs (results/): time_dependent_auc_revised.csv, tauc_censoring_sensitivity.csv,
  calibration_quintiles.csv
Output: figs/fig_auc_calibration.png
Style: figure-style skill supplies apply_figure_style/panel_letter; fallbacks defined below.
"""
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

ORDER = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203", "GSE20711", "GSE58812"]
SHORT = {"TCGA": "TCGA", "METABRIC": "META-\nBRIC", "SCANB_GSE96058": "SCAN-B\n96058",
         "SCANB_GSE202203": "SCAN-B\n202203", "GSE20711": "GSE\n20711",
         "GSE58812": "GSE\n58812"}
CC = {"TCGA": "#2B5C8A", "METABRIC": "#8A6D3B", "SCANB_GSE96058": "#4A6B4A",
      "SCANB_GSE202203": "#6B4A6B", "GSE20711": "#B3312C", "GSE58812": "#C77B2B"}
SPEC = {"A_train_maxtau": "train max $\\tau$", "B_train_safetau": "train safe $\\tau$",
        "C_test_own": "test-cohort own $\\tau$"}
MK = {"A_train_maxtau": "o", "B_train_safetau": "s", "C_test_own": "^"}
OFF = {"A_train_maxtau": -0.20, "B_train_safetau": 0.0, "C_test_own": 0.20}

RES = "results"
tda = pd.read_csv(os.path.join(RES, "time_dependent_auc_revised.csv"))
tcs = pd.read_csv(os.path.join(RES, "tauc_censoring_sensitivity.csv"))
cal = pd.read_csv(os.path.join(RES, "calibration_quintiles.csv"))

a5 = tda[tda.horizon_years.eq(5.0)]
vals = [float(a5[a5.held_out_cohort.eq(o)].auc.iloc[0]) for o in ORDER]
v5 = a5.auc.dropna()
NEST = int(v5.notna().sum())


def draw(path="figs/fig_auc_calibration.png"):
    apply_figure_style(sizes=(8, 7, 6))
    mpl.rcParams.update({"axes.titlesize": 7.0, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6.0, "ytick.labelsize": 6.0})
    fig = plt.figure(figsize=(180 / 25.4, 104 / 25.4))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.30, left=0.085, right=0.975,
                          top=0.905, bottom=0.105)
    xs = np.arange(len(ORDER))

    axa = fig.add_subplot(gs[0, 0])
    for coh in ORDER:
        ok = tda[tda.held_out_cohort.eq(coh)].sort_values("horizon_years").dropna(
            subset=["auc"])
        if len(ok):
            axa.plot(ok.horizon_years, ok.auc, "-o", ms=3.0, lw=1.0, color=CC[coh],
                     label=coh.replace("SCANB_", "SCAN-B "))
    # axa.axhline(0.70, color="#B3312C", lw=0.8, ls=(0, (3, 2)))
    # axa.text(10.2, 0.703, "claimed 0.70", fontsize=5.6, color="#B3312C", ha="right",
    #          va="bottom")
    axa.set_xticks([3, 5, 10])
    axa.set_xlabel("Horizon (years)")
    axa.set_ylabel("Time-dependent AUC")
    axa.set_ylim(0.60, 0.80)
    axa.set_title("Five-year AUC reaches 0.70 in none of the\n"
                  "%d cohorts where it is estimable" % NEST)
    axa.legend(frameon=False, fontsize=5.2, loc="lower left", handlelength=1.2,
               borderpad=0.1, labelspacing=0.2)
    axa.text(0.985, 0.045, "METABRIC not estimable at any horizon",
             transform=axa.transAxes, fontsize=5.0, ha="right", va="bottom",
             color="#666666")

    axb = fig.add_subplot(gs[0, 1])
    for i, v in enumerate(vals):
        if np.isnan(v):
            axb.scatter(i, 0.681, s=30, marker="x", color="#9A9A9A", linewidth=1.1)
            axb.text(i, 0.6845, "not\nestimable", fontsize=4.9, ha="center",
                     va="bottom", color="#666666", linespacing=1.2)
        else:
            axb.scatter(i, v, s=30, color="#2B5C8A", zorder=3)
            axb.text(i, v + 0.0025, "%.3f" % v, fontsize=5.2, ha="center", va="bottom")
    # axb.axhline(0.70, color="#B3312C", lw=0.8, ls=(0, (3, 2)))
    # axb.text(5.35, 0.7015, "0.70", fontsize=5.4, color="#B3312C", ha="right",
    #          va="bottom")
    axb.set_xticks(xs)
    axb.set_xticklabels([SHORT[o] for o in ORDER], fontsize=5.4)
    axb.set_xlim(-0.6, 5.6)
    axb.set_ylim(0.652, 0.712)
    axb.set_ylabel("Five-year AUC")
    axb.set_title("Five-year AUC %.3f$-$%.3f; 0 of %d estimable\ncohorts reach 0.70"
                  % (v5.min(), v5.max(), NEST))

    axc = fig.add_subplot(gs[1, 0])
    for coh in cal.cohort.unique():
        s = cal[cal.cohort.eq(coh)].sort_values("quintile")
        lab = "monotone" if bool(s.strictly_monotone.iloc[0]) else \
              "%d reversals" % int(s.n_reversals.iloc[0])
        axc.plot(s.quintile, s.obs_surv, "-o", ms=3.0, lw=1.0, color=CC[coh],
                 label="%s (%s)" % (coh.replace("SCANB_", "SCAN-B "), lab))
        axc.fill_between(s.quintile, s.obs_lo, s.obs_hi, color=CC[coh], alpha=0.10, lw=0)
    nmono = int(cal.groupby("cohort").strictly_monotone.first().sum())
    ncoh = cal.cohort.nunique()
    axc.set_xticks([1, 2, 3, 4, 5])
    axc.set_xlabel("Risk-score quintile (1 = lowest risk)")
    axc.set_ylabel("Observed 5-year survival")
    axc.set_ylim(0.35, 1.0)
    axc.set_title("Risk ordering is monotone in %d of %d cohorts" % (nmono, ncoh))
    axc.legend(frameon=False, fontsize=5.2, loc="lower left", handlelength=1.2,
               borderpad=0.1, labelspacing=0.2)

    axd = fig.add_subplot(gs[1, 1])
    for sp, c in [("A_train_maxtau", "#2B5C8A"), ("B_train_safetau", "#6B4A6B"),
                  ("C_test_own", "#4A6B4A")]:
        xv, yv = [], []
        for i, o in enumerate(ORDER):
            q = tcs[tcs.held_out_cohort.eq(o) & tcs.spec.eq(sp)
                    & tcs.horizon_years.eq(5.0)].auc
            if len(q) and not np.isnan(float(q.iloc[0])):
                xv.append(i + OFF[sp])
                yv.append(float(q.iloc[0]))
        axd.scatter(xv, yv, s=24, marker=MK[sp], color=c, label=SPEC[sp], zorder=3)
    # axd.axhline(0.70, color="#B3312C", lw=0.8, ls=(0, (3, 2)))
    for i in range(len(ORDER)):
        axd.axvline(i + 0.5, color="#DDDDDD", lw=0.5, zorder=0)
    axd.annotate("only estimable\nwith test-cohort $\\tau$", xy=(1.20, 0.6805),
                 xytext=(1.02, 0.6555), fontsize=5.0, color="#555555", ha="center",
                 va="top", linespacing=1.25,
                 arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, shrinkA=1,
                                 shrinkB=2))
    axd.set_xticks(xs)
    axd.set_xticklabels([SHORT[o] for o in ORDER], fontsize=5.4)
    axd.set_xlim(-0.6, 5.6)
    axd.set_ylim(0.625, 0.716)
    axd.set_ylabel("Five-year AUC")
    axd.set_title("METABRIC is estimable only when $\\tau$ is\n"
                  "taken from the test cohort itself")
    axd.legend(frameon=False, fontsize=5.2, loc="lower right", handlelength=1.0,
               borderpad=0.1, labelspacing=0.22, handletextpad=0.3)

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
    print("figs/fig_auc_calibration.png written; text-text 0, text-patch 0")
