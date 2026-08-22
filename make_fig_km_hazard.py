"""
make_fig_km_hazard.py -- Figure 3: risk stratification and the time-varying effect.

Revised for the review. The original figure presented Kaplan-Meier curves plus a single
hazard ratio. The Grambsch-Therneau test shows proportional hazards is VIOLATED in the
pooled fit and in the two largest cohorts, so a single HR is a weighted average of an
effect that changes with follow-up. This figure therefore reports the decay explicitly:
the high-vs-low tertile HR falls from ~5.1 in the first three years to ~0.97 beyond ten,
and the cohort panel flags which cohorts violate the assumption.

Panels: a  pooled Kaplan-Meier by out-of-fold risk tertile, Greenwood 95% bands
        b  cohort-stratified HR within successive follow-up windows (the PH violation)
        c  per-cohort high-vs-low HR, red where PH is violated at 0.05
        d  pooled HR under three specifications (unstratified / stratified / per SD)

Note on splits: panels a and c use TERTILES (matching km_stratification.csv); panel d
uses the within-cohort MEDIAN split (matching the pooled Cox specification). The two are
not interchangeable and the axis labels state which is which.

Inputs (results/): loco_risk_scores.csv, km_stratification.csv, ph_tests.csv,
  pooled_hr_stratified.csv, pooled_hr_summary.json
Outputs: figs/fig_km_hazard.png, results/timevarying_hr_windows.csv
Survival estimators (Kaplan-Meier + Greenwood CI, Newton-Raphson Cox with strata) are
implemented here because lifelines/scikit-survival are not installed in this environment.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, NullLocator

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
FLAT = {"TCGA": "TCGA", "METABRIC": "METABRIC", "SCANB_GSE96058": "SCAN-B 96058",
        "SCANB_GSE202203": "SCAN-B 202203", "GSE20711": "GSE 20711",
        "GSE58812": "GSE 58812"}
GC = {"low": "#2B5C8A", "intermediate": "#8A8A8A", "high": "#B3312C"}
FOC, NEUT = "#B3312C", "#2B5C8A"
RES = "results"


def km_curve(t, e):
    """Kaplan-Meier survival with Greenwood pointwise 95% confidence limits."""
    o = np.argsort(t)
    t, e = np.asarray(t, float)[o], np.asarray(e, int)[o]
    S, V, out = 1.0, 0.0, [(0.0, 1.0, 1.0, 1.0)]
    for u in np.unique(t[e == 1]):
        nr = int((t >= u).sum())
        d = int(((t == u) & (e == 1)).sum())
        if nr == 0:
            break
        S *= (1 - d / nr)
        V += d / (nr * (nr - d)) if nr > d else 0.0
        se = S * np.sqrt(V)
        out.append((u, S, max(0.0, S - 1.96 * se), min(1.0, S + 1.96 * se)))
    return pd.DataFrame(out, columns=["t", "S", "lo", "hi"])


def coxph_beta(t, e, x, strata=None):
    """Cox partial-likelihood beta by Newton-Raphson, Breslow ties, optional strata."""
    t, e, x = np.asarray(t, float), np.asarray(e, int), np.asarray(x, float)
    s = np.zeros(len(t), int) if strata is None else pd.factorize(strata)[0]
    b = 0.0
    h = np.nan
    for _ in range(60):
        g = h = 0.0
        for k in np.unique(s):
            m = s == k
            tk, ek, xk = t[m], e[m], x[m]
            for u in np.unique(tk[ek == 1]):
                at = tk >= u
                d = int(((tk == u) & (ek == 1)).sum())
                w = np.exp(b * xk[at])
                s0 = w.sum()
                s1 = (w * xk[at]).sum()
                s2 = (w * xk[at] ** 2).sum()
                g += xk[(tk == u) & (ek == 1)].sum() - d * s1 / s0
                h += d * (s2 / s0 - (s1 / s0) ** 2)
        if h <= 0:
            break
        step = g / h
        b += step
        if abs(step) < 1e-9:
            break
    return b, (1 / np.sqrt(h) if h and h > 0 else np.nan)


def stratified_logrank(time, event, group, strata):
    """Cohort-stratified K-sample log-rank test (Mantel-Haenszel generalisation,
    matches R's survival::survdiff(..., strata=)). Returns (chi2, df, p)."""
    from scipy.stats import chi2 as chi2dist
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    group = np.asarray(group)
    strata = np.asarray(strata)
    groups = sorted(set(group))
    k = len(groups)
    OE = np.zeros(k)
    V = np.zeros((k, k))
    for s in set(strata):
        ms = strata == s
        ts, es, gs = time[ms], event[ms], group[ms]
        for u in np.unique(ts[es == 1]):
            at_risk = ts >= u
            n_total = at_risk.sum()
            d_total = ((ts == u) & (es == 1)).sum()
            if n_total <= 1 or d_total == 0:
                continue
            n_g = np.array([(at_risk & (gs == g)).sum() for g in groups])
            d_g = np.array([((ts == u) & (es == 1) & (gs == g)).sum() for g in groups])
            OE += (d_g - d_total * n_g / n_total)
            factor = d_total * (n_total - d_total) / (n_total - 1)
            for i in range(k):
                for j in range(k):
                    if i == j:
                        V[i, j] += factor * (n_g[i] / n_total) * (1 - n_g[i] / n_total)
                    else:
                        V[i, j] += -factor * (n_g[i] / n_total) * (n_g[j] / n_total)
    OEr, Vr = OE[:-1], V[:-1, :-1]
    try:
        chi2_stat = float(OEr @ np.linalg.solve(Vr, OEr))
    except np.linalg.LinAlgError:
        chi2_stat = float(OEr @ np.linalg.pinv(Vr) @ OEr)
    df = k - 1
    p = float(chi2dist.sf(chi2_stat, df))
    return chi2_stat, df, p


def windowed_hr(rs, windows=((0, 36), (36, 60), (60, 120), (120, 10 ** 6))):
    """Cohort-stratified high-vs-low tertile HR inside successive follow-up windows."""
    sub = rs[rs.risk_group_tertile.isin(["low", "high"])].copy()
    sub["x"] = (sub.risk_group_tertile == "high").astype(int)
    rows = []
    for lo, hi in windows:
        d = sub[sub.time_months > lo].copy()
        d["e2"] = np.where((d.time_months <= hi) & (d.event == 1), 1, 0)
        d["t2"] = np.clip(d.time_months, None, hi) - lo
        if d.e2.sum() < 10:
            continue
        b, se = coxph_beta(d.t2, d.e2, d.x, strata=d.cohort)
        rows.append(dict(window="%d-%s" % (lo, "end" if hi > 10 ** 5 else hi),
                         n=len(d), events=int(d.e2.sum()), hr=np.exp(b),
                         lo95=np.exp(b - 1.96 * se), hi95=np.exp(b + 1.96 * se)))
    return pd.DataFrame(rows)


def logticks(ax, vals):
    """Label exactly the given ticks on a log axis; suppress matplotlib's minor labels."""
    ax.xaxis.set_major_locator(FixedLocator(vals))
    ax.xaxis.set_major_formatter(FixedFormatter([("%g" % v) for v in vals]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def draw(rs, kmt, ph, ps, phs, tv, path="figs/fig_km_hazard.png"):
    apply_figure_style(sizes=(8, 7, 6))
    mpl.rcParams.update({"axes.titlesize": 7.0, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6.0, "ytick.labelsize": 6.0})
    fig = plt.figure(figsize=(180 / 25.4, 104 / 25.4))
    gs = fig.add_gridspec(2, 2, hspace=0.66, wspace=0.34, left=0.125, right=0.975,
                          top=0.905, bottom=0.115)

    axa = fig.add_subplot(gs[0, 0])
    for g in ["low", "intermediate", "high"]:
        d = rs[rs.risk_group_tertile.eq(g)]
        k = km_curve(d.time_months, d.event)
        k = k[k.t <= 180]
        axa.step(k.t, k.S, where="post", color=GC[g], lw=1.2,
                 label="%s (n = %s, %d events)"
                       % (g, f"{len(d):,}", int(d.event.sum())))
        axa.fill_between(k.t, k.lo, k.hi, step="post", color=GC[g], alpha=0.12, lw=0)
    prow = kmt[kmt.cohort.eq("POOLED")].iloc[0]
    axa.set_xlim(0, 180)
    axa.set_ylim(0.30, 1.0)
    axa.set_xticks([0, 36, 60, 120, 180])
    axa.set_xlabel("Months from diagnosis")
    axa.set_ylabel("Overall survival")
    axa.set_title("High-risk patients fare clearly worse ($p$ = %.0e)"
                  % prow.logrank_p_stratified)
    axa.legend(frameon=False, fontsize=5.2, loc="lower left", handlelength=1.2,
               borderpad=0.1, labelspacing=0.22)

    axb = fig.add_subplot(gs[0, 1])
    yy = np.arange(len(tv))[::-1]
    axb.errorbar(tv.hr, yy, xerr=[tv.hr - tv.lo95, tv.hi95 - tv.hr], fmt="o", ms=4.0,
                 color=NEUT, ecolor=NEUT, elinewidth=0.9, capsize=2.0)
    axb.axvline(1.0, color="k", lw=0.7, ls=(0, (4, 3)))
    axb.axvline(phs["stratified_hr"], color=FOC, lw=0.9, ls=(0, (3, 2)))
    axb.set_yticks(yy)
    axb.set_yticklabels(["%s mo\n(%d events)" % (w, e)
                         for w, e in zip(tv.window, tv.events)], fontsize=5.4)
    axb.set_xscale("log")
    axb.set_xlim(0.62, 9.0)
    logticks(axb, [1, 2, 4, 8])
    axb.set_ylim(-0.75, len(tv) - 0.2)
    axb.set_xlabel("Hazard ratio, high vs low risk tertile")
    axb.set_title("The extra danger fades with time: HR %.1f → %.2f"
                  % (tv.hr.iloc[0], tv.hr.iloc[-1]))
    axb.text(phs["stratified_hr"] * 1.09, -0.66,
             "one pooled HR would say %.2f" % phs["stratified_hr"], fontsize=5.0,
             color=FOC, ha="left", va="bottom")

    axc = fig.add_subplot(gs[1, 0])
    kk = kmt[~kmt.cohort.eq("POOLED")].copy()
    kk = kk.assign(ord_=[ORDER.index(c) for c in kk.cohort]).sort_values(
        "ord_", ascending=False)
    viol = dict(zip(ph.cohort, ph.ph_violated_05))
    for i, (_, r0) in enumerate(kk.iterrows()):
        c = FOC if viol.get(r0.cohort, False) else NEUT
        axc.plot([r0.hr_lo95, r0.hr_hi95], [i, i], color=c, lw=0.9)
        axc.scatter(r0.hr_high_vs_low, i, s=24, color=c, zorder=3)
    axc.axvline(1.0, color="k", lw=0.7, ls=(0, (4, 3)))
    axc.set_yticks(np.arange(len(kk)))
    axc.set_yticklabels([FLAT[c] for c in kk.cohort], fontsize=5.4)
    axc.set_xscale("log")
    axc.set_xlim(0.45, 9.5)
    logticks(axc, [0.5, 1, 2, 4, 8])
    axc.set_ylim(-0.6, len(kk) - 0.4)
    axc.set_xlabel("Hazard ratio, high vs low risk tertile")
    nv = int(ph[~ph.cohort.eq("POOLED")].ph_violated_05.sum())
    axc.set_title("The risk gap shifts over time in %d of 6 cohorts (red)" % nv)

    axd = fig.add_subplot(gs[1, 1])
    spec3 = ps[ps.label.str.startswith("POOLED")].copy()
    lab3 = ["unstratified\n(original)", "cohort-stratified\n(recommended)",
            "per 1 SD\n(continuous)"]
    yy3 = np.arange(len(spec3))[::-1]
    for y, (_, r0) in zip(yy3, spec3.iterrows()):
        axd.plot([r0.hr_lo, r0.hr_hi], [y, y], color=NEUT, lw=0.9)
        axd.scatter(r0.hr, y, s=26, color=NEUT, zorder=3)
        axd.text(r0.hr_hi * 1.015, y, "%.2f" % r0.hr, fontsize=5.2, va="center",
                 ha="left")
    axd.axvline(1.0, color="k", lw=0.7, ls=(0, (4, 3)))
    axd.set_yticks(yy3)
    axd.set_yticklabels(lab3, fontsize=5.4)
    axd.set_xlim(1.35, 2.38)
    axd.set_ylim(-0.55, len(spec3) - 0.45)
    axd.set_xlabel("Pooled hazard ratio (95% CI), median split")
    axd.set_title("Cohort barely moves the pooled result (%.1f%% shift)"
                  % phs["pct_change_hr_stratifying"])

    for ax, L in [(axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")]:
        panel_letter(ax, L, case="lower")
    fig.savefig(path, dpi=1000, facecolor="white")
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
    clip = []
    fb = fig.bbox
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
    return tt, tp, clip


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    rs = pd.read_csv(os.path.join(RES, "loco_risk_scores.csv"))
    rs = rs.rename(columns={"group": "risk_group_tertile"})
    rs["risk_group_tertile"] = rs["risk_group_tertile"].replace({"mid": "intermediate"})
    kmt = pd.read_csv(os.path.join(RES, "km_stratification.csv"))
    ph = pd.read_csv(os.path.join(RES, "ph_tests.csv"))
    ps = pd.read_csv(os.path.join(RES, "pooled_hr_stratified.csv"))
    phs = json.load(open(os.path.join(RES, "pooled_hr_summary.json")))

    # C11-review fix: panel (a) previously annotated an unstratified pooled log-rank
    # test, which Section on statistical methods argues against (all pooled inference
    # elsewhere uses the cohort-stratified fit). Recompute the 3-group tertile log-rank
    # test stratified by cohort (Mantel-Haenszel generalisation) and use that instead.
    chi2_strat, df_strat, p_strat = stratified_logrank(
        rs.time_months, rs.event, rs.risk_group_tertile, rs.cohort)
    kmt.loc[kmt.cohort.eq("POOLED"), "logrank_chi2_stratified"] = chi2_strat
    kmt.loc[kmt.cohort.eq("POOLED"), "logrank_p_stratified"] = p_strat
    kmt.to_csv(os.path.join(RES, "km_stratification.csv"), index=False)
    print("cohort-stratified pooled tertile log-rank: chi2=%.3f df=%d p=%.3e"
          % (chi2_strat, df_strat, p_strat))

    tv = windowed_hr(rs)
    tv.to_csv(os.path.join(RES, "timevarying_hr_windows.csv"), index=False)
    assert tv.hr.iloc[0] > tv.hr.iloc[-1], "windowed HR should decay with follow-up"

    fig = draw(rs, kmt, ph, ps, phs, tv)
    tt, tp, clip = verify(fig)
    assert not tt, "text-text collisions: %s" % tt
    assert not tp, "text-patch collisions: %s" % tp
    assert not clip, "tick labels clipped at figure edge: %s" % clip
    print("figs/fig_km_hazard.png written; text-text 0, text-patch 0, clipped 0")
    print(tv[["window", "events", "hr", "lo95", "hi95"]].to_string(index=False))
