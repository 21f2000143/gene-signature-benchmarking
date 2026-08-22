"""
make_fig_auc_calibration.py -- Figure 4: time-dependent AUC and calibration.

Panels:
	a  Time-dependent AUC across prediction horizons
	b  Five-year AUC by held-out cohort
	c  Observed five-year survival across risk-score quintiles
	d  Sensitivity of five-year AUC to censoring-time definition

Inputs:
	results/time_dependent_auc_revised.csv
	results/tauc_censoring_sensitivity.csv
	results/calibration_quintiles.csv

Output:
	figs/fig_auc_calibration.png
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
		mpl.rcParams.update({
			"font.family": "sans-serif",
			"font.size": sizes[0],
			"axes.spines.top": False,
			"axes.spines.right": False,
			"savefig.dpi": 600,
		})

	def panel_letter(ax, letter, case="lower"):
		ax.text(
			-0.16, 1.07, letter,
			transform=ax.transAxes,
			fontsize=9,
			fontweight="bold",
			va="bottom",
			ha="right",
		)


# ---------------------------------------------------------------------
# Cohort definitions
# ---------------------------------------------------------------------

ORDER = [
	"TCGA",
	"METABRIC",
	"SCANB_GSE96058",
	"SCANB_GSE202203",
	"GSE20711",
	"GSE58812",
]

SHORT = {
	"TCGA": "TCGA",
	"METABRIC": "META-\nBRIC",
	"SCANB_GSE96058": "SCAN-B\n96058",
	"SCANB_GSE202203": "SCAN-B\n202203",
	"GSE20711": "GSE\n20711",
	"GSE58812": "GSE\n58812",
}

CC = {
	"TCGA": "#2B5C8A",
	"METABRIC": "#8A6D3B",
	"SCANB_GSE96058": "#4A6B4A",
	"SCANB_GSE202203": "#6B4A6B",
	"GSE20711": "#B3312C",
	"GSE58812": "#C77B2B",
}

SPEC = {
	"A_train_maxtau": "Training-set maximum $\\tau$",
	"B_train_safetau": "Training-set safe $\\tau$",
	"C_test_own": "Test-cohort $\\tau$",
}

OFF = {
	"A_train_maxtau": -0.20,
	"B_train_safetau": 0.00,
	"C_test_own": 0.20,
}


# ---------------------------------------------------------------------
# Data-label collision resolution
# ---------------------------------------------------------------------
#
# Several panels place a numeric label above every plotted point, and some
# points sit close enough in value that a fixed offset makes labels overlap
# (e.g. three cohorts within 0.006 AUC of each other at horizon = 5y in
# panel a). Rather than dropping labels, nudge them apart vertically in
# rendered pixel space -- using the actual bounding boxes, not an analytic
# estimate -- until no pair overlaps.

def _shift_label_px(ax, text, dy_px):
	x, y = text.get_position()
	xd, yd = ax.transData.transform((x, y))
	x2, y2 = ax.transData.inverted().transform((xd, yd + dy_px))
	text.set_position((x, y2))


def declutter_labels(fig, ax, texts, max_iter=200, pad_px=1.0):
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


def label_gap(fig, ax, gap_pt=2.2):
	"""
	Data-unit y-offset that renders as a fixed `gap_pt`-point visual gap
	between a bar top and its label, on THIS axis's current y-scale. Panels
	share wildly different y-ranges (0.78 to 1.55), so a single hardcoded
	data-unit offset (e.g. 0.02) produces visibly inconsistent spacing --
	tight in a wide-range panel, loose in a narrow one. Computing the offset
	from the actual rendered pixels-per-data-unit keeps the gap uniform
	across panels. Requires xlim/ylim to already be set on `ax`.
	"""

	fig.canvas.draw()

	bbox = ax.get_window_extent(fig.canvas.get_renderer())
	y0, y1 = ax.get_ylim()

	px_per_unit = bbox.height / (y1 - y0)
	gap_px = gap_pt * fig.dpi / 72.0

	return gap_px / px_per_unit


# ---------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------

RES = "results"

tda = pd.read_csv(
	os.path.join(RES, "time_dependent_auc_revised.csv")
)

tcs = pd.read_csv(
	os.path.join(RES, "tauc_censoring_sensitivity.csv")
)

cal = pd.read_csv(
	os.path.join(RES, "calibration_quintiles.csv")
)


# ---------------------------------------------------------------------
# Five-year AUC summary
# ---------------------------------------------------------------------

a5 = tda[tda.horizon_years.eq(5.0)]

vals = []

for cohort in ORDER:
	q = a5[a5.held_out_cohort.eq(cohort)].auc

	if len(q):
		vals.append(float(q.iloc[0]))
	else:
		vals.append(np.nan)

v5 = a5.auc.dropna()

NEST = int(v5.notna().sum())


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

def draw(path="figs/fig_auc_calibration.png"):

	apply_figure_style(sizes=(8, 7, 6))

	mpl.rcParams.update({
		"axes.titlesize": 7.0,
		"axes.labelsize": 6.8,
		"xtick.labelsize": 6.0,
		"ytick.labelsize": 6.0,
	})

	fig = plt.figure(
		figsize=(230.8 / 25.4, 154.8 / 25.4)
	)

	gs = fig.add_gridspec(
		2,
		2,
		hspace=0.62,
		wspace=0.30,
		left=0.085,
		right=0.975,
		top=0.905,
		bottom=0.105,
	)

	xs = np.arange(len(ORDER))


	# =================================================================
	# Panel a: Time-dependent AUC
	# =================================================================

	axa = fig.add_subplot(gs[0, 0])

	horizons = [3, 5, 10]
	nb = len(ORDER)
	bw_a = 0.8 / nb

	axa.set_xticks(np.arange(len(horizons)))
	axa.set_xticklabels([str(h) for h in horizons])
	axa.set_xlabel("Horizon (years)")
	axa.set_ylabel("Time-dependent AUC")
	axa.set_ylim(0.0, 1.05)
	axa.set_xlim(-0.5, len(horizons) - 0.5)

	gap_a = label_gap(fig, axa)

	panel_a_labels = []

	for j, cohort in enumerate(ORDER):

		ok = (
			tda[tda.held_out_cohort.eq(cohort)]
			.sort_values("horizon_years")
			.dropna(subset=["auc"])
		)

		if len(ok):

			pos = [
				horizons.index(int(h)) + (j - (nb - 1) / 2) * bw_a
				for h in ok.horizon_years
			]

			axa.bar(
				pos,
				ok.auc,
				width=bw_a * 0.92,
				color=CC[cohort],
				label=cohort.replace("SCANB_", "SCAN-B "),
				zorder=3,
			)

			# Data labels
			for x, (_, row) in zip(pos, ok.iterrows()):

				panel_a_labels.append(
					axa.text(
						x,
						row.auc + gap_a,
						f"{row.auc:.3f}",
						fontsize=4.5,
						ha="center",
						va="bottom",
						color=CC[cohort],
						rotation=90,
					)
				)

	declutter_labels(fig, axa, panel_a_labels)

	# Every missing bar has a distinct, recorded reason (tda.note): METABRIC's
	# IPCW censoring-weight estimator is degenerate at every horizon (unrelated
	# to follow-up length), while the two SCAN-B cohorts are only missing at
	# the 10-year horizon because their maximum follow-up (81 / 105 months) is
	# shorter than 120 months. Tag each gap with its actual reason rather than
	# a single blanket "not estimable".

	def _reason_tag(note):
		if "maximum follow-up" in note:
			return "follow-up\n< 10y"
		if "censoring survival" in note:
			return "IPCW\nundefined"
		return "not\nestimable"

	missing = tda[tda.auc.isna()]

	for _, row in missing.iterrows():

		j = ORDER.index(row.held_out_cohort)
		gx = (
			horizons.index(int(row.horizon_years))
			+ (j - (nb - 1) / 2) * bw_a
		)

		axa.text(
			gx,
			0.02,
			_reason_tag(row.note),
			fontsize=3.3,
			ha="center",
			va="bottom",
			color="#8A8A8A",
			linespacing=1.1,
			rotation=90,
		)

	axa.text(
		1.0,
		0.915,
		"METABRIC: IPCW weights degenerate, all horizons  •  "
		"SCAN-B: excluded at 10y only, follow-up < 10y",
		fontsize=3.3,
		ha="center",
		va="top",
		color="#666666",
	)

	axa.set_title(
		"Time-dependent AUC across prediction horizons"
	)

	axa.legend(
		frameon=False,
		fontsize=5.0,
		loc="upper center",
		ncol=3,
		handlelength=1.1,
		borderpad=0.1,
		labelspacing=0.2,
		columnspacing=0.8,
	)


	# =================================================================
	# Panel b: Five-year AUC
	# =================================================================

	axb = fig.add_subplot(gs[0, 1])

	axb.set_xlim(-0.6, 5.6)
	axb.set_ylim(0.0, 0.78)

	gap_b = label_gap(fig, axb)

	for i, value in enumerate(vals):

		if np.isnan(value):

			axb.text(
				i,
				0.015,
				"not\nestimable",
				fontsize=4.9,
				ha="center",
				va="bottom",
				color="#666666",
				linespacing=1.2,
			)

		else:

			axb.bar(
				i,
				value,
				width=0.6,
				color="#2B5C8A",
				zorder=3,
			)

			# Data label
			axb.text(
				i,
				value + gap_b,
				f"{value:.3f}",
				fontsize=5.0,
				ha="center",
				va="bottom",
			)

	axb.set_xticks(xs)

	axb.set_xticklabels(
		[SHORT[o] for o in ORDER],
		fontsize=5.4,
	)

	axb.set_ylabel("Five-year AUC")

	axb.set_title(
		"Five-year AUC by held-out cohort"
	)


	# =================================================================
	# Panel c: Calibration by risk quintile
	# =================================================================

	axc = fig.add_subplot(gs[1, 0])

	axc.set_ylim(0.0, 1.45)

	gap_c = label_gap(fig, axc)

	cohorts_c = list(cal.cohort.unique())
	nb_c = len(cohorts_c)
	bw_c = 0.8 / nb_c

	panel_c_labels = []

	for j, cohort in enumerate(cohorts_c):

		s = (
			cal[cal.cohort.eq(cohort)]
			.sort_values("quintile")
		)

		monotone = bool(
			s.strictly_monotone.iloc[0]
		)

		if monotone:
			label = "monotone"
		else:
			label = "%d reversals" % int(
				s.n_reversals.iloc[0]
			)

		pos = s.quintile.values - 1 + (j - (nb_c - 1) / 2) * bw_c

		axc.bar(
			pos,
			s.obs_surv,
			width=bw_c * 0.92,
			# yerr=[
			# 	s.obs_surv - s.obs_lo,
			# 	s.obs_hi - s.obs_surv,
			# ],
			# error_kw=dict(elinewidth=0.6, capsize=1.2, ecolor=CC[cohort]),
			color=CC[cohort],
			label="%s (%s)" % (
				cohort.replace("SCANB_", "SCAN-B "),
				label,
			),
			zorder=3,
		)

		# Data labels
		for x, (_, row) in zip(pos, s.iterrows()):

			panel_c_labels.append(
				axc.text(
					x,
					row.obs_hi + gap_c,
					f"{row.obs_surv:.2f}",
					fontsize=4.0,
					ha="center",
					va="bottom",
					color=CC[cohort],
					rotation=90,
				)
			)

	declutter_labels(fig, axc, panel_c_labels)

	nmono = int(
		cal.groupby("cohort")
		.strictly_monotone
		.first()
		.sum()
	)

	ncoh = cal.cohort.nunique()

	axc.set_xticks(np.arange(5))
	axc.set_xticklabels([str(q) for q in [1, 2, 3, 4, 5]])
	axc.set_xlim(-0.5, 4.5)

	axc.set_xlabel(
		"Risk-score quintile (1 = lowest risk)"
	)

	axc.set_ylabel(
		"Observed 5-year survival"
	)

	axc.set_title(
		"Observed five-year survival across risk groups"
	)

	axc.legend(
		frameon=False,
		fontsize=4.8,
		loc="upper center",
		ncol=2,
		handlelength=1.1,
		borderpad=0.1,
		labelspacing=0.2,
		columnspacing=0.7,
	)


	# =================================================================
	# Panel d: Censoring-time sensitivity
	# =================================================================

	axd = fig.add_subplot(gs[1, 1])

	sensitivity_colors = {
		"A_train_maxtau": "#2B5C8A",
		"B_train_safetau": "#6B4A6B",
		"C_test_own": "#4A6B4A",
	}

	axd.set_xlim(-0.6, 5.6)
	axd.set_ylim(0.0, 1.55)

	gap_d = label_gap(fig, axd)

	bw_d = 0.18

	panel_d_labels = []

	for spec, color in sensitivity_colors.items():

		xv = []
		yv = []

		for i, cohort in enumerate(ORDER):

			q = tcs[
				tcs.held_out_cohort.eq(cohort)
				& tcs.spec.eq(spec)
				& tcs.horizon_years.eq(5.0)
			].auc

			if (
				len(q)
				and not np.isnan(float(q.iloc[0]))
			):

				auc_value = float(q.iloc[0])

				xv.append(i + OFF[spec])
				yv.append(auc_value)

				# Data label
				panel_d_labels.append(
					axd.text(
						i + OFF[spec],
						auc_value + gap_d,
						f"{auc_value:.3f}",
						fontsize=4.0,
						ha="center",
						va="bottom",
						color=color,
						rotation=90,
					)
				)

		axd.bar(
			xv,
			yv,
			width=bw_d,
			color=color,
			label=SPEC[spec],
			zorder=3,
		)

	declutter_labels(fig, axd, panel_d_labels)

	# for i in range(len(ORDER)):

	# 	axd.axvline(
	# 		i + 0.5,
	# 		color="#DDDDDD",
	# 		lw=0.5,
	# 		zorder=0,
	# 	)

	axd.text(
		1.20,
		1.45,
		"METABRIC: estimable only\nwith test-cohort $\\tau$",
		fontsize=4.6,
		color="#555555",
		ha="center",
		va="top",
		linespacing=1.25,
	)

	axd.set_xticks(xs)

	axd.set_xticklabels(
		[SHORT[o] for o in ORDER],
		fontsize=5.4,
	)

	axd.set_ylabel("Five-year AUC")

	axd.set_title(
		"Sensitivity of five-year AUC to\n"
		"censoring-time definition"
	)

	axd.legend(
		frameon=False,
		fontsize=5.0,
		loc="upper right",
		handlelength=1.0,
		borderpad=0.1,
		labelspacing=0.22,
		handletextpad=0.3,
	)


	# =================================================================
	# Panel labels
	# =================================================================

	for ax, letter in [
		(axa, "a"),
		(axb, "b"),
		(axc, "c"),
		(axd, "d"),
	]:

		panel_letter(
			ax,
			letter,
			case="lower",
		)


	# =================================================================
	# Save
	# =================================================================

	fig.savefig(
		path,
		dpi=1000,
		facecolor="white",
	)

	return fig


# ---------------------------------------------------------------------
# Figure verification
# ---------------------------------------------------------------------

def verify(fig):
	"""
	Check that text does not overlap other text or plotted patches.
	"""

	fig.canvas.draw()

	renderer = fig.canvas.get_renderer()

	tx = [
		(text, text.get_window_extent(renderer))
		for text in fig.findobj(mpl.text.Text)
		if text.get_text().strip()
		and text.get_visible()
	]

	axis_labels = (
		{id(ax.xaxis.label) for ax in fig.axes}
		|
		{id(ax.yaxis.label) for ax in fig.axes}
	)

	text_collisions = [
		(a.get_text()[:26], b.get_text()[:26])
		for i, (a, box_a) in enumerate(tx)
		for b, box_b in tx[i + 1:]
		if box_a.overlaps(box_b)
		and id(a) not in axis_labels
		and id(b) not in axis_labels
	]

	text_patch_collisions = []

	for ax in fig.axes:

		for text in ax.texts:

			if (
				not text.get_text().strip()
				or not text.get_visible()
			):
				continue

			text_box = text.get_window_extent(renderer)

			for patch in ax.patches:

				if patch.get_window_extent(renderer).overlaps(
					text_box
				):

					text_patch_collisions.append(
						(
							text.get_text()[:24],
							type(patch).__name__,
						)
					)

	return (
		text_collisions,
		text_patch_collisions,
	)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

	os.makedirs(
		"figs",
		exist_ok=True,
	)

	fig = draw()

	text_collisions, patch_collisions = verify(fig)

	assert not text_collisions, (
		"text-text collisions: %s"
		% text_collisions
	)

	assert not patch_collisions, (
		"text-patch collisions: %s"
		% patch_collisions
	)

	print(
		"figs/fig_auc_calibration.png written; "
		"text-text 0, text-patch 0"
	)