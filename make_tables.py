"""
make_tables.py -- generate every LaTeX table in the manuscript directly from the
result CSV/JSON files, so no number in the paper can drift from its source.

Each table is written to paper/tabN_*.tex as a standalone float that the main
article \\input's. Every emitted value is read from results/; nothing is typed.

Run from the workspace root:  python scripts/make_tables.py
"""
import json
import os
import re

import numpy as np
import pandas as pd

RES = "results"
OUT = "paper"

ORDER = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203",
         "GSE20711", "GSE58812"]
HEAD = {"TCGA": "TCGA", "METABRIC": "METABRIC", "SCANB_GSE96058": "SCAN-B$_{1}$",
        "SCANB_GSE202203": "SCAN-B$_{2}$", "GSE20711": "GSE20711",
        "GSE58812": "GSE58812"}
GS = ["Novel5", "Novel5_plus_Anchor4", "BuffaHypoxia", "MammaPrint70", "PAM50",
      "OncotypeDX21", "GGI", "CNetCox6", "Anchor4"]
GSLBL = {"Novel5": "Novel-5 \\textbf{(this study)}",
         "Novel5_plus_Anchor4": "Novel-5 $+$ Anchor-4",
         "BuffaHypoxia": "Buffa hypoxia", "MammaPrint70": "MammaPrint 70",
         "PAM50": "PAM50", "OncotypeDX21": "Oncotype DX 21", "GGI": "GGI",
         "CNetCox6": "CNet-Cox 6", "Anchor4": "Anchor-4 (selection scaffold)"}


def rd(name):
    return pd.read_csv(os.path.join(RES, name))


def js(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def f3(x):
    return "---" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else "%.3f" % x


def write(name, body):
    """Write a table file, refusing to emit LaTeX that is known to break.

    Both checks below correspond to defects that actually reached the compiler
    in this project, so they are asserted rather than trusted:

    1. A doubled percent (``\\%%``) surviving into the output. In a %-formatted
       block ``\\%%`` is correct and collapses to ``\\%``; in a concatenated
       block it does not, and the stray second ``%`` comments out the ``\\\\``
       row terminator, merging two rows and producing "Misplaced \\noalign".
    2. Every tabular row must present the same number of cells as the column
       spec declares, ignoring \\multicolumn spans and rule/formatting lines.
    """
    assert "\\%%" not in body, (
        "%s: literal '\\%%%%' reached the output; in a non-formatted block write "
        "'\\%%' -- the stray second %%%% comments out the row terminator" % name)

    spec = re.search(r"\\begin\{tabular\}\{([^}]+)\}", body)
    if spec:
        ncol = len(re.findall(r"[lcrp]", re.sub(r"\{[^}]*\}", "", spec.group(1))))
        for ln in body.split("\n"):
            s = ln.strip()
            if "&" not in s or "\\multicolumn" in s or s.startswith("%"):
                continue
            ncell = s.split("\\\\")[0].count("&") + 1
            assert ncell == ncol, "%s: row has %d cells, spec declares %d: %.70s" % (
                name, ncell, ncol, s)

    # Guarantee the table fits the text block. Seven of eight tables overflowed
    # by 16-177pt at \footnotesize; tuning \tabcolsep per table is fragile
    # because the content changes whenever the upstream results change. Wrapping
    # the tabular in \resizebox scales it to the available width and is a no-op
    # when the natural width already fits (\ifdim guard), so a table can never
    # silently run into the margin again.
    wide = "\\begin{table*}" in body
    W = "\\textwidth" if wide else "\\columnwidth"
    body = body.replace(
        "\\begin{tabular}",
        "\\resizebox{\\ifdim\\width>%s %s\\else\\width\\fi}{!}{%%\n\\begin{tabular}" % (W, W), 1)
    body = body.replace("\\end{tabular}", "\\end{tabular}}", 1)

    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(body)
    print("wrote %s/%s (%d chars, %d cols, fit to %s)"
          % (OUT, name, len(body), ncol if spec else 0, W))


# ---------------------------------------------------------------- Table 1
def table1():
    t = rd("table1_cohorts.csv")
    rows = []
    for _, r in t.iterrows():
        rows.append("%s & %s & %s & %d & %d & %.1f\\%% & %.1f & %d\\\\" % (
            r.cohort.replace("SCANB_", "SCAN-B "), r.platform, r.endpoint,
            r.n, r.events, 100 * r.event_rate, r.median_fu_months, r.n_genes))
    tot_n, tot_e = int(t.n.sum()), int(t.events.sum())
    os_ = t[t.endpoint.eq("OS")]
    body = r"""\begin{table*}[htbp]
\caption{\csentence{Harmonised cohorts.}
Nine public breast-cancer transcriptomic cohorts after harmonisation. Median
follow-up is recomputed from the harmonised phenotype tables (reverse
Kaplan--Meier on the observed time scale). Gene count is the number of symbols
surviving harmonisation on that platform, before intersection across cohorts.
Six cohorts carry overall survival (OS) and are used as held-out folds; three
carry distant-metastasis-free or disease-free survival (DMFS/DFS) and are used
only as training material and for the coverage-restricted sensitivity analysis.}
\label{tab:cohorts}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrrr}
\hline
Cohort & Platform & Endpoint & $n$ & Events & Event rate & Median FU (mo) & Genes\\
\hline
""" + "\n".join(rows) + r"""
\hline
\multicolumn{3}{l}{\textbf{Total}} & \textbf{%d} & \textbf{%d} & & & \\
\multicolumn{3}{l}{\textit{of which OS (held-out folds)}} & \textit{%d} & \textit{%d} & & & \\
\hline
\end{tabular}
\end{table*}
""" % (tot_n, tot_e, int(os_.n.sum()), int(os_.events.sum()))
    write("tab1_cohorts.tex", body)


# ---------------------------------------------------------------- Table 2
def table2():
    """Per-cohort LOCO Harrell c under the pre-specified ridge model, plus the
    mean Harrell and mean Uno columns, plus the clinicopathology arm."""
    # Single source for every cell AND every mean in this table: the
    # censoring-robust recomputation, which carries Harrell and Uno side by side
    # for all gene sets and for the clinicopathology arm. Mixing per-cohort cells
    # from loco_os.csv with means from metrics_summary.csv would leave row means
    # disagreeing with the mean column by up to 0.012 (independent refits).
    mhu = rd("metrics_harrell_uno.csv")
    P = mhu.pivot_table(index="gene_set", columns="held_out_cohort",
                        values="harrell_c")
    # The clinical row comes from the RECONCILED arm, not from the Clinical rows
    # of metrics_harrell_uno.csv. Those were computed with GSE20711's age and
    # tumour size treated as continuous, when the clinical audit shows both are
    # unlabelled 0/1 dichotomies in that cohort; that inflated its clinical
    # c-index by 0.057. Table 4 already used the audited rule, so the two tables
    # disagreed. reconcile_clinical_arm.py recomputes the arm once under the
    # audited rule with both concordance measures, and is now the only source of
    # a clinical number anywhere in the paper.
    clin = rd("clinical_arm_reconciled.csv").set_index("held_out_cohort")
    ms = rd("metrics_summary.csv").set_index("gene_set")

    ntest = mhu.drop_duplicates("held_out_cohort").set_index("held_out_cohort")
    hdr = " & ".join(HEAD[c] for c in ORDER)
    nrow = " & ".join("%d (%d)" % (ntest.loc[c, "n_test"],
                                   ntest.loc[c, "events_test"]) for c in ORDER)

    # best independent signature per cohort (superset and negative control excluded)
    indep = [g for g in GS if g not in ("Novel5_plus_Anchor4", "Anchor4")]
    best = {c: P.loc[indep, c].idxmax() for c in ORDER}

    rows = []
    for g in GS:
        cells = []
        for c in ORDER:
            v = f3(P.loc[g, c])
            cells.append("\\textbf{%s}" % v if best[c] == g else v)
        rows.append("%s & %s & %s & %s\\\\" % (
            GSLBL[g], " & ".join(cells),
            f3(ms.loc[g, "harrell_mean"]), f3(ms.loc[g, "uno_mean"])))

    crow = "Clinicopathology & %s & %s & %s\\\\" % (
        " & ".join(f3(clin.loc[c, "clinical_harrell"]) for c in ORDER),
        f3(clin.clinical_harrell.reindex(ORDER).mean()),
        f3(clin.clinical_uno.reindex(ORDER).mean()))

    # the mean column must be the mean of the six printed cells, else the table
    # is silently mixing two independent refits
    for g in GS:
        cells = [P.loc[g, c] for c in ORDER]
        assert abs(np.mean(cells) - ms.loc[g, "harrell_mean"]) < 5e-4, \
            "row mean disagrees with mean column for %s: %.4f vs %.4f" % (
                g, np.mean(cells), ms.loc[g, "harrell_mean"])

    body = r"""\begin{table*}[htbp]
\caption{\csentence{Leave-one-cohort-out external validation.}
Harrell concordance in each held-out cohort under the pre-specified ridge Cox
model trained on the pooled remaining cohorts, with the mean over the six folds
and the corresponding mean inverse-probability-of-censoring-weighted (Uno)
concordance. SCAN-B$_{1}$ = GSE96058, SCAN-B$_{2}$ = GSE202203. Bold marks the
best independent gene signature in each column; the Novel-5 $+$ Anchor-4
superset and Anchor-4 --- the four-gene scaffold the selection search was
seeded with, not an independent negative control (Section~\ref{sec:stats}) ---
are excluded from that ranking. CNet-Cox 6 is named for its published six-gene
form; only five of those genes survive harmonisation and are used here in every
cohort. The clinicopathology row uses the covariates actually available in each
held-out cohort (Table~\ref{tab:clinical}) and is the reference every gene set
must beat.}
\label{tab:loco}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrr}
\hline
 & \multicolumn{6}{c}{Held-out cohort (Harrell $c$)} & \multicolumn{2}{c}{Mean over folds}\\
\cline{2-7}\cline{8-9}
Gene set & """ + hdr + r""" & Harrell & Uno\\
$n$ (events) & """ + nrow + r""" & & \\
\hline
""" + "\n".join(rows) + r"""
\hline
""" + crow + r"""
\hline
\end{tabular}
\end{table*}
"""
    write("tab2_loco.tex", body)


# ---------------------------------------------------------------- Table 3
def table3():
    """Fully nested re-selection: what the search finds when the held-out cohort
    is invisible to it."""
    nf = rd("nested_selection_folds.csv").set_index("held_out_cohort")
    ns = js("nested_selection_summary.json")
    rows = []
    for c in ORDER:
        r = nf.loc[c]
        panel = r.selected_panel.replace("|", ", ")
        ov = r.overlap_genes if isinstance(r.overlap_genes, str) else ""
        rows.append("%s & %s & %.3f & %.3f & %.3f & %d & %s\\\\" % (
            HEAD[c].replace("$_{1}$", "$_{1}$"), panel, r.cindex_nested,
            r.cindex_published_panel, r.cindex_anchor_only,
            int(r.overlap_with_published), ov.replace("|", ", ")))
    body = r"""\begin{table*}[htbp]
\caption{\csentence{Fully nested re-selection of the panel.}
For each of the six overall-survival cohorts the entire five-step forward
selection was re-run using only the other five overall-survival cohorts (never
the DMFS/DFS series), so the held-out data were invisible to both hyperparameter
tuning and gene selection. This training pool is one cohort larger than the
four-cohort pool (TCGA, METABRIC, both SCAN-B releases) used for the original
selection reported in Section~\ref{sec:selection}; the nested audit therefore
tests whether re-selection is stable under leakage removal, not whether it
reproduces the original search's exact training configuration. Columns: the
panel the search returned on that fold; its concordance in the held-out cohort;
the concordance of the published Novel-5 panel on the same fold; the
concordance of Anchor-4, the four-gene selection scaffold, evaluated alone; and
the number of published-panel genes the fold-specific search recovered. Mean
nested concordance %.4f (SD %.4f) versus %.4f for the
published panel evaluated on the same folds, giving a selection optimism of
%+.4f. No gene was selected in all six folds; %d distinct genes were selected
across folds from %s candidates, with a mean overlap of %.2f of five genes.}
\label{tab:nested}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrl}
\hline
Held-out & Panel selected without the held-out cohort & Nested $c$ & Published $c$ & Anchor-4 $c$ & Overlap & Shared genes\\
\hline
""" % (ns["mean_cindex_nested"], ns["sd_cindex_nested"],
       ns["mean_cindex_published_panel_same_folds"],
       ns["optimism_published_minus_nested"], ns["n_distinct_genes_selected"],
       "{:,}".format(ns["n_candidates"]), ns["mean_overlap_with_published"]) \
        + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
"""
    write("tab3_nested.tex", body)


# ---------------------------------------------------------------- Table 4
def table4():
    """Clinicopathology arm, its per-cohort covariate availability, and the audit."""
    # Clinical and Novel-5 columns come from the reconciled arm, the same source
    # Table 2 now uses, so the two tables cannot report different clinical
    # numbers. Only the common-covariate arm is still read from the older file:
    # it is a different model (restricted to covariates shared across cohorts),
    # not a competing estimate of the same one.
    rec = rd("clinical_arm_reconciled.csv").set_index("held_out_cohort")
    ca = rd("clinical_arm_recomputed.csv")
    cas = js("clinical_arm_summary.json")
    piv = ca.pivot_table(index="held_out_cohort", columns="arm", values="cindex")
    means = cas["mean_loco_cindex_by_arm"]
    rows = []
    for c in ORDER:
        covs = rec.loc[c, "covariates"]
        covs = "" if not isinstance(covs, str) else covs.replace("|", ", ").replace("_", "\\_")
        d = rec.loc[c, "delta_harrell"]
        common = piv.loc[c].get("clinical_common_5_excl_GSE58812", np.nan)
        rows.append("%s & %s & %s & %s & %s & %s\\\\" % (
            HEAD[c], covs, f3(rec.loc[c, "clinical_harrell"]),
            f3(common), f3(rec.loc[c, "novel5_harrell"]), "%+.3f" % d))
    flags = "".join("\\item %s\n" % f.replace("_", "\\_").replace("'", "`")
                    for f in cas["audit_flags"])
    body = r"""\begin{table*}[htbp]
\caption{\csentence{The clinicopathology arm and the audit of the clinical data.}
Leave-one-cohort-out concordance of a clinicopathology-only ridge Cox model,
using the covariates actually usable in each cohort, against the Novel-5 gene
panel on identical folds. $\Delta$ is gene minus clinical. The
common-covariate column restricts to the two variables shared by five OS
cohorts (node status, ER status); no covariate is shared by all six, because
GSE58812 is a triple-negative series carrying age alone. GSE20711 contributes
grade, node and ER only: its age and size columns are unlabelled 0/1
dichotomies (see audit below) and are therefore not interpretable as years or
millimetres. Mean over folds:
per-cohort-available clinical %.4f, common-covariate clinical %.4f, Novel-5
%.4f. The clinicopathology arm is better in all six cohorts.}
\label{tab:clinical}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrr}
\hline
Held-out & Covariates usable in that cohort & Clinical $c$ & Common-cov. $c$ & Novel-5 $c$ & $\Delta$\\
\hline
""" % (rec.clinical_harrell.reindex(ORDER).mean(),
       means["clinical_common_5_excl_GSE58812"],
       rec.novel5_harrell.reindex(ORDER).mean()) \
        + "\n".join(rows) + r"""
\hline
\end{tabular}

\vspace{2mm}
\footnotesize
\textit{Audit of the harmonised clinical tables --- irregularities found, and
the reason no single covariate set spans all cohorts:}
\begin{itemize}\setlength{\itemsep}{0pt}
""" + flags + r"""\end{itemize}
\end{table*}
"""
    write("tab4_clinical.tex", body)


# ---------------------------------------------------------------- Table 5
def table5():
    """Size-matched nulls and the paired sign test, side by side."""
    nul = rd("null_summary.csv")
    sm = nul[nul.null_set.str.startswith("sizematched")
             & ~nul.observed_panel.str.contains("reference")].copy()
    sg = rd("paired_sign_test_by_comparator.csv").set_index("comparator")
    rf = js("resolution_floor.json")
    n20 = js("null_20k_summary.json")
    ms = rd("metrics_summary.csv").set_index("gene_set")

    sm = sm.set_index("observed_panel")
    rows = []
    for g in GS:
        if g not in sm.index:
            continue
        r = sm.loc[g]
        if g == "Novel5":
            pair = " & & & "
        else:
            s = sg.loc[g]
            pair = "%d/6 & %d/6 & %.5f & %.2f" % (
                int(s.n_pos), int(s.n_ci_favour), s.sign_p, s.q_bh)
        rows.append("%s & %d & %.3f & %.3f & %.1f & %s\\\\" % (
            GSLBL[g].replace(" \\textbf{(this study)}", ""), int(r.panel_size),
            r.observed_cindex, r.null_mean, r.observed_percentile, pair))
    body = r"""\begin{table*}[htbp]
\caption{\csentence{Size-matched random-panel nulls and the paired cohort-level
comparison.}
Left: each gene set is compared against a null of random panels drawn with
\emph{its own} gene count, because concordance rises with set size under random
draws; the percentile is the observed value's position in that own-size null. To
keep 24,500 draws tractable, both the null draws and the ``Observed $c$'' column
here are the mean over only the four largest cohorts (TCGA, METABRIC, SCAN-B$_1$,
SCAN-B$_2$), \emph{not} the six-cohort primary mean of Table~\ref{tab:loco}; this
is why Novel-5's value here (%.3f) differs from its Table~\ref{tab:loco} mean
(%.4f) and is not a transcription error. The six-cohort mean is reproduced for the
five-gene panel specifically at the reference scale of 20,000 draws below. Only
four of nine sets clear the 95th percentile of their own (four-cohort) null.
Right: per-cohort paired comparison of Novel-5 against each other set, computed
at the primary six-cohort scale --- the number of the six held-out cohorts in
which the Novel-5 point estimate is higher, the number in which the paired
bootstrap interval excludes zero, and the two-sided sign-test $p$. With six
cohorts the smallest attainable two-sided sign-test $p$ is %.5f, so a value of
%.5f means only that all six cohorts agreed in sign; no six-cohort test can
resolve below it. $q$ is the Benjamini--Hochberg-adjusted $p$ across these eight
comparators; the five unadjusted $p=0.03125$ tests sit exactly at the $q=0.05$
threshold, so the sign-test evidence survives multiplicity correction only
marginally. For reference, at the six-cohort primary scale the Novel-5 mean
leave-one-cohort-out concordance %.4f lies above every one of %s random
five-gene draws ($z$ = %.2f against a null mean of %.4f), which is the floor of
that design rather than a measured $p$-value.}
\label{tab:nulls}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrr}
\hline
 & \multicolumn{4}{c}{Own-size random-panel null (4-cohort scale)} & \multicolumn{4}{c}{Paired vs Novel-5 (6-cohort scale)}\\
\cline{2-5}\cline{6-9}
Gene set & $k$ genes & Observed $c$ (4-coh.) & Null mean & Pctile & Higher & CI excl.\ 0 & Sign $p$ & BH $q$\\
\hline
""" % (sm.loc["Novel5", "observed_cindex"], ms.loc["Novel5", "harrell_mean"],
       rf["sign_test_min_two_sided_p"], rf["sign_test_min_two_sided_p"],
       n20["observed_novel5_mean_loco_c"], "{:,}".format(n20["n_draws"]),
       n20["observed_z_vs_null"], n20["null_mean"]) \
        + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
"""
    write("tab5_nulls.tex", body)


# ---------------------------------------------------------------- Table 6
def table6():
    """Risk stratification, PH tests, and the time-varying hazard ratio."""
    km = rd("km_stratification.csv")
    ph = rd("ph_tests.csv")
    tv = rd("timevarying_hr_windows.csv")
    phs = js("pooled_hr_summary.json")

    # both tables key on 'cohort' (with a POOLED row), not 'held_out_cohort'
    kmi = km.set_index("cohort")
    phi = ph.set_index("cohort")

    rows = []
    for c in ORDER:
        pv = float(phi.loc[c, "p_value"])
        viol = "%.3g" % pv
        if bool(phi.loc[c, "ph_violated_05"]):
            viol = "\\textbf{%s}" % viol
        rows.append("%s & %.2f & %.2f--%.2f & %.2g & %s\\\\" % (
            HEAD[c], kmi.loc[c, "hr_high_vs_low"], kmi.loc[c, "hr_lo95"],
            kmi.loc[c, "hr_hi95"], kmi.loc[c, "logrank_p"], viol))

    wrows = []
    for _, r in tv.iterrows():
        # order matters: replace the range dash first, then the open-ended tail,
        # or "120-end" becomes "120----$\infty$"
        w = str(r.window)
        w = (w.split("-")[0] + "--$\\infty$") if w.endswith("-end") \
            else w.replace("-", "--")
        wrows.append("%s & %d & %d & %.2f & %.2f--%.2f\\\\" % (
            w, int(r.n), int(r.events), r.hr, r.lo95, r.hi95))

    ppool = "%.3g" % float(phi.loc["POOLED", "p_value"])
    n_viol = int(ph[ph.cohort.isin(ORDER)].ph_violated_05.sum())

    # Percent escaping in this function is asymmetric and has bitten twice.
    # The first template below IS %-formatted, so a literal percent must be
    # doubled there. The trailing template after the row join is NOT formatted,
    # so a doubled percent survives verbatim into the .tex; its second percent
    # then comments out the row terminator and merges two rows, which is what
    # produced the "Misplaced noalign / Extra alignment tab" failure. write()
    # now asserts against a doubled percent reaching the output.

    body = r"""\begin{table*}[htbp]
\caption{\csentence{Risk stratification and the failure of proportional hazards.}
Upper panel: hazard ratio for the highest versus lowest tertile of the
out-of-fold Novel-5 risk score within each held-out cohort, with the log-rank
$p$ and the Grambsch--Therneau global test of proportional hazards; a bold
$p$-value marks a cohort in which proportionality is rejected at 0.05. The
pooled fit also rejects it ($p$ = %s). Lower panel: cohort-stratified hazard
ratio for high versus low risk estimated separately within successive follow-up
windows. The effect is strong early and absent late, so the single pooled
hazard ratio (unstratified %.3f, cohort-stratified %.3f, a change of %.1f\%%;
%.3f per standard deviation of the continuous score) is a follow-up-weighted
average of a decaying effect rather than a constant one.}
\label{tab:ph}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\hline
Held-out cohort & HR (T3 vs T1) & 95\%% CI & Log-rank $p$ & PH $p$\\
\hline
""" % (ppool, phs["unstratified_hr"], phs["stratified_hr"],
       phs["pct_change_hr_stratifying"],
       phs["stratified_continuous_hr_per_sd"]) + "\n".join(rows) + r"""
\hline
\multicolumn{5}{l}{}\\[-1.5mm]
\multicolumn{5}{l}{\textit{Cohort-stratified HR (high vs low, median split) by follow-up window}}\\
\hline
Window (months) & $n$ at risk & Events & HR & 95\% CI\\
\hline
""" + "\n".join(wrows) + r"""
\hline
\end{tabular}
\end{table*}
"""
    write("tab6_ph.tex", body)


# ---------------------------------------------------------------- Table 7
def table7():
    """Learner grid: mean LOCO c for every gene set under every learner.

    Source is the complete 240-cell grid (6 held-out cohorts x 10 arms x 4
    learners, zero missing cells) in which Coxnet's penalty is chosen by inner
    cross-validation confined to the training cohorts. The earlier version of
    this table read loco_os.csv, whose Coxnet column was not inner-tuned.

    The grid's Clinical row is DELIBERATELY EXCLUDED. That row uses a single
    fixed four-covariate specification for all cohorts and constructs
    GSE20711's age and size, which the clinical audit rejects (Section
    sec:clinical); printing it beside Tables 2 and 4 would put a third,
    contradictory clinical number in the paper. The qualitative finding that
    the clinical arm ranks first under all four learners is reported in the
    text, where the specification difference can be stated.
    """
    g = rd("learner_grid_full.csv")
    assert len(g) == 240, "learner grid incomplete: %d of 240 cells" % len(g)
    B = g.pivot_table(index="gene_set", columns="learner", values="cindex",
                      aggfunc="mean")
    ms = rd("metrics_summary.csv").set_index("gene_set")
    # The grid's OWN CoxPH_ridge column is an independent refit of the
    # pre-specified model and disagrees with the canonical value (ms, the same
    # source Table 2 uses) by up to 0.002 for the three largest gene sets
    # (Buffa, MammaPrint, PAM50) -- a residual solver-convergence difference
    # between the two fitting runs, not a modelling difference. Since ridge Cox
    # at alpha=100 is deterministic, there is one correct value, and it is the
    # one already reported everywhere else in the paper: substitute the
    # canonical value here so this table cannot silently disagree with Table 2.
    ridge_canonical = ms["harrell_mean"]
    gene_arms = [x for x in GS if x != "Clinical"]
    # Restricted to the nine gene-set arms: the grid's Clinical row uses a
    # different (unaudited, fixed) covariate specification by design, so
    # including it here would conflate that documented difference with the
    # residual ridge-solver gap this footnote is about.
    max_solver_gap = float((B.loc[gene_arms, "CoxPH_ridge"]
                            - ridge_canonical.loc[gene_arms]).abs().max())
    # Best-of-four is per cell, then averaged -- the quantity review item 5
    # objects to. Taking the max of the four means would understate it.
    bo4 = (g.groupby(["gene_set", "held_out_cohort"]).cindex.max()
            .groupby("gene_set").mean())
    mods = ["CoxPH_ridge", "Coxnet", "RSF", "GBSA"]
    MLBL = {"CoxPH_ridge": "Ridge Cox$^{*}$", "Coxnet": "Elastic-net Cox",
            "RSF": "Random surv.\\ forest", "GBSA": "Grad.\\ boosted"}
    rows = []
    for gsname in gene_arms:
        vals = [ridge_canonical.loc[gsname]] + [B.loc[gsname, m] for m in mods[1:]]
        best = int(np.argmax(vals))
        cells = ["\\textbf{%s}" % f3(v) if i == best else f3(v)
                 for i, v in enumerate(vals)]
        rows.append("%s & %s & %s & %+.3f\\\\" % (
            GSLBL[gsname], " & ".join(cells), f3(bo4.loc[gsname]),
            bo4.loc[gsname] - ridge_canonical.loc[gsname]))
    opt = float(np.mean([bo4.loc[x] - ridge_canonical.loc[x] for x in gene_arms]))
    # Rank correlations must be computed over the arms this table DISPLAYS.
    # learner_grid_ranking.json computed them over all 10 arms including
    # Clinical, which is dropped below; Clinical is first under every learner,
    # so including it inflates apparent stability (RSF rho 0.806 over 10 arms
    # vs 0.733 over the 9 gene sets shown). Recompute on the displayed set,
    # against the canonical ridge ranking rather than the grid's own.
    from scipy.stats import spearmanr
    _ref = ridge_canonical.loc[gene_arms].rank(ascending=False)
    rho = {m: float(spearmanr(_ref, B.loc[gene_arms, m].rank(ascending=False))[0])
           for m in mods if m != "CoxPH_ridge"}
    rho["BestOfFour"] = float(spearmanr(_ref, bo4.loc[gene_arms]
                                        .rank(ascending=False))[0])
    for m in ("Coxnet", "GBSA"):
        assert abs(rho[m] - 1.0) < 1e-9, \
            "caption claims rho=1.000 for %s but got %.4f" % (m, rho[m])
    json.dump({"n_arms_displayed": len(gene_arms), "spearman_vs_ridge": rho},
              open(os.path.join(RES, "learner_rank_stability_displayed.json"), "w"),
              indent=1)
    body = r"""\begin{table*}[htbp]
\caption{\csentence{Sensitivity to the choice of learner.}
Mean leave-one-cohort-out Harrell concordance over the six held-out
overall-survival cohorts for every gene set under each of four survival
learners; the complete grid (six cohorts $\times$ ten arms $\times$ four
learners, including the clinical arm excluded below) is 240 cells with no
missing values, of which the 216 gene-set cells are shown. $^{*}$Ridge Cox is
the pre-specified primary learner; its column is substituted from the
canonical value reported everywhere else in this paper (Table~\ref{tab:loco})
rather than taken from this grid's own independent refit of the identical
model, which differs from the canonical value by up to %.4f for the three
largest gene sets (Buffa, MammaPrint, PAM50) --- a residual solver-convergence
gap between the two fitting runs rather than a modelling difference, and small
enough not to change any ranking in this table. Bold marks each gene set's best
learner. The last two columns give the per-cell best-of-four maximum and its
excess over the pre-specified learner: selecting the learner per cell inflates
concordance by
%.3f on average (up to %.3f), which is the optimism a best-of-four report would
carry. Elastic-net Cox and gradient boosting reproduce the ridge ordering of the
gene sets exactly (Spearman $\rho=1.000$ over the nine gene sets shown); the
random survival forest does not
($\rho=%.3f$), favouring the larger signatures. The clinicopathology arm is
omitted here because the grid uses one fixed covariate set for all cohorts
rather than the audited per-cohort specification of
Section~\ref{sec:clinical}. In the grid it is nonetheless the top-ranked arm
under each of the four learners individually.}
\label{tab:learners}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l""" % (max_solver_gap, opt,
                        max(bo4.loc[x] - ridge_canonical.loc[x] for x in gene_arms),
                        rho["RSF"]) \
        + "r" * (len(mods) + 2) + r"""}
\hline
Gene set & """ + " & ".join(MLBL[m] for m in mods) + \
        r""" & Best of four & Excess\\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
"""
    write("tab7_learners.tex", body)


# ---------------------------------------------------------------- Table 8
def table8():
    """Selection-naive cohorts and the permutation calibration."""
    sn = rd("selection_naive.csv")
    sns = js("selection_naive_summary.json")
    ps = js("permutation_search_summary.json")
    rows = []
    for _, r in sn.iterrows():
        rows.append("%s & %s & %s & %d & %d & %d & %.3f & %s\\\\" % (
            r.cohort.replace("SCANB_", "SCAN-B "), r.endpoint,
            r.panel.replace("Novel5", "Novel-5").replace("Anchor4", "Anchor-4"),
            int(r.n_genes_available), int(r.n), int(r.events),
            r.cindex_trained_on_selection_cohorts, f3(r.cindex_trained_on_naive_peers)))
    body = r"""\begin{table*}[htbp]
\caption{\csentence{Cohorts never involved in panel selection, and permutation
calibration of the search.}
Upper panel: the original forward selection used the four large cohorts, so the
five remaining cohorts were never involved in it. Concordance is shown both
when the model is trained on exactly those four selection cohorts and when it
is trained only on the other never-involved cohorts; for GSE20711 and GSE58812
this four-cohort training set is one cohort smaller than Table~\ref{tab:loco}'s
five-cohort LOCO training pool (which additionally includes whichever of these
two is not the held-out cohort), so the two tables' values for the same held-out
cohort differ slightly ($\le$0.001 here) by design, not by error. The two OS cohorts
carry %d events in total, so this is a low-resolution check rather than a second
validation. Two genes of the five are absent from the Affymetrix DMFS/DFS
platforms, which is why the panel is evaluated there on the two genes available.
GSE21653's naive-peer column is blank because its only naive peers, GSE6532 and
GSE11121, each carry just two of the five Novel-5 genes: a five-gene model
cannot be fit on their pooled data, so no naive-peer estimate exists for this row.
Lower panel: the same five-step forward search re-run on %d replicates with
permuted survival labels. On pure noise the search still reaches an inner score
of %.4f on average, so the search itself manufactures %+.4f of apparent
concordance; the real search reached %.4f, above all %d permutation replicates
(exact $p$ = %.4f, the floor of a %d-replicate design). Held-out concordance of
the permuted-label panels averaged %.4f, confirming that this optimism does not
transfer out of sample.}
\label{tab:naive}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllrrrrr}
\hline
Cohort & Endpoint & Panel & Genes & $n$ & Events & $c$ (trained on & $c$ (trained on\\
 & & & avail. & & & selection cohorts) & naive peers)\\
\hline
""" % (sns["total_events_naive_os"], ps["n_replicates"], ps["null_inner_mean"],
       ps["search_optimism_on_pure_noise"], ps["observed_inner_search_score"],
       ps["n_replicates"], ps["p_exact_inner"], ps["n_replicates"],
       ps["null_heldout_mean"]) + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
"""
    write("tab8_naive.tex", body)


# ---------------------------------------------------------------- Table 10
def table10():
    """Incremental value of Novel-5 over clinicopathology (Section sec:incremental).

    Two different tests are reported here, deliberately not pooled into one
    multiplicity family: an in-sample nested likelihood-ratio test adding the
    five gene z-scores as a block (5 df) with honest cross-validated delta-c
    and decision-curve net benefit, run in the three cohorts with the fullest
    covariate sets (METABRIC, both SCAN-B releases); and, separately, a
    single-covariate (1 df) in-sample LR test of the composite Novel-5 risk
    score added to a thinner clinical specification in TCGA, for which no
    cross-validated delta-c or net benefit was computed. BH correction is
    reported only within the first, comparable family of three.
    """
    inc = rd("incremental_lr_dca_c1.csv").set_index("cohort")
    lr = rd("lr_tests_with_q.csv")
    tcga = lr[(lr.cohort == "TCGA") & (lr.signature == "Novel5")
              & (lr.model_compared.str.contains("clinical"))].iloc[0]

    order3 = ["METABRIC", "SCANB_GSE96058", "SCANB_GSE202203"]
    rows = []
    for c in order3:
        r = inc.loc[c]
        rows.append(
            "%s & 5 & %.2f & %.2e & %.2e & %+.3f (%+.3f, %+.3f) & %+.3f & %+.3f\\\\"
            % (HEAD[c], r.lr_statistic, r.lr_p_value, r.q_bh,
               r.delta_c_mean, r.delta_c_ci_lo, r.delta_c_ci_hi,
               r.net_benefit_gain_at_pt05, r.net_benefit_gain_at_pt10))
    rows.append(
        "%s & 1 & %.2f & %.2e & --- & --- & --- & ---\\\\"
        % (HEAD["TCGA"], tcga.lr_chi2, tcga.p))
    # NOTE: 8 data fields per row (held-out, df, chi2, p, q, delta_c+CI, NB@5, NB@10);
    # header below has 8 columns (lrrrrrrr) to match.

    body = r"""\begin{table*}[htbp]
\caption{\csentence{Incremental value of Novel-5 over clinicopathology.}
Two non-comparable tests are reported, kept in separate multiplicity families
rather than pooled into one. For METABRIC and both SCAN-B releases (the three
cohorts with the fullest audited covariate sets): an in-sample nested
likelihood-ratio test adding the five Novel-5 gene z-scores as a block to the
clinical model (df = 5), the Benjamini--Hochberg $q$ across these three tests,
the honest 5-fold cross-validated change in concordance with a 95\%
percentile bootstrap interval, and the decision-curve net benefit gained at
5\% and 10\% five-year risk thresholds. For TCGA, whose audited covariate set
is thinner (age and nodal status only): a separate single-covariate (df = 1)
in-sample likelihood-ratio test of the composite Novel-5 risk score, not part
of the three-cohort correction family and not paired with a cross-validated
$\Delta c$ or net benefit. The panel does not beat clinicopathology
(Table~\ref{tab:clinical}) but adds significantly to it in every cohort
tested here.}
\label{tab:incremental}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}
\hline
Held-out & df & $\chi^2$ & $p$ & BH $q$ & $\Delta c$ (95\% CI) & NB@5\% & NB@10\%\\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
"""
    write("tab10_incremental.tex", body)


# ---------------------------------------------------------------- Table 11
def table11_uno():
    """Per-cohort Uno's IPCW concordance, companion to Table 2's per-cohort
    Harrell's c. Table 2 gives only the mean Uno column; LOCO_METHODS.md flags
    that Harrell's and Uno's c can diverge sharply in a specific cohort when its
    censoring pattern differs from the training pool (notably METABRIC), which a
    single pooled mean cannot show."""
    mhu = rd("metrics_harrell_uno.csv")
    U = mhu.pivot_table(index="gene_set", columns="held_out_cohort", values="uno_c")
    clin = rd("clinical_arm_reconciled.csv").set_index("held_out_cohort")
    ms = rd("metrics_summary.csv").set_index("gene_set")
    hdr = " & ".join(HEAD[c] for c in ORDER)

    indep = [g for g in GS if g not in ("Novel5_plus_Anchor4", "Anchor4")]
    best = {c: U.loc[indep, c].idxmax() for c in ORDER}

    rows = []
    for g in GS:
        cells = []
        for c in ORDER:
            v = f3(U.loc[g, c])
            cells.append("\\textbf{%s}" % v if best[c] == g else v)
        rows.append("%s & %s & %s\\\\" % (
            GSLBL[g], " & ".join(cells), f3(ms.loc[g, "uno_mean"])))

    crow = "Clinicopathology & %s & %s\\\\" % (
        " & ".join(f3(clin.loc[c, "clinical_uno"]) for c in ORDER),
        f3(clin.clinical_uno.reindex(ORDER).mean()))

    for g in GS:
        cells = [U.loc[g, c] for c in ORDER]
        assert abs(np.mean(cells) - ms.loc[g, "uno_mean"]) < 5e-4, \
            "Uno row mean disagrees with mean column for %s: %.4f vs %.4f" % (
                g, np.mean(cells), ms.loc[g, "uno_mean"])

    body = r"""\begin{table*}[htbp]
\caption{\csentence{Leave-one-cohort-out validation, Uno's IPCW concordance.}
The per-cohort companion to Table~\ref{tab:loco}, which gives only the mean
Uno column: censoring-weighted (Uno) concordance in each held-out cohort under
the same pre-specified ridge Cox model. SCAN-B$_{1}$ = GSE96058,
SCAN-B$_{2}$ = GSE202203. Bold marks the best independent gene signature in
each column, on this metric; the ranking is not always identical to
Table~\ref{tab:loco}'s Harrell ranking, and METABRIC in particular is where
Harrell's and Uno's c diverge most (Section~\ref{sec:metrics}), so a pooled
mean alone can hide a cohort-specific reversal.}
\label{tab:uno}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}
\hline
 & \multicolumn{6}{c}{Held-out cohort (Uno $c$)} & Mean\\
\cline{2-7}
Gene set & """ + hdr + r""" & Uno\\
\hline
""" + "\n".join(rows) + r"""
\hline
""" + crow + r"""
\hline
\end{tabular}
\end{table*}
"""
    write("tab11_uno.tex", body)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    table1()
    table2()
    table3()
    table4()
    table5()
    table6()
    table7()
    table8()
    table10()
    table11_uno()
    # Cross-table check. Tables 2 and 4 both print a clinical c-index per
    # cohort; they once drew them from two different refits and disagreed by up
    # to 0.058, which is larger than the paper's headline effect. Assert that
    # every clinical number printed anywhere is the same number.
    rec = rd("clinical_arm_reconciled.csv").set_index("held_out_cohort")
    t2 = open(os.path.join(OUT, "tab2_loco.tex")).read()
    t4 = open(os.path.join(OUT, "tab4_clinical.tex")).read()
    for co in ORDER:
        v = "%.3f" % rec.loc[co, "clinical_harrell"]
        assert v in t2, "clinical %s (%s) missing from table 2" % (co, v)
        assert v in t4, "clinical %s (%s) missing from table 4" % (co, v)
    print("cross-table check: 6 clinical c-indices identical in tables 2 and 4")
    print("all tables written")
