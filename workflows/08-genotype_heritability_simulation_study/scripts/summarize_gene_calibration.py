#!/usr/bin/env python
"""Per-gene calibration summary: {p_causal} x {method}.

Direct port of workflow 09's summarize_calibration.py to workflow 08's schema:
rows are still p_causal, but columns are one per method (e.g. exact_reml_mac_maf_bins_4)
instead of a kernel, and each panel's lines are one per GENE (parsed from the raw
'experiment' column, e.g. "CXCR1_h2_0.1_pc_0.01_rep_42" -> gene "CXCR1") instead of one per synthetic
geometry experiment. Unlike workflow 09, every gene here shares the identical
true_h2/p_causal grid, so genes are directly comparable panel to panel.

  p_causal   0.01 / 0.1            causal-variant density (Simulator_mafld)
  method     <estimator>_<arm>     e.g. exact_reml_mac_maf_bins_4, he_mac_maf_ld_bins_6: the
                                   estimator (exact REML, its --nonneg fit, or HE) on one
                                   binned genotype GRM design

Pages 1-6 use CONVERGED REPLICATES ONLY. A non-converged fit
(converged=False) still writes out a real, non-NaN h2 --
unlike a hard crash (h2=NaN, which pandas already drops from any mean/count)
-- so without this filter an unreliable soft-non-converged estimate would sit
in the same average as a converged one with no way to tell them apart on
pages 1-6. Page 7 is the only page computed on the unfiltered data, since
it's the one reporting the convergence rate this filter is built on; read it
alongside pages 1-6 to see how much data the filter removed.

  1. Calibration: estimated vs true h2, one line per gene.
  2. Error distribution: violin of (h2 - true_h2) per true_h2, genes pooled.
  3. H0: h2=0 QQ-plot, genes pooled, observed -log10 p capped at 12; overlaid (if
     the columns exist) the same on the fits with at_boundary=False, converged or not,
     and on the valid ones (see 8), and the --nonneg fit's LRT p-values (lrt_p).
  4. 95% CI coverage vs true h2 (colored = per gene, black = pooled).
  5. Fraction confidently too high (true_h2 < h2 - 1.96*se), same overlay.
  6. Bias: mean(h2_hat) - true_h2, one line per gene.
  7. Convergence (UNFILTERED): fraction converged vs true h2, same overlay.
  8. Validity (UNFILTERED, only if the input has 'valid', as exact_reml_heritability.py
     writes): fraction of fits converged with lambda_min/lambda_max of V >= 1e-5.

Pages 3-5 need h2_se; build(no_se=True) skips them (HE run with --jackknife none).

Bias table (bias_summary.parquet): one row per (method, true_h2, p_causal),
genes pooled, replicates restricted to converged ones (n_unconverged is still
reported, computed from the unfiltered data, purely as a diagnostic column --
same grain/schema as the original workflow 08 bias_summary.parquet, so
downstream readers of that table are unaffected by this script replacing
plot_calibration.py).
"""
from __future__ import annotations

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

GENE_RE = re.compile(r"^(.*)_h2_")
PC_RE = re.compile(r"pc_([0-9.eE+-]+)_rep")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a85"


def method_order(m):
    """single_grm first, then maf_bins_N ascending by N."""
    if m == "single_grm":
        return (0, 0)
    mm = re.search(r"(\d+)$", m)
    return (1, int(mm.group(1)) if mm else 0)


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color="#e6e5e1", linewidth=0.7, axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)


def load(paths):
    """Concat every per-replicate h2 parquet, deriving gene and p_causal from the
    'experiment' column."""
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["gene"] = df.experiment.str.extract(GENE_RE)[0]
    df["p_causal"] = df.experiment.str.extract(PC_RE)[0].astype(float)
    return df


def qq_data(df, h2_col, se_col, p_col=None):
    """One-sided Wald p = 1 - Phi(h2/se), or the p-values in p_col if given."""
    if p_col is not None:
        pval = df[p_col].to_numpy(dtype=float)
    else:
        zstat = df[h2_col] / df[se_col]
        pval = np.where(np.isfinite(zstat), 1.0 - norm.cdf(zstat), 1.0)
    p = np.sort(pval)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([]), np.nan, 0
    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    observed = -np.log10(np.clip(p, 1e-300, 1.0))
    frac_p1 = float(np.mean(p >= 1.0 - 1e-12))
    return expected, observed, frac_p1, n


def build(df, no_se=False):
    """Build all 7 figures plus the bias table. Returns (figs, bias_table),
    figs a list in page order. Shared by this script's own CLI and
    plot_summary.py, which folds these pages into one combined PDF alongside
    plot_bin_bias's. no_se: skip pages 3-5 (they need h2_se)."""
    z = 1.96
    df_all = df  # unfiltered -- needed only for page 7 (convergence itself)

    # Pages 1-6 and bias_table are computed on CONVERGED replicates only. A
    # non-converged fit (converged=False) still
    # writes out a real, non-NaN h2 -- unlike a hard crash (h2=NaN, which
    # pandas' mean/count already skip) -- so without this filter an unreliable
    # estimate would sit in the same average as a converged one with no way to
    # tell them apart on pages 1-6. Page 7 is the only page that still needs
    # the unfiltered data, since it's the one reporting the convergence rate
    # this filter is built on.
    if "converged" in df.columns:
        df = df[df["converged"]].copy()

    df["covered"] = (df["h2"] - df["true_h2"]).abs() <= z * df["h2_se"]
    df["sq_err"] = (df["h2"] - df["true_h2"]) ** 2
    zstat = df["h2"] / df["h2_se"]
    df["pval_h2_0"] = np.where(np.isfinite(zstat), 1.0 - norm.cdf(zstat), 1.0)
    df["over_high"] = df["true_h2"] < (df["h2"] - z * df["h2_se"])

    AGG = dict(mean_h2=("h2", "mean"), sd_h2=("h2", "std"),
               mean_se=("h2_se", "mean"), coverage95=("covered", "mean"),
               n_reps=("h2", "count"), mse=("sq_err", "mean"),
               n_over_high=("over_high", "sum"))

    per_gene = (df.groupby(["method", "gene", "p_causal", "true_h2"])
                  .agg(**AGG).reset_index())
    overall = (df.groupby(["method", "p_causal", "true_h2"])
                 .agg(**AGG).reset_index())
    for t in (per_gene, overall):
        t["bias"] = t["mean_h2"] - t["true_h2"]
        t["rmse"] = np.sqrt(t.pop("mse"))

    # Page 7 (and only page 7) uses the unfiltered data -- frac_converged
    # computed here would be trivially 1.0 on the already-converged-only df.
    if "converged" in df_all.columns:
        conv_agg = dict(frac_converged=("converged", "mean"))
        per_gene_conv = (df_all.groupby(["method", "gene", "p_causal", "true_h2"])
                              .agg(**conv_agg).reset_index())
        overall_conv = (df_all.groupby(["method", "p_causal", "true_h2"])
                             .agg(**conv_agg).reset_index())
    else:
        per_gene_conv = overall_conv = None

    bias_table = overall.drop(columns=["rmse", "n_over_high"]).copy()
    bias_table = bias_table.rename(columns={"sd_h2": "sd_bias", "n_reps": "n"})
    if "converged" in df_all.columns:
        n_unconv = (df_all.groupby(["method", "p_causal", "true_h2"])["converged"]
                      .apply(lambda c: int((c == False).sum())).reset_index(name="n_unconverged"))  # noqa: E712
        bias_table = bias_table.merge(n_unconv, on=["method", "p_causal", "true_h2"])
    else:
        bias_table["n_unconverged"] = 0
    bias_table = bias_table[["method", "true_h2", "p_causal", "bias", "sd_bias",
                              "mean_h2", "n", "n_unconverged"]]
    bias_table = bias_table.rename(columns={"bias": "mean_bias"})

    methods = sorted(df_all.method.unique(), key=method_order)
    p_causal_vals = sorted(df_all.p_causal.unique())
    genes = sorted(df_all.gene.unique())
    h2_values = sorted(df_all.true_h2.unique())
    nrow, ncol = len(p_causal_vals), len(methods)

    gene_colors = dict(zip(genes, plt.cm.tab10(np.linspace(0, 1, len(genes)))))

    def cell_title(pc, method):
        return f"{method}, p_causal = {pc:g}"

    def new_grid(figsize_row=3.1):
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, figsize_row * nrow),
                                 squeeze=False)
        fig.patch.set_facecolor("#fcfcfb")
        for i, pc in enumerate(p_causal_vals):
            for j, method in enumerate(methods):
                style(axes[i, j])
                axes[i, j].set_title(cell_title(pc, method), fontsize=9, color=INK)
        return fig, axes

    def cell(pc, method, frame=per_gene):
        return frame[(frame.p_causal == pc) & (frame.method == method)]

    def cell_overall(pc, method):
        return overall[(overall.p_causal == pc) & (overall.method == method)] \
            .sort_values("true_h2")

    def cell_raw(pc, method):
        return df[(df.p_causal == pc) & (df.method == method)]

    def finish(fig, title):
        fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    figs = []

    # Page 1: calibration -- mean h2 vs true h2, per gene.
    fig, axes = new_grid()
    for i, pc in enumerate(p_causal_vals):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            sub_all = cell(pc, method)
            for gene in genes:
                sub = sub_all[sub_all.gene == gene].sort_values("true_h2")
                if len(sub):
                    ax.plot(sub.true_h2, sub.mean_h2, marker="o", ms=3, lw=1.2,
                            color=gene_colors[gene], alpha=0.85, label=gene)
            ax.set_xlabel("true h2", fontsize=8, color=INK2)
            ax.set_ylabel("mean estimated h2", fontsize=8, color=INK2)
            if i == 0 and j == ncol - 1:
                ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncols=2,
                          bbox_to_anchor=(1.02, 1.0))
    figs.append(finish(fig, "Calibration: estimated vs true h2 (per gene)"))

    # Page 2: violin of (h2 - true_h2), one violin per true_h2, genes pooled.
    fig, axes = new_grid()
    for i, pc in enumerate(p_causal_vals):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            raw = cell_raw(pc, method)
            ax.axhline(0.0, color="k", ls="--", lw=1)
            err_by_h2 = [raw.loc[raw.true_h2 == h2, "h2"].dropna().to_numpy() - h2
                         for h2 in h2_values]
            for pos, e in enumerate(err_by_h2, start=1):
                if len(e) == 0:
                    continue
                if len(e) > 1 and np.ptp(e) > 1e-12:
                    parts = ax.violinplot([e], positions=[pos], widths=0.85,
                                          showextrema=True, showmedians=False)
                    for b in parts["bodies"]:
                        b.set_facecolor("C0")  # one colour for every violin, as in workflow 09
                        b.set_alpha(0.55)
                    for key in ("cmins", "cmaxes", "cbars"):
                        if key in parts:
                            parts[key].set_linewidth(0.8)
                            parts[key].set_color("0.35")
                else:
                    ax.plot([pos], [float(e[0])], marker="_", ms=14, color="0.35")
                q1, med, q3 = np.percentile(e, [25, 50, 75])
                ax.vlines(pos, q1, q3, color="black", lw=3, zorder=3)
                ax.plot(pos, med, marker="o", ms=3.0, color="white",
                        markeredgecolor="black", markeredgewidth=0.5, zorder=4)
            ax.set_xticks(range(1, len(h2_values) + 1))
            ax.set_xticklabels([f"{h2:g}" for h2 in h2_values])
            ax.tick_params(axis="x", labelsize=6)
            ax.set_xlabel("true h2", fontsize=8, color=INK2)
            ax.set_ylabel("h2 - true_h2", fontsize=8, color=INK2)
    figs.append(finish(fig, "Error distribution by true h2 (all genes pooled) -- "
                            "violin = KDE, bar = IQR, dot = median, caps = min/max"))

    if not no_se:  # pages 3-5 are built on h2_se
        # Page 3: QQ-plot of the H0:h2=0 p-values, genes pooled -- converged fits, and
        # (if at_boundary exists) overlaid, the fits not flagged at_boundary taken from
        # the unfiltered data (so non-converged fits that did not end at_boundary are
        # in). Observed -log10 p capped at 12.
        not_bnd = df_all[~df_all["at_boundary"].astype(bool)] if "at_boundary" in df_all.columns else None
        fig, axes = new_grid()
        for i, pc in enumerate(p_causal_vals):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                null_df = cell_raw(pc, method)
                null_df = null_df[null_df.true_h2 == 0]
                expected, observed, frac_p1, n_null = qq_data(null_df, "h2", "h2_se")
                # (label, color, marker size): the at_boundary=False set contains every
                # converged fit, so it is drawn first, larger, underneath the converged one
                series = []
                if not_bnd is not None:
                    nb = not_bnd[(not_bnd.p_causal == pc) & (not_bnd.method == method)
                                 & (not_bnd.true_h2 == 0)]
                    e_nb, o_nb, _, n_nb = qq_data(nb, "h2", "h2_se")
                    series.append(("at_boundary=False", "#eb6834", 14, e_nb, o_nb, n_nb))
                series.append(("converged=True", "#2a78d6", 5, expected, observed, n_null))
                if "valid" in df_all.columns:
                    # valid (converged and lambda_min/lambda_max of V >= 1e-5, written by
                    # exact_reml_heritability.py) is a subset of converged: drawn last, smallest
                    va = df_all[df_all["valid"].astype(bool) & (df_all.p_causal == pc)
                                & (df_all.method == method) & (df_all.true_h2 == 0)]
                    e_va, o_va, _, n_va = qq_data(va, "h2", "h2_se")
                    series.append(("valid=True", "#1baf7a", 2, e_va, o_va, n_va))
                if "lrt_p" in df_all.columns:
                    # exact_reml_heritability.py --nonneg: LRT vs h2 = 0, chi-bar-squared null, all fits
                    lr = df_all[(df_all.p_causal == pc) & (df_all.method == method) & (df_all.true_h2 == 0)]
                    e_lr, o_lr, _, n_lr = qq_data(lr, "h2", "h2_se", p_col="lrt_p")
                    series.append(("LRT, chi-bar-squared null (all fits)", "#7b3fb5", 3, e_lr, o_lr, n_lr))
                series = [(lab, c, s, e, np.minimum(o, 12.0), n) for lab, c, s, e, o, n in series if n]
                if series:
                    lim = max(max(e.max(), o.max()) for *_, e, o, _ in series) * 1.05
                    ax.plot([0, lim], [0, lim], "k--", lw=1)
                    for lab, c, s, e, o, n in series:
                        ax.scatter(e, o, s=s, alpha=0.6, color=c, label=f"{lab} (n={n})")
                    if len(series) > 1:
                        ax.legend(fontsize=6, loc="upper left", frameon=False, markerscale=1.5)
                    ax.set_xlim(0, lim)
                    ax.set_ylim(0, lim)
                ax.set_title(f"{cell_title(pc, method)}\nn={n_null}, {frac_p1:.1%} at p=1",
                             fontsize=8, color=INK)
                ax.set_xlabel("expected -log10(p)", fontsize=8, color=INK2)
                ax.set_ylabel("observed -log10(p)", fontsize=8, color=INK2)
        figs.append(finish(fig, "H0: h2=0 QQ-plot (observed -log10 p capped at 12)"))

        # Page 4: 95% CI coverage vs true h2, per gene + pooled (black, bold).
        fig, axes = new_grid()
        for i, pc in enumerate(p_causal_vals):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                ax.axhline(0.95, color="k", ls="--", lw=1)
                sub_all = cell(pc, method)
                for gene in genes:
                    sub = sub_all[sub_all.gene == gene].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.coverage95, marker="o", ms=3, lw=1,
                                color=gene_colors[gene], alpha=0.5)
                all_sub = cell_overall(pc, method)
                if len(all_sub):
                    ax.plot(all_sub.true_h2, all_sub.coverage95, marker="o",
                            color="black", lw=2)
                ax.set_ylim(0, 1.02)
                ax.set_xlabel("true h2", fontsize=8, color=INK2)
                ax.set_ylabel("coverage95", fontsize=8, color=INK2)
        figs.append(finish(fig, "h2_se coverage check (black = all genes pooled)"))

        # Page 5: fraction of replicates confidently too high.
        fig, axes = new_grid()
        for i, pc in enumerate(p_causal_vals):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                ax.axhline(0.025, color="k", ls="--", lw=1)
                sub_all = cell(pc, method)
                for gene in genes:
                    sub = sub_all[sub_all.gene == gene].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.n_over_high / sub.n_reps,
                                marker="o", ms=3, lw=1, color=gene_colors[gene], alpha=0.5)
                all_sub = cell_overall(pc, method)
                if len(all_sub):
                    ax.plot(all_sub.true_h2, all_sub.n_over_high / all_sub.n_reps,
                            marker="o", color="black", lw=2)
                ax.set_xlabel("true h2", fontsize=8, color=INK2)
                ax.set_ylabel("frac true_h2 < h2-1.96se", fontsize=8, color=INK2)
        figs.append(finish(fig, "Estimate confidently higher than truth (black = pooled)"))

    # Page 6: bias vs true h2, per gene.
    fig, axes = new_grid()
    for i, pc in enumerate(p_causal_vals):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            ax.axhline(0.0, color="k", ls="--", lw=1)
            sub_all = cell(pc, method)
            for gene in genes:
                sub = sub_all[sub_all.gene == gene].sort_values("true_h2")
                if len(sub):
                    ax.plot(sub.true_h2, sub.bias, marker="o", ms=3, lw=1.2,
                            color=gene_colors[gene], alpha=0.85, label=gene)
            ax.set_xlabel("true h2", fontsize=8, color=INK2)
            ax.set_ylabel("mean(h2_hat) - true_h2", fontsize=8, color=INK2)
    figs.append(finish(fig, "Bias (per gene)"))

    # Page 7: convergence, from the UNFILTERED data (per_gene_conv/overall_conv)
    # -- this is the one page pages 1-6's converged-only filter doesn't apply
    # to, since it's what that filter is based on.
    if per_gene_conv is not None:
        fig, axes = new_grid()
        for i, pc in enumerate(p_causal_vals):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                ax.axhline(1.0, color="k", ls="--", lw=1)
                sub_all = cell(pc, method, frame=per_gene_conv)
                for gene in genes:
                    sub = sub_all[sub_all.gene == gene].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.frac_converged, marker="o", ms=3,
                                lw=1, color=gene_colors[gene], alpha=0.5)
                all_sub = overall_conv[(overall_conv.p_causal == pc) & (overall_conv.method == method)] \
                    .sort_values("true_h2")
                if len(all_sub):
                    ax.plot(all_sub.true_h2, all_sub.frac_converged, marker="o",
                            color="black", lw=2)
                ax.set_ylim(-0.02, 1.02)
                ax.set_xlabel("true h2", fontsize=8, color=INK2)
                ax.set_ylabel("frac converged", fontsize=8, color=INK2)
        figs.append(finish(fig, "Convergence (black = pooled) -- pages 1-6 are already "
                                "restricted to converged replicates only; this page shows "
                                "how much data that filter removed"))

    # Page 8: fraction of valid fits (converged and lambda_min/lambda_max of V >= 1e-5),
    # from the UNFILTERED data, same layout as page 7. Only if the input has 'valid'.
    if "valid" in df_all.columns:
        val = df_all.assign(valid=df_all["valid"].astype(bool))
        per_gene_val = val.groupby(["method", "gene", "p_causal", "true_h2"]).valid.mean().reset_index()
        overall_val = val.groupby(["method", "p_causal", "true_h2"]).valid.mean().reset_index()
        fig, axes = new_grid()
        for i, pc in enumerate(p_causal_vals):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                ax.axhline(1.0, color="k", ls="--", lw=1)
                sub_all = cell(pc, method, frame=per_gene_val)
                for gene in genes:
                    sub = sub_all[sub_all.gene == gene].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.valid, marker="o", ms=3,
                                lw=1, color=gene_colors[gene], alpha=0.5)
                all_sub = overall_val[(overall_val.p_causal == pc) & (overall_val.method == method)] \
                    .sort_values("true_h2")
                if len(all_sub):
                    ax.plot(all_sub.true_h2, all_sub.valid, marker="o", color="black", lw=2)
                ax.set_ylim(-0.02, 1.02)
                ax.set_xlabel("true h2", fontsize=8, color=INK2)
                ax.set_ylabel("frac valid", fontsize=8, color=INK2)
        figs.append(finish(fig, "Validity (black = pooled): fraction of fits that converged with "
                                "lambda_min/lambda_max of V >= 1e-5"))

    return figs, bias_table


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquets", nargs="+", required=True,
                    help="every per-replicate h2 parquet")
    ap.add_argument("--out-pdf", required=True)
    ap.add_argument("--out-table", required=True)
    args = ap.parse_args()

    from matplotlib.backends.backend_pdf import PdfPages

    df = load(args.parquets)
    figs, table = build(df)
    with PdfPages(args.out_pdf) as pdf:
        for fig in figs:
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
    table.to_parquet(args.out_table, index=False)
    print(f"[done] wrote {len(figs)} pages -> {args.out_pdf}, {len(table)} rows -> {args.out_table}")


if __name__ == "__main__":
    main()
