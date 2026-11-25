#!/usr/bin/env python
"""Summarize the joint genotype+embedding calibration benchmark: {component} x {kernel}.

Structural copy of summarize_calibration.py's page layout (same qq_data helper, same
violin/coverage/bias/convergence patterns), faceted differently because each fit now
produces TWO h2 estimates (genotype, embedding) instead of one:

  ROWS      component   genotype    the genotype GRM's variance component
                         embedding   the embedding GRM's variance component -- by
                                     construction has NO true signal (phenotypes are
                                     simulated from genotype alone), so this row
                                     should show h2 centered at 0 across the WHOLE
                                     true_h2 grid, not just at true_h2=0
  COLUMNS   kernel      raw / white_k{k}   which embedding kernel shared the fit

x-axis on every page is true_h2 (the genotype experiment variable), for BOTH rows,
so you can see whether the embedding component stays null no matter how strong the
genotype signal is. All bias/coverage/QQ/error-distribution math is against
`target_h2` (see estimate_joint_genotype_embedding.py), not `true_h2`: for the
genotype row these are identical (target_h2 == true_h2); for the embedding row
target_h2 is always 0, so every replicate (not just a true_h2==0 subset) feeds the
null-calibration pages -- that's deliberate, it's the whole diagnostic this script
exists to produce.

The H0: h2 = target_h2 p-value (page 3 QQ, pval_h2_0) is the one-sided Wald test
1 - Phi(h2/h2_se), except for --nonneg runs, whose rows carry lrt_p (combine_grm_joint.py):
there it is that LRT p-value (chi-bar-squared null), since the Wald test is not valid
at the >= 0 constraint -- for the genotype row the joint LRT of every component = 0,
for the embedding row the LRT of the embedding component alone, given the genotype ones.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import norm


def component_order(c):
    return 0 if c == "genotype" else 1


def kernel_order(k):
    """raw first, then white_k{k} / white_frac{f} by their number."""
    if k == "raw":
        return (0, 0.0)
    num = k.split("_k")[-1] if "_k" in k else k.split("frac")[-1] if "frac" in k else "1"
    return (1, float(num))


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", nargs="+", required=True,
                    help="per-cell estimate_joint_genotype_embedding.py parquets")
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-pdf", required=True)
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(p) for p in args.calibration], axis=0, ignore_index=True)

    z = 1.96
    # All stats are against target_h2 (see module docstring), not true_h2 -- the
    # x-axis grouping variable used for facet position stays true_h2 throughout.
    zstat = df["h2"] / df["h2_se"]
    df["covered"] = (df["h2"] - df["target_h2"]).abs() <= z * df["h2_se"]
    df["sq_err"] = (df["h2"] - df["target_h2"]) ** 2
    # H0 p-value -- for --nonneg fits (lrt_p present) the LRT with its chi-bar-squared
    # null (see module docstring), otherwise the one-sided Wald test 1 - Phi(h2/h2_se)
    P_COL = "lrt_p" if "lrt_p" in df.columns and df["lrt_p"].notna().all() else None
    df["pval_h2_0"] = (df["lrt_p"] if P_COL else
                       np.where(np.isfinite(zstat), 1.0 - norm.cdf(zstat), 1.0))
    df["over_high"] = df["target_h2"] < (df["h2"] - z * df["h2_se"])

    AGG = dict(mean_h2=("h2", "mean"), sd_h2=("h2", "std"),
               mean_se=("h2_se", "mean"), coverage95=("covered", "mean"),
               n_reps=("h2", "count"), mse=("sq_err", "mean"),
               frac_at_boundary=("at_boundary", "mean"),
               frac_converged=("converged", "mean"),
               n_over_high=("over_high", "sum"),
               target_h2=("target_h2", "first"))

    per_exp = (df.groupby(["component", "kernel", "experiment", "true_h2"])
                 .agg(**AGG).reset_index())
    overall = (df.groupby(["component", "kernel", "true_h2"])
                 .agg(**AGG).reset_index())
    overall["experiment"] = "ALL"
    for t in (per_exp, overall):
        t["bias"] = t["mean_h2"] - t["target_h2"]
        t["rmse"] = np.sqrt(t.pop("mse"))

    summary = pd.concat([per_exp, overall], axis=0, ignore_index=True)
    summary.to_parquet(args.out_summary, index=False)
    print(f"[done] wrote {len(summary)} summary rows -> {args.out_summary}", flush=True)

    COMPONENTS = sorted(df["component"].unique(), key=component_order)
    KERNELS = sorted(df["kernel"].unique(), key=kernel_order)
    experiments = sorted(e for e in df["experiment"].unique())
    nrow, ncol = len(COMPONENTS), len(KERNELS)

    def cell_title(comp, kern):
        return f"{comp} / {kern}"

    def new_grid():
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.1 * nrow),
                                 squeeze=False)
        for i, comp in enumerate(COMPONENTS):
            for j, kern in enumerate(KERNELS):
                axes[i, j].set_title(cell_title(comp, kern), fontsize=9)
        return fig, axes

    def cell(comp, kern, frame=per_exp):
        return frame[(frame.component == comp) & (frame.kernel == kern)]

    def cell_overall(comp, kern):
        return overall[(overall.component == comp) & (overall.kernel == kern)] \
            .sort_values("true_h2")

    def cell_raw(comp, kern):
        return df[(df.component == comp) & (df.kernel == kern)]

    def finish(fig, title):
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        return fig

    h2_values = sorted(df["true_h2"].unique())

    with PdfPages(args.out_pdf) as pdf:
        # Page 1: calibration -- mean h2 vs true h2, per gene. Reference line is
        # y=x for genotype (its target moves with true_h2) but y=0 for embedding
        # (its target never moves, no matter what true_h2/x-position is).
        fig, axes = new_grid()
        for i, comp in enumerate(COMPONENTS):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                if comp == "genotype":
                    ax.plot([0, 1], [0, 1], "k--", lw=1)
                else:
                    ax.axhline(0.0, color="k", ls="--", lw=1)
                sub_all = cell(comp, kern)
                for exp in experiments:
                    sub = sub_all[sub_all.experiment == exp].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.mean_h2, marker="o", ms=3, lw=1, alpha=0.8)
                ax.set_xlabel("true h2 (genotype)", fontsize=8)
                ax.set_ylabel("mean estimated h2", fontsize=8)
        pdf.savefig(finish(fig, "Calibration: estimated vs true h2 (per gene)"))
        plt.close(fig)

        # Page 2: violin of (h2 - target_h2), one violin per true_h2, all genes pooled.
        # See summarize_calibration.py's page 2 docstring for why a violin needs the
        # IQR/median/min-max overlay and the degenerate-variance fallback.
        fig, axes = new_grid()
        for i, comp in enumerate(COMPONENTS):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                raw = cell_raw(comp, kern)
                ax.axhline(0.0, color="k", ls="--", lw=1)
                err_by_h2 = [(raw.loc[raw.true_h2 == h2, "h2"]
                              - raw.loc[raw.true_h2 == h2, "target_h2"]).dropna().to_numpy()
                             for h2 in h2_values]
                for pos, e in enumerate(err_by_h2, start=1):
                    if len(e) == 0:
                        continue
                    if len(e) > 1 and np.ptp(e) > 1e-12:
                        parts = ax.violinplot([e], positions=[pos], widths=0.85,
                                              showextrema=True, showmedians=False)
                        for b in parts["bodies"]:
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
                ax.set_xticklabels([f"{h2:.2f}" for h2 in h2_values])
                ax.tick_params(axis="x", labelsize=6)
                ax.set_xlabel("true h2 (genotype)", fontsize=8)
                ax.set_ylabel("h2 - target_h2", fontsize=8)
        pdf.savefig(finish(fig, "Error distribution by true h2 (all genes pooled) -- "
                                "violin = KDE, bar = IQR, dot = median, caps = min/max"))
        plt.close(fig)

        # Page 3: QQ-plot of the H0:h2=target_h2 p-values, all genes pooled. For the
        # genotype row this matches summarize_calibration.py's null-only page 3
        # (true_h2==0 -> target_h2==0). For the embedding row target_h2 is always 0,
        # so EVERY true_h2 contributes -- the null holds across the whole grid.
        fig, axes = new_grid()
        for i, comp in enumerate(COMPONENTS):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                null_df = cell_raw(comp, kern)
                null_df = null_df[null_df.target_h2 == 0]
                expected, observed, frac_p1, n_null = qq_data(null_df, "h2", "h2_se", p_col=P_COL)
                if n_null:
                    lim = max(expected.max(), observed.max()) * 1.05
                    ax.plot([0, lim], [0, lim], "k--", lw=1)
                    ax.scatter(expected, observed, s=6, alpha=0.5)
                    ax.set_xlim(0, lim)
                    ax.set_ylim(0, lim)
                ax.set_title(f"{cell_title(comp, kern)}\nn={n_null}, {frac_p1:.1%} at p=1",
                             fontsize=8)
                ax.set_xlabel("expected -log10(p)", fontsize=8)
                ax.set_ylabel("observed -log10(p)", fontsize=8)
        pdf.savefig(finish(fig, "H0: h2=target_h2 QQ-plot -- " +
                           ("LRT, chi-bar-squared null (genotype: all components = 0; "
                            "embedding: embedding = 0 given genotype)" if P_COL else
                            "one-sided Wald, h2/h2_se")))
        plt.close(fig)

        # Page 4: 95% CI coverage vs true h2, per gene + pooled (ALL, bold black).
        fig, axes = new_grid()
        for i, comp in enumerate(COMPONENTS):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                ax.axhline(0.95, color="k", ls="--", lw=1)
                sub_all = cell(comp, kern)
                for exp in experiments:
                    sub = sub_all[sub_all.experiment == exp].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.coverage95, marker="o", ms=3, lw=1, alpha=0.5)
                all_sub = cell_overall(comp, kern)
                if len(all_sub):
                    ax.plot(all_sub.true_h2, all_sub.coverage95, marker="o",
                            color="black", lw=2)
                ax.set_ylim(0, 1.02)
                ax.set_xlabel("true h2 (genotype)", fontsize=8)
                ax.set_ylabel("coverage95", fontsize=8)
        pdf.savefig(finish(fig, "h2_se coverage check vs target_h2 (black = all genes pooled)"))
        plt.close(fig)

        # Page 5: fraction of replicates confidently too high (target_h2 < h2-1.96se).
        fig, axes = new_grid()
        for i, comp in enumerate(COMPONENTS):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                ax.axhline(0.025, color="k", ls="--", lw=1)
                sub_all = cell(comp, kern)
                for exp in experiments:
                    sub = sub_all[sub_all.experiment == exp].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.n_over_high / sub.n_reps,
                                marker="o", ms=3, lw=1, alpha=0.5)
                all_sub = cell_overall(comp, kern)
                if len(all_sub):
                    ax.plot(all_sub.true_h2, all_sub.n_over_high / all_sub.n_reps,
                            marker="o", color="black", lw=2)
                ax.set_xlabel("true h2 (genotype)", fontsize=8)
                ax.set_ylabel("frac target_h2 < h2-1.96se", fontsize=8)
        pdf.savefig(finish(fig, "Estimate confidently higher than truth (black = pooled)"))
        plt.close(fig)

        # Page 6: bias vs true h2, per gene.
        fig, axes = new_grid()
        for i, comp in enumerate(COMPONENTS):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                ax.axhline(0.0, color="k", ls="--", lw=1)
                sub_all = cell(comp, kern)
                for exp in experiments:
                    sub = sub_all[sub_all.experiment == exp].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.bias, marker="o", ms=3, lw=1, alpha=0.8)
                ax.set_xlabel("true h2 (genotype)", fontsize=8)
                ax.set_ylabel("mean(h2_hat) - target_h2", fontsize=8)
        pdf.savefig(finish(fig, "Bias"))
        plt.close(fig)

        # Page 7: convergence. One joint fit per (gene, kernel, true_h2, rep) --
        # converged/at_boundary are shared by both component rows of that fit.
        fig, axes = new_grid()
        for i, comp in enumerate(COMPONENTS):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                ax.axhline(1.0, color="k", ls="--", lw=1)
                sub_all = cell(comp, kern)
                for exp in experiments:
                    sub = sub_all[sub_all.experiment == exp].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.frac_converged, marker="o", ms=3,
                                lw=1, alpha=0.5)
                all_sub = cell_overall(comp, kern)
                if len(all_sub):
                    ax.plot(all_sub.true_h2, all_sub.frac_converged, marker="o",
                            color="black", lw=2)
                ax.set_ylim(-0.02, 1.02)
                ax.set_xlabel("true h2 (genotype)", fontsize=8)
                ax.set_ylabel("frac converged", fontsize=8)
        pdf.savefig(finish(fig, "Joint REML convergence (black = pooled) -- read with pages 1-6"))
        plt.close(fig)

    print(f"[done] wrote plots -> {args.out_pdf}", flush=True)

    for kern in KERNELS:
        s = overall[overall.kernel == kern]
        print(f"\n{'=' * 78}\nKERNEL = {kern}\n{'=' * 78}")
        for label, col, r in (("mean estimated h2", "mean_h2", 3),
                              ("bias", "bias", 3),
                              ("95% CI coverage", "coverage95", 2),
                              ("frac converged", "frac_converged", 2)):
            print(f"\n{label}:")
            print(s.pivot_table(index="component", columns="true_h2", values=col)
                   .reindex(COMPONENTS).round(r).to_string())


if __name__ == "__main__":
    main()
