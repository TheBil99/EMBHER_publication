#!/usr/bin/env python
"""Summarize the calibration benchmark: {pheno_source} x {kernel}.

Every page of the PDF is one grid of axes -- ROWS are pheno_source, COLUMNS are kernel --
so the question the workflow exists to answer ("does whitening fix this geometry?") is a
left-to-right comparison within a row.

  pheno_source   dense / sparse / pcX   phenotype simulated in the kernel's eigenbasis,
                                        so the signal's position is known and controlled
                 genotype               Simulator_mafld phenotypes on real GTEx
                                        haplotypes -- realistic, but NO control over where
                                        the signal lands in the eigenbasis
  kernel         raw                    K = ZZ'/M, the full matrix
                 white_k{k}             the top-k whitened GRM

Coverage is the fraction of replicates where the true h2 falls within h2 +/- 1.96*h2_se.
If h2_se is correct this sits near 0.95 at every true_h2; systematic over/under-coverage
means the SE is mis-calibrated, not merely that the point estimate is biased. For the
genotype row, bias and coverage also reflect how much real genetic signal the embedding
captures, not just estimator correctness.

Page 7 shows convergence: on low-eigenvalue geometries the raw kernel can fail to converge
in many replicates, and those failures average into a flat line at zero that reads as
"no signal" when it means "failed to fit". Metrics on pages 1-6 are computed over ALL
replicates; page 7 is what tells you when to distrust them.

The H0: h2 = 0 p-value (page 3 QQ, pval_h2_0) is the one-sided Wald test 1 - Phi(h2/h2_se),
except for --nonneg runs, whose fits carry lrt_p: there it is that LRT p-value (chi-bar-
squared null), since the Wald test is not valid at the >= 0 constraint.
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


def pheno_order(s):
    """dense, sparse, the PC geometries by ascending rank fraction, then genotype."""
    if s == "dense":
        return (0, 0.0)
    if s == "sparse":
        return (1, 0.0)
    if s == "genotype":
        return (3, 0.0)
    if s == "randompc":
        return (2, 2.0)
    try:
        return (2, float(s[2:]))
    except ValueError:
        return (4, 0.0)


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
                    help="per-cell calibration parquets")
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-pdf", required=True)
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(p) for p in args.calibration], axis=0, ignore_index=True)

    z = 1.96
    df["covered"] = (df["h2"] - df["true_h2"]).abs() <= z * df["h2_se"]
    df["sq_err"] = (df["h2"] - df["true_h2"]) ** 2
    # H0: h2 = 0 p-value -- for --nonneg fits (lrt_p present) exact_reml_heritability.py's
    # LRT with its chi-bar-squared null, since the Wald test is not valid at the >= 0
    # constraint; otherwise the one-sided Wald test 1 - Phi(h2/h2_se)
    P_COL = "lrt_p" if "lrt_p" in df.columns and df["lrt_p"].notna().all() else None
    zstat = df["h2"] / df["h2_se"]
    df["pval_h2_0"] = (df["lrt_p"] if P_COL else
                       np.where(np.isfinite(zstat), 1.0 - norm.cdf(zstat), 1.0))
    df["over_high"] = df["true_h2"] < (df["h2"] - z * df["h2_se"])
    df["adjr2_gap"] = (df["h2"] - df["h2_adjr2"]).abs()

    AGG = dict(mean_h2=("h2", "mean"), sd_h2=("h2", "std"),
               mean_se=("h2_se", "mean"), coverage95=("covered", "mean"),
               n_reps=("h2", "count"), mse=("sq_err", "mean"),
               frac_at_boundary=("at_boundary", "mean"),
               frac_converged=("converged", "mean"),
               max_adjr2_gap=("adjr2_gap", "max"),
               n_over_high=("over_high", "sum"))

    per_exp = (df.groupby(["pheno_source", "kernel", "experiment", "true_h2"])
                 .agg(**AGG).reset_index())
    overall = (df.groupby(["pheno_source", "kernel", "true_h2"])
                 .agg(**AGG).reset_index())
    overall["experiment"] = "ALL"
    for t in (per_exp, overall):
        t["bias"] = t["mean_h2"] - t["true_h2"]
        t["rmse"] = np.sqrt(t.pop("mse"))

    summary = pd.concat([per_exp, overall], axis=0, ignore_index=True)
    summary.to_parquet(args.out_summary, index=False)
    print(f"[done] wrote {len(summary)} summary rows -> {args.out_summary}", flush=True)

    PHENO_SOURCES = sorted(df["pheno_source"].unique(), key=pheno_order)
    KERNELS = sorted(df["kernel"].unique(), key=kernel_order)
    experiments = sorted(e for e in df["experiment"].unique())
    nrow, ncol = len(PHENO_SOURCES), len(KERNELS)

    def cell_title(ps, kern):
        return f"{ps} / {kern}"

    def new_grid():
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.1 * nrow),
                                 squeeze=False)
        for i, ps in enumerate(PHENO_SOURCES):
            for j, kern in enumerate(KERNELS):
                axes[i, j].set_title(cell_title(ps, kern), fontsize=9)
        return fig, axes

    def cell(ps, kern, frame=per_exp):
        return frame[(frame.pheno_source == ps) & (frame.kernel == kern)]

    def cell_overall(ps, kern):
        return overall[(overall.pheno_source == ps) & (overall.kernel == kern)] \
            .sort_values("true_h2")

    def cell_raw(ps, kern):
        return df[(df.pheno_source == ps) & (df.kernel == kern)]

    def finish(fig, title):
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        return fig

    h2_values = sorted(df["true_h2"].unique())

    with PdfPages(args.out_pdf) as pdf:
        # Page 1: calibration -- mean h2 vs true h2, per gene.
        fig, axes = new_grid()
        for i, ps in enumerate(PHENO_SOURCES):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                ax.plot([0, 1], [0, 1], "k--", lw=1)
                sub_all = cell(ps, kern)
                for exp in experiments:
                    sub = sub_all[sub_all.experiment == exp].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.mean_h2, marker="o", ms=3, lw=1, alpha=0.8)
                ax.set_xlabel("true h2", fontsize=8)
                ax.set_ylabel("mean estimated h2", fontsize=8)
        pdf.savefig(finish(fig, "Calibration: estimated vs true h2 (per gene)"))
        plt.close(fig)

        # Page 2: violin of (h2 - true_h2), one violin per true_h2, all genes pooled.
        #
        # A violin is a KDE, which SMOOTHS. Several of these distributions are spikes at
        # a single value -- the raw kernel at pc0.3/pc0.6 pins sigma_g^2 at its boundary,
        # so thousands of replicates land on essentially one number -- and a KDE renders
        # that as a broad smooth bump extending past the data. Two things guard against
        # reading that as real spread: showextrema draws the true min/max, and the median
        # and IQR are overlaid so the actual mass is visible. A distribution with no
        # variance at all cannot be KDE'd (singular covariance), so it is drawn as a bare
        # marker at its value instead of being silently dropped.
        # Transposed grid: ROWS are kernels (raw, whitened), COLUMNS are pheno_source
        # (geometries), so each row compares all geometries under one approach. All axes
        # share y so the scale reads the same everywhere. Page 2b is the relative error
        # (h2 - true_h2) / true_h2, which is undefined at true_h2 = 0, so that level is
        # left out.
        def violin_page(relative, ylim=None):
            hv = [h for h in h2_values if h > 0] if relative else h2_values
            fig, axes = plt.subplots(ncol, nrow, figsize=(3.2 * nrow, 2.8 * ncol),
                                     squeeze=False, sharey=True)
            for j, kern in enumerate(KERNELS):
                for i, ps in enumerate(PHENO_SOURCES):
                    ax = axes[j, i]
                    ax.set_title(cell_title(ps, kern), fontsize=9)
                    raw = cell_raw(ps, kern)
                    ax.axhline(0.0, color="k", ls="--", lw=1)
                    err_by_h2 = [(raw.loc[raw.true_h2 == h2, "h2"].dropna().to_numpy() - h2)
                                 / (h2 if relative else 1.0) for h2 in hv]
                    for pos, e in enumerate(err_by_h2, start=1):
                        if len(e) == 0:
                            continue
                        if len(e) > 1 and np.ptp(e) > 1e-12:
                            parts = ax.violinplot([e], positions=[pos], widths=0.85,
                                                  showextrema=True, showmedians=False)
                            for b in parts["bodies"]:
                                b.set_facecolor("C0")
                                b.set_alpha(0.55)
                            for key in ("cmins", "cmaxes", "cbars"):
                                if key in parts:
                                    parts[key].set_linewidth(0.8)
                                    parts[key].set_color("0.35")
                        else:
                            # Degenerate: every replicate identical. Mark the value.
                            ax.plot([pos], [float(e[0])], marker="_", ms=14, color="0.35")
                        q1, med, q3 = np.percentile(e, [25, 50, 75])
                        ax.vlines(pos, q1, q3, color="black", lw=3, zorder=3)
                        ax.plot(pos, med, marker="o", ms=3.0, color="white",
                                markeredgecolor="black", markeredgewidth=0.5, zorder=4)
                    ax.set_xticks(range(1, len(hv) + 1))
                    ax.set_xticklabels([f"{h2:.2f}" for h2 in hv])
                    ax.tick_params(axis="x", labelsize=6)
                    ax.set_xlabel("true h2", fontsize=8)
                    ax.set_ylabel("(h2 - true_h2) / true_h2" if relative else "h2 - true_h2",
                                  fontsize=8)
            if ylim is not None:
                axes[0, 0].set_ylim(*ylim)   # sharey: applies to every axis
            what = "Relative error (h2 - true_h2) / true_h2" if relative else "Error"
            return finish(fig, f"{what} by true h2 (all genes pooled) -- "
                               + (f"ZOOM y in [{ylim[0]}, {ylim[1]}] -- " if ylim else "")
                               + "violin = KDE, bar = IQR, dot = median, caps = min/max")

        # page 2 (zoomed absolute error), page 2b relative (full range)
        for relative, ylim in ((False, (-0.4, 0.4)), (True, None)):
            fig = violin_page(relative, ylim)
            pdf.savefig(fig)
            plt.close(fig)

        # Page 3: QQ-plot of the H0:h2=0 p-values, all genes pooled.
        fig, axes = new_grid()
        for i, ps in enumerate(PHENO_SOURCES):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                null_df = cell_raw(ps, kern)
                null_df = null_df[null_df.true_h2 == 0]
                expected, observed, frac_p1, n_null = qq_data(null_df, "h2", "h2_se", p_col=P_COL)
                if n_null:
                    lim = max(expected.max(), observed.max()) * 1.05
                    ax.plot([0, lim], [0, lim], "k--", lw=1)
                    ax.scatter(expected, observed, s=6, alpha=0.5)
                    ax.set_xlim(0, lim)
                    ax.set_ylim(0, lim)
                ax.set_title(f"{cell_title(ps, kern)}\nn={n_null}, {frac_p1:.1%} at p=1",
                             fontsize=8)
                ax.set_xlabel("expected -log10(p)", fontsize=8)
                ax.set_ylabel("observed -log10(p)", fontsize=8)
        pdf.savefig(finish(fig, "H0: h2=0 QQ-plot -- " +
                           ("LRT, chi-bar-squared null (lrt_p)" if P_COL else "one-sided Wald, h2/h2_se")))
        plt.close(fig)

        # Page 4: 95% CI coverage vs true h2, per gene + pooled (ALL, bold black).
        fig, axes = new_grid()
        for i, ps in enumerate(PHENO_SOURCES):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                ax.axhline(0.95, color="k", ls="--", lw=1)
                sub_all = cell(ps, kern)
                for exp in experiments:
                    sub = sub_all[sub_all.experiment == exp].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.coverage95, marker="o", ms=3, lw=1, alpha=0.5)
                all_sub = cell_overall(ps, kern)
                if len(all_sub):
                    ax.plot(all_sub.true_h2, all_sub.coverage95, marker="o",
                            color="black", lw=2)
                ax.set_ylim(0, 1.02)
                ax.set_xlabel("true h2", fontsize=8)
                ax.set_ylabel("coverage95", fontsize=8)
        pdf.savefig(finish(fig, "h2_se coverage check (black = all genes pooled)"))
        plt.close(fig)

        # Page 5: fraction of replicates confidently too high (true_h2 < h2 - 1.96*se).
        fig, axes = new_grid()
        for i, ps in enumerate(PHENO_SOURCES):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                ax.axhline(0.025, color="k", ls="--", lw=1)
                sub_all = cell(ps, kern)
                for exp in experiments:
                    sub = sub_all[sub_all.experiment == exp].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.n_over_high / sub.n_reps,
                                marker="o", ms=3, lw=1, alpha=0.5)
                all_sub = cell_overall(ps, kern)
                if len(all_sub):
                    ax.plot(all_sub.true_h2, all_sub.n_over_high / all_sub.n_reps,
                            marker="o", color="black", lw=2)
                ax.set_xlabel("true h2", fontsize=8)
                ax.set_ylabel("frac true_h2 < h2-1.96se", fontsize=8)
        pdf.savefig(finish(fig, "Estimate confidently higher than truth (black = pooled)"))
        plt.close(fig)

        # Page 6: bias vs true h2, per gene.
        fig, axes = new_grid()
        for i, ps in enumerate(PHENO_SOURCES):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                ax.axhline(0.0, color="k", ls="--", lw=1)
                sub_all = cell(ps, kern)
                for exp in experiments:
                    sub = sub_all[sub_all.experiment == exp].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.bias, marker="o", ms=3, lw=1, alpha=0.8)
                ax.set_xlabel("true h2", fontsize=8)
                ax.set_ylabel("mean(h2_hat) - true_h2", fontsize=8)
        pdf.savefig(finish(fig, "Bias"))
        plt.close(fig)

        # Page 7: convergence. Pages 1-6 average non-converged fits in; where this drops
        # far below 100% those panels are reporting failures, not estimates.
        fig, axes = new_grid()
        for i, ps in enumerate(PHENO_SOURCES):
            for j, kern in enumerate(KERNELS):
                ax = axes[i, j]
                ax.axhline(1.0, color="k", ls="--", lw=1)
                sub_all = cell(ps, kern)
                for exp in experiments:
                    sub = sub_all[sub_all.experiment == exp].sort_values("true_h2")
                    if len(sub):
                        ax.plot(sub.true_h2, sub.frac_converged, marker="o", ms=3,
                                lw=1, alpha=0.5)
                all_sub = cell_overall(ps, kern)
                if len(all_sub):
                    ax.plot(all_sub.true_h2, all_sub.frac_converged, marker="o",
                            color="black", lw=2)
                ax.set_ylim(-0.02, 1.02)
                ax.set_xlabel("true h2", fontsize=8)
                ax.set_ylabel("frac converged", fontsize=8)
        pdf.savefig(finish(fig, "REML convergence (black = pooled) -- read with pages 1-6"))
        plt.close(fig)

    print(f"[done] wrote plots -> {args.out_pdf}", flush=True)

    # Console summary, one block per kernel.
    for kern in KERNELS:
        s = overall[overall.kernel == kern]
        print(f"\n{'=' * 78}\nKERNEL = {kern}\n{'=' * 78}")
        for label, col, r in (("mean estimated h2", "mean_h2", 3),
                              ("bias", "bias", 3),
                              ("95% CI coverage", "coverage95", 2),
                              ("frac converged", "frac_converged", 2)):
            print(f"\n{label}:")
            print(s.pivot_table(index="pheno_source", columns="true_h2", values=col)
                   .reindex(PHENO_SOURCES).round(r).to_string())
        gap = s.max_adjr2_gap.max()
        if np.isfinite(gap):
            print(f"\nsaturation check  max|REML - adjR2| = {gap:.2e}"
                  f"{'' if gap < 5e-3 else '   <-- WARN: whitening did not take'}")


if __name__ == "__main__":
    main()
