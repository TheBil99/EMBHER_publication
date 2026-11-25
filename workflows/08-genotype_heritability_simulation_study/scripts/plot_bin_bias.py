#!/usr/bin/env python
"""Per-bin calibration: the estimated h2_k1, h2_k2, ... of each binned method vs
the true per-bin heritability computed by compute_true_bin_h2.py from the
simulator's own betas, using the identical MAF bin edges.

One row of panels per binned method. Each panel: one x-slot per true_h2
condition, with one violin per bin side by side (full density shown, nothing
clipped or hidden as outliers), showing that bin's BIAS
(the h2_k{i} estimate minus the TRUE per-replicate h2_k{i} for that same
replicate) rather than the raw estimate against a single reference value.
Plotting the bias directly -- not raw estimates against one reference tick --
matters because the true per-bin split is itself not fixed: the same SNPs are
causal in every replicate, but each replicate draws its own effect sizes, so
how much variance lands in each bin varies replicate to replicate too (see
compute_true_bin_h2.py's per-replicate true_h2_k{i}). Subtracting the matched
per-replicate true value nets that spread out of the box, leaving only actual
estimation error -- a single "true" reference line could not do that, since it
collapses the true value's own variability into one number.

The panel and bias table are restricted to CONVERGED replicates only -- same
reasoning as summarize_gene_calibration.py's pages 1-6: a soft non-convergence
still writes a real h2_k{i}, so it's excluded here rather than silently
averaged in as if it were reliable.

Bias table (bin_bias_summary.parquet): one row per (method, bin, true_h2,
p_causal) with mean_bias (= estimated h2_k{i} - true per-replicate h2_k{i}), its
SD, mean estimated/true h2, and CONVERGED replicate count (n_unconverged is
still reported, computed from the unfiltered data, purely as a diagnostic).
"""
from __future__ import annotations

import argparse

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _method_style import (HE_HUE, REML_HUE, _shade, draw_violin, is_he,
                            method_nbins, method_strategy)


def _effective_nbins(sub):
    """Number of h2_k{i} columns actually populated for this method's rows --
    the actual bin count a method's fit produced, rather than trusting
    method_nbins()'s name-parsed guess."""
    i = 0
    while f"h2_k{i + 1}" in sub.columns and sub[f"h2_k{i + 1}"].notna().any():
        i += 1
    return i

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a85"


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
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def panel(ax, merged, nbins, true_h2_vals, method):
    style(ax)
    group_width = 0.5
    slot = group_width / nbins
    width = slot * 0.6
    # Same family hue as plot_calibration.py (blue=REML, orange=HE),
    # shaded by bin index within this one method's own panel -- keeps every
    # panel visually tagged to its estimator family even before the title is
    # read. Family is carried by hue alone (no dashed/solid edge split): the
    # gradients read clearly on their own.
    hue = HE_HUE if is_he(method) else REML_HUE
    colors = [_shade(hue, 0.45 + 0.55 * (i / max(nbins - 1, 1))) for i in range(nbins)]
    idx = np.arange(len(true_h2_vals))

    ax.axhline(0, color=MUTED, lw=0.9, zorder=1)

    for i in range(1, nbins + 1):
        colr = colors[i - 1]
        off = (i - 1 - (nbins - 1) / 2) * slot
        bias = [(merged.loc[merged.true_h2 == th2, f"h2_k{i}"]
                 - merged.loc[merged.true_h2 == th2, f"true_h2_k{i}"]).to_numpy(float)
                for th2 in true_h2_vals]

        draw_violin(ax, bias, idx + off, width, colr, INK, MUTED)
        ax.plot([], [], color=colr, marker="s", ls="", label=f"bin {i}")

    ax.set_xlim(idx.min() - 0.5, idx.max() + 0.5)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{v:g}" for v in true_h2_vals])
    ax.set_xlabel("true total $h^2$", fontsize=9, color=INK2)
    ax.set_ylabel("bias ($h^2_k$ estimate $-$ matched true $h^2_k$)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def build(est, truth):
    """Build the per-bin bias figure and its table from loaded dataframes.
    Returns (fig, table). Shared by this script's own CLI and plot_summary.py,
    which folds this figure into one combined PDF alongside plot_calibration's."""
    # a true_bin_h2 table without a 'strategy' column holds fixed-width bins
    if "strategy" not in truth.columns:
        truth["strategy"] = "width"
    else:
        truth["strategy"] = truth["strategy"].fillna("width")

    methods = sorted(est.method.unique(), key=method_nbins)
    p_causal_vals = sorted(est.p_causal.unique())
    true_h2_vals = sorted(est.true_h2.unique())

    fig, axes = plt.subplots(len(methods), len(p_causal_vals), sharey=True,
                              figsize=(6.5 * len(p_causal_vals), 4.6 * len(methods)), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    table_rows = []
    for row, method in enumerate(methods):
        nbins = _effective_nbins(est[est.method == method])
        strategy = method_strategy(method)
        for col, pc in enumerate(p_causal_vals):
            sub_est_all = est[(est.method == method) & (est.p_causal == pc)]
            # Panel and table are restricted to CONVERGED replicates only (see
            # summarize_gene_calibration.py's docstring for why a soft
            # non-convergence still writes a real, non-NaN h2_k{i} that would
            # otherwise sit unflagged in the same average as a converged one).
            # n_unconverged below is still computed from the unfiltered
            # sub_est_all, purely as a diagnostic count.
            sub_est = (sub_est_all[sub_est_all["converged"]]
                        if "converged" in sub_est_all.columns else sub_est_all)
            # Filter truth to the SAME bin count AND strategy as this method:
            # true_bin_h2 files just enumerate pheno_0..pheno_99 regardless of how
            # their bins were chosen, so two DIFFERENT bin definitions that happen to
            # produce the same bin count (e.g. a width-3 split and an eqcount split
            # that only achieved 3 effective bins) would otherwise be indistinguishable
            # by nbins alone, and the merge below could match a method against the
            # wrong strategy's truth.
            sub_truth = truth[(truth.nbins == nbins) & (truth.strategy == strategy) &
                               truth.experiment.isin(sub_est.experiment.unique())]
            merged = sub_est.merge(sub_truth, on=["experiment", "phenotype", "true_h2"],
                                     how="inner", suffixes=("", "_true"))
            if merged.empty or f"true_h2_k1" not in merged.columns:
                axes[row, col].axis("off")
                continue
            panel(axes[row, col], merged, nbins, true_h2_vals, method)
            axes[row, col].set_title(f"{method}, p_causal = {pc:g}", fontsize=10, color=INK, loc="left")

            n_unconverged_by_h2 = (
                sub_est_all.groupby("true_h2")["converged"]
                            .apply(lambda c: int((c == False).sum()))  # noqa: E712
                if "converged" in sub_est_all.columns else {})

            for i in range(1, nbins + 1):
                for th2 in true_h2_vals:
                    d = merged.loc[merged.true_h2 == th2]
                    bias = d[f"h2_k{i}"] - d[f"true_h2_k{i}"]
                    # named table_row, not row: `row` is the outer loop's subplot-row
                    # index (used right after by axes[row, col]) -- reusing that name
                    # here shadowed it and corrupted the next column's axes lookup.
                    table_row = dict(
                        method=method, bin=i, true_h2=th2, p_causal=pc,
                        mean_bias=bias.mean(), sd_bias=bias.std(),
                        mean_est_h2=d[f"h2_k{i}"].mean(), mean_true_h2=d[f"true_h2_k{i}"].mean(),
                        n=len(d))
                    # One fit produces every bin's h2_k{i} at once,
                    # so convergence is the same value across bins for a given replicate --
                    # reported once per group, not re-derived per bin.
                    if "converged" in sub_est_all.columns:
                        table_row["n_unconverged"] = int(n_unconverged_by_h2.get(th2, 0))
                    table_rows.append(table_row)

    fig.suptitle("Per-bin calibration: bias of each binned method against true per-bin $h^2$ "
                 "(from simulator betas, matched per replicate)\n"
                 "blue = REML, orange = HE regression",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    table = pd.DataFrame(table_rows).sort_values(["p_causal", "true_h2", "method", "bin"])

    if "n_unconverged" in table.columns:
        bad = table[(table.n_unconverged > 0) & (table.bin == 1)]  # dedupe across bins
        if len(bad):
            print(f"[warn] AI-REML did not converge for {bad.n_unconverged.sum()} "
                  f"replicate(s) across {len(bad)} (method, true_h2, p_causal) group(s):")
            print(bad[["method", "true_h2", "p_causal", "n_unconverged", "n"]].to_string(index=False))
        else:
            print("[info] AI-REML converged for every replicate")
    return fig, table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h2-parquets", nargs="+", required=True,
                     help="every per-replicate h2 parquet (per-bin estimates of the binned methods)")
    ap.add_argument("--true-bin-parquets", nargs="+", required=True,
                     help="every *_true_bins_N.parquet (compute_true_bin_h2.py output)")
    ap.add_argument("--out-pdf", required=True)
    ap.add_argument("--out-table", required=True)
    args = ap.parse_args()

    est = load(args.h2_parquets)
    truth = load(args.true_bin_parquets)
    est["p_causal"] = est.experiment.str.extract(r"pc_([0-9.eE+-]+)_rep")[0].astype(float)

    fig, table = build(est, truth)
    fig.savefig(args.out_pdf, facecolor=fig.get_facecolor())
    table.to_parquet(args.out_table, index=False)
    print(f"[done] -> {args.out_pdf}, {args.out_table}")


if __name__ == "__main__":
    main()
