#!/usr/bin/env python
"""Calibration of single_grm vs every maf_bins_N method against known true h2.

Unlike comparing methods directly on real GTEx expression (no ground truth available
there), every phenotype here
was simulated by Simulator_mafld at a known true_h2 and causal-variant density
(p_causal), so each method's bias (estimated h2 - true h2) can be measured
directly, across 100 replicates per (true_h2, p_causal) condition.

One row of panels per p_causal value. Each panel: one x-slot per true_h2 value,
with one violin per method side by side within the slot (100 replicate
estimates each, full density shown -- nothing clipped or hidden as outliers),
and a short dashed tick marking the true_h2 the slot targets.

Bias table (bias_summary.parquet): one row per (method, true_h2, p_causal) with
mean bias, its SD, and replicate count.
"""
from __future__ import annotations

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _method_style import draw_violin, method_style, sorted_bases

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a85"

PC_RE = re.compile(r"pc_([0-9.eE+-]+)_rep")


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
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["p_causal"] = df.experiment.str.extract(PC_RE)[0].astype(float)
    return df


def panel(ax, df, methods, true_h2_vals):
    style(ax)
    n = len(methods)
    group_width = min(0.5 + 0.08 * max(0, n - 4), 0.85)  # a bit more room as methods grow
    slot = group_width / n
    width = slot * 0.6
    all_bases = sorted_bases(methods)
    idx = np.arange(len(true_h2_vals))

    for j, th2 in enumerate(true_h2_vals):
        ax.plot([j - group_width / 2 - 0.03, j + group_width / 2 + 0.03], [th2, th2],
                ls=(0, (4, 3)), color=MUTED, lw=1.1, zorder=1)

    handles = []
    for off, method in zip((np.arange(n) - (n - 1) / 2) * slot, methods):
        colr, _ = method_style(method, all_bases)  # family is carried by hue alone here
        data = [df.loc[(df.method == method) & (df.true_h2 == th2), "h2"].to_numpy(float)
                for th2 in true_h2_vals]
        draw_violin(ax, data, idx + off, width, colr, INK, MUTED)
        handles.append(mpatches.Patch(facecolor=colr, edgecolor=INK,
                                       linewidth=0.9, label=method))

    ax.set_xlim(idx.min() - 0.5, idx.max() + 0.5)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{v:g}" for v in true_h2_vals])
    ax.set_xlabel("true $h^2$", fontsize=9, color=INK2)
    ax.set_ylabel("estimated $h^2$", fontsize=9, color=INK2)
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left", ncols=2)


def build(df):
    """Build the calibration figure and its bias table from a loaded dataframe.
    Returns (fig, table). Shared by this script's own CLI and plot_summary.py,
    which folds this figure into one combined PDF alongside plot_bin_bias's."""
    methods = sorted(df.method.unique(),
                      key=lambda m: (m != "single_grm", int(re.search(r"\d+$", m).group())
                                     if re.search(r"\d+$", m) else 0))
    true_h2_vals = sorted(df.true_h2.unique())
    p_causal_vals = sorted(df.p_causal.unique())
    print(f"[info] {df.experiment.nunique()} conditions, methods: {methods}, "
          f"true_h2: {true_h2_vals}, p_causal: {p_causal_vals}")

    fig, axes = plt.subplots(len(p_causal_vals), 1,
                              figsize=(9, 4.2 * len(p_causal_vals)), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for row, pc in enumerate(p_causal_vals):
        ax = axes[row, 0]
        panel(ax, df[df.p_causal == pc], methods, true_h2_vals)
        ax.set_title(f"p_causal = {pc:g}", fontsize=10, color=INK, loc="left")

    fig.suptitle("Genotype-heritability calibration: single GRM vs MAF-binned GRM "
                 "(simulated phenotypes, known true $h^2$)\n"
                 "blue = REML, orange = HE regression",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))  # a bit more headroom for the 2-line suptitle

    df["bias"] = df.h2 - df.true_h2
    aggs = dict(mean_bias=("bias", "mean"), sd_bias=("bias", "std"),
                mean_h2=("h2", "mean"), n=("bias", "size"))
    if "converged" in df.columns:
        aggs["n_unconverged"] = ("converged", lambda c: int((c == False).sum()))  # noqa: E712
    table = df.groupby(["method", "true_h2", "p_causal"]).agg(**aggs).reset_index()

    if "converged" in table.columns:
        bad = table[table.n_unconverged > 0]
        if len(bad):
            print(f"[warn] AI-REML did not converge for {bad.n_unconverged.sum()} "
                  f"replicate(s) across {len(bad)} (method, true_h2, p_causal) group(s):")
            print(bad[["method", "true_h2", "p_causal", "n_unconverged", "n"]].to_string(index=False))
        else:
            print("[info] AI-REML converged for every replicate")
    return fig, table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquets", nargs="+", required=True)
    ap.add_argument("--out-pdf", required=True)
    ap.add_argument("--out-table", required=True)
    args = ap.parse_args()

    df = load(args.parquets)
    fig, table = build(df)
    fig.savefig(args.out_pdf, facecolor=fig.get_facecolor())
    table.to_parquet(args.out_table, index=False)
    print(f"[done] -> {args.out_pdf}, {args.out_table}")


if __name__ == "__main__":
    main()
