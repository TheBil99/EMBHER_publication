#!/usr/bin/env python
"""Summary figures for the multi-gene embedding-variant comparison.

Reads the combined all_genes_comparison.parquet (concatenation of every gene's
compare_variants.py output) and writes ONE multi-page PDF, one page per figure:
  1. h2 distribution across genes, per variant, one panel per MAF condition.
  2. genotype vs joined_mean (the current workflow-06 approach) h2 per gene,
     one panel per MAF condition, with a y=x reference line.
  3. same genotype-vs-embedding scatter as (2), one row per embedding variant.
  4. genotype-minus-embedding h2 gap in the rare-only band, per embedding variant,
     across genes -- which variant leaves the smallest gap.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

# Okabe-Ito colorblind-safe qualitative palette. genotype gets black (it's the
# reference every embedding variant is compared against); the 6 embedding variants
# get the remaining 6 colors in a fixed order (never reassigned/cycled per subset).
COLORS = {
    "genotype": "#000000",
    "joined_mean": "#0072B2",
    "joined_exon": "#56B4E9",
    "trunk_mean": "#E69F00",
    "trunk_exon": "#D55E00",
    "res_tower_mean": "#009E73",
    "transformer_mean": "#CC79A7",
}
MAF_ORDER = ["0.0", "0.01", "0.0to0.01"]
MAF_TITLE = {"0.0": "MAF > 0.0", "0.01": "MAF > 0.01", "0.0to0.01": "0.0 < MAF ≤ 0.01"}
VARIANT_ORDER = ["genotype", "joined_mean", "joined_exon", "trunk_mean",
                 "trunk_exon", "res_tower_mean", "transformer_mean"]


def fig1_distributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, maf in zip(axes, MAF_ORDER):
        sub = df[df.maf == maf]
        data = [sub.loc[sub.variant == v, "h2"].to_numpy() for v in VARIANT_ORDER]
        bp = ax.boxplot(data, tick_labels=VARIANT_ORDER, showfliers=False, patch_artist=True,
                        widths=0.6, medianprops=dict(color="white"))
        for patch, v in zip(bp["boxes"], VARIANT_ORDER):
            patch.set_facecolor(COLORS[v])
            patch.set_alpha(0.85)
        for v_i, v in enumerate(VARIANT_ORDER):
            y = sub.loc[sub.variant == v, "h2"].to_numpy()
            x = np.full(len(y), v_i + 1) + np.random.default_rng(0).uniform(-0.08, 0.08, len(y))
            ax.scatter(x, y, s=6, color="black", alpha=0.25, zorder=3)
        ax.set_title(MAF_TITLE[maf])
        ax.tick_params(axis="x", rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.set_ylabel("h2" if maf == MAF_ORDER[0] else "")
        ax.axhline(0, color="#888888", lw=0.8, zorder=0)
    fig.suptitle(f"h2 across {df.gene.nunique()} genes, by variant and MAF condition")
    fig.tight_layout()
    return fig


def _plot_paired_scatter(ax, df, maf, variant, show_xlabel=True, show_ylabel=True):
    """One genotype-vs-`variant` scatter panel, one point per gene, for one MAF
    condition. Always scales the panel to its OWN data -- see fig2's note on why
    these panels must never share axes."""
    sub = df[df.maf == maf]
    geno = sub[sub.variant == "genotype"].set_index("gene")["h2"]
    emb = sub[sub.variant == variant].set_index("gene")["h2"]
    common = geno.index.intersection(emb.index)
    lo, hi = 0, max(geno.max(), emb.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1, ls="--", zorder=0)
    ax.scatter(geno.loc[common], emb.loc[common], s=18,
              color=COLORS[variant], alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if show_xlabel:
        ax.set_xlabel("genotype h2")
    if show_ylabel:
        ax.set_ylabel(f"{variant} h2")
    ax.set_aspect("equal")


def fig2_paired_scatter(df):
    # NB no sharex/sharey: the three MAF panels have very different h2 ranges (the
    # rare-only band is near-zero for embeddings), and shared axes only keep the
    # limits from whichever set_xlim/set_ylim call runs last -- silently squashing
    # every panel onto the smallest one's range.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, maf in zip(axes, MAF_ORDER):
        _plot_paired_scatter(ax, df, maf, "joined_mean", show_ylabel=(maf == MAF_ORDER[0]))
        ax.set_title(MAF_TITLE[maf])
    fig.suptitle("genotype vs. current embedding approach (joined_mean), per gene "
                f"(n={df.gene.nunique()})")
    fig.tight_layout()
    return fig


def fig4_paired_scatter_all_variants(df):
    """Same genotype-vs-embedding scatter as fig2, but one row per embedding
    variant (6 rows x 3 MAF columns) instead of just joined_mean."""
    variants = [v for v in VARIANT_ORDER if v != "genotype"]
    fig, axes = plt.subplots(len(variants), 3, figsize=(13, 3.6 * len(variants)))
    for row, variant in enumerate(variants):
        for col, maf in enumerate(MAF_ORDER):
            ax = axes[row, col]
            _plot_paired_scatter(ax, df, maf, variant,
                                show_xlabel=(row == len(variants) - 1),
                                show_ylabel=(col == 0))
            if row == 0:
                ax.set_title(MAF_TITLE[maf])
    # Explicit top margin, not left to tight_layout's default: on a figure this tall
    # (6 rows) the automatic suptitle reservation is too small and the top row's own
    # column titles overlap it -- reserve real space and place the suptitle above that.
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(f"genotype vs. every embedding variant, per gene (n={df.gene.nunique()})",
                y=0.995)
    return fig


def fig3_rare_gap(df):
    rare = df[df.maf == "0.0to0.01"]
    geno = rare[rare.variant == "genotype"].set_index("gene")["h2"]
    variants = [v for v in VARIANT_ORDER if v != "genotype"]
    gaps = []
    for v in variants:
        emb = rare[rare.variant == v].set_index("gene")["h2"]
        common = geno.index.intersection(emb.index)
        gaps.append((geno.loc[common] - emb.loc[common]).to_numpy())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(gaps, tick_labels=variants, showfliers=False, patch_artist=True,
                    widths=0.6, medianprops=dict(color="white"))
    for patch, v in zip(bp["boxes"], variants):
        patch.set_facecolor(COLORS[v])
        patch.set_alpha(0.85)
    rng = np.random.default_rng(0)
    for i, (v, g) in enumerate(zip(variants, gaps)):
        x = np.full(len(g), i + 1) + rng.uniform(-0.08, 0.08, len(g))
        ax.scatter(x, g, s=6, color="black", alpha=0.25, zorder=3)
    ax.axhline(0, color="#888888", lw=1, ls="--",
              label="0 = embedding matches genotype")
    ax.set_ylabel("genotype h2 − embedding h2  (rare-only band)")
    ax.set_title(f"Rare-variant heritability gap by embedding variant (n={len(geno)} genes)")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", required=True, help="all_genes_comparison.parquet")
    ap.add_argument("--out-pdf", required=True, help="single multi-page output PDF")
    args = ap.parse_args()

    df = pd.read_parquet(args.comparison)
    figs = [fig1_distributions(df), fig2_paired_scatter(df),
            fig4_paired_scatter_all_variants(df), fig3_rare_gap(df)]
    with PdfPages(args.out_pdf) as pdf:
        for fig in figs:
            # bbox_inches="tight" recomputes the page bounds from every rendered
            # artist (rotated tick labels, legend, suptitle) so nothing gets clipped.
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.3)
            plt.close(fig)
    print(f"[done] wrote {len(figs)}-page PDF -> {args.out_pdf}")


if __name__ == "__main__":
    main()
