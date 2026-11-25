#!/usr/bin/env python
"""fig_4b: mean rare-variant h2 over genes, rare genotype (2 lowest MAF/MAC bins) vs Borzoi
embedding, with bootstrap 95% CIs and the bootstrap p of the difference -- the mean panel of
workflow 16's bootstrap_bars_rare_h2.pdf, drawn with embher_plotting.

Reads workflow 16's already-published bootstrap_bars_rare_h2_summary.tsv (bar height = bootstrap
mean, CI = percentile bootstrap over genes; p = twice the smaller tail of the bootstrap
differences, never below 2 / n_boot) and per_gene_real_estimates.parquet (only for the number of genes); nothing is
recomputed. Titles go to the information page after the figure page.

The p reads "p < 2 / n_boot" when the summary's bootstrap_p_is_bound is true (no bootstrap
difference crossed zero). Without that column zero crossings cannot be told from one, so a p
at the floor reads "p ≤ 2 / n_boot".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-tsv", required=True,
                    help="results/16-rare_heritability_permutations/gtex/<run>/summary/"
                         "bootstrap_bars_rare_h2_summary.tsv")
    ap.add_argument("--real-estimates", required=True,
                    help="same run's summary/per_gene_real_estimates.parquet (gene count only)")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstraps the summary was built with")
    ap.add_argument("--embher-plotting-src", required=True,
                    help="embher_analysis's src/ directory, put on sys.path by explicit path")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.embher_plotting_src)
    from embher_plotting import apply_style, archetypes as A, annotations as N, layout as L
    from embher_plotting.theme import P2F_COLORS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(args.summary_tsv, sep="\t")
    t = t[t["statistic"] == "mean"].set_index("estimate")
    arms = ["h2_rare_genotype", "h2_rare_embedding"]
    est = t.loc[arms, "bootstrap_mean"].to_numpy(float)
    lo = t.loc[arms, "ci_lo"].to_numpy(float)
    hi = t.loc[arms, "ci_hi"].to_numpy(float)
    p_boot = float(t.loc["embedding - genotype", "bootstrap_p"])
    if "bootstrap_p_is_bound" in t.columns:
        is_bound = str(t.loc["embedding - genotype", "bootstrap_p_is_bound"]).lower() == "true"
        p_text = f"bootstrap p < {p_boot:.2g}" if is_bound else f"bootstrap p={p_boot:.2g}"
    else:                                    # older summary: at the floor, 0 or 1 crossings
        at_floor = p_boot <= 2 / args.n_boot + 1e-12
        p_text = f"bootstrap p ≤ {p_boot:.2g}" if at_floor else f"bootstrap p={p_boot:.2g}"
    n_genes = len(pd.read_parquet(args.real_estimates)[["h2_rare_genotype", "h2_rare_embedding"]].dropna())
    t.reset_index().to_csv(out_dir / "fig_4b_summary.tsv", sep="\t", index=False)

    apply_style()
    fig, ax = L.panel_grid(1, ncols=1, panel=L.figsize_cm(4.21, 6.19))   # narrow figure, 3.86 x 5.85 cm overall
    ax = ax[0]
    bars = A.bar(ax, ["$h^2_{G,rare}$", "$h^2_{E,rare}$"], est,
          err=np.vstack([est - lo, hi - est]), width=0.55,
          colors=[P2F_COLORS["genotype"], mcolors.to_rgb(P2F_COLORS["embedding"])])
    for cap in bars.errorbar.lines[1]:       # larger error-bar caps (library default is capsize 2)
        cap.set_markersize(2 * 3.5)
    span = hi.max() - min(lo.min(), 0)
    ax.set_ylim(min(lo.min(), 0) - 0.1 * span, hi.max() + 0.35 * span)
    N.bracket(ax, 0, 1, p_text, y=hi.max() + 0.12 * span)
    ax.set_ylabel("mean rare $h^2$ over genes")

    top = L.title(fig, "Rare-variant heritability: rare genotype vs embedding",
                  f"mean over genes; bars = bootstrap mean, error bars = 95% CI; {args.n_boot} "
                  f"bootstraps of {n_genes} genes; bracket = bootstrap p of the difference")
    fig.tight_layout(rect=(0, 0, 1, top))
    L.save(fig, out_dir / "fig_4b", formats=("pdf", "png"), png_dpi=300)
    print(f"[done] {n_genes} genes, {p_text} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
