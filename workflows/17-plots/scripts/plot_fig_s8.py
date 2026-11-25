#!/usr/bin/env python
"""fig_s8: rare-variant (MAF <= 0.01) heritability, embedding vs genotype, across genes -- workflow 16
(hi_q_gtex_genes_maf_le0.01).

  a    violins of the per-gene rare h2 of the two arms (rare genotype, rare embedding), the dot the
       median, dashed line at zero, y zoomed to [-0.1, 0.1]
  b, c null distribution of the across-gene mean (b) and median (c) of embedding - genotype rare h2
       from phenotype shuffling (one value per shuffle), with the observed value as a line; the text is
       the two-sided permutation P from permutation_test.tsv

Reads workflow 16's summary/per_gene_real_estimates.parquet, null_distribution.parquet and
permutation_test.tsv; nothing is recomputed. Writes fig_s8.{pdf,png} and fig_s8_summary.tsv to --out-dir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AX = 3.8          # cm: side of every square axes (same as fig_2 / fig_s5)
WSPACE = 0.7      # gap between columns, in axes widths
H = 5.6           # cm: row height (as fig_2 / fig_s5)
STATS = ("mean", "median")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-estimates", required=True,
                    help="results/16-rare_heritability_permutations/gtex/hi_q_gtex_genes_maf_le0.01/"
                         "summary/per_gene_real_estimates.parquet")
    ap.add_argument("--null-distribution", required=True, help="summary/null_distribution.parquet")
    ap.add_argument("--permutation-test", required=True, help="summary/permutation_test.tsv")
    ap.add_argument("--embher-plotting-src", required=True,
                    help="embher_analysis's src/ directory, put on sys.path by explicit path")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.embher_plotting_src)
    from embher_plotting import apply_style, archetypes as A, annotations as N, layout as L
    from embher_plotting.theme import P2F_COLORS as P2F

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    est = pd.read_parquet(args.real_estimates)[["h2_rare_genotype", "h2_rare_embedding"]].dropna()
    null = pd.read_parquet(args.null_distribution)
    tests = pd.read_csv(args.permutation_test, sep="\t")
    perm = tests[tests["test"] == "phenotype_permutation"].set_index("statistic")

    apply_style()
    W = AX * (3 + 2 * WSPACE) / 0.94
    fig = plt.figure(figsize=L.figsize_cm(W, H))
    grid = fig.add_gridspec(1, 3, wspace=WSPACE, left=0.06, right=1.0)

    # a: per-gene rare h2, the two arms
    ax = fig.add_subplot(grid[0, 0])
    # Own violin call (library's A.violin evaluates the KDE on 100 points over the whole data
    # range, up to h2 ~ 2 for genotype, which looks blocky once zoomed to +/-0.1): same style,
    # KDE on 4000 points.
    from embher_plotting.theme import BG, INK, VIOLIN_EDGE_LINEWIDTH
    data = [est["h2_rare_genotype"].to_numpy(float), est["h2_rare_embedding"].to_numpy(float)]
    parts = ax.violinplot(data, positions=[0, 1], widths=0.85, showextrema=False, points=4000)
    for body, color in zip(parts["bodies"], [P2F["genotype"], P2F["embedding"]]):
        body.set_facecolor(color)
        body.set_edgecolor(INK)
        body.set_linewidth(VIOLIN_EDGE_LINEWIDTH)
        body.set_alpha(1.0)
    for pos, d in enumerate(data):
        ax.plot(pos, np.median(d), "o", ms=2.2, color=BG, zorder=4)   # median dot
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$h^2_{G,rare}$", "$h^2_{E,rare}$"])
    N.ref_line(ax, 0.0, "y", ls=N.DASH, color=P2F["reference"])
    ax.set_ylim(-0.1, 0.1)                   # zoom: a few genes reach h2 > 2
    ax.set_ylabel("$h^2$")
    ax.set_box_aspect(1)

    # b, c: null distribution of the embedding - genotype difference, observed as a line
    rows = [{"quantity": f"{name} h2_rare_{arm}", "value": f(est[f"h2_rare_{arm}"])}
            for arm in ("genotype", "embedding")
            for name, f in (("mean", np.mean), ("median", np.median))]
    for col, stat in enumerate(STATS, start=1):
        ax = fig.add_subplot(grid[0, col])
        nd = null[f"{stat}_diff"].to_numpy(float)
        A.histogram(ax, nd, bins=30, color=P2F["random"], alpha=1.0)
        observed = float(perm.loc[f"{stat} diff", "observed"])
        p_perm = float(perm.loc[f"{stat} diff", "p_two_sided"])
        ax.set_ylim(top=ax.get_ylim()[1] * 1.25)      # headroom for the P text
        N.ref_line(ax, observed, "x", color=P2F["embedding"], lw=1.5)
        N.stat_text(ax, "P < 0.001" if p_perm < 0.001 else N.format_pvalue(p_perm),
                    loc="upper left")
        ax.set_xlabel(f"{stat} $h^2_E - h^2_G$")
        ax.set_ylabel("permutations")
        ax.set_box_aspect(1)
        rows.append({"quantity": f"{stat} diff (embedding - genotype)", "value": observed,
                     "permutation_p": p_perm, "null_mean": nd.mean(),
                     "null_sd": nd.std(ddof=1)})
    N.panel_letters(fig.axes, x=-0.32, y=1.05)
    pd.DataFrame(rows).to_csv(out / "fig_s8_summary.tsv", sep="\t", index=False)
    L.save(fig, out / "fig_s8", rasterize=True, transparent=False, png_dpi=300)
    print(f"[done] n={len(est)} genes, {len(null)} permutations -> {out}", flush=True)


if __name__ == "__main__":
    main()
