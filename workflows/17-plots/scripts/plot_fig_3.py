#!/usr/bin/env python
"""fig_3: Figure 3 at 183 mm wide -- (a) ancestry R^2 over genes, genotype vs unwhitened Borzoi
embedding (small violins); (b) PC1 vs PC2 of the gene window, genotype and Borzoi embedding, EUR
vs AFR donors (workflow 14's published PC scores); (c) per-layer pretrained-minus-random
enrichment of the attribution seqlets for RBP motifs, TF motifs, PhyloP and gnomAD (workflow 15's
published summary parquet, raw deltas with 95% CIs). Nothing is recomputed.

Panel a is workflow 13's in-sample R^2 of EUR vs AFR on the top 10 PCs, genotype vs the UNWHITENED (raw)
Borzoi embedding of --embedding-variant (default embeddings_final_pos), one value per gene
(paired, Wilcoxon).

Writes into --out-dir: fig_3.{pdf,png} (a and b on top, c below, one canvas), plus the tables
behind it, fig_3a_ancestry_r2.tsv (a) and fig_3b_summary.tsv (c). Titles go to the information
page after the figure page (database names in (c) are coloured text).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import pandas as pd

REPRESENTATIONS = ["Genotype", "Borzoi embedding"]
DBS = ["hocomoco_v13", "phylop241", "cisbp_rna", "gnomad"]   # panel order: strongest results first
COLOR_KEY = {"cisbp_rna": "RBP", "hocomoco_v13": "TF", "phylop241": "phyloP", "gnomad": "gnomAD"}
TITLES = {"hocomoco_v13": "TF motifs (n enriched / gene)", "phylop241": "Conservation (PhyloP obs − exp)",
          "cisbp_rna": "RBP motifs (n enriched / gene)", "gnomad": "Constraint (gnomAD log OR)"}
YLABEL = "Pretrained − random init"
W, H_A, H_B = 18.15, 5.6, 3.4    # cm: width (183 mm), scatter row height, line-plot row height
TICKS = [1, 6, 12, 18, 24]


def draw_a(fig, spec, pca, r2, mods, gene, letters=False):
    """Small R^2 violins (genotype vs unwhitened embedding, over genes) + two scatter-with-marginals
    panels in `spec`."""
    A, N, P2F = mods["A"], mods["N"], mods["P2F"]
    group_colors = {"EUR": P2F["EUR"], "AFR": P2F["AFR"]}   # earthy purple / gold, as the 3c palette
    df = pca[pca["group"].isin(group_colors)]
    inner = spec.subgridspec(1, 3, width_ratios=[0.8, 2, 2], wspace=0.5)
    vax = fig.add_subplot(inner[0, 0])
    firsts, mains = [], []
    for col, rep in enumerate(REPRESENTATIONS):
        sub = df[df["representation"] == rep]
        groups = {g: (sub.loc[sub["group"] == g, "PC1"], sub.loc[sub["group"] == g, "PC2"])
                  for g in group_colors if (sub["group"] == g).any()}
        main, top, _ = A.scatter_marginals(fig, groups, colors=group_colors,
                                           subplot_spec=inner[0, col + 1])
        main.set_box_aspect(1)
        main.set_xlabel("PC1")
        main.set_ylabel("PC2")
        if col == len(REPRESENTATIONS) - 1:
            leg = main.legend(loc="upper left", bbox_to_anchor=(1.02, 1.3), frameon=False,
                              handletextpad=0.1, borderaxespad=0)   # empty corner above the right marginal
            for t in leg.get_texts():                  # EUR / AFR written in their own colour
                t.set_color(group_colors[t.get_text()])
        elif main.get_legend() is not None:
            main.get_legend().remove()
        top.set_title(rep, loc="left")
        firsts.append(top)
        mains.append(main)
    # example gene, in italics, above the scatters next to the panel letter
    firsts[0].text(-0.12, 1.1, gene, transform=firsts[0].transAxes, ha="left", va="bottom",
                   style="italic", fontsize=mods["FONT"]["title"])
    # violins: same vertical extent as the scatters' main axes
    pos = mains[0].get_position()
    vp = vax.get_position()
    vax.set_position([vp.x0, pos.y0, vp.width, pos.height])
    A.violin(vax, [r2["genotype"], r2["embedding"]], width=0.8,
             colors=[P2F["genotype"], P2F["embedding"]])   # genotype / embedding colours
    for coll in [c for c in vax.collections if isinstance(c, LineCollection)]:
        coll.remove()                                    # no IQR bars; the median dot stays
    for coll in [c for c in vax.collections if isinstance(c, PolyCollection)]:
        coll.set_alpha(1.0)                  # opaque violin bodies (library default is 0.8)
    vax.set_xticks([0, 1])
    vax.set_xticklabels(["Genotype", "Embedding"], rotation=45, ha="right", rotation_mode="anchor")
    vax.set_ylabel("Ancestry $R^2$")
    vax.set_ylim(0, 1)
    vax.text(0.5, 1.03, N.format_pvalue(r2["p"]), transform=vax.transAxes, ha="center", va="bottom",
             fontsize=mods["FONT"]["tick"])
    if letters:
        N.panel_label(vax, "a", x=-0.6, y=1.1)
        N.panel_label(firsts[0], "b", x=-0.28, y=1.1)
    return vax


STAGES = [("encoder", 1, 10), ("transformer", 11, 18), ("decoder", 19, 24)]


def _stage_bands(ax, mods):
    """Grey shade over the transformer layers and a 4 pt stage label above the axes."""
    ax.axvspan(10.5, 18.5, color=mods["NEUTRAL"], alpha=0.15, lw=0, zorder=0)
    for name, lo, hi in STAGES:
        ax.text((lo + hi) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=mods["INK2"], fontsize=4)


def draw_b(fig, spec, summ, mods, letter=False, letter_text="c"):
    """2 x 2 per-layer line panels in `spec`; returns the first axes."""
    A, N, P2F, FONT = mods["A"], mods["N"], mods["P2F"], mods["FONT"]
    inner = spec.subgridspec(1, 4, wspace=0.55)
    axes = [fig.add_subplot(inner[0, i]) for i in range(len(DBS))]
    for i, (ax, db) in enumerate(zip(axes, DBS)):
        r = summ[summ["db"] == db].sort_values("layer_index")
        col = P2F[COLOR_KEY[db]]
        N.ref_line(ax, 0.0, ls="--", color=P2F["reference"])
        for ln in A.line(ax, r["layer_index"], r["delta"], r["ci_low"], r["ci_high"],
                         color=col, smooth=False):
            ln.set_markersize(2)
        _stage_bands(ax, mods)
        ax.text(0.0, 1.13, TITLES[db], transform=ax.transAxes, ha="left", va="bottom",
                color=col, fontsize=FONT["title"])
        ax.set_ylabel(YLABEL)
        ax.set_xticks(TICKS)
        ax.set_xlabel("Layer")
    if letter:
        N.panel_label(axes[0], letter_text, x=-0.28, y=1.33)
    return axes[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pca-tsv", required=True,
                    help="results/14-ancestry_effect_embeddings/<gene_dir>/<comparison>/"
                         "genes/<GENE>/<GENE>_pca_ancestry.tsv")
    ap.add_argument("--gene", default="FAAP20", help="name of the example gene of the scatters")
    ap.add_argument("--ancestry-summary", required=True,
                    help="results/13-ancestry_prediction/<run>/results/ancestry_prediction_summary.tsv")
    ap.add_argument("--embedding-rep", default="raw", choices=["raw", "whitened"],
                    help="92 `rep` of the embedding R^2: raw (unwhitened, default) or whitened")
    ap.add_argument("--embedding-variant", default="embeddings_final_pos",
                    help="92 `variant` of the embedding whose R^2 is plotted (the layer of the scatter)")
    ap.add_argument("--summary-parquet", required=True,
                    help="results/15-borzoi_attribution/<run>/summary/"
                         "pretrained_vs_random_adjusted-<dbs>_summary.parquet")
    ap.add_argument("--embher-plotting-src", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.embher_plotting_src)
    from embher_plotting import apply_style, archetypes as A, annotations as N, layout as L
    from embher_plotting.theme import FONT, INK2, NEUTRAL, P2F_COLORS
    mods = {"A": A, "N": N, "P2F": P2F_COLORS, "FONT": FONT, "INK2": INK2, "NEUTRAL": NEUTRAL}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pca = pd.read_csv(args.pca_tsv, sep="\t")
    from scipy.stats import wilcoxon
    anc = pd.read_csv(args.ancestry_summary, sep="\t")
    g = anc[anc["variant"] == "genotype"].set_index("gene")["r2"].rename("genotype")
    e = anc[(anc["variant"] == args.embedding_variant) & (anc["rep"] == args.embedding_rep)] \
        .set_index("gene")["r2"].rename("embedding")
    r2 = pd.concat([g, e], axis=1).dropna()
    r2.to_csv(out / "fig_3a_ancestry_r2.tsv", sep="\t")
    r2 = {"genotype": r2["genotype"].to_numpy(), "embedding": r2["embedding"].to_numpy(),
          "p": float(wilcoxon(r2["genotype"], r2["embedding"]).pvalue)}
    print(f"[info] {len(r2['genotype'])} genes; median R2 genotype {pd.Series(r2['genotype']).median():.3f}, "
          f"embedding {pd.Series(r2['embedding']).median():.3f}; Wilcoxon p {r2['p']:.2e}", flush=True)
    summ = pd.read_parquet(args.summary_parquet)
    summ = summ[summ["db"].isin(DBS)]
    summ.to_csv(out / "fig_3b_summary.tsv", sep="\t", index=False)

    apply_style()
    fig = plt.figure(figsize=L.figsize_cm(W, H_A + H_B + 2.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[H_A, H_B], hspace=0.5,
                             left=0.05, right=0.972, top=0.97, bottom=0.12)
    draw_a(fig, outer[0], pca, r2, mods, args.gene, letters=True)
    draw_b(fig, outer[1], summ, mods, letter=True)
    L.save(fig, out / "fig_3", rasterize=True, png_dpi=300)   # scatter dots raster, everything else vector

    print(f"[done] wrote {out}/fig_3.{{pdf,png}}", flush=True)


if __name__ == "__main__":
    main()
