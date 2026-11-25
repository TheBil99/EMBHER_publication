#!/usr/bin/env python
"""Collect predict_ancestry.py's per (gene, variant) OLS R^2 tables into one summary
TSV and one figure: median EUR-vs-AFR R^2 (OLS on the top 10 PCs -- whitened onto
round(0.9 * rank) components first, then truncated to 10 -- no held-out split) across
the trunk's depth, one page per pooling axis, genotype shown as a reference line (not a
point on the depth axis, same convention as workflow 17's compare_variants.py).

Variant labels (see src/scripts/extract_layer_variants.py / workflow 12's
heritability_over_layers.nf): 'genotype', 'rna_predictions', 'rna_predictions_whole_blood', or '<layer>_<pos|chan>'.

Plot style matches workflow 17's heritability_over_layers figure
(summarize_embedding_genotype_ratio.py): square panels, no gridlines, plain (no white
edge) markers, a smoothed CI band (shape-preserving PCHIP spline -- cosmetic only,
doesn't touch the median line/markers), a numeric 1-based layer index on the x-axis
instead of the long rotated layer names, and a second PDF page mapping index -> layer
name.
"""
import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# Architecture-depth order, forward-pass order -- duplicated from workflow 17's
# compare_variants.py (small closure, not worth promoting to src/scripts/ for two
# workflow-local plotting scripts -- see that script's own comment for the reasoning).
LAYER_GROUPS = ["encoder_conv_dna", "encoder_res_tower", "skip_horizontal_conv0",
                "encoder_unet1", "skip_horizontal_conv1", "encoder_trunk",
                "transformer_block", "decoder_upsample1", "decoder_separable1",
                "decoder_upsample0", "decoder_separable0", "embeddings_cropped",
                "embeddings_final"]

# Same palette as workflow 17's h2 figures (plot_embedding_vs_genotype_h2.py,
# plot_embedding_joint_vs_rare_genotype_h2.py, summarize_embedding_genotype_ratio.py):
# blue for the embedding quantity, orange for genotype.
C = {"raw": "#2a78d6", "whitened": "#1baf7a"}
LABEL = {"raw": "embedding (raw)", "whitened": "embedding (whitened)"}
INK, MUTED = "#0b0b0b", "#8b8a85"

CI_SMOOTH_POINTS_PER_SEGMENT = 20


def layer_sort_key(layer: str):
    for gi, g in enumerate(LAYER_GROUPS):
        if layer == g:
            return (gi, 0)
        if layer.startswith(g + "_") and layer[len(g) + 1:].isdigit():
            return (gi, int(layer[len(g) + 1:]))
    return (len(LAYER_GROUPS), 0)


def smooth_band(xs: np.ndarray, ys: np.ndarray, points_per_segment: int = CI_SMOOTH_POINTS_PER_SEGMENT):
    """Oversample ys (an IQR bound) onto a finer x grid via a shape-preserving
    monotonic cubic (PCHIP) spline -- purely cosmetic smoothing of the fill_between
    band's sharp point-to-point angles; does not touch the median line/markers
    themselves. Same helper as workflow 17's summarize_embedding_genotype_ratio.py."""
    fine_xs = np.linspace(xs[0], xs[-1], (len(xs) - 1) * points_per_segment + 1)
    return fine_xs, PchipInterpolator(xs, ys)(fine_xs)


def bootstrap_median_ci(v: np.ndarray, rng, n_boot: int = 2000):
    """95% percentile-bootstrap CI of the median of v (resampling genes)."""
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    return tuple(np.percentile(np.median(v[idx], axis=1), [2.5, 97.5]))


def parse_variant(variant: str) -> tuple[str | None, str | None]:
    if variant in ("genotype", "rna_predictions", "rna_predictions_whole_blood"):
        return None, None
    layer, _, axis = variant.rpartition("_")
    if axis not in ("pos", "chan") or not layer:
        raise SystemExit(f"[error] variant doesn't parse as <layer>_<pos|chan>: {variant}")
    return layer, axis


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of *_ancestry_prediction.tsv")
    a = ap.parse_args()

    # ignore_index=True is load-bearing: each per-file table has its own 0-based
    # RangeIndex, so without it the concatenated `res` has massively duplicated index
    # values (every genotype file contributes exactly one row at index 0, every batch
    # file the same handful of small indices). An index-based .join() against that
    # would cross-join same-indexed rows from UNRELATED files instead of attaching
    # each row's own parsed layer/axis.
    res = pd.concat((pd.read_csv(f, sep="\t")
                     for f in sorted(glob.glob(f"{a.dir}/*_ancestry_prediction.tsv"))),
                    ignore_index=True)
    parsed = res["variant"].map(parse_variant)
    res["layer"] = parsed.map(lambda t: t[0])
    res["axis"] = parsed.map(lambda t: t[1])
    res.to_csv(f"{a.dir}/ancestry_prediction_summary.tsv", sep="\t", index=False)

    pd.set_option("display.width", 200)
    geno = res[res.variant == "genotype"]
    print(f"\n== genotype: median R^2 over {geno.gene.nunique()} genes: {geno.r2.median():.3f} ==")

    # Per-gene ratio of the embedding R^2 to that gene's own genotype R^2 (genotype = 1).
    g_r2 = geno.groupby("gene")["r2"].median()
    g_r2 = g_r2[g_r2 > 0]
    res["ratio"] = res["r2"] / res["gene"].map(g_r2)

    layers_by_axis = {}

    def draw_page(axes, rep):
        for col, axis in enumerate(("pos", "chan")):
            ax = axes[col]
            sub = res[(res.axis == axis) & (res.rep == rep) & res.ratio.notna()]
            layers = sorted(res[res.axis == axis]["layer"].unique(), key=layer_sort_key)
            layers_by_axis[axis] = layers
            xs = np.arange(len(layers))
            r = sub.groupby("layer")["ratio"]
            med = r.median().reindex(layers)
            # 95% CI of the median: percentile bootstrap over genes, per layer.
            rng = np.random.default_rng(0)
            ci = np.array([bootstrap_median_ci(sub.loc[sub.layer == l, "ratio"].to_numpy(), rng)
                           for l in layers])
            smooth_xs, ci_lo = smooth_band(xs, ci[:, 0])
            _, ci_hi = smooth_band(xs, ci[:, 1])
            ax.fill_between(smooth_xs, ci_lo, ci_hi, color=C[rep], alpha=0.2, lw=0)
            ax.plot(xs, med.values, color=C[rep], lw=2, marker="o", markersize=4,
                    label=LABEL[rep])
            ax.axhline(1, color="#eb6834", ls="--", label="genotype (= 1)")
            ax.set_ylim(bottom=0)
            ax.set_box_aspect(1)
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([str(i + 1) for i in range(len(layers))], fontsize=7)
            ax.set_xlabel("layer index", fontsize=8)
            ax.set_ylabel("R^2 embedding / R^2 genotype\n(median over genes, band = 95% CI of median)", fontsize=8)
            pooling = "position axis (per-channel embedding)" if axis == "pos" \
                else "channel axis (per-position embedding)"
            ax.set_title(f"pooled over the {pooling}", fontsize=9.5, loc="left", color=INK)
            ax.legend(frameon=False, fontsize=7, loc="lower right")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7)

    title = ("Predicting EUR vs AFR ancestry across the Borzoi trunk's depth, relative to "
             "genotype (OLS on the top 10 PCs, no held-out split) -- {}")
    figs = {}
    for rep, name in (("raw", "unwhitened"), ("whitened", "whitened")):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), facecolor="white",
                                 gridspec_kw={"wspace": 0.3})
        draw_page(axes, rep)
        fig.suptitle(title.format(name), x=0.02, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
        figs[rep] = fig
    # PNG: both pages stacked in one image (the nf declares a single PNG)
    png_fig, png_axes = plt.subplots(2, 2, figsize=(9, 9), facecolor="white",
                                     gridspec_kw={"wspace": 0.3, "hspace": 0.35})
    draw_page(png_axes[0], "raw")
    draw_page(png_axes[1], "whitened")
    png_fig.suptitle("Ancestry prediction R^2 relative to genotype: unwhitened (top), "
                     "whitened (bottom)", x=0.02, ha="left", fontsize=10, color=INK)
    png_fig.tight_layout()
    png_fig.savefig(f"{a.dir}/ancestry_prediction_layers.png", dpi=150, bbox_inches="tight")
    plt.close(png_fig)

    # Index -> layer name legend page. layers_by_axis["pos"] and ["chan"] are the same
    # set of layers in practice (both poolings capture the same architecture), so one
    # shared mapping suffices; if they ever diverge, fall back to the union in depth
    # order so no index is left unlabeled.
    if layers_by_axis["pos"] == layers_by_axis["chan"]:
        legend_layers = layers_by_axis["pos"]
    else:
        legend_layers = sorted(set(layers_by_axis["pos"]) | set(layers_by_axis["chan"]),
                               key=layer_sort_key)
    legend_fig, legend_ax = plt.subplots(figsize=(7, max(4.5, 0.26 * len(legend_layers) + 1.5)))
    legend_ax.axis("off")
    cell_text = [[str(i + 1), l] for i, l in enumerate(legend_layers)]
    table = legend_ax.table(cellText=cell_text, colLabels=["index", "layer"],
                            cellLoc="left", loc="upper center", colWidths=[0.15, 0.6],
                            bbox=[0, 0, 1, 0.92])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    legend_ax.set_title("Index -> layer name (shared by all panels)", fontsize=10)
    legend_fig.tight_layout()

    with PdfPages(f"{a.dir}/ancestry_prediction_layers.pdf") as pdf:
        pdf.savefig(figs["raw"], bbox_inches="tight")
        pdf.savefig(figs["whitened"], bbox_inches="tight")
        pdf.savefig(legend_fig, bbox_inches="tight")
    plt.close(figs["raw"])
    plt.close(figs["whitened"])
    plt.close(legend_fig)
    print(f"[done] wrote {a.dir}/ancestry_prediction_summary.tsv and "
          f"{a.dir}/ancestry_prediction_layers.{{pdf,png}}")


if __name__ == "__main__":
    main()
