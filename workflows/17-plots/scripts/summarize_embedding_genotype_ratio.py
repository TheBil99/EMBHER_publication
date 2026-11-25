#!/usr/bin/env python
"""Workflow 17 (plots): for each gene, what fraction of the genotype-GRM's estimated
heritability (workflow 10, genotype_heritability) does each embedding variant's own
heritability (workflow 12, heritability_over_layers) recover
(h2_embedding / h2_genotype)? Plots the median ratio across genes at each layer/axis,
with a shaded 95% CI band on that median from a bootstrap over genes (percentile
method, 2000 resamples) -- median/bootstrap rather than mean/SEM since a few
extreme-ratio genes (e.g. a tiny or near-zero genotype h2 denominator) can otherwise
drag the mean and its CI far from where most genes actually sit (the full per-gene
distribution is still written to per_gene_ratio.parquet, just not plotted here).

Reads every `*_comparison.parquet` compare_variants.py already wrote (one row per
gene x variant, joining workflow 12's embedding h2 against workflow 10's genotype h2)
-- this computes nothing the rest of the pipeline doesn't already have, it just
re-aggregates across genes instead of within one gene. rna_predictions is plotted as
the final point on the depth axis (it isn't really one, but this is how it reads
best). rna_predictions_whole_blood (Borzoi's prediction head restricted to the
whole-blood tracks) is instead a horizontal
reference line across the whole panel, same treatment as genotype (ratio = 1 by
construction) -- it isn't a point on the depth axis either, and its ratio is small and
tightly clustered across genes (see per_gene_ratio.parquet), so a flat reference line
reads better than a 26th point squeezed at the very bottom of the y-axis.

The x-axis is a plain 1-based layer index rather than the (long, ~25-entry) layer
names, to keep the plot readable; a third page maps index -> layer name.

Panels are drawn only for axes (pos/chan) actually present in the data -- a
randomly-initialized-Borzoi run (workflow 12's --embeddings_model_type borzoi_random)
only has seq_mean pooling cached, so it produces pos-axis variants only, and this
script renders a single square panel instead of the usual pos+chan pair.

Usage:
    python summarize_embedding_genotype_ratio.py \
        --comparison-parquet /path/to/*_comparison.parquet \
        --out-dir results/17-plots/heritability_over_layers/embedding_genotype_ratio
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

CI_SMOOTH_POINTS_PER_SEGMENT = 20

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

# Shared palette across workflow 17's other h2 figures (plot_embedding_vs_genotype_h2.py,
# plot_embedding_joint_vs_rare_genotype_h2.py) and workflow 13's
# summarize_ancestry_prediction.py: blue for the embedding quantity, orange for genotype.
EMBEDDING_COLOR = "#2a78d6"
GENOTYPE_COLOR = "#eb6834"
WHOLE_BLOOD_COLOR = "#1baf7a"


def smooth_band(xs: np.ndarray, ys: np.ndarray, points_per_segment: int = CI_SMOOTH_POINTS_PER_SEGMENT):
    """Oversample ys (the CI bound) onto a finer x grid via a shape-preserving
    monotonic cubic (PCHIP) spline -- purely cosmetic smoothing of the fill_between
    band's sharp point-to-point angles; does not touch the median line/markers
    themselves, and PCHIP (unlike a plain cubic spline) never overshoots past the
    original values, which matters here since a bound can sit right at 0."""
    fine_xs = np.linspace(xs[0], xs[-1], (len(xs) - 1) * points_per_segment + 1)
    return fine_xs, PchipInterpolator(xs, ys)(fine_xs)


def median_bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                        n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    """Percentile-method bootstrap 95% CI on the median, resampling genes with
    replacement -- robust to the handful of extreme-ratio genes that would otherwise
    dominate a mean-based CI."""
    if len(values) < 2:
        m = float(np.median(values)) if len(values) else float("nan")
        return m, m
    boot_medians = np.median(
        rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return float(np.percentile(boot_medians, 2.5)), float(np.percentile(boot_medians, 97.5))

# Same architecture-depth order as compare_variants.py's own LAYER_GROUPS/layer_sort_key
# (kept as a plain copy here, not an import -- this is a one-off script, not a shared
# module, and the two are small enough that duplication is cheaper than coupling them).
LAYER_GROUPS = ["encoder_conv_dna", "encoder_res_tower", "skip_horizontal_conv0",
                "encoder_unet1", "skip_horizontal_conv1", "encoder_trunk",
                "transformer_block", "decoder_upsample1", "decoder_separable1",
                "decoder_upsample0", "decoder_separable0", "embeddings_cropped",
                "embeddings_final"]

# Not really a point on the depth axis, but plotted as the final one anyway (see
# module docstring). rna_predictions_whole_blood is instead a horizontal reference
# line (REFERENCE_LINE_VARIANT below), like genotype.
TRAILING_VARIANTS = ["rna_predictions"]
REFERENCE_LINE_VARIANT = "rna_predictions_whole_blood"


def layer_sort_key(layer: str):
    for gi, g in enumerate(LAYER_GROUPS):
        if layer == g:
            return (gi, 0)
        if layer.startswith(g + "_") and layer[len(g) + 1:].isdigit():
            return (gi, int(layer[len(g) + 1:]))
    for ti, t in enumerate(TRAILING_VARIANTS):
        if layer == t:
            return (len(LAYER_GROUPS) + ti, 0)
    return (len(LAYER_GROUPS) + len(TRAILING_VARIANTS), 0)   # unknown, sort last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comparison-parquet", nargs="+", required=True,
                    help="one or more *_comparison.parquet files (one per gene, from "
                         "compare_variants.py)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.comparison_parquet)

    per_gene = []
    skipped_no_genotype = []
    for f in files:
        df = pd.read_parquet(f)
        geno = df[df["variant"] == "genotype"]
        if geno.empty:
            skipped_no_genotype.append(f)
            continue
        h_geno = geno["h2"].iloc[0]
        emb = df[df["method"] == "embedding"].copy()
        emb["ratio"] = emb["h2"] / h_geno
        emb["h2_genotype"] = h_geno
        per_gene.append(emb[["gene", "variant", "layer", "axis", "h2", "h2_genotype", "ratio"]])
    if skipped_no_genotype:
        print(f"[warn] {len(skipped_no_genotype)} gene(s) had no genotype row, skipped: "
              f"{skipped_no_genotype}")

    long = pd.concat(per_gene, ignore_index=True)
    long.to_parquet(out_dir / "per_gene_ratio.parquet", index=False)

    summary = (long.groupby(["variant", "layer", "axis"], dropna=False)["ratio"]
                    .agg(mean="mean", std="std",
                         median="median",
                         p25=lambda s: s.quantile(0.25),
                         p75=lambda s: s.quantile(0.75),
                         n="count")
                    .reset_index())
    summary.to_parquet(out_dir / "ratio_summary.parquet", index=False)
    print(summary.to_string(index=False))

    n_genes = long["gene"].nunique()
    trailing_long = long[long["variant"].isin(TRAILING_VARIANTS)]
    whole_blood_ratio = long.loc[long["variant"] == REFERENCE_LINE_VARIANT, "ratio"].median()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Only axes actually present in the data get a panel -- a pretrained run has both
    # pos and chan; a random-init run (seq_mean pooling only) has pos alone.
    present_axes = [a for a in ("pos", "chan") if (long["axis"] == a).any()]
    if not present_axes:
        raise SystemExit("[error] no pos/chan layer rows in the comparison tables")
    # pos on top, chan on bottom (when both are present), one figure, shared y-axis --
    # so the two poolings' ratio magnitudes are directly comparable at a glance, not
    # just shape. Each panel's own box is forced square (set_box_aspect below)
    # regardless of the data ranges, so figsize is sized for N square panels stacked
    # vertically.
    fig, axes = plt.subplots(len(present_axes), 1, figsize=(4.5, 4.5 * len(present_axes)),
                             sharey=True, squeeze=False)
    axes = axes[:, 0]
    layers_by_axis = {}
    for ax, axis in zip(axes, present_axes):
        # rna_predictions has no axis (pos/chan) of its own -- appended to both panels
        # as the final point on the depth axis, using each row's own variant name as
        # its "layer" label.
        sub = pd.concat([long[long["axis"] == axis], trailing_long], ignore_index=True)
        sub["layer"] = sub["layer"].fillna(sub["variant"])
        layers = sorted(sub["layer"].unique(), key=layer_sort_key)
        layers_by_axis[axis] = layers

        # Median +/- bootstrap 95% CI per layer across genes -- the per-gene ratios
        # themselves still live in per_gene_ratio.parquet for anyone who wants the
        # full distribution.
        medians, ci_lo, ci_hi = [], [], []
        for l in layers:
            vals = sub.loc[sub["layer"] == l, "ratio"].dropna().to_numpy()
            medians.append(float(np.median(vals)) if len(vals) else float("nan"))
            lo, hi = median_bootstrap_ci(vals, rng)
            ci_lo.append(lo)
            ci_hi.append(hi)
        xs = range(len(layers))
        smooth_xs, smooth_lo = smooth_band(np.arange(len(layers)), np.array(ci_lo))
        _, smooth_hi = smooth_band(np.arange(len(layers)), np.array(ci_hi))
        ax.fill_between(smooth_xs, smooth_lo, smooth_hi, color=EMBEDDING_COLOR, alpha=0.18,
                        lw=0, label="95% CI (bootstrap)")
        ax.plot(xs, medians, "o-", color=EMBEDDING_COLOR, lw=1.5, markersize=4, zorder=3,
                label="median")
        ax.axhline(1.0, color=GENOTYPE_COLOR, ls="--", label="genotype (ratio = 1)")
        if pd.notna(whole_blood_ratio):
            ax.axhline(whole_blood_ratio, color=WHOLE_BLOOD_COLOR, ls=":",
                      label=f"rna_predictions_whole_blood (median {whole_blood_ratio:.3f})")

        ax.set_ylabel("median h2_embedding / h2_genotype")
        # Numeric layer index on the x-axis (1-based) instead of the full layer name
        # -- the names are long and rotating them crowds the plot; the index-to-name
        # mapping is on its own page instead (same index numbering on both axes'
        # panels, since layers_by_axis["pos"] == layers_by_axis["chan"] in practice).
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(i + 1) for i in range(len(layers))], fontsize=7)
        for lbl, layer in zip(ax.get_xticklabels(), layers):
            if layer in TRAILING_VARIANTS:
                lbl.set_color(GENOTYPE_COLOR)
        ax.legend(fontsize=7, frameon=False)
        ax.set_box_aspect(1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        axis_label = "position axis (per-channel embedding)" if axis == "pos" \
            else "channel axis (per-position embedding)"
        ax.set_title(f"pooled over the {axis_label}")
    fig.suptitle(f"h2_embedding / h2_genotype across trunk depth (n={n_genes} genes)")
    fig.tight_layout()

    # Index -> layer name legend page. When both pos and chan are present they're the
    # same set of variants in practice (both poolings capture the same architecture
    # layers, plus the same trailing variants), so one shared mapping suffices; if they
    # ever diverge (or only one axis is present at all, e.g. a random-init run), fall
    # back to the union in depth order so no index is left unlabeled.
    axis_layer_sets = list(layers_by_axis.values())
    if all(s == axis_layer_sets[0] for s in axis_layer_sets):
        legend_layers = axis_layer_sets[0]
    else:
        legend_layers = sorted(set().union(*axis_layer_sets), key=layer_sort_key)
    legend_fig, legend_ax = plt.subplots(figsize=(7, max(4, 0.22 * len(legend_layers) + 1)))
    legend_ax.axis("off")
    cell_text = [[str(i + 1), l] for i, l in enumerate(legend_layers)]
    cell_colors = [["#fff3e6" if l in TRAILING_VARIANTS else "white"] * 2
                   for l in legend_layers]
    table = legend_ax.table(cellText=cell_text, colLabels=["index", "layer"],
                            cellColours=cell_colors, cellLoc="left", loc="center",
                            colWidths=[0.15, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    legend_ax.set_title("Index -> layer name (shared by both panels above)",
                        fontsize=10)
    legend_fig.tight_layout()

    with PdfPages(out_dir / "ratio_across_layers.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.3)
        pdf.savefig(legend_fig, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    plt.close(legend_fig)
    print(f"[done] wrote {out_dir}/per_gene_ratio.parquet, ratio_summary.parquet, "
          f"ratio_across_layers.pdf")


if __name__ == "__main__":
    main()
