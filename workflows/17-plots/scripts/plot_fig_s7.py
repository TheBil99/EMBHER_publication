#!/usr/bin/env python
"""fig_s7: how well Borzoi's embedding predicts EUR vs AFR ancestry across the trunk's depth,
relative to genotype -- median R^2_embedding / R^2_genotype per layer (unwhitened embedding,
pooled over the position axis = per-channel embedding), drawn with embher_plotting.

Reads workflow 13's already-published ancestry_prediction_summary.tsv (OLS R^2 on the top 10
PCs, no held-out split); nothing is recomputed. Per gene and layer the ratio is the embedding R^2
over that gene's own genotype R^2 (genes with genotype R^2 <= 0 left out). Each layer is the
median over genes with a percentile-bootstrap 95% CI on the median (workflow 17's
summarize_embedding_genotype_ratio.py's median_bootstrap_ci, imported by path, same convention
as fig_2b in fig_2); genotype is the dashed reference line at 1, not a point on the depth axis.

Titles go to the information page after the figure page (embher_plotting).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-tsv", required=True,
                    help="results/13-ancestry_prediction/hi_q_random500/results/"
                         "ancestry_prediction_summary.tsv")
    ap.add_argument("--rep", default="raw", choices=["raw", "whitened"],
                    help="embedding representation (default: %(default)s = unwhitened)")
    ap.add_argument("--axis", default="pos", choices=["pos", "chan"],
                    help="pooling axis (default: %(default)s = position axis, per-channel embedding)")
    ap.add_argument("--ratio-script", required=True,
                    help="workflows/17-plots/scripts/summarize_embedding_genotype_ratio.py -- "
                         "imported by path for layer_sort_key / median_bootstrap_ci / BOOTSTRAP_SEED")
    ap.add_argument("--embher-plotting-src", required=True,
                    help="embher_analysis's src/ directory, put on sys.path by explicit path")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.embher_plotting_src)
    from embher_plotting import apply_style, archetypes as A, annotations as N, layout as L
    from embher_plotting.theme import P2F_COLORS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratio_mod = _load_module("ratio_mod", Path(args.ratio_script))

    res = pd.read_csv(args.summary_tsv, sep="\t")
    geno = res[res["variant"] == "genotype"].groupby("gene")["r2"].median()
    geno = geno[geno > 0]
    emb = res[(res["axis"] == args.axis) & (res["rep"] == args.rep)].copy()
    layers = sorted(emb["layer"].unique(), key=ratio_mod.layer_sort_key)
    emb["ratio"] = emb["r2"] / emb["gene"].map(geno)

    rng = np.random.default_rng(ratio_mod.BOOTSTRAP_SEED)
    rows = []
    for layer in layers:
        vals = emb.loc[emb["layer"] == layer, "ratio"].dropna().to_numpy()
        lo, hi = ratio_mod.median_bootstrap_ci(vals, rng)
        rows.append(dict(layer=layer, median=float(np.median(vals)), ci_lo=lo, ci_hi=hi,
                         n_genes=len(vals)))
    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "fig_s7.parquet", index=False)

    xs = np.arange(len(layers))
    apply_style()
    fig, ax = L.panel_grid(1, ncols=1, panel=L.figsize_cm(9, 8))   # single-column figure
    ax = ax[0]
    # drop the palette colour's baked-in alpha so the line and markers are opaque (as in fig_2)
    A.line(ax, xs, table["median"].to_numpy(), table["ci_lo"].to_numpy(), table["ci_hi"].to_numpy(),
           color=mcolors.to_rgb(P2F_COLORS["embedding"]), label="embedding")
    N.ref_line(ax, 1.0, "y", color=P2F_COLORS["reference"], ls=N.DASH, label="genotype")

    ax.set_ylabel("$R^2_E$ / $R^2_G$")
    ax.set_xlabel("layer index")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i + 1) for i in xs], fontsize=5.5)
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)

    pooling = "position axis (per-channel embedding)" if args.axis == "pos" \
        else "channel axis (per-position embedding)"
    top = L.title(fig, f"EUR vs AFR ancestry prediction across Borzoi's depth, relative to genotype "
                       f"({'unwhitened' if args.rep == 'raw' else 'whitened'}, {pooling})",
                  "R^2_E / R^2_G: OLS on the top 10 PCs, no held-out split; line = median across genes, "
                  f"band = bootstrap 95% CI on the median; {len(geno)} genes with a genotype R^2 > 0")
    fig.tight_layout(rect=(0, 0, 1, top))
    L.save(fig, out_dir / "fig_s7", formats=("pdf", "png"), png_dpi=300)
    print(f"[done] {len(layers)} layers, {len(geno)} genes -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
