#!/usr/bin/env python
"""fig_s1_s4: error distribution of heritability against the true h2 in the two
simulation studies, and the fraction of converged fits, as a 4-page PDF (embher_plotting
archetypes): one page per (simulation, estimator). Each page has three panel groups:
a = error, b = relative error (h2 - true_h2) / true_h2, with true h2 = 0 dropped and y
zoomed to [-1, 1], c = fraction of converged fits per true h2, on the same panel grid.

  page 1  workflow 08 (genotype GRMs), HE: one panel per (p_causal, binning), genes
          pooled. Causal-MAF band 0-0.5 (the replicate files without a `_cmaf` tag).
          a/b: converged fits only -- as in 06's summary_cmaf0_0.5.pdf page 2;
          c: all fits, before that filter.
  page 2  as page 1, exact REML.
  page 3  workflow 09 (embedding kernels), HE: one panel per (kernel, phenotype source),
          shared y, all replicates -- as in 07's calibration.pdf page 2.
  page 4  as page 3, exact REML.

Each violin is a KDE, the bar the IQR, the dot the median. Reads the two workflows'
already-published per-replicate parquets; nothing is recomputed.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection

PC_RE = re.compile(r"pc_([0-9.eE+-]+)_rep")


def method_order(m):
    mm = re.search(r"(\d+)$", m)
    return (0 if m == "single_grm" else 1, int(mm.group(1)) if mm else 0)


def pheno_order(s):
    if s == "dense":
        return (0, 0.0)
    if s == "sparse":
        return (1, 0.0)
    if s == "randompc":
        return (2, 0.0)
    if s == "genotype":
        return (3, 0.0)
    try:
        return (2, float(s[2:]))
    except ValueError:
        return (4, 0.0)


def kernel_order(k):
    if k == "raw":
        return (0, 0.0)
    num = k.split("_k")[-1] if "_k" in k else k.split("frac")[-1] if "frac" in k else "1"
    return (1, float(num))


def load_genotype(h2_dir: str, converged_only: bool = True) -> pd.DataFrame:
    """Workflow 08 replicate fits of the 0-0.5 causal-MAF band (no `_cmaf` in the name),
    converged only unless `converged_only=False`, with p_causal parsed from the experiment
    name."""
    files = [f for f in sorted(glob.glob(f"{h2_dir}/*.parquet")) if "_cmaf" not in Path(f).name]
    if not files:
        raise SystemExit(f"[error] no band-0-0.5 parquets under {h2_dir}")
    df = pd.concat([pd.read_parquet(f, columns=["experiment", "method", "h2", "true_h2", "converged"])
                    for f in files], ignore_index=True)
    df = df[df["converged"]].copy() if converged_only else df.copy()
    df["p_causal"] = df["experiment"].str.extract(PC_RE)[0].astype(float)
    return df


def violin_panel(ax, df, h2_values, color, A, N, label, relative=False):
    """Error (h2 - true_h2) violins, one per true h2, on `ax`."""
    err = [(df.loc[df["true_h2"] == h, "h2"].dropna().to_numpy() - h) / (h if relative else 1.0)
           for h in h2_values]
    A.violin(ax, err, positions=np.arange(len(h2_values)), width=0.85, colors=color,
             labels=[f"{h:g}" for h in h2_values])
    for coll in [c for c in ax.collections if isinstance(c, LineCollection)]:
        coll.remove()                        # drop the IQR bars (ax.vlines)
    for coll in [c for c in ax.collections if isinstance(c, PolyCollection)]:
        coll.set_alpha(1.0)                  # opaque violin bodies (library default is 0.8)
    from embher_plotting.theme import P2F_COLORS   # sys.path is set by main() before this runs
    N.ref_line(ax, 0.0, "y", ls=N.DASH, color=P2F_COLORS["reference"])
    for ln in ax.lines:                      # shrink the median dot (library default is larger)
        if ln.get_marker() == "o":
            ln.set_markersize(0.9)
    if label:
        N.stat_text(ax, label, loc="upper left")


def outer_labels(axes, ncols, relative=False):
    """Axis labels on the bottom row (x) and first column (y) only; axes are shared."""
    for k, ax in enumerate(axes):
        if k >= len(axes) - ncols:
            ax.set_xlabel("$h^2_{true}$")
        if k % ncols == 0:
            ax.set_ylabel("$(h^2 - h^2_{true}) / h^2_{true}$" if relative else "$h^2 - h^2_{true}$")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genotype-h2-dir", required=True,
                    help="results/08-genotype_heritability_simulation_study/"
                         "10_random_genes_he/h2")
    ap.add_argument("--embedding-replicates-dir", required=True,
                    help="results/09-embedding_heritability_simulation_study/"
                         "10_random_genes_he/replicates")
    ap.add_argument("--genotype-reml-h2-dir", required=True,
                    help="results/08-genotype_heritability_simulation_study/"
                         "10_random_genes_reml/h2")
    ap.add_argument("--embedding-reml-replicates-dir", required=True,
                    help="results/09-embedding_heritability_simulation_study/"
                         "10_random_genes_reml/replicates")
    ap.add_argument("--embher-plotting-src", required=True,
                    help="embher_analysis's src/ directory, put on sys.path by explicit path")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--embedding-ylim", type=float, default=None,
                    help="symmetric y zoom for pages 2 and 4 (default: none, shared autoscale)")
    args = ap.parse_args()

    sys.path.insert(0, args.embher_plotting_src)
    from embher_plotting import apply_style, archetypes as A, annotations as N, layout as L
    from embher_plotting.theme import P2F_COLORS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    def load_embedding(replicates_dir):
        files = sorted(glob.glob(f"{replicates_dir}/*.parquet"))
        if not files:
            raise SystemExit(f"[error] no parquets under {replicates_dir}")
        return pd.concat([pd.read_parquet(f, columns=["h2", "true_h2", "kernel", "pheno_source",
                                                      "converged"])
                          for f in files], ignore_index=True)

    def h2_list(df, relative):
        return [h for h in sorted(df["true_h2"].unique()) if not (relative and h == 0)]

    def draw_genotype(sf, geno, relative):
        methods = sorted(geno["method"].unique(), key=method_order)
        pcs = sorted(geno["p_causal"].unique())
        h2_values = h2_list(geno, relative)
        axes = list(sf.subplots(len(pcs), len(methods), sharex=True, sharey=True).ravel())
        for k, ax in enumerate(axes):
            pc, method = pcs[k // len(methods)], methods[k % len(methods)]
            sub = geno[(geno["p_causal"] == pc) & (geno["method"] == method)]
            violin_panel(ax, sub, h2_values, P2F_COLORS["genotype"], A, N, None, relative)
            if k < len(methods):                 # column header: genotype-kernel design
                ax.text(0.5, 1.04, method.rsplit("_", 1)[-1] + " bins", transform=ax.transAxes,
                        ha="center", va="bottom", fontweight="bold")
            if k % len(methods) == 0:            # row label: causal fraction, left of the y label
                ax.text(-0.35, 0.5, f"p_causal = {pc:g}", transform=ax.transAxes, ha="center",
                        va="center", rotation=90, fontweight="bold")
        if relative:
            axes[0].set_ylim(-1, 1)          # shared y: zoom on the relative bias
        outer_labels(axes, len(methods), relative)
        return axes

    def draw_embedding(sf, emb, relative):
        sources = sorted(emb["pheno_source"].unique(), key=pheno_order)
        kernels = sorted(emb["kernel"].unique(), key=kernel_order)
        h2_values = h2_list(emb, relative)
        axes = list(sf.subplots(len(kernels), len(sources), sharex=True, sharey=True).ravel())
        for k, ax in enumerate(axes):
            kern, src = kernels[k // len(sources)], sources[k % len(sources)]
            sub = emb[(emb["kernel"] == kern) & (emb["pheno_source"] == src)]
            violin_panel(ax, sub, h2_values, P2F_COLORS["embedding"], A, N, None, relative)
            ax.tick_params(axis="x", labelrotation=90)
            if k < len(sources):                 # column header: phenotype geometry
                ax.text(0.5, 1.04, src.replace("randompc", "random PC"), transform=ax.transAxes,
                        ha="center", va="bottom", fontweight="bold")
            if k % len(sources) == 0:            # row label: kernel, left of the y label
                name = kern.replace("white_frac", "whitening ")
                ax.text(-0.75, 0.5, name, transform=ax.transAxes, ha="center", va="center",
                        rotation=90, fontweight="bold")
        if args.embedding_ylim is not None and not relative:
            axes[0].set_ylim(-args.embedding_ylim, args.embedding_ylim)
        if relative:
            axes[0].set_ylim(-1, 1)          # shared y: zoom on the relative bias
        outer_labels(axes, len(sources), relative)
        return axes

    def draw_convergence(sf, df, row_key, col_key, row_order, col_order, row_name, col_name,
                         color, row_label_x, compact=False):
        """Fraction of converged fits per true h2 (line, one point per true h2 at the violins'
        positions), one panel per (row, column) cell of the same grid as the matching error
        panels. `compact` (small panels): x tick labels turned 90 degrees, y ticks 0/0.5/1, one
        y label on the middle row, row labels on two lines."""
        rows = sorted(df[row_key].unique(), key=row_order)
        cols = sorted(df[col_key].unique(), key=col_order)
        h2_values = h2_list(df, False)
        axes = list(sf.subplots(len(rows), len(cols), sharex=True, sharey=True).ravel())
        for k, ax in enumerate(axes):
            r, c = rows[k // len(cols)], cols[k % len(cols)]
            sub = df[(df[row_key] == r) & (df[col_key] == c)]
            frac = [sub.loc[sub["true_h2"] == h, "converged"].mean() for h in h2_values]
            A.line(ax, np.arange(len(h2_values)), frac, color=color)
            N.ref_line(ax, 1.0, "y", ls=N.DASH, color=P2F_COLORS["reference"])
            ax.set_xticks(np.arange(len(h2_values)))
            ax.set_xticklabels([f"{h:g}" for h in h2_values])
            if compact:
                ax.tick_params(axis="x", labelrotation=90)
            if k < len(cols):
                ax.text(0.5, 1.04, col_name(c), transform=ax.transAxes,
                        ha="center", va="bottom", fontweight="bold")
            if k % len(cols) == 0:
                name = row_name(r).replace(" ", "\n", 1) if compact else row_name(r)
                ax.text(row_label_x, 0.5, name, transform=ax.transAxes, ha="center",
                        va="center", rotation=90, fontweight="bold")
        axes[0].set_xlim(-0.5, len(h2_values) - 0.5)
        axes[0].set_ylim(0, 1.08)            # room for the markers at 1
        axes[0].set_yticks([0, 0.5, 1.0] if compact else [0, 0.25, 0.5, 0.75, 1.0])
        for k, ax in enumerate(axes):
            if k >= len(axes) - len(cols):
                ax.set_xlabel("$h^2_{true}$")
            if not compact and k % len(cols) == 0:
                ax.set_ylabel("fraction\nconverged")
        if compact:                          # one label spanning the rows, on the middle one
            axes[(len(rows) // 2) * len(cols)].set_ylabel("fraction converged")
        return axes

    def page(draw, df, conv_df, grid, width_cm, heights_cm, letter_x, title, subtitle):
        """One page: error (a), relative error (b) and fraction converged (c), stacked, panel
        letters in the house style (embher_plotting.annotations.panel_label, as in fig_3)."""
        fig = plt.figure(figsize=L.figsize_cm(width_cm, sum(heights_cm)), layout="constrained")
        top, mid, bottom = fig.subfigures(3, 1, height_ratios=heights_cm)
        N.panel_label(draw(top, df, False)[0], "a", x=letter_x, y=1.25)
        N.panel_label(draw(mid, df, True)[0], "b", x=letter_x, y=1.25)
        N.panel_label(draw_convergence(bottom, conv_df, **grid)[0], "c", x=letter_x, y=1.25)
        L.title(fig, title, subtitle)
        return fig

    def log_convergence(df, keys, label):
        print(f"[converged] {label}: {df['converged'].mean():.3f} of {len(df)} fits", flush=True)
        print(df.groupby(keys + ["true_h2"])["converged"].mean().unstack().round(3).to_string(),
              flush=True)

    geno_grid = dict(row_key="p_causal", col_key="method", row_order=float, col_order=method_order,
                     row_name=lambda pc: f"p_causal = {pc:g}",
                     col_name=lambda m: m.rsplit("_", 1)[-1] + " bins",
                     color=P2F_COLORS["genotype"], row_label_x=-0.35)
    emb_grid = dict(row_key="kernel", col_key="pheno_source", row_order=kernel_order,
                    col_order=pheno_order,
                    row_name=lambda kern: kern.replace("white_frac", "whitening "),
                    col_name=lambda src: src.replace("randompc", "random PC"),
                    color=P2F_COLORS["embedding"], row_label_x=-0.75, compact=True)

    runs = [("HE regression", args.genotype_h2_dir, args.embedding_replicates_dir),
            ("Exact REML", args.genotype_reml_h2_dir, args.embedding_reml_replicates_dir)]
    figs = []
    for est, geno_dir, _ in runs:            # pages 1-2: genotype
        geno = load_genotype(geno_dir, converged_only=False)
        log_convergence(geno, ["method", "p_causal"], f"{est} genotype")
        figs.append(page(draw_genotype, geno[geno["converged"]], geno, geno_grid,
                         18.0, [7.6, 7.6, 6.0], -0.6,
                         "Genotype heritability: error, relative error and convergence by true h2",
                         f"{est}, causal MAF 0-0.5, 10 genes pooled; a-b converged fits, "
                         "b drops true h2 = 0; c all fits"))
    for est, _, emb_dir in runs:             # pages 3-4: embedding
        emb = load_embedding(emb_dir)
        log_convergence(emb, ["kernel", "pheno_source"], f"{est} embedding")
        figs.append(page(draw_embedding, emb, emb, emb_grid, 12.0, [8.0, 8.0, 5.0], -1.4,
                         "Embedding heritability: error, relative error and convergence by true h2",
                         f"{est}, all replicates, 10 genes pooled; b drops true h2 = 0"))

    with L.PdfBook(out_dir / "fig_s1_s4.pdf", info_page=False, png_dpi=300) as book:
        for fig in figs:
            book.add(fig)
    print(f"[done] {out_dir / 'fig_s1_s4.pdf'}", flush=True)


if __name__ == "__main__":
    main()
