#!/usr/bin/env python
"""PC1 vs PC2 of genotypes and of raw Borzoi embeddings for one gene's window, by ancestry.

All donors; columns centred only (no scaling), so the PCA is the best 2-D view of the
same Euclidean distances intra_inter_distances.py uses. Marginal densities of PC1 (top)
and PC2 (right) for EUR and the other group. Genotypes: 0/1/2 non-reference counts
(genotype_features.py); embeddings: haplotypes summed, not whitened
(combine_haplotypes.py). Colours: EUR and the comparison's other group from its groups
file; every remaining donor is 'Other'.

Outputs (prefix <gene>_pca_ancestry): PC scores .tsv and the plot (.png/.pdf).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

COLORS = ["#2a78d6", "#eb6834"]   # EUR, other group
OTHER_COLOR = "#a3a29c"


def pca2(X):
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :2] * S[:2], S[:2] ** 2 / (S ** 2).sum()


def marginal(ax, values, color, vertical=False):
    grid = np.linspace(values.min(), values.max(), 200)
    dens = gaussian_kde(values)(grid)
    if vertical:
        ax.fill_betweenx(grid, 0, dens, color=color, alpha=0.25, linewidth=0)
        ax.plot(dens, grid, color=color, lw=1.5)
    else:
        ax.fill_between(grid, 0, dens, color=color, alpha=0.25, linewidth=0)
        ax.plot(grid, dens, color=color, lw=1.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene", required=True, help="label used in titles and file names")
    ap.add_argument("--groups", required=True, help="IID / GROUP TSV (make_ancestry_groups.py)")
    ap.add_argument("--genotype", required=True, help="donor x variant parquet (genotype_features.py)")
    ap.add_argument("--embedding", required=True, help="donor x feature parquet (combine_haplotypes.py)")
    ap.add_argument("--window-bp", type=int, required=True, help="window size, for the title")
    a = ap.parse_args()

    groups = pd.read_csv(a.groups, sep="\t").set_index("IID").GROUP
    order = ["EUR", groups[groups != "EUR"].iloc[0]]
    reps = {"Genotype": a.genotype, "Borzoi embedding": a.embedding}

    fig = plt.figure(figsize=(11, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, width_ratios=[4, 1, 0.35, 4, 1], height_ratios=[1, 4])
    rows, main_axes = [], []
    for col, (name, path) in zip((0, 3), reps.items()):
        ax = fig.add_subplot(gs[1, col])
        top = fig.add_subplot(gs[0, col], sharex=ax)
        right = fig.add_subplot(gs[1, col + 1], sharey=ax)
        main_axes.append(ax)
        df = pd.read_parquet(path)
        ids = df.index.astype(str)
        lab = groups.reindex(ids).fillna("Other").to_numpy()
        pcs, var = pca2(df.to_numpy(np.float64))
        rows.append(pd.DataFrame({"donor": ids, "group": lab, "representation": name,
                                  "PC1": pcs[:, 0], "PC2": pcs[:, 1]}))
        # larger groups first, so the smaller ones stay visible on top
        for g, c in (("Other", OTHER_COLOR), (order[0], COLORS[0]), (order[1], COLORS[1])):
            m = lab == g
            ax.scatter(pcs[m, 0], pcs[m, 1], s=20, color=c, alpha=0.7, linewidths=0,
                       label=f"{g} (n = {m.sum()})", rasterized=True)
        for g, c in zip(order, COLORS):
            m = lab == g
            marginal(top, pcs[m, 0], c)
            marginal(right, pcs[m, 1], c, vertical=True)
        ax.set_xlabel(f"PC1 ({100 * var[0]:.1f}% of variance)")
        ax.set_ylabel(f"PC2 ({100 * var[1]:.1f}% of variance)")
        top.set_title(name, fontsize=10, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
        for m_ax in (top, right):
            m_ax.axis("off")
    handles, labels = main_axes[0].get_legend_handles_labels()
    keep = [labels.index(l) for l in labels if not l.startswith("Other")] + \
           [i for i, l in enumerate(labels) if l.startswith("Other")]
    main_axes[1].legend([handles[i] for i in keep], [labels[i] for i in keep], frameon=False,
                        fontsize=8, markerscale=1.2, loc="best")
    fig.suptitle(f"{a.gene}: PCA of the {a.window_bp / 1000:.0f} kb window, all donors, "
                 f"coloured by ancestry", fontsize=11, x=0.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.gene}_pca_ancestry.{ext}", dpi=200)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(f"{a.gene}_pca_ancestry.tsv", sep="\t", index=False)
    print(res.groupby(["representation", "group"])[["PC1", "PC2"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
