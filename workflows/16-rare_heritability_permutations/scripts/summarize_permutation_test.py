#!/usr/bin/env python
"""Permutation test of the rare-variant heritability difference, embedding vs genotype.

Input: the per-gene parquets from fit_rare_h2_permutations.py (row perm 0 = real phenotype, rows
1..n_perm = donor-label permutations; the same permutation feeds both arms of a gene).

Statistic: across genes, the mean and the median of h2_diff = h2_rare_embedding - h2_rare_genotype.
Null: permutation k's statistic is the mean/median over genes of each gene's own permutation-k
difference (genes are shuffled independently; k only indexes the draw). p = (1 + #{null at least as
extreme}) / (n_perm + 1): two-sided on |statistic|, and one-sided for "embedding > genotype". With no
permutation as extreme, figures show the bound "p < 1 / n_perm".

A gene enters only if its real fit and every permutation fit are finite, so every permutation
statistic is taken over exactly the genes of the observed statistic.

Secondary, cheap and on the real estimates only: a paired sign-flip test (each gene's difference
flipped with probability 1/2) and a paired Wilcoxon signed-rank test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

S1, S2 = "#2a78d6", "#eb6834"
VIOLIN_YLIM = 0.1          # y-axis half-range of the observed-values violin (display only)
VIOLIN_POINTS = 4000       # grid points of the violins' density curve over the whole data range (smooth inside the window)
INK2, MUTED = "#52514e", "#8b8a85"


def p_values(obs, null):
    n = len(null)
    two = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-15)) / (n + 1)
    greater = (1 + np.sum(null >= obs - 1e-15)) / (n + 1)
    return two, greater


def perm_p_label(p, n_perm, prefix="p", fmt=".2g"):
    """Label of a permutation p = (1 + b) / (n_perm + 1). With b = 0 (no permutation as extreme as the
    observed) the p is only a bound and reads 'p < 1 / n_perm'."""
    return f"{prefix} < {1 / n_perm:.2g}" if p * (n_perm + 1) < 1.5 else f"{prefix}={p:{fmt}}"


def draw_pdf(out, n_genes, n_perm, null, tests, R0, E0, png=False):
    """permutation_test.pdf: page 1 the null histograms of the mean / median difference with the observed value,
    page 2 the violins of the observed rare genotype and rare embedding h2 with the permutation p-values.
    R0 / E0: the real-phenotype h2 per gene (rare genotype / rare embedding)."""
    with PdfPages(out / "permutation_test.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, name in zip(axes, ["mean", "median"]):
            r = tests[(tests.test == "phenotype_permutation") & (tests.statistic == f"{name} diff")].iloc[0]
            ax.hist(null[f"{name}_diff"], bins=40, color=MUTED, alpha=0.7)
            ax.axvline(r.observed, color=S2, lw=2, label=f"observed {r.observed:.4f}")
            ax.set_xlabel(f"{name} over genes of (embedding − genotype) rare $h^2$", fontsize=9)
            ax.set_ylabel("permutations", fontsize=9)
            ax.set_title(f"{name}: {perm_p_label(r.p_two_sided, n_perm, 'permutation p', '.3g')} (two-sided), "
                         f"{perm_p_label(r.p_embedding_greater, n_perm, 'p', '.3g')} (embedding>genotype)",
                         fontsize=8, loc="left")
            ax.legend(frameon=False, fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.suptitle(f"{n_genes} genes, {n_perm} phenotype permutations", fontsize=9, x=0.02, ha="left",
                     color=INK2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(3.2, 4.2))
        parts = ax.violinplot([R0, E0], showmeans=False, showmedians=True, showextrema=False, widths=0.7,
                              points=VIOLIN_POINTS)
        for body, c in zip(parts["bodies"], (S2, S1)):
            body.set_facecolor(c); body.set_edgecolor(c); body.set_alpha(0.6)
        parts["cmedians"].set_color(INK2)
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["rare genotype\n(2 lowest bins)", "embedding"], fontsize=8)
        ax.set_ylabel("rare-variant $h^2$ (HE)", fontsize=9)
        ax.set_ylim(-VIOLIN_YLIM, VIOLIN_YLIM)             # display range only; the tests use every gene
        # bracket between the two violins with the permutation p of the mean and of the median difference
        pm = tests[(tests.test == "phenotype_permutation")].set_index("statistic")
        by, tk = 0.80 * VIOLIN_YLIM, 0.03 * VIOLIN_YLIM
        ax.plot([1, 1, 2, 2], [by - tk, by, by, by - tk], color=INK2, lw=1)
        ax.text(1.5, by + tk,
                f"mean diff: {perm_p_label(pm.loc['mean diff', 'p_two_sided'], n_perm, 'permutation p')}\n"
                f"median diff: {perm_p_label(pm.loc['median diff', 'p_two_sided'], n_perm, 'permutation p')}",
                ha="center", va="bottom", fontsize=7, color="#0b0b0b")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        pdf.savefig(fig)
        if png:
            fig.savefig(out / "permutation_test_violin.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-gene", nargs="+", default=None, help="fit_rare_h2_permutations.py parquets")
    ap.add_argument("--replot-from", default=None,
                    help="redraw permutation_test.pdf from the tables already in this folder (per_gene_real_estimates.parquet, "
                         "null_distribution.parquet, permutation_test.tsv) instead of recomputing from --per-gene")
    ap.add_argument("--seed", type=int, default=0, help="for the sign-flip test")
    ap.add_argument("--config-tsv", default=None,
                    help="optional (gene_id, gene_name / experiment columns): adds a gene_name column to the per-gene table")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--png", action="store_true", help="also save the violin page as a PNG (for a quick look)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.replot_from:
        src = Path(args.replot_from)
        real = pd.read_parquet(src / "per_gene_real_estimates.parquet")
        null = pd.read_parquet(src / "null_distribution.parquet")
        tests = pd.read_csv(src / "permutation_test.tsv", sep="\t")
        n_genes = int(tests[tests.test == "phenotype_permutation"].n_genes.iloc[0])
        draw_pdf(out, n_genes, len(null), null, tests, real.h2_rare_genotype.to_numpy(),
                 real.h2_rare_embedding.to_numpy(), args.png)
        print(f"[done] replotted {n_genes} genes -> {out}", flush=True)
        return
    if not args.per_gene:
        ap.error("--per-gene is required unless --replot-from is given")

    slim_cols = ["experiment", "perm", "h2_rare_genotype", "h2_rare_embedding", "h2_diff", "converged"]
    real_rows, slim = [], []
    for p in args.per_gene:                       # keep every file's full real row, only slim columns otherwise
        g = pd.read_parquet(p)
        real_rows.append(g.iloc[[0]])
        slim.append(g[slim_cols])
    df = pd.concat(slim, ignore_index=True)
    n_perm = int(df.perm.max())
    df["finite"] = df.converged & np.isfinite(df.h2_diff)
    agg = df.groupby("experiment").agg(n=("perm", "size"), ok=("finite", "all"))
    dropped = agg.index[(agg.n != n_perm + 1) | (~agg.ok)].tolist()
    if dropped:
        print(f"[warn] dropping {len(dropped)} gene(s) with a non-finite or incomplete fit: {dropped[:10]}", flush=True)
    df = df[~df.experiment.isin(dropped)]
    genes = sorted(df.experiment.unique())
    real = pd.concat(real_rows, ignore_index=True).set_index("experiment").loc[genes]
    if args.config_tsv:
        cfg = pd.read_csv(args.config_tsv, sep="\t", dtype=str)
        ens = cfg.gene_id.str.split(".").str[0]
        names = cfg.gene_name if "gene_name" in cfg.columns else cfg.get("experiment", ens)
        by_key = dict(zip(ens, names))
        if "experiment" in cfg.columns:                     # configs keyed by the gene symbol
            by_key.update(dict(zip(cfg.experiment, names)))
        real.insert(0, "gene_name", [by_key.get(g, g) for g in real.index])
    real.to_parquet(out / "per_gene_real_estimates.parquet")

    # genes x (n_perm + 1) matrices, column 0 = real phenotype
    piv = lambda col: df.pivot(index="experiment", columns="perm", values=col).loc[genes].to_numpy()
    D, R, E = piv("h2_diff"), piv("h2_rare_genotype"), piv("h2_rare_embedding")
    stat_fn = {"mean": np.mean, "median": np.median}
    null = pd.DataFrame({"perm": np.arange(1, n_perm + 1)})
    rows = []
    for name, f in stat_fn.items():
        s_diff = f(D, axis=0)
        obs, nul = s_diff[0], s_diff[1:]
        null[f"{name}_diff"] = nul
        null[f"{name}_h2_rare_genotype"] = f(R, axis=0)[1:]
        null[f"{name}_h2_rare_embedding"] = f(E, axis=0)[1:]
        two, greater = p_values(obs, nul)
        rows.append(dict(test="phenotype_permutation", statistic=f"{name} diff", observed=obs,
                         null_mean=nul.mean(), null_sd=nul.std(ddof=1), p_two_sided=two,
                         p_embedding_greater=greater, n_genes=len(genes), n_perm=n_perm))
        rows.append(dict(test="observed_estimates", statistic=f"{name} h2_rare_genotype",
                         observed=f(R[:, 0])))
        rows.append(dict(test="observed_estimates", statistic=f"{name} h2_rare_embedding",
                         observed=f(E[:, 0])))
    # secondary, real estimates only
    d0 = D[:, 0]
    rng = np.random.default_rng(args.seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d0)))
    flips = signs * d0
    for name, f in stat_fn.items():
        obs, nul = f(d0), f(flips, axis=1)
        two, greater = p_values(obs, nul)
        rows.append(dict(test="sign_flip", statistic=f"{name} diff", observed=obs,
                         null_mean=nul.mean(), null_sd=nul.std(ddof=1), p_two_sided=two,
                         p_embedding_greater=greater, n_genes=len(genes), n_perm=n_perm))
    w = stats.wilcoxon(d0)
    rows.append(dict(test="wilcoxon_signed_rank", statistic="diff", observed=float(w.statistic),
                     p_two_sided=float(w.pvalue), n_genes=len(genes)))
    tests = pd.DataFrame(rows)
    tests.to_csv(out / "permutation_test.tsv", sep="\t", index=False)
    null.to_parquet(out / "null_distribution.parquet", index=False)
    print(tests.round(5).to_string(index=False), flush=True)

    draw_pdf(out, len(genes), n_perm, null, tests, R[:, 0], E[:, 0], args.png)
    print(f"[done] {len(genes)} genes -> {out}", flush=True)


if __name__ == "__main__":
    main()
