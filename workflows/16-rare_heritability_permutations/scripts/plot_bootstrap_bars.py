#!/usr/bin/env python
"""Bar plots of the mean and the median rare-variant h2 over genes, rare genotype vs embedding, with
bootstrap 95% confidence intervals (percentile). Genes are resampled with replacement; both arms use
the same resampled genes each time, so the difference gets its own bootstrap CI and p.

The significance bracket shows the bootstrap p of the difference (twice the smaller tail of the bootstrap
differences, never below 2 / n_boot; drawn as "p < 2 / n_boot" when no bootstrap difference crossed zero,
the summary's bootstrap_p_is_bound column). The phenotype-permutation p (permutation_test.tsv) is not drawn on the
figure; it is written to the summary table next to the bootstrap p.

Input: per_gene_real_estimates.parquet (summarize_permutation_test.py) and permutation_test.tsv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = ("#eb6834", "#2a78d6")


def boot_p(d, n_boot):
    """Two-sided bootstrap p of a difference: twice the smaller tail of the bootstrap differences, never
    below 2 / n_boot. Also returns True when no bootstrap difference fell on the far side of zero: the p is
    then only a bound (the true p is below 2 / n_boot), and a label should read "p < 2 / n_boot"."""
    tail = min((d <= 0).mean(), (d >= 0).mean())
    return min(1.0, 2 * max(tail, 1 / n_boot)), bool(tail == 0)


def p_label(p, is_bound, prefix="p"):
    """'p=0.034', or 'p < 0.002' when p is only the bound of the test."""
    return f"{prefix} < {p:.2g}" if is_bound else f"{prefix}={p:.2g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-estimates", required=True)
    ap.add_argument("--permutation-test", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="bootstrap_bars_rare_h2")
    ap.add_argument("--png", action="store_true", help="also save a PNG (for a quick look)")
    args = ap.parse_args()

    df = pd.read_parquet(args.real_estimates)[["h2_rare_genotype", "h2_rare_embedding"]].dropna()
    x = df.to_numpy(float)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(df), size=(args.n_boot, len(df)))
    res = x[idx]                                            # (n_boot, n_genes, 2)
    stat = {"mean": res.mean(axis=1), "median": np.median(res, axis=1)}
    obs = {"mean": x.mean(axis=0), "median": np.median(x, axis=0)}

    tests = pd.read_csv(args.permutation_test, sep="\t")
    perm_p = {k: float(tests[(tests.test == "phenotype_permutation") & (tests.statistic == f"{k} diff")]
                       .p_two_sided.iloc[0]) for k in stat}

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.4))
    for ax, k in zip(axes, ["mean", "median"]):
        b = stat[k]
        lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
        est = b.mean(axis=0)
        ax.bar([1, 2], est, width=0.55, color=COL, alpha=0.65, edgecolor=COL)
        ax.errorbar([1, 2], est, yerr=[est - lo, hi - est], fmt="none", ecolor="#0b0b0b", capsize=5, lw=1.2)
        d = b[:, 1] - b[:, 0]
        p_boot, p_is_bound = boot_p(d, args.n_boot)
        top = hi.max(); span = top - min(lo.min(), 0); by = top + 0.08 * span; tk = 0.02 * span
        ax.plot([1, 1, 2, 2], [by - tk, by, by, by - tk], color="#52514e", lw=1)
        ax.text(1.5, by + tk, p_label(p_boot, p_is_bound, "bootstrap p"), ha="center", va="bottom", fontsize=8)
        ax.set_ylim(min(lo.min(), 0) - 0.1 * span, by + 6 * tk)
        ax.axhline(0, color="#8b8a85", lw=0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["rare genotype\n(2 lowest bins)", "embedding"], fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylabel(f"{k} rare $h^2$ over genes (bootstrap, 95% CI)", fontsize=9)
        ax.set_title(k.capitalize(), fontsize=10, loc="left")
        for j, c in enumerate(df.columns):
            rows.append(dict(statistic=k, estimate=c, observed=obs[k][j], bootstrap_mean=est[j],
                             ci_lo=lo[j], ci_hi=hi[j]))
        rows.append(dict(statistic=k, estimate="embedding - genotype", observed=obs[k][1] - obs[k][0],
                         bootstrap_mean=d.mean(), ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                         bootstrap_p=p_boot, bootstrap_p_is_bound=p_is_bound,
                         permutation_p=perm_p[k]))
    fig.suptitle(f"{args.n_boot} bootstraps of {len(df)} genes", fontsize=9, x=0.02, ha="left", color="#52514e")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.pdf")
    if args.png:
        fig.savefig(f"{args.out_prefix}.png", dpi=200)
    t = pd.DataFrame(rows)
    t.to_csv(f"{args.out_prefix}_summary.tsv", sep="\t", index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
