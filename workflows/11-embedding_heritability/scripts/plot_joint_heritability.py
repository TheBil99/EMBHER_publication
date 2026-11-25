#!/usr/bin/env python
"""Summary page for embedding_heritability.nf's joint_heritability process -- the
region's genotype GRM (default MAC/MAF 4-bin cohort split) and this gene's embedding
kernel fit as ONE joint REML (exact_reml_heritability.py, --bfile ... --grm-npz ...),
so genotype and embedding are reported as shares of one total h2 rather than two
independently-fit h2 values that would double-count whatever variance the two kernels
tag in common. Same approach as workflow 09's exact_from_grm_joint, on real GTEx
genotypes instead of a simulated panel.

One panel, one gene per point, against the joint fit's own y=x line: genotype-from-joint
(h2_genotype) vs embedding-from-joint (h2_k{last}) -- how the joint fit itself splits
one gene's total h2 between the two kernels. No comparison against pure genotype
heritability here -- that is workflow 10's own result; cross-workflow comparison is
workflow 17's job, not this one's.

A gene is dropped only if the joint fit neither converged nor ended at_boundary
(at_boundary keeps the total h2 but not its per-component split/SE, same convention as
workflow 10's plot_genotype_heritability.py).
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a85"
POINT_COLOR = "#2a78d6"
BOUNDARY_COLOR = "#eb6834"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color="#e6e5e1", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)


def load(paths):
    rows = []
    for p in paths:
        d = pd.read_parquet(p)
        rows.append(d[d["phenotype"] == "pheno_expression"])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def embedding_component_col(joint_df):
    comp_idx = [int(c[4:]) for c in joint_df.columns
                if c.startswith("h2_k") and c[4:].isdigit()]
    if not comp_idx:
        raise ValueError("no h2_k* component columns in the joint-fit parquet -- expected "
                          "one per genotype bin plus one for the embedding kernel")
    # "the highest-index h2_k* is the embedding" only holds for a SINGLE embedding
    # kernel (joint_heritability's own exact_reml_joint_mac4bin_covar: h2_k1..h2_k4
    # genotype bins, h2_k5 the one embedding). joint_heritability_dual_emb's
    # exact_reml_joint_mac4bin_2emb_covar has TWO (h2_k5 rare, h2_k6 common) -- silently
    # returning just h2_k6 here would drop h2_k5 with no warning. Fail loudly instead of
    # guessing wrong.
    method = joint_df["method"].iloc[0] if "method" in joint_df.columns and len(joint_df) else ""
    if "2emb" in method:
        raise ValueError(
            f"'{method}' joint fits have TWO embedding kernels (h2_k5, h2_k6), not one -- "
            "embedding_component_col() only knows how to pick a single embedding column. "
            "Use plot_joint_heritability_dual_emb.py instead of this single-embedding helper."
        )
    return f"h2_k{max(comp_idx)}"


def scatter_pair(ax, df, col_x, col_y, label_x, label_y, title_suffix=""):
    d = df.set_index("experiment")
    genes = sorted(d.index)

    def flag(g, col):
        return col in d.columns and d.loc[g, col] == True  # noqa: E712

    # A fit can report converged with every h2 NaN (zero variance components, e.g. a
    # degenerate phenotype) -- it has no point to plot and would make the axis limits NaN.
    def finite(g):
        return bool(np.isfinite(d.loc[g, [col_x, col_y]].to_numpy(dtype=float)).all())

    usable = {g: (flag(g, "converged") or flag(g, "at_boundary")) and finite(g) for g in genes}
    dropped = [g for g in genes if not usable[g]]
    genes = [g for g in genes if usable[g]]
    if dropped:
        print(f"[warn] {label_x} vs {label_y}: dropping {len(dropped)} gene(s) with a "
              f"non-converged fit: {dropped}", flush=True)
    boundary = np.array([flag(g, "at_boundary") for g in genes], dtype=bool)

    xs = d.loc[genes, col_x].to_numpy(dtype=float)
    ys = d.loc[genes, col_y].to_numpy(dtype=float)

    style(ax)
    ax.scatter(xs[~boundary], ys[~boundary], s=45, color=POINT_COLOR, alpha=0.85,
               edgecolors="#fcfcfb", linewidths=1.0, zorder=3,
               label=f"converged (n={int((~boundary).sum())})")
    if boundary.any():
        ax.scatter(xs[boundary], ys[boundary], s=45, color=BOUNDARY_COLOR, alpha=0.85,
                   edgecolors="#fcfcfb", linewidths=1.0, zorder=4,
                   label=f"at_boundary (n={int(boundary.sum())})")

    # Data-driven per panel, not anchored to a fixed [0,1] -- forcing every panel to
    # span at least 0-1 (the old behaviour) made panels with small values (e.g. a
    # near-zero genotype bin) look effectively identical in scale to one spanning
    # 0-1, squeezing their real spread into a tiny corner. Each panel now autoscales
    # to its own xs/ys, so panels are independent even though they share one figure.
    lo = min(xs.min(), ys.min()) - 0.05
    hi = max(xs.max(), ys.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1, label="y=x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.axvline(0, color=MUTED, lw=0.5)

    title = f"{label_x} vs {label_y}{title_suffix}"
    if len(genes) >= 2:
        r, p = stats.pearsonr(xs, ys)
        title += f"\nPearson r={r:.3f} (p={p:.2g}, n={len(genes)})"
    else:
        title += f"\n(n={len(genes)})"
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(f"{label_x} $h^2$", fontsize=9, color=INK2)
    ax.set_ylabel(f"{label_y} $h^2$", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", nargs="+", required=True,
                     help="per-gene *_joint_heritability.parquet")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    joint = load(args.joint)
    geno_col = "h2_genotype"
    emb_col = embedding_component_col(joint)

    joint.to_parquet(f"{args.out_prefix}.parquet", index=False)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    fig.patch.set_facecolor("#fcfcfb")

    scatter_pair(ax, joint, geno_col, emb_col,
                 "genotype-from-joint", "embedding-from-joint",
                 title_suffix=" (one joint fit's own split)")

    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[done] {joint.experiment.nunique()} genes -> {args.out_prefix}.pdf", flush=True)


if __name__ == "__main__":
    main()
