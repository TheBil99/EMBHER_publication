#!/usr/bin/env python
"""Cross-gene scatterplots: does Borzoi's own head express what the trunk
demonstrably encodes?

One point per gene, from the per-gene summary parquets:
  (a) Borzoi head prediction R2 (squared Pearson)   vs EREML h2
  (b) Borzoi head prediction rho2 (squared Spearman) vs EREML h2

Both are shown, not just R2: R2 assumes a linear predicted-vs-measured relationship,
while rho2 is transformation-free (rank-based) -- a prediction that tracks the
phenotype monotonically but non-linearly would be unfairly penalized by R2 alone.
(Genotype-GRM heritability on these same genes now lives in workflow 10; compare
against it separately if needed.)

Both panels carry the y = x bisectrix on equal axes, so distance below the line reads
directly as the shortfall -- which is the result, not a plotting artefact.

Points carry error bars from the embedding estimator's own SE
(summarize_embedding_heritability.py's h2_se). Borzoi head R2/rho2 have none --
nothing is fitted there, so there is no sampling variance to report.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

S1, S2 = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a85"
EMB = "EREML"
PRED_R2, PRED_RHO2 = "Borzoi head prediction $R^2$", "Borzoi head prediction $\\rho^2$"


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


def panel(ax, x, y, genes, colr, xlab, ylab, title, identity, xerr=None, yerr=None):
    style(ax)
    if identity:
        lim = [min(0, np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y)) * 1.08]
        ax.plot(lim, lim, ls=(0, (4, 3)), color=MUTED, lw=0.9, zorder=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.annotate("equal", (lim[1] * 0.82, lim[1] * 0.86), fontsize=7.5,
                    color=MUTED, rotation=38)
    # errorbar (not plot) so a per-point SE renders as a cap-tipped bar underneath the
    # marker when available; genes with no SE (NaN) just get no bar -- errorbar treats
    # NaN err the same as omitting it for that point.
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color=colr, ms=6, alpha=0.75,
                mec="#fcfcfb", mew=0.8, ecolor=colr, elinewidth=0.9, capsize=2.5,
                capthick=0.9, zorder=3)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 3:
        rho, rp = stats.spearmanr(x[ok], y[ok])
        r, pp = stats.pearsonr(x[ok], y[ok])
        ax.annotate(f"Spearman $\\rho$ = {rho:+.3f} (p = {rp:.2g})\n"
                    f"Pearson r = {r:+.3f} (p = {pp:.2g})\n"
                    f"n = {int(ok.sum())} genes",
                    (0.03, 0.97), xycoords="axes fraction", va="top",
                    fontsize=8, color=INK2)
    # label the genes with the largest embedding h2
    for i in np.argsort(np.nan_to_num(x))[-4:]:
        ax.annotate(genes[i], (x[i], y[i]), textcoords="offset points",
                    xytext=(6, 3), fontsize=7, color=INK2)
    ax.set_xlabel(xlab, fontsize=9, color=INK2)
    ax.set_ylabel(ylab, fontsize=9, color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summaries", nargs="+", required=True,
                    help="per-gene *_heritability_summary.parquet files")
    ap.add_argument("--variant", default="ancestry corrected")
    ap.add_argument("--out-pdf", required=True)
    ap.add_argument("--out-table", required=True)
    ap.add_argument("--out-png", default=None, help="optional, for quick viewing")
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(p) for p in args.summaries], ignore_index=True)
    df = df[df.variant == args.variant]
    wide = df.pivot_table(index="experiment", columns="method", values="h2", aggfunc="first")
    for c in (EMB, PRED_R2, PRED_RHO2):
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide.dropna(subset=[EMB]).sort_values(EMB, ascending=False)

    # Per-gene SE, same pivot shape as `wide` (missing entirely, e.g. an older summary
    # parquet with no h2_se column, becomes all-NaN -- errorbar then just draws no bar).
    if "h2_se" in df.columns:
        wide_se = df.pivot_table(index="experiment", columns="method", values="h2_se", aggfunc="first")
    else:
        wide_se = pd.DataFrame(index=wide.index)
    for c in (EMB, PRED_R2, PRED_RHO2):
        if c not in wide_se.columns:
            wide_se[c] = np.nan
    wide_se = wide_se.reindex(wide.index)

    print(f"[info] {len(wide)} genes, variant '{args.variant}'")
    print(wide.head(12).round(4).to_string())

    genes = wide.index.to_numpy()
    x = wide[EMB].to_numpy(float)
    x_se = wide_se[EMB].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(5.6 * 2, 4.6))
    fig.patch.set_facecolor("#fcfcfb")
    panel(axes[0], x, wide[PRED_R2].to_numpy(float), genes, S1,
          "EREML $h^2$", "Borzoi head prediction $R^2$",
          "(a) Borzoi head prediction $R^2$ vs EREML", identity=True,
          xerr=x_se)   # R2/rho2 have no SE (nothing fitted, no sampling variance to report)
    panel(axes[1], x, wide[PRED_RHO2].to_numpy(float), genes, S2,
          "EREML $h^2$", "Borzoi head prediction $\\rho^2$",
          "(b) Borzoi head prediction $\\rho^2$ vs EREML", identity=True,
          xerr=x_se)

    wide.to_parquet(args.out_table)

    fig.tight_layout()
    fig.savefig(args.out_pdf, facecolor=fig.get_facecolor())
    if args.out_png:
        fig.savefig(args.out_png, dpi=150, facecolor=fig.get_facecolor())
    print(f"[done] -> {args.out_pdf}, {args.out_table}")


if __name__ == "__main__":
    main()
