#!/usr/bin/env python
"""Summarize GTEx expression heritability: embedding GRM (EREML) vs a permutation null,
alongside Borzoi head prediction R2.

There is no true h2 on real data, so each estimate is reported against the permutation
null built from the pheno_perm_* columns. At n ~ 800 the null is what makes an estimate
readable -- it gives the sampling spread of h2 directly, for this gene, kernel and
sample size. (Genotype-GRM heritability on these genes is computed by workflow 10.)

Borzoi's prediction is reported as BOTH r2 (squared Pearson) and rho2 (squared
Spearman) against the same phenotype. r2 assumes a linear relationship; rho2 is
transformation-free (rank-based), so a prediction that tracks the phenotype
monotonically but non-linearly isn't penalized by r2 alone.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a85"
METHODS = [("h2", "EREML", S1),
           ("r2", "Borzoi head prediction $R^2$", S2),
           ("rho2", "Borzoi head prediction $\\rho^2$", S3)]
# Which load() source each column comes from -- the embedding heritability parquet
# carries its own "converged" column, so "source" (set by load()) disambiguates a
# method's row unconditionally rather than by column-notna alone.
METHOD_SOURCE = {"h2": "embedding"}
# Each estimator's own SE column, carried through into the summary as "h2_se" so
# plot_across_genes.py can draw error bars. Borzoi head R2 has no SE: nothing is
# fitted, so there is no sampling variance to report.
METHOD_SE_COL = {"h2": "h2_se"}
VARIANTS = [("pheno_expression", "ancestry corrected")]


def load(paths, kind):
    out = []
    for p in paths:
        df = pd.read_parquet(p)
        df["source"] = kind
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, axis="y", color="#e6e5e1", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)


def analyse(embedding_herit, prediction_corr, out_summary,
            out_pdf=None, out_png=None, variant=None, quiet=False):
    """One gene's summary. Importable so a driver can loop in a single process rather
    than paying pandas/scipy/matplotlib import cost per gene.

    Returns (summary, nonconverged): nonconverged is a list of (experiment, variant)
    for EREML (h2) rows dropped because ai_reml_one reported converged=False for the
    real phenotype."""
    df = load(embedding_herit, "embedding")
    df["is_perm"] = df.phenotype.str.startswith("pheno_perm_")

    # Borzoi's own predicted expression, as R2 against the same phenotype variants.
    # It carries no permutation null (nothing is fitted, so there is none to build).
    corr = load(prediction_corr, "prediction") if prediction_corr else pd.DataFrame()

    rows = []
    nonconverged = []
    for exp, g in df.groupby("experiment"):
        for col, label, _ in METHODS:
            if col not in g.columns:
                continue
            null = g[g.is_perm & g[col].notna()][col]
            for pcol, vlabel in VARIANTS:
                src = METHOD_SOURCE.get(col)
                if src and "source" in g.columns:
                    real = g[(g.phenotype == pcol) & (g.source == src)]
                else:
                    real = g[(g.phenotype == pcol) & g[col].notna()]
                if real.empty:
                    continue
                real_row = real.iloc[0]
                # ai_reml_one (h2) can stop at max_iter without meeting its |delta LL|
                # tolerance -- that fit should not be plotted or trusted silently.
                # 'converged' is absent from custom_reml's output (no such concept
                # for its bounded 1-D search), hence the "in real_row" guard.
                if col == "h2" and "converged" in real_row \
                        and real_row["converged"] == False:  # noqa: E712 (numpy bool_, "is False" fails)
                    nonconverged.append((exp, vlabel, label))
                    continue
                if pd.isna(real_row[col]):
                    continue
                r = float(real_row[col])
                se_col = METHOD_SE_COL.get(col)
                h2_se = float(real_row[se_col]) if se_col and se_col in real_row \
                    and pd.notna(real_row[se_col]) else np.nan
                rec = dict(experiment=exp, variant=vlabel, method=label, h2=r, h2_se=h2_se,
                           n_perm=int(null.size))
                if null.size:
                    rec.update(null_mean=float(null.mean()), null_sd=float(null.std()),
                               null_p95=float(np.percentile(null, 95)),
                               p_empirical=float((null >= r).mean()))
                rows.append(rec)
    for _, c in corr.iterrows():
        if not c.get("variant"):
            continue
        # Both r2 (squared Pearson, assumes a linear relationship) and rho2 (squared
        # Spearman, transformation-free) are reported as separate rows/points -- r2
        # alone would penalize a prediction that tracks the phenotype monotonically
        # but non-linearly.
        rows.append(dict(experiment=c["experiment"], variant=c["variant"],
                         method="Borzoi head prediction $R^2$", h2=float(c["r2"]),
                         h2_se=np.nan, n_perm=0, spearman=float(c["spearman"]),
                         spearman_p=float(c["spearman_p"])))
        rows.append(dict(experiment=c["experiment"], variant=c["variant"],
                         method="Borzoi head prediction $\\rho^2$", h2=float(c["rho2"]),
                         h2_se=np.nan, n_perm=0, spearman=float(c["spearman"]),
                         spearman_p=float(c["spearman_p"])))
    summary = pd.DataFrame(rows)
    print("=== GTEx expression heritability ===")
    print(summary.round(4).to_string(index=False))
    summary.to_parquet(out_summary, index=False)

    # Per-gene figures are optional and off by default: with a single gene each is
    # three dots and a null band, i.e. the same content as one row of the parquet,
    # and rendering hundreds of them dominated the task.
    if not out_pdf:
        if not quiet:
            print(f"[done] -> {out_summary}", flush=True)
        return summary, nonconverged

        # ── figure: one panel per covariate set ───────────────────────────────── //
        csets = [v for _, v in VARIANTS if (summary.variant == v).any()]
        titles = {"ancestry corrected": "sex + age + GTEx genotype PCs removed",
                  "no ancestry correction": "sex + age removed, genotype PCs kept in"}
        fig, axes = plt.subplots(1, len(csets), figsize=(5.4 * len(csets), 4.0), squeeze=False)
        fig.patch.set_facecolor("#fcfcfb")
        genes = sorted(summary.experiment.unique())

        for ax, cs in zip(axes[0], csets):
            style(ax)
            sub = summary[summary.variant == cs]
            for mi, (col, label, colr) in enumerate(METHODS):
                s = sub[sub.method == label]
                if s.empty:
                    continue
                x = np.arange(len(genes)) + (mi - 0.5) * 0.26
                keep = [g in set(s.experiment) for g in genes]
                xs = x[np.array(keep)]
                ys = [float(s[s.experiment == g].h2.iloc[0]) for g in genes if g in set(s.experiment)]
                # permutation null as a band per gene
                for xi, g in zip(xs, [g for g in genes if g in set(s.experiment)]):
                    r = s[s.experiment == g].iloc[0]
                    if np.isfinite(r.get("null_p95", np.nan)):
                        ax.add_patch(plt.Rectangle(
                            (xi - 0.1, r["null_mean"] - 1.96 * r["null_sd"]), 0.2,
                            3.92 * r["null_sd"], color=MUTED, alpha=0.20, lw=0, zorder=1))
                ax.plot(xs, ys, "o", color=colr, ms=8, mec="#fcfcfb", mew=1.4,
                        label=label, zorder=3, ls="none")
                if col == "r2":                       # annotate: it sits on the zero line
                    for xi, yi in zip(xs, ys):
                        ax.annotate(f"{yi:.2g}", (xi, yi), textcoords="offset points",
                                    xytext=(9, -3), fontsize=7.5, color=INK2)
            ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
            ax.set_xlim(-0.55, len(genes) - 0.45 + 0.45)   # room for the R2 label
            ax.set_xticks(np.arange(len(genes)))
            ax.set_xticklabels(genes, rotation=0 if len(genes) < 8 else 45,
                               ha="center" if len(genes) < 8 else "right")
            ax.set_ylabel("$h^2$", fontsize=9, color=INK2)
            ax.set_title(titles[cs], fontsize=10, color=INK, loc="left")
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.22 * (hi - lo))     # headroom so the legend clears the marks
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", ncol=2)
        axes[0][0].annotate("grey band = permutation null (95%)", (0.02, 0.02),
                            xycoords="axes fraction", fontsize=7.5, color=INK2)

        genes_title = ", ".join(genes)
        fig.suptitle(genes_title, fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_pdf, facecolor=fig.get_facecolor())
        if out_png:
            fig.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
            plt.close(fig)
        if not quiet:
            print(f"[done] -> {out_summary}", flush=True)




def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding-herit", nargs="+", required=True)
    ap.add_argument("--prediction-corr", nargs="*", default=[],
                    help="per-gene parquets from correlate_predictions.py; their R2 is "
                         "plotted alongside h2 (both are variance-explained fractions)")
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-pdf", default=None,
                    help="optional; omit to skip the figure and only write the table")
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()
    result = analyse(args.embedding_herit, args.prediction_corr,
                      args.out_summary, args.out_pdf, args.out_png)
    nonconverged = result[1] if result else []
    if nonconverged:
        print(f"[warn] {len(nonconverged)} non-converged EREML fit(s) excluded: "
              f"{nonconverged}", flush=True)


if __name__ == "__main__":
    main()
