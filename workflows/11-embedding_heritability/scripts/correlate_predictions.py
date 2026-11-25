#!/usr/bin/env python
"""Do Borzoi's predicted expressions track measured GTEx expression across donors?

Correlates the per-donor predicted expression (combine_haplotypes.py's
pred_expression: haplotypes summed per track, then averaged across the same-tissue
tracks, on Borzoi's raw unsquashed-coverage scale -- no log transform) against the
measured phenotype workflow 02 (expression_data_preparation) prepared for this gene
(pheno_expression, "ancestry corrected": sex + age + genotype PCs removed).

With --covar, the SAME covariates embedding_heritability adds to the REML fixed
effects (GTEx's own per-tissue PEER covariates, see build_gtex_eqtl_covar.py) are
regressed out of the measured phenotype before correlating -- so the R2/rho2 reported
here is on the same footing as h2, which is fit on a model that already accounts for
those covariates as fixed effects. Only the measured side is corrected; pred_expression
is used exactly as combine_haplotypes.py produced it -- the model has no way to know
about a donor's covariates, so there's nothing to regress out of it.

The `r2` column -- the one plotted next to h2 -- is a plain Pearson r2. Spearman is
also reported since it doesn't assume the relationship is linear.

The decisive companion measurement is the DISPERSION of the predictions. A null
correlation means something different depending on it: if predicted expression barely
varies between donors, the model is not responding to personal sequence at all, which
is a different failure from responding in the wrong direction.
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
# the phenotype variant label used by summarize_embedding_heritability.py
VARIANT = "ancestry corrected"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a85"


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


def residualize(y, covar):
    """OLS-residualize y against [intercept, covar] -- same fixed-effect design
    embedding_heritability builds from the identical covariate file."""
    X = np.column_stack([np.ones(len(y)), covar])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def boot_ci(x, y, stat, n=500, seed=0):
    """Percentile bootstrap CI. 500 resamples is ample for the 3 decimals reported;
    2000 cost ~2.3 s per gene, which dominated a 237-gene run."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) == 0 or np.std(y[i]) == 0:
            continue
        vals.append(stat(x[i], y[i]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) \
        if vals else (np.nan, np.nan)


def analyse(predictions, pheno_path, gene_id, experiment,
            out_summary, out_pdf=None, out_png=None, quiet=False, covar_path=None):
    """One gene. Importable, so a driver can loop over genes in a single process and pay
    the ~1.5 s pandas/scipy/matplotlib import cost once for the whole run instead of once
    per gene."""
    pred = pd.read_parquet(predictions)
    pred.index = pred.index.astype(str)
    pheno = pd.read_parquet(pheno_path)
    pheno.index = pheno.index.astype(str)

    donors = [d for d in pheno.index if d in pred.index]
    print(f"[info] {len(donors)} donors with both prediction and measurement", flush=True)

    p = pred.loc[donors, "pred_expression"].to_numpy(float)
    m = pheno.loc[donors, "pheno_expression"].to_numpy(float)

    if covar_path:
        covar = pd.read_csv(covar_path, sep="\t", index_col=0)
        covar.index = covar.index.astype(str)
        missing = [d for d in donors if d not in covar.index]
        if missing:
            raise SystemExit(
                f"[error] {len(missing)} donor(s) missing from --covar {covar_path}: "
                f"{missing[:10]}" + (" ..." if len(missing) > 10 else "")
            )
        m = residualize(m, covar.loc[donors].to_numpy(dtype=float))

    sr, sp = stats.spearmanr(p, m)
    pr, pp = stats.pearsonr(p, m)
    lo, hi = boot_ci(p, m, lambda a, b: stats.spearmanr(a, b)[0])
    summary = pd.DataFrame([dict(
        experiment=experiment, gene=gene_id, measure="pred_expression",
        variant=VARIANT, n=len(donors), spearman=float(sr), spearman_p=float(sp),
        spearman_ci_lo=lo, spearman_ci_hi=hi,
        pearson=float(pr), pearson_p=float(pp),
        r2=float(pr) ** 2, rho2=float(sr) ** 2,
    )])

    # dispersion: is the model moving at all? pred and m are on unrelated scales
    # (raw Borzoi coverage vs the workflow-02-transformed phenotype), so only each
    # side's own spread is reported -- no cross-scale ratio.
    disp = dict(
        pred_mean=float(p.mean()), pred_sd=float(p.std()),
        meas_mean=float(m.mean()), meas_sd=float(m.std()),
        n_unique_pred=int(len(np.unique(np.round(p, 6)))),
    )
    print("\n=== correlation of Borzoi-predicted vs measured expression ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== dispersion (is the model responding to personal sequence?) ===")
    for k, v in disp.items():
        print(f"  {k:22s} {v:.6g}")

    for k, v in disp.items():
        summary[k] = v
    summary.to_parquet(out_summary, index=False)

    # Per-gene figures are optional and off by default: across a few hundred genes
    # they dominated the runtime and are not used downstream.
    if not out_pdf:
        if not quiet:
            print(f"[done] {experiment} -> {out_summary}", flush=True)
        return summary

        # ── figure ────────────────────────────────────────────────────────────── //
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
        fig.patch.set_facecolor("#fcfcfb")

        ax = axes[0]
        style(ax)
        ax.plot(p, m, "o", color=S1, ms=5, alpha=0.55, mec="none")
        sr_row = summary.iloc[0]
        ax.set_xlabel("Borzoi predicted expression\n(unsquashed coverage, GTEx blood tracks)",
                      fontsize=9, color=INK2)
        ax.set_ylabel("measured (workflow 02 phenotype)", fontsize=9, color=INK2)
        ax.set_title(f"(a) predicted vs measured, {experiment}", fontsize=10, color=INK, loc="left")
        ax.annotate(f"Spearman $\\rho$ = {sr_row.spearman:+.3f}\n"
                    f"95% CI [{sr_row.spearman_ci_lo:+.3f}, {sr_row.spearman_ci_hi:+.3f}]\n"
                    f"p = {sr_row.spearman_p:.3g}   n = {int(sr_row.n)}",
                    (0.03, 0.83), xycoords="axes fraction", fontsize=8, color=INK2, va="top")

        ax = axes[1]
        style(ax)
        for vals, lab, colr in ((p - p.mean(), "Borzoi predicted", S1),
                                (m - m.mean(), "measured", S2)):
            ax.hist(vals, bins=40, color=colr, alpha=0.6, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("deviation from cohort mean (own units)", fontsize=9, color=INK2)
        ax.set_ylabel("donors", fontsize=9, color=INK2)
        ax.set_title("(b) how much each varies between donors", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
        ax.annotate(f"SD: predicted {disp['pred_sd']:.3g}, measured {disp['meas_sd']:.3g}",
                    (0.03, 0.03), xycoords="axes fraction", fontsize=8, color=INK2)

        fig.tight_layout()
        fig.savefig(out_pdf, facecolor=fig.get_facecolor())
        if out_png:                        # PNG is optional; the PDF is the deliverable
            fig.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    if not quiet:
        print(f"[done] {experiment} -> {out_summary}")




def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--pheno", required=True, help="pheno.parquet from prepare_expression_matrix.py, via workflow 02")
    ap.add_argument("--gene-id", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-pdf", required=True)
    ap.add_argument("--out-png", default=None, help="optional; the PDF is the deliverable")
    ap.add_argument("--covar", default=None,
                    help="optional TSV of fixed-effect covariates, indexed by IID (same "
                         "file embedding_heritability uses); regressed out of the "
                         "measured phenotype before correlating")
    args = ap.parse_args()
    analyse(args.predictions, args.pheno, args.gene_id,
            args.experiment, args.out_summary, args.out_pdf, args.out_png,
            covar_path=args.covar)


if __name__ == "__main__":
    main()
