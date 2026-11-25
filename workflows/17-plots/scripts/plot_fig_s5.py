#!/usr/bin/env python
"""fig_s5: GEUVADIS counterpart of fig_2a -- genotype h2 (HE regression, workflow 10) against
Borzoi head prediction h2 and against Borzoi embedding h2 (both HE regression, workflow 11), two
square scatters, same style as fig_2 (AX cm axes, ticks every 0.25, palette colours, y = x line).

Cohort: results/11-embedding_heritability/geuvadis/hi_q_geuvadis_random500 (embeddings and head
predictions, 500 genes) joined to results/10-genotype_heritability/geuvadis/hi_q_geuvadis_genes
(genotype, many more genes). Everything here is keyed by Ensembl gene id (no gene map needed):
the embedding across_genes.parquet index, the per-gene prediction parquets' `experiment` column
and the genotype parquets' <ENSG>_<method>.parquet names. Converged fits only (HE is
unconstrained, so a converged fit can still have h2 < 0 -- kept). Inner join per panel.

Writes fig_s5.{pdf,png} plus the two joined tables behind the panels into --out-dir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

EMBEDDING_COL = "EREML"   # across_genes.parquet's own column name, regardless of estimator
GENOTYPE_METHOD = "he_mac_ld_6_covar"
AX = 3.8          # cm: side of every square axes (same as fig_2)
WSPACE = 0.7      # gap between the two scatters, in axes widths
H = 5.6           # cm: row height (as fig_2)


def load_genotype(h2_dir: str, method: str, genes: list[str]) -> pd.Series:
    """Genotype h2 (Ensembl id -> value) of converged `method` fits; <ENSG>_*.parquet per gene."""
    out, missing, unconverged = {}, [], []
    for ensg in genes:
        matches = sorted(Path(h2_dir).glob(f"{ensg}_*.parquet"))
        if not matches:
            missing.append(ensg)
            continue
        df = pd.read_parquet(matches[0])
        df = df[df["method"] == method]
        if df.empty:
            missing.append(ensg)
        elif df["converged"].iloc[0] != True:  # noqa: E712
            unconverged.append(ensg)
        else:
            out[ensg] = float(df["h2"].iloc[0])
    if missing:
        print(f"[warn] {len(missing)} gene(s) missing/no {method} row in {h2_dir}: "
              f"{missing[:10]}{' ...' if len(missing) > 10 else ''}", flush=True)
    if unconverged:
        print(f"[warn] dropping {len(unconverged)} gene(s) with a non-converged {method} fit",
              flush=True)
    return pd.Series(out, name="h2_genotype")


def load_prediction_h2(pred_dir: str) -> pd.Series:
    """One parquet per gene (`experiment` = Ensembl id); converged fits only."""
    files = sorted(Path(pred_dir).glob("*_prediction_heritability.parquet"))
    if not files:
        raise SystemExit(f"[error] no *_prediction_heritability.parquet files in {pred_dir}")
    pred = pd.DataFrame([pd.read_parquet(f).iloc[0] for f in files]).set_index("experiment")
    keep = pred["converged"] == True  # noqa: E712
    if (~keep).any():
        print(f"[warn] dropping {int((~keep).sum())} gene(s) with a non-converged prediction h2 fit",
              flush=True)
    return pred.loc[keep, "h2"].rename("h2_prediction")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding-parquet", required=True,
                    help="results/11-embedding_heritability/geuvadis/hi_q_geuvadis_random500/"
                         "summary/overall/across_genes.parquet")
    ap.add_argument("--prediction-heritability-dir", required=True,
                    help="same run's prediction_heritability/")
    ap.add_argument("--genotype-h2-dir", required=True,
                    help="results/10-genotype_heritability/geuvadis/hi_q_geuvadis_genes/h2/he_mac_ld_6")
    ap.add_argument("--genotype-method", default=GENOTYPE_METHOD)
    ap.add_argument("--embher-plotting-src", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.embher_plotting_src)
    from embher_plotting import apply_style, archetypes as A, annotations as N, layout as L
    from embher_plotting.theme import P2F_COLORS as P2F

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    embedding = pd.read_parquet(args.embedding_parquet)[EMBEDDING_COL].rename("h2_embedding")
    prediction = load_prediction_h2(args.prediction_heritability_dir)
    genotype = load_genotype(args.genotype_h2_dir, args.genotype_method,
                             sorted(set(embedding.index) | set(prediction.index)))
    df_embed = pd.concat([embedding, genotype], axis=1, join="inner").dropna()
    df_pred = pd.concat([prediction, genotype], axis=1, join="inner").dropna()
    df_embed.to_parquet(out / "fig_s5_genotype_h2_vs_embedding_h2.parquet")
    df_pred.to_parquet(out / "fig_s5_genotype_h2_vs_prediction_h2.parquet")

    W = AX * (2 + WSPACE) / 0.94           # grid spans 94% of the canvas width
    apply_style()
    fig = plt.figure(figsize=L.figsize_cm(W, H))
    inner = fig.add_gridspec(1, 2, wspace=WSPACE, left=0.06, right=1.0)
    panels = [(df_pred["h2_prediction"], df_pred["h2_genotype"], "$h^2_P$", P2F["prediction"]),
              (df_embed["h2_embedding"], df_embed["h2_genotype"], "$h^2_E$", P2F["embedding"])]
    for col, (x, y, xlabel, color) in enumerate(panels):
        ax = fig.add_subplot(inner[0, col])
        A.scatter(ax, x.to_numpy(float), y.to_numpy(float), color=color, stats="pearson")
        N.identity_line(ax, color=P2F["reference"])
        ax.set_box_aspect(1)
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("$h^2_G$")
    L.save(fig, out / "fig_s5", rasterize=True, transparent=False, png_dpi=300)
    print(f"[done] n={len(df_embed)} genes (genotype vs embedding), n={len(df_pred)} genes "
          f"(genotype vs prediction h2) -> {out}", flush=True)


if __name__ == "__main__":
    main()
