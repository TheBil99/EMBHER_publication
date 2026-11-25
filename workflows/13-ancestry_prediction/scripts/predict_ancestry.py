#!/usr/bin/env python
"""EUR vs AFR ancestry predictive power (OLS R^2) for one or more donor x feature
tables, no held-out split -- we only want a quick read of how separable the two groups
are in each representation, not a generalization estimate. Takes any mix of files in
one call (a batch of many genes/variants is just as cheap as one), so the Nextflow side
can pool thousands of tiny per-variant fits into a handful of tasks instead of spawning
one task each.

gene/variant/rep(s) are inferred from each --features file's name, not passed on the
command line (see src/scripts/genotype_features.py / extract_layer_variants.py, the two
producers of these files):
  <gene>_genotype.parquet              -> variant 'genotype', rep 'genotype'
  <gene>__<variant>_embedding.parquet  -> rep 'raw' AND 'whitened' (<variant> is a
                                           layer:pooling label or 'rna_predictions')

Every representation is first reduced to its top --n-pcs principal components (default
10, far below the donor count) before the fit -- NOT the full feature matrix. Column
count is almost always >= donor count here (genotype windows, most embedding layers),
so an unreduced OLS fit would trivially interpolate the labels (R^2 = 1 regardless of
real signal); capping every representation at the same small, fixed number of
components keeps R^2 comparable across genotype/raw/whitened and meaningful.
  raw/genotype   columns centred -> SVD -> top n-pcs scores (U*S), unscaled
  whitened       columns standardised (mean 0, unit variance) -> SVD -> top
                 round(--pc-frac * rank) components rescaled to unit variance, EXACTLY
                 src/scripts/pca_embeddings.py's construction (default --pc-frac 0.9)
                 -> then keep only the first n-pcs columns of that whitened basis for
                 the fit. Whitening first and truncating to n-pcs after (rather than
                 going straight to n-pcs) matches the whitening workflow 11/12 use for
                 GRMs; in practice it gives the identical first n-pcs columns either
                 way (whitening rescales each column independently to unit variance,
                 so truncating to n-pcs before or after that rescaling doesn't change
                 which components or their magnitude -- only the pc-frac truncation
                 point could matter, and only if it fell below n-pcs).

OLS fit: y (EUR=0, AFR=1) ~ 1 + top-n-pcs scores, ordinary least squares (closed form,
no regularization, no split). Reported as R^2 -- the in-sample fraction of label
variance the representation's leading components explain.

Donors are restricted to --groups' EUR/AFR individuals (others excluded), matching
workflow 14's make_ancestry_groups.py --comparison eur_afr.

Output: one TSV, one row per (file, rep), to --out.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from embher.embedding.whiten import svd_robust, whiten_embeddings

RANK_EIGVAL_RATIO = 1e-8  # matches pca_embeddings.py's numerical-rank threshold


def parse_features_path(path):
    """(gene, variant, reps) from a genotype_features.py / extract_layer_variants.py
    output filename -- see module docstring."""
    name = path.name
    if name.endswith("_genotype.parquet"):
        return name[: -len("_genotype.parquet")], "genotype", ["genotype"]
    if name.endswith("_embedding.parquet") and "__" in name:
        gene, variant = name[: -len("_embedding.parquet")].split("__", 1)
        return gene, variant, ["raw", "whitened"]
    raise SystemExit(f"[error] can't parse gene/variant from filename: {name}")


def top_pcs(X, rep, n_pcs, pc_frac):
    """Top min(n_pcs, rank) PC scores: centred (raw/genotype) or whitened (whitened).

    'whitened' whitens onto round(pc_frac * rank) components first
    (embher.embedding.whiten.whiten_embeddings, the same call pca_embeddings.py makes), THEN keeps
    only the first n_pcs columns of that whitened basis.
    """
    if rep == "whitened":
        Z = whiten_embeddings(X, pc_frac=pc_frac, verbose=False)[:, :n_pcs]
        return Z, Z.shape[1]
    A = X - X.mean(0)
    U, S, _ = svd_robust(A)
    rank_keep = S ** 2 > (S[0] ** 2) * RANK_EIGVAL_RATIO
    U, S = U[:, rank_keep], S[rank_keep]
    k = min(n_pcs, len(S))
    return (U * S)[:, :k], k


def ols_r2(Z, y):
    design = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", required=True,
                    help="one or more donor x feature parquet files (genotype and/or "
                         "embedding-variant tables, any mix)")
    ap.add_argument("--groups", required=True, help="IID/GROUP TSV (EUR/AFR donors only)")
    ap.add_argument("--n-pcs", type=int, default=10)
    ap.add_argument("--pc-frac", type=float, default=0.9,
                    help="'whitened' only: whiten onto this fraction of rank before "
                         "truncating to --n-pcs (pca_embeddings.py's --pc-frac)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    groups = pd.read_csv(a.groups, sep="\t").set_index("IID").GROUP
    n_files = len(a.features)
    rows = []
    for i, fp in enumerate(a.features, 1):
        path = Path(fp)
        gene, variant, reps = parse_features_path(path)
        df = pd.read_parquet(path)
        df.index = df.index.astype(str)
        df = df[df.index.isin(groups.index)]
        X = df.to_numpy(np.float64)
        y = (groups.loc[df.index] == "AFR").to_numpy().astype(float)
        print(f"[{i}/{n_files} {100 * i / n_files:5.1f}%] {gene} {variant}: "
              f"{len(df)} donors ({int(y.sum())} AFR, {int((1 - y).sum())} EUR), "
              f"{X.shape[1]} features", flush=True)
        for rep in reps:
            Z, k = top_pcs(X, rep, a.n_pcs, a.pc_frac)
            r2 = ols_r2(Z, y)
            rows.append({"gene": gene, "variant": variant, "rep": rep, "n": len(df),
                         "n_afr": int(y.sum()), "m": k, "r2": r2})
            print(f"[{i}/{n_files}] {gene} {variant} {rep}: R^2 = {r2:.3f} (k={k})", flush=True)
    pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)
    print(f"[done] {n_files}/{n_files} files, {len(rows)} rows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
