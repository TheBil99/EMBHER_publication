#!/usr/bin/env python
"""Add donor-label permutation columns to a resolved phenotype parquet -- the
empirical null for a heritability estimate (the estimators loop over every pheno_*
column generically).

Kept separate from expression preparation (workflow 02/prepare_expression_matrix.py):
permutation count and seed are an estimator-side choice (how big a null does THIS
heritability estimate want), not a phenotype-construction one. Nothing downstream
ever joins two different estimators' outputs by permutation index (each aggregates its
own null distribution independently), so there is no need for this to be part of the
cached phenotype -- keeping it out means workflows that estimate heritability
(11-embedding_heritability, 12-heritability_over_layers) can change --n-perm/--seed without invalidating workflow 02's (much more expensive)
covariate-correction cache.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pheno", required=True, help="resolved pheno.parquet (pheno_expression column)")
    ap.add_argument("--pheno-col", default="pheno_expression")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pheno = pd.read_parquet(args.pheno)
    y = pheno[args.pheno_col].to_numpy()
    rng = np.random.default_rng(args.seed)

    cols = {args.pheno_col: y}
    cols.update({f"pheno_perm_{k:03d}": rng.permutation(y) for k in range(args.n_perm)})
    out = pd.DataFrame(cols, index=pheno.index)
    out.index.name = pheno.index.name
    out.to_parquet(args.out)
    print(f"[done] {out.shape[1]} phenotype columns ({args.n_perm} permutations) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
