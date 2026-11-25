#!/usr/bin/env python
"""Build a GRM from per-donor embeddings: K = Z Z' / M.

Z is the embedding matrix with each feature standardized (mean 0, unit variance)
across donors; M is the number of retained features. Zero-variance features are
dropped. Writes:
  - <out>.npz          : K (float64) + donor ids  (used by the Python REML)
  - <out>.grm.bin/.N.bin/.grm.id : GCTA-format GRM

Optional --min-eigval-ratio guards against a degenerate GRM: when the embedding's
raw feature count is comparable to or exceeds the number of donors and the features
are collinear (e.g. 1543 correlated RNA-seq tracks over 800 donors), K's eigenvalue
spectrum can carry a long tail of near-zero-but-technically-nonzero directions that
are numerically indistinguishable from noise. REML can then treat that tail as
donor-specific "genetic" signal and saturate h2 toward 1, regardless of any real
relationship between the embedding and the phenotype. Setting this flag drops
eigen-components below the given fraction of the top eigenvalue before writing K, so
only the effective (non-degenerate) rank contributes to the genetic component.
Off by default, so existing callers (workflows 08/06) are unaffected unless they
explicitly opt in.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from embher.heritability.grm_io import write_gcta_grm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", nargs="+", required=True,
                    help="one or more embeddings parquet(s) (shards/panels are concatenated)")
    ap.add_argument("--pheno", default=None,
                    help="optional pheno.parquet; the GRM is built ONLY on its individuals "
                         "(so features are standardized over the analysis subset, not the whole panel)")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--min-eigval-ratio", type=float, default=None,
                    help="if set, zero out K's eigen-components whose eigenvalue is below "
                         "this fraction of the top eigenvalue -- see module docstring. "
                         "Default: off (exact previous behavior).")
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(p) for p in args.embeddings], axis=0)
    df.index = df.index.astype(str)
    df = df[~df.index.duplicated(keep="first")]

    # Subset to the analysis individuals BEFORE standardizing, so the GRM's
    # feature scaling reflects exactly the population used for heritability.
    if args.pheno:
        pid = pd.read_parquet(args.pheno).index.astype(str)
        keep_ids = [i for i in df.index if i in set(pid)]
        missing = len(set(pid)) - len(keep_ids)
        df = df.loc[keep_ids]
        print(f"[info] subset to {len(keep_ids)} phenotyped individuals "
              f"({missing} in pheno lacked embeddings)", flush=True)

    ids = df.index.astype(str).tolist()
    Z = df.to_numpy(dtype=np.float64)

    # Standardize features; drop zero-variance columns.
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    keep = sd > 0
    Z = (Z[:, keep] - mu[keep]) / sd[keep]
    M = Z.shape[1]
    print(f"[info] {Z.shape[0]} donors, {M} retained features "
          f"({int((~keep).sum())} zero-variance dropped)", flush=True)

    K = (Z @ Z.T) / M

    if args.min_eigval_ratio is not None:
        eigvals, eigvecs = np.linalg.eigh(K)
        thresh = eigvals.max() * args.min_eigval_ratio
        keep = eigvals > thresh
        eigvals_kept = np.where(keep, eigvals, 0.0)
        K = (eigvecs * eigvals_kept) @ eigvecs.T
        print(f"[info] eigenvalue truncation (ratio={args.min_eigval_ratio}): "
              f"kept {int(keep.sum())}/{len(eigvals)} components, dropped "
              f"{int((~keep).sum())} near-degenerate directions", flush=True)

    np.savez(f"{args.out}.npz", K=K, ids=np.array(ids), n_markers=M)
    write_gcta_grm(args.out, K.copy(), ids, M)
    print(f"[done] GRM {K.shape} -> {args.out}.npz (+ GCTA {args.out}.grm.*)", flush=True)


if __name__ == "__main__":
    main()
