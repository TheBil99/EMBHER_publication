#!/usr/bin/env python
"""Precompute everything the per-cell simulation jobs share, once per gene.

Splitting the sweep into hundreds of small jobs is only worthwhile if each one does not
redo the setup. Every (geometry, true h2) cell needs the same three things, none of which
depend on the cell:

  K     the raw kernel        -- ai_reml_one fits against it directly
  lam,U its eigendecomposition -- custom_reml needs the eigenbasis, AND the geometries
                                  are DEFINED in it (pcX is literally a column of U, and
                                  the `dense` draw is U @ (sqrt(lam) * z))
  Z     the column-standardised embedding -- only the `sparse` geometry needs it

`rank` is stored rather than the per-geometry PC indices, so the consumer can resolve any
--pc-fracs value without this step being re-run.

Written as a single .npz so a cell job stages one file. ~29 MB per gene (K and U are
944x944 float64, Z is 944x1920); float64 throughout because REML's conditioning does not
tolerate float32 here.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def load_embedding(path):
    df = pd.read_parquet(path)
    df.index = df.index.astype(str)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grm-npz", required=True)
    ap.add_argument("--emb-parquet", required=True,
                    help="the gene's embeddings parquet, as staged by the workflow")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.grm_npz, allow_pickle=True)
    K = np.ascontiguousarray(z["K"], dtype=np.float64)
    ids = [str(i) for i in z["ids"]]
    n = K.shape[0]

    lam, U = np.linalg.eigh(K)
    lam, U = lam[::-1].copy(), U[:, ::-1].copy()
    rank = int((lam > 1e-8 * lam[0]).sum())
    lam_pos = np.clip(lam, 0.0, None)
    eff_rank = float(lam_pos.sum() ** 2 / (lam_pos ** 2).sum())

    emb = load_embedding(args.emb_parquet)
    missing = [i for i in ids if i not in emb.index]
    if missing:
        raise SystemExit(f"[error] {len(missing)} GRM donors absent from the embedding "
                         f"(first few: {missing[:3]})")
    Z = emb.loc[ids].to_numpy(dtype=np.float64)
    sd = Z.std(axis=0)
    Z = (Z[:, sd > 0] - Z.mean(axis=0)[sd > 0]) / sd[sd > 0]

    # K is loaded from build_grm.py's output and Z is rebuilt here from the embedding.
    # Nothing upstream forces them to describe the same matrix, and every geometry below
    # is defined in eigh(K)'s basis while `sparse` draws from Z -- so a mismatch would
    # silently place signal in one basis and estimate it in another.
    gap = float(np.abs(K - Z @ Z.T / Z.shape[1]).max())
    if gap > 1e-8:
        raise SystemExit(f"[error] K does not match ZZ'/M for this embedding "
                         f"(max|K - ZZ'/M| = {gap:.3e}); the GRM and the embedding "
                         f"parquet describe different feature matrices")

    np.savez(args.out, K=K, lam=lam, U=U, Z=Z, ids=np.array(ids),
             rank=rank, eff_rank=eff_rank)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[done] {args.experiment}: n={n} M={Z.shape[1]} rank={rank} "
          f"eff_rank={eff_rank:.1f} lambda_1={lam[0]:.4g} -> {args.out} "
          f"({size_mb:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
