#!/usr/bin/env python
"""Simulate one (gene, geometry, true_h2) phenotype set and write it to a pheno.parquet, so
exact_reml_heritability.py (which only reads phenotypes from disk, via --pheno) can fit
the IDENTICAL simulated signal in every kernel arm.

The phenotype is simulated in the RAW kernel's eigenbasis (from --basis), where the
signal sits is chosen by --geometry:

  dense    g ~ N(0, K)                    the model's own assumption, true by construction
  sparse   g = one real embedding feature  a single causal feature
  pcX      g = one eigenvector of K        placed at a FRACTION of the kernel's rank
  randompc g = one eigenvector of K        a DIFFERENT uniformly random PC in every
                                           replicate (like `sparse` with features), drawn
                                           among the leading PCs that together explain
                                           --random-pc-var (default 90%) of K's variance

Seeded on (experiment, geometry, true_h2, seed) only -- NOT on the kernel -- so this runs
ONCE per (gene, geometry, true_h2) and every kernel arm's estimate reuses it.
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd


def make_g(kind, U, lam, Z, idx, rng):
    if kind == "dense":
        return U @ (np.sqrt(lam) * rng.normal(size=U.shape[0]))
    if kind == "sparse":
        return Z[:, rng.integers(Z.shape[1])].astype(float).copy()
    if kind == "pc":
        return U[:, idx].copy()
    if kind == "randompc":
        return U[:, rng.integers(idx)].copy()  # idx = number of eligible PCs
    raise ValueError(kind)


def make_y(g, h2, n, rng):
    if h2 == 0:
        y = rng.normal(size=n)
    else:
        g = (g - g.mean()) / g.std()
        e = rng.normal(size=n)
        e = (e - e.mean()) / e.std()
        y = np.sqrt(h2) * g + np.sqrt(1.0 - h2) * e
    return (y - y.mean()) / y.std()


def stable_seed(*parts):
    """Deterministic across processes, unlike builtin hash() on str -- which is salted by
    PYTHONHASHSEED and so differs between interpreter runs."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basis", required=True, help="prepare_basis.py .npz")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--geometry", required=True,
                    help="'dense', 'sparse', 'randompc', or 'pc<frac>' (frac of the kernel's rank)")
    ap.add_argument("--random-pc-var", type=float, default=0.9,
                    help="randompc: draw among the leading PCs explaining this fraction of "
                         "the kernel's variance")
    ap.add_argument("--true-h2", type=float, required=True)
    ap.add_argument("--n-reps", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    b = np.load(args.basis, allow_pickle=True)
    lam = b["lam"]
    U = np.ascontiguousarray(b["U"], dtype=np.float64)
    rank = int(b["rank"])
    ids = [str(i) for i in b["ids"]]
    lam_pos = np.clip(lam, 0.0, None)
    n = U.shape[0]

    geom = args.geometry
    if geom in ("dense", "sparse"):
        kind, idx = geom, None
        Z = np.ascontiguousarray(b["Z"], dtype=np.float64) if geom == "sparse" else None
    elif geom == "randompc":
        kind, Z = "randompc", None
        cum = np.cumsum(lam_pos[:rank]) / lam_pos[:rank].sum()
        idx = min(int(np.searchsorted(cum, args.random_pc_var)) + 1, rank)
        print(f"[randompc] drawing among the top {idx} of {rank} PCs "
              f"({args.random_pc_var:.0%} of variance)", flush=True)
    elif geom.startswith("pc"):
        kind, Z = "pc", None
        frac = float(geom[2:])
        idx = min(max(int(round(frac * rank)), 0), rank - 1)
    else:
        raise SystemExit(f"[error] unrecognised geometry '{geom}'")

    # Seeded on (experiment, geom, true_h2, seed), NOT the kernel, so every kernel arm
    # fits the identical phenotype.
    rng = np.random.default_rng(stable_seed(args.experiment, geom, args.true_h2, args.seed))

    cols = {}
    for rep in range(args.n_reps):
        y = make_y(make_g(kind, U, lam_pos, Z, idx, rng), args.true_h2, n, rng)
        cols[f"pheno_{rep}"] = y

    df = pd.DataFrame(cols, index=pd.Index(ids, name="IID"))
    df.to_parquet(args.out)
    print(f"[done] {args.experiment} {geom} true_h2={args.true_h2}: {args.n_reps} reps "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
