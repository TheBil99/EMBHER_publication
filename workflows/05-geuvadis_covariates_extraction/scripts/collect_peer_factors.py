#!/usr/bin/env python
"""Attach sample IDs to peertool's headerless X.csv (n_factors x samples) output.

Confirmed empirically (not from reading src/peer/src/peerutil.cpp's write_output,
which suggested the opposite orientation for a non---transpose run): with our
peertool invocation (no --transpose flag), X.csv comes out n_factors rows x
n_samples columns, not samples x n_factors.
"""
from __future__ import annotations

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x-csv", required=True, help="peertool's X.csv (no header, n_factors x samples)")
    ap.add_argument("--samples", required=True, help="individual IDs, one per line, column order of --x-csv")
    ap.add_argument("--n-factors", type=int, required=True)
    ap.add_argument("--out", required=True, help="writes a parquet, index=IID, columns=PEER1..PEER<n>")
    args = ap.parse_args()

    samples = [l.strip() for l in open(args.samples)]
    x = pd.read_csv(args.x_csv, header=None).T
    if len(x) != len(samples):
        raise SystemExit(f"[error] X.csv has {len(x)} samples (after transpose) but {len(samples)} were expected")
    if x.shape[1] != args.n_factors:
        raise SystemExit(f"[error] X.csv has {x.shape[1]} factors (after transpose) but --n-factors={args.n_factors}")

    x.index = pd.Index(samples, name="IID")
    x.columns = [f"PEER{i + 1}" for i in range(args.n_factors)]
    x.to_parquet(args.out)
    print(f"[done] {x.shape[0]} samples x {x.shape[1]} PEER factors -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
