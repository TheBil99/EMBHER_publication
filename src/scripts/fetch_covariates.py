#!/usr/bin/env python
"""Copy out workflow 02's covar_full.tsv for one tissue (fixed at n_gpcs/num_expr_pcs/
r2_cutoff), read directly from its pheno_matrix manifest row.

resolve_expression.py's 'pheno' product does not expose covar_file_path (only
pheno_path/donors_path). Only the 'full' cache-key row has the covariates (sex,
age, DTHHRDY, genotype PCs, expression PCs, QC metrics) on disk, so that row is read;
the covariates are then used as REML fixed effects.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--transform", required=True)
    ap.add_argument("--n-gpcs", required=True)
    ap.add_argument("--num-expr-pcs", required=True)
    ap.add_argument("--r2-cutoff", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tk = slug(args.tissue)
    cache_key = (f"{tk}_t{args.transform}_vfull_g{args.n_gpcs}"
                 f"_p{args.num_expr_pcs}_r{args.r2_cutoff}")
    manifest_row = Path(args.manifest_dir) / tk / f"{cache_key}.tsv"
    if not manifest_row.exists():
        raise SystemExit(f"[error] no manifest row at {manifest_row}")
    row = pd.read_csv(manifest_row, sep="\t", dtype=str).iloc[0]
    covar_path = Path(args.data_dir) / row["covar_file_path"]
    shutil.copy(covar_path, args.out)
    print(f"[done] copied {covar_path} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
