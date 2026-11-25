#!/usr/bin/env python
"""Transpose GTEx's own published per-tissue eQTL covariates file (donors as
columns, one row per genotype-PC/PEER-factor/platform/pcr/sex covariate) into
the donor-indexed TSV the heritability scripts' --covariate-file expects (IID + one column
per covariate). This is GTEx's real covariate set (60 PEER factors for Whole
Blood's ~800 samples, per their published N>=350 rule), not this project's own
expression-PC approximation (src/scripts/fetch_covariates.py)."""
from __future__ import annotations

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--covariates-file", required=True,
                     help="GTEx_Analysis_v10_eQTL_covariates.tar's per-tissue "
                          "<Tissue>.v10.covariates.txt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cov = pd.read_csv(args.covariates_file, sep="\t", index_col=0)
    covT = cov.T
    covT.index.name = "IID"
    covT.to_csv(args.out, sep="\t")
    print(f"[done] {covT.shape[0]} donors x {covT.shape[1]} covariates -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
