#!/usr/bin/env python
"""Combine workflow 05's (geuvadis_covariates_extraction) three covariate sources into
the donor-indexed TSV exact_reml_heritability.py's
--covariate-file expects (IID + one column per covariate) -- the GEUVADIS-side
counterpart of build_gtex_eqtl_covar.py,
which does the equivalent transpose for GTEx's own published per-tissue eQTL covariates
file. Unlike GTEx's single published file, GEUVADIS's three pieces come from different
places (workflow 05's own PEER run; a third-party 1000G ancestry PCA panel for genotype
PCs; E-GEUV-1.sdrf.txt for sex/sequencing-center), so this script's job is specifically
the join, not just a transpose.

Inner-joined on IID: a donor missing from ANY of the three sources (in practice, only
the genotype-PCs file is ever short a few donors -- see extract_genotype_pcs.py's own
docstring) is dropped from the combined covariate file entirely, since a heritability
model needs every covariate for every donor it uses.
"""
from __future__ import annotations

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--peer-factors", required=True, help="workflow 05's peer_factors.parquet")
    ap.add_argument("--genotype-pcs", required=True, help="workflow 05's genotype_pcs.tsv")
    ap.add_argument("--technical-covariates", required=True,
                     help="workflow 05's technical_covariates.tsv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    peer = pd.read_parquet(args.peer_factors)
    pcs = pd.read_csv(args.genotype_pcs, sep="\t", index_col="IID")
    tech = pd.read_csv(args.technical_covariates, sep="\t", index_col="IID")

    donors = peer.index.intersection(pcs.index).intersection(tech.index)
    dropped = (set(peer.index) | set(pcs.index) | set(tech.index)) - set(donors)
    if dropped:
        print(f"[warn] {len(dropped)} donor(s) missing from at least one covariate source, "
              f"dropped: {sorted(dropped)}", flush=True)

    out = pd.concat([pcs.loc[donors], tech.loc[donors], peer.loc[donors]], axis=1)
    out.index.name = "IID"
    out.to_csv(args.out, sep="\t")
    print(f"[done] {out.shape[0]} donors x {out.shape[1]} covariates "
          f"({pcs.shape[1]} genotype PCs + {tech.shape[1]} technical + {peer.shape[1]} PEER) "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
