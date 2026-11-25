#!/usr/bin/env python
"""Concatenate one (gene, kernel, pheno_source) task's exact_reml_heritability.py
outputs into a single parquet matching summarize_calibration.py's own schema, tagged
with the kernel/region/pheno_source this task ran on -- exact_reml_heritability.py's
own output carries none of those (they are a per-gene-region-condition Nextflow-level
fact, not something in the pheno/GRM files themselves).

Two callers in embedding_heritability_simulation_study.nf, both passing --pheno-source explicitly:
  exact_from_grm_genosim   one parquet per true_h2 value (its own pheno.parquet per h2),
                          --pheno-source 'genotype' (matches estimate_genotype_simulated.py's
                          own convention, since this replaces it)
  exact_from_grm_geometry  one parquet (single true_h2, single geometry per task already),
                          --pheno-source '<geometry>' ('dense'/'sparse'/'pc<frac>')

at_boundary/h2_adjr2 placeholder columns are added where the estimator has none
(exact_reml_heritability.py's own at_boundary flag is kept as is), so the combined
output can feed summarize_calibration.py directly, unmodified.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquets", nargs="+", required=True,
                     help="one or more exact_reml_heritability.py outputs")
    ap.add_argument("--kernel-label", required=True)
    ap.add_argument("--region", default="")
    ap.add_argument("--pheno-source", required=True,
                     help="'genotype' for the genosim arm, or the geometry "
                          "('dense'/'sparse'/'pc<frac>') for the geometry arm")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(p) for p in args.parquets], ignore_index=True)
    df["kernel"] = args.kernel_label
    df["region"] = args.region
    df["pheno_source"] = args.pheno_source
    if "at_boundary" not in df.columns:
        df["at_boundary"] = False
    df["h2_adjr2"] = np.nan
    df.to_parquet(args.out, index=False)
    print(f"[done] {len(df)} rows ({df.experiment.iloc[0]}, {args.kernel_label}) -> "
          f"{args.out}", flush=True)


if __name__ == "__main__":
    main()
