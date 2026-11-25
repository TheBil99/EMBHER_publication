#!/usr/bin/env python
"""Extract sex + sequencing-center covariates for GEUVADIS donors from
E-GEUV-1.sdrf.txt -- the GEUVADIS-side counterpart of GTEx's covariate file's own
'sex'/'platform'/'pcr' rows (see build_gtex_eqtl_covar.py). 'Characteristics[laboratory]'
(GEUVADIS's 7 sequencing centers, confirmed one value per individual -- no ambiguity)
stands in for GTEx's platform/pcr batch covariates: it's GEUVADIS's own well-known major
technical batch effect. One-hot encoded with the first lab dropped as reference (k-1
dummies for a k-level categorical, avoiding collinearity with an intercept), unlike
GTEx's platform/pcr, which are already binary.

Population/superpopulation is deliberately NOT extracted as a covariate here, matching
GTEx's own covariate recipe (no population column there either) -- that's what the
genotype PCs (extract_genotype_pcs.py) are for; adding both would double-correct for
the same population structure.
"""
from __future__ import annotations

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sdrf", required=True, help="E-GEUV-1.sdrf.txt")
    ap.add_argument("--samples", required=True, help="GEUVADIS donor IDs, one per line")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    samples = [l.strip() for l in open(args.samples) if l.strip()]

    sdrf = pd.read_csv(args.sdrf, sep="\t", dtype=str)
    per_donor = sdrf.drop_duplicates("Characteristics[individual]").set_index("Characteristics[individual]")

    missing = [s for s in samples if s not in per_donor.index]
    if missing:
        raise SystemExit(f"[error] {len(missing)} donor(s) not found in {args.sdrf}: {missing}")

    sex = per_donor.loc[samples, "Characteristics[sex]"].map({"male": 1, "female": 2})
    if sex.isna().any():
        raise SystemExit(f"[error] unexpected sex value(s) for: {sex[sex.isna()].index.tolist()}")

    lab = per_donor.loc[samples, "Characteristics[laboratory]"]
    lab_dummies = pd.get_dummies(lab, prefix="lab", drop_first=True, dtype=int)

    out = pd.concat([sex.rename("sex"), lab_dummies], axis=1)
    out.index = pd.Index(samples, name="IID")
    out.to_csv(args.out, sep="\t")
    print(f"[done] {out.shape[0]} donors x {out.shape[1]} technical covariates "
          f"(sex + {lab_dummies.shape[1]} lab dummies) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
