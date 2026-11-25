#!/usr/bin/env python
"""Donor x variant table of non-reference allele counts (0/1/2) for one gene's window.

Reads workflow 04's region bfile with embher.heritability.io.read_bed,
which counts the .bim's second allele; REF is taken from the variant ID
(chr_pos_REF_ALT_b38), so the output counts ALT whichever allele plink stored second.
Missing calls are counted as reference (0).
"""
import argparse

import numpy as np
import pandas as pd

from embher.heritability.io import read_bed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfile", required=True, help="PLINK prefix (bed/bim/fam)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    iids, dosage = read_bed(a.bfile)                                   # counts of bim A2
    bim = pd.read_csv(f"{a.bfile}.bim", sep=r"\s+", header=None)
    parts = bim[1].str.split("_")
    if not (parts.str.len() == 5).all():
        raise SystemExit(f"{a.bfile}.bim: variant IDs are not chr_pos_REF_ALT_b38")
    ref = parts.str[2].to_numpy()
    a2 = bim[5].to_numpy()
    if not np.all((a2 == ref) | (bim[4].to_numpy() == ref)):
        raise SystemExit(f"{a.bfile}.bim: REF from the variant ID is neither allele")
    nonref = np.where(a2 == ref, 2 - dosage, dosage)
    print(f"{nonref.shape[0]} donors x {nonref.shape[1]} variants; "
          f"{int(np.isnan(nonref).sum())} missing calls counted as reference")
    nonref = np.nan_to_num(nonref, nan=0.0)
    df = pd.DataFrame(nonref, index=pd.Index(iids, name="donor_id"), columns=bim[1].to_numpy())
    df.to_parquet(a.out)


if __name__ == "__main__":
    main()
