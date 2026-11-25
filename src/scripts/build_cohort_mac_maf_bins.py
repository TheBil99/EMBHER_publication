#!/usr/bin/env python
"""Per-variant bin assignment for the estimators' --bin-assignment mode,
MAC-and-MAF composite bins computed on the ACTUAL analysis cohort -- the donors with a
non-missing phenotype for this gene -- rather than on workflow 04's own maf.txt, which
is computed once over the whole 944-donor GTEx WGS panel regardless of tissue (see that
workflow's own "TISSUE-AGNOSTIC BY CONSTRUCTION" docstring). A tissue's expression donor
set is a subset of the WGS panel, so MAF/MAC computed over all 944 donors can differ
from what this gene's actual REML fit sees.

Three-bin mode (--n-bins 3):
    bin 1  MAC <= --mac-max
    bin 2  MAC >  --mac-max  and MAF <= --maf-common
    bin 3  MAF >= --maf-common
Two-bin mode (--n-bins 2): same as bin 2/3 above, bin-1 variants (MAC <= --mac-max) are
    assigned 0 (excluded from every bin, not just left out of the joint fit's variance
    accounting); any value <1 means "in no bin".

MAC is derived as round(MAF * OBS_CT) from a plink2 --freq --keep run restricted to the
cohort (not workflow 04's own unfiltered maf.txt) -- OBS_CT is plink2's own allele
observation count for that variant among the kept donors (2 * non-missing individuals
for an autosomal SNP), so this is the true per-variant, per-cohort count, not a
--fam-row-count approximation.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def cohort_ids(bfile, pheno_path):
    """Donors present in both the region's .fam and with a non-missing phenotype
    value -- the set the heritability fit uses (a genotyped but unphenotyped donor
    is dropped)."""
    fam = pd.read_csv(f"{bfile}.fam", sep=r"\s+", header=None)
    fam[1] = fam[1].astype(str)
    ph = pd.read_parquet(pheno_path)
    ph.index = ph.index.astype(str)
    pheno_cols = [c for c in ph.columns if c.startswith("pheno_")]
    if not pheno_cols:
        raise ValueError(f"{pheno_path} has no pheno_* column")
    non_missing = set(ph.index[ph[pheno_cols[0]].notna()])
    keep = fam[fam[1].isin(non_missing)]
    if keep.empty:
        raise ValueError(f"no donor in {bfile}.fam has a non-missing phenotype in {pheno_path}")
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bfile", required=True, help="region genotypes PLINK prefix")
    ap.add_argument("--pheno", required=True, help="this gene's pheno.parquet (index IID, cols pheno_*)")
    ap.add_argument("--plink-bin", default="plink2")
    ap.add_argument("--mac-max", type=int, required=True,
                     help="variants with cohort MAC <= this go in bin 1 (3-bin mode) "
                          "or are excluded entirely (2-bin mode)")
    ap.add_argument("--maf-common", type=float, required=True,
                     help="the common-variant MAF floor for the last bin")
    ap.add_argument("--n-bins", type=int, choices=[2, 3], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as wd:
        wd = Path(wd)
        keep = cohort_ids(args.bfile, args.pheno)
        keep_path = wd / "keep.txt"
        # Explicit #FID/IID header -- plink2's own unambiguous sample-ID-file format,
        # rather than relying on its headerless-file fallback.
        keep[[0, 1]].to_csv(keep_path, sep="\t", header=["#FID", "IID"], index=False)

        freq_prefix = wd / "cohort_freq"
        subprocess.run([args.plink_bin, "--bfile", args.bfile, "--keep", str(keep_path),
                         "--freq", "--out", str(freq_prefix)],
                        check=True, capture_output=True, text=True)
        freq = pd.read_csv(f"{freq_prefix}.afreq", sep=r"\s+")

    bim = pd.read_csv(f"{args.bfile}.bim", sep=r"\s+", header=None,
                       names=["chr", "id", "cm", "bp", "a1", "a2"])
    freq = freq.set_index("ID").reindex(bim["id"])
    if freq["ALT_FREQS"].isna().any():
        missing = freq.index[freq["ALT_FREQS"].isna()].tolist()
        raise ValueError(f"{len(missing)} bim variant(s) missing from cohort .afreq, "
                          f"e.g. {missing[:5]}")

    af = freq["ALT_FREQS"].to_numpy(dtype=float)
    obs_ct = freq["OBS_CT"].to_numpy(dtype=float)
    maf = np.minimum(af, 1.0 - af)
    mac = np.round(maf * obs_ct).astype(int)

    rare = mac <= args.mac_max
    common = maf >= args.maf_common
    mid = ~rare & ~common

    assignment = np.zeros(len(bim), dtype=int)
    if args.n_bins == 3:
        assignment[rare] = 1
        assignment[mid] = 2
        assignment[common] = 3
    else:
        # rare (MAC <= mac_max) left at 0 -- excluded from every bin.
        assignment[mid] = 1
        assignment[common] = 2

    np.savetxt(args.out, assignment, fmt="%d")
    counts = pd.Series(assignment).value_counts().sort_index()
    print(f"[done] {args.bfile}: n_bins={args.n_bins}, cohort n={len(keep)}, "
          f"bin counts={counts.to_dict()} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
