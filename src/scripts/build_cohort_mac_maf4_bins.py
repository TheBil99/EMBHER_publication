#!/usr/bin/env python
"""Four-bin MAC/MAF cohort split for exact_reml_heritability.py --bin-assignment:

    bin 1  MAC <= --mac-max
    bin 2  MAC >  --mac-max  and MAF <= 0.01
    bin 3  0.01 < MAF <= 0.1
    bin 4  0.1  < MAF <= 0.5

i.e. the 3-bin MAC/MAF arm with its common bin split once, at MAF 0.1. It replaces a
5-bin version (0.01-0.05 and 0.05-0.1 as separate bins) whose exact-REML fits ended
at_boundary for 384/1000 genes in workflow 10, often via a negative 0.01-0.1 bin. Used
by workflows 08 (genotype_heritability_simulation_study) and 08
(genotype_heritability). MAC and MAF are
computed on this gene's phenotyped cohort exactly as src/scripts/
build_cohort_mac_maf_bins.py does (plink2 --freq --keep on the cohort, MAC =
round(MAF * OBS_CT)), whose cohort_ids() is reused. Rules apply in the order above,
so a variant is in bin 1 whenever MAC <= --mac-max.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cohort_mac_maf_bins import cohort_ids  # noqa: E402

MAF_EDGES = [0.01, 0.1, 0.5]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfile", required=True, help="region genotypes PLINK prefix")
    ap.add_argument("--pheno", required=True, help="this gene's pheno.parquet (index IID, cols pheno_*)")
    ap.add_argument("--plink-bin", default="plink2")
    ap.add_argument("--mac-max", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as wd:
        wd = Path(wd)
        keep = cohort_ids(args.bfile, args.pheno)
        keep_path = wd / "keep.txt"
        keep[[0, 1]].to_csv(keep_path, sep="\t", header=["#FID", "IID"], index=False)
        subprocess.run([args.plink_bin, "--bfile", args.bfile, "--keep", str(keep_path),
                        "--freq", "--out", str(wd / "cohort_freq")],
                       check=True, capture_output=True, text=True)
        freq = pd.read_csv(wd / "cohort_freq.afreq", sep=r"\s+")

    bim = pd.read_csv(f"{args.bfile}.bim", sep=r"\s+", header=None,
                      names=["chr", "id", "cm", "bp", "a1", "a2"])
    freq = freq.set_index("ID").reindex(bim["id"])
    if freq["ALT_FREQS"].isna().any():
        raise ValueError(f"{int(freq['ALT_FREQS'].isna().sum())} bim variant(s) missing from cohort .afreq")
    af = freq["ALT_FREQS"].to_numpy(dtype=float)
    maf = np.minimum(af, 1.0 - af)
    mac = np.round(maf * freq["OBS_CT"].to_numpy(dtype=float)).astype(int)

    assignment = np.zeros(len(bim), dtype=int)
    for i, (lo, hi) in enumerate(zip(MAF_EDGES[:-1], MAF_EDGES[1:]), start=3):
        assignment[(maf > lo) & (maf <= hi)] = i
    assignment[(mac > args.mac_max) & (maf <= MAF_EDGES[0])] = 2
    assignment[mac <= args.mac_max] = 1

    np.savetxt(args.out, assignment, fmt="%d")
    counts = pd.Series(assignment).value_counts().sort_index()
    print(f"[done] {args.bfile}: cohort n={len(keep)}, bin counts={counts.to_dict()} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
