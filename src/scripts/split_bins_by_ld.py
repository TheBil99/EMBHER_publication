#!/usr/bin/env python
"""Split bins of a per-variant bin assignment into a low- and a high-LD half.

Input: an assignment file (one integer per .bim variant, 0 = in no bin), e.g.
build_cohort_mac_maf4_bins.py's 4-bin MAC/MAF split. Output: the same format, bins renumbered
in order: an unsplit bin keeps one id, a split bin gets two consecutive ids (low LD = the
variants below the bin's median LD score, then high LD). --bins picks which input bins to split
(default: all). On the 4 MAC/MAF bins:
  all bins (8 bins)  1/2 = MAC <= 5 low/high LD, 3/4 = MAC > 5 & MAF <= 0.01 low/high,
                     5/6 = (0.01, 0.1] low/high, 7/8 = (0.1, 0.5] low/high
  --bins 3,4 (6)     1 = MAC <= 5, 2 = MAC > 5 & MAF <= 0.01, 3/4 = (0.01, 0.1] low/high,
                     5/6 = (0.1, 0.5] low/high -- the mac_ld_6 binning
Used by workflows 08 (genotype_heritability_simulation_study: mac_ld_6, mac_maf_ld_bins_8) and
08 (genotype_heritability: mac_ld_6, its default binning).

LD score: l_j = sum_l r_jl^2 over every variant in the region (the variant itself included,
no distance window), r_jl = z_j'z_l / n on the phenotyped cohort, with the same
standardization the GRMs use (embher.heritability.reml.build_grm: HWE-scaled, allele frequency on the
cohort, missing calls z = 0). r^2 is not corrected for sample size: the usual correction
r^2 - (1 - r^2)/(n - 2) is the same linear map for every pair, and every variant sums over
the same variant set, so it would not change which half any variant falls in.

The split is by rank (the lower floor(n_b / 2) variants of bin b go to 2b-1), so the two
halves are balanced even with ties. Monomorphic variants (no LD score; build_grm drops them
anyway) go to the low half and are not counted in the split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from embher.heritability.io import read_bed

# cohort_ids: the phenotyped-cohort definition the MAC/MAF assignment itself was built on
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cohort_mac_maf_bins import cohort_ids  # noqa: E402


def ld_scores(dosage, chunk=2000):
    """Per-variant region-wide LD score on the rows of `dosage` (NaN for monomorphic)."""
    n, m = dosage.shape
    p = np.nanmean(dosage, axis=0) / 2.0
    keep = (p > 0) & (p < 1)
    z = (dosage[:, keep] - 2 * p[keep]) / np.sqrt(2 * p[keep] * (1 - p[keep]))
    z[np.isnan(z)] = 0.0
    z = z.astype(np.float32)
    ell_keep = np.empty(z.shape[1])
    for s in range(0, z.shape[1], chunk):
        r = z[:, s:s + chunk].T @ z / n
        ell_keep[s:s + chunk] = (r.astype(np.float64) ** 2).sum(axis=1)
    ell = np.full(m, np.nan)
    ell[keep] = ell_keep
    return ell


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfile", required=True, help="region genotypes PLINK prefix")
    ap.add_argument("--pheno", required=True, help="pheno.parquet; its first pheno_* column defines the cohort")
    ap.add_argument("--bin-assignment", required=True, help="input assignment, one integer per bim variant")
    ap.add_argument("--bins", default=None,
                    help="comma-separated input bins to split (default: all)")
    ap.add_argument("--out", required=True, help="output assignment (see module docstring for the numbering)")
    args = ap.parse_args()

    iids, dosage = read_bed(args.bfile)
    keep_ids = cohort_ids(args.bfile, args.pheno)[1].astype(str).to_numpy()   # .fam rows -> IIDs
    rows = np.isin(np.asarray(iids).astype(str), keep_ids)
    a = np.loadtxt(args.bin_assignment, dtype=int)
    if len(a) != dosage.shape[1]:
        raise ValueError(f"assignment has {len(a)} entries but {args.bfile}.bim has {dosage.shape[1]}")

    n_in = int(a.max())
    to_split = set(range(1, n_in + 1)) if args.bins is None else {int(x) for x in args.bins.split(",")}
    if not to_split <= set(range(1, n_in + 1)):
        raise ValueError(f"--bins {args.bins} names bins outside 1..{n_in}")
    ell = ld_scores(dosage[rows])
    out = np.zeros_like(a)
    print(f"[info] {rows.sum()} cohort donors, {len(a)} variants", flush=True)
    nxt = 1
    for b in range(1, n_in + 1):
        idx = np.where(a == b)[0]
        if len(idx) == 0:
            raise ValueError(f"input bin {b} is empty")
        if b not in to_split:
            out[idx] = nxt
            print(f"[info] bin {b}: {len(idx)} variants -> bin {nxt} (not split)", flush=True)
            nxt += 1
            continue
        poly = idx[np.isfinite(ell[idx])]
        mono = idx[~np.isfinite(ell[idx])]
        order = poly[np.argsort(ell[poly], kind="stable")]
        half = len(order) // 2
        low, high = np.concatenate([order[:half], mono]), order[half:]
        if len(low) == 0 or len(high) == 0:
            raise ValueError(f"input bin {b} has too few variants ({len(idx)}) to split")
        out[low], out[high] = nxt, nxt + 1
        print(f"[info] bin {b}: {len(idx)} variants -> bin {nxt} low LD {len(low)} (median l "
              f"{np.median(ell[order[:half]]):.1f}), bin {nxt + 1} high LD {len(high)} (median l "
              f"{np.median(ell[high]):.1f}); {len(mono)} monomorphic", flush=True)
        nxt += 2
    np.savetxt(args.out, out, fmt="%d")
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
