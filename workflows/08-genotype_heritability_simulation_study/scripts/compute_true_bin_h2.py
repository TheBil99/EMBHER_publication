#!/usr/bin/env python
"""True per-MAF-bin heritability, computed directly from the simulator's own
per-SNP effect sizes (betas.parquet) -- no changes to Simulator_mafld or
workflow 07 needed.

For each replicate, the true per-SNP betas already fix a genetic component
G = X_std @ beta (X_std = mean-imputed, per-SNP standardized genotypes, the
GCTA-model convention the estimators' GRMs use). This is computed with
plink2 --score (variance-standardize, scoresums), never loading the genotype
matrix into Python -- one call per MAF bin (score file restricted to that
bin's variants) plus one "total" call (all variants), each covering all 100
replicates at once via --score-col-nums.

Splitting the SNPs into the same bins the estimators use, each bin's SHARE of the total genetic variance is the covariance-based
partition

    share_bin = Cov(G_bin, G_total) / Var(G_total)

(not Var(G_bin) / Var(G_total) alone -- causal SNPs in different bins can
still be in real LD in the 1000G panel even though ld_ex=0 only turns off
LD-dependent effect-size scaling, not genotype correlation itself, so a
covariance-based split is the one that sums to exactly 1 across bins). Each
bin's true h2 is then share_bin * true_h2 (the simulator's h2 for
this condition), so sum_bin true_h2_bin == true_h2 by construction --
comparable directly to the estimators' own h2_k1 + h2_k2 + ... which target the same
identity.

Output columns: experiment, phenotype, true_h2, true_h2_k1, true_h2_k2, ...
(one true_h2_k{i} per bin, same order as --maf-bins' breakpoints).
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def read_bim(bfile):
    return pd.read_csv(f"{bfile}.bim", sep=r"\s+", header=None,
                        names=["chr", "id", "cm", "bp", "a1", "a2"])


def run_score(plink_bin, bfile, score_file, out_prefix, n_pheno):
    subprocess.run([plink_bin, "--bfile", bfile, "--score", str(score_file),
                    "1", "2", "header-read", "variance-standardize", "cols=scoresums",
                    "--score-col-nums", f"3-{2 + n_pheno}",
                    "--out", str(out_prefix)],
                   check=True, capture_output=True, text=True)
    sscore = pd.read_csv(f"{out_prefix}.sscore", sep=r"\s+")
    sscore = sscore.rename(columns={"#IID": "IID"}).set_index("IID")
    return sscore[[c for c in sscore.columns if c.endswith("_SUM")]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bfile", required=True, help="region genotypes PLINK prefix")
    ap.add_argument("--betas", required=True, help="betas.parquet (chrom,pos,ref,alt,pheno_*)")
    ap.add_argument("--maf-ld", required=True, help="maf_ld.txt (header + MAF/ldscore, BIM order)")
    ap.add_argument("--maf-bins", default=None,
                     help="comma-separated ascending MAF breakpoints, bins (lo, hi], "
                          "e.g. '0,0.01,0.5'. "
                          "Mutually exclusive with --bin-assignment.")
    ap.add_argument("--bin-assignment", default=None,
                     help="path to a per-variant bin-assignment file (one integer "
                          "1..n_bins per line, same order as --bfile's .bim), the same "
                          "file the estimator used, so the true-h2 split matches its "
                          "bins exactly, variant for variant.")
    ap.add_argument("--true-h2", type=float, required=True)
    ap.add_argument("--experiment", default="")
    ap.add_argument("--strategy", default="width",
                     help="how --maf-bins' breakpoints were chosen -- 'width' (fixed MAF "
                          "ranges) or 'eqcount' (equal variant count per bin, rank-based). "
                          "Carried through as a column so results from the two strategies, "
                          "which can share the same bin COUNT with different edges, are never "
                          "matched against each other downstream.")
    ap.add_argument("--plink-bin", default="plink2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.maf_bins and args.bin_assignment:
        raise ValueError("--maf-bins and --bin-assignment are mutually exclusive")
    if not args.maf_bins and not args.bin_assignment:
        raise ValueError("one of --maf-bins or --bin-assignment is required")

    bim = read_bim(args.bfile)
    maf = pd.read_csv(args.maf_ld, sep=r"\s+")["MAF"].to_numpy()
    if len(maf) != len(bim):
        raise ValueError(f"maf_ld.txt has {len(maf)} SNPs but bim has {len(bim)}")

    betas = pd.read_parquet(args.betas)
    pheno_cols = [c for c in betas.columns if c.startswith("pheno_")]
    if len(betas) != len(bim):
        raise ValueError(f"betas.parquet has {len(betas)} SNPs but bim has {len(bim)}")

    weights = pd.DataFrame({"id": bim["id"].values, "a1": bim["a1"].values})
    for c in pheno_cols:
        weights[c] = betas[c].values

    if args.maf_bins:
        breakpoints = [float(x) for x in args.maf_bins.split(",")]
        n_bins = len(breakpoints) - 1
        bin_masks = [(maf > breakpoints[i]) & (maf <= breakpoints[i + 1]) for i in range(n_bins)]
    else:
        # An explicit per-variant bin assignment -- must be the SAME partition
        # --bin-assignment the estimator used for this method, or the
        # "true" bin split and the estimated bin split would disagree on
        # which bin a variant landed in, silently comparing mismatched bins.
        assignment = np.loadtxt(args.bin_assignment, dtype=int)
        if len(assignment) != len(bim):
            raise ValueError(f"bin assignment has {len(assignment)} entries but bim has {len(bim)}")
        n_bins = int(assignment.max())
        bin_masks = [assignment == i for i in range(1, n_bins + 1)]

    with tempfile.TemporaryDirectory() as wd:
        wd = Path(wd)
        total_file = wd / "score_total.tsv"
        weights.to_csv(total_file, sep="\t", index=False)
        gtot = run_score(args.plink_bin, args.bfile, total_file, wd / "total", len(pheno_cols))

        gbins = []
        for i, mask in enumerate(bin_masks, start=1):
            bin_file = wd / f"score_bin{i}.tsv"
            weights.loc[mask].to_csv(bin_file, sep="\t", index=False)
            gbins.append(run_score(args.plink_bin, args.bfile, bin_file, wd / f"bin{i}", len(pheno_cols)))

    rows = []
    for col in pheno_cols:
        sum_col = f"{col}_SUM"
        Gtot = gtot[sum_col].to_numpy()
        var_tot = Gtot.var()
        row = dict(experiment=args.experiment, phenotype=col, true_h2=args.true_h2,
                   nbins=n_bins, strategy=args.strategy)
        for i, gbin in enumerate(gbins, start=1):
            # var_tot == 0 means every beta was exactly 0 (true_h2 == 0 scales every
            # causal SNP's effect variance to 0), not an undefined 0/0 share -- with no
            # genetic signal at all, every bin's true h2 is trivially 0 too, and that's
            # the one condition where showing the estimator's bias matters most (it's exactly
            # the false-positive/inflation check: how much h2 does it attribute to a
            # bin when there truly is none).
            if var_tot == 0:
                row[f"true_h2_k{i}"] = 0.0
            else:
                Gbin = gbin[sum_col].to_numpy()
                share = np.cov(Gbin, Gtot, ddof=0)[0, 1] / var_tot
                row[f"true_h2_k{i}"] = share * args.true_h2
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    means = [df[f"true_h2_k{i}"].mean() for i in range(1, n_bins + 1)]
    print(f"[done] {args.experiment}: true_h2={args.true_h2}, mean true_h2_k*={means} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
