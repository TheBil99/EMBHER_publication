#!/usr/bin/env python
"""Turn one tissue's WHOLE GTEx expression matrix into a phenotype matrix (donors x
genes), in one pass -- for every gene in the tissue at once, not one gene at a time.

No covariate correction is applied: the transformed phenotype is used exactly as
constructed, not even mean-centered or an intercept removed.

Phenotype -- exactly one of two pipelines, chosen with --transform:

  quantile_int  Quantile-normalize the raw TPM matrix (normalize_quantiles(), GTEx's
                own eQTL-pipeline recipe: each donor's distribution of values across
                ALL genes is normalized to the same reference distribution, the
                average -- across donors -- of each sorted rank; ties averaged), THEN
                a rank-based inverse-normal transform per gene across donors. No log2
                anywhere in this pipeline -- quantile normalization already fixes each
                donor's marginal distribution, so a log transform beforehand is
                redundant.
  log2_plus_1   log2(TPM+1), used as-is.

The donor list is still filtered to donors with both genotype PCs and subject
phenotypes available (--genotype-pcs/--subject-phenotypes), matching the donor
population used elsewhere in this project (e.g. the genotype-GRM branch), even though
neither file's values are used as a covariate here -- this is a donor-membership check,
not a covariate. Both are optional: omit them for a cohort with no such files (e.g.
GEUVADIS), and every donor column in --tissue-matrix is kept as-is.

Everything above is gene-independent for a fixed (tissue, transform): the donor list
doesn't depend on which gene is being processed. This script computes the phenotype for
every gene in one vectorized pass instead of once per gene.

Only the real phenotype is produced here -- no permutation columns. Permutation
count/seed is an estimator-side choice, not a phenotype-construction one (see
src/scripts/add_permutations.py, applied downstream by whichever workflow actually
estimates heritability), so it stays out of this cache entirely.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def _rankdata_chunk(chunk: np.ndarray) -> np.ndarray:
    return rankdata(chunk, axis=0, method="average")


def rank_avg(M: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    """Average-tie rank of every column of M independently (axis=0), vectorized via
    scipy.stats.rankdata's own axis support (a single C-level call per chunk, not a
    Python-level loop over columns like pandas' DataFrame.rank). Each column is fully
    independent of every other, so for n_jobs > 1 the columns are split into that many
    chunks and ranked in separate processes -- genuine parallel use of multiple CPUs,
    since numpy/scipy's rank/sort routines are single-threaded on their own.
    """
    if n_jobs <= 1 or M.shape[1] < n_jobs:
        return rankdata(M, axis=0, method="average")
    chunks = np.array_split(M, n_jobs, axis=1)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        ranked_chunks = list(ex.map(_rankdata_chunk, chunks))
    return np.concatenate(ranked_chunks, axis=1)


def inverse_normal_matrix(Y: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    """Rank-based inverse-normal transform, applied independently to each column
    (gene) of a donors x genes matrix. Genes are ranked in parallel across n_jobs
    processes."""
    ranks = rank_avg(Y, n_jobs)
    return norm.ppf(ranks / (Y.shape[0] + 1.0))


def normalize_quantiles(M: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    """Quantile normalization to the average empirical distribution across columns --
    GTEx's own eQTL-pipeline preprocessing step (matches R's
    preprocessCore::normalize.quantiles; Bolstad et al. 2003, Bioinformatics 19(2),
    pp. 185-193). M is genes x donors; normalizes each donor's (column's) distribution
    of values across ALL genes to the same reference distribution -- the average,
    across donors, of each sorted rank. Ties within a column are averaged.

    Vectorized reformulation of the reference algorithm (rank, then linear-interpolate
    into the sorted reference distribution at that rank, rather than looping over
    columns and explicitly re-averaging each tie block): for a block of L consecutive
    tied sorted values, linear interpolation at the block's own (fractional, for even
    L) average rank is algebraically identical to the median of those L reference
    values, so this reproduces the reference's tie-averaging exactly with no explicit
    per-column loop. Donors are ranked in parallel across n_jobs processes.
    """
    m, _ = M.shape
    quantiles = np.sort(M, axis=0).mean(axis=1)   # length m, ascending reference distribution
    ranks = rank_avg(M, n_jobs)   # 1..m, ties averaged
    lo = np.clip(np.floor(ranks).astype(int) - 1, 0, m - 1)
    hi = np.clip(lo + 1, 0, m - 1)
    frac = ranks - np.floor(ranks)
    return quantiles[lo] * (1 - frac) + quantiles[hi] * frac


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tissue-matrix", required=True, help="cached genes x donors TPM parquet")
    ap.add_argument("--genotype-pcs", default=None,
                     help="GTEx 20_Genotype_PCs.eigenvec.txt.gz; omit for a cohort with none")
    ap.add_argument("--subject-phenotypes", default=None,
                     help="GTEx SubjectPhenotypesDS.txt; omit for a cohort with none")
    ap.add_argument("--transform", choices=["quantile_int", "log2_plus_1"], required=True)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="parallelize the per-gene/per-donor rank computations "
                         "(quantile normalization, inverse-normal transform) across "
                         "this many processes -- match to --cpus-per-task")
    ap.add_argument("--out-pheno-matrix", required=True)
    ap.add_argument("--out-donors", required=True)
    args = ap.parse_args()

    expr = pd.read_parquet(args.tissue_matrix)   # genes x donors

    if args.genotype_pcs and args.subject_phenotypes:
        pcs = pd.read_csv(args.genotype_pcs, sep="\t")
        pcs["donor"] = pcs["IID"].astype(str).str.split("-").str[:2].str.join("-")
        pcs = pcs.drop_duplicates("donor").set_index("donor")

        sub = pd.read_csv(args.subject_phenotypes, sep="\t").set_index("SUBJID")

        donors = [d for d in expr.columns if d in pcs.index and d in sub.index]
        dropped = expr.shape[1] - len(donors)
        if dropped:
            print(f"[warn] {dropped} donors dropped for missing GTEx PCs/phenotypes", flush=True)
        print(f"[info] {len(donors)} donors with expression, genotype PCs and phenotypes; "
              f"{expr.shape[0]} genes", flush=True)
    else:
        donors = list(expr.columns)
        print(f"[info] {len(donors)} donors (no genotype-PCs/subject-phenotypes membership "
              f"filter applied); {expr.shape[0]} genes", flush=True)

    if args.transform == "quantile_int":
        qn = normalize_quantiles(expr.loc[:, donors].to_numpy(dtype=np.float64), args.n_jobs)   # genes x donors
        Y = inverse_normal_matrix(qn.T, args.n_jobs)   # donors x genes
    else:   # log2_plus_1
        Y = np.log2(expr.loc[:, donors].T.astype(float).to_numpy() + 1.0)   # donors x genes

    pheno_matrix = pd.DataFrame(Y, index=donors, columns=expr.index)
    pheno_matrix.index.name = "IID"

    pheno_matrix.to_parquet(args.out_pheno_matrix)
    with open(args.out_donors, "w") as fh:
        fh.write("\n".join(donors) + "\n")
    print(f"[done] {pheno_matrix.shape[1]} genes x {pheno_matrix.shape[0]} donors, "
          f"transform={args.transform} -> {args.out_pheno_matrix}", flush=True)


if __name__ == "__main__":
    main()
