#!/usr/bin/env python
"""Rare-variant heritability of one gene, from genotype or from the embedding, with a phenotype
permutation null -- both by Haseman-Elston regression (embher.heritability.he), never jackknifed.

Two arms, fit on the SAME donors and the SAME phenotype columns:

  genotype   the 6 mac_ld_6 genotype bins (--bin-assignment, split_bins_by_ld.py --bins 3,4):
               1 MAC <= mac_max | 2 MAC > mac_max & MAF <= 0.01 |
               3/4 (0.01, 0.1] low/high LD | 5/6 (0.1, 0.5] low/high LD
             rare h2 = h2_k1 + h2_k2 (the two lowest bins).
  embedding  bins 1-2 replaced by the embedding kernel (--grm-npz); the four common LD bins stay:
             kernels = [bin3, bin4, bin5, bin6, embedding]
             rare h2 = the embedding's own component.

Column 0 of the phenotype matrix is the real phenotype; columns 1..n_perm are donor-label
permutations of it (same permutation fed to both arms, so the per-gene difference
h2_embedding - h2_rare_genotype is computed on one shuffled phenotype). Kernels and covariates are
fixed across columns: only the phenotype is shuffled. he_fit solves the kernel system once and every
phenotype column costs a few matrix products, so n_perm columns are far cheaper than n_perm refits.

Donors/cohort follow he_heritability.py with --bfile + --grm-npz + --restrict-grm-to-cohort: the
bed is cut to the donors present in the embedding kernel, genotype GRMs (and their allele
frequencies) are built on donors with a phenotype, covariates are fixed effects, and donors without
a complete covariate row are dropped from the fit. Both arms use these same donors.

Output: one row per phenotype column (perm 0 = real) with the two rare h2, their difference, the
common-variant h2 of each arm, and (every row) each arm's per-component h2.
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd

from embher.heritability.he import _h2, he_fit
from embher.heritability.io import read_bed
from embher.heritability.reml import bin_masks, build_grm, design_matrix, load_npz_kernels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfile", required=True, help="the gene's region PLINK prefix (workflow 04)")
    ap.add_argument("--bin-assignment", required=True,
                    help="mac_ld_6 assignment file (one integer 0-6 per .bim variant)")
    ap.add_argument("--grm-npz", required=True, help="embedding kernel .npz (build_grm.py: K + ids)")
    ap.add_argument("--pheno", required=True, help="pheno.parquet (index IID, column pheno_expression)")
    ap.add_argument("--pheno-col", default="pheno_expression")
    ap.add_argument("--covariate-file", default=None)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0,
                    help="per-gene permutations are seeded by (seed, crc32(experiment)), so they "
                         "do not depend on how genes are batched")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # ── donors: bed cut to the embedding kernel's donors, GRM rows = phenotyped donors ── //
    npz_ids, (K_emb,) = load_npz_kernels([args.grm_npz])
    iids, dosage = read_bed(args.bfile)
    keep = np.isin(iids, npz_ids)
    if not keep.any():
        raise ValueError(f"no donor shared between {args.bfile}.fam and {args.grm_npz}")
    iids, dosage = iids[keep], dosage[keep]
    ph = pd.read_parquet(args.pheno)
    ph.index = ph.index.astype(str)
    y_all = ph[args.pheno_col].reindex(iids).to_numpy(float)
    grm_rows = ~np.isnan(y_all)
    if not grm_rows.any():
        raise ValueError(f"no donor in {args.bfile}.fam has a non-missing phenotype")
    grm_iids = iids[grm_rows]

    # ── kernels: genotype bins from the bed, embedding from the npz ────────── //
    masks = bin_masks(dosage.shape[1], bin_assignment=args.bin_assignment)
    if len(masks) != 6:
        raise ValueError(f"expected the 6-bin mac_ld_6 assignment, got {len(masks)} bins")
    K_geno, n_var = [], []
    for i, mask in enumerate(masks, start=1):
        if not mask.any():
            raise ValueError(f"mac_ld_6 bin {i} has no variants")
        K, m_used = build_grm(dosage[np.ix_(grm_rows, mask)])
        K_geno.append(K)
        n_var.append(m_used)
    del dosage
    pos = {i: k for k, i in enumerate(npz_ids)}
    idx = [pos[i] for i in grm_iids]
    K_emb = K_emb[np.ix_(idx, idx)]

    # ── fixed effects and the donors actually fitted ───────────────────────── //
    X_all = design_matrix(args.covariate_file, grm_iids)
    ok = X_all.notna().all(axis=1).to_numpy()
    X = X_all.to_numpy(float)[ok]
    y = y_all[grm_rows][ok]
    n = len(y)

    # ── phenotype columns: real, then n_perm donor-label permutations ──────── //
    rng = np.random.default_rng([args.seed, zlib.crc32(args.experiment.encode())])
    Y = np.column_stack([y] + [rng.permutation(y) for _ in range(args.n_perm)])

    sub = np.ix_(ok, ok)
    Kg = [K[sub] for K in K_geno]
    Ke = K_emb[sub]
    arms = {"G": Kg,                                   # genotype: 6 bins
            "E": Kg[2:] + [Ke]}                        # embedding: 4 common bins + embedding
    h2 = {}
    finite = np.ones(Y.shape[1], dtype=bool)
    for arm, Ks in arms.items():
        r = he_fit(Ks, Y, X, jackknife=False)
        h2[arm], _ = _h2(r["theta"], r["a"])           # (k, P), scaled h2 per component
        finite &= np.isfinite(r["theta"]).all(axis=0)

    out = pd.DataFrame({
        "experiment": args.experiment,
        "perm": np.arange(Y.shape[1]),
        "h2_rare_genotype": h2["G"][0] + h2["G"][1],
        "h2_rare_embedding": h2["E"][-1],
        "h2_common_genotype": h2["G"][2:].sum(axis=0),             # genotype arm's four common bins
        "h2_common_genotype_embedding_arm": h2["E"][:-1].sum(axis=0),
        "converged": finite,
    })
    out["h2_diff"] = out["h2_rare_embedding"] - out["h2_rare_genotype"]
    for i in range(6):
        out[f"h2_geno_k{i+1}"] = h2["G"][i]
    for i in range(5):
        out[f"h2_emb_arm_k{i+1}"] = h2["E"][i]
    out["n_samples"] = n
    for i, m in enumerate(n_var, start=1):
        out[f"n_variants_bin{i}"] = m
    out.to_parquet(args.out, index=False)
    real = out.iloc[0]
    print(f"[done] {args.experiment}: n={n}, real rare h2 genotype={real.h2_rare_genotype:.4f} "
          f"embedding={real.h2_rare_embedding:.4f}; {args.n_perm} permutations -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
