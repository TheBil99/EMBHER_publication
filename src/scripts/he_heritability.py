#!/usr/bin/env python
"""Heritability by Haseman-Elston (HE) regression on genotype GRMs and/or prebuilt
kernels (--grm-npz), covariates as fixed effects, with delete-one jackknife SEs.

Drop-in companion of exact_reml_heritability.py: same inputs (--bfile, --pheno,
--covariate-file, --maf-bins/--maf-ld or --bin-assignment, --restrict-grm-to-cohort,
--grm-npz) and the same output columns (method, h2, h2_se, true_h2, converged,
at_boundary, n_samples, h2_k{i}/h2_k{i}_se, h2_genotype/h2_genotype_se). The GRMs, bin
masks, covariate design (intercept + VIF<=10 filter) and kernel loading come from
embher.heritability.reml, the same functions exact_reml_heritability.py uses, so HE and REML always fit
the identical kernels and fixed effects.

Model:  y = X b + sum_k g_k + e,  g_k ~ N(0, s_k K_k),  e ~ N(0, s_e I).

Estimator (one component set, no measurement-noise term): with M = I - X (X'X)^-1 X' and r = M y, regress the outer product t = r r' on
Q_l = M A_l M, A_l in {K_1, ..., K_k, I}, by least squares over ALL n x n entries
(diagonal included):  theta = (Q'Q)^-1 Q't, where (Q'Q)_lm = tr(M A_l M A_m) and
(Q't)_l = r' A_l r. theta = (s_1..s_k, s_e). No constraint: components may be negative.

h2: h2_k = a_k s_k / (sum_j a_j s_j + s_e), a_k = mean(diag K_k) -- the same
total-variance-share convention exact_reml_heritability.py uses, so the two estimators
are on one scale. The alternative s_k / (sum_j s_j + s_e) assumes mean(diag K) = 1; on
standardized GRMs (mean diag ~1.01-1.05) that differs by a few percent, so it is also
written, as h2_unscaled / h2_unscaled_se.

SEs (--jackknife loo): delete-one jackknife -- drop donor i (its row and column of every
kernel, its row of y and of X), refit HE on the remaining n-1 (projection M rebuilt), and
Var = (n-1) * np.var(estimates) (ddof 0). The n refits are computed in closed form, not by n explicit
refits: removing one donor is a rank-one change to X'X and drops one row/column from each
kernel, so every fold's Q'Q and Q't follow from full-sample quantities in O(n p^2) per
fold -- algebraically identical to refitting (checked against brute-force refits), and shared across all phenotype columns with the same donors.
--jackknife none skips it (h2_se = NaN).

converged is True whenever Q'Q is invertible (HE is closed-form: nothing to converge);
at_boundary is always False (no positive-definiteness requirement). Both are written only
so the output joins exact_reml_heritability.py's schema.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from embher.heritability.he import _h2, he_fit, jk_var
from embher.heritability.io import read_bed
from embher.heritability.reml import bin_masks, build_grm, design_matrix, load_npz_kernels


# ── CLI ──────────────────────────────────────────────────────────────────── //

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfile", default=None,
                    help="region genotypes PLINK prefix; required unless --grm-npz is given")
    ap.add_argument("--grm-npz", nargs="+", default=None,
                    help="one or more prebuilt GRM .npz files (build_grm.py: K + ids), alone or "
                         "jointly with --bfile's genotype GRM(s) -- as exact_reml_heritability.py")
    ap.add_argument("--pheno", required=True, help="pheno.parquet (index IID, cols pheno_*)")
    ap.add_argument("--covariate-file", default=None,
                    help="optional TSV of fixed-effect covariates indexed by IID; an intercept is "
                         "always fit")
    ap.add_argument("--maf-bins", default=None,
                    help="comma-separated ascending MAF breakpoints, bins (lo, hi]")
    ap.add_argument("--maf-ld", default=None, help="MAF column in bim order; required with --maf-bins")
    ap.add_argument("--bin-assignment", default=None,
                    help="one integer per bim variant, 0 = excluded from every bin")
    ap.add_argument("--restrict-grm-to-cohort", action="store_true",
                    help="GRM allele frequencies on the phenotyped donors only (one pheno_* column)")
    ap.add_argument("--jackknife", choices=["loo", "none"], default="loo",
                    help="loo (default): delete-one jackknife SEs; none: no SEs")
    ap.add_argument("--experiment", default="")
    ap.add_argument("--true-h2", type=float, default=np.nan)
    ap.add_argument("--method", default=None,
                    help="label for the output 'method' column (default: he_single_grm / he_maf_bins / "
                         "he_bin_assignment / he_from_grm / he_joint_from_grm / he_joint_genotype_grm)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not args.bfile and not args.grm_npz:
        raise ValueError("at least one of --bfile or --grm-npz is required")
    if (args.maf_bins or args.bin_assignment or args.restrict_grm_to_cohort) and not args.bfile:
        raise ValueError("--maf-bins/--bin-assignment/--restrict-grm-to-cohort require --bfile")
    if args.maf_bins and args.bin_assignment:
        raise ValueError("--maf-bins and --bin-assignment are mutually exclusive")
    breakpoints = [float(x) for x in args.maf_bins.split(",")] if args.maf_bins else None
    if breakpoints and not args.maf_ld:
        raise ValueError("--maf-ld is required when --maf-bins is given")
    binned = breakpoints is not None or bool(args.bin_assignment)
    if args.bfile and args.grm_npz:
        default_method = "he_joint_genotype_grm"
    elif args.grm_npz:
        default_method = "he_joint_from_grm" if len(args.grm_npz) > 1 else "he_from_grm"
    else:
        default_method = ("he_maf_bins" if breakpoints else
                          "he_bin_assignment" if args.bin_assignment else "he_single_grm")
    method = args.method or default_method

    # ── kernels, phenotypes, covariates: exactly as exact_reml_heritability.py main() ── //
    npz_Ks = []
    if args.grm_npz:
        npz_ids, npz_Ks = load_npz_kernels(args.grm_npz)
    if args.bfile:
        iids, dosage = read_bed(args.bfile)
        if args.grm_npz:
            keep = np.isin(iids, npz_ids)
            if not keep.any():
                raise ValueError(f"no donor shared between {args.bfile}.fam and --grm-npz")
            iids, dosage = iids[keep], dosage[keep]
    else:
        iids = npz_ids
    ph = pd.read_parquet(args.pheno)
    ph.index = ph.index.astype(str)
    ph = ph.reindex(iids)
    cols = [c for c in ph.columns if c.startswith("pheno_")]

    grm_rows = np.ones(len(iids), dtype=bool)
    if args.restrict_grm_to_cohort:
        if len(cols) != 1:
            raise ValueError(f"--restrict-grm-to-cohort requires exactly one pheno_* column, got {cols}")
        grm_rows = ph[cols[0]].notna().to_numpy()
        if not grm_rows.any():
            raise ValueError(f"no donor in {args.bfile}.fam has a non-missing phenotype")
    grm_iids = iids[grm_rows]
    Ks = []
    if args.bfile:
        masks = bin_masks(dosage.shape[1], args.maf_ld, breakpoints, args.bin_assignment)
        for i, mask in enumerate(masks, start=1):
            if not mask.any():
                raise ValueError(f"component {i} has no variants")
            K, m_used = build_grm(dosage[np.ix_(grm_rows, mask)])
            print(f"[info] {args.experiment}: GRM {i}/{len(masks)} from {m_used} variants "
                  f"over {len(grm_iids)} donors", flush=True)
            Ks.append(K)
        del dosage
    if npz_Ks:
        pos = {i: k for k, i in enumerate(npz_ids)}
        idx = [pos[i] for i in grm_iids]
        Ks.extend(K[np.ix_(idx, idx)] for K in npz_Ks)
    multi = binned or len(Ks) > 1
    n_geno = len(Ks) - len(npz_Ks) if (args.bfile and npz_Ks) else 0

    X_all = design_matrix(args.covariate_file, grm_iids)
    Xok = X_all.notna().all(axis=1).to_numpy()
    Y_all = ph[cols].to_numpy(float)[grm_rows]

    # columns sharing a missingness pattern share the kernels/folds: fit them together
    ok_all = ~np.isnan(Y_all) & Xok[:, None]
    groups = {}
    for j, col in enumerate(cols):
        groups.setdefault(ok_all[:, j].tobytes(), []).append(j)

    rows = [None] * len(cols)
    for key, js in groups.items():
        ok = ok_all[:, js[0]]
        sub = np.ix_(ok, ok)
        Kok = [K[sub] for K in Ks]
        r = he_fit(Kok, Y_all[np.ix_(ok, js)], X_all.to_numpy(float)[ok], args.jackknife == "loo")
        h2k, h2k_u = _h2(r["theta"], r["a"])                        # (k, P)
        if args.jackknife == "loo":
            h2k_l, h2k_ul = _h2(r["theta_loo"], r["a_loo"])         # (n, k, P)
        for c, j in enumerate(js):
            row = dict(experiment=args.experiment, phenotype=cols[j], method=method,
                       h2=float(h2k[:, c].sum()), h2_se=np.nan, true_h2=args.true_h2,
                       converged=bool(np.all(np.isfinite(r["theta"][:, c]))), at_boundary=False,
                       n_samples=int(ok.sum()), h2_unscaled=float(h2k_u[:, c].sum()),
                       h2_unscaled_se=np.nan)
            row.update({f"sigma_k{i}": float(t) for i, t in enumerate(r["theta"][:-1, c], start=1)})
            row["sigma_e"] = float(r["theta"][-1, c])
            if args.jackknife == "loo":
                row["h2_se"] = float(np.sqrt(jk_var(h2k_l[:, :, c].sum(axis=1))))
                row["h2_unscaled_se"] = float(np.sqrt(jk_var(h2k_ul[:, :, c].sum(axis=1))))
            if multi:
                for i in range(h2k.shape[0]):
                    row[f"h2_k{i+1}"] = float(h2k[i, c])
                    row[f"h2_k{i+1}_se"] = (float(np.sqrt(jk_var(h2k_l[:, i, c])))
                                            if args.jackknife == "loo" else np.nan)
            if n_geno:
                row["h2_genotype"] = float(h2k[:n_geno, c].sum())
                row["h2_genotype_se"] = (float(np.sqrt(jk_var(h2k_l[:, :n_geno, c].sum(axis=1))))
                                         if args.jackknife == "loo" else np.nan)
            rows[j] = row

    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"[done] {args.experiment} ({method}): mean h2={df.h2.mean():.4f} "
          f"(true {args.true_h2}) over {len(df)} phenotypes -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
