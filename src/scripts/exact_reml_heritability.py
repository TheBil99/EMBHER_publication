#!/usr/bin/env python
"""Heritability by exact multi-component AI-REML (numpy) on genotype GRMs and/or
prebuilt kernels (--grm-npz), covariates as REML fixed effects.

Inputs: --bfile, --pheno, --covariate-file, --maf-bins/--maf-ld or --bin-assignment,
--restrict-grm-to-cohort. Output parquet columns: method, h2, h2_se, true_h2, converged,
h2_k{i}/h2_k{i}_se, at_boundary, logLL, n_iter,
n_starts_converged, n_samples, lam_ratio and valid (see Validity below), and with
--nonneg also lrt and lrt_p (and lrt_npz, lrt_npz_p, converged_genotype_only in joint
--bfile + --grm-npz mode).

Model:  y = X b + sum_k g_k + e,  g_k ~ N(0, s_k K_k),  e ~ N(0, s_e I).
X is an intercept plus every --covariate-file column left after the same VIF<=10
collinearity filter he_heritability.py applies, so both estimators fit the
identical design.

GRM (one per bin): K = Z Z' / m, Z_ij = (x_ij - 2p_j) / sqrt(2 p_j (1 - p_j)), p_j
estimated on the donors the GRM is built over, monomorphic variants dropped -- the
standardized GCTA-model (alpha=-1) GRM. A missing genotype call is mean-imputed (z=0)
and left out of p (counting it as a heterozygote would turn a missing call on a rare
variant into a fake near-singleton, z ~ sqrt(n), and shift p for every donor).

Fit: AI-REML from three starting points (h2 split evenly across components at 0.5,
0.1 and 0.9 of the covariate-adjusted phenotypic variance). Each step solves the
average-information system for the score, then halves the step until V stays
positive definite and the REML log-likelihood does not decrease. A run is converged
only when the Newton-PREDICTED log-likelihood gain at the current point falls below
--tol -- not when the realized gain does, which is how a stalled optimizer creeping
along with tiny accepted steps would otherwise pass. The
reported fit is the highest-likelihood converged start (or, if none converged, the
highest-likelihood start, flagged converged=False). Variance components are
unconstrained (may go negative).

Boundary: unconstrained, a negative component can push V toward singularity (its
smallest eigenvalue -> 0), and for some gene/bin designs the REML likelihood keeps
rising all the way to that edge -- there is then no interior maximum to converge to.
Such a fit stops at the last positive-definite
point and is reported with converged=False, at_boundary=True, rather than forced
anywhere.

--grm-npz: prebuilt kernels (build_grm.py's .npz: K + ids), e.g. embedding GRMs,
instead of or together with --bfile. One npz ->
single-component fit; several -> joint fit, their donors intersected onto the first
file's order. With --bfile too, the genotype component(s) are built from the bfile
exactly as without --grm-npz, over the donors present in both, and always occupy the
leading h2_k{i} slots, the npz component(s) the trailing ones (the fixed order
workflow 09's combine_grm_joint.py relies on). h2_k{i} columns are written whenever
there is more than one component or the genotype GRM is binned. In that joint mode the
genotype part as a whole is also written as h2_genotype/h2_genotype_se: the sum of
the genotype components, SE from their full sampling covariance (e.g. a 4-bin
genotype GRM + an embedding kernel).

h2_k = s_k * mean(diag K_k) / (sum_j s_j * mean(diag K_j) + s_e): each component's
share of total variance. mean(diag K) is ~1 for a standardized GRM, so this is
s_k / sum of s up to HWE deviations and missingness. h2 = sum_k h2_k.
SEs: inverse average-information matrix at the optimum, delta method for h2_k/h2.

Validity: lam_ratio = lambda_min / lambda_max of V at the reported fit, and valid =
converged and lam_ratio >= VALID_LAM_RATIO (1e-5), both written for every fit. With two
or more unconstrained kernels the REML likelihood can rise without bound toward a
singular V; a fit can then pass the convergence test at a numerically singular V (AI
indefinite, h2_se ~ 0) while at_boundary is only checked for non-converged starts.
Such fits have lam_ratio ~ 1e-16, ordinary fits ~ 1e-3 to 0.5, so valid separates them.

--nonneg: every kernel's variance component constrained >= 0 (projected AI-Newton,
same three starts and --tol; a component the likelihood pushes below 0 is held at 0).
V >= s_e I then, so the fit cannot reach a singular V: at_boundary is always False.
h2_se/h2_k_se cover only the components not held at 0, and are a poor basis for a test
of h2 = 0 at the constraint; instead lrt = 2 x (REML log-likelihood - that of the
no-kernel model) and lrt_p, its p-value under the chi-bar-squared null (mixture of
chi2_0..chi2_k with weights by Monte Carlo from the null information, seed 1) are
written. Negative estimates are truncated at 0, so the fit is biased slightly upward
at h2 = 0.
In joint --bfile + --grm-npz mode lrt tests every component (genotype and npz) = 0. To
test the npz component(s) alone, the genotype-only model (same donors, genotype
components and fixed effects, also >= 0) is fitted too, and lrt_npz = 2 x (joint REML
log-likelihood - genotype-only one) and lrt_npz_p are written. Its null is
chi-bar-squared over the npz components: 1/2 chi2_0 + 1/2 chi2_1 for one npz kernel,
otherwise weights by Monte Carlo from the npz components' efficient information at the
genotype-only fit (npz_information). lrt_npz is set to exactly 0 when every npz
component is held at 0 in the joint fit (the two fits otherwise differ by --tol noise).
This doubles the cost of each phenotype.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from embher.heritability.io import read_bed
from embher.heritability.reml import (VALID_LAM_RATIO, bin_masks, build_grm, chibar_p, chibar_weights,
                        design_matrix, fit_reml, load_npz_kernels, npz_information,
                        null_information)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfile", default=None,
                    help="region genotypes PLINK prefix; required unless --grm-npz is given")
    ap.add_argument("--grm-npz", nargs="+", default=None,
                    help="one or more prebuilt GRM .npz files (build_grm.py: K + ids), fit "
                         "alone or jointly with --bfile's genotype GRM(s) -- see module docstring")
    ap.add_argument("--pheno", required=True, help="pheno.parquet (index IID, cols pheno_*)")
    ap.add_argument("--covariate-file", default=None,
                    help="optional TSV of fixed-effect covariates indexed by IID (e.g. "
                         "build_gtex_eqtl_covar.py's output); an intercept is always fit")
    ap.add_argument("--maf-bins", default=None,
                    help="comma-separated ascending MAF breakpoints, bins (lo, hi]; "
                         "omit for one whole-region GRM")
    ap.add_argument("--maf-ld", default=None,
                    help="file with a MAF column in bim order (e.g. workflow 04's maf.txt); "
                         "required with --maf-bins")
    ap.add_argument("--bin-assignment", default=None,
                    help="one integer per bim variant, 0 = excluded from every bin; "
                         "exclusive with --maf-bins")
    ap.add_argument("--restrict-grm-to-cohort", action="store_true",
                    help="estimate the GRM's allele frequencies on only the donors with a "
                         "non-missing phenotype (requires exactly one pheno_* column), "
                         "instead of every donor in --bfile")
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="convergence: Newton-predicted REML log-likelihood gain below this")
    ap.add_argument("--nonneg", action="store_true",
                    help="constrain every kernel's variance component to >= 0 and add the "
                         "LRT against the no-kernel model (lrt, lrt_p); with --bfile + "
                         "--grm-npz also the LRT of the npz component(s) alone (lrt_npz, "
                         "lrt_npz_p) -- see module docstring")
    ap.add_argument("--experiment", default="")
    ap.add_argument("--true-h2", type=float, default=np.nan)
    ap.add_argument("--method", default=None,
                    help="label for the output 'method' column (default: exact_reml_single_grm / "
                         "exact_reml_maf_bins / exact_reml_bin_assignment / exact_reml_from_grm / "
                         "exact_reml_joint_from_grm / exact_reml_joint_genotype_grm)")
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
        default_method = "exact_reml_joint_genotype_grm"
    elif args.grm_npz:
        default_method = "exact_reml_joint_from_grm" if len(args.grm_npz) > 1 else "exact_reml_from_grm"
    else:
        default_method = ("exact_reml_maf_bins" if breakpoints else
                          "exact_reml_bin_assignment" if args.bin_assignment else
                          "exact_reml_single_grm")
    if args.nonneg:
        default_method = default_method.replace("exact_reml_", "exact_reml_nonneg_", 1)
    method = args.method or default_method

    npz_Ks = []
    if args.grm_npz:
        npz_ids, npz_Ks = load_npz_kernels(args.grm_npz)
    if args.bfile:
        iids, dosage = read_bed(args.bfile)
        if args.grm_npz:
            # donors in both, in the bfile's own order;
            # the genotype GRM is standardized over exactly these
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
    rows = []
    chibar_cache = {}
    for col in cols:
        y_all = ph[col].to_numpy()[grm_rows]
        ok = ~np.isnan(y_all) & X_all.notna().all(axis=1).to_numpy()
        sub = np.ix_(ok, ok)
        r = fit_reml([K[sub] for K in Ks], y_all[ok], X_all.to_numpy(float)[ok],
                     args.max_iter, args.tol, nonneg=args.nonneg)
        row = dict(experiment=args.experiment, phenotype=col, method=method, h2=r["h2"],
                   h2_se=r["h2_se"], true_h2=args.true_h2, converged=r["converged"],
                   at_boundary=r["at_boundary"], logLL=r["logLL"], n_iter=r["n_iter"],
                   n_starts_converged=r["n_starts_converged"], n_samples=int(ok.sum()),
                   lam_ratio=r["lam_ratio"],
                   valid=bool(r["converged"] and r["lam_ratio"] >= VALID_LAM_RATIO))
        if args.nonneg:
            # chi-bar-squared weights depend only on the kernels/fixed effects of this
            # donor set: computed once per distinct set of donors
            key = ok.tobytes()
            if key not in chibar_cache:
                chibar_cache[key] = chibar_weights(null_information(
                    [K[sub] for K in Ks], X_all.to_numpy(float)[ok]))
            row["lrt"] = r["lrt"]
            row["lrt_p"] = chibar_p(r["lrt"], chibar_cache[key])
            if n_geno:
                # LRT of the npz component(s) alone: against the genotype-only fit on
                # the same donors and fixed effects
                X_ok = X_all.to_numpy(float)[ok]
                r0 = fit_reml([K[sub] for K in Ks[:n_geno]], y_all[ok], X_ok,
                              args.max_iter, args.tol, nonneg=True)
                n_npz = len(Ks) - n_geno
                if n_npz == 1:
                    w = np.array([0.5, 0.5])
                else:
                    theta0 = np.r_[r0["sigma"][:-1], np.zeros(n_npz), r0["sigma"][-1]]
                    w = chibar_weights(npz_information(
                        theta0, [K[sub] for K in Ks], X_ok, range(n_geno, len(Ks))))
                # every npz component held at 0: the joint optimum is in the null model,
                # so T = 0 exactly (the two fits otherwise differ by up to --tol noise)
                at0 = all(h == 0 for h in r["h2_k"][n_geno:])
                row["lrt_npz"] = 0.0 if at0 else float(2 * (r["logLL"] - r0["logLL"]))
                row["lrt_npz_p"] = chibar_p(row["lrt_npz"], w)
                row["converged_genotype_only"] = r0["converged"]
        if multi:
            for i, (kh, kse) in enumerate(zip(r["h2_k"], r["h2_k_se"]), start=1):
                row[f"h2_k{i}"], row[f"h2_k{i}_se"] = kh, kse
        if n_geno:
            # joint --bfile + --grm-npz: the genotype part as a whole (sum of its
            # components, SE from their full sampling covariance)
            c = r["h2_k_cov"][:n_geno, :n_geno]
            row["h2_genotype"] = float(np.sum(r["h2_k"][:n_geno]))
            row["h2_genotype_se"] = float(np.sqrt(max(c.sum(), 0.0)))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    n_unconverged = int((~df.converged).sum())
    if n_unconverged:
        print(f"[warn] {args.experiment} ({method}): {n_unconverged}/{len(df)} "
              f"phenotypes did NOT converge", flush=True)
    print(f"[done] {args.experiment} ({method}): mean h2={df.h2.mean():.4f} "
          f"(true {args.true_h2}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
