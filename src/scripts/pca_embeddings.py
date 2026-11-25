#!/usr/bin/env python
"""Whiten a per-donor embedding onto its top-k principal components.

Why this exists
---------------
`build_grm.py` forms K = Z Z' / M from column-standardised features. That is the
right construction when the columns are close to independent (genotype dosages,
where LD decays), but embedding columns are massively redundant -- 701 RNA tracks
all read the same locus, 6144 channel values are smooth functions of one another.
Column standardisation is a MARGINAL normalisation: it equalises each column's
variance and does nothing to the correlations between columns. The result is a
kernel whose eigenvalue spectrum can span 5+ orders of magnitude.

Single-component REML then fails, not subtly. In K's eigenbasis the model is
    ytilde_i^2 ~ sigma_g^2 * lambda_i + sigma_e^2
i.e. a STRAIGHT LINE through 800 points (lambda_i, ytilde_i^2). A 2-parameter line
cannot cover a lambda range that wide, and the Gaussian likelihood's cost is
asymmetric -- under-predicting a variance costs ytilde^2/v (blows up linearly),
over-predicting costs only log(v) (nearly free). REML therefore inflates sigma_g^2
until the low-lambda bulk fits and eats the cheap log penalty on the top
directions, and h2 is pushed toward 1 mechanically.

Whitening fixes it by construction. Keeping k components with their variances
equalised makes K = (n/k) U_k U_k', a scaled projection with only TWO distinct
eigenvalues. The model is then SATURATED -- two parameters, two distinct lambdas --
so there is no line to extrapolate and nothing left to misspecify. It reduces to a
two-group variance contrast (inside the subspace vs outside), and the resulting h2
equals the df-adjusted R2 of y on those k directions.

Note this is NOT the same as truncating K while keeping its eigenvalues (what
build_grm.py's --min-eigval-ratio does): that leaves the spectrum, and so the problem,
essentially unchanged. The eigenvalue flattening is what matters, not the truncation.

What it does
------------
    embedding
      -> subset to the analysis donors (so the basis is built on exactly the
         individuals the GRM and REML will use)
      -> standardise each ORIGINAL feature column (mean 0, unit variance)
      -> SVD, keep the top k components
      -> rescale each component to unit variance   <- the whitening
      -> write PC scores parquet

The first standardisation makes the PCA operate on the correlation rather than the
covariance matrix, so components are not chosen by whichever Borzoi channels happen
to have the largest activation magnitude (those scales are an artefact of the
network, not a statement of importance). It can move the retained subspace noticeably
but changes h2 little; the second standardisation is the load-bearing step.

Output columns are already unit-variance, so build_grm.py's own column
standardisation is a no-op on them -- deliberately, so this script's correctness
does not depend on that behaviour of a shared script.

k is chosen via --n-pcs (an absolute count) or --pc-frac (a fraction of the analysis
cohort's numerical rank, k = round(pc_frac * rank)). --pc-frac exists because a fixed
absolute k silently stops meaning "most of the variance" if the cohort shrinks -- e.g.
workflow 09's original 944-donor panel vs its 800-donor 04b panel, where the same
n_pcs=800 went from leaving 143 residual directions to leaving 0. Since rank is
computed AFTER the --pheno subset below, --pc-frac is always relative to whatever
cohort is ACTUALLY being analyzed, never the full embedding panel -- the same
guarantee holds when a real phenotype's cohort is smaller than the panel with
embeddings.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from embher.embedding.whiten import whiten_embeddings


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", nargs="+", required=True,
                    help="one or more embeddings parquet(s) (shards/panels are concatenated)")
    ap.add_argument("--pheno", default=None,
                    help="optional pheno.parquet; the PCA basis is built ONLY on its "
                         "individuals, matching build_grm.py's own subsetting so the "
                         "components are orthonormal on exactly the analysis set")
    ap.add_argument("--n-pcs", type=int, default=None,
                    help="absolute number of principal components to retain (k). "
                         "Exactly one of --n-pcs / --pc-frac is required.")
    ap.add_argument("--pc-frac", type=float, default=None,
                    help="retain this FRACTION of the analysis cohort's numerical rank "
                         "instead of a fixed k (k = round(pc_frac * rank)) -- panel-size "
                         "agnostic, unlike --n-pcs, which silently stops meaning "
                         "'most of the variance' if the cohort shrinks (see module "
                         "docstring). Because rank is computed AFTER the --pheno subset "
                         "below, this is always a fraction of the ANALYSIS cohort's rank, "
                         "never the full embedding panel's -- exactly what matters when "
                         "a real phenotype's cohort is smaller than the embedding panel. "
                         "Exactly one of --n-pcs / --pc-frac is required.")
    ap.add_argument("--out", required=True, help="output parquet path")
    args = ap.parse_args()

    if (args.n_pcs is None) == (args.pc_frac is None):
        raise SystemExit("[error] exactly one of --n-pcs / --pc-frac must be given")

    df = pd.concat([pd.read_parquet(p) for p in args.embeddings], axis=0)
    df.index = df.index.astype(str)
    df = df[~df.index.duplicated(keep="first")]

    if args.pheno:
        pid = set(pd.read_parquet(args.pheno).index.astype(str))
        keep_ids = [i for i in df.index if i in pid]
        print(f"[info] subset to {len(keep_ids)} phenotyped individuals "
              f"({len(pid) - len(keep_ids)} in pheno lacked embeddings)", flush=True)
        df = df.loc[keep_ids]

    ids = df.index.tolist()
    Z = df.to_numpy(dtype=np.float64)
    n = Z.shape[0]

    scores = whiten_embeddings(Z, n_pcs=args.n_pcs, pc_frac=args.pc_frac)
    k = scores.shape[1]

    out = pd.DataFrame(scores, index=ids,
                       columns=[f"pc{i + 1:04d}" for i in range(k)])
    out.index.name = df.index.name
    out.to_parquet(args.out)
    print(f"[done] {out.shape[0]} donors x {k} whitened PCs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
