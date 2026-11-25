#!/usr/bin/env python
"""Concatenate one (gene, kernel) task's per-h2 exact_reml_heritability.py --grm-npz
TWO-kernel outputs (genotype GRM + embedding GRM, one parquet per true_h2 value, from
exact_from_grm_joint in embedding_heritability_simulation_study.nf) into
estimate_joint_genotype_embedding.py's own long format (one row per component per
true_h2/rep).

The wrapper scripts were called with `--grm-npz <genotype.npz> <embedding.npz>`, so by
that fixed argument order h2_k1/h2_k1_se is always the genotype component and
h2_k2/h2_k2_se the embedding component -- matching estimate_joint_genotype_embedding.py's
own (0, "genotype"), (1, "embedding") convention, including target_h2 (true_h2 for
genotype, always 0.0 for embedding, since the phenotype has no embedding-derived signal
by construction).

With a binned genotype part (exact_reml_heritability.py --bin-assignment + --grm-npz,
workflow 09's --joint_genotype_bins mac4) there are more than two components: the
genotype ones first, the embedding last. The genotype component is then read from
h2_genotype/h2_genotype_se (the sum of the genotype bins, SE from their covariance) and
the embedding from the last h2_k{i}.

--nonneg fits (exact_reml_heritability.py lrt_p/lrt_npz_p present) also get an lrt_p
column, each component row its own test: genotype = the joint LRT of every component
= 0 (lrt_p; a valid null test of genotype h2 = 0, since the embedding is null by
construction), embedding = the LRT of the embedding component alone against the
genotype-only fit (lrt_npz_p; valid at every true_h2).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquets", nargs="+", required=True,
                     help="one exact_reml_heritability.py output per true_h2 "
                          "value, each with h2_k1/h2_k1_se (genotype) and h2_k2/h2_k2_se "
                          "(embedding) columns")
    ap.add_argument("--kernel-label", required=True)
    ap.add_argument("--region", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(p) for p in args.parquets], ignore_index=True)
    df["rep"] = df["phenotype"].str.extract(r"pheno_(\d+)").astype(int)

    n_comp = max(int(c[4:]) for c in df.columns if c.startswith("h2_k") and c[4:].isdigit())
    if n_comp > 2:
        geno_cols, emb_cols = ("h2_genotype", "h2_genotype_se"), (f"h2_k{n_comp}", f"h2_k{n_comp}_se")
    else:
        geno_cols, emb_cols = ("h2_k1", "h2_k1_se"), ("h2_k2", "h2_k2_se")

    has_lrt = "lrt_p" in df.columns and "lrt_npz_p" in df.columns
    rows = []
    for _, r in df.iterrows():
        for comp_col, se_col, comp_name, target, p_col in (
            (*geno_cols, "genotype", r["true_h2"], "lrt_p"),
            (*emb_cols, "embedding", 0.0, "lrt_npz_p"),
        ):
            rows.append(dict(
                experiment=r["experiment"], region=args.region, kernel=args.kernel_label,
                component=comp_name, true_h2=r["true_h2"], target_h2=target,
                rep=r["rep"], n=np.nan, h2=r[comp_col], h2_se=r[se_col],
                sigma2=np.nan, converged=r["converged"],
                at_boundary=r["at_boundary"] if "at_boundary" in r.index else np.nan,
                estimator=r["method"]))
            if has_lrt:
                rows[-1]["lrt_p"] = r[p_col]

    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print(f"[done] {len(out)} rows ({df.experiment.iloc[0]}, {args.kernel_label}) -> "
          f"{args.out}", flush=True)


if __name__ == "__main__":
    main()
