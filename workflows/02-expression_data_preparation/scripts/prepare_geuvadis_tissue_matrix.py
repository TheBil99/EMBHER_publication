#!/usr/bin/env python
"""Cache GEUVADIS's gene-level TPM as a parquet (genes x donors) -- the GEUVADIS
counterpart of prepare_tissue_matrix.py's GTEx TPM cache, same output schema (index
name "gene_id", one column per donor), so the same prepare_expression_matrix.py and
resolve_expression.py downstream work unchanged for either cohort.

data/raw/GEUVADIS/GD660.GeneQuantRPKM.txt.gz is RPKM for all 660 GEUVADIS sequencing
runs, INCLUDING technical replicate runs (confirmed against E-GEUV-1.sdrf.txt's Assay
Name field: run columns are "<individual>.<lane>.<machine_date_index>", e.g.
"HG00096.1.M_111124_6").

RPKM is converted to TPM per sequencing run, on the full gene set, before anything else:
TPM_g = RPKM_g / sum_j(RPKM_j) * 1e6, summing over every gene of that run's column. The
462-unique-individual ("GD462") collapse then averages TPM across an individual's
technical replicate runs. TPM is the only output; there is no RPKM option.

No gene filtering and no transform here -- matching prepare_tissue_matrix.py, which
caches GTEx's raw TPM as-is with no gene-level QC either; both filtering and transform
are prepare_expression_matrix.py's job (quantile_int), downstream of this cache. GTEx's
tissue_matrix also restricts donors to those with phased WGS (only they can be
genotyped) -- --vcf-samples mirrors that restriction here, against the 1000G
high-coverage VCF's own sample panel (--vcf-samples is optional: omit it to keep every
collapsed individual with no genotype-availability filter).
"""
from __future__ import annotations

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rpkm", required=True, help="GD660.GeneQuantRPKM.txt.gz")
    ap.add_argument("--vcf-samples", default=None,
                     help="one donor ID per line (1000G VCF sample panel); omit for no filter")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.rpkm, sep="\t")
    meta_cols = ["TargetID", "Gene_Symbol", "Chr", "Coord"]
    run_cols = [c for c in df.columns if c not in meta_cols]
    print(f"[info] {len(df)} gene units x {len(run_cols)} sequencing runs", flush=True)

    # RPKM -> TPM per run (column), over the full gene set, before replicate averaging
    tpm = df[run_cols] / df[run_cols].sum(axis=0) * 1e6
    print(f"[info] converted RPKM to TPM per run (column sums now "
          f"{tpm.sum(axis=0).min():.0f}..{tpm.sum(axis=0).max():.0f})", flush=True)

    individuals = pd.Index([c.split(".")[0] for c in run_cols])
    expr = tpm.T
    expr.index = individuals
    expr = expr.groupby(level=0).mean()
    print(f"[info] collapsed to {expr.shape[0]} unique individuals (technical-replicate "
          f"TPM averaged)", flush=True)

    if args.vcf_samples:
        vcf_samples = {l.strip() for l in open(args.vcf_samples) if l.strip()}
        keep = [d for d in expr.index if d in vcf_samples]
        dropped = expr.shape[0] - len(keep)
        if dropped:
            print(f"[warn] {dropped} donors dropped for missing 1000G genotype", flush=True)
        expr = expr.loc[keep]

    out = expr.T   # genes x donors, matching prepare_tissue_matrix.py's schema
    out.index = df["TargetID"]
    out.index.name = "gene_id"
    out.to_parquet(args.out)
    print(f"[done] {out.shape[0]} genes x {out.shape[1]} donors -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
