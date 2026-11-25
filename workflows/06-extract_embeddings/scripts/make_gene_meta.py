#!/usr/bin/env python3
"""
Build an embher-capture gene metadata TSV for a fixed gene list.

embher-capture -> load_gene_metadata() expects gene_id, chr_hg38, tss_hg38,
start_hg38, end_hg38 (see embher-prepare-gene-meta for the
chromosome-scan version this is adapted from). Here the gene set is not a
chromosome scan but a fixed list -- gene_symbol/gene_id pairs from a gene panel
(e.g. paper/genes/hi_q_random500.tsv) --
so coordinates are looked up by ENSG rather than filtered by chromosome.

Built offline, not by a live Nextflow process.

Usage:
    python make_gene_meta.py \
        --genes-tsv paper/genes/hi_q_random500.tsv \
        --out results/03-gene_panels/gene_meta/hi_q_random500.tsv
"""

import argparse
from pathlib import Path

import pandas as pd

# GENCODE v46 parquet made by workflow 01 (gencode_gtf_to_parquet), the one embher-capture reads
GENCODE_PARQUET = (
    'data/raw/GENCODE/human/release_46/'
    'gencode.v46.annotation.gtf.parquet'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--genes-tsv', required=True,
                    help='TSV with gene_symbol, gene_id columns (versioned ENSG)')
    p.add_argument('--gencode', default=GENCODE_PARQUET)
    p.add_argument('--out', required=True)
    return p.parse_args()


def main():
    args = parse_args()

    wanted = pd.read_csv(args.genes_tsv, sep='\t')
    wanted_ensg = wanted['gene_id'].str.split('.').str[0]

    df = pd.read_parquet(
        args.gencode,
        columns=['Chromosome', 'Feature', 'Start', 'End', 'Strand', 'gene_id'],
    )
    genes = df[(df['Feature'] == 'gene') & (df['gene_id'].str.split('.').str[0].isin(wanted_ensg))].copy()
    genes['gene_id'] = genes['gene_id'].str.split('.').str[0]

    # The parquet is pyranges-style: Start is already 0-based half-open, so it needs
    # no GTF-style -1. TSS is the 5' end: Start itself on the plus strand, but on the
    # minus strand it's the last base of the interval, i.e. End - 1 -- End alone is the
    # half-open exclusive bound (one past the last base), not a coordinate on the gene.
    genes['tss_hg38'] = genes['Start'].where(genes['Strand'] == '+', genes['End'] - 1)
    genes['chr_hg38'] = genes['Chromosome'].str.removeprefix('chr')
    genes = genes.rename(columns={'Start': 'start_hg38', 'End': 'end_hg38'})

    genes = genes.merge(wanted.assign(gene_id=wanted_ensg), on='gene_id', how='inner',
                         suffixes=('', '_wanted'))

    missing = set(wanted_ensg) - set(genes['gene_id'])
    if missing:
        raise SystemExit(f"{len(missing)} gene(s) from {args.genes_tsv} not found in "
                          f"{args.gencode}: {sorted(missing)}")

    out_cols = ['gene_id', 'gene_symbol', 'chr_hg38', 'start_hg38', 'end_hg38', 'tss_hg38']
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    genes[out_cols].sort_values('gene_id').to_csv(out_path, sep='\t', index=False)

    print(f"Wrote {len(genes)} genes to {out_path}")
    print(f"  chromosomes: {sorted(genes['chr_hg38'].unique())}")


if __name__ == '__main__':
    main()
