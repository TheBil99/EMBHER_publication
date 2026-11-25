#!/usr/bin/env python
"""Extract the first --n-pcs genotype PCs for GEUVADIS donors from a third-party 1000G
high-coverage ancestry PCA panel (data/raw/GenomiAncestryPanel/, github.com/
exon-research/genomi-ancestry-panel) -- used in place of a 1000G/IGSR-published PC
file, since neither 1000G nor IGSR publishes one (unlike GTEx, which ships its own
20_Genotype_PCs.eigenvec.txt.gz). This panel's reference_scores.tsv was built from the
same 1000G high-coverage phased VCFs this project already uses for GEUVADIS genotypes
(see its own manifest.json), and covers every GEUVADIS donor this project's own
1000G-VCF genotype-availability filter keeps (confirmed: this panel is missing exactly
the same 13 GD462 donors workflows/02-expression_data_preparation's geuvadis branch
already drops for lacking 1000G coverage)."""
from __future__ import annotations

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-scores", required=True, help="panel's reference_scores.tsv")
    ap.add_argument("--samples", required=True, help="GEUVADIS donor IDs, one per line")
    ap.add_argument("--n-pcs", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    samples = [l.strip() for l in open(args.samples) if l.strip()]
    pc_cols = [f"PC{i + 1}" for i in range(args.n_pcs)]

    scores = pd.read_csv(args.reference_scores, sep="\t", index_col="sample_id")
    present = [s for s in samples if s in scores.index]
    dropped = [s for s in samples if s not in scores.index]
    if dropped:
        print(f"[warn] {len(dropped)}/{len(samples)} donors have no genotype PCs in this panel: "
              f"{dropped}", flush=True)

    out = scores.loc[present, pc_cols].copy()
    out.index.name = "IID"
    out.to_csv(args.out, sep="\t")
    print(f"[done] {out.shape[0]} donors x {out.shape[1]} genotype PCs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
