#!/usr/bin/env python
"""table_s6: rare-variant heritability of the Borzoi embedding, per gene (workflow 16,
gtex/hi_q_gtex_genes_maf_le0.01; HE regression, real phenotype). Genotype rare h2 is in table_s1.

Workflow 16's embedding arm fits the embedding kernel jointly with the four common-variant
genotype bins; the two rarest genotype bins are left out. Columns:
  ensg_id, gene_name
  h2_rare_embedding               h2 of the embedding kernel (the rare-variant embedding h2)
  h2_common_genotype_embedding_arm  sum of the four common-variant genotype bins in that fit
  h2_MAF_0.01_0.1_low_LD / _high_LD, h2_MAF_0.1_0.5_low_LD / _high_LD
                                  the four common bins separately (workflow 16's h2_emb_arm_k1..k4)
Reads summary/per_gene_real_estimates.parquet (one row per gene, real phenotype = perm 0);
nothing is recomputed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RENAME = {
    "experiment": "ensg_id",
    "gene_name": "gene_name",
    "h2_rare_embedding": "h2_rare_embedding",
    "h2_common_genotype_embedding_arm": "h2_common_genotype_embedding_arm",
    "h2_emb_arm_k1": "h2_MAF_0.01_0.1_low_LD",
    "h2_emb_arm_k2": "h2_MAF_0.01_0.1_high_LD",
    "h2_emb_arm_k3": "h2_MAF_0.1_0.5_low_LD",
    "h2_emb_arm_k4": "h2_MAF_0.1_0.5_high_LD",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-estimates", required=True,
                    help="results/16-rare_heritability_permutations/gtex/hi_q_gtex_genes_maf_le0.01/"
                         "summary/per_gene_real_estimates.parquet")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    d = pd.read_parquet(args.real_estimates)
    if "experiment" not in d.columns:          # the Ensembl id is the index
        d = d.rename_axis("experiment").reset_index()
    if "perm" in d.columns:
        d = d[d["perm"] == 0]
    t = d[list(RENAME)].rename(columns=RENAME).sort_values("ensg_id")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / "table_s6.tsv", sep="\t", index=False)
    print(f"[done] {len(t)} genes -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
