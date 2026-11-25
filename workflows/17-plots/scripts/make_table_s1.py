#!/usr/bin/env python
"""table_s1: genotype (HE, mac_ld_6) heritability of every gene of workflow 10's
gtex/hi_q_gtex_genes run -- total and per component -- plus whether the gene is in workflow 03's
hi_q_random500 panel.

One row per gene (one <ENSG>_he_mac_ld_6.parquet each). Columns:
  ensg_id, gene_name            Ensembl id; gene symbol from the hi_q_gtex_genes panel TSV
  h2_genotype                   total genotype h2 (the fit's `h2`, sum of the components)
  h2_<bin> x6                   h2 of each of the six variant bins (k1..k6, split_bins_by_ld.py's
                                mac_ld_6: MAC <= 5 | MAC > 5 & MAF <= 0.01 | MAF (0.01,0.1] low / high
                                LD | MAF (0.1,0.5] low / high LD)
  in_random_gene_set            True if the gene is in the hi_q_random500 panel TSV
Reads the published per-gene parquets and the two panel TSVs; nothing is recomputed.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

COMPONENTS = {
    "h2_k1": "h2_MAC_le5",
    "h2_k2": "h2_MAC_gt5_MAF_le0.01",
    "h2_k3": "h2_MAF_0.01_0.1_low_LD",
    "h2_k4": "h2_MAF_0.01_0.1_high_LD",
    "h2_k5": "h2_MAF_0.1_0.5_low_LD",
    "h2_k6": "h2_MAF_0.1_0.5_high_LD",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genotype-h2-dir", required=True,
                    help="results/10-genotype_heritability/gtex/hi_q_gtex_genes/h2/he_mac_ld_6")
    ap.add_argument("--genotype-method", default="he_mac_ld_6_covar")
    ap.add_argument("--panel", required=True,
                    help="results/03-gene_panels/panels/hi_q_gtex_genes.tsv (gene names)")
    ap.add_argument("--in-panel", required=True,
                    help="results/03-gene_panels/panels/hi_q_random500.tsv (defines in_random_gene_set)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.genotype_h2_dir}/*.parquet"))
    if not files:
        raise SystemExit(f"[error] no parquets under {args.genotype_h2_dir}")
    cols = ["method", "h2", *COMPONENTS]
    rows = []
    for f in files:
        d = pd.read_parquet(f, columns=cols)
        d = d[d["method"] == args.genotype_method]
        if d.empty:
            continue
        r = d.iloc[0]
        rows.append({"ensg_id": Path(f).name.split("_")[0], "h2_genotype": r["h2"],
                     **{new: r[old] for old, new in COMPONENTS.items()}})
    t = pd.DataFrame(rows)

    panel = pd.read_csv(args.panel, sep="\t", dtype=str)
    names = dict(zip(panel["gene_id"], panel["gene_symbol"]))
    t.insert(1, "gene_name", t["ensg_id"].map(names))
    in_panel = pd.read_csv(args.in_panel, sep="\t", dtype=str)
    t["in_random_gene_set"] = t["ensg_id"].isin(set(in_panel["gene_id"].str.split(".").str[0]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / "table_s1.tsv", sep="\t", index=False)
    print(f"[done] {len(t)} genes ({int(t['in_random_gene_set'].sum())} in the in-panel set of {len(in_panel)} genes)"
          f" -> {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
