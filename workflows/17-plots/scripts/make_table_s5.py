#!/usr/bin/env python
"""table_s5: gene-level inputs and Fig. 3c summary statistics of the attribution-seqlet enrichments,
pretrained vs random-init Borzoi, per layer (workflow 15, hi_q_random500).

Reads workflow 15's already-published per-gene tables (summary/<db>/layers_per_gene.parquet;
layers_normalized_per_gene.parquet for cisbp_rna, the seqlet-count-adjusted counts Fig. 3c uses) and
the pretrained_vs_random_adjusted-<dbs>_summary.parquet behind Fig. 3c. Nothing is computed for the table:
gene-level columns are the stored per-gene values; summary_* columns are copied from the summary parquet
and repeat on every row of that database and layer.

One row per gene x database x layer (24 layers; the expression target is not in the Fig. 3c summary and
is left out). Columns:
  ensg_id, gene_name, database, feature, layer_index, layer
  hocomoco_v13   n_enriched_{pretrained,random_init}
  cisbp_rna      n_enriched_adjusted_*  (used in Fig. 3c), n_enriched_raw_*, n_seqlets_*, adjustment_slope
  phylop241, gnomad   observed_*, expected_*, n_positions_*
                 (phylop241: summed PhyloP of seqlet positions, summed context-expected PhyloP, positions;
                  gnomad: positions seen in gnomAD, expected seen, positions)
  summary_pretrained, summary_random_init, summary_delta (= pretrained - random init, the y axis of Fig. 3c),
  summary_ci_low, summary_ci_high (95% bootstrap over genes), n_genes
Columns that do not apply to a database, and genes without a row for a target in the source table, are empty.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DBS = ["cisbp_rna", "hocomoco_v13", "phylop241", "gnomad"]
FEATURE = {"cisbp_rna": "RBP motifs (CisBP-RNA)", "hocomoco_v13": "TF motifs (HOCOMOCO v13)",
           "phylop241": "PhyloP (241-way)", "gnomad": "gnomAD"}
ROLE = {"reference": "pretrained", "random_init": "random_init"}
GENE_COLS = {   # source column -> table column stem (suffixed _pretrained / _random_init)
    "hocomoco_v13": {"n_enriched": "n_enriched"},
    "cisbp_rna": {"n_enriched": "n_enriched_adjusted", "n_enriched_raw": "n_enriched_raw",
                  "n_seqlets": "n_seqlets"},
    "phylop241": {"observed": "observed", "expected": "expected", "n_positions": "n_positions"},
    "gnomad": {"observed": "observed", "expected": "expected", "n_positions": "n_positions"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-dir", required=True, help="results/15-borzoi_attribution/<run>/summary")
    ap.add_argument("--summary-parquet", required=True,
                    help="<summary-dir>/pretrained_vs_random_adjusted-<dbs>_summary.parquet")
    ap.add_argument("--gene-map", required=True,
                    help="workflows/11-embedding_heritability/configs/hi_q_random500.tsv")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    summ = pd.read_parquet(args.summary_parquet)
    summ = summ.rename(columns={"db": "database", "label": "layer", "pretrained": "summary_pretrained",
                                "random_init": "summary_random_init", "delta": "summary_delta",
                                "ci_low": "summary_ci_low", "ci_high": "summary_ci_high"})
    summ = summ[["database", "layer_index", "layer", "summary_pretrained", "summary_random_init",
                 "summary_delta", "summary_ci_low", "summary_ci_high", "n_genes"]]

    gm = pd.read_csv(args.gene_map, sep="\t", dtype=str)
    gm["ensg_id"] = gm["gene_id"].str.split(".").str[0]
    info = gm.drop_duplicates("experiment").set_index("experiment")[["ensg_id", "gene_name"]]

    parts = []
    for db in DBS:
        fname = "layers_normalized_per_gene.parquet" if db == "cisbp_rna" else "layers_per_gene.parquet"
        d = pd.read_parquet(Path(args.summary_dir) / db / fname)
        d = d[d["label"].isin(summ.loc[summ["database"] == db, "layer"])]      # 24 layers, no expression
        cols = GENE_COLS[db]
        wide = d.pivot(index=["gene", "label"], columns="role", values=list(cols))
        wide.columns = [f"{cols[v]}_{ROLE[r]}" for v, r in wide.columns]
        wide = wide.reset_index().rename(columns={"label": "layer"})
        if db == "cisbp_rna":                                                   # one slope per layer, shared by roles
            slope = d.groupby("label")["slope"].first().rename("adjustment_slope")
            wide["adjustment_slope"] = wide["layer"].map(slope)
        # full gene x layer grid: a gene without a row for a layer stays empty instead of dropping out
        genes = sorted(gm["experiment"].unique())
        grid = pd.MultiIndex.from_product([genes, summ.loc[summ["database"] == db, "layer"]],
                                          names=["gene", "layer"]).to_frame(index=False)
        wide = grid.merge(wide, on=["gene", "layer"], how="left")
        wide.insert(0, "database", db)
        wide.insert(1, "feature", FEATURE[db])
        parts.append(wide)
    t = pd.concat(parts, ignore_index=True)
    t = t.merge(summ, on=["database", "layer"], how="left")
    t.insert(0, "gene_name", t["gene"].map(info["gene_name"]))
    t.insert(0, "ensg_id", t["gene"].map(info["ensg_id"]))
    t = t.drop(columns="gene")

    id_cols = ["ensg_id", "gene_name", "database", "feature", "layer_index", "layer"]
    gene_cols = [c for c in t.columns if c not in id_cols and not c.startswith("summary_") and c != "n_genes"]
    t = t[id_cols + gene_cols + [c for c in t.columns if c.startswith("summary_")] + ["n_genes"]]
    t["database"] = pd.Categorical(t["database"], DBS, ordered=True)
    t = t.sort_values(["database", "layer_index", "ensg_id"]).reset_index(drop=True)
    t["database"] = t["database"].astype(str)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / "table_s5.tsv.gz", sep="\t", index=False, compression="gzip")
    print(f"[done] {len(t)} rows x {t.shape[1]} columns; missing gene-level rows per database:", flush=True)
    key = {"cisbp_rna": "n_enriched_adjusted_pretrained", "hocomoco_v13": "n_enriched_pretrained",
           "phylop241": "observed_pretrained", "gnomad": "observed_pretrained"}
    for db in DBS:
        s = t[t["database"] == db]
        print(f"  {db}: {int(s[key[db]].isna().sum())} of {len(s)}", flush=True)


if __name__ == "__main__":
    main()
