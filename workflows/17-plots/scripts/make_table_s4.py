#!/usr/bin/env python
"""table_s4: EUR vs AFR ancestry prediction R^2 per gene -- genotype and every Borzoi embedding /
output head (workflow 13, hi_q_random500; OLS on the top 10 PCs, no held-out split).

Reads workflow 13's already-published ancestry_prediction_summary.tsv; nothing is recomputed.
One row per gene:
  ensg_id, gene_name           from the run's gene map (workflow 11 config TSV)
  n_donors, n_afr              donors in the fit and how many are AFR
  genotype_r2                  R^2 of the gene's genotype PCs
  <layer>_r2                   R^2 of the unwhitened Borzoi embedding of that layer, pooled over the
                               position axis (per-channel embedding); layers in forward-pass order
  rna_predictions_r2, rna_predictions_whole_blood_r2
                               the same for the output heads (all RNA-seq tracks / whole-blood tracks)
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

HEADS = {"rna_predictions": 1, "rna_predictions_whole_blood": 2}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-tsv", required=True,
                    help="results/13-ancestry_prediction/hi_q_random500/results/"
                         "ancestry_prediction_summary.tsv")
    ap.add_argument("--gene-map", required=True,
                    help="workflows/11-embedding_heritability/configs/hi_q_random500.tsv")
    ap.add_argument("--ratio-script", required=True,
                    help="workflows/17-plots/scripts/summarize_embedding_genotype_ratio.py "
                         "(imported by path for layer_sort_key)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ratio_mod = _load_module("ratio_mod", Path(args.ratio_script))
    res = pd.read_csv(args.summary_tsv, sep="\t")

    geno = res[res["rep"] == "genotype"].set_index("gene")
    # unwhitened embeddings only; position-axis layers and the output heads (no channel axis)
    emb = res[(res["rep"] == "raw") & ~res["variant"].str.endswith("_chan")].copy()
    emb["variant"] = emb["variant"].str.replace(r"_pos$", "", regex=True)
    emb["col"] = emb["variant"] + "_r2"
    wide = emb.pivot(index="gene", columns="col", values="r2")

    def sort_key(col):
        variant = col[: -len("_r2")]
        if variant in HEADS:
            return (1, 0, HEADS[variant])
        return (0, ratio_mod.layer_sort_key(variant), 0)

    wide = wide[sorted(wide.columns, key=sort_key)]
    t = pd.concat([geno[["n", "n_afr"]].rename(columns={"n": "n_donors"}),
                   geno["r2"].rename("genotype_r2"), wide], axis=1)

    gm = pd.read_csv(args.gene_map, sep="\t", dtype=str)
    gm["ensg_id"] = gm["gene_id"].str.split(".").str[0]
    info = gm.drop_duplicates("experiment").set_index("experiment")[["ensg_id", "gene_name"]]
    t.insert(0, "gene_name", t.index.map(info["gene_name"]))
    t.insert(0, "ensg_id", t.index.map(info["ensg_id"]))
    t = t.sort_values("ensg_id")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / "table_s4.tsv", sep="\t", index=False)
    print(f"[done] {len(t)} genes x {wide.shape[1]} embedding R^2 columns "
          f"(genes missing from the gene map: {int(t['ensg_id'].isna().sum())}) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
