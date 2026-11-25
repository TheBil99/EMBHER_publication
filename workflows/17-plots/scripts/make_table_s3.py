#!/usr/bin/env python
"""table_s3: Borzoi heritability (HE regression) per gene, for the pretrained and the
randomly-initialised model -- every trunk layer's embedding plus the RNA-seq output heads.

Reads workflow 12's already-published per-gene heritability files of the two runs
(<run>/heritability/<gene>__<variant>_heritability.parquet, real-phenotype row); nothing is
recomputed. One row per gene, one h2 column per (model, variant):

  ensg_id, gene_name              from the run's gene map (workflow 11 config TSV)
  pretrained_<variant>            workflow 12's hi_q_random500 run
  random_init_<variant>           its hi_q_random500_borzoi_random run
with <variant> = <layer> (that layer's embedding pooled over the position axis, per-channel),
rna_predictions (all RNA-seq output tracks in one kernel) or rna_predictions_whole_blood (the
whole-blood tracks alone). Layers in forward-pass order. Channel-pooled embeddings are not
included. A cell is empty where that fit does not exist or did not converge.
"""
from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SUFFIX = "_heritability.parquet"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_h2(path: Path):
    gene, variant = path.name[: -len(SUFFIX)].split("__", 1)
    d = pd.read_parquet(path, columns=["phenotype", "h2", "converged"])
    d = d[d["phenotype"] == "pheno_expression"]
    if d.empty:
        return gene, variant, float("nan"), True
    return gene, variant, float(d["h2"].iloc[0]), bool(d["converged"].iloc[0])


def load_run(run_dir: str, prefix: str) -> tuple[pd.DataFrame, int]:
    files = sorted(f for f in Path(run_dir, "heritability").glob(f"*{SUFFIX}")
                   if not f.name.endswith(f"_chan{SUFFIX}"))    # position-axis layers and heads only
    if not files:
        raise SystemExit(f"[error] no heritability files under {run_dir}/heritability")
    with ThreadPoolExecutor(8) as ex:        # many small files on a network disk: I/O-bound
        rows = list(ex.map(read_h2, files))
    df = pd.DataFrame(rows, columns=["gene", "variant", "h2", "converged"])
    df["variant"] = df["variant"].str.replace(r"_pos$", "", regex=True)
    n_unconverged = int((~df["converged"]).sum())
    df.loc[~df["converged"], "h2"] = float("nan")
    wide = df.pivot(index="gene", columns="variant", values="h2")
    wide.columns = [f"{prefix}_{c}" for c in wide.columns]
    return wide, n_unconverged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained-run-dir", required=True,
                    help="results/12-heritability_over_layers/hi_q_random500")
    ap.add_argument("--random-init-run-dir", required=True,
                    help="results/12-heritability_over_layers/hi_q_random500_borzoi_random")
    ap.add_argument("--gene-map", required=True,
                    help="workflows/11-embedding_heritability/configs/hi_q_random500.tsv")
    ap.add_argument("--ratio-script", required=True,
                    help="workflows/17-plots/scripts/summarize_embedding_genotype_ratio.py "
                         "(imported by path for layer_sort_key)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ratio_mod = _load_module("ratio_mod", Path(args.ratio_script))
    pre, n_pre = load_run(args.pretrained_run_dir, "pretrained")
    rnd, n_rnd = load_run(args.random_init_run_dir, "random_init")
    t = pre.join(rnd, how="outer")

    heads = {"rna_predictions": 1, "rna_predictions_whole_blood": 2}

    def sort_key(col):
        """pretrained before random-init; layers in depth order (pos before chan), then heads."""
        random = col.startswith("random_init_")
        variant = col[len("random_init_"):] if random else col[len("pretrained_"):]
        if variant in heads:
            return (random, 1, 0, heads[variant])
        return (random, 0, ratio_mod.layer_sort_key(variant), 0)

    t = t[sorted(t.columns, key=sort_key)]

    gm = pd.read_csv(args.gene_map, sep="\t", dtype=str)
    gm["ensg_id"] = gm["gene_id"].str.split(".").str[0]
    info = gm.drop_duplicates("experiment").set_index("experiment")[["ensg_id", "gene_name"]]
    t.insert(0, "gene_name", t.index.map(info["gene_name"]))
    t.insert(0, "ensg_id", t.index.map(info["ensg_id"]))
    t = t.sort_values("ensg_id")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / "table_s3.tsv", sep="\t", index=False)
    print(f"[done] {len(t)} genes x {t.shape[1] - 2} h2 columns "
          f"(unconverged left empty: pretrained {n_pre}, random-init {n_rnd}; "
          f"genes missing from the gene map: {int(t['ensg_id'].isna().sum())}) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
