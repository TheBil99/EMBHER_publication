#!/usr/bin/env python
"""Produce every per-gene summary AND the cross-gene scatter in one task.

Replaces one short task per gene (hundreds of them) with a single task. Each gene is
rendered by invoking summarize_embedding_heritability.py as a subprocess, which keeps that
script independently usable and gives per-gene failure isolation for free: a gene with
malformed inputs is recorded and skipped instead of taking the whole summary down with
it. The cross-gene figure is then built from the genes that succeeded.

Input files are grouped by gene from their filenames, as published upstream:
    <GENE>_heritability.parquet             (EREML)
    <GENE>_prediction_correlation.parquet   (Borzoi head)
Output layout keeps per-gene summaries out of the overall (cross-gene) ones, each flat
(no per-gene subfolder -- each gene's summary is a single file):
    per_gene/<GENE>_heritability_summary.{parquet,pdf}
    overall/across_genes.{pdf,parquet}
    overall/nonconverged.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


def by_gene(paths, suffix):
    out = {}
    for p in paths:
        name = Path(p).name
        if name.endswith(suffix):
            out[name[: -len(suffix)]] = str(p)
    return out


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding-herit", nargs="+", required=True)
    ap.add_argument("--prediction-corr", nargs="+", required=True)
    ap.add_argument("--per-gene-script", required=True)
    ap.add_argument("--across-script", required=True)
    ap.add_argument("--variant", default="ancestry corrected")
    ap.add_argument("--figures", action="store_true",
                    help="also write a per-gene figure; off by default (with one gene it "
                         "is three dots, i.e. the same content as the parquet row)")
    ap.add_argument("--out-nonconverged", default="overall/nonconverged.txt",
                    help="genes whose GREML or EREML (ai_reml_one) fit did not "
                         "converge, one 'gene\\tvariant\\tmethod' line per exclusion -- "
                         "that gene's point for that method is dropped from "
                         "across_genes.pdf rather than plotted")
    args = ap.parse_args()

    per_gene = load_module(args.per_gene_script, "per_gene")   # imports paid once
    across = load_module(args.across_script, "across")

    emb = by_gene(args.embedding_herit, "_heritability.parquet")
    corr = by_gene(args.prediction_corr, "_prediction_correlation.parquet")

    genes = sorted(set(emb) & set(corr))
    missing = (set(emb) | set(corr)) - set(genes)
    print(f"[info] {len(genes)} genes with both inputs", flush=True)
    if missing:
        print(f"[warn] {len(missing)} gene(s) missing an input, skipped: "
              f"{sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}", flush=True)

    os.makedirs("per_gene", exist_ok=True)
    os.makedirs("overall", exist_ok=True)

    done, failed, nonconverged = [], {}, []
    for i, g in enumerate(genes, 1):
        out_parquet = f"per_gene/{g}_heritability_summary.parquet"
        try:
            _, gene_nonconverged = per_gene.analyse(
                [emb[g]], [corr[g]], out_parquet,
                f"per_gene/{g}_heritability_summary.pdf" if args.figures else None,
                quiet=True)
            nonconverged.extend(gene_nonconverged)
            done.append(out_parquet)
        except Exception as e:                  # one bad gene must not sink the task
            failed[g] = f"{type(e).__name__}: {e}"
        if i % 50 == 0 or i == len(genes):
            print(f"[info] {i}/{len(genes)} genes ({len(failed)} failed)", flush=True)

    if failed:
        print(f"[warn] {len(failed)} gene(s) failed:", flush=True)
        for g, msg in list(failed.items())[:20]:
            print(f"    {g}: {msg}", flush=True)
    if not done:
        raise SystemExit("[error] no gene rendered successfully; nothing to summarise")

    if nonconverged:
        print(f"[warn] {len(nonconverged)} fit(s) did not converge, excluded "
              f"from across_genes.pdf -- see {args.out_nonconverged}", flush=True)
    with open(args.out_nonconverged, "w") as f:
        for gene, vlabel, method in nonconverged:
            f.write(f"{gene}\t{vlabel}\t{method}\n")
        if not nonconverged:
            f.write("none\n")

    import sys
    argv = ["plot_across_genes", "--summaries", *done, "--variant", args.variant,
            "--out-pdf", "overall/across_genes.pdf",
            "--out-table", "overall/across_genes.parquet"]
    print(f"[info] cross-gene figure over {len(done)} genes", flush=True)
    saved, sys.argv = sys.argv, argv
    try:
        across.main()
    finally:
        sys.argv = saved
    print(f"[done] {len(done)} per-gene summaries + across_genes.pdf", flush=True)


if __name__ == "__main__":
    main()
