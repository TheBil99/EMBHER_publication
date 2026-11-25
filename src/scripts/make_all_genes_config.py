#!/usr/bin/env python
"""Build a workflow config from every GENCODE gene expressed in a tissue.

The gene universe is GENCODE itself: every gene that clears the expression and
window-size filters below is included.

Genes are dropped when they cannot be analysed correctly:
  - longer than the model's predicted interval (window_size/2 - crop), because
    output_slice silently clips exons beyond it;
  - not expressed in the tissue, so the phenotype would be near-constant.
Chromosomes in --exclude-chrom (default chrX) are dropped outright.

--no-expression-filter drops the tissue-expression criterion entirely (and makes
--tissue-matrix optional): Borzoi embeddings/layer-activations/rna_predictions_wide
don't depend on tissue at all, only downstream phenotype prep does, so a purely
sequence-driven extraction workflow (workflow 06) wants every window-length-eligible
gene, not just the ones expressed in one tissue. The flag defaults to off.
"""
from __future__ import annotations

import argparse
import gzip
import re

import pandas as pd

BIN_SIZE = 32
N_BINS = 6144            # Borzoi's cropped output length, in bins


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtf", required=True, help="GENCODE annotation GTF (gzipped)")
    ap.add_argument("--tissue-matrix", default=None,
                    help="cached genes x donors TPM parquet; required unless --no-expression-filter")
    ap.add_argument("--tissue", default="Whole Blood",
                    help="only used to label the output's tissue column; with "
                         "--no-expression-filter it need not match any real tissue")
    ap.add_argument("--no-expression-filter", action="store_true",
                    help="keep every window-length-eligible gene regardless of tissue "
                         "expression (see module docstring); --tissue-matrix is then optional")
    ap.add_argument("--exclude-chrom", nargs="*", default=["chrX"],
                    help="chromosomes to drop entirely, e.g. chrX chrY chrM")
    ap.add_argument("--min-median-tpm", type=float, default=0.1)
    ap.add_argument("--min-frac-expressed", type=float, default=0.2,
                    help="minimum fraction of donors above 0.1 TPM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-report", default=None, help="optional parquet of the full table")
    args = ap.parse_args()

    if not args.no_expression_filter and not args.tissue_matrix:
        raise SystemExit("[error] --tissue-matrix is required unless --no-expression-filter is set")

    max_len = N_BINS * BIN_SIZE // 2      # half the predicted interval
    exclude = set(args.exclude_chrom)

    # GENCODE is the gene universe: TSS + span, straight from the GTF.
    ann = {}
    with gzip.open(args.gtf, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 9)
            if f[2] != "gene":
                continue
            if f[0] in exclude:
                continue
            gid = re.search(r'gene_id "([^"]+)"', f[8])
            if not gid:
                continue
            base = gid.group(1).split(".")[0]
            name = re.search(r'gene_name "([^"]+)"', f[8])
            start, end = int(f[3]), int(f[4])
            ann[base] = dict(gene_id=gid.group(1), gene_name=name.group(1) if name else base,
                             chrom=f[0], tss=start if f[6] == "+" else end,
                             length=end - start)
    print(f"[info] {len(ann)} genes in {args.gtf} (excluding {sorted(exclude)})", flush=True)

    expr = None
    if not args.no_expression_filter:
        expr = pd.read_parquet(args.tissue_matrix)
        expr.index = expr.index.astype(str).str.split(".").str[0]
        expr = expr[~expr.index.duplicated(keep="first")]

    rows, drop = [], dict(too_long=0, not_expressed=0)
    for g, a in ann.items():
        if a["length"] > max_len:
            drop["too_long"] += 1
            continue
        median_tpm = None
        if not args.no_expression_filter:
            if g not in expr.index:
                drop["not_expressed"] += 1
                continue
            t = expr.loc[g].astype(float)
            if t.median() < args.min_median_tpm or (t > 0.1).mean() < args.min_frac_expressed:
                drop["not_expressed"] += 1
                continue
            median_tpm = float(t.median())
        rows.append(dict(experiment=a["gene_name"].replace(" ", "_"), gene_id=a["gene_id"],
                         gene_name=a["gene_name"], region=f"{a['chrom']}:{a['tss']}-{a['tss']}",
                         tissue=args.tissue, length=a["length"], median_tpm=median_tpm))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("[error] no genes survived filtering")
    df.drop_duplicates("experiment", inplace=True)
    df = df.sort_values("experiment")
    df[["experiment", "gene_id", "gene_name", "region", "tissue"]].to_csv(
        args.out, sep="\t", index=False)
    if args.out_report:
        df.to_parquet(args.out_report, index=False)

    print(f"[info] dropped while walking GENCODE: {drop}", flush=True)
    print(f"[done] {len(df)} genes -> {args.out}")
    print(f"       gene length: {df.length.min():,} - {df.length.max():,} bp (limit {max_len:,})")
    print(df.head(10)[["experiment", "region", "median_tpm"]].to_string(index=False))


if __name__ == "__main__":
    main()
