#!/usr/bin/env python3
"""
GENCODE GTF -> parquet, one row per GTF feature, attributes expanded to columns (the table embher-capture,
workflow 06 and workflow 15 read as --gencode_parquet).

Columns: Chromosome, Source, Feature, Start (0-based, = GTF start - 1), End, Score, Strand, Frame, then one
column per attribute key (gene_id, gene_type, gene_name, transcript_id, exon_number, ...). A key that occurs
more than once in a feature (GENCODE's `tag`) keeps its last value, as pyranges.read_gtf does.

Usage: python gtf_to_parquet.py --gtf gencode.v46.annotation.gtf.gz --out gencode.v46.annotation.gtf.parquet
"""
import argparse
import re

import pandas as pd

ATTR = re.compile(r'(\w+) (?:"([^"]*)"|([^;\s]+))')
COLUMNS = ["Chromosome", "Source", "Feature", "Start", "End", "Score", "Strand", "Frame"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gtf = pd.read_csv(a.gtf, sep="\t", comment="#", header=None, names=COLUMNS + ["attributes"],
                      dtype={"Chromosome": str, "Score": str, "Frame": str, "Strand": str})
    gtf["Start"] = gtf["Start"].astype("int64") - 1
    gtf["End"] = gtf["End"].astype("int64")

    keys, rows = [], []
    for s in gtf.pop("attributes"):
        d = {}
        for k, quoted, bare in ATTR.findall(s):
            v = quoted or bare
            d[k] = v
            if k not in keys:
                keys.append(k)
        rows.append(d)
    attrs = pd.DataFrame(rows, columns=keys)

    out = pd.concat([gtf.reset_index(drop=True), attrs], axis=1)
    out["Chromosome"] = out["Chromosome"].astype("category")
    out["Strand"] = pd.Categorical(out["Strand"], categories=["+", "-"], ordered=True)
    out.to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
