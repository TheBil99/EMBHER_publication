#!/usr/bin/env python
"""Cache one GTEx tissue's gene-level TPM as a parquet (genes x donors).

The v10 TPM GCT is 2.2 GB gzipped and holds every tissue, so it is streamed once
per tissue and cached; every downstream per-gene step then reads the small parquet.

Samples are mapped to donors (GTEX-XXXXX-0626-SM-XXXXX -> GTEX-XXXXX); a donor with
several samples in the tissue keeps the first. Donors are restricted to those present
in the phased WGS VCF, since only they can be embedded or genotyped.
"""
from __future__ import annotations

import argparse
import gzip

import pandas as pd


def donor_of(sampid: str) -> str:
    return "-".join(sampid.split("-")[:2])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tpm-gct", required=True)
    ap.add_argument("--sample-attributes", required=True)
    ap.add_argument("--vcf-samples", required=True, help="one donor ID per line")
    ap.add_argument("--tissue", required=True, help="SMTSD value, e.g. 'Whole Blood'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sa = pd.read_csv(args.sample_attributes, sep="\t", low_memory=True,
                     usecols=["SAMPID", "SMTSD", "SMAFRZE"])
    sa = sa[(sa.SMTSD == args.tissue) & (sa.SMAFRZE == "RNASEQ")]
    if sa.empty:
        raise SystemExit(f"[error] no RNASEQ samples for tissue '{args.tissue}'")
    wgs = {l.strip() for l in open(args.vcf_samples) if l.strip()}

    # one sample per donor, donor must have phased WGS
    keep = {}
    for s in sa.SAMPID:
        d = donor_of(s)
        if d in wgs and d not in keep:
            keep[d] = s
    sample_to_donor = {v: k for k, v in keep.items()}
    print(f"[info] {args.tissue}: {len(sa)} samples -> {len(keep)} donors with phased WGS",
          flush=True)

    with gzip.open(args.tpm_gct, "rt") as fh:
        fh.readline(); fh.readline()                      # GCT version + dims
        header = fh.readline().rstrip("\n").split("\t")
        cols = [i for i, c in enumerate(header) if c in sample_to_donor]
        names = [sample_to_donor[header[i]] for i in cols]
        gene_i = header.index("Name")
        rows, index = [], []
        for k, line in enumerate(fh):
            f = line.rstrip("\n").split("\t")
            index.append(f[gene_i])
            rows.append([float(f[i]) for i in cols])
            if (k + 1) % 20000 == 0:
                print(f"[info] {k + 1} genes read", flush=True)

    df = pd.DataFrame(rows, index=pd.Index(index, name="gene_id"), columns=names)
    df.to_parquet(args.out)
    print(f"[done] {df.shape[0]} genes x {df.shape[1]} donors -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
