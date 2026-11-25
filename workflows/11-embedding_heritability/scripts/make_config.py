#!/usr/bin/env python
"""Build a workflow-09 config TSV from gene symbols or IDs, using GENCODE for the TSS.

The Borzoi window is centred on the region midpoint, so each gene's region is written
as chrN:TSS-TSS -- the window is then centred on the transcription start site
(gene start on +, gene end on -).

Run once, offline; the pipeline consumes the TSV.
"""
from __future__ import annotations

import argparse
import gzip
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtf", required=True, help="GENCODE annotation GTF (gzipped)")
    ap.add_argument("--genes", required=True, nargs="+",
                    help="gene symbols and/or Ensembl IDs")
    ap.add_argument("--tissue", default="Whole Blood")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    want = {g.upper() for g in args.genes}
    want_ens = {g.split(".")[0].upper() for g in args.genes if g.upper().startswith("ENSG")}
    found = {}
    with gzip.open(args.gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^"]+)"', f[8])
            gname = re.search(r'gene_name "([^"]+)"', f[8])
            if not gid:
                continue
            gid, gname = gid.group(1), (gname.group(1) if gname else "")
            key = None
            if gname.upper() in want:
                key = gname
            elif gid.split(".")[0].upper() in want_ens:
                key = gname or gid
            if key and key not in found:
                tss = int(f[3]) if f[6] == "+" else int(f[4])
                found[key] = (gid, gname, f[0], tss, f[6])

    missing = want - {k.upper() for k in found} - {v[0].split(".")[0].upper() for v in found.values()}
    if missing:
        print(f"[warn] not found in GTF: {sorted(missing)}")

    with open(args.out, "w") as out:
        out.write("experiment\tgene_id\tgene_name\tregion\ttissue\n")
        for key, (gid, gname, chrom, tss, strand) in sorted(found.items()):
            label = (gname or gid).replace(" ", "_")
            out.write(f"{label}\t{gid}\t{gname}\t{chrom}:{tss}-{tss}\t{args.tissue}\n")
            print(f"[info] {label}: {chrom}:{tss} (strand {strand}, {gid})")
    print(f"[done] {len(found)} genes -> {args.out}")


if __name__ == "__main__":
    main()
