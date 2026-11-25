#!/usr/bin/env python
"""Per-gene extract of a constraint track over the reference window of one gene (torch-free).

The window (chromosome, win_start, reference sequence) is read from
<genes_dir>/<gene>/attribution/reference/expression/attribution.h5. Writes <out_dir>/<gene>.parquet (zstd):

  --kind gnomad   one row per PASS single-nucleotide alternate allele in the window with AC > 0, read from the
                  per-chromosome gnomAD sites VCFs (tabix) under --source:
                  idx (0-based index in the window), alt, ac, an. The reference allele of every record is checked
                  against the window sequence (this also validates the window convention).
  --kind phylop   one row per window position: idx, phylop (float32, NaN where the track has no value), read
                  from the bigWig --source.

A gene already extracted (file newer than the attribution) is skipped (--force recomputes).
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

BASES = np.array(list("ACGT"))


def read_window(genes_dir, gene):
    f = Path(genes_dir) / gene / "attribution" / "reference" / "expression" / "attribution.h5"
    with h5py.File(f, "r") as h:
        return f, str(h.attrs["chrom"]), int(h.attrs["win_start"]), h["sequence"][...].astype(np.int8)


def gnomad_alleles(source, chrom, ws, seq):
    import pysam
    vcfs = sorted(Path(source).glob(f"*.sites.{chrom}.vcf.bgz"))
    if len(vcfs) != 1:
        raise SystemExit(f"[error] expected one VCF for {chrom} in {source}, found {len(vcfs)}")
    L = len(seq)
    idx, alt, ac, an, ref_ok, n_checked = [], [], [], [], 0, 0
    tbx = pysam.TabixFile(str(vcfs[0]))
    for line in tbx.fetch(chrom, ws, ws + L):
        f = line.split("\t", 8)
        if f[6] != "PASS" or len(f[3]) != 1:
            continue
        alts = f[4].split(",")
        info = f[7].split(";", 2)                           # AC=..;AN=..;AF=.. come first in the v4 sites files
        if not (info[0].startswith("AC=") and info[1].startswith("AN=")):
            raise SystemExit(f"[error] unexpected INFO order at {chrom}:{f[1]}: {f[7][:80]}")
        acs = info[0][3:].split(",")
        i = int(f[1]) - 1 - ws
        if not 0 <= i < L or seq[i] > 3:
            continue
        n_checked += 1
        ref_ok += BASES[seq[i]] == f[3]
        for a, c in zip(alts, acs):
            if len(a) == 1 and int(c) > 0:
                idx.append(i); alt.append(a); ac.append(int(c)); an.append(int(info[1][3:]))
    if n_checked == 0 or ref_ok / n_checked < 0.99:
        raise SystemExit(f"[error] reference allele matches the window sequence at {ref_ok} of {n_checked} "
                         f"records: the window coordinates do not agree with the VCF")
    return pd.DataFrame(dict(idx=np.array(idx, np.int32), alt=alt, ac=np.array(ac, np.int32),
                             an=np.array(an, np.int32)))


def phylop_values(source, chrom, ws, L):
    import pyBigWig
    bw = pyBigWig.open(str(source))
    v = np.full(L, np.nan, np.float32)                      # windows reaching past a chromosome end stay NaN (N-padded)
    lo, hi = max(ws, 0), min(ws + L, bw.chroms()[chrom])
    if hi > lo:
        v[lo - ws:hi - ws] = bw.values(chrom, lo, hi, numpy=True)
    return pd.DataFrame(dict(idx=np.arange(L, dtype=np.int32), phylop=v))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes_dir", required=True, help="<run>/genes; one folder per gene")
    ap.add_argument("--gene", required=True)
    ap.add_argument("--kind", required=True, choices=["gnomad", "phylop"])
    ap.add_argument("--source", required=True, help="gnomad: folder of the sites VCFs; phylop: bigWig")
    ap.add_argument("--out_dir", required=True, help="e.g. data/prep/constraint/<track>")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    att, chrom, ws, seq = read_window(a.genes_dir, a.gene)
    out = Path(a.out_dir) / f"{a.gene}.parquet"
    if not a.force and out.exists() and out.stat().st_mtime >= att.stat().st_mtime:
        print(f"[skip] {out} is up to date")
        return
    df = gnomad_alleles(a.source, chrom, ws, seq) if a.kind == "gnomad" else \
        phylop_values(a.source, chrom, ws, len(seq))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, compression="zstd", index=False)
    tmp.replace(out)
    print(f"[done] {a.gene} {chrom}:{ws}-{ws + len(seq)} {a.kind}: {len(df)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
