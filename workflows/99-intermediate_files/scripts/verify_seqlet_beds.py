#!/usr/bin/env python
"""Check the seqlet_beds bundle against its sources; exits non-zero on any mismatch.

For each of the 50 files (bgzipped BED + .tbi): row count equals the sum of the genes' seqlet counts, nine
columns, start < end, every interval inside its gene's window, rows sorted. Then a random sample of rows per
file is compared base by base with the reference FASTA (bgzip -b/-s with the .gzi) and with the window sequence
stored in the gene's attribution.h5, which checks the window-to-genome coordinate rule independently.
"""
import argparse
import gzip
import random
import subprocess
from pathlib import Path

import h5py
import pandas as pd

from bundle_common import LAYERS, MODELS, bed_name

ORDER = {f"chr{c}": i for i, c in enumerate(list(range(1, 23)) + ["X", "Y", "M"])}
ROLE = {m: r for r, m in MODELS.items()}
TARGET = {i: t for i, _, t in LAYERS}


def fasta_fetch(fasta, fai, bgzip, chrom, start, end):
    _, _, off, lb, lw = fai[chrom]
    def pos(p):
        return off + p // lb * lw + p % lb
    o0, o1 = pos(start), pos(end - 1)
    r = subprocess.run([bgzip, "-b", str(o0), "-s", str(o1 - o0 + 1), "-d", fasta],
                       capture_output=True, check=True, text=True)
    return r.stdout.replace("\n", "").upper()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bed_dir", required=True)
    ap.add_argument("--packed_dir", required=True)
    ap.add_argument("--genes", required=True)
    ap.add_argument("--genes_dir", required=True, help="workflow 15 <run>/genes")
    ap.add_argument("--fasta", required=True, help="bgzipped FASTA with .fai and .gzi")
    ap.add_argument("--bgzip", required=True)
    ap.add_argument("--sample", type=int, default=20, help="rows per file compared with the FASTA")
    a = ap.parse_args()

    genes = [g.strip() for g in open(a.genes) if g.strip()]
    meta = pd.concat([pd.read_csv(Path(a.packed_dir) / g / "meta.tsv", sep="\t") for g in genes])
    win = meta.drop_duplicates("gene").set_index("gene")[["chrom", "win_start", "window_len"]]
    fai = {l.split("\t")[0]: (l.split("\t")[0], *map(int, l.split("\t")[1:5]))
           for l in open(a.fasta + ".fai")}
    rng = random.Random(0)
    bad = []
    n_checked = 0
    for model in MODELS.values():
        for idx, lname, _ in LAYERS:
            f = Path(a.bed_dir) / bed_name(model, idx, lname)
            if not f.exists() or not Path(str(f) + ".tbi").exists():
                bad.append(f"{f.name}: file or .tbi missing")
                continue
            expect = int(meta[(meta.model == model) & (meta.layer_index == idx)]["n_seqlets"].sum())
            df = pd.read_csv(f, sep="\t", comment="#", header=None, dtype={0: str})
            if df.shape[1] != 9:
                bad.append(f"{f.name}: {df.shape[1]} columns")
                continue
            if len(df) != expect:
                bad.append(f"{f.name}: {len(df)} rows, sources have {expect}")
            if not (df[1] < df[2]).all():
                bad.append(f"{f.name}: start >= end")
            key = list(zip(df[0].map(ORDER), df[1]))
            if key != sorted(key):
                bad.append(f"{f.name}: not sorted")
            g = df[3].str.split("|").str[0]
            w = win.loc[g]
            if not ((df[0].to_numpy() == w["chrom"].to_numpy())
                    & (df[1].to_numpy() >= w["win_start"].to_numpy())
                    & (df[2].to_numpy() <= (w["win_start"] + w["window_len"]).to_numpy())).all():
                bad.append(f"{f.name}: interval outside its gene's window")
            for i in rng.sample(range(len(df)), min(a.sample, len(df))):
                row = df.iloc[i]
                gene = row[3].split("|")[0]
                ws = int(win.loc[gene, "win_start"])
                with h5py.File(Path(a.genes_dir) / gene / "attribution" / ROLE[model] / TARGET[idx]
                               / "attribution.h5", "r") as h:
                    seq = "".join("ACGTN"[min(int(v), 4)] for v in h["sequence"][row[1] - ws:row[2] - ws])
                ref = fasta_fetch(a.fasta, fai, a.bgzip, row[0], int(row[1]), int(row[2]))
                n_checked += 1
                if seq != ref:
                    bad.append(f"{f.name}: {row[3]} {row[0]}:{row[1]}-{row[2]} window seq {seq} != FASTA {ref}")
    if bad:
        print("[FAIL]\n" + "\n".join(bad[:50]), flush=True)
        raise SystemExit(1)
    print(f"[ok] 50 files verified, {n_checked} intervals matched the FASTA", flush=True)


if __name__ == "__main__":
    main()
