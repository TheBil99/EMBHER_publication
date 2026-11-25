#!/usr/bin/env python
"""Merge the genes' rows of one model x layer into one sorted BED (bundle seqlet_beds, step merge_bed).

Writes <out_dir>/<model>_layer-<NN>_<layer_name>.bed (header line starting with '#'); the caller bgzips and
tabix-indexes it. Sorted by chromosome (chr1..22, X, Y, M) then start then end.
"""
import argparse
from pathlib import Path

import pandas as pd

from bundle_common import LAYERS, bed_name
from seqlets_to_bed import COLS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packed_dir", required=True)
    ap.add_argument("--genes", required=True, help="text file, one gene per line")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer_index", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    genes = [g.strip() for g in open(a.genes) if g.strip()]
    parts = [pd.read_csv(Path(a.packed_dir) / g / f"{a.model}__{a.layer_index:02d}.tsv.gz", sep="\t",
                         header=None, names=COLS, dtype={"chrom": str}) for g in genes]
    df = pd.concat(parts, ignore_index=True)
    order = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y", "M"]]
    extra = sorted(set(df["chrom"]) - set(order))
    df["chrom"] = pd.Categorical(df["chrom"], categories=order + extra, ordered=True)
    df = df.sort_values(["chrom", "start", "end", "name"], kind="stable")
    layer_name = next(n for i, n, _ in LAYERS if i == a.layer_index)
    out = Path(a.out_dir) / bed_name(a.model, a.layer_index, layer_name)[:-3]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("#" + "\t".join(COLS) + "\n")
        df.to_csv(fh, sep="\t", header=False, index=False)
    print(f"[done] {out.name}: {len(df)} rows from {len(genes)} genes", flush=True)


if __name__ == "__main__":
    main()
