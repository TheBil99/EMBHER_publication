#!/usr/bin/env python
"""One gene's seqlets as BED rows, one file per model and layer (bundle seqlet_beds, step pack_gene).

Reads <genes_dir>/<gene>/seqlets/{reference,random_init}/<target>/seqlets.parquet (workflow 15) for the 24
layers + expression and the window (chrom, win_start) from the gene's attribution.h5 attrs. The window is on
the forward strand, so hg38 [win_start + start, win_start + end) is 0-based half-open. Writes into
<out_dir>/<gene>/: <model>__<NN>.tsv.gz (BED rows without header, columns of the bundle's README) and
meta.tsv (gene, model, layer_index, layer_name, n_seqlets, chrom, tss, win_start, window_len).
A missing target is an error (a bundle is only complete if all 50 files are complete).
"""
import argparse
import gzip
from pathlib import Path

import h5py
import pandas as pd

from bundle_common import LAYERS, MODELS

COLS = ["chrom", "start", "end", "name", "score", "strand", "p_value", "pattern_type", "gene"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes_dir", required=True, help="workflow 15 <run>/genes")
    ap.add_argument("--gene", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    gdir = Path(a.genes_dir) / a.gene
    out = Path(a.out_dir) / a.gene
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for role, model in MODELS.items():
        win = None
        for idx, lname, target in LAYERS:
            h5 = gdir / "attribution" / role / target / "attribution.h5"
            pq = gdir / "seqlets" / role / target / "seqlets.parquet"
            for f in (h5, pq):
                if not f.exists():
                    raise SystemExit(f"[error] {a.gene}: missing {f}")
            with h5py.File(h5, "r") as h:
                w = (str(h.attrs["chrom"]), int(h.attrs["win_start"]), int(h.attrs["tss"]),
                     int(h["sequence"].shape[0]))
            if win is None:
                win = w
            elif w != win:
                raise SystemExit(f"[error] {a.gene}: window differs between targets of {role}: {w} vs {win}")
            chrom, ws, tss, L = win
            s = pd.read_parquet(pq).reset_index(drop=True)
            bed = pd.DataFrame({
                "chrom": chrom, "start": ws + s["start"], "end": ws + s["end"],
                "name": a.gene + "|" + s.index.astype(str), "score": s["attribution"], "strand": ".",
                "p_value": s["p-value"], "pattern_type": s["pattern_type"], "gene": a.gene})[COLS]
            with gzip.open(out / f"{model}__{idx:02d}.tsv.gz", "wt") as fh:
                bed.to_csv(fh, sep="\t", header=False, index=False, float_format="%.6g")
            meta.append(dict(gene=a.gene, model=model, layer_index=idx, layer_name=lname,
                             n_seqlets=len(bed), chrom=chrom, tss=tss, win_start=ws, window_len=L))
    pd.DataFrame(meta).to_csv(out / "meta.tsv", sep="\t", index=False)
    print(f"[done] {a.gene}: {len(meta)} files, {sum(m['n_seqlets'] for m in meta)} seqlets", flush=True)


if __name__ == "__main__":
    main()
