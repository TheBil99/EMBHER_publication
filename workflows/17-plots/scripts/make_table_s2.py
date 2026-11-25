#!/usr/bin/env python
"""table_s2: the 24 Borzoi layers whose embeddings workflows 12/15 use, in workflow 15's order
(layer_order.LABELS, imported by path): 1-based index, layer name and embedding dimension.

The dimension is read from one workflow-06 H5 (pooled_embeddings/<group>/<layer>__seq_mean, last
axis = the layer's channel count, which is the dimension of the position-pooled embedding), not
typed in. Writes table_s2.tsv: layer_index, layer_name, embedding_dimension.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import h5py
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer-order", required=True,
                    help="workflows/15-borzoi_attribution/scripts/layer_order.py")
    ap.add_argument("--h5", required=True,
                    help="any workflow-06 borzoi gene H5 (pooled_embeddings group, seq_mean)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("layer_order", args.layer_order)
    order = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(order)

    rows = []
    with h5py.File(args.h5) as h:
        pooled = h[f"genes/{next(iter(h['genes']))}/pooled_embeddings"]
        dims = {f"{g}_{k[: -len('__seq_mean')]}": pooled[g][k].shape[-1]
                for g in pooled for k in pooled[g] if k.endswith("__seq_mean")}
    for i, name in enumerate(order.LABELS, start=1):
        rows.append(dict(layer_index=i, layer_name=name, embedding_dimension=int(dims[name])))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "table_s2.tsv", sep="\t", index=False)
    print(f"[done] {len(rows)} layers -> {out}/table_s2.tsv", flush=True)


if __name__ == "__main__":
    main()
