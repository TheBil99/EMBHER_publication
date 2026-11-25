#!/usr/bin/env python
"""Collapse one gene's workflow-06 haplotype axis into donor-indexed tables.

Workflow 06 (extract_embeddings) writes one embher-capture H5 per gene, in the
"stacked" layout: sample and haplotype are array axes, so nothing is pre-combined
across haplotypes. This script does that combining, for the two products workflow 11 needs:

  embedding    pooled_embeddings/embeddings/final__seq_mean -- Borzoi's 1920-dim trunk
               output, mean-pooled over the sequence axis, the tensor immediately
               before the output head. Haplotypes are SUMMED.
  prediction   outputs/rna_seq/exon_sum_unsquashed, restricted to --tracks (original
               Borzoi channel indices, resolved via meta/tracks/rna_seq/track_index).
               Haplotypes are SUMMED per track, then the tracks are averaged (plain
               mean, no log2) -- several of --tracks are different assays/replicates
               for the same tissue, so they must be combined by averaging, not by
               summing them into one pooled track. Linear (unsquashed coverage) scale,
               not log2 -- no log transform is applied here.

bfloat16 tensors are stored as their raw bit pattern in a uint16 dataset (h5py has no
native bfloat16), decoded here via ml_dtypes the same way h5_writer.py encoded them.
"""
from __future__ import annotations

import argparse

import h5py
import ml_dtypes
import numpy as np
import pandas as pd


def decode_bfloat16(raw: np.ndarray) -> np.ndarray:
    return raw.view(ml_dtypes.bfloat16).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", required=True, help="one gene's workflow-06 H5 file")
    ap.add_argument("--gene-id", required=True,
                     help="unversioned ENSG, as stored in the H5 (genes/<gene-id>)")
    ap.add_argument("--tracks", required=True,
                     help="comma-separated original Borzoi track indices to sum")
    ap.add_argument("--out-embedding", required=True)
    ap.add_argument("--out-prediction", required=True)
    args = ap.parse_args()

    want_tracks = {int(t) for t in args.tracks.split(",")}

    with h5py.File(args.h5, "r") as f:
        samples = [s.decode() for s in f["samples"][...]]
        gene = f[f"genes/{args.gene_id}"]

        emb_raw = gene["pooled_embeddings/embeddings/final__seq_mean"][...]  # (n, 2, 1920)
        emb = decode_bfloat16(emb_raw).sum(axis=1)  # sum haplotypes -> (n, 1920)
        emb_df = pd.DataFrame(emb, index=samples,
                               columns=[f"e{i}" for i in range(emb.shape[1])])
        emb_df.index.name = "donor_id"
        emb_df.to_parquet(args.out_embedding)

        pred_raw = gene["outputs/rna_seq/exon_sum_unsquashed"][...]  # (n, 2, n_tracks)
        track_index = f["meta/tracks/rna_seq/track_index"][...]
        cols = [i for i, t in enumerate(track_index) if int(t) in want_tracks]
        missing = want_tracks - {int(track_index[i]) for i in cols}
        if missing:
            raise SystemExit(
                f"track(s) {sorted(missing)} not found in meta/tracks/rna_seq/track_index "
                f"for {args.gene_id} ({args.h5})"
            )
        per_track = pred_raw[:, :, cols].sum(axis=1)   # sum haplotypes -> (n, n_tracks)
        pred_expression = per_track.mean(axis=1)       # average across same-tissue tracks
        pred_df = pd.DataFrame({"pred_expression": pred_expression}, index=samples)
        pred_df.index.name = "donor_id"
        pred_df.to_parquet(args.out_prediction)


if __name__ == "__main__":
    main()
