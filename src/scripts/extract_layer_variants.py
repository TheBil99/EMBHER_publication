#!/usr/bin/env python
"""Split one gene's workflow-06 H5 into one donor x feature embedding parquet per
(layer, pooling) variant, plus one "rna_predictions" variant of the full RNA-seq
track vector. Shared by workflows 12 (heritability_over_layers) and 14
(correlation_matrices), so both read the identical per-variant tables.

Workflow 06 (extract_embeddings) already captures the FULL Borzoi trunk depth per
gene -- encoder/conv_dna, encoder/res_tower_{0..4}, encoder/unet1, encoder/trunk,
transformer/block_{0..7}, skip/horizontal_conv{0,1}, decoder/upsample{0,1},
decoder/separable{0,1}, embeddings/{cropped,final} -- each pooled BOTH over the
position axis (__seq_mean) and the channel axis (__channel_mean), under
genes/{gene}/pooled_embeddings/ (capture/borzoi.py's _capture_specs, reduce.py's
Pooling.suffix). This script just reads that cache; it never runs the model.

  pos   __seq_mean pooling: mean over the position axis, one value per channel.
  chan  __channel_mean pooling: mean over the channel axis, positions left
        untouched, one value per genomic position.

__seq_max/__channel_max are ignored -- the pos/chan axes are mean pooling only.
Haplotype combination is the additive SUM of the two haplotypes' pooled vectors, same
as src/scripts/combine_haplotypes.py's embeddings/final -- do not change this without
revisiting every cached GRM built from it.

rna_predictions stacks Borzoi's predicted coverage across EVERY RNA-seq track
(genes/{gene}/outputs/rna_seq/exon_sum_unsquashed, haplotype-summed, linear scale,
no log transform) -- the model's own prediction-head output, not a trunk-pooling
choice, so it is reported as its own reference variant rather than a point on the
layer-depth axis.

rna_predictions_whole_blood is the same prediction-head output restricted to
--whole-blood-tracks (default 7531,7532,7533 -- the three GTEx whole-blood RNA-seq
tracks among Borzoi's training targets, same default as workflow 11's own
params.pred_tracks) and pooled exactly as src/scripts/combine_haplotypes.py's own
pred_expression: haplotypes SUMMED per track, then the tracks AVERAGED (plain mean,
no log2, linear coverage scale) -- several of a tissue's tracks are different
assays/replicates for the same tissue, so they must be combined by averaging, not
summed into one pooled track. A single-column donor x feature table (one point, not
a vector like rna_predictions), so it compares directly against workflow 11's own
embedding/joint h2 on the identical whole-blood phenotype.

Output: one parquet per written variant, `<out-prefix>__<variant>_embedding.parquet`,
donor_id-indexed.
"""
from __future__ import annotations

import argparse

import h5py
import ml_dtypes
import numpy as np
import pandas as pd

POOL_SUFFIXES = {"__seq_mean": "pos", "__channel_mean": "chan"}
RNA_PREDICTIONS = "rna_predictions"
RNA_PREDICTIONS_WHOLE_BLOOD = "rna_predictions_whole_blood"


def decode_bfloat16(raw: np.ndarray) -> np.ndarray:
    return raw.view(ml_dtypes.bfloat16).astype(np.float32)


def list_pooled_variants(gene_group: h5py.Group) -> dict[str, str]:
    """{variant_label: dataset_path} for every __seq_mean/__channel_mean leaf under
    pooled_embeddings/, e.g. 'encoder_conv_dna_pos': 'pooled_embeddings/encoder/conv_dna__seq_mean'."""
    variants: dict[str, str] = {}
    if "pooled_embeddings" not in gene_group:
        return variants

    def visit(name: str, obj) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        for suffix, axis in POOL_SUFFIXES.items():
            if name.endswith(suffix):
                layer = name[: -len(suffix)].replace("/", "_")
                variants[f"{layer}_{axis}"] = f"pooled_embeddings/{name}"
                return

    gene_group["pooled_embeddings"].visititems(visit)
    return variants


def extract_pooled(gene_group: h5py.Group, dataset_path: str, samples: list[str]) -> pd.DataFrame:
    raw = gene_group[dataset_path][...]  # (n, 2, d), bfloat16 bit pattern
    emb = decode_bfloat16(raw).sum(axis=1)  # sum haplotypes -> (n, d)
    df = pd.DataFrame(emb, index=samples, columns=[f"e{i}" for i in range(emb.shape[1])])
    df.index.name = "donor_id"
    return df


def extract_rna_predictions(gene_group: h5py.Group, samples: list[str]) -> pd.DataFrame:
    raw = gene_group["outputs/rna_seq/exon_sum_unsquashed"][...]  # (n, 2, n_tracks) float32
    coverage = raw.astype(np.float32).sum(axis=1)  # sum haplotypes -> (n, n_tracks), linear scale
    df = pd.DataFrame(coverage, index=samples, columns=[f"t{i}" for i in range(coverage.shape[1])])
    df.index.name = "donor_id"
    return df


def extract_rna_predictions_whole_blood(
    gene_group: h5py.Group, samples: list[str], track_index: np.ndarray, want_tracks: set[int]
) -> pd.DataFrame:
    """Same pooling as combine_haplotypes.py's pred_expression: haplotypes SUMMED per
    track, then want_tracks AVERAGED (plain mean, no log2, linear scale)."""
    raw = gene_group["outputs/rna_seq/exon_sum_unsquashed"][...]  # (n, 2, n_tracks) float32
    cols = [i for i, t in enumerate(track_index) if int(t) in want_tracks]
    missing = want_tracks - {int(track_index[i]) for i in cols}
    if missing:
        raise SystemExit(
            f"[error] --whole-blood-tracks {sorted(missing)} not found in "
            "meta/tracks/rna_seq/track_index"
        )
    per_track = raw.astype(np.float32)[:, :, cols].sum(axis=1)  # sum haplotypes -> (n, n_tracks)
    pred_expression = per_track.mean(axis=1)  # average across same-tissue tracks
    df = pd.DataFrame({"pred_expression": pred_expression}, index=samples)
    df.index.name = "donor_id"
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="one gene's workflow-06 H5 file")
    ap.add_argument("--gene-id", required=True,
                    help="unversioned ENSG, as stored in the H5 (genes/<gene-id>)")
    ap.add_argument("--out-prefix", required=True,
                    help="output files are <out-prefix>__<variant>_embedding.parquet")
    ap.add_argument("--variants", default=None,
                    help="comma-separated variant labels to restrict to, e.g. "
                         "'encoder_conv_dna_pos,rna_predictions' (default: every "
                         "variant found in the H5)")
    ap.add_argument("--whole-blood-tracks", default="7531,7532,7533",
                    help="comma-separated original Borzoi track indices pooled into "
                         "rna_predictions_whole_blood (default: the three GTEx "
                         "whole-blood RNA-seq tracks, same default as workflow 11's "
                         "own --pred_tracks)")
    args = ap.parse_args()
    want_whole_blood_tracks = {int(t) for t in args.whole_blood_tracks.split(",")}

    with h5py.File(args.h5, "r") as f:
        samples = [s.decode() for s in f["samples"][...]]
        gene = f[f"genes/{args.gene_id}"]

        available = list_pooled_variants(gene)
        available[RNA_PREDICTIONS] = None  # special-cased below, not a plain dataset path
        available[RNA_PREDICTIONS_WHOLE_BLOOD] = None  # ditto
        if not available:
            raise SystemExit(f"[error] no pooled_embeddings found for {args.gene_id} in {args.h5}")

        wanted = None
        if args.variants:
            wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
            missing = wanted - set(available)
            if missing:
                raise SystemExit(
                    f"[error] requested variant(s) not available for {args.gene_id} "
                    f"({args.h5}): {sorted(missing)}"
                )

        track_index = f["meta/tracks/rna_seq/track_index"][...] \
            if RNA_PREDICTIONS_WHOLE_BLOOD in (wanted or available) else None

        written = []
        for label in sorted(available):
            if wanted is not None and label not in wanted:
                continue
            if label == RNA_PREDICTIONS:
                df = extract_rna_predictions(gene, samples)
            elif label == RNA_PREDICTIONS_WHOLE_BLOOD:
                df = extract_rna_predictions_whole_blood(
                    gene, samples, track_index, want_whole_blood_tracks)
            else:
                df = extract_pooled(gene, available[label], samples)
            df.to_parquet(f"{args.out_prefix}__{label}_embedding.parquet")
            written.append(label)

    print(f"[done] wrote {len(written)} variant(s) for {args.gene_id}: {sorted(written)}")


if __name__ == "__main__":
    main()
