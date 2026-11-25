#!/usr/bin/env python
"""Copy a prepare_tissue_matrix.py/prepare_expression_matrix.py output into workflow
02's (expression_data_preparation) cache and write its manifest row.

Mirrors workflow 06's (extract_borzoi_embeddings) gather_activations manifest-row
pattern: many small per-key files, one per completed task, so concurrent or rerun
tasks never race on a shared write -- see src/nf/borzoi_manifest.nf's header for the
same rationale. Cache keys come from resolve_expression.py (the read side), so the
write and read sides can never independently drift apart.
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil
import uuid

import pandas as pd

from resolve_expression import pheno_matrix_cache_key, slug, tissue_cache_key


def atomic_copy(src: str, dst: str) -> None:
    """Copy src to dst without ever leaving a partially-written dst visible: copy to a
    temp file in the same directory (same filesystem, so the final rename is atomic),
    then os.replace. Guards against two tasks that resolve to the SAME cache_key (e.g.
    two concurrent workflow-12 runs over the same tissue, whose tissue_matrix entry is
    transform-independent) racing on the same destination file -- shutil.copyfile alone
    is not atomic and a torn write would corrupt whichever reader gets there first.
    """
    tmp = f"{dst}.tmp-{uuid.uuid4().hex}"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="workflow 02's results root")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--product", choices=["tissue_matrix", "pheno_matrix"], required=True)
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--transform", default="")
    ap.add_argument("--matrix-file", help="product=tissue_matrix only")
    ap.add_argument("--pheno-matrix-file", help="product=pheno_matrix only")
    ap.add_argument("--donors-file", help="product=pheno_matrix only")
    args = ap.parse_args()

    tissue = args.tissue.strip()
    tk = slug(tissue)

    if args.product == "tissue_matrix":
        cache_key = tissue_cache_key(tk)
        dest_dir = os.path.join(args.data_dir, "tissue_matrix")
        os.makedirs(dest_dir, exist_ok=True)
        file_path = f"tissue_matrix/{cache_key}.parquet"
        atomic_copy(args.matrix_file, os.path.join(args.data_dir, file_path))
        row = dict(product_type=args.product, tissue=tissue, tissue_key=tk,
                  cache_key=cache_key, file_path=file_path,
                  created_at=datetime.datetime.utcnow().isoformat())
    else:
        cache_key = pheno_matrix_cache_key(tk, args.transform)
        dest_dir = os.path.join(args.data_dir, "pheno_matrix", tk)
        os.makedirs(dest_dir, exist_ok=True)
        file_path = f"pheno_matrix/{tk}/{cache_key}.parquet"
        donors_file_path = f"pheno_matrix/{tk}/{cache_key}_donors.txt"
        atomic_copy(args.pheno_matrix_file, os.path.join(args.data_dir, file_path))
        atomic_copy(args.donors_file, os.path.join(args.data_dir, donors_file_path))
        row = dict(product_type=args.product, tissue=tissue, tissue_key=tk,
                  transform=args.transform, cache_key=cache_key, file_path=file_path,
                  donors_file_path=donors_file_path,
                  created_at=datetime.datetime.utcnow().isoformat())

    manifest_dir = os.path.join(args.manifest_dir, tk)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, f"{cache_key}.tsv")
    tmp_manifest_path = f"{manifest_path}.tmp-{uuid.uuid4().hex}"
    pd.DataFrame([row]).to_csv(tmp_manifest_path, sep="\t", index=False)
    os.replace(tmp_manifest_path, manifest_path)
    print(f"[manifest] {args.product} {cache_key} -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
