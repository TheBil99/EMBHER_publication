#!/usr/bin/env python
"""Resolve genes/tissues in a config TSV against workflow 02's
(expression_data_preparation) manifest -- the single source of truth for "where is
this tissue's cached TPM matrix" / "where is this tissue's cached phenotype MATRIX,
under this exact prepare_expression_matrix.py --transform".

Mirrors workflow 171's own scripts/resolve_embeddings.py (matches the manifest's own
semantic columns instead of re-deriving workflow 02's cache-key formula independently
-- see that script's docstring for why re-derivation is what to avoid).

Two products, each with its own required config columns and cache key:
  tissue_matrix  key = tissue alone (raw TPM has no transform dependence). Requires a
                 'tissue' column.
  pheno          key = tissue + transform -- two workflows asking for the same tissue
                 under different transforms get different, both-cached pheno_matrix
                 entries; conflating them would silently mix e.g. workflow 11/12's
                 log2_plus_1 phenotype with workflow 170/171's quantile_int one. No
                 covariate correction is applied (see prepare_expression_matrix.py) --
                 the phenotype is the transformed value as-is. The manifest entry is a
                 whole-tissue donors x genes matrix (workflow 02 computes every gene's
                 phenotype in one pass, not one file per gene -- see
                 prepare_expression_matrix.py); this script extracts the one column
                 (gene) each config row asks for and writes a per-gene pheno.parquet
                 (just pheno_expression -- no permutation columns; add those separately
                 with src/scripts/add_permutations.py if the caller needs a
                 heritability null) plus donors.txt. That extraction is cheap (a
                 single-column parquet read) and isn't itself cached.
                 Requires experiment/gene_id/region/tissue columns.

Fails loudly, for everything unresolvable at once, before any per-gene job runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


def tissue_cache_key(tissue_key: str) -> str:
    return tissue_key


def pheno_matrix_cache_key(tissue_key, transform) -> str:
    return f"{tissue_key}_t{transform}"


def load_manifest(manifest_dir: Path) -> dict:
    idx = {}
    if not manifest_dir.exists():
        return idx
    for f in manifest_dir.rglob("*.tsv"):
        row = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False).iloc[0].to_dict()
        idx[(row["product_type"], row["cache_key"])] = row
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="tissue, or experiment/gene_id/region/tissue TSV")
    ap.add_argument("--manifest-dir", required=True,
                    help="workflow 02's manifest dir, e.g. "
                         "results/02-expression_data_preparation/manifest")
    ap.add_argument("--data-dir", required=True,
                    help="workflow 02's results root (file_path is relative to this), "
                         "e.g. results/02-expression_data_preparation")
    ap.add_argument("--product", choices=["tissue_matrix", "pheno"], required=True)
    ap.add_argument("--transform", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = pd.read_csv(args.config, sep="\t", dtype=str)
    manifest = load_manifest(Path(args.manifest_dir))
    data_dir = Path(args.data_dir)

    if args.product == "tissue_matrix":
        if "tissue" not in config.columns:
            raise SystemExit("[error] --product tissue_matrix requires a 'tissue' column")
        tissues = sorted(config["tissue"].str.strip().unique())
        resolved, unresolved = [], []
        for tissue in tissues:
            tk = slug(tissue)
            hit = manifest.get(("tissue_matrix", tissue_cache_key(tk)))
            if hit is None:
                unresolved.append(tissue)
                continue
            resolved.append({"tissue": tissue, "tissue_key": tk,
                             "tissue_matrix_path": str((data_dir / hit["file_path"]).resolve())})
        if unresolved:
            raise SystemExit(
                f"[error] {len(unresolved)} tissue(s) have no cached tissue_matrix under "
                f"{args.manifest_dir}: {unresolved}\n"
                "Run workflow 02 (expression_data_preparation) for these tissues first."
            )
        pd.DataFrame(resolved).to_csv(args.out, sep="\t", index=False)
        print(f"[done] resolved {len(resolved)} tissue(s) -> {args.out}", flush=True)
        return

    required = ("experiment", "gene_id", "region", "tissue")
    missing_cols = [c for c in required if c not in config.columns]
    if missing_cols:
        raise SystemExit(f"[error] config missing required column(s): {missing_cols}")

    # one pheno_matrix per distinct tissue -- the transform is fixed for this whole
    # invocation, so the only thing that varies row-to-row is which tissue, and within
    # a tissue, which gene column to pull out of it.
    tissues = sorted(config["tissue"].str.strip().unique())
    matrices = {}
    unresolved_tissues = []
    for tissue in tissues:
        tk = slug(tissue)
        key = pheno_matrix_cache_key(tk, args.transform)
        hit = manifest.get(("pheno_matrix", key))
        if hit is None:
            unresolved_tissues.append(tissue)
            continue
        matrices[tissue] = pd.read_parquet(data_dir / hit["file_path"])
    if unresolved_tissues:
        raise SystemExit(
            f"[error] {len(unresolved_tissues)} tissue(s) have no cached pheno_matrix under "
            f"{args.manifest_dir} for transform={args.transform}: {unresolved_tissues}\n"
            "Run workflow 02 (expression_data_preparation) with matching parameters for these tissues first."
        )

    resolved, unresolved = [], []
    for row in config.to_dict("records"):
        tissue = row["tissue"].strip()
        matrix = matrices[tissue]
        # unversioned Ensembl ID, matching prepare_expression_matrix.py's own gene matching
        want = row["gene_id"].strip().split(".")[0]
        hits = [g for g in matrix.columns if str(g).split(".")[0] == want]
        if not hits:
            unresolved.append(f"{row['experiment']}: {row['gene_id']}@{tissue}")
            continue

        y = matrix[hits[0]].to_numpy()
        out_df = pd.DataFrame({"pheno_expression": y}, index=matrix.index)
        out_df.index.name = "IID"

        tag = slug(row["experiment"].strip())
        pheno_path = Path(f"{tag}_pheno.parquet").resolve()
        donors_path = Path(f"{tag}_donors.txt").resolve()
        out_df.to_parquet(pheno_path)
        with open(donors_path, "w") as fh:
            fh.write("\n".join(matrix.index) + "\n")

        resolved.append({**row, "pheno_path": str(pheno_path), "donors_path": str(donors_path)})

    if unresolved:
        lines = "\n".join(f"  {m}" for m in unresolved)
        raise SystemExit(
            f"[error] {len(unresolved)} gene(s) not found in their tissue's cached pheno_matrix "
            f"under {args.manifest_dir}:\n{lines}\n"
            "Run workflow 02 (expression_data_preparation) with matching parameters for these genes first."
        )
    pd.DataFrame(resolved).to_csv(args.out, sep="\t", index=False)
    print(f"[done] resolved {len(resolved)}/{len(config)} gene(s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
