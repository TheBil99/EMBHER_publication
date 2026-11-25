#!/usr/bin/env python
"""Build PEER's input expression matrix from workflow 02's GEUVADIS TPM cache.

--tissue-matrix is workflow 02's cached genes x donors TPM parquet for GEUVADIS_LCL
(GD660 RPKM -> TPM per run, technical replicates averaged per individual, restricted to
the donors in the 1000G high-coverage VCF; see 02's prepare_geuvadis_tissue_matrix.py).
It is NOT GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz, GEUVADIS's own
already-PEER-corrected output (k=10 factors already regressed out), which would mean
computing PEER factors on residuals that already had PEER applied.

Gene filter mirrors GEUVADIS's own "50FN" ("50% filtered, non-zero") convention
described in data/raw/GEUVADIS/GeuvadisRNASeqAnalysisFiles_README.txt: drop a gene if
TPM==0 in more than --zero-frac-max of the donors. GEUVADIS's own filter was applied to
raw counts; TPM==0 iff the underlying count is 0, so this is equivalent for this binary
filter.

log2(TPM + --pseudocount) is applied gene-wise before writing the PEER input -- not
part of GEUVADIS's own published pipeline but standard practice for PEER on RNA-seq,
which is heavily right-skewed on the raw scale.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tissue-matrix", required=True,
                     help="workflow 02's GEUVADIS_LCL tissue_matrix parquet (genes x donors TPM)")
    ap.add_argument("--zero-frac-max", type=float, default=0.5,
                     help="drop a gene if TPM==0 in more than this fraction of the donors")
    ap.add_argument("--pseudocount", type=float, default=1.0, help="log2(TPM + pseudocount)")
    ap.add_argument("--out-expr", required=True, help="PEER input: samples x genes, headerless CSV")
    ap.add_argument("--out-samples", required=True, help="individual IDs, one per line, row order of --out-expr")
    ap.add_argument("--out-genes", required=True, help="TargetID, one per line, column order of --out-expr")
    args = ap.parse_args()

    expr = pd.read_parquet(args.tissue_matrix).T   # donors x genes
    print(f"[info] {expr.shape[1]} genes x {expr.shape[0]} donors from {args.tissue_matrix}",
          flush=True)

    zero_frac = (expr == 0).mean(axis=0)
    keep = (zero_frac <= args.zero_frac_max).to_numpy()
    print(f"[info] keeping {keep.sum()}/{len(keep)} genes with TPM==0 in <= "
          f"{args.zero_frac_max:.0%} of donors", flush=True)

    kept_targets = expr.columns[keep].to_numpy()
    expr = np.log2(expr.loc[:, keep] + args.pseudocount)
    expr = expr.sort_index()

    np.savetxt(args.out_expr, expr.to_numpy(), delimiter=",")
    with open(args.out_samples, "w") as fh:
        fh.write("\n".join(expr.index.tolist()) + "\n")
    with open(args.out_genes, "w") as fh:
        fh.write("\n".join(kept_targets.tolist()) + "\n")

    print(f"[done] wrote {expr.shape[0]} samples x {expr.shape[1]} genes -> {args.out_expr}", flush=True)


if __name__ == "__main__":
    main()
