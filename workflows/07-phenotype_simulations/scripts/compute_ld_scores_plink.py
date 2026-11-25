"""
Aggregate plink --r2 long-format output into an ldsc-compatible ldscore file.

LD score definition (same as ldsc):
    l_i = sum_{j in window} r²(i, j)   (including self: r²(i,i) = 1)

Input:
    --r2_ld   : plink .ld file produced by plink --r2 (long format)
    --bim     : plink BIM of the filtered variants (defines variant order)
    --out     : output path for ldscore.l2.ldscore.gz

Output format (tab-separated, gzipped):
    CHR  SNP  BP  L2
"""

import argparse
import gzip

import pandas as pd


def main(args):
    bim = pd.read_csv(
        args.bim, sep="\t", header=None,
        names=["CHR", "ID", "CM", "BP", "A1", "A2"],
        dtype={"CHR": str, "ID": str, "BP": int},
    )
    print(f"[ld_plink] BIM: {len(bim)} variants", flush=True)

    # plink --r2 columns: CHR_A  BP_A  SNP_A  CHR_B  BP_B  SNP_B  R2
    r2 = pd.read_csv(args.r2_ld, sep=r"\s+", dtype={"SNP_A": str, "SNP_B": str})
    print(f"[ld_plink] r2 pairs loaded: {len(r2)}", flush=True)

    if r2.empty:
        bim["L2"] = 1.0
        _write(bim, args.out)
        return

    scores_a = r2.groupby("SNP_A", sort=False)["R2"].sum().rename("L2")
    scores_b = r2.groupby("SNP_B", sort=False)["R2"].sum().rename("L2")

    scores = (
        pd.concat([scores_a, scores_b])
        .groupby(level=0)
        .sum()
        .rename_axis("ID")
        .reset_index()
    )

    result = bim[["CHR", "ID", "BP"]].merge(scores, on="ID", how="left")
    result["L2"] = result["L2"].fillna(0.0) + 1.0   # add self r²=1

    n_miss = (result["L2"] == 1.0).sum()
    print(f"[ld_plink] Variants with no off-diagonal r² pairs: {n_miss}/{len(result)}", flush=True)

    _write(result, args.out)


def _write(df: pd.DataFrame, out_path: str) -> None:
    with gzip.open(out_path, "wt") as fh:
        fh.write("CHR\tSNP\tBP\tL2\n")
        for _, row in df.iterrows():
            fh.write(f"{row['CHR']}\t{row['ID']}\t{int(row['BP'])}\t{row['L2']:.6f}\n")
    print(f"[ld_plink] Written {out_path} ({len(df)} variants)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert plink --r2 output to ldsc-compatible ldscore.gz"
    )
    parser.add_argument("--r2_ld",  required=True,
                        help="plink .ld file from plink --r2 --ld-window-r2 0")
    parser.add_argument("--bim",    required=True,
                        help="plink BIM file of the filtered variants")
    parser.add_argument("--out",    default="ldscore.l2.ldscore.gz",
                        help="output gzip path")
    main(parser.parse_args())
