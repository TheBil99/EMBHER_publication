"""
Prepare maf_ld.txt and annot.txt for Simulator_mafld.

maf_ld.txt format (header + M rows, space-separated, in BIM order):
    MAF ldscore

annot.txt format (M rows, no header, single column of 1s):
    1
    1
    ...

MAF source: bcftools-extracted AF INFO field from the 1KG VCF (columns: ID  AF).
            MAF is computed as min(AF, 1-AF).  One or more per-chromosome
            {chr}_af.txt files are supplied via --af_files and concatenated.

LD scores:  ldscore.l2.ldscore.gz produced by compute_ld_scores_plink.py
            Columns (tab-sep, gzipped): CHR  SNP  BP  L2
            All-zero when ld_ex == 0 (placeholder path).
"""

import argparse
import gzip

import pandas as pd


MAF_DEFAULT     = 0.01
LDSCORE_DEFAULT = 1.0


def read_bim(bim_path):
    return pd.read_csv(
        bim_path, sep=r"\s+", header=None,
        names=["CHR", "ID", "CM", "BP", "A1", "A2"],
        dtype={"CHR": str, "ID": str, "BP": int},
    )


def read_af_files(af_file_paths):
    """
    Read one or more per-chromosome af.txt files (columns: ID  AF, no header).
    Returns dict {variant_id -> AF}.
    """
    af_dict = {}
    for path in af_file_paths:
        with open(path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                vid, af_str = parts[0], parts[1]
                try:
                    af_dict[vid] = float(af_str)
                except ValueError:
                    pass
    return af_dict


def read_ldscore(ldscore_gz):
    ld_dict = {}
    with gzip.open(ldscore_gz, "rt") as fh:
        header = fh.readline().rstrip("\n").split()
        snp_idx = header.index("SNP")
        l2_idx  = header.index("L2")
        for line in fh:
            parts = line.rstrip("\n").split()
            if len(parts) <= max(snp_idx, l2_idx):
                continue
            try:
                ld_dict[parts[snp_idx]] = float(parts[l2_idx])
            except ValueError:
                pass
    return ld_dict


def main(args):
    bim = read_bim(args.bim)
    M = len(bim)
    print(f"[prepare_maf_ld] BIM: {M} variants", flush=True)

    af_dict = read_af_files(args.af_files)
    print(f"[prepare_maf_ld] AF entries loaded: {len(af_dict)}", flush=True)

    ld_dict = read_ldscore(args.ldscore)
    print(f"[prepare_maf_ld] LD score entries loaded: {len(ld_dict)}", flush=True)

    mafs     = []
    ldscores = []
    n_maf_miss = 0
    n_ld_miss  = 0

    for _, row in bim.iterrows():
        vid = row["ID"]

        if vid in af_dict:
            af  = af_dict[vid]
            maf = min(af, 1.0 - af)
        else:
            maf = MAF_DEFAULT
            n_maf_miss += 1

        if vid in ld_dict:
            ldscore = max(ld_dict[vid], 0.0)
        else:
            ldscore = LDSCORE_DEFAULT
            n_ld_miss += 1

        mafs.append(maf)
        ldscores.append(ldscore)

    print(f"[prepare_maf_ld] MAF fallbacks (ID not found in AF file): {n_maf_miss}/{M}", flush=True)
    print(f"[prepare_maf_ld] LD fallbacks:                            {n_ld_miss}/{M}", flush=True)

    with open(args.maf_ld_out, "w") as fh:
        fh.write("MAF ldscore\n")
        for maf, ldscore in zip(mafs, ldscores):
            fh.write(f"{maf:.8f} {ldscore:.6f}\n")

    with open(args.annot_out, "w") as fh:
        for _ in range(M):
            fh.write("1\n")

    print(f"[prepare_maf_ld] Written {args.maf_ld_out} and {args.annot_out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare maf_ld.txt and annot.txt for Simulator_mafld."
    )
    parser.add_argument("--bim",        required=True,
                        help="plink BIM of the merged filtered variants")
    parser.add_argument("--af_files",   required=True, nargs="+",
                        help="Per-chromosome AF files (ID<tab>AF, no header)")
    parser.add_argument("--ldscore",    required=True,
                        help="ldscore.l2.ldscore.gz (CHR SNP BP L2, gzipped)")
    parser.add_argument("--maf_ld_out", default="maf_ld.txt")
    parser.add_argument("--annot_out",  default="annot.txt")
    main(parser.parse_args())
