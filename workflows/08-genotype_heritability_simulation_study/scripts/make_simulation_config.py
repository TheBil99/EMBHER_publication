#!/usr/bin/env python3
"""
Build workflow 08's --config_tsv (experiment, true_h2, p_causal, bfile_prefix, pheno_parquet) from workflow 07's
config and output directory, one row per (gene, h2, p_causal) condition (replicate seed = the config row's seed).

Workflow 07 names its outputs after the region and the simulation settings:
  <sim_out>/genotypes/region_<chrom>_<start>_<end>_maf<min>_<max>_ld<ld_ex>_mafex<maf_ex>_<panel>/merged{.bed,...}
  <sim_out>/experiments/h2_<h2>_pc_<p_causal>_rep_<seed>[_cmaf<min>_<max>]/simulated_phenotypes/pheno.parquet
The experiment is prefixed with the gene symbol: the GENCODE gene whose TSS is at the centre of the region.

Usage (from the project root):
  python workflows/08-genotype_heritability_simulation_study/scripts/make_simulation_config.py \
    --sim-config workflows/07-phenotype_simulations/configs/ten_random_genes_maf_bins_calibration.tsv \
    --sim-out-dir results/07-phenotype_simulations/simulations/ten_random_genes_maf_bins_calibration \
    --out <config.tsv>
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-config", required=True, help="workflow 07's config TSV")
    ap.add_argument("--sim-out-dir", required=True, help="workflow 07's results/.../simulations/<out_dir>")
    ap.add_argument("--gencode-parquet", default="data/raw/GENCODE/human/release_46/gencode.v46.annotation.gtf.parquet")
    ap.add_argument("--panel", default="gtex", choices=["gtex", "1000g"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = pd.read_csv(a.sim_config, sep="\t", dtype=str)
    g = pd.read_parquet(a.gencode_parquet, columns=["Chromosome", "Feature", "Start", "End", "Strand", "gene_name"])
    g = g[g["Feature"] == "gene"].copy()
    g["tss_hg38"] = g["Start"].where(g["Strand"] == "+", g["End"]) + (g["Strand"] == "+").astype(int)  # 1-based
    genes = g.rename(columns={"Chromosome": "chr_hg38", "gene_name": "gene_symbol"}).astype({"chr_hg38": str})
    out = Path(a.sim_out_dir).resolve()
    rows = []
    for _, r in cfg.iterrows():
        chrom, start, end = re.fullmatch(r"(chr[0-9XY]+):(\d+)-(\d+)", r["region"]).groups()
        mid = (int(start) + int(end)) // 2
        hit = genes[(genes["chr_hg38"] == chrom) & ((genes["tss_hg38"] - mid).abs() <= 1)]
        if len(hit) != 1:
            raise SystemExit(f"{r['region']}: expected one gene with its TSS at the region centre, found {len(hit)}")
        sym = hit["gene_symbol"].iloc[0]
        geno = out / "genotypes" / (f"region_{chrom}_{start}_{end}_maf{r['min_maf']}_{r['max_maf']}"
                                    f"_ld{r['ld_ex']}_mafex{r['maf_ex']}_{a.panel}") / "merged"
        for h2 in r["h2"].split(","):
            for pc in r["p_causal"].split(","):
                exp = f"h2_{h2}_pc_{pc}_rep_{r['seed']}"
                cmin, cmax = r.get("causal_min_maf"), r.get("causal_max_maf")
                if pd.notna(cmin) and pd.notna(cmax) and (float(cmin), float(cmax)) != (0.0, 0.5):
                    exp += f"_cmaf{cmin}_{cmax}"      # causal variants drawn from a MAF band (workflow 07)
                rows.append({"experiment": f"{sym}_{exp}", "true_h2": h2, "p_causal": pc,
                             "bfile_prefix": str(geno),
                             "pheno_parquet": str(out / "experiments" / exp / "simulated_phenotypes" / "pheno.parquet")})
    pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)
    print(f"[done] {len(rows)} conditions -> {a.out}")


if __name__ == "__main__":
    main()
