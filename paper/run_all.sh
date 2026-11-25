#!/usr/bin/env bash
# The commands that produced the paper, in dependency order. Run from the project root. This is documentation
# you can execute step by step, not a one-shot script: every step is a Nextflow run (-profile slurm) that takes
# hours to days, and steps 4-6 need controlled-access GTEx data (see README.md).
#
# Gene lists are in paper/genes/ (Ensembl ids only, no donor information). paper/figure_map.tsv says which
# result feeds which figure/table.
set -euo pipefail
NF="nextflow run -profile slurm"
GENES=paper/genes
GTEX=results/06-extract_embeddings/gtex
GEUV=results/06-extract_embeddings/geuvadis
KEY=borzoi_replicate_0_release_46_tss_centered_out_rna_seq_sum_mean_gene_exon_intron_bfloat16   # 06's cache_key
RANDKEY=random_init_seed0_release_46_tss_centered_out_rna_seq_sum_mean_gene_exon_intron_bfloat16
DONORS=whole_blood_donors_config                                                                  # name of the donors YAML (step 4)

# 0. environments, clones, raw data ----------------------------------------------------------------------------
#   conda env create -f envs/embher_env.yaml                       # CPU analysis env (installs embher)
#   mkdir -p logs && sbatch src/scripts/build_embher_gpu_env.sh   # GPU env: yaml + --no-deps installs (embher-capture, embher-attribute)
#   git clone https://github.com/sriramlab/Simulator src/Simulator # + build Simulator_mafld (workflow 07, see its header)
#   snakemake -s workflows/01-obtain_data/Snakefile --use-conda --cores 1     # public inputs
#   (place the dbGaP files by hand: README.md)
#   sbatch workflows/05-geuvadis_covariates_extraction/build_peer.sh          # PEER, for the GEUVADIS covariates

# 1. expression phenotypes (quantile normalisation + inverse-normal transform) ---------------------------------
$NF workflows/02-expression_data_preparation/expression_data_preparation.nf \
  --config_tsv $GENES/11_hi_q_random500.tsv --transform quantile_int
$NF workflows/02-expression_data_preparation/expression_data_preparation.nf --cohort geuvadis \
  --config_tsv $GENES/11_hi_q_geuvadis_random500.tsv --transform quantile_int

# 2. gene panels: high-quality genes (TPM > 0.1 in >= 50% of donors, Borzoi-viable) --------------------------------
python workflows/03-gene_panels/scripts/build_hi_q_panels.py --help     # writes results/03-gene_panels/{panels,gene_meta}/
#   paper/genes/hi_q_*.tsv are the panels it produced for the paper.

# 3. genotype extraction (region PLINK bfiles per gene) and GEUVADIS covariates ----------------------------------
$NF workflows/04-genotype_extraction/genotype_extraction.nf --config_tsv $GENES/10_hi_q_gtex_genes.tsv --maf_bands 0.0
$NF workflows/04-genotype_extraction/genotype_extraction.nf --cohort geuvadis --config_tsv $GENES/10_hi_q_geuvadis_genes.tsv --maf_bands 0.0
$NF workflows/05-geuvadis_covariates_extraction/geuvadis_covariates_extraction.nf

# 4. Borzoi embeddings (GPU). donors YAMLs: python workflows/06-extract_embeddings/scripts/make_donors_config.py --help;
#    gene_meta: results/03-gene_panels/gene_meta/<panel>.tsv ---------------------------------------------------------
$NF workflows/06-extract_embeddings/extract_embeddings.nf --include_reference false --pooling_axis seq_dim --pooling_func mean \
  --gene_meta_tsv results/03-gene_panels/gene_meta/hi_q_random500.tsv --donors_config configs/all_donors_config.yaml
$NF workflows/06-extract_embeddings/extract_embeddings.nf --run_borzoi false --run_borzoi_random true \
  --include_reference false --pooling_axis seq_dim --pooling_func mean \
  --gene_meta_tsv results/03-gene_panels/gene_meta/hi_q_random500.tsv --donors_config configs/all_donors_config.yaml
$NF workflows/06-extract_embeddings/extract_embeddings.nf --max_maf 0.01 --maf_label le0.01 \
  --include_reference false --pooling_axis seq_dim --pooling_func mean \
  --gene_meta_tsv results/03-gene_panels/gene_meta/hi_q_gtex_genes.tsv --donors_config configs/${DONORS}.yaml
$NF workflows/06-extract_embeddings/extract_embeddings.nf --cohort geuvadis --include_reference false \
  --pooling_axis seq_dim --pooling_func mean \
  --gene_meta_tsv results/03-gene_panels/gene_meta/hi_q_geuvadis_random500.tsv --donors_config configs/geuvadis_donors_config.yaml

# 5. heritability: genotype (h2G), embedding/prediction (h2E, h2P), layers -----------------------------------------
$NF workflows/10-genotype_heritability/genotype_heritability.nf --config_tsv $GENES/10_hi_q_gtex_genes.tsv \
  --estimator he --jackknife none --binning mac_ld_6 --batch_size 5 --out_dir hi_q_gtex_genes
$NF workflows/10-genotype_heritability/genotype_heritability.nf --cohort geuvadis --config_tsv $GENES/10_hi_q_geuvadis_genes.tsv \
  --estimator he --jackknife none --binning mac_ld_6 --batch_size 5 --out_dir hi_q_geuvadis_genes
$NF workflows/11-embedding_heritability/embedding_heritability.nf --config_tsv $GENES/11_hi_q_random500.tsv \
  --embeddings_dir $GTEX/borzoi/$KEY --estimator he --joint false --out_dir hi_q_random500
$NF workflows/11-embedding_heritability/embedding_heritability.nf --cohort geuvadis --config_tsv $GENES/11_hi_q_geuvadis_random500.tsv \
  --embeddings_dir $GEUV/borzoi/$KEY --estimator he --joint false --out_dir hi_q_geuvadis_random500
$NF workflows/12-heritability_over_layers/heritability_over_layers.nf --config_tsv $GENES/12_hi_q_random500.tsv \
  --embeddings_dir $GTEX/borzoi/$KEY --estimator he --out_dir hi_q_random500
$NF workflows/12-heritability_over_layers/heritability_over_layers.nf --config_tsv $GENES/12_hi_q_random500.tsv \
  --embeddings_dir $GTEX/borzoi_random/$RANDKEY --embeddings_model_type borzoi_random --estimator he \
  --out_dir hi_q_random500_borzoi_random

# 6. ancestry, attribution, rare variants -------------------------------------------------------------------------
$NF workflows/13-ancestry_prediction/ancestry_prediction.nf --config_tsv $GENES/11_hi_q_random500.tsv \
  --embeddings_dir $GTEX/borzoi/$KEY --out_dir hi_q_random500
$NF workflows/14-ancestry_effect_embeddings/ancestry_effect_embeddings.nf --config_tsv $GENES/14_faap20.tsv \
  --embeddings_dir $GTEX/borzoi/$KEY --comparison eur_afr --out_dir faap20
$NF workflows/15-borzoi_attribution/borzoi_attribution.nf --config_tsv $GENES/15_hi_q_random500.tsv --run_name hi_q_random500
$NF workflows/16-rare_heritability_permutations/rare_heritability_permutations.nf --config_tsv $GENES/16_hi_q_gtex_genes.tsv \
  --embeddings_dir $GTEX/borzoi/${KEY}_maf_le0.01_cohort_${DONORS} --n_perm 1000 --pc_frac 0.7 --out_dir hi_q_gtex_genes_maf_le0.01

# 7. simulations (Supp. Note 2): phenotypes on real GTEx genotypes (07), genotype estimator (08), embedding estimator (09) -------
$NF workflows/07-phenotype_simulations/simulate_phenotypes.nf \
  --config_tsv $GENES/07_ten_random_genes_maf_bins_calibration.tsv --out_dir ten_random_genes_maf_bins_calibration
$NF workflows/07-phenotype_simulations/simulate_phenotypes.nf \
  --config_tsv $GENES/07_ten_random_genes_calibration.tsv --out_dir ten_random_genes_calibration
python workflows/08-genotype_heritability_simulation_study/scripts/make_simulation_config.py \
  --sim-config $GENES/07_ten_random_genes_maf_bins_calibration.tsv \
  --sim-out-dir results/07-phenotype_simulations/simulations/ten_random_genes_maf_bins_calibration \
  --out results/08_config_ten_random_genes_maf_bins_calibration.tsv
$NF workflows/08-genotype_heritability_simulation_study/genotype_heritability_simulation_study.nf \
  --config_tsv results/08_config_ten_random_genes_maf_bins_calibration.tsv --estimator he --no_se \
  --out_dir 10_random_genes_he
$NF workflows/09-embedding_heritability_simulation_study/embedding_heritability_simulation_study.nf \
  --config_tsv $GENES/09_ten_random_genes.tsv --embeddings_dir $GTEX/borzoi/$KEY \
  --genosim_config_tsv $GENES/07_ten_random_genes_calibration.tsv --genosim_out_dir ten_random_genes_calibration \
  --whiten_pc_frac 0.7,0.9 --estimator he --n_reps 100 --out_dir 10_random_genes_he

# 8. figures and tables (reads the results above), and the Zenodo seqlet bundle -----------------------------------
$NF workflows/17-plots/plots.nf
$NF workflows/99-intermediate_files/intermediate_files.nf --config_tsv $GENES/15_hi_q_random500.tsv --run_name hi_q_random500
