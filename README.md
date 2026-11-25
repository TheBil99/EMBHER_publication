# EMBHER publication: analysis pipeline

Code to reproduce the results of *Sequence-to-function embeddings across individuals capture rare and common
variant heritability* (Billeci\*, Anglada-Girotto\*, Gusev, Frazer & Dias).

| Repository | Content |
|---|---|
| this repository | the full analysis pipeline (Nextflow + Snakemake), figures and tables |
| [TheBil99/EMBHER](https://github.com/TheBil99/EMBHER) | the EMBHER package (variance components, Haseman-Elston, whitening, extraction of per-layer Borzoi embeddings and attributions for personal genomes); pinned at commit `ebbad74` |
| Zenodo, [10.5281/zenodo.24246557](https://doi.org/10.5281/zenodo.24246557) | intermediate files: Borzoi attribution seqlets (hg38) and the supplementary tables |

## What is (and is not) here

Results derived from individual-level GTEx data are **not** distributed: GTEx genotypes, per-donor embeddings and
phenotypes, ancestry labels and the donor lists are controlled-access (dbGaP phs000424) and are rebuilt by the
pipeline from your own authorised copy. The repository contains gene lists (Ensembl ids), code and run commands
only. The supplementary tables are on Zenodo, not in this repository.

## Layout

```
workflows/      one directory per stage, in run order; 01 is Snakemake, all others Nextflow
  01-obtain_data  02-expression_data_preparation  03-gene_panels  04-genotype_extraction
  05-geuvadis_covariates_extraction  06-extract_embeddings  07-phenotype_simulations
  08-genotype_heritability_simulation_study  09-embedding_heritability_simulation_study
  10-genotype_heritability  11-embedding_heritability  12-heritability_over_layers
  13-ancestry_prediction  14-ancestry_effect_embeddings  15-borzoi_attribution
  16-rare_heritability_permutations  17-plots  99-intermediate_files
src/scripts/    scripts shared by two or more workflows         src/nf/   shared Nextflow helpers
src/embher_plotting/   plotting style and archetypes             envs/     conda environments
paper/          genes/ (gene lists), run_all.sh (the commands of the paper), figure_map.tsv
conf/ + nextflow.config   executor and site settings
```

`paper/figure_map.tsv` lists which workflow and script produce each figure and supplementary table, and
`paper/run_all.sh` the commands in dependency order.

## Setup

```bash
conda env create -f envs/embher_env.yaml      # CPU analysis (plink2, bcftools, htslib, embher @ ebbad74, ...)
mkdir -p logs && sbatch src/scripts/build_embher_gpu_env.sh   # Borzoi, attribution (workflows 06 and 15); or: bash ...
```

`embher_gpu` is built in two steps by `src/scripts/build_embher_gpu_env.sh` (`envs/embher_gpu.yaml`, then
decima, grelu, enformer-pytorch and embher installed with `--no-deps`, see the yaml header), so
`conda env create -f envs/embher_gpu.yaml` alone does not give a working env.

Everything is launched **from the repository root**: that directory is the project root (`data/`, `results/`
and `work/` hang off it, nothing else is configured). Paths, conda environment names and the SLURM
partitions are set once in `nextflow.config`; override them on the command line (`--env_analysis`,
`--cpu_queue`, `--gpu_queue`, `--gpu_gres`, `--conda_sh`) or in a git-ignored `conf/site.config`.

```bash
nextflow run workflows/10-genotype_heritability/genotype_heritability.nf -profile slurm --config_tsv ...
```

## Data

Public inputs are downloaded into `data/raw/` by `snakemake -s workflows/01-obtain_data/Snakefile --use-conda`
(GENCODE v46, Borzoi replicate 0, GEUVADIS, 1000 Genomes 30x phased panel, genomi-ancestry-panel, public GTEx v10
files, HOCOMOCO v13, phyloP, gnomAD v4.1.1). By hand: the GTEx phased WGS VCF, subject phenotypes and genotype
PCs from dbGaP (placed as read by workflow 02/04/06/13), the GTEx v10 eQTL covariates tar, and the CisBP-RNA
motifs (see the notes in `workflows/01-obtain_data/rules/`).

## Citation

If you use EMBHER, please cite:

> Federico Billeci, Miquel Anglada-Girotto, Alexander Gusev, Jonathan Frazer & Mafalda Dias. *Sequence-to-function embeddings across individuals capture rare and common variant heritability.* bioRxiv (2026). [Preprint](https://www.biorxiv.org/content/10.64898/2026.11.23.768687v1)

Software versions are in `envs/`.
