#!/usr/bin/env nextflow

/*
 * expression_data_preparation -- the single producer of per-tissue/per-cohort TPM
 * caches and expression phenotypes used across this project (GTEx and GEUVADIS;
 * see --cohort below).
 *
 * It caches its output in a manifest (src/scripts/resolve_expression.py /
 * write_expression_manifest_row.py, mirroring workflow 06's manifest pattern), and every
 * consumer looks a gene/tissue up there, failing fast if a required entry is missing.
 *
 * No covariate correction is applied (see prepare_expression_matrix.py): the
 * phenotype is the transformed value as-is, not even mean-centered. The phenotype
 * side still computes every gene in a tissue AT ONCE, not one gene at a time -- the
 * donor list is gene-independent for a fixed (tissue, transform), so there is one
 * process invocation per tissue, not one per gene. Consumers still get a per-gene
 * pheno.parquet/donors.txt back (resolve_expression.py slices one column out of the
 * cached matrix on demand).
 *
 * Two product types, both content-addressed by their own parameters (not by whatever
 * "experiment" label a consumer's config happens to use):
 *   tissue_matrix  raw genes x donors TPM for one tissue. Key = tissue alone.
 *   pheno_matrix   every gene's phenotype for one tissue, in one matrix
 *                  (prepare_expression_matrix.py). Key = tissue + --transform -- run
 *                  this workflow once per distinct transform a consumer needs (e.g.
 *                  once with --transform log2_plus_1 for workflows 11/12's flavor,
 *                  once with --transform quantile_int for 05/12's). No permutation
 *                  columns here: permutation count/seed is an estimator-side choice
 *                  (see src/scripts/add_permutations.py), applied downstream by
 *                  whichever workflow actually estimates heritability, not part of
 *                  this cache.
 *
 * Config TSV columns (tab-separated, with header), same format as workflow 11's:
 *   experiment  gene_id  gene_name  region  tissue
 * Only the 'tissue' column drives production now (every gene in that tissue's cached
 * matrix gets a phenotype, not just the ones listed) -- the other columns are kept so
 * any existing per-workflow config can be passed straight in unchanged.
 *
 * --cohort (gtex | geuvadis) picks which raw source this run's whole pipeline draws
 * from, and results land in a cohort-named subdirectory (results/
 * 02-expression_data_preparation/{gtex,geuvadis}/...) -- two independent manifests,
 * never sharing cache keys. The gtex branch is exactly the original pipeline
 * (vcf_metadata -> tissue_matrix -> prepare_expression_matrix). The geuvadis branch is
 * its counterpart over data/raw/GEUVADIS/GD660.GeneQuantRPKM.txt.gz (converted to TPM
 * per run, then collapsed from 660 sequencing runs to GEUVADIS's own 462 unique individuals -- see
 * scripts/prepare_geuvadis_tissue_matrix.py) restricted to the 1000G high-coverage
 * VCF's own sample panel (parity with the gtex branch's phased-WGS-only
 * restriction), then through the SAME prepare_expression_matrix.py --transform
 * quantile_int (GEUVADIS has no genotype-PCs/subject-phenotypes files, so that
 * script's membership filter is skipped for this cohort -- see its own docstring).
 * No covariate correction happens in this workflow either way (never did, for gtex) --
 * covariates are the heritability-estimation step's own concern downstream, not
 * phenotype construction's.
 *
 * GEUVADIS has no tissue dimension (GD660 is one matrix, not split by tissue); a
 * GEUVADIS config_tsv's 'tissue' column is just a cache-key tag, conventionally
 * "GEUVADIS_LCL" for every row -- the geuvadis branch does not use it to filter data,
 * only to key the manifest like every other tissue string.
 *
 * Usage:
 *   nextflow run workflows/02-expression_data_preparation/expression_data_preparation.nf \
 *     -profile slurm \
 *     --config_tsv workflows/11-embedding_heritability/configs/<config>.tsv \
 *     --transform log2_plus_1
 *
 *   nextflow run workflows/02-expression_data_preparation/expression_data_preparation.nf \
 *     -profile slurm \
 *     --cohort geuvadis \
 *     --config_tsv workflows/05-geuvadis_covariates_extraction/configs/all_genes.tsv \
 *     --transform quantile_int
 */

// ── Constants ─────────────────────────────────────────────────────────────── //

def SHARED_DIR    = "${projectDir}/../../src/scripts"
def SCRIPTS_DIR   = "${projectDir}/scripts"
def DATA          = "${launchDir}/data"
def GTEX_VCF      = "${DATA}/raw/dbgap/gtex/genotypes/phg001796.v1.GTEx_v9_WGS_phased.genotype-calls-vcf.c1/GTEx_Analysis_2021-02-11_v9_WholeGenomeSeq_944Indiv_Analysis_Freeze.SHAPEIT2_phased.vcf.gz"
def TPM_GCT       = "${DATA}/raw/GTEx/v10/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"
def SAMPLE_ATTR   = "${DATA}/raw/GTEx/v10/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"
def SUBJECT_PHEN  = "${DATA}/raw/GTEx/v10/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt"
def GENOTYPE_PCS  = "${DATA}/raw/GTEx/dbgap/phg001796.v1.GTEx_v9.genotype-qc.MULTI/GTEx_Analysis_2021-02-11_v9_WholeGenomeSeq_support_files/GTEx_Analysis_2021-02-11_v9_WholeGenomeSeq_953Indiv.20_Genotype_PCs.eigenvec.txt.gz"
def GEUVADIS_RPKM = "${DATA}/raw/GEUVADIS/GD660.GeneQuantRPKM.txt.gz"
def ONEKG_VCF     = "${DATA}/raw/1000genomes/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr21.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
def ANALYSIS_ENV  = params.env_analysis
def BCFTOOLS_ENV  = params.env_analysis
def CONDA_SH      = params.conda_sh

// ── Parameters ────────────────────────────────────────────────────────────── //

params.config_tsv = null
params.transform  = 'log2_plus_1'     // quantile_int | log2_plus_1
params.cohort     = 'gtex'            // gtex | geuvadis

if (!params.config_tsv) error "Please specify --config_tsv"
if (!(params.cohort in ['gtex', 'geuvadis'])) error "--cohort must be 'gtex' or 'geuvadis', got '${params.cohort}'"

def DATA_ROOT    = "${launchDir}/results/02-expression_data_preparation/${params.cohort}"
def MANIFEST_DIR = "${DATA_ROOT}/manifest"

def slug = { s -> s.replaceAll(/[^A-Za-z0-9]/, '_') }


// ──────────────────────────────────────────────────────────────────────────── //
// Process 1: donor IDs for the phased WGS panel (identical to every consumer's).
// ──────────────────────────────────────────────────────────────────────────── //

process vcf_metadata {
    cpus 1
    memory { 4.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    output:
    path("wgs_donors.txt")

    script:
    """
    source ${CONDA_SH}
    conda activate ${BCFTOOLS_ENV}
    bcftools query -l ${GTEX_VCF} > wgs_donors.txt
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 2: cache one tissue's gene-level TPM (genes x donors), and register it
//            in the manifest. Cached per tissue -- MAF/covariate/transform-
//            independent, so this is reused by every "pheno" flavor.
// ──────────────────────────────────────────────────────────────────────────── //

process tissue_matrix {
    array 1000
    cpus 2
    memory { 24.GB * task.attempt }
    time { 3.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${tissue}"

    input:
    tuple val(tissue_key), val(tissue), path(donors)

    output:
    tuple val(tissue_key), path("matrix.parquet")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${SHARED_DIR}/prepare_tissue_matrix.py \\
        --tpm-gct ${TPM_GCT} \\
        --sample-attributes ${SAMPLE_ATTR} \\
        --vcf-samples ${donors} \\
        --tissue '${tissue}' \\
        --out matrix.parquet

    python ${SHARED_DIR}/write_expression_manifest_row.py \\
        --data-dir ${DATA_ROOT} \\
        --manifest-dir ${MANIFEST_DIR} \\
        --product tissue_matrix \\
        --tissue '${tissue}' \\
        --matrix-file matrix.parquet
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 3: EVERY gene's phenotype in one tissue, at once (prepare_expression_
//            matrix.py) -- the donor list is gene-independent for a fixed (tissue,
//            transform), so this replaces one process invocation per gene with one
//            per tissue. Registers a single pheno_matrix manifest entry keyed by
//            tissue + --transform. No covariate correction is applied.
// ──────────────────────────────────────────────────────────────────────────── //

process prepare_expression_matrix {
    array 1000
    // The rank-based steps (quantile normalization, inverse-normal transform) are
    // ranked per-column (per donor, then per gene) independently, so
    // prepare_expression_matrix.py's --n-jobs parallelizes them across processes --
    // matched to this process's own cpu allocation.
    cpus 4
    memory { 8.GB * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${tissue}"

    input:
    tuple val(tissue_key), val(tissue), path(matrix)

    output:
    val(tissue_key)

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${SHARED_DIR}/prepare_expression_matrix.py \\
        --tissue-matrix ${matrix} \\
        --genotype-pcs ${GENOTYPE_PCS} \\
        --subject-phenotypes ${SUBJECT_PHEN} \\
        --transform ${params.transform} \\
        --n-jobs ${task.cpus} \\
        --out-pheno-matrix pheno_matrix.parquet \\
        --out-donors donors.txt

    python ${SHARED_DIR}/write_expression_manifest_row.py \\
        --data-dir ${DATA_ROOT} \\
        --manifest-dir ${MANIFEST_DIR} \\
        --product pheno_matrix \\
        --tissue '${tissue}' \\
        --transform ${params.transform} \\
        --pheno-matrix-file pheno_matrix.parquet \\
        --donors-file donors.txt
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 1b (geuvadis branch): 1000G high-coverage VCF's own sample panel -- parity
//            with vcf_metadata's "only donors with phased WGS can be genotyped"
//            restriction, against the genotype source workflow 05's PEER prep and any
//            future GEUVADIS genotype-extraction step both use.
// ──────────────────────────────────────────────────────────────────────────── //

process geuvadis_vcf_metadata {
    cpus 1
    memory { 4.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    output:
    path("onekg_donors.txt")

    script:
    """
    source ${CONDA_SH}
    conda activate ${BCFTOOLS_ENV}
    bcftools query -l ${ONEKG_VCF} > onekg_donors.txt
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 2b (geuvadis branch): GEUVADIS's counterpart of tissue_matrix -- one cached
//            genes x donors TPM matrix (GD660 RPKM -> TPM, collapsed to GD462), registered in the
//            same manifest convention. 'tissue' is a cache-key tag only here (see
//            module docstring); the data read doesn't depend on it.
// ──────────────────────────────────────────────────────────────────────────── //

process geuvadis_tissue_matrix {
    array 1000
    cpus 1
    memory { 4.GB * task.attempt }
    time { 20.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${tissue}"

    input:
    tuple val(tissue_key), val(tissue), path(donors)

    output:
    tuple val(tissue_key), path("matrix.parquet")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${SCRIPTS_DIR}/prepare_geuvadis_tissue_matrix.py \\
        --rpkm ${GEUVADIS_RPKM} \\
        --vcf-samples ${donors} \\
        --out matrix.parquet

    python ${SHARED_DIR}/write_expression_manifest_row.py \\
        --data-dir ${DATA_ROOT} \\
        --manifest-dir ${MANIFEST_DIR} \\
        --product tissue_matrix \\
        --tissue '${tissue}' \\
        --matrix-file matrix.parquet
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 3b (geuvadis branch): same prepare_expression_matrix.py as the gtex branch,
//            just without --genotype-pcs/--subject-phenotypes (GEUVADIS has neither;
//            that script's membership filter is skipped entirely -- see its
//            docstring). No covariate correction here either, same as gtex.
// ──────────────────────────────────────────────────────────────────────────── //

process geuvadis_prepare_expression_matrix {
    array 1000
    cpus 4
    memory { 8.GB * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${tissue}"

    input:
    tuple val(tissue_key), val(tissue), path(matrix)

    output:
    val(tissue_key)

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${SHARED_DIR}/prepare_expression_matrix.py \\
        --tissue-matrix ${matrix} \\
        --transform ${params.transform} \\
        --n-jobs ${task.cpus} \\
        --out-pheno-matrix pheno_matrix.parquet \\
        --out-donors donors.txt

    python ${SHARED_DIR}/write_expression_manifest_row.py \\
        --data-dir ${DATA_ROOT} \\
        --manifest-dir ${MANIFEST_DIR} \\
        --product pheno_matrix \\
        --tissue '${tissue}' \\
        --transform ${params.transform} \\
        --pheno-matrix-file pheno_matrix.parquet \\
        --donors-file donors.txt
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
workflow {

    rows = Channel
        .fromPath(params.config_tsv)
        .splitCsv(header: true, sep: '\t')
        .map { r ->
            if (!r.experiment || !r.gene_id || !r.region || !r.tissue)
                error "config row missing a required field: ${r}"
            tuple(r.experiment.trim(), r.gene_id.trim(), r.region.trim(), r.tissue.trim())
        }

    // one cached TPM matrix, and one cached pheno_matrix, per distinct tissue
    tissues = rows.map { e, g, reg, t -> tuple(slug(t), t) }.unique()

    if (params.cohort == 'gtex') {
        donors = vcf_metadata()
        matrices = tissue_matrix(tissues.combine(donors))
        prepare_expression_matrix(matrices.combine(tissues, by: 0).map { tk, m, t -> tuple(tk, t, m) })
    } else {
        donors = geuvadis_vcf_metadata()
        matrices = geuvadis_tissue_matrix(tissues.combine(donors))
        geuvadis_prepare_expression_matrix(matrices.combine(tissues, by: 0).map { tk, m, t -> tuple(tk, t, m) })
    }
}
