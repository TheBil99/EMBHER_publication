#!/usr/bin/env nextflow

/*
 * ancestry_effect_embeddings -- how strongly do Borzoi embeddings separate ancestry
 * groups, compared with the genotypes they are computed from?
 *
 * Per gene, on the same window (Borzoi's 524,288 bp around the TSS), all per donor:
 *   donor_features    genotype = 0/1/2 non-reference allele counts from workflow 04's
 *                     full region bfile (scripts/genotype_features.py, via the shared
 *                     read_bed; missing calls counted as reference); embedding =
 *                     final__seq_mean with the two haplotypes summed, from workflow
 *                     04b's H5 (src/scripts/combine_haplotypes.py; its prediction output
 *                     is not used). Embeddings are never whitened here.
 *   ancestry_effect   intra vs inter distances (scripts/intra_inter_distances.py): each
 *                     donor's mean Euclidean distance to the other donors of its own
 *                     group and to the donors of the other group; and PC1 vs PC2 with
 *                     EUR / other-group marginals (scripts/pca_ancestry.py, all donors,
 *                     columns centred only). Both pages merged into <gene>_report.pdf.
 * Then (merge_report): every gene's report in one PDF, and the per-gene intra/inter
 * summaries in one TSV.
 *
 * Groups (scripts/make_ancestry_groups.py) come from GTEx self-reported race (dbGaP
 * pht002742): --comparison eur_afr (White vs Black or African American; other donors
 * left out of the intra/inter distances, grey in the PCA) or eur_noneur (White vs all
 * others).
 *
 * Inputs: workflow 04's region bfiles (full variant set, looked up via its manifest)
 * and an unfiltered workflow 06 run (--embeddings_dir, looked up via its manifest), so
 * the genotypes and the embeddings see the same variants.
 *
 * Config TSV columns: experiment  gene_id  region  [extra columns ignored] -- workflow
 * 11's configs work as-is.
 *
 * Usage (from the project root):
 *   nextflow run workflows/14-ancestry_effect_embeddings/ancestry_effect_embeddings.nf \
 *     -profile slurm \
 *     --config_tsv workflows/11-embedding_heritability/configs/whole_blood_16_new.tsv \
 *     --embeddings_dir results/06-extract_embeddings/gtex/borzoi/<unfiltered cache_key> \
 *     --comparison eur_afr --out_dir custom16
 */

include { loadGenotypeManifest; lookupGenoBfile } from '../../src/nf/genotype_manifest.nf'

// ── Constants ─────────────────────────────────────────────────────────────── //

def SCRIPTS_DIR  = "${projectDir}/scripts"
def SHARED_DIR   = "${projectDir}/../../src/scripts"
def ANALYSIS_ENV = params.env_analysis
def CONDA_SH     = params.conda_sh
def RACE_FILE    = "${projectDir}/../../data/raw/dbgap/gtex/phenotypes/phs000424.v11.pht002742.v9.p2.c1.GTEx_Subject_Phenotypes.GRU.txt.gz"
// combine_haplotypes.py always writes a prediction table too; any valid tracks will do
// (GTEx whole-blood RNA-seq), the prediction output is discarded.
def UNUSED_TRACKS = '7531,7532,7533'

// ── Parameters ────────────────────────────────────────────────────────────── //

params.out_dir               = null
params.config_tsv            = null
params.embeddings_dir        = null   // unfiltered 04b run
params.comparison            = 'eur_afr'
params.window_size           = 524288
params.vcf_filter            = null
params.genotype_manifest_dir = null

if (!params.out_dir)        error "Please specify --out_dir"
if (!params.config_tsv)     error "Please specify --config_tsv"
if (!params.embeddings_dir) error "Please specify --embeddings_dir (unfiltered 04b run)"
if (!(params.comparison in ['eur_afr', 'eur_noneur']))
    error "--comparison must be eur_afr or eur_noneur (got '${params.comparison}')"

def OUT_PREFIX        = "${launchDir}/results/14-ancestry_effect_embeddings/${params.out_dir}/${params.comparison}"
def GENOTYPE_DATA_DIR = "${launchDir}/results/04-genotype_extraction/gtex"
def GENOTYPE_MANIFEST = loadGenotypeManifest(params.genotype_manifest_dir ?: "${GENOTYPE_DATA_DIR}/manifest")
def FULL_BAND = [min: 0.0, max: null]

// 04's manifest reader (src/nf/embeddings_manifest.nf): the 04 directory name no longer carries the
// pooling, so the rows are filtered on the manifest's own pooling columns.
include { loadEmbeddingsManifest } from '../../src/nf/embeddings_manifest.nf'
def MANIFEST = loadEmbeddingsManifest(params.embeddings_dir, [pooling_axes: ['seq_dim'], pooling_funcs: ['mean']])
def anyRow = MANIFEST.values().find { true }
if (anyRow?.min_maf || anyRow?.max_maf)
    error "--embeddings_dir must be an unfiltered 04b run (got min_maf='${anyRow.min_maf}', max_maf='${anyRow.max_maf}')"


// ──────────────────────────────────────────────────────────────────────────── //
// Process: ancestry groups for --comparison (scripts/make_ancestry_groups.py).
// ──────────────────────────────────────────────────────────────────────────── //

process make_groups {
    cpus 1
    memory { 2.GB * task.attempt }
    time { 10.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}" }, mode: 'copy'

    output:
    path("groups_${params.comparison}.tsv")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    python ${SHARED_DIR}/make_ancestry_groups.py \\
        --phenotypes ${RACE_FILE} --comparison ${params.comparison} \\
        --out groups_${params.comparison}.tsv
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process: donor x feature tables for one gene -- genotype counts
//          (scripts/genotype_features.py) and haplotype-summed embedding
//          (src/scripts/combine_haplotypes.py).
// ──────────────────────────────────────────────────────────────────────────── //

process donor_features {
    array 1000
    cpus 1
    memory { 8.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${gene}"

    input:
    tuple val(gene), val(gene_id), path(h5), path(bed), path(bim), path(fam)

    output:
    tuple val(gene), path("${gene}_genotype.parquet"), path("${gene}_embedding.parquet")

    script:
    def gid = gene_id.tokenize('.')[0]
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    python ${SHARED_DIR}/genotype_features.py \\
        --bfile ${bed.baseName} --out ${gene}_genotype.parquet
    python ${SHARED_DIR}/combine_haplotypes.py \\
        --h5 ${h5} --gene-id ${gid} --tracks ${UNUSED_TRACKS} \\
        --out-embedding ${gene}_embedding.parquet --out-prediction unused_prediction.parquet
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process: intra vs inter distances and PCA for one gene, merged into one PDF
//          (scripts/intra_inter_distances.py, scripts/pca_ancestry.py).
// ──────────────────────────────────────────────────────────────────────────── //

process ancestry_effect {
    array 1000
    cpus 2
    memory { 8.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${gene}"

    publishDir { "${OUT_PREFIX}/genes/${gene}" }, mode: 'copy'

    input:
    tuple val(gene), path(genotype), path(embedding)
    path(groups)

    output:
    path("${gene}_intra_inter_distances*")
    path("${gene}_pca_ancestry*")
    path("${gene}_report.pdf"), emit: report
    path("${gene}_intra_inter_distances_summary.tsv"), emit: summary

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    python ${SCRIPTS_DIR}/intra_inter_distances.py \\
        --gene '${gene}' --groups ${groups} --genotype ${genotype} --embedding ${embedding}
    python ${SCRIPTS_DIR}/pca_ancestry.py \\
        --gene '${gene}' --groups ${groups} --genotype ${genotype} --embedding ${embedding} \\
        --window-bp ${params.window_size}
    gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=${gene}_report.pdf \\
        ${gene}_intra_inter_distances_violin.pdf ${gene}_pca_ancestry.pdf
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process: every gene's report in one PDF (config order) and the per-gene
//          intra/inter summaries in one TSV.
// ──────────────────────────────────────────────────────────────────────────── //

process merge_report {
    cpus 1
    memory { 4.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}" }, mode: 'copy'

    input:
    val(gene_order)
    path(reports)
    path(summaries)

    output:
    path("ancestry_effect_report.pdf")
    path("intra_inter_distances_summary.tsv")

    script:
    def ordered = gene_order.collect { "'${it}_report.pdf'" }.join(' ')
    """
    gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=ancestry_effect_report.pdf ${ordered}
    head -n 1 \$(ls *_intra_inter_distances_summary.tsv | head -n 1) > intra_inter_distances_summary.tsv
    for g in ${gene_order.collect { "'${it}'" }.join(' ')}; do
        tail -n +2 "\${g}_intra_inter_distances_summary.tsv" >> intra_inter_distances_summary.tsv
    done
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
workflow {

    def missing = []
    def genes = []      // [gene, gene_id, h5, bed, bim, fam]
    new File(params.config_tsv).withReader { r ->
        def header = r.readLine().split('\t')
        r.eachLine { line ->
            if (!line.trim()) return
            def vals = line.split('\t', -1)
            def row = [:]
            header.eachWithIndex { h, i -> row[h] = i < vals.size() ? vals[i].trim() : '' }
            def gid = row.gene_id.tokenize('.')[0]
            def er = MANIFEST[gid]
            def bf = lookupGenoBfile(GENOTYPE_MANIFEST, row.region, params.window_size, FULL_BAND, params.vcf_filter)
            if (!er) missing << "${row.experiment}: no H5 in ${params.embeddings_dir}"
            if (!bf) missing << "${row.experiment}: region_bfile (${row.region}, full variant set)"
            if (er && bf) genes << [row.experiment, row.gene_id,
                file("${params.embeddings_dir}/${er.file_path}"),
                file("${GENOTYPE_DATA_DIR}/${bf.bed_path}"), file("${GENOTYPE_DATA_DIR}/${bf.bim_path}"),
                file("${GENOTYPE_DATA_DIR}/${bf.fam_path}")]
        }
    }
    if (missing) error "Missing inputs:\n" + missing.unique().join('\n')
    def dup = genes.countBy { it[0] }.findAll { it.value > 1 }.keySet()
    if (dup) error "Duplicate experiment names in ${params.config_tsv}: ${dup}"

    groups = make_groups()
    features = donor_features(Channel.fromList(genes))
    effect = ancestry_effect(features, groups)
    merge_report(Channel.value(genes.collect { it[0] }), effect.report.collect(), effect.summary.collect())
}
