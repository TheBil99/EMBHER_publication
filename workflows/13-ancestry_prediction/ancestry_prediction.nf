#!/usr/bin/env nextflow

/*
 * ancestry_prediction -- how well does EUR vs AFR ancestry decode from genotype,
 * raw Borzoi embeddings and whitened Borzoi embeddings, across the FULL depth of the
 * trunk?
 *
 * Per gene, three representations, each reduced to its own top --n_pcs principal
 * components (default 10) and scored by plain in-sample OLS R^2 on EUR vs AFR (EUR=0,
 * AFR=1) -- no held-out split (scripts/predict_ancestry.py): this is a quick read of
 * how separable the two groups are in a representation, not a generalization
 * estimate, so a train/test split buys nothing here. The top-10-PC cap matters more
 * than usual: feature count is almost always >= donor count (genotype windows, most
 * embedding layers), so an unreduced OLS fit would trivially interpolate the labels
 * (R^2 = 1 regardless of real signal) -- capping every representation at the same
 * small fixed component count keeps R^2 comparable and meaningful across them.
 *   genotype    0/1/2 non-reference allele counts from workflow 04's full region
 *               bfile (src/scripts/genotype_features.py), top-10-PC OLS R^2 per gene
 *               -- a reference line, not a point on the depth axis.
 *   raw         every layer:pooling variant workflow 06's H5 has cached for the gene
 *               (src/scripts/extract_layer_variants.py -- stem, each res_tower conv
 *               block, each transformer block, the upsampling path, trunk, joined,
 *               human_head, under BOTH position- and channel-axis pooling, plus
 *               rna_predictions), haplotype-summed, columns centred -> top 10 PCs
 *               (unscaled).
 *   whitened    the same layer:pooling variant, columns standardised -> SVD -> top
 *               round(--whiten_pc_frac * rank) components rescaled to unit variance,
 *               EXACTLY src/scripts/pca_embeddings.py's construction (default 0.9) --
 *               then only the first 10 columns of that whitened basis are kept for
 *               the fit (whitening first, truncating to 10 after).
 *
 * Donors are restricted to GTEx self-reported EUR/AFR individuals
 * (src/scripts/make_ancestry_groups.py --comparison eur_afr, as workflow 14
 * ancestry_effect_embeddings already uses) -- every other ancestry is left out, since
 * there are too few donors of any other group for a meaningful fit.
 *
 * Every layer/axis variant for a gene comes from a SINGLE forward pass, already
 * computed by workflow 06 (extract_embeddings); this workflow never runs the model,
 * exactly like workflow 12 (heritability_over_layers), whose extract_layer_variants
 * process this one reuses unchanged.
 *
 * A gene's ~45 layer:pooling variants fan out to that many OLS fits, and at hundreds
 * of genes that's tens of thousands of (gene, variant) fits -- each is a single cheap
 * closed-form fit (no CV, no split), so spawning one SLURM task per fit is pure
 * scheduling overhead. --batch_size (default 200) pools that many (gene, variant)
 * embedding tables into ONE task instead (scripts/predict_ancestry.py accepts any
 * number of --features files and loops over them internally). A smaller batch means a crash or retry redoes less work, and each file
 * is read/SVD'd one at a time regardless of batch size, so batch size trades task
 * count against blast radius, not peak memory. Genotype fits stay one task per gene
 * (only hundreds of those, not tens of thousands, so batching buys little there).
 *
 * The batch list is sorted by filename before --batch_size chunking
 * (`.toSortedList()` below) so batch membership is deterministic across runs: without
 * it, the (gene, variant) files land in whatever order the many parallel
 * extract_layer_variants SLURM tasks happened to finish/glob in, which can differ
 * between a run and its own -resume even when every underlying file is an identical
 * cache hit -- reshuffling almost every batch's exact file set and defeating caching
 * for predict_ancestry_embedding.
 *
 * Config TSV columns (tab-separated, header) -- same format as workflow 11's:
 *   experiment  gene_id  gene_name  region  tissue
 *
 * Usage (from the project root):
 *   nextflow run workflows/13-ancestry_prediction/ancestry_prediction.nf \
 *     -profile slurm \
 *     --config_tsv workflows/11-embedding_heritability/configs/whole_blood_04b_random539.tsv \
 *     --embeddings_dir results/06-extract_embeddings/gtex/borzoi/<unfiltered cache_key> \
 *     --out_dir random539
 */

include { loadGenotypeManifest; lookupGenoBfile } from '../../src/nf/genotype_manifest.nf'

// ── Constants ─────────────────────────────────────────────────────────────── //

def SCRIPTS_DIR  = "${projectDir}/scripts"
def SHARED_DIR   = "${projectDir}/../../src/scripts"
def ANALYSIS_ENV = params.env_analysis
def CONDA_SH     = params.conda_sh
def RACE_FILE    = "${projectDir}/../../data/raw/dbgap/gtex/phenotypes/phs000424.v11.pht002742.v9.p2.c1.GTEx_Subject_Phenotypes.GRU.txt.gz"

// ── Parameters ────────────────────────────────────────────────────────────── //

params.out_dir               = null
params.config_tsv            = null
params.embeddings_dir        = null   // unfiltered 04b run, as workflow 12/14 require
params.genotype_manifest_dir = null
params.comparison            = 'eur_afr'
params.window_size           = 524288
params.vcf_filter            = null
// null (default) = every layer/pooling variant cached for a gene, plus
// rna_predictions -- same param as workflow 12's --variants. Comma-separated
// "<layer>_<pos|chan>,...,rna_predictions" to restrict.
params.variants              = null
// Every representation (genotype, raw, whitened) is capped to its top n_pcs
// components before the OLS fit -- see module docstring for why a small fixed k,
// not a fraction of rank, is what keeps R^2 meaningful here (no held-out split to
// penalize an inflated component count).
params.n_pcs                 = 10
// 'whitened' only: whiten onto this FRACTION of rank first (pca_embeddings.py's own
// --pc-frac construction), then keep only the first n_pcs columns of that whitened
// basis -- see module docstring.
params.whiten_pc_frac        = 0.9
// Number of (gene, variant) embedding tables pooled into one predict_ancestry task --
// see module docstring.
params.batch_size            = 200

if (!params.out_dir)        error "Please specify --out_dir"
if (!params.config_tsv)     error "Please specify --config_tsv"
if (!params.embeddings_dir) error "Please specify --embeddings_dir (unfiltered 04b run)"
if (!(params.comparison in ['eur_afr']))
    error "--comparison must be eur_afr (got '${params.comparison}') -- too few donors " +
        "of any other GTEx-reported ancestry for a meaningful fit"

def OUT_PREFIX        = "${launchDir}/results/13-ancestry_prediction/${params.out_dir}"
def GENOTYPE_DATA_DIR = "${launchDir}/results/04-genotype_extraction/gtex"
def GENOTYPE_MANIFEST = loadGenotypeManifest(params.genotype_manifest_dir ?: "${GENOTYPE_DATA_DIR}/manifest")
def FULL_BAND = [min: 0.0, max: null]

// 04's manifest reader (src/nf/embeddings_manifest.nf): the 04 directory name no longer carries the
// pooling; no pooling filter here: each gene contributes whatever pooling variants its H5 holds.
include { loadEmbeddingsManifest } from '../../src/nf/embeddings_manifest.nf'
def EMBEDDINGS_MANIFEST = loadEmbeddingsManifest(params.embeddings_dir)


// ──────────────────────────────────────────────────────────────────────────── //
// Process: ancestry groups for --comparison (src/scripts/make_ancestry_groups.py) --
//          same script workflow 14 uses.
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
// Process: donor x variant genotype counts for one gene (src/scripts/genotype_features.py).
// ──────────────────────────────────────────────────────────────────────────── //

process genotype_features {
    array 1000
    cpus 1
    memory { 8.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${gene}"

    input:
    tuple val(gene), path(bed), path(bim), path(fam)

    output:
    tuple val(gene), path("${gene}_genotype.parquet")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    python ${SHARED_DIR}/genotype_features.py \\
        --bfile ${bed.baseName} --out ${gene}_genotype.parquet
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process: split one gene's workflow-06 H5 into one donor x feature embedding
//          parquet per (layer, pooling) variant, plus rna_predictions -- identical
//          to workflow 12's own process (src/scripts/extract_layer_variants.py).
// ──────────────────────────────────────────────────────────────────────────── //

process extract_layer_variants {
    array 1000
    cpus 1
    memory { 8.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${gene}"

    input:
    tuple val(gene), val(geneIdH5), path(h5)
    path(driver)

    output:
    tuple val(gene), path("${gene}__*_embedding.parquet")

    script:
    def variantsFlag = params.variants ? "--variants '${params.variants}'" : ""
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${driver} \\
        --h5 ${h5} \\
        --gene-id ${geneIdH5} \\
        --out-prefix ${gene} \\
        ${variantsFlag}
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process: EUR-vs-AFR top-10-PC OLS R^2 for one gene's genotype table
//          (scripts/predict_ancestry.py --reps genotype). No CV/split -- a single
//          closed-form fit, so this is far cheaper than the process name's workflow-12
//          analogue.
// ──────────────────────────────────────────────────────────────────────────── //

process predict_ancestry_genotype {
    array 1000
    cpus 1
    memory { 16.GB * task.attempt }
    time { 20.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${gene}"

    publishDir { "${OUT_PREFIX}/results" }, mode: 'copy'

    input:
    tuple val(gene), path(genotype)
    path(groups)

    output:
    path("${gene}__genotype_ancestry_prediction.tsv")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python -u ${SCRIPTS_DIR}/predict_ancestry.py \\
        --features ${genotype} --groups ${groups} --n-pcs ${params.n_pcs} \\
        --out ${gene}__genotype_ancestry_prediction.tsv
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process: EUR-vs-AFR top-10-PC OLS R^2 for a BATCH of up to --batch_size (gene,
//          embedding variant) tables, raw AND whitened
//          (scripts/predict_ancestry.py --features <batch's files>, looped
//          internally -- see module docstring for why batching, not one task per
//          variant). Each task's work directory holds exactly its own batch's staged
//          files, so the `*_embedding.parquet` glob below can't pick up another
//          task's files.
// ──────────────────────────────────────────────────────────────────────────── //

process predict_ancestry_embedding {
    array 1000
    cpus 1
    memory { 16.GB * task.attempt }
    time { 2.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "batch_${task.index}"

    publishDir { "${OUT_PREFIX}/results" }, mode: 'copy'

    input:
    path(embs)
    path(groups)

    output:
    path("batch_${task.index}_ancestry_prediction.tsv")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python -u ${SCRIPTS_DIR}/predict_ancestry.py \\
        --features *_embedding.parquet --groups ${groups} \\
        --n-pcs ${params.n_pcs} --pc-frac ${params.whiten_pc_frac} \\
        --out batch_${task.index}_ancestry_prediction.tsv
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process: collect every (gene, variant)'s R^2 table into one summary TSV and one
//          figure (scripts/summarize_ancestry_prediction.py).
// ──────────────────────────────────────────────────────────────────────────── //

process summarize {
    cpus 1
    memory { 2.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}/results" }, mode: 'copy'

    input:
    path(tables)

    output:
    path("ancestry_prediction_summary.tsv")
    path("ancestry_prediction_layers.pdf")
    path("ancestry_prediction_layers.png")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    python ${SCRIPTS_DIR}/summarize_ancestry_prediction.py --dir .
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
workflow {

    def missing = []
    def genoGenes = []  // [gene, bed, bim, fam]
    def embGenes  = []  // [gene, geneIdH5, h5]
    new File(params.config_tsv).withReader { r ->
        def header = r.readLine().split('\t')
        r.eachLine { line ->
            if (!line.trim()) return
            def vals = line.split('\t', -1)
            def row = [:]
            header.eachWithIndex { h, i -> row[h] = i < vals.size() ? vals[i].trim() : '' }
            def gene = row.experiment
            def geneIdH5 = row.gene_id.tokenize('.')[0]

            def bf = lookupGenoBfile(GENOTYPE_MANIFEST, row.region, params.window_size, FULL_BAND, params.vcf_filter)
            def er = EMBEDDINGS_MANIFEST[geneIdH5]
            if (!bf) missing << "${gene}: region_bfile (${row.region}, full variant set)"
            if (!er) missing << "${gene}: ${row.gene_id} (${geneIdH5}) not in workflow 06's manifest"
            if (bf) genoGenes << [gene, file("${GENOTYPE_DATA_DIR}/${bf.bed_path}"),
                file("${GENOTYPE_DATA_DIR}/${bf.bim_path}"), file("${GENOTYPE_DATA_DIR}/${bf.fam_path}")]
            if (er) embGenes << [gene, geneIdH5, file("${params.embeddings_dir}/${er.file_path}")]
        }
    }
    if (missing) error "Missing inputs:\n" + missing.unique().join('\n')

    groups = make_groups()

    geno_r2 = predict_ancestry_genotype(
        genotype_features(Channel.fromList(genoGenes)), groups)

    // Flatten every gene's bundle of "<gene>__<variant>_embedding.parquet" files into
    // one stream of plain files (predict_ancestry.py parses gene/variant back out of
    // each filename itself, so no (gene, variant) tuple needs carrying through here),
    // sort by filename for a run-to-run DETERMINISTIC order (the glob output of many
    // parallel extract_layer_variants tasks is not otherwise guaranteed stable -- see
    // module docstring), then pool them into batches of --batch_size for
    // predict_ancestry_embedding -- see that process's comment for why batching
    // instead of one task per file.
    extracted = extract_layer_variants(
        Channel.fromList(embGenes),
        Channel.value(file("${SHARED_DIR}/extract_layer_variants.py")))
    // toSortedList collects ALL extract_layer_variants outputs before sorting, so
    // batching no longer starts until every one of them has finished -- a short delay
    // (that process is fast, and fully cached on any resume) traded for deterministic,
    // resumable batches.
    emb_batches = extracted
        .flatMap { gene, files -> (files instanceof List ? files : [files]) }
        .toSortedList { a, b -> a.name <=> b.name }
        .flatMap { it }
        .collate(params.batch_size)
    emb_r2 = predict_ancestry_embedding(emb_batches, groups)

    summarize(geno_r2.mix(emb_r2).collect())
}
