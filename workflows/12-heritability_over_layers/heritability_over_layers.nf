#!/usr/bin/env nextflow

/*
 * heritability_over_layers -- how does heritability, estimated from a frozen Borzoi
 * embedding-GRM, evolve across the FULL depth of the trunk, from the one-hot input to
 * the human output head, and does that depend on which axis you pool? Every checkpoint
 * (stem, each res_tower conv block, each transformer block, the upsampling path,
 * trunk, joined, human_head -- see src/scripts/extract_layer_variants.py's
 * list_pooled_variants()) is read out under BOTH poolings:
 *   pos   mean over the position axis (this workflow's original approach): one value
 *         per channel.
 *   chan  mean over the channel axis instead, positions left untouched: one value per
 *         genomic position.
 * Haplotype combination is unchanged throughout: additive SUM of the two haplotypes'
 * pooled vectors, matching src/scripts/combine_haplotypes.py -- do not change this
 * without revisiting every cached GRM built from it.
 *
 * A second, independent embedding stacks Borzoi's own predicted log2(coverage+1)
 * across EVERY RNA-seq track (not just a hand-picked few, and not averaged across
 * them) -- see extract_layer_variants.py's extract_rna_predictions(). This is the
 * model's actual prediction head output, not a trunk-pooling choice, so it is reported
 * as its own reference variant ("rna_predictions") rather than a point on the
 * layer-depth axis.
 *
 * A third reference variant, "rna_predictions_whole_blood" (--whole_blood_tracks,
 * extract_layer_variants.py's extract_rna_predictions_whole_blood()), restricts the
 * same prediction head to just the tissue's own RNA-seq tracks (default 7531,7532,7533,
 * the three GTEx whole-blood tracks, same default as workflow 11's own params.pred_tracks)
 * and pools them exactly as combine_haplotypes.py's pred_expression does -- haplotypes
 * SUMMED per track, then the tracks AVERAGED (plain mean, no log2, linear scale) -- so
 * this single-feature point compares directly against workflow 11's own embedding/joint
 * h2 on the identical whole-blood phenotype.
 *
 * This workflow only computes the EMBEDDING side. Genotype heritability is already
 * carried out by its own dedicated workflow (10-genotype_heritability); comparing this
 * workflow's per-layer embedding h2 against that genotype h2 -- and the "what fraction
 * of genotype h2 does each layer recover" plot -- lives in workflow 17 (plots), since
 * that is a comparison of results across two different workflows, not something this
 * workflow computes itself.
 *
 * Every layer/axis variant for a gene comes from a SINGLE forward pass, already
 * computed by workflow 06 (extract_embeddings) -- the trunk is a strictly sequential
 * U-Net, so every checkpoint was a byproduct of computing the deepest one when that
 * workflow ran. This workflow itself never runs the model; it only reads workflow 06's
 * per-gene HDF5 (embher-capture's "stacked" layout) via its manifest, exactly the
 * way workflows 09 (embedding_heritability_simulation_study) and 11 (embedding_heritability) already do.
 *
 * Each embedding variant's kernel is whitened onto the top FRACTION of its analysis
 * cohort's numerical rank before the GRM is built (src/scripts/pca_embeddings.py,
 * --pc-frac) -- see that script's docstring for why: plain column-standardisation
 * leaves embedding columns' massive mutual redundancy untouched, which saturates h2
 * near 1.0. This applies uniformly to every layer/pooling/rna_predictions variant, via
 * the same embedding_heritability process.
 *
 * The embedding REML fit includes GTEx's own published per-tissue eQTL covariates
 * (genotype PCs, PEER factors, platform, sex -- src/scripts/build_gtex_eqtl_covar.py)
 * as fixed effects, the same covariate set workflows 10/09/11 use -- assumes a single
 * tissue across the whole config, since the covariate file is shared by every gene's
 * fit.
 *
 * Config TSV columns (tab-separated, header) -- same format as workflow 11's:
 *   experiment  gene_id  gene_name  region  tissue
 *
 * Usage:
 *   nextflow run workflows/12-heritability_over_layers/heritability_over_layers.nf \
 *     -profile slurm \
 *     --config_tsv workflows/12-heritability_over_layers/configs/<config>.tsv \
 *     --embeddings_dir results/06-extract_embeddings/gtex/borzoi/<cache_key> \
 *     --out_dir <run_name>
 */

// null (default) means every layer/pooling variant workflow 06's H5 has cached for a
// gene, plus rna_predictions and rna_predictions_whole_blood -- see
// extract_layer_variants.py's list_pooled_variants().
// This workflow no longer runs the model itself, so it cannot discover "every layer"
// beyond whatever embher-capture actually captured when workflow 06 ran. Set to a
// comma-separated "<layer>_<pos|chan>,...,rna_predictions" string (labels exactly as
// extract_layer_variants.py names its output files) to require/restrict to a specific
// subset (fails fast if any requested one isn't cached).
params.variants      = null
// Tracks pooled into the rna_predictions_whole_blood variant (haplotypes SUMMED per
// track, then these tracks AVERAGED -- plain mean, no log2, linear coverage scale --
// identical to combine_haplotypes.py's own pred_expression). Default: the three GTEx
// whole-blood RNA-seq tracks among Borzoi's training targets, same default as workflow
// 11's own params.pred_tracks -- override if this run's config isn't Whole Blood.
params.whole_blood_tracks = '7531,7532,7533'

// ── Constants ─────────────────────────────────────────────────────────────── //

def SCRIPTS_DIR  = "${projectDir}/scripts"
def SHARED_DIR   = "${projectDir}/../../src/scripts"
def ANALYSIS_ENV = params.env_analysis
def CONDA_SH     = params.conda_sh
// GTEx's own published per-tissue eQTL covariates (60 PEER factors + genotype PCs +
// sex/platform/pcr for Whole Blood) -- the same covariate set every other real-data
// heritability workflow (08, 09, 11) uses as REML fixed effects
// (exact_reml_heritability.py --covariate-file).
def GTEX_COVAR_DIR = "${projectDir}/../../data/raw/GTEx/v10/GTEx_Analysis_v10_eQTL_covariates"

// ── Parameters ────────────────────────────────────────────────────────────── //

params.out_dir     = null
params.config_tsv  = null
// Workflow 06's (extract_embeddings) borzoi/<cache_key> output dir for this run, e.g.
// results/06-extract_embeddings/gtex/borzoi/borzoi_replicate_0_release_46_..._bfloat16 --
// see the manifest lookup in the workflow{} block below. Same param name/meaning as
// workflow 11's.
params.embeddings_dir = null
// model_type of the manifest rows to read: 'borzoi' (pretrained, the default) or
// 'borzoi_random' (workflow 06's --run_borzoi_random output, results/06-extract_embeddings/
// {cohort}/borzoi_random/<cache_key>). A random-init run stores only the __seq_mean pooling, so it
// yields the *_pos variants (and the rna_predictions ones), no *_chan.
params.embeddings_model_type = 'borzoi'
// The phenotype is always the quantile_int transform (no --transform option).
def TRANSFORM = 'quantile_int'
// Optional: drop K's eigen-components below this fraction of the top eigenvalue
// before REML (see build_grm.py's docstring). Default OFF (null) -- toy calibration
// tests showed plain rank/collinearity does not bias AI-REML, and forcing K to be
// exactly singular instead introduced its own numerical instability (BLAS-threading-
// dependent non-convergence). Set explicitly (e.g. --min_eigval_ratio 1e-6) only if
// deliberately re-testing this.
params.min_eigval_ratio = null
// Whiten each embedding onto its top-k principal components before the GRM is built
// (src/scripts/pca_embeddings.py). ON by default, at --whiten_pc_frac 0.7 -- see that
// script's docstring for why this whitening step exists at all: column standardisation
// is a MARGINAL normalisation, and embedding columns are massively redundant, so an
// unwhitened kernel's eigenvalue spectrum spans 5+ orders of magnitude and saturates
// h2 near 1.0.
//
// --whiten_pc_frac is a FRACTION of the analysis cohort's numerical rank (k = round(
// whiten_pc_frac * rank), recomputed per variant from that variant's own analysis
// cohort/rank -- see pca_embeddings.py's --pc-frac), preferred here over a fixed
// --n_pcs because a fixed absolute k means something different for every layer (layer
// widths here range from 512 to 262144 features -- see the H5 dump in this workflow's
// own history) and would need re-tuning per layer to stay "most of the variance" the
// way a single fraction does not. --n_pcs still exists as an escape hatch (e.g. to
// reproduce an old fixed-k run); when --whiten_pc_frac is null, it's used instead. Set
// --whiten_pc_frac null (and leave --n_pcs unset) to disable whitening entirely and
// fall back to the historical behaviour, K = Z Z'/M straight from the
// column-standardised embedding.
params.whiten_pc_frac = 0.7
params.n_pcs       = null
// exact_reml_heritability.py --nonneg (default true): embedding variance component
// constrained >= 0, same constrained-REML default workflows 09/10/09/11/100 use.
params.nonneg      = true
def NONNEG_FLAG = params.nonneg ? '--nonneg' : ''
// --estimator he (default): Haseman-Elston regression (src/scripts/he_heritability.py;
// --nonneg ignored). Its delete-one jackknife SEs are OFF by default here (--jackknife none,
// h2_se = NaN; nothing downstream reads them) -- pass --he_jackknife true to turn them on.
// --estimator reml: exact_reml_heritability.py, per --nonneg above.
params.estimator    = 'he'
params.he_jackknife = false
if (!(params.estimator in ['reml', 'he'])) error "--estimator must be reml or he, got '${params.estimator}'"
def HE = params.estimator == 'he'
def ESTIMATOR_SCRIPT = HE ? 'he_heritability.py' : 'exact_reml_heritability.py'
def FIT_FLAGS = HE ? "--jackknife ${params.he_jackknife ? 'loo' : 'none'}" : NONNEG_FLAG
// embedding_heritability_batch: (gene, variant) fits per SLURM task -- batched because
// a real run is ~500 variants x 539 genes = tens of thousands of ~1 s fits, and
// array-per-fit scheduling made SLURM's own per-task overhead dominate (same reasoning
// workflow 09's exact_reml_single_grm_batch docstring gives). Batches are built from a
// channel explicitly sorted by experiment (gene__variant) before collate() -- see the
// workflow{} block -- so batch composition/order never depends on upstream SLURM-array
// completion order, which would otherwise change between runs and invalidate every
// batch's Nextflow cache under -resume even when nothing about the actual inputs
// changed.
params.batch_size = 500

if (!params.out_dir)    error "Please specify --out_dir"
if (!params.config_tsv) error "Please specify --config_tsv"
if (!params.embeddings_dir) error "Please specify --embeddings_dir (workflow 06's " +
    "borzoi/<cache_key> output directory)"

def OUT_PREFIX   = "${launchDir}/results/12-heritability_over_layers/${params.out_dir}"

// 04's manifest reader (src/nf/embeddings_manifest.nf): the 04 directory name no longer carries the
// pooling, so the rows are filtered on the manifest's own pooling columns.
include { loadEmbeddingsManifest } from '../../src/nf/embeddings_manifest.nf'
def EMBEDDINGS_MANIFEST = loadEmbeddingsManifest(params.embeddings_dir, [model_type: params.embeddings_model_type])

// Workflow 02's (expression_data_preparation) cache -- the tissue TPM matrix and every
// gene's phenotype are looked up there instead of computed here.
def EXPRESSION_DATA_DIR     = "${launchDir}/results/02-expression_data_preparation/gtex"
def EXPRESSION_MANIFEST_DIR = "${EXPRESSION_DATA_DIR}/manifest"


// ──────────────────────────────────────────────────────────────────────────── //
// Process: resolve every gene's cached phenotype (workflow 02's
//          expression_data_preparation), under the quantile_int transform, once, up
//          front.
// ──────────────────────────────────────────────────────────────────────────── //

process resolve_expression {
    cpus 1
    memory { 4.GB * task.attempt }
    time { 20.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    input:
    path(config_tsv)

    output:
    path("resolved_pheno.tsv"), emit: table
    // resolve_expression.py also writes a {experiment}_pheno.parquet/_donors.txt pair
    // per config row (see its docstring); they are declared here so Nextflow stages them
    // back from the node-local scratch dir (`scratch = '/tmp'`) before it is deleted.
    path("*_pheno.parquet"), emit: phenos
    path("*_donors.txt"), emit: donors

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${SHARED_DIR}/resolve_expression.py \\
        --config ${config_tsv} \\
        --manifest-dir ${EXPRESSION_MANIFEST_DIR} \\
        --data-dir ${EXPRESSION_DATA_DIR} \\
        --product pheno \\
        --transform ${TRANSFORM} \\
        --out resolved_pheno.tsv
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process: GTEx's own published per-tissue eQTL covariates, transposed to the
//          IID-indexed TSV exact_reml_heritability.py's --covariate-file expects -- same
//          script and source workflows 10/09/11 use. Assumes a single tissue across the
//          whole config (same assumption those workflows make), since the covariate
//          file is shared across every gene's fit.
// ──────────────────────────────────────────────────────────────────────────── //

process build_peer_covar {
    cpus 1
    memory { 4.GB * task.attempt }
    time { 20.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    input:
    val(tissue)

    output:
    path("covar_peer.tsv")

    script:
    def tissueFile = tissue.replaceAll(' ', '_')
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    python ${SHARED_DIR}/build_gtex_eqtl_covar.py \\
        --covariates-file ${GTEX_COVAR_DIR}/${tissueFile}.v10.covariates.txt \\
        --out covar_peer.tsv
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process: split one gene's workflow-06 H5 into one donor x feature embedding
//          parquet per (layer, pooling) variant, plus rna_predictions -- see
//          src/scripts/extract_layer_variants.py. Cheap (plain HDF5 reads, no model
//          forward pass), unlike this workflow's retired extract_embeddings_variant.
// ──────────────────────────────────────────────────────────────────────────── //

process extract_layer_variants {
    array 1000        // pack tasks of this process into one SLURM array
    cpus 1
    memory { 8.GB * task.attempt }
    time { 3.h * task.attempt }
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
        --whole-blood-tracks ${params.whole_blood_tracks} \\
        ${variantsFlag}
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process: embedding GRM + REML for a BATCH of (gene, layer, pooling) variants --
//          ~params.batch_size variants per SLURM task, not one task per variant (see
//          params.batch_size's own comment on why). Estimated by this project's own
//          exact multi-component AI-REML (src/scripts/exact_reml_heritability.py,
//          --grm-npz single-kernel mode), as in workflow 11. The PCA-whitening step
//          below (pca_embeddings.py, --whiten_pc_frac/--n_pcs) is
//          independent of which algorithm then estimates h2 from the resulting kernel,
//          and (like --min_eigval_ratio/--nonneg) is the same for every item in a batch,
//          so it's computed once outside the per-item loop below.
//
//          batch items are plain absolute-path strings (val, not path) -- pheno/donors/
//          embedding filenames are already unique per item (gene__variant), so there is
//          no staging collision to avoid; this only works because compute nodes here
//          have direct access to the same shared filesystem the upstream tasks' work
//          dirs point at, so there is nothing to actually stage (same reasoning
//          workflow 09's exact_reml_single_grm_batch gives for its own bedPrefix/
//          phenoPath strings). A failing item is skipped (|| echo ... >&2), not fatal
//          to the whole batch.
// ──────────────────────────────────────────────────────────────────────────── //

process embedding_heritability_batch {
    array 1000        // pack tasks of this process into one SLURM array
    cpus 1
    memory { 16.GB * task.attempt }
    time { 6.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "batch_${task.index}"

    publishDir { "${OUT_PREFIX}/heritability" }, mode: 'copy',
        pattern: "*_heritability.parquet"

    input:
    // batch: List of [experiment, gene, phenoPath, donorsPath, embPath] 5-tuples, built
    // and sorted deterministically in the workflow{} block before collate() -- see
    // params.batch_size's own comment.
    val(batch)
    path(covar)

    output:
    path("*_heritability.parquet"), emit: h2

    script:
    def eigFlag = params.min_eigval_ratio ? "--min-eigval-ratio ${params.min_eigval_ratio}" : ""
    // --whiten_pc_frac (a fraction of the analysis cohort's rank) takes precedence over
    // --n_pcs (a fixed absolute k) when set -- see params.whiten_pc_frac above for why
    // a fraction is this workflow's default. Neither set -> whitening is skipped
    // entirely (grmInput falls back to the raw embedding).
    def pcArg = params.whiten_pc_frac ? "--pc-frac ${params.whiten_pc_frac}" :
                (params.n_pcs ? "--n-pcs ${params.n_pcs}" : null)
    def cmds = batch.collect { item ->
        def (experiment, gene, phenoPath, donorsPath, embPath) = item
        def pcaStep = pcArg ? """
    python ${SHARED_DIR}/pca_embeddings.py \\
        --embeddings ${embPath} \\
        --pheno ${phenoPath} \\
        ${pcArg} \\
        --out ${experiment}_pcs.parquet
""" : ""
        def grmInput = pcArg ? "${experiment}_pcs.parquet" : "${embPath}"
        """
${pcaStep}
    python ${SHARED_DIR}/build_grm.py \\
        --embeddings ${grmInput} \\
        --pheno ${phenoPath} ${eigFlag} \\
        --out ${experiment}

    python ${SHARED_DIR}/${ESTIMATOR_SCRIPT} \\
        --grm-npz ${experiment}.npz \\
        --pheno ${phenoPath} \\
        --covariate-file ${covar} ${FIT_FLAGS} \\
        --experiment '${experiment}' \\
        --out ${experiment}_heritability.parquet \\
        || echo '[warn] embedding_heritability failed for ${experiment}' >&2
"""
    }.join('\n')
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
${cmds}
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
workflow {

    // GTEx's own PEER covariates (build_peer_covar) assume one shared tissue across
    // the whole config -- same assumption workflows 10/09/11 make, since the
    // covariate file is shared by every gene's fit.
    def tissues = [] as Set
    new File(params.config_tsv).withReader { r ->
        def header = r.readLine().split('\t')
        r.eachLine { line ->
            if (!line.trim()) return
            def vals = line.split('\t', -1)
            def row = [:]
            header.eachWithIndex { h, i -> row[h] = i < vals.size() ? vals[i].trim() : '' }
            tissues << row.tissue
        }
    }
    if (tissues.size() != 1)
        error "This workflow assumes a single tissue for the shared PEER covariate file; got: ${tissues}"
    // build_peer_covar's input is a Channel.value(), so its output is automatically a
    // value channel too (broadcasts to every embedding_heritability invocation instead
    // of being exhausted after the first).
    covar = build_peer_covar(Channel.value(tissues.first()))

    // per-gene phenotype: looked up from workflow 02's
    // (expression_data_preparation) manifest instead of computed here.
    resolved_pheno = resolve_expression(Channel.value(file(params.config_tsv)))
    // Joined by tag (not by the absolute path string resolved_pheno.tsv records --
    // see that process's output comment) against the actually-staged phenos/donors
    // emitted alongside the table, since those are what Nextflow tracks as real,
    // persisted files.
    pheno_files = resolved_pheno.phenos.flatten().map { f -> tuple(f.name - '_pheno.parquet', f) }
    donor_files = resolved_pheno.donors.flatten().map { f -> tuple(f.name - '_donors.txt', f) }
    pheno = resolved_pheno.table
        .splitCsv(header: true, sep: '\t')
        .map { r -> tuple(new File(r.pheno_path).name - '_pheno.parquet', r.experiment.trim(), r.region.trim()) }
        .combine(pheno_files, by: 0)
        .combine(donor_files, by: 0)
        .map { tag, e, reg, pf, df -> tuple(e, reg, pf, df) }

    // ── embeddings: resolved against workflow 06's manifest, one gene at a time --
    // fails the whole run immediately for any config gene 04b hasn't captured, same
    // "never silently fall back" pattern workflows 09/11 use for the identical lookup.
    def missing = []
    def resolvedGenes = []  // [gene, geneIdH5, h5]

    new File(params.config_tsv).withReader { r ->
        def header = r.readLine().split('\t')
        r.eachLine { line ->
            if (!line.trim()) return
            def vals = line.split('\t', -1)
            def row = [:]
            header.eachWithIndex { h, i -> row[h] = i < vals.size() ? vals[i].trim() : '' }
            def gene = row.experiment
            def geneIdH5 = row.gene_id.split('\\.')[0]

            def manRow = EMBEDDINGS_MANIFEST[geneIdH5]
            if (!manRow) {
                missing << "${gene}: ${row.gene_id} (${geneIdH5}) not in workflow 06's manifest"
            } else {
                resolvedGenes << [gene, geneIdH5, file("${params.embeddings_dir}/${manRow.file_path}")]
            }
        }
    }
    if (missing) {
        error "Missing from workflow 06's manifest (${params.embeddings_dir}) -- run " +
            "workflow 06 (extract_embeddings) for these genes first:\n" +
            missing.unique().join('\n')
    }

    // "__" (double underscore) separates the gene prefix from the variant name in
    // every experiment label below -- see workflow 17's compare_variants.py
    // parse_experiment() and extract_layer_variants.py's own file-naming, which is the
    // same delimiter.
    extracted = extract_layer_variants(
        Channel.fromList(resolvedGenes),
        Channel.value(file("${SHARED_DIR}/extract_layer_variants.py")))

    // Each task emits a variable-length list of "<gene>__<variant>_embedding.parquet"
    // files (one per layer/pooling/rna_predictions variant found for that gene) --
    // flatten to one (gene, experiment, file) tuple per variant, parsing the variant
    // label back out of the filename the same way resolve_expression's own
    // pheno/donor joins strip a known suffix (`-` removes the first match).
    emb_variants = extracted.flatMap { gene, files ->
        (files instanceof List ? files : [files]).collect { f ->
            def variant = f.name - "${gene}__" - "_embedding.parquet"
            tuple(gene, "${gene}__${variant}", f)
        }
    }

    // ── final step: embedding-GRM h2 per (gene, layer, pooling) variant, published
    // for workflow 17 (plots) to compare against genotype heritability (already
    // computed by its own dedicated workflow, 10-genotype_heritability) ────────── //
    //
    // Sorted by experiment (gene__variant, a unique string) before collate() -- without
    // this, batch composition/order would follow extract_layer_variants' own SLURM-array
    // completion order, which differs run to run and would invalidate every batch's
    // Nextflow cache under -resume even when nothing about the actual inputs changed
    // (see embedding_heritability_batch's own docstring and params.batch_size's comment).
    // toSortedList()/flatMap{it} materializes the whole (gene, variant) list before
    // batching -- unavoidable for a deterministic sort, and collate() already requires
    // buffering a full batch at a time regardless.
    batches = emb_variants
        .combine(pheno.map { e, reg, p, d -> tuple(e, p, d) }, by: 0)
        .map { e, exp, emb, p, d -> [exp, e, p.toString(), d.toString(), emb.toString()] }
        .toSortedList { a, b -> a[0] <=> b[0] }
        .flatMap { it }
        .collate(params.batch_size)

    embedding_heritability_batch(batches, covar)
}
