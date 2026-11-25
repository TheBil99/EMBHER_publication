#!/usr/bin/env nextflow

/*
 * genotype_heritability -- genotype-only cis-h2 for real GTEx gene expression, by
 * Haseman-Elston regression (default, src/scripts/he_heritability.py) or, with
 * --estimator reml, exact multi-component AI-REML (src/scripts/exact_reml_heritability.py:
 * exact traces, three starts, variance components constrained >= 0 by default -- see
 * --nonneg below), on a cohort MAC/MAF partition of the region's variants, jointly fit.
 * --binning picks it:
 *
 *   mac_ld_6 (default)  MAC<=mac_max | MAC>mac_max & MAF<=0.01 | (0.01,0.1] low LD |
 *                       (0.01,0.1] high LD | (0.1,0.5] low LD | (0.1,0.5] high LD
 *   mac4                MAC<=mac_max | MAC>mac_max & MAF<=0.01 | (0.01,0.1] | (0.1,0.5]
 *
 * mac4 is src/scripts/build_cohort_mac_maf4_bins.py; mac_ld_6 splits its two common bins
 * at their median region-wide LD score (src/scripts/split_bins_by_ld.py --bins 3,4).
 * MAC/MAF and LD scores are
 * computed fresh per gene on the donors with a non-missing phenotype for THIS gene (a
 * tissue's RNA-seq donor set is a subset of the WGS panel).
 * Method is he_mac_ld_6_covar / he_mac_maf_bins_4_covar by default (exact_reml_... with
 * --estimator reml); per-gene outputs go to h2/he_mac_ld_6/ / h2/he_mac4bin/
 * (h2/exact_.../). Workflow 17 selects workflow 10 results by method
 * (params.genotype_method, default exact_reml_mac_maf_bins_4_covar).
 * --jackknife none (Haseman-Elston) drops the SEs; see --jackknife below.
 *
 * --nonneg (default true; --estimator reml only): every fit runs exact_reml_heritability.py --nonneg -- each
 * bin's variance component constrained >= 0, so V cannot become singular and the
 * at_boundary/valid flags of the unconstrained fit no longer apply. Each fit also gets
 * lrt/lrt_p: the LRT of the total genotype h2 = 0 (every bin = 0) against the
 * covariates-only model, chi-bar-squared null. --nonneg false restores the
 * unconstrained fit. Method labels are the same either way (workflow 17 selects by
 * them), so the two are told apart by out_dir and the lrt columns.
 *
 * Every fit passes --restrict-grm-to-cohort: each GRM's per-SNP allele-frequency
 * standardization is computed on just this gene's phenotyped donors, not the full
 * tissue-agnostic 944-donor panel. Otherwise a variant that is rare elsewhere in GTEx
 * but invariant within the analyzed cohort would still be standardized by a nonzero
 * whole-panel frequency, adding a small constant to every kinship entry instead of the
 * zero contribution an invariant SNP should have. This concerns the GRM's own values
 * and is independent of the cohort-restricted bin *classification*.
 *
 * Phenotype and covariates:
 *
 *   1. Covariates enter the estimator as REML fixed effects (--covariate-file), not
 *      regressed out of the phenotype beforehand (residualizing first biases
 *      h2 low). Workflow 02 applies no covariate
 *      correction at all (see prepare_expression_matrix.py), so the phenotype resolved
 *      here is its published, quantile+int-transformed expression.
 *   2. The covariate set is GTEx's own published per-tissue eQTL covariates (60 PEER
 *      factors + 5 genotype PCs + platform + pcr + sex,
 *      data/raw/GTEx_Analysis_v10_eQTL_covariates/, transposed by
 *      src/scripts/build_gtex_eqtl_covar.py). Assumes a single tissue across the whole
 *      config, since the covariate file is shared across every gene.
 *
 * No embedding, no permutation null, no true h2, on the window workflow 04
 * (genotype_extraction) already produced bfiles for (this project's usual
 * window_size/min_maf convention).
 *
 * Config TSV columns (tab-separated, with header) -- same schema as workflow 11/16:
 *   experiment  gene_id  gene_name  region  tissue
 *
 * Usage:
 *   nextflow run workflows/10-genotype_heritability/genotype_heritability.nf \
 *     -profile slurm \
 *     --config_tsv workflows/10-genotype_heritability/configs/whole_blood_genes.tsv \
 *     --out_dir whole_blood_genes
 *
 * --cohort gtex|geuvadis (default gtex) selects workflow 02's/03's cohort-specific
 * results dirs (results/02-.../{gtex,geuvadis}, results/03-.../{gtex,geuvadis}) and,
 * for the covariate step,
 * build_gtex_eqtl_covar.py's single published per-tissue file vs.
 * build_geuvadis_covar.py's join of workflow 05's three GEUVADIS covariate sources.
 * Output goes to results/10-genotype_heritability/${cohort}/${out_dir}.
 */

include { loadGenotypeManifest; lookupGenoBfile } from '../../src/nf/genotype_manifest.nf'

// ── Constants ─────────────────────────────────────────────────────────────── //

def SCRIPTS_DIR    = "${projectDir}/scripts"
def SHARED_DIR      = "${projectDir}/../../src/scripts"
def CONDA_SH        = params.conda_sh
def ANALYSIS_ENV    = params.env_analysis
def PLINK_BIN       = params.plink
def GTEX_COVAR_DIR  = "${projectDir}/../../data/raw/GTEx/v10/GTEx_Analysis_v10_eQTL_covariates"

// ── Parameters ────────────────────────────────────────────────────────────── //

params.out_dir     = null
params.config_tsv  = "${projectDir}/configs/whole_blood_genes.tsv"
params.window_size = 524288
params.min_maf     = 0.0
params.max_maf     = null
params.vcf_filter  = null
// gtex | geuvadis -- selects workflow 02's/03's cohort-specific results dirs and, for
// the covariate step, build_gtex_eqtl_covar.py's single published per-tissue file vs.
// build_geuvadis_covar.py's join of workflow 05's three GEUVADIS covariate sources.
params.cohort      = 'gtex'
// Phenotype construction (workflow 02, via resolve_expression.py). No covariate
// correction is applied there at all -- see module docstring on why covariates are
// added as fixed effects instead. Transform is fixed to quantile_int
// (prepare_expression_matrix.py's own --transform, see its docstring) -- the
// log2_plus_1 arm isn't used by this workflow, so it isn't exposed as a param here
// (same as workflow 170).
def TRANSFORM = 'quantile_int'
// Genes per SLURM task -- batched (not one task per gene) so the
// per-task SLURM scheduling overhead isn't paid once per gene when each gene's actual
// fit only takes seconds (same batching as workflow 09).
params.batch_size = 20
// MAC cutoff of the lowest bin (src/scripts/build_cohort_mac_maf4_bins.py).
params.mac_max     = 5
// mac_ld_6 (default) | mac4 -- see module docstring.
params.binning     = 'mac_ld_6'
if (!(params.binning in ['mac_ld_6', 'mac4'])) error "--binning must be mac_ld_6 or mac4, got '${params.binning}'"
def LD6 = params.binning == 'mac_ld_6'
def BIN_TAG = LD6 ? 'mac_ld_6' : 'mac4bin'
def BIN_METHOD = LD6 ? 'exact_reml_mac_ld_6_covar' : 'exact_reml_mac_maf_bins_4_covar'
// Variance components >= 0 (exact_reml_heritability.py --nonneg), see module docstring.
params.nonneg      = true
def NONNEG_FLAG = params.nonneg ? '--nonneg ' : ''
// --estimator he (default): Haseman-Elston regression (src/scripts/he_heritability.py, jackknife
// SEs per --jackknife, no lrt columns, --nonneg ignored); method he_<binning>_covar, output in
// h2/he_<binning>/. --estimator reml: exact_reml_heritability.py per --nonneg. One estimator per
// run, each into its own --out_dir (same convention as workflow 08).
params.estimator   = 'he'
if (!(params.estimator in ['reml', 'he'])) error "--estimator must be reml or he, got '${params.estimator}'"
def HE = params.estimator == 'he'
def EST_TAG = HE ? 'he' : 'exact'
def ESTIMATOR_SCRIPT = HE ? 'he_heritability.py' : 'exact_reml_heritability.py'
if (HE) NONNEG_FLAG = ''
// --jackknife loo (default): he_heritability.py's delete-one jackknife SEs. --jackknife none
// (--estimator he only): no jackknife, so h2_se and the per-bin/genotype SE columns are NaN.
// The refit per dropped donor is skipped too, which is most of HE's cost.
params.jackknife   = 'loo'
if (!(params.jackknife in ['loo', 'none'])) error "--jackknife must be loo or none, got '${params.jackknife}'"
if (!HE && params.jackknife != 'loo') error "--jackknife none needs --estimator he (exact REML has no jackknife)"
// he_heritability.py writes no lrt/lrt_p; pad them as NaN so the output has the REML run's columns.
def padCmd = { String out -> HE ? "&& python ${SCRIPTS_DIR}/pad_he_columns.py ${out} " : '' }
def JK_FLAG = HE ? "--jackknife ${params.jackknife} " : ''
def methodLabel = { String m -> HE ? m.replaceFirst('^exact_reml_', 'he_') : m }
if (!params.out_dir)    error "Please specify --out_dir"
if (!params.config_tsv) error "Please specify --config_tsv"
if (!(params.cohort in ['gtex', 'geuvadis'])) error "--cohort must be 'gtex' or 'geuvadis', got '${params.cohort}'"

def OUT_PREFIX = "${launchDir}/results/10-genotype_heritability/${params.cohort}/${params.out_dir}"

def EXPRESSION_DATA_DIR     = "${launchDir}/results/02-expression_data_preparation/${params.cohort}"
def EXPRESSION_MANIFEST_DIR = "${EXPRESSION_DATA_DIR}/manifest"

def GENOTYPE_DATA_DIR     = "${launchDir}/results/04-genotype_extraction/${params.cohort}"
def GENOTYPE_MANIFEST_DIR = "${GENOTYPE_DATA_DIR}/manifest"
def GENOTYPE_MANIFEST     = loadGenotypeManifest(GENOTYPE_MANIFEST_DIR)
def GENO_BAND = [min: params.min_maf as double, max: params.max_maf]

// GEUVADIS's three covariate sources (workflow 05: geuvadis_covariates_extraction) --
// fixed paths, no per-tissue parameterization (GEUVADIS is LCL-only, unlike GTEx's
// per-tissue published covariates).
def GEUVADIS_COVAR_DIR = "${launchDir}/results/05-geuvadis_covariates_extraction"

// CPU processes deliberately declare NO `queue` directive: a directive in the process
// body overrides the generic `process.queue` that `-profile slurm` sets, which would
// pin them to the default CPU queue.


// ──────────────────────────────────────────────────────────────────────────── //
// Process 1: raw phenotype (no covariate correction, see module docstring), resolved
//            from workflow 02.
// ──────────────────────────────────────────────────────────────────────────── //

process resolve_pheno {
    cpus 1
    memory { 4.GB * task.attempt }
    time { 20.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    input:
    path(config_tsv)

    output:
    path("resolved_pheno.tsv"), emit: table
    path("*_pheno.parquet"), emit: phenos

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
// Process 2: GTEx's own published per-tissue eQTL covariates, transposed to the
//            IID-indexed TSV every estimator's --covar/--covariate-file expects.
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
    def covarCmd = params.cohort == 'geuvadis' ?
        "python ${SHARED_DIR}/build_geuvadis_covar.py " +
            "--peer-factors ${GEUVADIS_COVAR_DIR}/peer_factors.parquet " +
            "--genotype-pcs ${GEUVADIS_COVAR_DIR}/genotype_pcs.tsv " +
            "--technical-covariates ${GEUVADIS_COVAR_DIR}/technical_covariates.tsv " +
            "--out covar_peer.tsv" :
        "python ${SHARED_DIR}/build_gtex_eqtl_covar.py " +
            "--covariates-file ${GTEX_COVAR_DIR}/${tissueFile}.v10.covariates.txt " +
            "--out covar_peer.tsv"
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    ${covarCmd}
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 3: heritability fit on the --binning partition (see module docstring). BATCHED
//            -- one SLURM task fits params.batch_size genes: each gene's fit takes
//            seconds to minutes, so a task-per-gene array would let SLURM's own
//            scheduling overhead dominate. Genotype/phenotype paths are passed as plain
//            absolute-path strings (`val`, not `path`): every gene's region.bed/.bim/.fam
//            sits in its own workflow-04 manifest directory under an identical filename,
//            so staging N genes' files into one flat task directory would collide; nodes
//            here have direct access to the same shared filesystem the manifest paths
//            point at, so nothing needs staging. The process name, inputs and the mac4
//            command are kept as they were before --binning existed, so earlier mac4 runs
//            still -resume from cache; split_bins_by_ld.py is called by absolute path for
//            the same reason (a new staged input would change every task's hash).
// ──────────────────────────────────────────────────────────────────────────── //

process heritability_mac4bin_batch {
    array 1000
    cpus 1
    memory { 2.GB * task.attempt }
    time { 4.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${BIN_TAG}:batch_${task.index}"

    publishDir { "${OUT_PREFIX}/h2/${EST_TAG}_${BIN_TAG}" }, mode: 'copy', pattern: "*_${EST_TAG}_${BIN_TAG}.parquet"

    input:
    val(batch)
    path(covar)
    path(exact_reml_script)
    path(bin_assignment_script)

    output:
    path("*_${EST_TAG}_${BIN_TAG}.parquet"), emit: h2

    script:
    def cmds = batch.collect { g ->
        def (experiment, bedPrefix, mafPath, phenoPath) = g
        // mac_ld_6: the 4-bin assignment's two common bins split at their median LD score
        def ldSplit = LD6 ?
            "python ${SHARED_DIR}/split_bins_by_ld.py --bfile ${bedPrefix} --pheno ${phenoPath} " +
            "--bin-assignment ${experiment}_mac4bin.assign.txt --bins 3,4 " +
            "--out ${experiment}_${BIN_TAG}.assign.txt && " : ''
        "python ${bin_assignment_script} --bfile ${bedPrefix} --pheno ${phenoPath} " +
        "--plink-bin ${PLINK_BIN} --mac-max ${params.mac_max} --out ${experiment}_mac4bin.assign.txt && " +
        ldSplit +
        "python ${exact_reml_script} --bfile ${bedPrefix} --pheno ${phenoPath} " +
        "--bin-assignment ${experiment}_${BIN_TAG}.assign.txt " +
        "--restrict-grm-to-cohort --covariate-file ${covar} ${NONNEG_FLAG}${JK_FLAG}" +
        "--experiment '${experiment}' --method ${methodLabel(BIN_METHOD)} " +
        "--out ${experiment}_${EST_TAG}_${BIN_TAG}.parquet " +
        padCmd("${experiment}_${EST_TAG}_${BIN_TAG}.parquet") +
        "|| echo '[warn] ${EST_TAG} fit (${BIN_TAG}) failed for ${experiment}' >&2"
    }.join('\n')
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    ${cmds}
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
workflow {

    def tissues = [] as Set
    new File(params.config_tsv).withReader { r ->
        def header = r.readLine().split('\t')
        r.eachLine { line ->
            if (!line.trim()) return
            def vals = line.split('\t', -1)
            def row = [:]
            header.eachWithIndex { h, i -> row[h] = i < vals.size() ? vals[i].trim() : '' }
            if (!row.experiment || !row.gene_id || !row.region || !row.tissue)
                error "config row missing a required field: ${row}"
            tissues << row.tissue
        }
    }
    if (tissues.size() != 1)
        error "This workflow assumes a single tissue for the shared PEER covariate file; got: ${tissues}"

    resolved = resolve_pheno(Channel.value(file(params.config_tsv)))
    pheno_files = resolved.phenos.flatten().map { f -> tuple(f.name - '_pheno.parquet', f) }
    pheno = resolved.table
        .splitCsv(header: true, sep: '\t')
        .map { r -> tuple(new File(r.pheno_path).name - '_pheno.parquet', r.experiment.trim()) }
        .combine(pheno_files, by: 0)
        .map { tag, e, pf -> tuple(e, pf) }

    covar = build_peer_covar(Channel.value(tissues.first()))

    // ── genotype: resolved synchronously from workflow 04's manifest, same
    //    fail-fast pattern as workflows 11/16 ──────────────────────────────── //
    def missingGeno = []
    def resolvedGeno = []
    new File(params.config_tsv).withReader { r ->
        def header = r.readLine().split('\t')
        r.eachLine { line ->
            if (!line.trim()) return
            def vals = line.split('\t', -1)
            def row = [:]
            header.eachWithIndex { h, i -> row[h] = i < vals.size() ? vals[i].trim() : '' }
            def bfRow = lookupGenoBfile(GENOTYPE_MANIFEST, row.region, params.window_size, GENO_BAND, params.vcf_filter)
            if (!bfRow) {
                missingGeno << "${row.experiment}: region_bfile for ${row.region}"
            } else {
                // Plain absolute-path STRINGS (not Nextflow `file()`/`path`) -- consumed
                // downstream as `val` by the batched processes below, deliberately never
                // staged (see Process 3's own comment on why).
                def bedPrefix = "${GENOTYPE_DATA_DIR}/${bfRow.bed_path}" - '.bed'
                def mafPath   = "${GENOTYPE_DATA_DIR}/${bfRow.maf_path}"
                resolvedGeno << tuple(row.experiment, bedPrefix, mafPath)
            }
        }
    }
    if (missingGeno) {
        error "Missing from workflow 04's manifest (${GENOTYPE_MANIFEST_DIR}) under " +
            "window_size=${params.window_size}, min_maf=${params.min_maf}, " +
            "max_maf=${params.max_maf}, vcf_filter=${params.vcf_filter} -- run workflow 04 " +
            "(genotype_extraction) for these regions first:\n" + missingGeno.join('\n')
    }

    // tuple(experiment, bedPrefix, mafPath, phenoPath) -- phenoPath is still the
    // Nextflow Path object resolve_pheno emitted, just consumed as `val` (no
    // re-staging) by the batched processes below.
    geno_pheno = Channel.fromList(resolvedGeno)
        .combine(pheno, by: 0)
        .map { e, bedPrefix, mafPath, p -> tuple(e, bedPrefix, mafPath, p) }
    batches = geno_pheno.collate(params.batch_size)

    // covar.first(): covar is a single-item process-output (queue) channel, but each
    // batched process runs once per batch (not once per gene) -- without .first() to
    // make it replay/broadcast, it would pair with only the first batch and starve
    // every other invocation.
    heritability_mac4bin_batch(
        batches, covar.first(),
        Channel.value(file("${SHARED_DIR}/${ESTIMATOR_SCRIPT}")),
        Channel.value(file("${SHARED_DIR}/build_cohort_mac_maf4_bins.py")))
}
