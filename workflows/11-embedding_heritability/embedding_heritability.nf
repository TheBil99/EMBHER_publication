#!/usr/bin/env nextflow

/*
 * embedding_heritability -- heritability of a real GTEx gene's expression from frozen
 * Borzoi embeddings, plus the R2 of Borzoi's own predicted-expression output track
 * against the same measured phenotype.
 *
 * Genotype-GRM heritability on these genes used to be computed here too; it has moved
 * to workflow 10 (genotype_heritability), so this workflow does the embedding-GRM side
 * plus the prediction-R2 side, plus (joint_heritability process) the joint
 * genotype+embedding fit needed to compare the two kernels' shares of one total h2 --
 * same approach as workflow 09's exact_from_grm_joint, ported from simulated genotypes
 * onto workflow 04's real per-gene GTEx bfiles (the panel workflow 10 fits on): the
 * region's genotype GRM (default MAC/MAF 4-bin split, build_cohort_mac_maf4_bins.py,
 * --mac_max) and the embedding kernel (the same .npz embedding_heritability already
 * builds) are fit jointly in one exact_reml_heritability.py call
 * (--bfile ... --bin-assignment ... --grm-npz ..., method exact_reml_joint_mac4bin_covar),
 * giving h2_genotype (sum of the 4 geno bins) and h2_k5 (the embedding component) as
 * shares of one joint total h2 -- not the sum of the two kernels' independently-fit h2,
 * which double-counts whatever variance the two kernels tag in common. Deliberately no
 * standalone genotype-only fit here: pure genotype heritability on these genes is
 * already workflow 10's result, and comparing it against this workflow's own numbers
 * is workflow 17's job, not this one's.
 *
 * The embedding kernel's h2 (embedding_heritability process) is estimated by Haseman-
 * Elston regression by default (src/scripts/he_heritability.py, --estimator he), with the
 * jackknife off (h2_se = NaN; --he_jackknife true turns it on). --estimator reml switches
 * to this project's exact multi-component AI-REML (src/scripts/exact_reml_heritability.py,
 * --grm-npz single-kernel mode). The embedding is PCA-whitened (pca_embeddings.py,
 * hardcoded PC_FRAC = 0.7, i.e. 70% of the cohort's rank) before the GRM is built, under
 * either estimator.
 *
 * Embeddings/predictions come from workflow 06 (extract_embeddings), not workflow 06:
 * 04b writes one HDF5 per gene (embher-capture's "stacked" layout -- sample and
 * haplotype are array axes, nothing pre-combined across haplotypes), for a fixed
 * ~100-gene list under one fixed configuration. This workflow resolves each gene
 * against 04b's manifest (one row per gene_id -- no window/MAF/tracks cache-key
 * variants to disambiguate, unlike workflow 06's), then runs
 * src/scripts/combine_haplotypes.py to sum the two haplotypes into the donor x feature
 * embedding table (embeddings/final__seq_mean, the 1920-dim tensor immediately before
 * Borzoi's output head) and the donor x pred_expression table (exon_sum_unsquashed,
 * summed over haplotypes and over --pred_tracks). A config gene not yet captured by
 * 04b fails the whole run immediately, same "never silently fall back" pattern as the
 * expression/genotype manifest lookups below.
 *
 * Same estimator as workflow 08 (both call the same scripts in src/scripts, so
 * the two analyses cannot drift apart); what changes is that the phenotype is measured
 * rather than simulated, which brings in two things simulations did not need:
 *
 *   covariates   GTEx's own released genotype PCs + sex + age (never PCs computed
 *                here), added to the fit as fixed effects (--covariate-file),
 *                not regressed out of the phenotype beforehand.
 *   n ~ 800      workflow 08's heritability-vs-N curves show that at this N the
 *                sampling spread of h2 is comparable to h2 itself, so no single
 *                gene's value means much on its own (this workflow no longer reports
 *                a permutation null for it -- see above).
 *
 * --nonneg (default true; applies only with --estimator reml, ignored by HE):
 * embedding_heritability and joint_heritability run
 * exact_reml_heritability.py --nonneg -- every variance component constrained >= 0
 * (methods exact_reml_nonneg_*), with chi-bar-squared LRTs in place of Wald tests:
 *   embedding_heritability   lrt_p      embedding h2 = 0
 *   joint_heritability       lrt_p      total h2 = 0 (4 genotype bins + embedding)
 *                            lrt_npz_p  embedding h2 = 0 given the genotype bins
 *                                       (needs one extra genotype-only fit per gene)
 * --nonneg false restores the unconstrained fits.
 *
 * The tissue TPM matrix and the per-gene phenotype are not computed here either:
 * workflow 02 (expression_data_preparation) is the sole producer of both, and this
 * workflow resolves each tissue/gene against its manifest
 * (src/scripts/resolve_expression.py) instead, keyed by tissue + --transform so this
 * workflow's own flavor of the phenotype is never silently conflated with another
 * consumer's. No covariate correction is applied -- the phenotype is the transformed
 * value as-is (see prepare_expression_matrix.py).
 * No permutation null is computed here.
 *
 * Config TSV columns (tab-separated, with header) -- build with scripts/make_config.py:
 *   experiment  gene_id  gene_name  region  tissue
 *     region : chrN:TSS-TSS (only used to resolve the phenotype; 04b's own window is
 *     fixed and TSS-centred already)
 *
 * Usage:
 *   nextflow run workflows/11-embedding_heritability/embedding_heritability.nf \
 *     --config_tsv workflows/11-embedding_heritability/configs/<config>.tsv \
 *     --embeddings_dir results/06-extract_embeddings/gtex/borzoi/<cache_key> \
 *     --out_dir <run_name>
 *
 * --cohort gtex|geuvadis (default gtex) selects workflow 02's/03's cohort-specific
 * results dirs and, for the covariate step, build_gtex_eqtl_covar.py vs.
 * build_geuvadis_covar.py (workflow 05). Output goes to
 * results/11-embedding_heritability/${cohort}/${out_dir}. --embeddings_dir and
 * --pred_tracks are NOT switched by --cohort -- pass a GEUVADIS-cohort workflow 06
 * run's borzoi/<cache_key> dir and an explicit, non-GTEx --pred_tracks yourself.
 */

// Region PLINK bfiles for the joint_heritability process are looked up from workflow
// 03's (genotype_extraction) manifest, same "sole producer + manifest lookup" pattern
// as the embeddings manifest below -- this workflow never extracts genotypes itself.
include { loadGenotypeManifest; lookupGenoBfile } from '../../src/nf/genotype_manifest.nf'

// ── Constants ─────────────────────────────────────────────────────────────── //

def SCRIPTS_DIR  = "${projectDir}/scripts"
// Scripts used by two or more workflows (estimation code shared with 03; expression
// manifest resolution shared with 05/06/08) live in src/scripts, not in one workflow's
// own dir.
def SHARED_DIR   = "${projectDir}/../../src/scripts"
// GTEx's own published per-tissue eQTL covariates (60 PEER factors + genotype PCs +
// sex/platform/pcr for Whole Blood) -- same source and script workflow 10 uses, so
// embedding heritability's fixed effects and genotype heritability's fixed effects
// are the identical covariate set.
def GTEX_COVAR_DIR = "${projectDir}/../../data/raw/GTEx/v10/GTEx_Analysis_v10_eQTL_covariates"
// GEUVADIS's three covariate sources (workflow 05: geuvadis_covariates_extraction) --
// fixed paths, no per-tissue parameterization (GEUVADIS is LCL-only, unlike GTEx's
// per-tissue published covariates).
def GEUVADIS_COVAR_DIR = "${launchDir}/results/05-geuvadis_covariates_extraction"
// Matches correlate_all.py's own slug() exactly -- that script pairs a prediction
// file to its gene by checking `slug(gene_id) in filename`, so the prediction
// filename must contain the SLUGGED gene_id (dots replaced), not the raw versioned
// one (e.g. "ENSG00000110455.16" would never match its own slug "ENSG00000110455_16"
// as a substring).
def slug = { s -> s.replaceAll(/[^A-Za-z0-9]/, '_') }
def ANALYSIS_ENV = params.env_analysis
def CONDA_SH     = params.conda_sh
// plink2 -- used by build_cohort_mac_maf4_bins.py (joint_heritability process), same
// path workflows 04/09/10 use.
def PLINK = params.plink

// ── Parameters ────────────────────────────────────────────────────────────── //

params.out_dir     = null
params.config_tsv  = null
// gtex | geuvadis -- selects workflow 02's/03's cohort-specific results dirs and, for
// the covariate step, build_gtex_eqtl_covar.py's single published per-tissue file vs.
// build_geuvadis_covar.py's join of workflow 05's three GEUVADIS covariate sources.
// Does NOT affect --embeddings_dir (always passed explicitly) or --pred_tracks (see
// below -- GTEx's default blood tracks don't apply to a non-GTEx cohort).
params.cohort      = 'gtex'
// Workflow 06's (extract_embeddings) borzoi/<cache_key> output dir for this run, e.g.
// results/06-extract_embeddings/gtex/borzoi/borzoi_replicate_0_release_46_..._bfloat16 --
// see the manifest lookup in the workflow{} block below.
params.embeddings_dir = null
// Phenotype construction (prepare_expression_matrix.py, via workflow 02). No covariate
// correction is applied -- see that script's docstring. Defaults to quantile_int to match
// workflow 10's TRANSFORM -- so
// embedding h2 (here), genotype h2 (workflow 10) and prediction R2 (here) are all
// computed against the identical phenotype definition and stay comparable.
params.transform     = 'quantile_int'     // quantile_int | log2_plus_1
// Whiten the embedding onto its top-k principal components before the GRM is built
// (src/scripts/pca_embeddings.py), instead of the raw column-standardised embedding --
// the redundant-column problem it fixes (see pca_embeddings.py's docstring) is a
// property of the Borzoi embedding. Retain PC_FRAC of the analysis cohort's own
// numerical rank (pca_embeddings.py's own --pc-frac), not a fixed absolute count: a
// fixed count stops meaning "most of the variance" once the cohort's embedding rank
// falls under it (e.g. GEUVADIS's ~449 donors vs. GTEx's ~800), leaving ~0 residual
// degrees of freedom. Hardcoded, not exposed
// as a param -- every run uses the identical whitening fraction.
def PC_FRAC = 0.7
// Variance components >= 0 (exact_reml_heritability.py --nonneg), see module docstring.
params.nonneg      = true
def NONNEG_FLAG = params.nonneg ? '--nonneg' : ''

// --estimator he (default): Haseman-Elston regression (he_heritability.py; no LRT
// p-values; --nonneg is ignored). Jackknife SEs are off by default (h2_se = NaN) -- pass
// --he_jackknife true to turn them on. Methods he_from_grm / he_joint_mac4common_covar.
// --estimator reml: exact_reml_heritability.py, per --nonneg above.
params.estimator   = 'he'
params.he_jackknife = false
if (!(params.estimator in ['reml', 'he'])) error "--estimator must be reml or he, got '${params.estimator}'"
def HE = params.estimator == 'he'
def ESTIMATOR_SCRIPT = HE ? 'he_heritability.py' : 'exact_reml_heritability.py'
if (HE) NONNEG_FLAG = ''
def FIT_FLAGS = HE ? "--jackknife ${params.he_jackknife ? 'loo' : 'none'}" : NONNEG_FLAG
// Borzoi human-head tracks used as the predicted-expression readout. 7531-7533 are the
// three GTEx whole-blood RNA-seq samples (recount3, all under datasets/.../blood/)
// among Borzoi's training targets -- the closest available match to the phenotype.
// NB blood_vessel (7534-7536) is a different tissue and PBMC tracks are a cell subset.
// This default is GTEx-whole-blood-specific: for --cohort geuvadis (LCL), there is no
// GTEx-donor-matched track, so --pred_tracks must be passed explicitly (e.g. one of
// the ENCODE GM12878 whole-cell RNA tracks, NOT its cytosolic/nuclear/nucleolus
// fraction tracks) -- see the validation below.
params.pred_tracks = '7531,7532,7533'

// Genes per task for combine_haplotypes, embedding_heritability and joint_heritability.
params.batch_size = 25
// joint_heritability's genotype side: which of workflow 04's (genotype_extraction)
// cached region bfiles this run reads -- must match whatever workflow 04 was run with
// for these genes, or the manifest lookup below will not find them. Same
// window_size/min_maf/max_maf/vcf_filter defaults as workflow 10, so the two
// workflows' genotype components are built from the identical panel.
params.window_size = 524288
params.min_maf     = 0.0
params.max_maf     = null
params.vcf_filter  = null
// Cohort MAC/MAF 4-bin split (build_cohort_mac_maf4_bins.py): MAC <= mac_max | MAC >
// mac_max & MAF <= 0.01 | (0.01, 0.1] | (0.1, 0.5] -- same default as workflow 10.
params.mac_max     = 5
// Genotype part of joint_heritability: 'mac4' = the 4 bins above; 'mac4_common' = only
// the two common bins, (0.01, 0.1] and (0.1, 0.5] -- the MAC <= mac_max and MAF <= 0.01
// bins are left out of the fit (e.g. to set a rare-variant embedding against common
// genotype only). Method exact_reml[_nonneg]_joint_mac4bin_covar / _joint_mac4common_covar.
params.joint_genotype_bins = 'mac4'
// --joint false skips the whole joint genotype+embedding side (genotype manifest lookup,
// joint_heritability, plot_joint_heritability); only the embedding and prediction fits run.
params.joint = true
if (!(params.joint_genotype_bins in ['mac4', 'mac4_common']))
    error "--joint_genotype_bins must be 'mac4' or 'mac4_common', got '${params.joint_genotype_bins}'"
def JOINT_TAG = params.joint_genotype_bins == 'mac4' ? 'mac4bin' : 'mac4common'

if (!params.out_dir)    error "Please specify --out_dir"
if (!params.config_tsv) error "Please specify --config_tsv"
if (!params.embeddings_dir) error "Please specify --embeddings_dir (workflow 06's " +
    "borzoi/<cache_key> output directory)"
if (!(params.cohort in ['gtex', 'geuvadis'])) error "--cohort must be 'gtex' or 'geuvadis', got '${params.cohort}'"
if (params.cohort == 'geuvadis' && params.pred_tracks == '7531,7532,7533')
    error "--pred_tracks is still at its GTEx-whole-blood default (7531,7532,7533), " +
        "which doesn't apply to --cohort geuvadis (LCL) -- pass an explicit " +
        "--pred_tracks (e.g. a GM12878 whole-cell RNA track from Borzoi's targets.txt; " +
        "avoid its cytosolic/nuclear/nucleolus fraction tracks, not comparable to bulk " +
        "GEUVADIS RNA-seq)."

def OUT_PREFIX   = "${launchDir}/results/11-embedding_heritability/${params.cohort}/${params.out_dir}"

// 04's manifest reader (src/nf/embeddings_manifest.nf): the 04 directory name no longer carries the
// pooling, so the rows are filtered on the manifest's own pooling columns.
include { loadEmbeddingsManifest } from '../../src/nf/embeddings_manifest.nf'
def EMBEDDINGS_MANIFEST = loadEmbeddingsManifest(params.embeddings_dir, [pooling_axes: ['seq_dim'], pooling_funcs: ['mean']])

// Whether --embeddings_dir points at a MAF-restricted 04b capture (workflow 06's own
// cacheKeyFor folds min_maf/max_maf/donors_config into the directory whenever a band is
// set) is otherwise only visible by eyeballing the --embeddings_dir path string. Every
// row of this run's manifest carries the identical min_maf/max_maf/maf_cohort/maf_label
// (one fixed configuration per 04b run, same reasoning as EMBEDDINGS_MANIFEST itself),
// so any one row's values describe the whole run -- recorded into this workflow's own
// output (record_embeddings_metadata below) so a MAF-bounded run is traceable from its
// results alone, not just from the launch command that produced them.
def mafInfoFrom = { manifest ->
    def row = manifest.values().find { true }
    [
        min_maf   : row?.min_maf ?: '',
        max_maf   : row?.max_maf ?: '',
        maf_cohort: row?.maf_cohort ?: '',
        maf_label : row?.maf_label ?: '',
    ]
}
def EMBEDDINGS_MAF_INFO = mafInfoFrom(EMBEDDINGS_MANIFEST)

// Workflow 02's (expression_data_preparation) cache: the tissue TPM matrix and every
// gene's residualized phenotype are looked up there (src/scripts/resolve_expression.py)
// instead of computed here -- same "sole producer + manifest lookup" pattern as the
// embeddings manifest above, applied to expression prep.
def EXPRESSION_DATA_DIR     = "${launchDir}/results/02-expression_data_preparation/${params.cohort}"
def EXPRESSION_MANIFEST_DIR = "${EXPRESSION_DATA_DIR}/manifest"

// Workflow 04's (genotype_extraction) manifest -- joint_heritability's genotype side is
// looked up here instead of extracted, same "sole producer + manifest lookup" pattern
// as the two manifests above.
def GENOTYPE_DATA_DIR     = "${launchDir}/results/04-genotype_extraction/${params.cohort}"
def GENOTYPE_MANIFEST_DIR = "${GENOTYPE_DATA_DIR}/manifest"
def GENOTYPE_MANIFEST     = loadGenotypeManifest(GENOTYPE_MANIFEST_DIR)
// This workflow only ever asks for one MAF band (its own scalar min_maf/max_maf) --
// built once here for every genotype lookup.
def GENO_BAND = [min: params.min_maf as double, max: params.max_maf]

// CPU processes deliberately declare NO `queue` directive: a directive in the process
// body overrides the generic `process.queue` that `-profile slurm` sets, which would pin
// them to the default CPU queue.
// (Config precedence: withName selectors > process body directives > generic process scope.)


// ──────────────────────────────────────────────────────────────────────────── //
// Process 3: resolve every gene in the config to its cached phenotype (workflow 02's
//            expression_data_preparation), under this run's own --transform, once, up
//            front -- fails on any unresolvable gene before spending a single per-gene
//            SLURM job.
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
        --transform ${params.transform} \\
        --out resolved_pheno.tsv
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process: GTEx's own published per-tissue eQTL covariates, transposed to the
//          IID-indexed TSV exact_reml_heritability.py's --covariate-file and
//          correlate_predictions.py's --covar both expect -- same script and source
//          workflow 10 uses, so
//          embedding heritability's fixed effects and the R2/rho2 correction below
//          are the identical covariate set. This workflow assumes a single tissue
//          across the whole config (same assumption workflow 10 makes), since the
//          covariate file is shared across every gene's fit.
// ──────────────────────────────────────────────────────────────────────────── //

// ──────────────────────────────────────────────────────────────────────────── //
// Process: record which embedding MAF band/cohort (if any) --embeddings_dir was built
//          under, straight from EMBEDDINGS_MAF_INFO above -- a run against a
//          MAF-bounded 04b capture is otherwise traceable only by eyeballing the
//          --embeddings_dir path, not from this workflow's own results/ directory.
//          exec: (no conda activate, no shell) since this only ever writes five values
//          already resolved in Groovy at workflow-definition time.
// ──────────────────────────────────────────────────────────────────────────── //

process record_embeddings_metadata {
    cpus 1
    memory { 1.GB * task.attempt }
    time { 5.min * task.attempt }

    publishDir { "${OUT_PREFIX}" }, mode: 'copy'

    input:
    val(embeddings_dir)
    val(min_maf)
    val(max_maf)
    val(maf_cohort)
    val(maf_label)
    val(out_name)   // always "embeddings_metadata.tsv"

    output:
    path("${out_name}")

    exec:
    task.workDir.resolve(out_name).text =
        "embeddings_dir\tmin_maf\tmax_maf\tmaf_cohort\tmaf_label\n" +
        "${embeddings_dir}\t${min_maf}\t${max_maf}\t${maf_cohort}\t${maf_label}\n"
}

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
// Process 5: embedding GRM on this gene's donors. (shared with workflow 08)
// ──────────────────────────────────────────────────────────────────────────── //

process embedding_heritability {
    array 1000        // pack tasks of this process into one SLURM array
    cpus 1
    memory { 1.GB * task.attempt }
    time { 6.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "batch_${task.index}"

    // GRM construction and the REML fit were separate processes; merged because they
    // are 1:1 per gene, nothing else consumes the GRM, and each was only ~15 s. The GRMs
    // are not published (several GB per run; rebuilt from the embedding in seconds).
    publishDir { "${OUT_PREFIX}/heritability" }, mode: 'copy',
        pattern: "*_heritability.parquet"

    input:
    // BATCHED: one task runs params.batch_size genes one after another. The four
    // lists are parallel (same gene order); every file is named after its experiment,
    // so staging many genes' files into one task directory cannot collide.
    tuple val(experiments), path(phenos), path(donors), path(embs)
    // Staged explicitly (not just interpolated via ${SHARED_DIR} in the script
    // string below) so an edit to either script invalidates this process's
    // Nextflow cache -- a path referenced only by string interpolation isn't
    // part of the task hash, so -resume would otherwise silently replay a
    // stale result computed by a since-changed exact_reml_heritability.py/build_grm.py.
    // Same pattern the summarize process already uses for its own driver scripts.
    path(build_grm_script)
    path(exact_reml_script)
    path(pca_embeddings_script)
    path(covar)

    output:
    path("*_heritability.parquet"), emit: h2
    // emit:npz so joint_heritability can reuse the identical kernel (same whitening)
    // instead of rebuilding it -- the experiment is the filename prefix, which the
    // workflow block below turns back into the by:0 join key.
    path("*.npz"), emit: npz
    path("*.grm.bin")
    path("*.grm.N.bin")
    path("*.grm.id")

    script:
    // The GRM is built from the whitened top-PC_FRAC-of-rank PC scores instead of the
    // raw embedding; pca_embeddings.py emits unit-variance columns, so build_grm.py's
    // own column standardisation is a deliberate no-op on them. See PC_FRAC's own
    // comment above for why this whitening step exists and why it's a fraction of
    // rank, not a fixed count. Whitening is independent of which algorithm then
    // estimates h2 from the resulting kernel, so it is unchanged by which estimator
    // runs next.
    def cmds = experiments.indices.collect { i ->
        def experiment = experiments[i]
        def pheno = phenos[i]
        def emb = embs[i]
        """
    python ${pca_embeddings_script} \\
        --embeddings ${emb} \\
        --pheno ${pheno} \\
        --pc-frac ${PC_FRAC} \\
        --out ${experiment}_pcs.parquet

    python ${build_grm_script} \\
        --embeddings ${experiment}_pcs.parquet \\
        --pheno ${pheno} \\
        --out ${experiment}

    python ${exact_reml_script} \\
        --grm-npz ${experiment}.npz \\
        --pheno ${pheno} \\
        --covariate-file ${covar} ${FIT_FLAGS} \\
        --experiment '${experiment}' \\
        --out ${experiment}_heritability.parquet
"""
    }.join('\n')
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
${cmds}
    """
}

// ──────────────────────────────────────────────────────────────────────────── //
// Process 5b: joint genotype+embedding heritability -- same approach as workflow 09's
//             exact_from_grm_joint, on workflow 04's real per-gene GTEx bfile instead
//             of a simulated one. Fits the region's genotype GRM (default MAC/MAF
//             4-bin cohort split) and this gene's embedding kernel (the identical .npz
//             embedding_heritability above already built) as ONE joint REML, so
//             h2_genotype and h2_k5 (the embedding's own component) are shares of one
//             total h2, not two independently-fit h2 values that would double-count
//             whatever variance the two kernels tag in common. Deliberately no
//             standalone genotype-only arm here -- pure genotype heritability is
//             workflow 10's result already; comparing it against this workflow's own
//             numbers is workflow 17's job, not this one's. --joint_genotype_bins
//             mac4_common keeps only the two common bins (MAF > 0.01): h2_k1..h2_k2
//             genotype, h2_k3 the embedding.
// ──────────────────────────────────────────────────────────────────────────── //

process joint_heritability {
    array 1000        // pack tasks of this process into one SLURM array
    cpus 1
    memory { 2.GB * task.attempt }
    time { 8.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "batch_${task.index}"

    publishDir { "${OUT_PREFIX}/heritability_joint" }, mode: 'copy',
        pattern: "*_joint_heritability.parquet"

    input:
    // BATCHED: one task fits params.batch_size genes one after another. `genos` is a
    // list of (experiment, bfile prefix) read by path, not staged -- every gene's
    // region.bed/.bim/.fam sits in its own workflow-04 manifest directory under an
    // identical filename, so staging N genes' files into one task directory would
    // collide (same reasoning as workflow 10's batched processes). phenos/emb_npzs are
    // named after their experiment and are staged, in the same gene order as genos.
    tuple val(genos), path(phenos), path(emb_npzs)
    // Staged explicitly, same reasoning as embedding_heritability's own
    // build_grm_script/exact_reml_script -- an edit to either script invalidates this
    // process's Nextflow cache instead of being replayed unchanged under -resume.
    path(bin_assignment_script)
    path(exact_reml_script)
    path(covar)

    output:
    path("*_joint_heritability.parquet"), emit: joint

    script:
    def cmds = genos.indices.collect { i ->
        def (experiment, bfile) = genos[i]
        def pheno = phenos[i]
        def emb_npz = emb_npzs[i]
        // --joint_genotype_bins mac4_common: bins 1-2 (MAC <= mac_max, MAF <= 0.01) -> 0
        // (in no bin), bins 3-4 -> 1-2
        def commonOnly = params.joint_genotype_bins == 'mac4_common' ?
            "awk '{ print (\$1 >= 3 ? \$1 - 2 : 0) }' ${experiment}_mac4bin.assign.txt > ${experiment}_mac4common.assign.txt" : ""
        """
    python ${bin_assignment_script} \\
        --bfile ${bfile} \\
        --pheno ${pheno} \\
        --plink-bin ${PLINK} \\
        --mac-max ${params.mac_max} \\
        --out ${experiment}_mac4bin.assign.txt
    ${commonOnly}

    python ${exact_reml_script} \\
        --bfile ${bfile} \\
        --bin-assignment ${experiment}_${JOINT_TAG}.assign.txt \\
        --grm-npz ${emb_npz} \\
        --pheno ${pheno} \\
        --restrict-grm-to-cohort \\
        --covariate-file ${covar} ${FIT_FLAGS} \\
        --experiment '${experiment}' \\
        --method ${HE ? 'he_' : 'exact_reml_' + (params.nonneg ? 'nonneg_' : '')}joint_${JOINT_TAG}_covar \\
        --out ${experiment}_joint_heritability.parquet
"""
    }.join('\n')
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${cmds}
    """
}

// ──────────────────────────────────────────────────────────────────────────── //
// Process 6: resolve every gene in the config against workflow 06's manifest and sum
//            its two haplotypes into the donor x feature embedding table and the
//            donor x pred_expression table -- see the file header and
//            src/scripts/combine_haplotypes.py for what gets summed and why.
// ──────────────────────────────────────────────────────────────────────────── //

process combine_haplotypes {
    array 1000        // pack tasks of this process into one SLURM array
    cpus 1
    memory { 8.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "batch_${task.index}"

    publishDir { "${OUT_PREFIX}/predictions_raw" }, mode: 'copy', pattern: "*_prediction.parquet"

    input:
    // BATCHED: one task combines params.batch_size genes, each a tuple of
    // (experiment, gene_id_slug, gene_id_h5, h5_path). gene_id_slug is the config's
    // (versioned) gene_id, slugged -- needed in the prediction's filename since
    // correlate_all.py matches a prediction to its gene by checking
    // `slug(gene_id) in filename`, and the raw versioned id (dots) would never match
    // its own slug as a substring. gene_id_h5 is the unversioned id 04b's H5 actually
    // stores its gene group under. The H5s are read by path (not staged as `path`,
    // same as workflow 10's batched processes) -- nodes see the same shared
    // filesystem, and staging many large H5s per task would be wasted work.
    val(batch)
    // Staged explicitly (see embedding_heritability's build_grm_script for why) so an
    // edit to combine_haplotypes.py invalidates this process's Nextflow cache.
    path(combine_script)

    output:
    path("*_embedding.parquet"), emit: emb
    path("*_prediction.parquet"), emit: pred

    script:
    def cmds = batch.collect { g ->
        def (experiment, geneSlug, geneH5, h5) = g
        "python ${combine_script} --h5 ${h5} --gene-id ${geneH5} --tracks ${params.pred_tracks} " +
        "--out-embedding ${experiment}_embedding.parquet " +
        "--out-prediction ${experiment}_${geneSlug}_prediction.parquet"
    }.join('\n')
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    ${cmds}
    """
}

// ──────────────────────────────────────────────────────────────────────────── //
// Process 9c: do Borzoi's predictions track measured expression across donors?
// ──────────────────────────────────────────────────────────────────────────── //

process correlate_predictions {
    cpus 1
    memory { 16.GB * task.attempt }
    time { 4.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    // One task for every gene, not one per gene: the per-gene tasks spent their ~30 s
    // on conda/matplotlib startup and on rendering a figure nothing downstream consumed
    // (--figures re-enables them). ~20 min -> ~2 min. A gene that errors is skipped.
    publishDir { "${OUT_PREFIX}/predictions" }, mode: 'copy'

    input:
    path(phenos)
    path(preds)
    path(driver)
    path(worker)
    path(config_tsv)
    path(covar)

    output:
    path("*_prediction_correlation.parquet"), emit: summary

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${driver} \\
        --config-tsv ${config_tsv} \\
        --pheno ${phenos} \\
        --predictions ${preds} \\
        --worker ${worker} \\
        --covar ${covar}
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 9d: heritability of Borzoi's predicted expression -- the one-component
//             counterpart of correlate_predictions' R2. The per-donor pred_expression
//             is a single feature, so build_grm.py's standardised K = z z' is rank 1;
//             the same estimator (--estimator) and covariates as the embedding fit then
//             give h2 of the predictions on the same footing as the embedding h2.
// ──────────────────────────────────────────────────────────────────────────── //

process prediction_heritability {
    array 1000        // pack tasks of this process into one SLURM array
    cpus 1
    memory { 1.GB * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "batch_${task.index}"

    publishDir { "${OUT_PREFIX}/prediction_heritability" }, mode: 'copy',
        pattern: "*_prediction_heritability.parquet"

    input:
    // BATCHED like embedding_heritability: parallel lists, one gene after another
    tuple val(experiments), path(phenos), path(preds)
    path(build_grm_script)
    path(estimator_script)
    path(covar)

    output:
    path("*_prediction_heritability.parquet"), emit: h2

    script:
    def cmds = experiments.indices.collect { i ->
        def experiment = experiments[i]
        """
    python ${build_grm_script} \\
        --embeddings ${preds[i]} \\
        --pheno ${phenos[i]} \\
        --out ${experiment}_prediction

    python ${estimator_script} \\
        --grm-npz ${experiment}_prediction.npz \\
        --pheno ${phenos[i]} \\
        --covariate-file ${covar} ${FIT_FLAGS} \\
        --experiment '${experiment}' \\
        --out ${experiment}_prediction_heritability.parquet
"""
    }.join('\n')
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
${cmds}
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 10: ALL summaries in one task -- the per-gene PDFs and the cross-gene
//             scatter. Merged because the per-gene render is a few seconds, so one
//             task per gene meant hundreds of near-instant jobs. Publishing keeps the
//             per-gene summaries (per_gene/<GENE>_heritability_summary.*) apart from
//             the overall, cross-gene outputs (overall/across_genes.*,
//             overall/nonconverged.txt) so the overall summary is never buried among
//             hundreds of per-gene files. Trade-off: any change to any gene re-renders
//             all of them (a few minutes), and a malformed gene is skipped rather than
//             failing the task.
// ──────────────────────────────────────────────────────────────────────────── //

process summarize {
    array 1000
    cpus 1
    memory { 12.GB * task.attempt }
    time { 2.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}/summary" }, mode: 'copy'

    input:
    path(embedding_herit)
    path(prediction_corr)
    path(driver)            // staged so edits invalidate the cache (see correlate_predictions)
    path(per_gene_script)
    path(across_script)

    output:
    path("per_gene/*_heritability_summary.parquet")
    path("overall/across_genes.pdf")
    path("overall/across_genes.parquet")
    path("overall/nonconverged.txt")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${driver} \\
        --embedding-herit ${embedding_herit} \\
        --prediction-corr ${prediction_corr} \\
        --per-gene-script ${per_gene_script} \\
        --across-script ${across_script} \\
        --variant 'ancestry corrected' \\
        --out-nonconverged overall/nonconverged.txt
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 11: an additional summary page for the joint genotype+embedding fit --
//             genotype-from-joint (h2_genotype) vs embedding-from-joint (h2_k5), both
//             from the SAME joint fit. No pure-genotype comparison here -- that is
//             workflow 10's result; cross-workflow comparison is workflow 17's job.
// ──────────────────────────────────────────────────────────────────────────── //

process plot_joint_heritability {
    cpus 1
    memory { 4.GB * task.attempt }
    time { 20.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}/summary" }, mode: 'copy'

    input:
    path(joint_h2)
    path(script)

    output:
    path("overall/joint_heritability_summary.parquet")
    path("overall/joint_heritability_summary.pdf")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    mkdir -p overall

    python ${script} \\
        --joint ${joint_h2} \\
        --out-prefix overall/joint_heritability_summary
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

    // GTEx's own PEER covariates (build_peer_covar) assume one shared tissue across
    // the whole config -- same assumption workflow 10 makes, since the covariate
    // file is shared by every gene's fit. Resolved synchronously, same style as the
    // embeddings-manifest scan below.
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
    // build_peer_covar's input is a Channel.value(), so its output is automatically
    // a value channel too (broadcasts to every embedding_heritability/
    // correlate_predictions invocation instead of being exhausted after the first).
    covar = build_peer_covar(Channel.value(tissues.first()))

    record_embeddings_metadata(
        Channel.value(params.embeddings_dir),
        Channel.value(EMBEDDINGS_MAF_INFO.min_maf),
        Channel.value(EMBEDDINGS_MAF_INFO.max_maf),
        Channel.value(EMBEDDINGS_MAF_INFO.maf_cohort),
        Channel.value(EMBEDDINGS_MAF_INFO.maf_label),
        Channel.value("embeddings_metadata.tsv"))

    // per-gene residualized phenotype, looked up from workflow 02's
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

    // ── embeddings + predictions: looked up from workflow 06's manifest ───── //
    // workflow 06 (extract_embeddings) is the sole producer of the per-gene H5s now.
    // Resolved synchronously (not as a reactive channel) so a missing gene fails the
    // whole run immediately, before any downstream process launches -- never silently
    // fall back to local extraction. gene_id is matched unversioned (04b's own
    // gene_meta.tsv/H5 groups carry no GENCODE version suffix).
    def resolveEmbeddingsRows = { manifest, dir ->
        def missing = []
        def resolvedRows = []
        new File(params.config_tsv).withReader { r ->
            def header = r.readLine().split('\t')
            r.eachLine { line ->
                if (!line.trim()) return
                def vals = line.split('\t', -1)
                def row = [:]
                header.eachWithIndex { h, i -> row[h] = i < vals.size() ? vals[i].trim() : '' }
                def geneIdH5 = row.gene_id.split('\\.')[0]
                def manRow = manifest[geneIdH5]
                if (!manRow) {
                    missing << "${row.experiment}: ${row.gene_id} (${geneIdH5})"
                } else {
                    resolvedRows << tuple(row.experiment, slug(row.gene_id), geneIdH5,
                        file("${dir}/${manRow.file_path}"))
                }
            }
        }
        if (missing) {
            error "Missing from workflow 06's manifest (${dir}) -- " +
                "run workflow 06 (extract_embeddings) for these genes first:\n" +
                missing.join('\n')
        }
        resolvedRows
    }

    def EMB_ROWS = resolveEmbeddingsRows(EMBEDDINGS_MANIFEST, params.embeddings_dir)
    // combine_haplotypes names each prediction <experiment>_<gene slug>_prediction.parquet;
    // this recovers the experiment from the exact name (experiments may contain '_')
    def PRED_EXPERIMENT = EMB_ROWS.collectEntries { r -> ["${r[0]}_${r[1]}_prediction.parquet".toString(), r[0]] }

    combined = combine_haplotypes(
        Channel.fromList(EMB_ROWS)
            .collate(params.batch_size),
        Channel.value(file("${SHARED_DIR}/combine_haplotypes.py")))
    // one (experiment, parquet) per gene again; the experiment is the filename prefix
    embeddings  = combined.emb.flatten()
        .map { f -> tuple(f.name - '_embedding.parquet', f) }
    predictions = combined.pred.flatten()

    // params.batch_size genes per task: build the embedding GRM, then fit REML on it
    // (covariates as fixed effects, same covar file correlate_predictions below
    // regresses out of the measured phenotype)
    pheno_with_emb = pheno.map { e, reg, p, d -> tuple(e, reg, p, d) }
        .combine(embeddings, by: 0)
        .map { e, reg, p, d, emb -> tuple(e, reg, p, d, emb) }

    emb_out = embedding_heritability(
        pheno_with_emb.collate(params.batch_size).map { b ->
            tuple(b.collect { it[0] }, b.collect { it[2] }, b.collect { it[3] }, b.collect { it[4] }) },
        Channel.value(file("${SHARED_DIR}/build_grm.py")),
        Channel.value(file("${SHARED_DIR}/${ESTIMATOR_SCRIPT}")),
        Channel.value(file("${SHARED_DIR}/pca_embeddings.py")),
        covar)
    emb_h2  = emb_out.h2.flatten()
    // back to (experiment, npz) for the by:0 join with the genotype side
    emb_npz = emb_out.npz.flatten().map { f -> tuple(f.name - '.npz', f) }

    if (params.joint) {
        // ── genotype: resolved synchronously from workflow 04's manifest, same fail-fast
        //    pattern as the embeddings manifest above ─────────────────────────────── //
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
                    resolvedGeno << tuple(row.experiment,
                        file("${GENOTYPE_DATA_DIR}/${bfRow.bed_path}"),
                        file("${GENOTYPE_DATA_DIR}/${bfRow.bim_path}"),
                        file("${GENOTYPE_DATA_DIR}/${bfRow.fam_path}"))
                }
            }
        }
        if (missingGeno) {
            error "Missing from workflow 04's manifest (${GENOTYPE_MANIFEST_DIR}) under " +
                "window_size=${params.window_size}, min_maf=${params.min_maf}, " +
                "max_maf=${params.max_maf}, vcf_filter=${params.vcf_filter} -- run workflow 04 " +
                "(genotype_extraction) for these regions first:\n" + missingGeno.join('\n')
        }

        joint_out = joint_heritability(
            Channel.fromList(resolvedGeno)
                .combine(pheno.map { e, reg, p, d -> tuple(e, p) }, by: 0)
                .combine(emb_npz, by: 0)
                .collate(params.batch_size)
                .map { b ->
                    tuple(b.collect { [it[0], it[1].toString().replaceAll(/\.bed$/, '')] },
                          b.collect { it[4] }, b.collect { it[5] }) },
            Channel.value(file("${SHARED_DIR}/build_cohort_mac_maf4_bins.py")),
            Channel.value(file("${SHARED_DIR}/${ESTIMATOR_SCRIPT}")),
            covar)

        plot_joint_heritability(
            joint_out.joint.flatten().collect(),
            Channel.value(file("${SCRIPTS_DIR}/plot_joint_heritability.py")))
    }

    corr = correlate_predictions(
        pheno.map { e, reg, p, d -> p }.collect(),
        predictions.collect(),
        Channel.value(file("${SCRIPTS_DIR}/correlate_all.py")),
        Channel.value(file("${SCRIPTS_DIR}/correlate_predictions.py")),
        Channel.value(file(params.config_tsv)),
        covar)

    pred_h2 = prediction_heritability(
        pheno.map { e, reg, p, d -> tuple(e, p) }
            .combine(predictions.map { f -> tuple(PRED_EXPERIMENT[f.name], f) }, by: 0)
            .collate(params.batch_size)
            .map { b -> tuple(b.collect { it[0] }, b.collect { it[1] }, b.collect { it[2] }) },
        Channel.value(file("${SHARED_DIR}/build_grm.py")),
        Channel.value(file("${SHARED_DIR}/${ESTIMATOR_SCRIPT}")),
        covar)

    summarize(emb_h2.collect(),
              corr.summary,
              Channel.value(file("${SCRIPTS_DIR}/summarize_all.py")),
              Channel.value(file("${SCRIPTS_DIR}/summarize_embedding_heritability.py")),
              Channel.value(file("${SCRIPTS_DIR}/plot_across_genes.py")))
}
