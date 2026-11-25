#!/usr/bin/env nextflow

/*
 * genotype_heritability_simulation_study -- bias of MAC/MAF-binned genotype heritability on
 * SIMULATED phenotypes with a KNOWN true h2, so each estimate can be scored directly against
 * the truth it was simulated with.
 *
 * Estimator (--estimator, one per run, each into its own --out_dir):
 *   exact (default)  src/scripts/exact_reml_heritability.py -- exact-trace multi-component
 *                    AI-REML, unconstrained, three starts; converged / at_boundary flags as
 *                    documented there.
 *   exact_nonneg     exact_reml_heritability.py --nonneg -- every bin's variance component
 *                    >= 0; also writes lrt/lrt_p.
 *   he               src/scripts/he_heritability.py -- Haseman-Elston regression, on
 *                    the same GRMs and bins, with its delete-one jackknife SEs unless --no_se.
 *
 * --no_se: skip SE / p-value work. HE runs with --jackknife none (h2_se = NaN); for every
 * estimator the summary drops the pages built on SEs (H0 QQ-plot, CI coverage, "confidently
 * too high"). REML's SEs come free with the fit, so its estimation is unchanged.
 *
 * Arms (method <estimator prefix>_<arm>: exact_reml_, exact_reml_nonneg_ or he_):
 *   mac_maf_bins_4     MAC <= mac_max | MAC > mac_max & MAF <= 0.01 | (0.01, 0.1] | (0.1, 0.5]
 *                      (src/scripts/build_cohort_mac_maf4_bins.py; same arm as workflow 10's).
 *   mac_maf_ld_bins_8  mac_maf_bins_4 with every bin split at its median region-wide LD score
 *                      (src/scripts/split_bins_by_ld.py): bins 2b-1 / 2b = low / high LD of bin b.
 *   mac_maf_ld_bins_6  mac_maf_bins_4 with only the two common bins split by LD: 1 = MAC <= 5,
 *                      2 = MAC > 5 & MAF <= 0.01, 3/4 = (0.01, 0.1] low/high, 5/6 = (0.1, 0.5]
 *                      low/high. The low-LD half of the MAC <= 5 bin is close to the identity
 *                      and roughly doubles HE's null SD in mac_maf_ld_bins_8. Same bins as
 *                      workflow 10's mac_ld_6.
 * All assignments are computed per condition on its phenotyped cohort, and each assignment
 * file is shared by the estimator and its matching true-per-bin h2, so the two cannot
 * partition the variants differently.
 *
 * No --restrict-grm-to-cohort: every simulated phenotype covers all panel donors, so the
 * cohort and the panel are the same set.
 *
 * Upstream: workflow 07 (simulate_phenotypes.nf) simulates phenotypes with Simulator_mafld on
 * one shared region genotype panel per gene, grid-expanding h2 x p_causal into one
 * pheno.parquet per condition (100 replicate columns). Every arm runs at the condition's own
 * --true-h2, so bias = h2 - true_h2 falls straight out.
 *
 * Per-bin truth: scripts/compute_true_bin_h2.py computes each replicate's TRUE per-bin h2 from
 * the simulator's betas (betas.parquet; same causal SNPs every replicate, fresh effect sizes),
 * and summarize compares each arm's h2_k1, h2_k2, ... against it.
 *
 * A config may mix causal-variant MAF bands (workflow 07's causal_min_maf/causal_max_maf,
 * a _cmaf<lo>_<hi> suffix in the experiment name). They are NOT pooled: summarize emits one
 * summary_<band>.pdf per band, and its tables gain a causal_band column.
 *
 * Config TSV columns (tab-separated, with header):
 *   experiment  true_h2  p_causal  bfile_prefix  pheno_parquet
 *     bfile_prefix : PLINK prefix of the condition's genotype panel (workflow 07's merged bfile)
 *     pheno_parquet: workflow 07's simulated_phenotypes/pheno.parquet (IID index, pheno_*)
 *     betas.parquet (same dir as pheno_parquet) and maf_ld.txt (same dir as bfile_prefix)
 *     are derived from workflow 07's publishDir layout, not their own columns.
 *
 * Usage (from the project root):
 *   nextflow run workflows/08-genotype_heritability_simulation_study/genotype_heritability_simulation_study.nf \
 *     --config_tsv workflows/08-genotype_heritability_simulation_study/configs/ten_random_genes_maf_bins_calibration.tsv \
 *     --estimator he --no_se --out_dir <out_dir>
 */

// ── Constants ─────────────────────────────────────────────────────────────── //

def SCRIPTS_DIR  = "${projectDir}/scripts"
def SHARED_DIR   = "${projectDir}/../../src/scripts"
def PLINK        = params.plink
def ANALYSIS_ENV = params.env_analysis
def CONDA_SH     = params.conda_sh

// ── Parameters ────────────────────────────────────────────────────────────── //

params.out_dir    = null
params.config_tsv = null
params.estimator  = 'exact'   // exact / exact_nonneg / he -- see module docstring
params.no_se      = false     // see module docstring
params.mac_max    = 5         // bin 1 of mac_maf_bins_4: MAC <= mac_max

if (!params.out_dir)    error "Please specify --out_dir"
if (!params.config_tsv) error "Please specify --config_tsv"
if (!(params.estimator in ['exact', 'exact_nonneg', 'he'])) error "--estimator must be exact, exact_nonneg or he, got ${params.estimator}"

def OUT_PREFIX = "${launchDir}/results/08-genotype_heritability_simulation_study/${params.out_dir}"

// Arm label -> the 'strategy' tag written into its true-bin parquets and matched by
// _method_style.method_strategy(); plot_bin_bias joins each arm to its truth on (strategy, bin
// count), which tells all three arms apart. LD arms: split (the input bins split_bins_by_ld.py
// splits, '' = all) and suffix (its assignment file name).
def ARM_4   = [arm: 'mac_maf_bins_4',    strategy: 'mac_cohort']
def ARM_LD8 = [arm: 'mac_maf_ld_bins_8', strategy: 'mac_ld_cohort', split: '',    suffix: 'ld']
def ARM_LD6 = [arm: 'mac_maf_ld_bins_6', strategy: 'mac_ld_cohort', split: '3,4', suffix: 'ld6']
def LD_ARMS = [ARM_LD8, ARM_LD6]
def LD_STRATEGY = LD_ARMS.collectEntries { m -> [(m.arm): m.strategy] }


// ──────────────────────────────────────────────────────────────────────────── //
// Process 1: 4-bin MAC/MAF assignment for one condition, on its phenotyped cohort.
//            Built once and fed to the LD split, the estimator and the truth, so all
//            of them partition the variants identically.
// ──────────────────────────────────────────────────────────────────────────── //

process build_mac_bin_assignment {
    array 1000
    cpus 1
    memory { 8.GB * task.attempt }
    time { 10.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${experiment}:mac_assign_4"

    input:
    tuple val(experiment), path(bed), path(bim), path(fam), path(pheno)

    output:
    tuple val(experiment), path("${experiment}_mac4bin.assign.txt"), emit: assign

    script:
    def bfile = bed.baseName
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${SHARED_DIR}/build_cohort_mac_maf4_bins.py \\
        --bfile ${bfile} \\
        --pheno ${pheno} \\
        --plink-bin ${PLINK} \\
        --mac-max ${params.mac_max} \\
        --out ${experiment}_mac4bin.assign.txt
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 2: an LD arm's assignment -- the chosen 4-bin MAC/MAF bins split at their
//            median LD score on the same cohort (src/scripts/split_bins_by_ld.py).
// ──────────────────────────────────────────────────────────────────────────── //

process split_bins_by_ld {
    array 1000
    cpus 1
    memory { 2.GB * task.attempt }
    time { 15.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${experiment}:${arm}"

    input:
    tuple val(experiment), val(arm), val(split_bins), val(suffix), path(assign),
          path(bed), path(bim), path(fam), path(pheno)

    output:
    tuple val(experiment), val(arm), path("${experiment}_mac4bin_${suffix}.assign.txt"), emit: assign

    script:
    def bfile = bed.baseName
    def bins = split_bins ? "--bins ${split_bins}" : ""
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}

    python ${SHARED_DIR}/split_bins_by_ld.py \\
        --bfile ${bfile} \\
        --pheno ${pheno} \\
        --bin-assignment ${assign} \\
        ${bins} \\
        --out ${experiment}_mac4bin_${suffix}.assign.txt
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 3: heritability of one (condition, arm), estimator per --estimator.
// ──────────────────────────────────────────────────────────────────────────── //

process run_mac_bins_exact {
    array 1000
    cpus 1
    memory { 2.GB * task.attempt }
    // 100 replicates per task at 944 donors, one thread: typically ~5-10 min, slowest
    // observed ~2 h (boundary-heavy binned fits). Retry twice on a longer limit, then
    // ignore rather than let one stuck task block summarize.
    time { 3.h * task.attempt }
    errorStrategy { task.attempt <= 2 ? 'retry' : 'ignore' }
    maxRetries 2
    tag "${experiment}:exact_reml_${arm}"

    publishDir { "${OUT_PREFIX}/h2" }, mode: 'copy'

    input:
    tuple val(experiment), val(true_h2), val(arm), path(assign),
          path(bed), path(bim), path(fam), path(pheno)

    output:
    path("${experiment}_exact_reml_${arm}.parquet"), emit: h2

    script:
    def bfile = bed.baseName
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}

    python ${SHARED_DIR}/exact_reml_heritability.py \\
        --bfile ${bfile} \\
        --pheno ${pheno} \\
        --bin-assignment ${assign} \\
        --experiment '${experiment}' \\
        --true-h2 ${true_h2} \\
        --method "exact_reml_${arm}" \\
        --out "${experiment}_exact_reml_${arm}.parquet"
    """
}


process run_mac_bins_exact_nonneg {
    array 1000
    cpus 1
    memory { 2.GB * task.attempt }
    time { 3.h * task.attempt }
    errorStrategy { task.attempt <= 2 ? 'retry' : 'ignore' }
    maxRetries 2
    tag "${experiment}:exact_reml_nonneg_${arm}"

    publishDir { "${OUT_PREFIX}/h2" }, mode: 'copy'

    input:
    tuple val(experiment), val(true_h2), val(arm), path(assign),
          path(bed), path(bim), path(fam), path(pheno)

    output:
    path("${experiment}_exact_reml_nonneg_${arm}.parquet"), emit: h2

    script:
    def bfile = bed.baseName
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}

    python ${SHARED_DIR}/exact_reml_heritability.py \\
        --bfile ${bfile} \\
        --pheno ${pheno} \\
        --bin-assignment ${assign} \\
        --nonneg \\
        --experiment '${experiment}' \\
        --true-h2 ${true_h2} \\
        --method "exact_reml_nonneg_${arm}" \\
        --out "${experiment}_exact_reml_nonneg_${arm}.parquet"
    """
}


process run_mac_bins_he {
    array 1000
    cpus 1
    // the delete-one jackknife holds per-donor copies of the projected kernels
    memory { (params.no_se ? 2.GB : 4.GB) * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${experiment}:he_${arm}"

    publishDir { "${OUT_PREFIX}/h2" }, mode: 'copy'

    input:
    tuple val(experiment), val(true_h2), val(arm), path(assign),
          path(bed), path(bim), path(fam), path(pheno)

    output:
    path("${experiment}_he_${arm}.parquet"), emit: h2

    script:
    def bfile = bed.baseName
    def jk = params.no_se ? "--jackknife none" : ""
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}

    python ${SHARED_DIR}/he_heritability.py \\
        --bfile ${bfile} \\
        --pheno ${pheno} \\
        --bin-assignment ${assign} \\
        ${jk} \\
        --experiment '${experiment}' \\
        --true-h2 ${true_h2} \\
        --method "he_${arm}" \\
        --out "${experiment}_he_${arm}.parquet"
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 4: true per-bin h2 of one (condition, arm), from the simulator's betas and the
//            arm's own assignment (see compute_true_bin_h2.py), tagged with the arm's
//            strategy.
// ──────────────────────────────────────────────────────────────────────────── //

process run_true_bin_h2_mac {
    array 1000
    cpus 1
    memory { 8.GB * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${experiment}:true_${arm}"

    publishDir { "${OUT_PREFIX}/true_bin_h2" }, mode: 'copy'

    input:
    tuple val(experiment), val(true_h2), val(arm), val(strategy), path(assign),
          path(bed), path(bim), path(fam), path(betas), path(maf_ld)

    output:
    path("${experiment}_true_${arm}.parquet"), emit: h2

    script:
    def bfile = bed.baseName
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${SCRIPTS_DIR}/compute_true_bin_h2.py \\
        --bfile ${bfile} \\
        --betas ${betas} \\
        --maf-ld ${maf_ld} \\
        --bin-assignment ${assign} \\
        --strategy ${strategy} \\
        --true-h2 ${true_h2} \\
        --experiment '${experiment}' \\
        --plink-bin ${PLINK} \\
        --out "${experiment}_true_${arm}.parquet"
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 5: one combined summary -- per-gene calibration/bias/error-dist/(SE pages unless
//            --no_se)/convergence (summarize_gene_calibration.py) plus the per-bin bias page
//            (plot_bin_bias.py) in one PDF per causal band, plus their tables.
// ──────────────────────────────────────────────────────────────────────────── //

process summarize {
    cpus 1
    memory { 8.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}/summary" }, mode: 'copy'

    input:
    path(h2_parquets)
    path(true_bin_parquets)
    path(driver)
    // imported by plot_summary.py as sibling modules, so staged alongside it
    path(gene_calibration_module)
    path(bin_bias_module)
    path(style_module)

    output:
    path("summary_*.pdf")
    path("bias_summary.parquet")
    path("bin_bias_summary.parquet")
    path("convergence_summary.parquet")

    script:
    def no_se = params.no_se ? "--no-se" : ""
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${driver} \\
        --h2-parquets ${h2_parquets} \\
        --true-bin-parquets ${true_bin_parquets} \\
        --out-pdf summary.pdf \\
        --split-causal-band \\
        ${no_se} \\
        --out-bias-table bias_summary.parquet \\
        --out-bin-bias-table bin_bias_summary.parquet \\
        --out-convergence-table convergence_summary.parquet
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
workflow {

    // keyed by experiment: (experiment, true_h2, bed, bim, fam, pheno, betas, maf_ld)
    rows = Channel
        .fromPath(params.config_tsv)
        .splitCsv(header: true, sep: '\t')
        .map { r ->
            if (!r.experiment || !r.true_h2 || !r.p_causal || !r.bfile_prefix || !r.pheno_parquet)
                error "config row missing a required field: ${r}"
            def bfilePrefix = r.bfile_prefix.trim()
            def phenoParquet = r.pheno_parquet.trim()
            def betasParquet = phenoParquet.replaceAll(/pheno\.parquet$/, 'betas.parquet')
            def mafLd = new File(bfilePrefix).getParent() + "/maf_ld.txt"
            tuple(r.experiment.trim(), r.true_h2.trim(),
                  file("${bfilePrefix}.bed"), file("${bfilePrefix}.bim"), file("${bfilePrefix}.fam"),
                  file(phenoParquet), file(betasParquet), file(mafLd))
        }

    assign4 = build_mac_bin_assignment(
        rows.map { e, th2, bed, bim, fam, pheno, betas, mafLd -> tuple(e, bed, bim, fam, pheno) }).assign

    assign_ld = split_bins_by_ld(
        assign4.combine(rows, by: 0)
               .combine(Channel.fromList(LD_ARMS))
               .map { e, a, th2, bed, bim, fam, pheno, betas, mafLd, m ->
                      tuple(e, m.arm, m.split, m.suffix, a, bed, bim, fam, pheno) }).assign

    // (experiment, arm, strategy, assignment) for every arm, joined back onto the rows so the
    // estimator and its truth read the same assignment file
    arms = assign4.map { e, a -> tuple(e, ARM_4.arm, ARM_4.strategy, a) }
        .mix(assign_ld.map { e, arm, a -> tuple(e, arm, LD_STRATEGY[arm], a) })
        .combine(rows, by: 0)

    est_rows = arms.map { e, arm, strat, a, th2, bed, bim, fam, pheno, betas, mafLd ->
                          tuple(e, th2, arm, a, bed, bim, fam, pheno) }
    def est = params.estimator
    h2 = est == 'he' ? run_mac_bins_he(est_rows).h2
       : est == 'exact_nonneg' ? run_mac_bins_exact_nonneg(est_rows).h2
       : run_mac_bins_exact(est_rows).h2

    true_bin_h2 = run_true_bin_h2_mac(
        arms.map { e, arm, strat, a, th2, bed, bim, fam, pheno, betas, mafLd ->
                   tuple(e, th2, arm, strat, a, bed, bim, fam, betas, mafLd) }).h2

    summarize(
        h2.collect(),
        true_bin_h2.collect(),
        Channel.value(file("${SCRIPTS_DIR}/plot_summary.py")),
        Channel.value(file("${SCRIPTS_DIR}/summarize_gene_calibration.py")),
        Channel.value(file("${SCRIPTS_DIR}/plot_bin_bias.py")),
        Channel.value(file("${SCRIPTS_DIR}/_method_style.py")))
}
