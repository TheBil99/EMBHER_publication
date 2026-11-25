#!/usr/bin/env nextflow

/*
 * geuvadis_covariates_extraction -- GEUVADIS's covariate matrix for heritability
 * models: 60 PEER factors, genotype PCs and technical covariates (named
 * geuvadis_prep until the latter two were added). PEER factors are computed with the
 * PEER standalone CLI (PMBio/peer, vendored unmodified at src/peer/, built locally
 * with -DBUILD_PEERTOOL=ON -- see src/peer/build/src/peertool, gitignored; built via
 * workflows/05-geuvadis_covariates_extraction/build_peer.sh, a one-off sbatch job,
 * not part of this pipeline).
 *
 * Source data: workflow 02's GEUVADIS TPM cache (results/02-expression_data_preparation/
 * geuvadis/, tissue_matrix GEUVADIS_LCL: GD660 RPKM -> TPM per run, technical replicates
 * averaged, restricted to the 1000G high-coverage donors), looked up through 02's manifest
 * with src/scripts/resolve_expression.py -- run workflow 02 (--cohort geuvadis) first.
 * Deliberately NOT data/raw/GEUVADIS/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz,
 * which is GEUVADIS's own already-PEER-corrected output (k=10 factors already regressed
 * out -- see data/raw/GEUVADIS/GeuvadisRNASeqAnalysisFiles_README.txt, "Library depth &
 * expressed & PEER" normalization level): using that file here would mean computing
 * PEER factors on residuals that already had PEER applied to them.
 *
 * scripts/prepare_geuvadis_peer_input.py filters genes the same way GEUVADIS's own "50FN"
 * convention does (TPM==0 in more than 50% of donors) and log2(TPM+1)-transforms the
 * rest (not part of GEUVADIS's own published pipeline, but standard practice for PEER on
 * RNA-seq, which is heavily right-skewed on the raw scale -- see the script's own
 * docstring).
 *
 * peertool's input format (confirmed against src/peer/src/peerutil.cpp): headerless
 * CSV, rows=samples, columns=genes; -n 60 requests 60 factors. X.csv, its output
 * factor matrix, comes out n_factors (rows) x samples (columns) for this data --
 * confirmed empirically, the opposite of what write_output's code suggests for a
 * non---transpose run -- so scripts/collect_peer_factors.py transposes it back.
 *
 * peertool's binary links against GCCcore/14.3.0's libstdc++ (built with the GCC
 * toolchain brought in by the CMake module); the system
 * libstdc++ is too old, so `module load GCCcore/14.3.0` is required at runtime,
 * not just build time (confirmed: running it unloaded fails with "GLIBCXX_3.4.32
 * not found").
 *
 * Two more covariate sources, alongside the PEER factors above, published as their own
 * files here (combined later by src/scripts/build_geuvadis_covar.py, called by whichever
 * downstream workflow -- 08, 11 -- needs the full GEUVADIS covariate matrix, mirroring
 * how build_gtex_eqtl_covar.py is called by those same workflows for GTEx):
 *   genotype_pcs.tsv         first --n_geno_pcs genotype PCs, from a third-party 1000G
 *                            high-coverage ancestry PCA panel (data/raw/
 *                            GenomiAncestryPanel/; neither 1000G nor IGSR publishes
 *                            one itself, unlike GTEx -- see extract_genotype_pcs.py).
 *   technical_covariates.tsv sex + sequencing-center (GEUVADIS's own major technical
 *                            batch effect, standing in for GTEx's platform/pcr), from
 *                            E-GEUV-1.sdrf.txt -- see extract_technical_covariates.py.
 *
 * Usage:
 *   nextflow run workflows/05-geuvadis_covariates_extraction/geuvadis_covariates_extraction.nf \
 */

// ── Constants ─────────────────────────────────────────────────────────────── //

def DATA           = "${launchDir}/data"
def EXPR_DIR        = "${launchDir}/results/02-expression_data_preparation/geuvadis"
def EXPR_TISSUE     = "GEUVADIS_LCL"
def SDRF            = "${DATA}/raw/GEUVADIS/E-GEUV-1.sdrf.txt"
def REFERENCE_SCORES = "${DATA}/raw/GenomiAncestryPanel/panel-1000g-30x-grch38-1.0.0/reference_scores.tsv"
def ANALYSIS_ENV    = params.env_analysis
def CONDA_SH        = params.conda_sh
def PEER_BIN        = params.peer_bin
def SCRIPTS_DIR     = "${projectDir}/scripts"
def SHARED_DIR      = "${projectDir}/../../src/scripts"

// ── Parameters ────────────────────────────────────────────────────────────── //

params.peer_bin = "${launchDir}/src/peer/build/src/peertool"   // PEER command-line tool (build_peer.sh)
params.n_factors     = 60
params.n_iter        = 1000
params.zero_frac_max = 0.5
params.pseudocount   = 1.0
params.n_geno_pcs    = 5

def OUT_PREFIX = "${launchDir}/results/05-geuvadis_covariates_extraction"


// ──────────────────────────────────────────────────────────────────────────── //
// Process 1: filter + log2-transform workflow 02's GEUVADIS TPM matrix.
// ──────────────────────────────────────────────────────────────────────────── //

process prepare_expression {
    cpus 1
    memory { 4.GB * task.attempt }
    time { 20.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}/input" }, mode: 'copy'

    output:
    path("peer_input.csv"), emit: expr
    path("samples.txt"), emit: samples
    path("genes.txt")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    printf 'tissue\\n${EXPR_TISSUE}\\n' > tissue.tsv
    python ${SHARED_DIR}/resolve_expression.py \\
        --config tissue.tsv \\
        --manifest-dir ${EXPR_DIR}/manifest \\
        --data-dir ${EXPR_DIR} \\
        --product tissue_matrix \\
        --out resolved.tsv
    MATRIX=\$(awk -F'\\t' 'NR==2 {print \$3}' resolved.tsv)

    python ${SCRIPTS_DIR}/prepare_geuvadis_peer_input.py \\
        --tissue-matrix \$MATRIX \\
        --zero-frac-max ${params.zero_frac_max} \\
        --pseudocount ${params.pseudocount} \\
        --out-expr peer_input.csv \\
        --out-samples samples.txt \\
        --out-genes genes.txt
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 2: run peertool, -n params.n_factors hidden factors.
// ──────────────────────────────────────────────────────────────────────────── //

process run_peer {
    cpus 1
    memory { 8.GB * task.attempt }
    time { 12.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}/peer_out" }, mode: 'copy'

    input:
    path(expr)

    output:
    path("peer_out/X.csv"), emit: x
    path("peer_out/Alpha.csv")
    path("peer_out/residuals.csv")

    script:
    """
    module load GCCcore/14.3.0

    ${PEER_BIN} -f ${expr} -n ${params.n_factors} -i ${params.n_iter} -o peer_out
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 3: attach sample IDs to peertool's headerless X.csv.
// ──────────────────────────────────────────────────────────────────────────── //

process collect_factors {
    cpus 1
    memory { 2.GB * task.attempt }
    time { 10.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}" }, mode: 'copy'

    input:
    path(x_csv)
    path(samples)

    output:
    path("peer_factors.parquet")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${SCRIPTS_DIR}/collect_peer_factors.py \\
        --x-csv ${x_csv} \\
        --samples ${samples} \\
        --n-factors ${params.n_factors} \\
        --out peer_factors.parquet
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 4: genotype PCs for GEUVADIS donors from the third-party 1000G ancestry PCA
//            panel (see module docstring and extract_genotype_pcs.py).
// ──────────────────────────────────────────────────────────────────────────── //

process extract_genotype_pcs {
    cpus 1
    memory { 2.GB * task.attempt }
    time { 10.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}" }, mode: 'copy'

    input:
    path(samples)

    output:
    path("genotype_pcs.tsv")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${SCRIPTS_DIR}/extract_genotype_pcs.py \\
        --reference-scores ${REFERENCE_SCORES} \\
        --samples ${samples} \\
        --n-pcs ${params.n_geno_pcs} \\
        --out genotype_pcs.tsv
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 5: sex + sequencing-center covariates from E-GEUV-1.sdrf.txt (see module
//            docstring and extract_technical_covariates.py).
// ──────────────────────────────────────────────────────────────────────────── //

process extract_technical_covariates {
    cpus 1
    memory { 2.GB * task.attempt }
    time { 10.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}" }, mode: 'copy'

    input:
    path(samples)

    output:
    path("technical_covariates.tsv")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}

    python ${SCRIPTS_DIR}/extract_technical_covariates.py \\
        --sdrf ${SDRF} \\
        --samples ${samples} \\
        --out technical_covariates.tsv
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
workflow {

    prepped = prepare_expression()
    peer_out = run_peer(prepped.expr)
    collect_factors(peer_out.x, prepped.samples)

    extract_genotype_pcs(prepped.samples)
    extract_technical_covariates(prepped.samples)
}
