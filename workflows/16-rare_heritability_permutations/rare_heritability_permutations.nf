#!/usr/bin/env nextflow

/*
 * rare_heritability_permutations -- rare-variant heritability from genotype or from the embedding, with
 * a phenotype-permutation null for their difference. Haseman-Elston regression (embher.heritability.he), always
 * WITHOUT the jackknife (no SE columns, no switch to turn it on).
 *
 * Per gene, on the cohort's phenotyped donors (workflow 02 expression, GTEx PEER/genotype-PC covariates
 * as fixed effects), the genotype is the 6-bin mac_ld_6 partition used by workflow 10's default
 * (--binning mac_ld_6; src/scripts/build_cohort_mac_maf4_bins.py, then split_bins_by_ld.py --bins 3,4):
 *   1 MAC <= mac_max | 2 MAC > mac_max & MAF <= 0.01 | 3/4 (0.01, 0.1] low/high LD | 5/6 (0.1, 0.5] low/high LD
 *
 *   genotype arm   all six bins;                    rare h2 = h2_k1 + h2_k2 (the two lowest bins)
 *   embedding arm  bins 1-2 replaced by the rare-variant embedding kernel (workflow 06 MAF <= 0.01
 *                  embeddings, haplotypes summed and PC-whitened as in workflow 11); the four common LD
 *                  bins stay separate;              rare h2 = the embedding's own component
 *
 * Both arms are fit on the same donors and the same phenotype columns: the real phenotype and
 * --n_perm donor-label permutations of it (the same permutation for both arms of a gene; kernels and
 * covariates fixed). All columns of a gene are solved together (he_fit), so the permutations cost a few
 * matrix products each, not a refit. summarize then compares the observed mean/median of
 * (embedding - genotype) rare h2 across genes with the same statistic under the permutations.
 *
 * Outputs (results/16-rare_heritability_permutations/<cohort>/<out_dir>/):
 *   per_gene/<experiment>_rare_h2_permutations.parquet   one row per phenotype column (perm 0 = real)
 *   summary/permutation_test.tsv, permutation_test.pdf, null_distribution.parquet,
 *   summary/per_gene_real_estimates.parquet
 *   summary/bootstrap_bars_rare_h2.pdf, _summary.tsv   bootstrap CIs of the mean / median rare h2 per arm
 *   embeddings_metadata.tsv   (the combined embeddings are not copied into the results: they already exist in workflow 06/11)
 *
 * Genes are identified by their unversioned Ensembl gene id: the per-gene files, every table and plot are keyed by
 * ENSG00000... (the config's own `experiment` column is ignored; the symbol is kept in the `gene_name` column).
 *
 * Launch from the project root. Per-gene work is batched (--batch_size genes per SLURM task).
 */

include { loadGenotypeManifest; lookupGenoBfile } from '../../src/nf/genotype_manifest.nf'
include { loadEmbeddingsManifest } from '../../src/nf/embeddings_manifest.nf'

// ── Constants ─────────────────────────────────────────────────────────────── //

def SCRIPTS_DIR  = "${projectDir}/scripts"
def SHARED_DIR   = "${projectDir}/../../src/scripts"
def GTEX_COVAR_DIR = "${projectDir}/../../data/raw/GTEx/v10/GTEx_Analysis_v10_eQTL_covariates"
def GEUVADIS_COVAR_DIR = "${launchDir}/results/05-geuvadis_covariates_extraction"
def slug = { s -> s.replaceAll(/[^A-Za-z0-9]/, '_') }
def ANALYSIS_ENV = params.env_analysis
def CONDA_SH     = params.conda_sh
def PLINK        = params.plink

// ── Parameters ────────────────────────────────────────────────────────────── //

params.out_dir        = null
params.config_tsv     = null     // gene_id, region, tissue (+ experiment / gene_name: symbol, only a label), one tissue
params.cohort         = 'gtex'
params.embeddings_dir = null     // workflow 06 results dir with the rare-variant (MAF <= 0.01) embeddings
params.transform      = 'quantile_int'     // quantile_int | log2_plus_1
params.n_perm         = 1000
params.seed           = 0
params.mac_max        = 5
params.window_size    = 524288
params.min_maf        = 0.0
params.max_maf        = null
params.vcf_filter     = null
params.batch_size     = 25
params.n_boot         = 1000     // bootstraps for the bar plots' confidence intervals
params.pred_tracks    = '7531,7532,7533'   // combine_haplotypes.py needs it; the prediction is not used
// Top-PC fraction of the embedding's whitening (pca_embeddings.py --pc-frac); default 0.7 (workflow 11 uses 0.9)
params.pc_frac        = 0.7
// Any non-empty value is written into the fit tasks' script, so a resumed run never reuses fits computed
// without it (the haplotype and expression steps do not depend on it and stay cached).
params.fit_tag        = ''
def PC_FRAC = params.pc_frac

if (!params.out_dir)        error "Please specify --out_dir"
if (!params.config_tsv)     error "Please specify --config_tsv"
if (!params.embeddings_dir) error "Please specify --embeddings_dir"
if (!(params.cohort in ['gtex', 'geuvadis'])) error "--cohort must be 'gtex' or 'geuvadis', got '${params.cohort}'"

def OUT_PREFIX = "${launchDir}/results/16-rare_heritability_permutations/${params.cohort}/${params.out_dir}"

def EMBEDDINGS_MANIFEST = loadEmbeddingsManifest(params.embeddings_dir, [pooling_axes: ['seq_dim'], pooling_funcs: ['mean']])
def mafInfoFrom = { manifest ->
    def row = manifest.values().find { true }
    [min_maf: row?.min_maf ?: '', max_maf: row?.max_maf ?: '',
     maf_cohort: row?.maf_cohort ?: '', maf_label: row?.maf_label ?: '']
}
def EMBEDDINGS_MAF_INFO = mafInfoFrom(EMBEDDINGS_MANIFEST)

def EXPRESSION_DATA_DIR     = "${launchDir}/results/02-expression_data_preparation/${params.cohort}"
def EXPRESSION_MANIFEST_DIR = "${EXPRESSION_DATA_DIR}/manifest"
def GENOTYPE_DATA_DIR     = "${launchDir}/results/04-genotype_extraction/${params.cohort}"
def GENOTYPE_MANIFEST_DIR = "${GENOTYPE_DATA_DIR}/manifest"
def GENOTYPE_MANIFEST     = loadGenotypeManifest(GENOTYPE_MANIFEST_DIR)
def GENO_BAND = [min: params.min_maf as double, max: params.max_maf]

// ──────────────────────────────────────────────────────────────────────────── //
// Expression, covariates, embedding metadata: as in workflow 11
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

process combine_haplotypes {
    array 1000        // pack tasks of this process into one SLURM array
    cpus 1
    memory { 8.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "batch_${task.index}"

    input:
    // BATCHED: one task combines params.batch_size genes, each (experiment, gene_id_slug, gene_id_h5, h5).
    // The H5s are read by path, not staged (nodes share the filesystem).
    val(batch)
    // Staged explicitly (see embedding_heritability's build_grm_script for why) so an
    // edit to combine_haplotypes.py invalidates this process's Nextflow cache.
    path(combine_script)

    output:
    path("*_embedding.parquet"), emit: emb

    script:
    def cmds = batch.collect { g ->
        def (experiment, geneSlug, geneH5, h5) = g
        "python ${combine_script} --h5 ${h5} --gene-id ${geneH5} --tracks ${params.pred_tracks} " +
        "--out-embedding ${experiment}_embedding.parquet " +
        "--out-prediction ${experiment}_${geneSlug}_prediction.parquet.unused"
    }.join('\n')
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    ${cmds}
    """
}

// ──────────────────────────────────────────────────────────────────────────── //
// Per gene: mac_ld_6 assignment, whitened embedding kernel, then the two arms and the permutations
// (scripts/fit_rare_h2_permutations.py). Batched: params.batch_size genes per task, run one after
// another. Genotype bfiles are read by path (not staged, nodes share the filesystem).
// ──────────────────────────────────────────────────────────────────────────── //

process rare_h2_permutations {
    array 1000        // pack tasks of this process into one SLURM array
    cpus 1
    memory { 4.GB * task.attempt }
    time { 6.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "batch_${task.index}"

    publishDir { "${OUT_PREFIX}/per_gene" }, mode: 'copy', pattern: "*_rare_h2_permutations.parquet"
    publishDir { "${OUT_PREFIX}/skipped" }, mode: 'copy', pattern: "*.skipped"

    input:
    tuple val(genos), path(phenos), path(embs)
    // staged explicitly so an edit to any of these invalidates the cache under -resume
    path(bin_assignment_script)
    path(ld_split_script)
    path(pca_script)
    path(grm_script)
    path(fit_script)
    path(covar)

    output:
    path("*_rare_h2_permutations.parquet"), emit: h2, optional: true
    // a gene that fails any step (e.g. a MAC/MAF bin with too few variants to split by LD) is skipped, not fatal:
    // its marker, with the last lines of the error, goes to skipped/
    path("*.skipped"), emit: skipped, optional: true

    script:
    def phenoList = [phenos].flatten()
    def embList = [embs].flatten()
    def cmds = genos.indices.collect { i ->
        def (experiment, bfile) = genos[i]
        def pheno = phenoList[i]
        def emb = embList[i]
        """
    ( python ${bin_assignment_script} --bfile ${bfile} --pheno ${pheno} --plink-bin ${PLINK} \\
          --mac-max ${params.mac_max} --out ${experiment}_mac4bin.assign.txt &&
      python ${ld_split_script} --bfile ${bfile} --pheno ${pheno} \\
          --bin-assignment ${experiment}_mac4bin.assign.txt --bins 3,4 --out ${experiment}_mac_ld_6.assign.txt &&
      python ${pca_script} --embeddings ${emb} --pheno ${pheno} --pc-frac ${PC_FRAC} --out ${experiment}_pcs.parquet &&
      python ${grm_script} --embeddings ${experiment}_pcs.parquet --pheno ${pheno} --out ${experiment} &&
      python ${fit_script} --bfile ${bfile} --bin-assignment ${experiment}_mac_ld_6.assign.txt \\
          --grm-npz ${experiment}.npz --pheno ${pheno} --covariate-file ${covar} \\
          --n-perm ${params.n_perm} --seed ${params.seed} --experiment '${experiment}' \\
          --out ${experiment}_rare_h2_permutations.parquet
    ) 2> ${experiment}.step.err || { echo "[warn] ${experiment} skipped: \$(tail -n 1 ${experiment}.step.err)" >&2; tail -n 3 ${experiment}.step.err > ${experiment}.skipped; rm -f ${experiment}_rare_h2_permutations.parquet; }
    cat ${experiment}.step.err >&2
"""
    }.join('\n')
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    ${params.fit_tag ? '# fit_tag: ' + params.fit_tag + ' (pc_frac ' + PC_FRAC + ')' : ''}
${cmds}
    """
}

// ──────────────────────────────────────────────────────────────────────────── //
// Permutation test across genes, tables and plots
// ──────────────────────────────────────────────────────────────────────────── //

process summarize {
    cpus 1
    memory { 8.GB * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}/summary" }, mode: 'copy'

    input:
    path(per_gene)
    path(config_tsv)
    path(script)

    output:
    path("permutation_test.tsv"), emit: tests
    path("permutation_test.pdf")
    path("null_distribution.parquet")
    path("per_gene_real_estimates.parquet"), emit: real

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    python ${script} --per-gene ${per_gene} --config-tsv ${config_tsv} --seed ${params.seed} --out-dir .
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Genes are identified by their unversioned Ensembl gene id (experiment = ENSG00000...), whatever the config's
// own `experiment` column holds (usually the symbol): this writes the config the pipeline uses, with `experiment`
// set to the ENSG id and the symbol kept in `gene_name`.
// ──────────────────────────────────────────────────────────────────────────── //

process ensg_config {
    cpus 1
    memory { 1.GB * task.attempt }
    time { 5.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    input:
    path(config_tsv)

    output:
    path("config_ensg.tsv")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    python - <<'PY'
import csv
tab = chr(9)
rows = list(csv.DictReader(open("${config_tsv}"), delimiter=tab))
out, seen = [], set()
for r in rows:
    ens = r["gene_id"].split(".")[0]
    if ens in seen:
        raise SystemExit("duplicate gene_id in the config: " + ens)
    seen.add(ens)
    out.append({"experiment": ens, "gene_id": r["gene_id"], "gene_name": r.get("gene_name") or r.get("experiment") or ens,
                "region": r["region"], "tissue": r["tissue"]})
w = csv.DictWriter(open("config_ensg.tsv", "w"), fieldnames=list(out[0]), delimiter=tab, lineterminator=chr(10))
w.writeheader()
w.writerows(out)
print(len(out), "genes")
PY
    """
}

// ──────────────────────────────────────────────────────────────────────────── //
// Bootstrap bar plots (mean / median rare h2, rare genotype vs embedding) with the permutation p
// ──────────────────────────────────────────────────────────────────────────── //

process bootstrap_bars {
    cpus 1
    memory { 2.GB * task.attempt }
    time { 20.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${OUT_PREFIX}/summary" }, mode: 'copy'

    input:
    path(real_estimates)
    path(permutation_test)
    path(script)

    output:
    path("bootstrap_bars_rare_h2.pdf")
    path("bootstrap_bars_rare_h2_summary.tsv")

    script:
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    python ${script} --real-estimates ${real_estimates} --permutation-test ${permutation_test} \\
        --n-boot ${params.n_boot} --seed ${params.seed} --out-prefix bootstrap_bars_rare_h2
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
            if (!row.gene_id || !row.region || !row.tissue)
                error "config row missing a required field (gene_id, region, tissue): ${row}"
            tissues << row.tissue
        }
    }
    if (tissues.size() != 1)
        error "This workflow assumes a single tissue for the shared covariate file; got: ${tissues}"
    covar = build_peer_covar(Channel.value(tissues.first()))

    record_embeddings_metadata(
        Channel.value(params.embeddings_dir),
        Channel.value(EMBEDDINGS_MAF_INFO.min_maf),
        Channel.value(EMBEDDINGS_MAF_INFO.max_maf),
        Channel.value(EMBEDDINGS_MAF_INFO.maf_cohort),
        Channel.value(EMBEDDINGS_MAF_INFO.maf_label),
        Channel.value("embeddings_metadata.tsv"))

    // per-gene phenotype from workflow 02's manifest (same join as workflow 11)
    cfg_ensg = ensg_config(Channel.value(file(params.config_tsv)))
    resolved_pheno = resolve_expression(cfg_ensg)
    pheno_files = resolved_pheno.phenos.flatten().map { f -> tuple(f.name - '_pheno.parquet', f) }
    pheno = resolved_pheno.table
        .splitCsv(header: true, sep: '\t')
        .map { r -> tuple(new File(r.pheno_path).name - '_pheno.parquet', r.experiment.trim()) }
        .combine(pheno_files, by: 0)
        .map { tag, e, pf -> tuple(e, pf) }

    // embeddings (workflow 06 manifest) and genotypes (workflow 04 manifest), resolved up front so a
    // missing gene fails the run before any job is submitted
    def missing = []
    def embRows = []
    def resolvedGeno = []
    new File(params.config_tsv).withReader { r ->
        def header = r.readLine().split('\t')
        r.eachLine { line ->
            if (!line.trim()) return
            def vals = line.split('\t', -1)
            def row = [:]
            header.eachWithIndex { h, i -> row[h] = i < vals.size() ? vals[i].trim() : '' }
            def geneIdH5 = row.gene_id.split('\\.')[0]
            def expId = geneIdH5                       // genes are identified by the unversioned ENSG id
            def manRow = EMBEDDINGS_MANIFEST[geneIdH5]
            if (!manRow) missing << "${expId}: ${row.gene_id} not in workflow 06's manifest (${params.embeddings_dir})"
            else embRows << tuple(expId, slug(row.gene_id), geneIdH5, file("${params.embeddings_dir}/${manRow.file_path}"))
            def bfRow = lookupGenoBfile(GENOTYPE_MANIFEST, row.region, params.window_size, GENO_BAND, params.vcf_filter)
            if (!bfRow) missing << "${expId}: workflow 04 region_bfile for ${row.region}"
            else resolvedGeno << tuple(expId, "${GENOTYPE_DATA_DIR}/${bfRow.bed_path}" - '.bed')
        }
    }
    if (missing) error "Cannot resolve every gene:\n" + missing.join('\n')

    combined = combine_haplotypes(
        Channel.fromList(embRows).collate(params.batch_size),
        Channel.value(file("${SHARED_DIR}/combine_haplotypes.py")))
    embeddings = combined.emb.flatten().map { f -> tuple(f.name - '_embedding.parquet', f) }

    fits = rare_h2_permutations(
        Channel.fromList(resolvedGeno)
            .combine(pheno, by: 0)
            .combine(embeddings, by: 0)
            .collate(params.batch_size)
            .map { b -> tuple(b.collect { [it[0], it[1]] }, b.collect { it[2] }, b.collect { it[3] }) },
        Channel.value(file("${SHARED_DIR}/build_cohort_mac_maf4_bins.py")),
        Channel.value(file("${SHARED_DIR}/split_bins_by_ld.py")),
        Channel.value(file("${SHARED_DIR}/pca_embeddings.py")),
        Channel.value(file("${SHARED_DIR}/build_grm.py")),
        Channel.value(file("${SCRIPTS_DIR}/fit_rare_h2_permutations.py")),
        covar)

    summ = summarize(fits.h2.flatten().collect(),
                     Channel.value(file(params.config_tsv)),
                     Channel.value(file("${SCRIPTS_DIR}/summarize_permutation_test.py")))

    bootstrap_bars(summ.real, summ.tests,
                   Channel.value(file("${SCRIPTS_DIR}/plot_bootstrap_bars.py")))
}
