#!/usr/bin/env nextflow

/*
 * intermediate_files -- compact, documented files from finished runs of other workflows, for sharing (Zenodo).
 *
 * A bundle is one named recipe (source, packer, naming, manifest). First bundle, `seqlet_beds`: the seqlets of
 * workflow 15's attributions (24 layers + expression, pretrained and random init) as one bgzipped, tabix-indexed
 * BED per model and layer across genes (50 files), in hg38.
 *
 *   1. pack_gene   per gene: its 50 seqlet tables as BED rows (window coordinates -> hg38)
 *   2. merge_bed   per model x layer: genes merged, sorted, bgzip + tabix
 *   3. verify      row counts, windows, order, and a sample of intervals against the FASTA
 *   4. docs        manifest.tsv, layers.tsv, README.md
 *   5. zip_bundle  YYYYMMDD-intermediate_files.zip (md5sums.txt inside, <zip>.md5 beside it), tested after writing
 *
 * Usage (from the project root):
 *   nextflow run workflows/99-intermediate_files/intermediate_files.nf \
 *     --config_tsv workflows/99-intermediate_files/configs/hi_q_random500.tsv --run_name hi_q_random500
 *
 * Outputs, in results/99-intermediate_files/<run_name>/: seqlet_beds/*.bed.gz(.tbi), manifest.tsv, layers.tsv,
 * README.md, <date>-intermediate_files.zip(.md5). _packed/ (per-gene intermediates) is removed at the end.
 */

// ── Constants ─────────────────────────────────────────────────────────────── //

def CONDA_SH     = params.conda_sh
def ANALYSIS_ENV = params.env_analysis      // python (pandas, h5py, pyarrow) + bgzip, tabix
def FASTA        = "${launchDir}/data/raw/GENCODE/human/release_46/GRCh38.primary_assembly.genome.fa.gz"

// ── Parameters ────────────────────────────────────────────────────────────── //

params.config_tsv  = null        // workflow 15 config of the genes (gene_name column)
params.run_name    = null        // folder under results/99-intermediate_files/
params.source_run  = 'hi_q_random500'    // workflow 15 run read
params.bundles     = 'seqlet_beds'
params.date        = new Date().format('yyyyMMdd')
params.cpu_queue   = 'cpu'
params.max_tasks   = 20

if (!params.config_tsv || !params.run_name) {
    error "--config_tsv and --run_name are required"
}
if (params.bundles != 'seqlet_beds') {
    error "only the seqlet_beds bundle exists so far"
}

def GENES_DIR  = "${launchDir}/results/15-borzoi_attribution/${params.source_run}/genes"
def OUT_PREFIX = "${launchDir}/results/99-intermediate_files/${params.run_name}"
def PACKED     = "${OUT_PREFIX}/_packed"
def PY         = "source ${CONDA_SH}; conda activate ${ANALYSIS_ENV}; export PYTHONNOUSERSITE=1 PYTHONPATH=${projectDir}/scripts; "

// ── Processes ─────────────────────────────────────────────────────────────── //

process pack_gene {
    array 100
    queue params.cpu_queue
    cpus 1
    memory { 1.GB * task.attempt }
    time { 5.min * task.attempt }
    maxForks params.max_tasks
    errorStrategy { task.attempt <= 1 ? 'retry' : 'terminate' }
    maxRetries 1

    input:
    val gene_name

    output:
    val gene_name

    script:
    """
    ${PY}
    python ${projectDir}/scripts/seqlets_to_bed.py --genes_dir ${GENES_DIR} --gene ${gene_name} --out_dir ${PACKED}
    """
}

process merge_bed {
    queue params.cpu_queue
    cpus 1
    memory { 3.GB * task.attempt }
    time { 10.min * task.attempt }
    maxForks params.max_tasks
    errorStrategy { task.attempt <= 1 ? 'retry' : 'terminate' }
    maxRetries 1

    input:
    tuple val(model), val(layer_index), path(genes_file)

    output:
    val "${model}_${layer_index}"

    script:
    """
    ${PY}
    python ${projectDir}/scripts/merge_bed.py --packed_dir ${PACKED} --genes ${genes_file} --model ${model} \\
        --layer_index ${layer_index} --out_dir ${OUT_PREFIX}/seqlet_beds
    for f in ${OUT_PREFIX}/seqlet_beds/${model}_layer-\$(printf %02d ${layer_index})_*.bed; do
        bgzip -f "\$f"
        tabix -f -p bed "\$f.gz"
    done
    """
}

process verify {
    queue params.cpu_queue
    cpus 1
    memory 4.GB
    time 30.min

    input:
    val merged
    path genes_file

    output:
    val true

    script:
    """
    ${PY}
    python ${projectDir}/scripts/verify_seqlet_beds.py --bed_dir ${OUT_PREFIX}/seqlet_beds --packed_dir ${PACKED} \\
        --genes ${genes_file} --genes_dir ${GENES_DIR} --fasta ${FASTA} --bgzip bgzip
    """
}

process docs {
    queue params.cpu_queue
    cpus 1
    memory 2.GB
    time 10.min

    input:
    val ok
    path genes_file

    output:
    val true

    script:
    """
    ${PY}
    python ${projectDir}/scripts/write_bundle_docs.py --packed_dir ${PACKED} --genes ${genes_file} \\
        --genes_dir ${GENES_DIR} --source_run ${params.source_run} --out_dir ${OUT_PREFIX}
    """
}

process zip_bundle {
    queue params.cpu_queue
    cpus 1
    memory 2.GB
    time 20.min

    input:
    val ok

    output:
    val true

    script:
    def zipname = "${params.date}-intermediate_files.zip"
    """
    set -euo pipefail
    cd ${OUT_PREFIX}
    rm -rf intermediate_files ${zipname} ${zipname}.md5 _check
    mkdir intermediate_files
    cp -r seqlet_beds manifest.tsv layers.tsv README.md intermediate_files/
    (cd intermediate_files && find . -type f ! -name md5sums.txt | sort | xargs md5sum | sed 's# \\./# #' > md5sums.txt)
    # the .bed.gz are already compressed: store, do not recompress
    zip -q -0 -r ${zipname} intermediate_files
    md5sum ${zipname} > ${zipname}.md5
    unzip -tq ${zipname}
    mkdir _check && unzip -q ${zipname} -d _check && (cd _check/intermediate_files && md5sum -c --quiet md5sums.txt)
    rm -rf _check intermediate_files _packed
    ls -l ${zipname}
    """
}

// ── Workflow ──────────────────────────────────────────────────────────────── //

workflow {
    genes = Channel.fromPath(params.config_tsv).splitCsv(header: true, sep: '\t').map { it.gene_name }
    packed = pack_gene(genes)
    genes_file = packed.collectFile(name: 'genes.txt', newLine: true, sort: true)
    jobs = Channel.of('pretrained', 'random_init').combine(Channel.of(1..25))
        .combine(genes_file).map { m, i, f -> tuple(m, i, f) }
    merged = merge_bed(jobs).collect()
    ok = verify(merged, genes_file.first())
    d = docs(ok, genes_file.first())
    zip_bundle(d)
}
