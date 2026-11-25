#!/usr/bin/env nextflow

/*
 * extract_embeddings -- pooled Borzoi embeddings (plus rna_seq gene/exon/intron aggregates) for a gene list
 * on a donor cohort, via embher-capture (embher.embedding, https://github.com/TheBil99/EMBHER). One gene is one
 * forward pass over every donor and haplotype, and one H5 shard: embher-capture's --resume/--seed_h5
 * only tracks a gene as complete once every donor has been written to its file, so gene is both the
 * parallelism unit and the shard unit.
 *
 *   pretrained Borzoi (replicate 0)        results/06-extract_embeddings/<cohort>/borzoi/<cache_key>/
 *   randomly initialised Borzoi (control)  results/06-extract_embeddings/<cohort>/borzoi_random/<cache_key>/
 *                                          (--run_borzoi_random true; same architecture, seeded torch init)
 *
 * Needs: the `embher_gpu` conda env (envs/embher_gpu.yaml: embher[attribution] with torch, borzoi-pytorch, h5py,
 * pysam; it provides the embher-capture command).
 *
 * Genes whose body reaches past Borzoi's centre crop (6144 bins x 32 bp, minus a 1 kb margin around the
 * TSS) are dropped before any job is submitted: embher-capture would drop them anyway, after spending a
 * GPU slot. publish_shard also refuses to publish an H5 without a /genes group.
 *
 * Cache key: results/<cohort>/<model_type>/<cache_key>/chr<N>/ holds one file per gene. The cache_key names
 * the weights, GENCODE release, window type, output aggregation, dtype and (when active) the MAF filter and
 * the donor cohort. Pooling, reference row and stored layers are recorded in each H5's meta and in the
 * gene's manifest row (chr<N>/manifest/<gene>.tsv, read by src/nf/embeddings_manifest.nf). A re-run of a gene
 * replaces both; when the stored configuration matches this run's, the old file seeds the new one
 * (--seed_h5: only donors not yet in the file are computed).
 *
 * Inputs: --gene_meta_tsv (gene_id, gene_symbol, chr_hg38, tss_hg38, start_hg38, end_hg38;
 * scripts/make_gene_meta.py) and --donors_config (YAML list of donor ids; scripts/make_donors_config.py).
 * Both are derived from GTEx/GEUVADIS files by those scripts; neither is distributed with this repository.
 *
 * --min_maf/--max_maf (min_maf < MAF <= max_maf, computed over the donors_config samples) keep only the
 * variants in that band on every haplotype, all other positions stay reference: --max_maf 0.01 gives the
 * rare-variant embeddings of workflow 16.
 *
 * --cohort gtex|geuvadis picks the genotype source: GTEx's single phased WGS VCF (params.gtex_vcf), or the
 * 1000 Genomes 30x phased panel, one VCF per autosome (GEUVADIS donors).
 *
 * Usage (from the project root):
 *   nextflow run workflows/06-extract_embeddings/extract_embeddings.nf -profile slurm \
 *     --gene_meta_tsv paper/genes/gene_meta_hi_q_random500.tsv --donors_config <donors_config.yaml>
 *   ... --run_borzoi false --run_borzoi_random true            # random-init control
 *   ... --max_maf 0.01 --maf_label le0.01                      # rare-variant embeddings
 *   ... --cohort geuvadis --donors_config <geuvadis_donors_config.yaml>
 */

// ── Constants ─────────────────────────────────────────────────────────────── //

// Shell blocks bind only to params and process inputs, so everything a script needs is a params.*.
def DATA = "${launchDir}/data"

params.hg38_fasta        = "${DATA}/raw/GENCODE/human/release_46/GRCh38.primary_assembly.genome.fa.gz"
params.gtex_vcf          = "${DATA}/raw/dbgap/gtex/genotypes/phg001796.v1.GTEx_v9_WGS_phased.genotype-calls-vcf.c1/GTEx_Analysis_2021-02-11_v9_WholeGenomeSeq_944Indiv_Analysis_Freeze.SHAPEIT2_phased.vcf.gz"
params.onekg_vcf_dir     = "${DATA}/raw/1000genomes/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"
params.gencode_parquet   = "${DATA}/raw/GENCODE/human/release_46/gencode.v46.annotation.gtf.parquet"
params.borzoi_weights    = "${DATA}/raw/articles/Hingerl2024/model_weights/borzoi-replicate-0"

params.cohort = 'gtex'   // gtex | geuvadis
if (!(params.cohort in ['gtex', 'geuvadis'])) error "--cohort must be 'gtex' or 'geuvadis', got '${params.cohort}'"

// Per-gene VCF: one run-wide file for gtex, one file per autosome for geuvadis (chrom is gene_meta's bare
// chr_hg38, e.g. '7'; the VCF contigs are chr-prefixed, handled by embher-capture --contig_prefix chr).
def vcfPathFor = { chrom ->
    if (params.cohort == 'gtex') return params.gtex_vcf
    if (!(chrom ==~ /^([1-9]|1[0-9]|2[0-2])$/))
        error "--cohort geuvadis only supports autosomal genes (chr1-chr22); got chrom '${chrom}'"
    "${params.onekg_vcf_dir}/1kGP_high_coverage_Illumina.chr${chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
}

// ── Parameters ────────────────────────────────────────────────────────────── //

params.gene_meta_tsv    = null
params.donors_config    = null
if (!params.gene_meta_tsv || !params.donors_config) error "--gene_meta_tsv and --donors_config are required"

params.run_borzoi        = true
params.run_borzoi_random = false      // untrained control, same genes/donors/pooling/outputs
params.borzoi_random_seed = 0

params.pooling_axis     = 'channel_dim,seq_dim'
params.pooling_func     = 'mean,max'
params.output_type      = 'rna_seq'
params.output_func      = 'sum,mean'
params.output_crop      = 'gene,exon,intron'
params.dtype            = 'bfloat16'
// Cohort-MAF variant filter (min_maf < MAF <= max_maf, null = unbounded). Changes the output directory.
params.min_maf          = null
params.max_maf          = null
params.maf_args         = [params.min_maf != null ? "--min_maf ${params.min_maf}" : null,
                           params.max_maf != null ? "--max_maf ${params.max_maf}" : null].findAll { it }.join(' ')
// Cosmetic stand-in for the numeric thresholds in the directory name only (e.g. 'le0.01'); the thresholds
// actually applied are recorded verbatim in every manifest row.
params.maf_label         = null
// Also forward the plain reference sequence once per gene as an extra 'REFERENCE' row.
params.include_reference = true
// Embedding layers to store: 'all' or 'final' (only those feeding the output head).
params.embedding_layers  = 'all'
if (!(params.embedding_layers in ['all', 'final'])) error "--embedding_layers must be 'all' or 'final', got '${params.embedding_layers}'"
params.include_reference_arg = [params.include_reference ? '--include_reference' : null,
                                params.embedding_layers != 'all' ? "--embedding_layers ${params.embedding_layers}" : null].findAll { it }.join(' ')

// GPU request of the extraction tasks (SLURM); see nextflow.config / conf/site.config
params.extract_time_h    = 2

def RESULTS_DIR = "${launchDir}/results/06-extract_embeddings/${params.cohort}"

// embher-capture always centres a gene's window on its TSS; named in the cache key so an alternative
// windowing would have to pick a new tag.
def WINDOWING_TAG = 'tss_centered'

def slug = { s -> s.replaceAll(/[^A-Za-z0-9]/, '_') }

// Every content-affecting setting folded into one directory name per model type.
def cacheKeyFor = { weightsPath ->
    def gencodeTag = file(params.gencode_parquet).parent.name      // release_46
    def mafActive = (params.min_maf != null || params.max_maf != null)
    def base = slug("${file(weightsPath).name}_${gencodeTag}_${WINDOWING_TAG}" +
         "_out_${params.output_type}_${params.output_func}_${params.output_crop}_${params.dtype}")
    if (!mafActive) return base
    // MAF is computed over the donors_config samples, so the donor list is part of the name
    def cohortTag = "_cohort_${slug(file(params.donors_config).baseName)}"
    def mafValueTag = params.maf_label
        ?: "${params.min_maf != null ? params.min_maf : 'na'}_${params.max_maf != null ? params.max_maf : 'na'}"
    "${base}_maf_${mafValueTag}${cohortTag}"
}
def MODEL_DIR = [
    borzoi       : "${RESULTS_DIR}/borzoi/${cacheKeyFor(params.borzoi_weights)}",
    borzoi_random: "${RESULTS_DIR}/borzoi_random/${cacheKeyFor("random_init_seed${params.borzoi_random_seed}")}",
]

// Borzoi centre-crops to the middle BORZOI_CROP_BP before its head, so a gene reaching further than that
// from its TSS would only be aggregated over the surviving part. `margin` absorbs indel shifts.
def BORZOI_CROP_BP     = 6144 * 32
def BORZOI_CROP_MARGIN = 1000
def fitsBorzoiCrop = { tss, start, end ->
    def half = (BORZOI_CROP_BP.intdiv(2)) - BORZOI_CROP_MARGIN
    (start as long) >= (tss as long) - half && (end as long) <= (tss as long) + half
}

// A results directory holds one file per gene whatever pooling/reference/layers it was made with (they live
// in the gene's manifest row). The file can only seed a new run when its stored configuration equals this
// run's (embher-capture raises on a mismatch), so the seed is dropped otherwise and the gene recomputed.
def RUN_CONFIG = [
    pooling_axis     : params.pooling_axis,
    pooling_func     : params.pooling_func,
    include_reference: params.include_reference ? 'true' : 'false',
    embedding_layers : params.embedding_layers,
]
def seedMatchesConfig = { modelType, gid, sym, chrom, start, end ->
    def f = new File("${MODEL_DIR[modelType]}/chr${chrom}/manifest/${gid}_${slug(sym)}_chr${chrom}_${start}_${end}.tsv")
    if (!f.exists()) return false
    def lines = f.readLines()
    if (lines.size() < 2) return false
    def header = lines[0].split('\t')
    def vals = lines[1].split('\t', -1)
    def have = [:]
    header.eachWithIndex { h, i -> have[h] = i < vals.size() ? vals[i] : '' }
    return RUN_CONFIG.every { k, v -> have[k] == v }
}

// ── Processes ─────────────────────────────────────────────────────────────── //

// One gene's pooled embeddings + rna_seq aggregates, every donor, in one forward pass (GPU).
process extract_gene {
    queue params.gpu_queue
    clusterOptions "--gres=${params.gpu_gres}"
    array 1000
    cpus 1
    memory { 8.GB * task.attempt }
    time { params.extract_time_h.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${model_type}:${gene_id}"

    input:
    tuple val(model_type), val(gene_id), val(gene_symbol), val(chrom), val(start), val(end), val(vcf_file_path), path(gene_meta), path(donors_config), val(seed_h5)

    output:
    tuple val(model_type), val(gene_id), val(gene_symbol), val(chrom), val(start), val(end), path("out.h5")

    script:
    def weights = model_type == 'borzoi_random'
        ? "--weights_path ${params.borzoi_weights} --random_init --seed ${params.borzoi_random_seed}"
        : "--weights_path ${params.borzoi_weights}"
    """
    source ${params.conda_sh}
    conda activate ${params.env_gpu}
    export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

    embher-capture \\
        --config ${donors_config} \\
        --model_type borzoi \\
        ${weights} \\
        --gene_meta_path ${gene_meta} --select_genes ${gene_id} \\
        --vcf_file_path ${vcf_file_path} --hg38_file_path ${params.hg38_fasta} --contig_prefix chr \\
        --gencode_parquet ${params.gencode_parquet} \\
        --pooling_axis ${params.pooling_axis} --pooling_func ${params.pooling_func} \\
        --output_type ${params.output_type} --output_func ${params.output_func} --output_crop ${params.output_crop} \\
        --dtype ${params.dtype} --compression gzip \\
        ${params.maf_args} \\
        ${params.include_reference_arg} \\
        --seed_h5 ${seed_h5} \\
        --output_h5 out.h5
    """
}

// Publish one gene's H5 under a name that encodes its identity and the coordinates it was windowed on, into
// its cache_key directory, and write a one-row manifest fragment named to match (per-gene fragments, not one
// shared TSV, so parallel writers never race). Files are nested under chr<N>/.
process publish_shard {
    array 1000
    cpus 1
    memory { 4.GB * task.attempt }
    time { 20.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 5
    maxForks 30
    tag "${model_type}:${gene_id}"

    input:
    tuple val(model_type), val(gene_id), val(gene_symbol), val(chrom), val(start), val(end), path(h5)

    output:
    path("manifest_row.tsv")

    script:
    def outDir = "${MODEL_DIR[model_type]}/chr${chrom}"
    def name = "${gene_id}_${slug(gene_symbol)}_chr${chrom}_${start}_${end}"
    """
    source ${params.conda_sh}
    conda activate ${params.env_gpu}

    # A gene the model computed nothing for (e.g. a gene body past Borzoi's output window) still exits 0 with
    # an otherwise-empty H5: it must not appear in results/ or the manifest as a "complete" empty file.
    # Everything the manifest records about how the shard was made comes from the H5's own meta, except the
    # run-level labels (donors config name, GENCODE release, windowing).
    python - <<'PY'
import csv, datetime, sys
import h5py
with h5py.File("${h5}", "r") as f:
    if not ("genes" in f and len(f["genes"]) > 0):
        sys.exit(42)
    a = {k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in f["meta"].attrs.items()}
    n_donors = len([s for s in f["samples"][()] if s != b"REFERENCE"])
    has_ref = len(f["samples"]) > 0 and f["samples"][-1] == b"REFERENCE"
weights = a.get("weights_path", "")
random_init = weights.startswith("random_init")
row = {
    "gene_id": "${gene_id}", "gene_symbol": "${gene_symbol}", "model_type": "${model_type}",
    "chrom": "${chrom}", "start": "${start}", "end": "${end}",
    "file_path": "chr${chrom}/${name}.h5",
    "min_maf": a.get("min_maf", ""), "max_maf": a.get("max_maf", ""),
    "maf_cohort": "${(params.min_maf != null || params.max_maf != null) ? file(params.donors_config).baseName : ''}",
    "maf_label": "${params.maf_label ?: ''}",
    "created_at": datetime.datetime.utcnow().isoformat(),
    "pooling_axis": a.get("pooling_axis", ""), "pooling_func": a.get("pooling_func", ""),
    "embedding_layers": a.get("embedding_layers", "") or "all",
    "include_reference": "true" if has_ref else "false",
    "output_type": a.get("output_type", ""), "output_func": a.get("output_func", ""),
    "output_crop": a.get("output_crop", ""), "dtype": a.get("dtype", ""),
    "sequence_length_bp": a.get("sequence_length_bp", ""), "crop_flank_bp": a.get("crop_flank_bp", ""),
    "n_donors": str(n_donors), "donors_config": "${file(params.donors_config).baseName}",
    "weights": "random_init" if random_init else weights,
    "random_seed": weights.rsplit("seed", 1)[-1] if random_init else "",
    "gencode_release": "${file(params.gencode_parquet).parent.name}", "windowing": "${WINDOWING_TAG}",
}
with open("manifest_row.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\\t")
    w.writerow(list(row)); w.writerow(list(row.values()))
PY
    if [ \$? -ne 0 ]; then
        echo "[skip] ${gene_id} (${model_type}): no /genes group in output -- not publishing" >&2
        touch manifest_row.tsv
        exit 0
    fi

    mkdir -p ${outDir} ${outDir}/manifest
    cp ${h5} ${outDir}/${name}.h5
    cp manifest_row.tsv ${outDir}/manifest/${name}.tsv
    """
}

// ──────────────────────────────────────────────────────────────────────────── //
workflow {

    geneMetaFile = Channel.fromPath(params.gene_meta_tsv)
    donorsConfig = Channel.fromPath(params.donors_config)

    // seed_h5 is the gene's already-published shard path, whether or not it exists (embher-capture
    // tolerates a missing seed): a donor-config change then only computes the added donors.
    def seedPathFor = { modelType, chrom, gene_id, gene_symbol, start, end ->
        def path = "${MODEL_DIR[modelType]}/chr${chrom}/${gene_id}_${slug(gene_symbol)}_chr${chrom}_${start}_${end}.h5"
        seedMatchesConfig(modelType, gene_id, gene_symbol, chrom, start, end) ? path : "${path}.no_seed"
    }

    // .toList(), not .collect(): collect() would flatten each row's fields across genes
    genes = geneMetaFile
        .splitCsv(header: true, sep: '\t')
        .map { r -> [r.gene_id, r.gene_symbol, r.chr_hg38, r.start_hg38, r.end_hg38, r.tss_hg38] }
        .toList()
        .map { rows ->
            def byFit = rows.groupBy { r -> fitsBorzoiCrop(r[5], r[3], r[4]) }
            def dropped = byFit[false] ?: []
            if (dropped)
                log.warn "Skipping ${dropped.size()} gene(s) exceeding the ${BORZOI_CROP_BP}bp crop window (never submitted): " +
                    dropped.collect { "${it[0]} (${it[1]})" }.join(', ')
            (byFit[true] ?: []).collect { it[0..4] }
        }
        .flatMap { it }
        .combine(geneMetaFile)
        .combine(donorsConfig)

    def jobsFor = { modelType ->
        genes.map { gene_id, gene_symbol, chrom, start, end, gene_meta, donors_config ->
            tuple(modelType, gene_id, gene_symbol, chrom, start, end, vcfPathFor(chrom), gene_meta, donors_config,
                  seedPathFor(modelType, chrom, gene_id, gene_symbol, start, end))
        }
    }
    jobs = (params.run_borzoi ? jobsFor('borzoi') : Channel.empty())
        .mix(params.run_borzoi_random ? jobsFor('borzoi_random') : Channel.empty())

    publish_shard(extract_gene(jobs))
}
