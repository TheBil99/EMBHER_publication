#!/usr/bin/env nextflow

/*
 * borzoi_attribution -- motif enrichment of Borzoi attributions along the architecture.
 *
 * For each gene of the config, on the REFERENCE window (524,288 bp around the TSS, no population):
 *
 *   1. attribute_layers  input x gradient attribution to the L2 norm of each of the 24 checkpoints
 *                        workflow 06 stores (the heritability-over-layers plots' layers and order)
 *                        and to the whole-blood expression prediction (the last point of the figures).
 *                        Two roles, one GPU task each: `reference` (pretrained weights) and
 *                        `random_init` (same architecture and head, random weights, params.seed), the
 *                        control. The targets are fixed in embher-attribute:
 *                        expression + one embedding_norm_<layer> per layer (params.layers restricts).
 *                        Targets already done with the same settings are skipped (params.force
 *                        recomputes), so a rerun only computes what is missing or was redefined.
 *   2. seqlets           Seqlets of every target's attribution (Decima's find_peaks, tangermeme), called
 *                        once per gene and shared by the annotations below.
 *   3. motifs            FIMO scans of the seqlets against a motif database (as Decima), tested per
 *                        TF / RBP against random windows of the same sequence: enriched motifs per
 *                        layer (the expression target's count alongside), plus seqlet counts. One
 *                        run per database in params.motif_dbs_run (HOCOMOCO v13 TFs over the whole
 *                        window; CISBP-RNA RBPs on the gene's own span and sense strand).
 *   4. motif_plot        One figure per database over all genes: enriched motifs per gene across layers,
 *                        mean and 95% CI over genes, pretrained vs random init, the expression target as
 *                        the last point (plot_motif_layers.py; cisbp_rna also gets a seqlet-count-adjusted version, layers_normalized).
 *   5. constraint_track  Per gene, the window's gnomAD alleles (PASS SNVs of the v4.1.1 genomes) or PhyloP
 *                        scores, extracted once to data/prep/constraint/<track>/<gene>.parquet.
 *   6. constraint        Per gene, positions inside the seqlets of each target and every other position
 *                        (background), counted by sequence context (trinucleotide + CpG): positions with
 *                        and without a gnomAD variant (gnomad; low-coverage positions dropped by
 *                        an AN proxy) or summed PhyloP (phylop). Nothing is excluded for biology.
 *   7. constraint_plot   One figure per track over all genes (plot_constraint_layers.py): log odds ratio of
 *                        unseen : seen positions (gnomad) or PhyloP of the seqlets minus the background
 *                        expectation by context (phylop; --phylop_stat ratio divides instead), pooled over
 *                        genes, CI by bootstrap over genes, the expression target as the last point.
 *                        Tracks in params.constraint_dbs_run; --constraint_only runs steps 5-7 on a
 *                        finished run.
 *   8. pretrained_vs_random  One small figure, run once the figures of params.pvr_dbs are done: per layer,
 *                        pretrained minus random init across databases, normalized by each database's
 *                        maximum and in raw units (plot_pretrained_vs_random.py).
 *
 * Per-gene inputs come from the config TSV (gene_name, gene_meta): `gene_meta` is the gene
 * table the 04b capture used for that gene (TSS and gene body, so the window is identical).
 *
 * Usage (from the project root):
 *   nextflow run workflows/15-borzoi_attribution/borzoi_attribution.nf \
 *     --config_tsv workflows/15-borzoi_attribution/configs/hi_q_random500.tsv \
 *     --run_name hi_q_random500
 *
 * Outputs, written straight into results/15-borzoi_attribution/<run_name>/ (not via publishDir):
 *   genes/<gene>/attribution/{reference,random_init}/<target>/{attribution.h5,summary.json}
 *   genes/<gene>/seqlets/{reference,random_init}/<target>/{seqlets.parquet,seqlets.json}
 *   genes/<gene>/annotation/<db>/motif_layers.parquet  (+ motif_layers.png with --gene_plots)
 *   genes/<gene>/annotation/<db>/{reference,random_init}/<target>/{motif_enrichment.parquet,
 *                                                                  motif_rows.json}   db: hocomoco_v13, ...
 *   genes/<gene>/annotation/<track>/{constraint_counts.parquet,constraint_counts.json}   track: gnomad, phylop241, phylop100
 *   data/prep/constraint/<track>/<gene>.parquet   the window's alleles / scores
 *   summary/<db>/layers.{png,pdf}     value per layer and for expression, one figure per motif database or
 *                                     constraint track (+ layers_per_gene.parquet, layers_summary.parquet)
 *   summary/pretrained_vs_random-{normalized,raw,raw2x2}-<dbs>.{pdf,png}   (+ _summary.parquet)
 */

// ── Constants ─────────────────────────────────────────────────────────────── //

def FASTA        = "${launchDir}/data/raw/GENCODE/human/release_46/GRCh38.primary_assembly.genome.fa.gz"
def GENCODE_PARQUET = "${launchDir}/data/raw/GENCODE/human/release_46/gencode.v46.annotation.gtf.parquet"
def WEIGHTS      = "${launchDir}/data/raw/articles/Hingerl2024/model_weights/borzoi-replicate-0"
// Borzoi + captum/tangermeme/gReLU/decima and embher (embher-attribute) live in the GPU env, the one workflow 06 uses
def MODEL_SETUP  = "source ${params.conda_sh}; conda activate ${params.env_gpu}; " +
                   "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; " +
                   "export PYTHONNOUSERSITE=1 MPLBACKEND=Agg"

// ── Parameters ────────────────────────────────────────────────────────────── //

params.config_tsv  = null
params.run_name    = null
params.tracks      = '7531,7532,7533'              // Whole Blood RNA tracks, as in workflows 06/11
params.layers      = null                          // comma-separated 04b layers (default: all 24)
params.precision   = 'bf16'                        // bf16 = how the cohort values were computed
params.gene_plots  = false                        // per-gene figure of the motif step (off for many genes)
params.genes_per_task  = 10                        // genes per attribute_layers task
params.max_gpu_tasks   = 7                         // concurrent attribute_layers tasks
params.max_motif_tasks = 10                        // concurrent motifs tasks (1 CPU each)
// motif databases (name -> MEME file, motif table, label, extra motif_layers.py arguments); each gets its
// own folder under every gene and its own summary/<name>/layers figure. params.motif_dbs_run picks
// the ones to run. db: MEME file read in place from data/raw; names: motif -> TF / RBP table
// (scripts/prepare_motif_tables.py). cisbp_rna (RNA-binding proteins) is scanned on the gene's own
// span and sense strand only; every motif hit is kept (motif_hits.parquet) for later filtering.
params.motif_dbs     = [
    hocomoco_v13: [db   : "${launchDir}/data/raw/HOCOMOCO/hocomoco13/H13CORE/formatted_motifs/H13CORE_meme_format.meme",
                   names: "${launchDir}/data/prep/motif_tables/hocomoco_v13.tsv",
                   unit : 'TFs',
                   args : '',
                   normalize: false],
    cisbp_rna   : [db   : "${launchDir}/data/raw/CISBP-RNA/pwms_all_motifs.as_dna.meme",
                   names: "${launchDir}/data/prep/motif_tables/cisbp_rna.tsv",
                   unit : 'RBPs',
                   args : '--region gene_body --strand sense --save_hits',
                   normalize: true]       // plus the plot adjusted for the seqlet count (plot_motif_layers.py --normalize)
]
params.motif_dbs_run = 'hocomoco_v13,cisbp_rna'
// constraint tracks ("databases" of the constraint step, one folder under annotation/ and one summary/<name>/
// figure each): gnomad = positions with / without a PASS SNV in the gnomAD genomes sites VCFs, phylop241 = PhyloP bigWig (241-way
// mammals, primary), phylop100 = 100-way vertebrates (robustness). params.constraint_dbs_run picks the ones to
// run ('' = none); params.constraint_only skips attribution, seqlets and motifs (their outputs must exist).
params.constraint_dbs     = [
    gnomad: [kind: 'gnomad', source: "${launchDir}/data/raw/gnomad/v4.1.1"],
    phylop241: [kind: 'phylop', source: "${launchDir}/data/raw/phylop/cactus241way.phyloP.bw"],
    phylop100: [kind: 'phylop', source: "${launchDir}/data/raw/phylop/hg38.phyloP100way.bw"]
]
params.constraint_dbs_run = 'gnomad,phylop241,phylop100'
params.constraint_only    = false
params.pvr_dbs             = 'cisbp_rna,hocomoco_v13,phylop241,gnomad'   // databases of the pretrained-vs-random figure ('' = skip it)
params.pvr_adjusted_dbs   = 'cisbp_rna'          // motif databases of a second pretrained-vs-random set read from the seqlet-count-adjusted tables ('' = skip it)
params.pvr_n_boot         = 10000                  // its bootstrap draws over genes
params.phylop_stat        = 'difference'           // phylop figures: 'difference' (observed - expected) or 'ratio'
params.force       = false                         // recompute targets already done (same settings)
params.checkpoint  = true                          // a 524 kb window needs it on a small slice
params.random_init = true                          // also run the random-weights control
params.seed        = 0                             // random_init weights seed
params.n_null      = 20000                         // random windows in the motif null pool
params.pthresh     = 1e-4                          // FIMO p-value
// GPU request: nextflow.config (--gpu_queue, --gpu_gres); a 12 GB slice is enough with checkpointing

if (!params.config_tsv || !params.run_name) {
    error "--config_tsv and --run_name are required"
}

def OUT_PREFIX = "${launchDir}/results/15-borzoi_attribution/${params.run_name}"
def GPU_OPT    = "--gres=${params.gpu_gres}"

// ── Processes ─────────────────────────────────────────────────────────────── //

// Reference-window attribution to the norm of every 04b checkpoint (and to expression), one GPU task
// per role and batch of params.genes_per_task genes (a gene takes about a minute, so one gene per
// task would be mostly queue time); reference = pretrained weights, random_init = same architecture,
// random weights. Genes already done are skipped, so a failed batch resumes where it stopped.
process attribute_layers {
    beforeScript MODEL_SETUP
    cpus 1                                         // ~0.7 core, 1.8 GB measured (GPU-bound)
    memory { 4.GB * task.attempt }
    time { (20 + 2 * params.genes_per_task).min * task.attempt }   // ~1 min per gene plus model loading; 40 min for 10 genes
    queue params.gpu_queue
    clusterOptions GPU_OPT
    maxForks params.max_gpu_tasks
    errorStrategy 'retry'
    maxRetries 1

    input:
    tuple val(role), val(batch)                    // batch: [[gene_name, gene_meta], ...]

    output:
    tuple val(role), val(batch), emit: layers_done

    script:
    def init     = role == 'random_init' ? "--random_init --seed ${params.seed}" : ""
    def layers   = params.layers ? "--layers ${params.layers}" : ""
    def force    = params.force ? "--force" : ""
    def ckptFlag = params.checkpoint ? "" : "--no_checkpoint"
    batch.collect { gene_name, gene_meta ->
        """
    embher-attribute \\
        --weights_path ${WEIGHTS} \\
        --gene_meta_path ${launchDir}/${gene_meta} \\
        --gene ${gene_name} \\
        --hg38_file_path ${FASTA} \\
        --gencode_parquet ${GENCODE_PARQUET} \\
        --tracks ${params.tracks} \\
        --precision ${params.precision} \\
        ${init} ${layers} ${force} ${ckptFlag} \\
        --out_dir ${OUT_PREFIX}/genes"""
    }.join("\n")
}

// Seqlets of every target of one gene (both roles), once for all the annotations (CPU)
process seqlets {
    array 100
    beforeScript MODEL_SETUP
    queue params.cpu_queue
    cpus 1
    memory { 4.GB * task.attempt }
    time { 20.min * task.attempt }
    errorStrategy { task.attempt <= 1 ? 'retry' : 'ignore' }   // a failure here must not stop the attribution tasks
    maxRetries 1

    input:
    tuple val(gene_name), val(gene_meta)

    output:
    tuple val(gene_name), val(gene_meta), val("${OUT_PREFIX}/genes/${gene_name}/seqlets")   // the folder it fills

    script:
    def force = params.force ? "--force" : ""
    """
    python ${projectDir}/scripts/call_seqlets.py --genes_dir ${OUT_PREFIX}/genes --gene ${gene_name} ${force}
    """
}

// FIMO motif scans of the seqlets and enrichment against random windows, plot per gene (CPU)
process motifs {
    array 100
    // per-task genomepy/diskcache dir: concurrent tasks racing on the shared ~/.cache/genomepy crashed `import genomepy`
    beforeScript MODEL_SETUP + "; export XDG_CACHE_HOME=\$PWD/.cache"
    queue params.cpu_queue
    maxForks params.max_motif_tasks
    cpus 1                                         // single-threaded: ~1 core, 1.4 GB, ~1 h per gene
    memory { 4.GB * Math.min(task.attempt, 2) }
    time { 3.h * Math.min(task.attempt, 2) }
    errorStrategy { task.attempt <= 3 ? 'retry' : 'ignore' }   // idem; the genes that failed are redone by a rerun
    maxRetries 3

    input:
    tuple val(gene_name), val(gene_meta), path(seqlets_dir), val(db_name)

    output:
    tuple val(gene_name), val(db_name)

    script:
    def plots = params.gene_plots ? "" : "--no_plot"
    def force = params.force ? "--force" : ""
    """
    python ${projectDir}/scripts/motif_layers.py \\
        --genes_dir ${OUT_PREFIX}/genes \\
        --gene ${gene_name} --seqlets_dir ${seqlets_dir} \\
        --gene_meta ${launchDir}/${gene_meta} \\
        --motif_db ${params.motif_dbs[db_name].db} --motif_names ${params.motif_dbs[db_name].names} \\
        --db_name ${db_name} --unit ${params.motif_dbs[db_name].unit} ${params.motif_dbs[db_name].args} \\
        --pthresh ${params.pthresh} --n_null ${params.n_null} --seed ${params.seed} ${plots} ${force}
    """
}

// One figure per motif database over all genes: enriched motifs per gene across layers, mean and 95% CI (CPU)
process motif_plot {
    beforeScript MODEL_SETUP
    queue params.cpu_queue
    cpus 1
    memory { 2.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'ignore'

    input:
    tuple val(genes_done), val(db_name)

    output:
    val(db_name), emit: plotted

    script:
    def base = "python ${projectDir}/scripts/plot_motif_layers.py --genes_dir ${OUT_PREFIX}/genes --db_name ${db_name} " +
               "--unit ${params.motif_dbs[db_name].unit} --out_prefix ${OUT_PREFIX}/summary/${db_name}/layers"
    """
    ${base}
    ${params.motif_dbs[db_name].normalize ? base + ' --normalize' : ''}
    """
}

// Extract of a constraint track over one gene's reference window (CPU, I/O only)
process constraint_track {
    array 100
    beforeScript MODEL_SETUP
    queue params.cpu_queue
    cpus 1
    memory { 2.GB * task.attempt }                 // ~0.2 GB measured
    time { 20.min * task.attempt }
    errorStrategy { task.attempt <= 1 ? 'retry' : 'ignore' }
    maxRetries 1

    input:
    tuple val(gene_name), val(db_name)

    output:
    tuple val(gene_name), val(db_name), val("${launchDir}/data/prep/constraint/${db_name}/${gene_name}.parquet")

    script:
    def force = params.force ? "--force" : ""
    """
    python ${projectDir}/scripts/prepare_constraint_track.py \\
        --genes_dir ${OUT_PREFIX}/genes --gene ${gene_name} \\
        --kind ${params.constraint_dbs[db_name].kind} --source ${params.constraint_dbs[db_name].source} \\
        --out_dir ${launchDir}/data/prep/constraint/${db_name} ${force}
    """
}

// Constraint counts of the seqlets of every target of one gene, by sequence context (CPU)
process constraint {
    array 100
    beforeScript MODEL_SETUP
    queue params.cpu_queue
    cpus 1
    memory { 2.GB * task.attempt }                 // ~0.2 GB, 2 s measured
    time { 20.min * task.attempt }
    errorStrategy { task.attempt <= 1 ? 'retry' : 'ignore' }
    maxRetries 1

    input:
    tuple val(gene_name), path(seqlets_dir), val(db_name), path(track)

    output:
    tuple val(gene_name), val(db_name)

    script:
    def force = params.force ? "--force" : ""
    """
    python ${projectDir}/scripts/constraint_layers.py \\
        --genes_dir ${OUT_PREFIX}/genes --gene ${gene_name} --db_name ${db_name} \\
        --kind ${params.constraint_dbs[db_name].kind} --track ${track} ${force}
    """
}

// One constraint figure per track over all genes: value across layers (CPU)
process constraint_plot {
    beforeScript MODEL_SETUP
    queue params.cpu_queue
    cpus 1
    memory { 2.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'ignore'

    input:
    tuple val(genes_done), val(db_name)

    output:
    val(db_name), emit: plotted

    script:
    """
    python ${projectDir}/scripts/plot_constraint_layers.py \\
        --genes_dir ${OUT_PREFIX}/genes --db_name ${db_name} --kind ${params.constraint_dbs[db_name].kind} \\
        --phylop_stat ${params.phylop_stat} --out_prefix ${OUT_PREFIX}/summary/${db_name}/layers
    """
}

// Pretrained minus random init across databases, one small figure (CPU); needs the per-database figures' tables
process pretrained_vs_random {
    beforeScript MODEL_SETUP
    queue params.cpu_queue
    cpus 1
    memory { 2.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'ignore'

    input:
    val(plotted)                                   // the databases plotted in this run (only gates the start)

    script:
    """
    python ${projectDir}/scripts/plot_pretrained_vs_random.py \\
        --summary_dir ${OUT_PREFIX}/summary --dbs ${params.pvr_dbs} --n_boot ${params.pvr_n_boot}
    ${params.pvr_adjusted_dbs ? "python ${projectDir}/scripts/plot_pretrained_vs_random.py --summary_dir ${OUT_PREFIX}/summary --dbs ${params.pvr_dbs} --n_boot ${params.pvr_n_boot} --adjusted_dbs ${params.pvr_adjusted_dbs}" : ""}
    """
}

// ── Workflow ──────────────────────────────────────────────────────────────── //

workflow {
    cfg   = Channel.fromPath(params.config_tsv)
    genes = cfg.splitCsv(header: true, sep: '\t').map { [it.gene_name, it.gene_meta] }
    roles = params.random_init ? Channel.of('reference', 'random_init') : Channel.of('reference')

    // one task per role and batch of genes
    batches = roles.combine(genes.toList().map { gl -> [gl] })              // [role, all genes]
        .flatMap { role, gl -> gl.collate(params.genes_per_task).collect { b -> [role, b] } }

    mplot = Channel.empty()
    if (params.constraint_only) {                  // the attributions, seqlets and motifs are already there
        sdone = genes.map { gene, meta -> [gene, meta, "${OUT_PREFIX}/genes/${gene}/seqlets"] }
    } else {
        ldone = attribute_layers(batches)
        dbs   = Channel.fromList(params.motif_dbs_run.tokenize(','))
        sdone = seqlets(ldone.layers_done
            .flatMap { role, batch -> batch.collect { g -> [g[0], role, g[1]] } }      // [gene, role, gene_meta]
            .groupTuple(size: params.random_init ? 2 : 1).map { gene, roles, metas -> [gene, metas[0]] })
        mdone = motifs(sdone.combine(dbs))
        // one figure per motif database, started as soon as that database's last gene is done (groupKey carries
        // the gene count, so a database does not wait for the others)
        mplot = motif_plot(mdone.combine(genes.count())
            .map { gene, db, n -> [groupKey(db, n), gene] }
            .groupTuple()
            .map { db, gs -> [gs, db] }).plotted
    }

    // constraint (gnomAD / PhyloP): track extract per gene, counts per gene, one figure per track
    cdbs = Channel.fromList(params.constraint_dbs_run ? params.constraint_dbs_run.tokenize(',') : [])
    ctrack = constraint_track(sdone.map { gene, meta, sdir -> gene }.combine(cdbs))
    cdone  = constraint(sdone.map { gene, meta, sdir -> [gene, sdir] }.combine(ctrack, by: 0))   // [gene, seqlets, db, track]
    cplot = constraint_plot(cdone.combine(genes.count())
        .map { gene, db, n -> [groupKey(db, n), gene] }
        .groupTuple()
        .map { db, gs -> [gs, db] }).plotted

    // pretrained vs random across databases: once, after the figures of the databases it uses (those plotted in
    // this run; with --constraint_only the motif tables are already in place)
    if (params.pvr_dbs) {
        pvr_dbs = params.pvr_dbs.tokenize(',')
        pretrained_vs_random(mplot.mix(cplot).filter { it in pvr_dbs }.toList())
    }
}
