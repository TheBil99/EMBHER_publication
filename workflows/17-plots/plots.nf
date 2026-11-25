#!/usr/bin/env nextflow

/*
 * plots -- the paper's figures and supplementary tables. Nothing is recomputed here: every item reads the
 * already-published results/ of other workflows (see paper/figure_map.tsv) and renders one figure (PDF) or
 * table (TSV) into results/17-plots/<item>/.
 *
 *   fig_2 fig_3 fig_4b fig_s1_s4 fig_s5 fig_s6 fig_s7 fig_s8   table_s1 ... table_s6
 *
 * Usage (from the project root, once the workflows of paper/run_all.sh have finished):
 *   nextflow run workflows/17-plots/plots.nf -profile slurm
 *   nextflow run workflows/17-plots/plots.nf -profile slurm --items fig_2,table_s3
 *
 * The result folders read are the run names of paper/run_all.sh; override one with its --*_dir parameter.
 * --embeddings_dir is workflow 06's pretrained-Borzoi cache directory (only table_s2 reads one H5 from it).
 */

def SCRIPTS_DIR  = "${projectDir}/scripts"
def CONDA_SH     = params.conda_sh
def ANALYSIS_ENV = params.env_analysis

params.items = 'all'

// run directories (relative to the project root)
params.genotype_gtex_dir    = 'results/10-genotype_heritability/gtex/hi_q_gtex_genes/h2/he_mac_ld_6'
params.genotype_geuvadis_dir = 'results/10-genotype_heritability/geuvadis/hi_q_geuvadis_genes/h2/he_mac_ld_6'
params.embedding_gtex_dir   = 'results/11-embedding_heritability/gtex/hi_q_random500'
params.embedding_geuvadis_dir = 'results/11-embedding_heritability/geuvadis/hi_q_geuvadis_random500'
params.layers_dir           = 'results/12-heritability_over_layers/hi_q_random500'
params.layers_random_dir    = 'results/12-heritability_over_layers/hi_q_random500_borzoi_random'
params.ancestry_dir         = 'results/13-ancestry_prediction/hi_q_random500'
params.faap20_dir           = 'results/14-ancestry_effect_embeddings/faap20/eur_afr'
params.attribution_dir      = 'results/15-borzoi_attribution/hi_q_random500'
params.rare_dir             = 'results/16-rare_heritability_permutations/gtex/hi_q_gtex_genes_maf_le0.01'
params.sim_genotype_dir     = 'results/08-genotype_heritability_simulation_study/10_random_genes_he'
params.sim_embedding_dir    = 'results/09-embedding_heritability_simulation_study/10_random_genes_he'
params.sim_genotype_reml_dir  = 'results/08-genotype_heritability_simulation_study/10_random_genes_reml'
params.sim_embedding_reml_dir = 'results/09-embedding_heritability_simulation_study/10_random_genes_reml'
params.embeddings_dir       = 'results/06-extract_embeddings/gtex/borzoi/borzoi_replicate_0_release_46_tss_centered_out_rna_seq_sum_mean_gene_exon_intron_bfloat16'
params.gene_map             = 'paper/genes/11_hi_q_random500.tsv'
params.panel_dir            = 'paper/genes'
params.he_genotype_method   = 'he_mac_ld_6_covar'

def S = SCRIPTS_DIR
def SUMMARY15 = "${params.attribution_dir}/summary"
def PQ15      = "${SUMMARY15}/pretrained_vs_random_adjusted-cisbp_rna__hocomoco_v13__phylop241__gnomad_summary.parquet"
def RARE      = "${params.rare_dir}/summary"
def PLOT_SRC  = "--embher-plotting-src ${launchDir}/src --out-dir ."

// item -> script and arguments (paths relative to the project root)
def ITEMS = [
    fig_2: "plot_fig_2.py --embedding-parquet ${params.embedding_gtex_dir}/summary/overall/across_genes.parquet " +
           "--gene-map ${params.gene_map} --genotype-h2-dir ${params.genotype_gtex_dir} " +
           "--prediction-heritability-dir ${params.embedding_gtex_dir}/prediction_heritability " +
           "--pretrained-run-dir ${params.layers_dir} --random-init-run-dir ${params.layers_random_dir} " +
           "--ratio-script ${S}/summarize_embedding_genotype_ratio.py ${PLOT_SRC}",
    fig_3: "plot_fig_3.py --gene FAAP20 --pca-tsv ${params.faap20_dir}/genes/FAAP20/FAAP20_pca_ancestry.tsv " +
           "--ancestry-summary ${params.ancestry_dir}/results/ancestry_prediction_summary.tsv " +
           "--summary-parquet ${PQ15} ${PLOT_SRC}",
    fig_4b: "plot_fig_4b.py --summary-tsv ${RARE}/bootstrap_bars_rare_h2_summary.tsv " +
            "--real-estimates ${RARE}/per_gene_real_estimates.parquet ${PLOT_SRC}",
    fig_s1_s4: "plot_fig_s1_s4.py --genotype-h2-dir ${params.sim_genotype_dir}/h2 " +
               "--embedding-replicates-dir ${params.sim_embedding_dir}/replicates " +
               "--genotype-reml-h2-dir ${params.sim_genotype_reml_dir}/h2 " +
               "--embedding-reml-replicates-dir ${params.sim_embedding_reml_dir}/replicates ${PLOT_SRC}",
    fig_s5: "plot_fig_s5.py --embedding-parquet ${params.embedding_geuvadis_dir}/summary/overall/across_genes.parquet " +
            "--prediction-heritability-dir ${params.embedding_geuvadis_dir}/prediction_heritability " +
            "--genotype-h2-dir ${params.genotype_geuvadis_dir} ${PLOT_SRC}",
    fig_s6: "plot_fig_s6.py --pretrained-heritability-dir ${params.layers_dir}/heritability " +
            "--genotype-h2-dir ${params.genotype_gtex_dir} --genotype-method ${params.he_genotype_method} " +
            "--gene-map ${params.gene_map} ${PLOT_SRC}",
    fig_s7: "plot_fig_s7.py --summary-tsv ${params.ancestry_dir}/results/ancestry_prediction_summary.tsv " +
            "--ratio-script ${S}/summarize_embedding_genotype_ratio.py ${PLOT_SRC}",
    fig_s8: "plot_fig_s8.py --real-estimates ${RARE}/per_gene_real_estimates.parquet " +
            "--null-distribution ${RARE}/null_distribution.parquet --permutation-test ${RARE}/permutation_test.tsv ${PLOT_SRC}",
    table_s1: "make_table_s1.py --genotype-h2-dir ${params.genotype_gtex_dir} --panel ${params.panel_dir}/hi_q_gtex_genes.tsv " +
              "--in-panel ${params.panel_dir}/hi_q_random500.tsv --out-dir .",
    table_s2: "make_table_s2.py --layer-order ${launchDir}/workflows/15-borzoi_attribution/scripts/layer_order.py " +
              "--h5 \$(ls ${launchDir}/${params.embeddings_dir}/chr18/*.h5 | head -1) --out-dir .",
    table_s3: "make_table_s3.py --pretrained-run-dir ${params.layers_dir} --random-init-run-dir ${params.layers_random_dir} " +
              "--gene-map ${params.gene_map} --ratio-script ${S}/summarize_embedding_genotype_ratio.py --out-dir .",
    table_s4: "make_table_s4.py --summary-tsv ${params.ancestry_dir}/results/ancestry_prediction_summary.tsv " +
              "--gene-map ${params.gene_map} --ratio-script ${S}/summarize_embedding_genotype_ratio.py --out-dir .",
    table_s5: "make_table_s5.py --summary-dir ${SUMMARY15} --summary-parquet ${PQ15} --gene-map ${params.gene_map} --out-dir .",
    table_s6: "make_table_s6.py --real-estimates ${RARE}/per_gene_real_estimates.parquet --out-dir .",
]

def selected = params.items == 'all' ? ITEMS.keySet().toList() : params.items.split(',').collect { it.trim() }
selected.each { if (!ITEMS.containsKey(it)) error "unknown item '${it}' (one of ${ITEMS.keySet().join(', ')})" }

process render {
    cpus 1
    memory { 8.GB * task.attempt }
    time { 30.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "${name}"

    publishDir "${launchDir}/results/17-plots/${name}", mode: 'copy'

    input:
    tuple val(name), val(cmd)

    output:
    path("*")

    script:
    // run dirs in `cmd` are relative to the project root: make them absolute, the task runs in its work dir
    def resolved = cmd.replaceAll(/(^| )(results|paper)\//) { all, sp, top -> sp + launchDir + '/' + top + '/' }
    """
    source ${CONDA_SH}
    conda activate ${ANALYSIS_ENV}
    python ${SCRIPTS_DIR}/${resolved}
    """
}

workflow {
    render(Channel.fromList(selected.collect { [it, ITEMS[it]] }))
}
