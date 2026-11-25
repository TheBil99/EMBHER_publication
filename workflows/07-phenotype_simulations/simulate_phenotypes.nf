#!/usr/bin/env nextflow

/*
 * Phenotype simulation pipeline using Simulator_mafld (plain simulations,
 * no GxE / GxG interactions).
 *
 * For each row in the config TSV, this pipeline expands all combinations of
 * h2 and p_causal (comma-separated lists allowed) and runs Simulator_mafld
 * once per combination.
 *
 * Expensive genotype-preparation steps (VCF filtering, BED conversion, MAF/LD
 * file creation) run ONCE per unique (region, min_maf, max_maf, ld_ex, maf_ex)
 * configuration and are shared across all h2/p_causal combinations.
 *
 * Region format (in TSV):
 *   "genome"            — all autosomes chr1–22
 *   "chrN"              — entire chromosome N
 *   "chrN:start-end"    — specific genomic interval
 *
 * The "seed" column in the TSV is passed to Simulator_mafld via -seed for
 *       reproducible runs, and is also encoded in the experiment directory name.
 *
 * Config TSV columns (tab-separated, with header):
 *   region  min_maf  max_maf
 *   h2  p_causal          <- comma-separated lists for grid expansion
 *   ld_ex  maf_ex
 *   n_simulations  seed
 *   causal_min_maf  causal_max_maf   <- optional, default 0 / 0.5. MAF band the
 *                        CAUSAL variants are drawn from, independent of the
 *                        min_maf/max_maf panel filter above: the panel (and hence
 *                        every GRM built from it) keeps all its variants, only the
 *                        variants given a non-zero effect are restricted. Setting
 *                        causal_min_maf=0.01 on an unfiltered panel reproduces the
 *                        real-expression architecture, where the rare MAF bins have
 *                        a TRUE h2 of exactly zero -- the boundary case a benchmark
 *                        needs in order to see a MAF-binned estimator's rare-bin
 *                        component sit on the zero boundary (see workflow 08).
 *   panel             <- optional, 'gtex' (default) or '1000g'. 'gtex' uses the
 *                        GTEx v9 WGS panel (944 samples, one whole-genome VCF --
 *                        the same genotypes workflows 11/09 use for real GTEx
 *                        expression); '1000g' uses the 1000G high-coverage panel
 *                        (3202 samples, one VCF per chromosome).
 *
 * Usage:
 *   nextflow run workflows/07-phenotype_simulations/simulate_phenotypes.nf \
 *     --config_tsv workflows/07-phenotype_simulations/configs/example_grid.tsv \
 *     --out_dir my_simulations
 */

// ── Constants ─────────────────────────────────────────────────────────────── //

def SCRIPTS_DIR  = "${projectDir}/scripts"
def SIMULATOR    = "${launchDir}/src/Simulator/build/Simulator_mafld"
def PLINK        = params.plink
def VCF_DIR      = "${launchDir}/data/raw/1000genomes/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"
def VCF_PREFIX   = "1kGP_high_coverage_Illumina.chr"
def VCF_SUFFIX   = ".filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
// chrX has a .v2 suffix; genome-wide mode uses only autosomes chr1–22.
// GTEx v9 WGS panel (944 donors): one phased whole-genome VCF (already indexed),
// unlike 1000G's one-file-per-chromosome layout -- same file covers every
// chromosome, so the per-chr VCF_DIR/VCF_PREFIX/VCF_SUFFIX pattern above doesn't
// apply to it. Same file workflows 11/09 use for their own genotype branches.
def GTEX_VCF     = "${launchDir}/data/raw/dbgap/gtex/genotypes/phg001796.v1.GTEx_v9_WGS_phased.genotype-calls-vcf.c1/GTEx_Analysis_2021-02-11_v9_WholeGenomeSeq_944Indiv_Analysis_Freeze.SHAPEIT2_phased.vcf.gz"
// bcftools comes from a conda env, not `module load BCFtools`: the module doesn't
// are not available on every node, while conda envs work on any node --
// same reasoning workflows 11/09 already follow for their own bcftools steps.
def BCFTOOLS_ENV = params.env_analysis
def CONDA_SH     = params.conda_sh

// ── Parameters ────────────────────────────────────────────────────────────── //

params.out_dir    = null
params.config_tsv = null

if (!params.out_dir)
    error "Please specify --out_dir"
if (!params.config_tsv)
    error "Please specify --config_tsv"

def OUT_PREFIX = "${launchDir}/results/07-phenotype_simulations/simulations/${params.out_dir}"
def EXP_PREFIX = "${OUT_PREFIX}/experiments"
def GEN_PREFIX = "${OUT_PREFIX}/genotypes"

// ── Helpers ───────────────────────────────────────────────────────────────── //

// Convert a region string to a list of [chr_str, start_or_null, end_or_null].
def parseRegion(String region) {
    if (region == 'genome') {
        return (1..22).collect { n -> [n.toString(), null, null] }
    }
    def m_iv  = (region =~ /^chr(\w+):(\d+)-(\d+)$/)
    def m_chr = (region =~ /^chr(\w+)$/)
    if (m_iv) {
        return [[m_iv[0][1], m_iv[0][2].toLong(), m_iv[0][3].toLong()]]
    } else if (m_chr) {
        return [[m_chr[0][1], null, null]]
    }
    error "Unrecognised region format: '${region}'. Use 'genome', 'chrN', or 'chrN:start-end'."
}

// Filesystem-safe base name shared by all combinations with the same geno config.
// Panel is part of the key: switching 1000g <-> gtex for the same region/maf/ld/maf_ex
// must never collide with (or silently reuse) the other panel's cached genotypes.
def makeBaseName = { cfg ->
    def r = cfg.region.replace(':', '_').replace('-', '_')
    "region_${r}_maf${cfg.min_maf}_${cfg.max_maf}_ld${cfg.ld_ex}_mafex${cfg.maf_ex}_${cfg.panel}".toLowerCase()
}

// Unique directory name per simulation combination. The causal-MAF band is part
// of the name ONLY when it is non-default: two conditions differing just in which
// variants are allowed to be causal share every other field, so without it the
// second would silently reuse the first's published directory. Keeping the
// suffix off the default (0/0.5) leaves every pre-existing experiment directory
// name byte-identical.
def makeExpName = { cfg ->
    def base = "h2_${cfg.h2}_pc_${cfg.p_causal}_rep_${cfg.seed}"
    def causalBand = (cfg.causal_min_maf != 0.0 || cfg.causal_max_maf != 0.5)
        ? "_cmaf${cfg.causal_min_maf}_${cfg.causal_max_maf}" : ""
    "${base}${causalBand}".toLowerCase()
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 1: Filter VCF by AF INFO field and region, then convert to BED.
//            Runs once per base_name × chromosome.
// ──────────────────────────────────────────────────────────────────────────── //

process filter_and_convert_vcf {
    cpus   2
    memory { 8.GB * task.attempt }
    time   { 1.h  * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    tag "${base_name}:chr${chr}"

    input:
    tuple val(base_name), val(cfg), val(chr), val(start), val(end), path(vcf_gz), path(vcf_tbi)

    output:
    tuple val(base_name), val(cfg), val(chr),
          path("chr${chr}_filtered.bed"),
          path("chr${chr}_filtered.bim"),
          path("chr${chr}_filtered.fam"),
          path("chr${chr}_af.txt")

    script:
    // GTEx's one file spans every chromosome, so a whole-chromosome/genome-wide
    // request (start == null) still needs --regions chrN to avoid re-scanning the
    // entire genome per "chromosome" task; 1000G's files are already per-chromosome,
    // so it never needed this.
    def regions_flag = (start != null) ? "--regions chr${chr}:${start}-${end}" :
                        (cfg.panel == 'gtex' ? "--regions chr${chr}" : "")
    // Filter on MAF = min(AF, 1-AF), expressed via INFO/AF (consistent with the
    // AF used downstream in prepare_maf_ld.py). Complements are precomputed here
    // because bcftools -i expressions cannot do arithmetic inline.
    //   MAF >= min_maf  <=>  AF >= min_maf AND AF <= 1-min_maf
    //   MAF <= max_maf  <=>  AF <= max_maf OR  AF >= 1-max_maf
    def min_maf_comp = 1.0 - cfg.min_maf
    def max_maf_comp = 1.0 - cfg.max_maf
    def maf_filter = "(INFO/AF >= ${cfg.min_maf} && INFO/AF <= ${min_maf_comp}) && " +
                     "(INFO/AF <= ${cfg.max_maf} || INFO/AF >= ${max_maf_comp})"
    """
    source ${CONDA_SH}
    conda activate ${BCFTOOLS_ENV}

    # 1. Filter VCF by MAF range (and optionally by coordinate interval)
    bcftools view \\
        ${regions_flag} \\
        -i "${maf_filter}" \\
        -Oz -o chr${chr}_filtered.vcf.gz \\
        ${vcf_gz}

    bcftools index chr${chr}_filtered.vcf.gz

    # 2. Extract variant ID + AF for passing variants
    bcftools query -f '%ID\\t%INFO/AF\\n' chr${chr}_filtered.vcf.gz > chr${chr}_af.txt

    # 3. Convert filtered VCF to PLINK BED
    ${PLINK} \\
        --vcf chr${chr}_filtered.vcf.gz \\
        --double-id \\
        --make-bed \\
        --out chr${chr}_filtered

    echo "chr${chr}: \$(wc -l < chr${chr}_filtered.bim) variants retained"
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 2: Merge per-chromosome PLINK BED files into one.
//            For single-chromosome regions, copies files without calling plink.
// ──────────────────────────────────────────────────────────────────────────── //

process merge_plink {
    cpus   2
    memory { 16.GB * task.attempt }
    time   { 1.h   * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    tag "${base_name}"

    input:
    tuple val(base_name), val(cfg),
          path(beds), path(bims), path(fams), path(af_txts)

    output:
    tuple val(base_name), val(cfg),
          path("merged.bed"), path("merged.bim"), path("merged.fam"),
          path("merged_af.txt")

    publishDir { "${GEN_PREFIX}/${base_name}" }, mode: 'copy',
        pattern: 'merged.{bed,bim,fam}'

    script:
    """
    # Concatenate per-chromosome AF files (BIM order is established in prepare_maf_ld)
    cat ${af_txts} > merged_af.txt

    # Build merge list (one BED prefix per line)
    for bed in ${beds}; do
        echo \${bed%.bed}
    done > merge_list.txt

    n=\$(wc -l < merge_list.txt)

    if [ "\$n" -eq 1 ]; then
        prefix=\$(cat merge_list.txt)
        cp \${prefix}.bed merged.bed
        cp \${prefix}.bim merged.bim
        cp \${prefix}.fam merged.fam
    else
        ${PLINK} --pmerge-list merge_list.txt bfile --make-bed --out merged
    fi

    echo "Merged: \$(wc -l < merged.bim) variants, \$(wc -l < merged.fam) samples"
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 3: Compute per-variant LD scores.
//            If ld_ex == 0, writes a zero-score placeholder (fast path).
//            Otherwise, runs plink --r2 and aggregates pairs.
// ──────────────────────────────────────────────────────────────────────────── //

process compute_ld_scores {
    cpus   4
    memory { 8.GB * task.attempt }
    time   { 1.h  * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    tag "${base_name}"

    input:
    tuple val(base_name), val(cfg),
          path(bed), path(bim), path(fam), path(af_txt)

    output:
    tuple val(base_name), val(cfg),
          path(bed), path(bim), path(fam), path(af_txt),
          path("ldscore.l2.ldscore.gz")

    script:
    def geno_prefix = bed.baseName
    """
    source ${CONDA_SH}
    conda activate ${params.env_analysis}

    if [ "${cfg.ld_ex}" = "0.0" ] || [ "${cfg.ld_ex}" = "0" ]; then
        # ── ld_ex == 0: write zero-score placeholder ─────────────────── #
        awk 'BEGIN{print "CHR\\tSNP\\tBP\\tL2"} {print \$1"\\t"\$2"\\t"\$4"\\t0.0"}' \\
            ${bim} | gzip > ldscore.l2.ldscore.gz
    else
        # ── ld_ex != 0: compute LD scores with plink ──────────────────── #

        ${PLINK} \\
            --bfile ${geno_prefix} \\
            --r2 \\
            --ld-window-kb 500 \\
            --ld-window 999999 \\
            --ld-window-r2 0 \\
            --out ld_pairs \\
            --threads ${task.cpus} \\
            --memory ${task.memory.toMega()}

        conda activate ${params.env_analysis}

        python ${SCRIPTS_DIR}/compute_ld_scores_plink.py \\
            --r2_ld ld_pairs.ld \\
            --bim   ${bim} \\
            --out   ldscore.l2.ldscore.gz
    fi
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 4: Prepare maf_ld.txt (MAF + LD score per variant, in BIM order)
//            and annot.txt (single annotation category, all 1s).
//            MAF comes from the VCF INFO/AF field extracted in Process 1.
// ──────────────────────────────────────────────────────────────────────────── //

process prepare_maf_ld {
    cpus   1
    memory { 4.GB * task.attempt }
    time   { 20.m * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    tag "${base_name}"

    input:
    tuple val(base_name), val(cfg),
          path(bed), path(bim), path(fam), path(af_txt),
          path(ldscore_gz)

    output:
    tuple val(base_name), val(cfg),
          path(bed), path(bim), path(fam),
          path("maf_ld.txt"), path("annot.txt")

    publishDir { "${GEN_PREFIX}/${base_name}" }, mode: 'copy',
        pattern: '{maf_ld.txt,annot.txt}'

    script:
    """
    source ${CONDA_SH}
    conda activate ${params.env_analysis}

    python ${SCRIPTS_DIR}/prepare_maf_ld.py \\
        --bim        ${bim} \\
        --af_files   ${af_txt} \\
        --ldscore    ${ldscore_gz} \\
        --maf_ld_out maf_ld.txt \\
        --annot_out  annot.txt
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process 5: Run Simulator_mafld to generate simulated phenotypes.
//
// param.mafld.txt layout (single data row):
//   p_causal  ld_ex  maf_ex  min_maf  max_maf  total_h2  num_simul
//
// Columns 4/5 (min_maf/max_maf) are the MAF band the CAUSAL variants are drawn
// from, not a panel filter -- the simulator keeps every SNP in the genotype matrix
// and only restricts which ones get a non-zero effect (see make_maf_ld_effect in
// src/Simulator/src/simulator_maf_ld_arch.cpp, which rejects a sampled index when
// maf_ld(i,2) falls outside the band). They are fed from cfg.causal_min_maf /
// cfg.causal_max_maf, whose 0/0.5 default spans the whole spectrum and so
// reproduces the previous hardcoded behaviour; panel-level MAF filtering remains
// upstream in bcftools + plink via the separate min_maf/max_maf config columns.
//
// GUARD: the simulator draws a FIXED number of causals, ncausals =
// p_causal * Nsnp, by rejection-sampling uniform SNP indices until it has that
// many inside the band. Nsnp is the count of ALL panel SNPs, so when the band
// excludes most of the panel (e.g. causal_min_maf=0.01 on an unfiltered panel,
// where ~75% of variants are MAF<0.005) ncausals can exceed the number of
// eligible SNPs -- and the while loop then spins forever with no error. The
// eligibility check below fails the task fast instead of hanging for the job's
// full walltime.
//
// The "seed" field is passed via -seed for reproducible runs.
// ──────────────────────────────────────────────────────────────────────────── //

process run_simulator {
    cpus   1
    memory { 8.GB  * task.attempt }
    time   { 4.h   * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    tag "${exp_name}"

    input:
    tuple val(exp_name), val(cfg),
          path(bed), path(bim), path(fam),
          path(maf_ld), path(annot)

    output:
    tuple val(exp_name), val(cfg), path("simulated_phenotypes/"), path("region.bed")

    publishDir { "${EXP_PREFIX}/${exp_name}" }, mode: 'copy', pattern: 'simulated_phenotypes'
    publishDir { "${OUT_PREFIX}" },             mode: 'copy', pattern: 'region.bed', overwrite: true

    script:
    def geno_prefix = bed.baseName
    """
    module load GCCcore/14.3.0

    printf 'p_causal ld_ex maf_ex min_maf max_maf total_h2 num_simul\\n' > param.mafld.txt
    printf '${cfg.p_causal} ${cfg.ld_ex} ${cfg.maf_ex} ${cfg.causal_min_maf} ${cfg.causal_max_maf} ${cfg.h2} ${cfg.n_simulations}\\n' >> param.mafld.txt

    # See the GUARD note above: ncausals = p_causal * Nsnp must be drawable from
    # the eligible band, or Simulator_mafld's rejection loop never terminates.
    awk -v pc=${cfg.p_causal} -v lo=${cfg.causal_min_maf} -v hi=${cfg.causal_max_maf} '
        NR > 1 { n++; if (\$1 >= lo && \$1 <= hi) elig++ }
        END {
            ncausal = int(pc * n)
            printf "causal band [%s,%s]: %d/%d SNPs eligible, %d causals requested\\n", lo, hi, elig, n, ncausal
            if (ncausal > 0.8 * elig && ncausal <= elig)
                printf "WARNING: %d causals from %d eligible SNPs (%.0f%% saturation) -- the rejection sampler slows sharply near full occupancy.\\n", ncausal, elig, 100*ncausal/elig > "/dev/stderr"
            if (ncausal > elig) {
                printf "ERROR: p_causal=%s over %d panel SNPs needs %d causals but only %d are inside the band -- Simulator_mafld would loop forever. Lower p_causal or widen the band.\\n", pc, n, ncausal, elig > "/dev/stderr"
                exit 1
            }
        }' ${maf_ld}

    mkdir -p simulated_phenotypes

    ${SIMULATOR} \\
        -g         ${geno_prefix} \\
        -simul_par param.mafld.txt \\
        -maf_ld    ${maf_ld} \\
        -annot     ${annot} \\
        -k 10 \\
        -jn 100 \\
        -seed      ${cfg.seed} \\
        -o         simulated_phenotypes/

    echo "Simulation complete: \$(ls simulated_phenotypes/ | wc -l) files written"

    source ${CONDA_SH}
    conda activate ${params.env_analysis}

    python - <<PYEOF
import pandas as pd, glob

# ── pheno.parquet ─────────────────────────────────────────────────────────── #
phen_files = sorted(
    glob.glob('simulated_phenotypes/*.phen'),
    key=lambda f: int(f.split('/')[-1].split('.')[0])
)
n_simul = len(phen_files)
col_names = [f'pheno_{i}' for i in range(n_simul)]

dfs = []
for i, phen_file in enumerate(phen_files):
    df = pd.read_csv(phen_file, sep=' ')[['IID', 'pheno']]
    df = df.rename(columns={'pheno': col_names[i]}).set_index('IID')
    dfs.append(df)
pd.concat(dfs, axis=1).to_parquet('simulated_phenotypes/pheno.parquet')

# ── betas.parquet ─────────────────────────────────────────────────────────── #
bim = pd.read_csv('${bim}', sep='\t', header=None,
                  names=['chrom', 'variant_id', 'cm', 'pos', 'alt', 'ref'])
bim['chrom'] = bim['chrom'].astype(str)

betas = pd.read_csv('simulated_phenotypes/betas_add.txt', sep=r'\s+',
                    header=None, engine='python')
betas.columns = col_names
betas.insert(0, 'alt',   bim['alt'].values)
betas.insert(0, 'ref',   bim['ref'].values)
betas.insert(0, 'pos',   bim['pos'].values)
betas.insert(0, 'chrom', bim['chrom'].values)
betas.to_parquet('simulated_phenotypes/betas.parquet', index=False)
PYEOF

    rm simulated_phenotypes/*.phen simulated_phenotypes/betas_add.txt

    # Write region BED from BIM (min/max bp per chromosome, 0-based half-open)
    awk 'BEGIN{OFS="\\t"} {
        chr="chr"\$1; bp=\$4+0;
        if (!(chr in mn) || bp < mn[chr]) mn[chr] = bp;
        if (!(chr in mx) || bp > mx[chr]) mx[chr] = bp;
    } END {
        for (chr in mn) print chr, mn[chr]-1, mx[chr]
    }' ${bim} | sort -k1,1V -k2,2n > region.bed
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Workflow
// ──────────────────────────────────────────────────────────────────────────── //

workflow {

    // ── 1. Parse config TSV and expand h2 × p_causal grid ────────────────── //
    combinations_ch = Channel
        .fromPath(params.config_tsv)
        .splitCsv(header: true, sep: '\t')
        .flatMap { row ->
            def h2_vals = row.h2.tokenize(',')*.trim()
            def pc_vals = row.p_causal.tokenize(',')*.trim()
            [h2_vals, pc_vals].combinations().collect { h2, pc ->
                // panel is optional, defaults to 'gtex'. Configs predating this
                // default that still want 1000G (deep_heritability_rare.tsv etc.) now
                // set panel=1000g explicitly, so their behaviour doesn't silently change.
                def panel = (row.panel?.trim() ?: 'gtex')
                if (!(panel in ['1000g', 'gtex']))
                    error "Unrecognised panel '${panel}'; use '1000g' or 'gtex'"
                def cfg = [
                    region        : row.region.trim(),
                    min_maf       : row.min_maf.trim().toDouble(),
                    max_maf       : row.max_maf.trim().toDouble(),
                    h2            : h2.toDouble(),
                    p_causal      : pc.toDouble(),
                    ld_ex         : row.ld_ex.trim().toDouble(),
                    maf_ex        : row.maf_ex.trim().toDouble(),
                    n_simulations : row.n_simulations.trim().toInteger(),
                    seed          : row.seed.trim().toInteger(),
                    panel         : panel,
                    // MAF band the causal variants are drawn from -- NOT a panel
                    // filter (that is min_maf/max_maf above). Optional; the 0/0.5
                    // default is the whole spectrum, i.e. the behaviour of every
                    // config written before these columns existed.
                    causal_min_maf: (row.causal_min_maf?.trim() ?: '0').toDouble(),
                    causal_max_maf: (row.causal_max_maf?.trim() ?: '0.5').toDouble(),
                ]
                if (cfg.causal_min_maf >= cfg.causal_max_maf)
                    error "causal_min_maf (${cfg.causal_min_maf}) must be < " +
                          "causal_max_maf (${cfg.causal_max_maf})"
                tuple(makeExpName(cfg), cfg)
            }
        }

    // ── 2. Deduplicate: run expensive shared steps once per base config ──── //
    base_ch = combinations_ch
        .map  { exp_name, cfg -> tuple(makeBaseName(cfg), cfg) }
        .groupTuple()
        .map  { base_name, cfgs -> tuple(base_name, cfgs[0]) }

    // ── 3. Expand chromosomes and run filter_and_convert_vcf ─────────────── //
    vcf_input_ch = base_ch
        .flatMap { base_name, cfg ->
            parseRegion(cfg.region).collect { chr, start, end ->
                // GTEx is one whole-genome VCF (every chromosome shares the same file);
                // 1000G is one file per chromosome. Same downstream process either way --
                // filter_and_convert_vcf's --regions flag does the actual chromosome
                // restriction for GTEx (see its own comment).
                def vcf = cfg.panel == 'gtex' ? file(GTEX_VCF) : file("${VCF_DIR}/${VCF_PREFIX}${chr}${VCF_SUFFIX}")
                def tbi = cfg.panel == 'gtex' ? file("${GTEX_VCF}.tbi") : file("${VCF_DIR}/${VCF_PREFIX}${chr}${VCF_SUFFIX}.tbi")
                tuple(base_name, cfg, chr, start, end, vcf, tbi)
            }
        }

    per_chr_ch = filter_and_convert_vcf(vcf_input_ch)
    // emits: (base_name, cfg, chr, bed, bim, fam, af_txt)

    // ── 4. Group chromosomes per base_name and merge ──────────────────────── //
    merged_ch = per_chr_ch
        .groupTuple(by: 0)
        .map { base_name, cfgs, chrs, beds, bims, fams, af_txts ->
            tuple(base_name, cfgs[0], beds, bims, fams, af_txts)
        }
        | merge_plink
    // emits: (base_name, cfg, bed, bim, fam, af_txt)

    // ── 5. LD scores and MAF/LD file preparation ──────────────────────────── //
    merged_ch
        | compute_ld_scores
        | prepare_maf_ld
    // emits: (base_name, cfg, bed, bim, fam, maf_ld, annot)

    // ── 6. Fan out: attach shared geno outputs to each h2/p_causal combo ─── //
    sim_input_ch = combinations_ch
        .map { exp_name, cfg -> tuple(makeBaseName(cfg), exp_name, cfg) }
        .combine(prepare_maf_ld.out, by: 0)
        // emits: (base_name, exp_name, cfg, base_cfg, bed, bim, fam, maf_ld, annot)
        .map { base_name, exp_name, cfg, _base_cfg, bed, bim, fam, maf_ld, annot ->
            tuple(exp_name, cfg, bed, bim, fam, maf_ld, annot)
        }

    run_simulator(sim_input_ch)
}
