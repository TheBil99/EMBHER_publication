#!/usr/bin/env nextflow

/*
 * genotype_extraction -- the single producer of region PLINK bfiles used across this
 * project's genotype-GRM branches.
 *
 * Each region is extracted once from the GTEx phased WGS VCF (bcftools region+MAF-band
 * filter | plink2 --make-bed) and the bfile is cached in a manifest
 * (src/nf/genotype_manifest.nf: loadGenotypeManifest/lookupGenoBfile, mirroring
 * workflow 06's manifest pattern); every consumer looks a region up there, failing fast
 * if a required entry is missing.
 *
 * Deliberately NOT centralized: any downstream kinship/GRM built from that bfile. Which
 * kinship a heritability method needs -- single whole-region GRM, several MAF-binned
 * GRMs jointly fit -- is specific to that method's own
 * partitioning choice, not a property of the region itself, so caching one shared
 * kinship would either force every consumer onto the same partitioning or multiply into
 * as many cached kinships as there are (partitioning x power) combinations, for a
 * build step that's already cheap (milliseconds of compute per gene/band --
 * see the region_bfile process comment for the equivalent extraction timing). Each
 * consumer builds whatever kinship/GRM its own estimator needs, on demand, from the
 * bfile this workflow resolves for it.
 *
 * TISSUE-AGNOSTIC BY CONSTRUCTION, like workflow 14's own genotype branch was: a
 * region's genotypes depend only on (region, window_size, MAF band, vcf_filter), never
 * on gene identity, tissue or phenotype, so they're computed once over the whole
 * 944-donor WGS panel and reused by any gene at that locus.
 *
 * One product type:
 *   region_bfile  PLINK bed/bim/fam for one (region, window, band), plus a maf.txt
 *                 (plink2 --freq derived MAF file, .bim order) for any consumer that
 *                 needs a --maf-ld file. Key = region + window_size +
 *                 MAF band + vcf_filter.
 *
 * Config TSV: reuses whatever config workflow 11/08/11 already has (experiment gene_id
 * gene_name region [tissue]) unchanged -- only the 'region' column is read, deduplicated
 * before extraction (every gene sharing a region/window/band combo reuses one
 * extraction, same as those workflows' own prior per-region caching).
 *
 * --cohort (gtex | geuvadis) picks which genotype source this run draws from, and
 * results land in a cohort-named subdirectory (results/04-genotype_extraction/
 * {gtex,geuvadis}/...) -- two independent manifests, never sharing cache keys (the
 * manifest row schema itself has no cohort column, so sharing one tree would let a
 * GTEx and a GEUVADIS bfile for the same region/window/band silently collide).
 *
 * The gtex branch is exactly the original pipeline (region_bfile_batch, one VCF for
 * every chromosome). The geuvadis branch (geuvadis_region_bfile_batch) differs in
 * three ways:
 *   1. One VCF per chromosome (the 1000G high-coverage phased release is split that
 *      way), not one VCF for everything -- the per-item VCF path is built from the
 *      region's own chrom instead of a single fixed constant.
 *   2. Restricted to GEUVADIS donors explicitly (bcftools --samples-file): unlike
 *      GTEx's VCF, which already IS the relevant donor panel, the 1000G VCF holds all
 *      3202 global 1000G samples. geuvadis_donor_list derives GEUVADIS's own 462
 *      individuals from GD660's own column headers, intersected with one
 *      chromosome's actual sample panel (identical across all 22 autosome files in
 *      this release, so computed once, not per-region) -- the same ~449-donor set
 *      workflow 02's geuvadis branch and workflow 05 both already land on
 *      independently, from the same 1000G source.
 *   3. No MAF-band (INFO/AF) filtering: the 1000G VCF's own INFO/AF reflects all 3202
 *      global samples, not GEUVADIS's ~449-donor subset -- using it here would band
 *      variants by the wrong population's frequencies. --cohort geuvadis therefore
 *      requires --maf_bands to be the no-filter default ('0.0'); anything else is a
 *      hard error. Per-variant MAF/MAC assignment for GEUVADIS is workflow 10's own
 *      concern (recomputed on its own cohort, after sample restriction), not this
 *      workflow's -- geuvadis_region_bfile_batch still writes maf.txt via plink2
 *      --freq on the extracted (already donor-restricted) bfile, which is
 *      cohort-correct regardless.
 *
 * Usage:
 *   nextflow run workflows/04-genotype_extraction/genotype_extraction.nf \
 *     -profile slurm \
 *     --config_tsv workflows/11-embedding_heritability/configs/<config>.tsv \
 *     --maf_bands 0.0,0.01,0.0-0.01
 *
 *   nextflow run workflows/04-genotype_extraction/genotype_extraction.nf \
 *     -profile slurm \
 *     --cohort geuvadis \
 *     --config_tsv workflows/05-geuvadis_covariates_extraction/configs/all_genes.tsv
 */

// ── Constants ─────────────────────────────────────────────────────────────── //

def DATA          = "${launchDir}/data"
def GTEX_VCF      = "${DATA}/raw/dbgap/gtex/genotypes/phg001796.v1.GTEx_v9_WGS_phased.genotype-calls-vcf.c1/GTEx_Analysis_2021-02-11_v9_WholeGenomeSeq_944Indiv_Analysis_Freeze.SHAPEIT2_phased.vcf.gz"
def ONEKG_VCF_DIR = "${DATA}/raw/1000genomes/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"
def GEUVADIS_RPKM = "${DATA}/raw/GEUVADIS/GD660.GeneQuantRPKM.txt.gz"
def PLINK         = params.plink
def BCFTOOLS_ENV  = params.env_analysis
def CONDA_SH      = params.conda_sh

// ── Parameters ────────────────────────────────────────────────────────────── //

params.config_tsv  = null
params.window_size = 524288
// Same comma-separated floor-("X")-or-capped-band-("A-B") syntax as workflows 12/14's
// own params.maf_bands -- a plain floor "0.0" means no filter.
params.maf_bands   = '0.0'
// Kept only for cache-key parity with the Borzoi embedding manifest's key shape (see
// src/nf/borzoi_manifest.nf) -- already a no-op against the actual bcftools call in
// every one of the three copies this workflow replaces, so this carries that behavior
// forward unchanged rather than fixing it here.
params.vcf_filter  = null
// Items (region x MAF band) per SLURM task -- extraction itself is short (~8s/region
// on a busy node), so one-task-per-item flooded the scheduler with very-short-lived tasks at
// any real gene count (1000+ genes x several bands). Batched into one task per N
// items instead, a plain sequential bash loop over N bcftools/plink2 calls, same
// pattern workflows 10/09 already use for their own per-gene fits.
params.batch_size = 50
params.cohort     = 'gtex'   // gtex | geuvadis

if (!params.config_tsv) error "Please specify --config_tsv"
if (!(params.cohort in ['gtex', 'geuvadis'])) error "--cohort must be 'gtex' or 'geuvadis', got '${params.cohort}'"

def DATA_ROOT     = "${launchDir}/results/04-genotype_extraction/${params.cohort}"
def MANIFEST_DIR  = "${DATA_ROOT}/manifest"

def slug = { s -> s.replaceAll(/[^A-Za-z0-9]/, '_') }

// A "band" is [min, max-or-null]; maxMaf==null means floor-only. Identical semantics to
// workflows 12/14's own parseBand/bandTag (see those files) -- kept here, not shared,
// since only this workflow still needs them for cache-directory naming; consumers no
// longer build a genotype cache-directory name at all, only a manifest lookup.
def parseBand = { s ->
    if (s.contains('-')) {
        def parts = s.split('-').collect { it as double }
        return [min: parts[0], max: parts[1]]
    }
    return [min: s as double, max: null]
}
def bandLabel = { band -> band.max != null ? "${band.min}to${band.max}" : "${band.min}" }
def bandTag = { band ->
    band.max != null ? "_maf${bandLabel(band)}" : ((band.min as double) > 0 ? "_maf${band.min}" : "") }

def genoKey = { region, band ->
    slug("${params.cohort}_${region}_w${params.window_size}_${params.vcf_filter ?: 'nofilt'}${bandTag(band)}_allvar") }

def windowOf = { region ->
    def m = (region =~ /^(chr[0-9XYMT]+):(\d+)-(\d+)$/)
    if (!m) error "Unrecognised region '${region}'; expected chrN:start-end"
    def chrom = m[0][1]
    // The 1000G high-coverage release's autosome files all share this exact naming
    // pattern; chrX is a differently-named file (..._phased_panel.v2.vcf.gz, see
    // workflows/01-obtain_data/rules/1000genomes.smk) and there's no phased chrY/MT
    // file in this release at all -- fail fast rather than build a path to a file
    // that doesn't exist.
    if (params.cohort == 'geuvadis' && !(chrom ==~ /^chr([1-9]|1[0-9]|2[0-2])$/)) {
        error "--cohort geuvadis only supports autosomal regions (chr1-chr22); got '${region}' " +
            "(chrX/Y/MT aren't in the same 1000G high-coverage file set this workflow reads)."
    }
    long s = m[0][2] as long, e = m[0][3] as long
    long center = (s + e).intdiv(2)
    long half = (params.window_size as long).intdiv(2)
    long ws = Math.max(0L, center - half)
    return [chrom, ws, ws + (params.window_size as long)]
}

def bandList = params.maf_bands.toString().split(',').collect { parseBand(it.trim()) }

// GEUVADIS's genotype source (the 1000G high-coverage VCF) has no cohort-matched
// INFO/AF -- see module docstring -- so MAF-band filtering isn't available for it here.
if (params.cohort == 'geuvadis' &&
        !(bandList.size() == 1 && (bandList[0].min as double) == 0.0 && bandList[0].max == null)) {
    error "--cohort geuvadis does not support MAF-band filtering in this workflow " +
        "(got --maf_bands '${params.maf_bands}'); per-variant MAF/MAC assignment for " +
        "GEUVADIS is workflow 10's own concern, recomputed on its own cohort. Leave " +
        "--maf_bands at its default ('0.0')."
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process: params.batch_size regions' PLINK bfiles (each at its own MAF band), one
//          plain sequential bash loop per SLURM task -- ~8s/region, so
//          one-task-per-region flooded the scheduler with very-short-lived tasks at
//          any real gene count. Each item gets its own `${geno_key}/` subdirectory
//          (so N items' identically-named region.bed/.bim/.fam/maf.txt never collide
//          within one task dir) and its own `${geno_key}.tsv` manifest row -- both
//          published with the exact same paths/names the single-item version used,
//          so every consumer's lookup (src/nf/genotype_manifest.nf) is unaffected.
// ──────────────────────────────────────────────────────────────────────────── //

process region_bfile_batch {
    array 1000
    cpus 1
    memory { 8.GB * task.attempt }
    // Scaled off params.batch_size, NOT items.size() -- `array` packs every batched
    // task into ONE SLURM job array sharing a single #SBATCH header, so a directive
    // closure referencing a per-task INPUT (items) can't be resolved when that shared
    // header is built (confirmed directly: NullPointerException on items.size(),
    // items unbound in that context) -- same incompatibility workflow 06's own
    // header comment documents for clusterOptions. params.batch_size is a plain
    // top-level param, identical for every task, so it's safe here; the true last
    // (possibly smaller) batch just gets a harmlessly generous budget.
    time { (10.min + 20.sec * params.batch_size) * task.attempt }
    errorStrategy 'retry'
    maxRetries 2
    tag "batch of ${items.size()}"

    publishDir { "${DATA_ROOT}/region_bfile" }, mode: 'copy',
        pattern: "*/{region.bed,region.bim,region.fam,maf.txt}"
    publishDir { "${MANIFEST_DIR}/region_bfile" }, mode: 'copy',
        pattern: "manifest/*.tsv", saveAs: { it - 'manifest/' }

    input:
    val(items)   // list of [geno_key, region, chrom, win_start, win_end, band]

    output:
    path("*/{region.bed,region.bim,region.fam,maf.txt}")
    // Must be declared here, not just named in publishDir's pattern above: with
    // `scratch = '/tmp'` in the shared nextflow.config the task runs node-local and
    // only declared outputs are staged back before publishDir runs.
    path("manifest/*.tsv")

    script:
    // Built as one shell script string, one item's worth of commands per iteration --
    // MAF filtered on the VCF's own INFO/AF field via bcftools, not plink2's --maf/
    // --max-maf (which recompute frequency from the genotypes they're given) -- moved
    // verbatim from the three workflows this replaces; see their own history for the
    // verification that this matches plink2's own recomputed frequency exactly.
    def perItem = items.collect { item ->
        def (geno_key, region, chrom, win_start, win_end, band) = item
        def start1 = (win_start as long) + 1   // bcftools regions are 1-based inclusive
        def afConds = []
        if ((band.min as double) > 0) {
            afConds << "INFO/AF>=${band.min} && INFO/AF<=${1.0 - (band.min as double)}"
        }
        if (band.max != null) {
            afConds << "(INFO/AF<=${band.max} || INFO/AF>=${1.0 - (band.max as double)})"
        }
        def afFilter = afConds ? "-i '${afConds.join(' && ')}'" : ""
        """
        echo "[region_bfile_batch] ${geno_key}"
        mkdir -p ${geno_key}
        bcftools view --regions ${chrom}:${start1}-${win_end} -Oz -o ${geno_key}/region_raw.vcf.gz ${GTEX_VCF}
        bcftools index ${geno_key}/region_raw.vcf.gz
        bcftools view ${afFilter} -Oz -o ${geno_key}/region.vcf.gz ${geno_key}/region_raw.vcf.gz
        bcftools index ${geno_key}/region.vcf.gz

        ${PLINK} --vcf ${geno_key}/region.vcf.gz --double-id --make-bed --out ${geno_key}/region
        echo "region variants: \$(wc -l < ${geno_key}/region.bim)"

        ${PLINK} --bfile ${geno_key}/region --freq --out ${geno_key}/region_freq
        awk 'NR==1 {for (i=1; i<=NF; i++) if (\$i == "ALT_FREQS") c=i; print "MAF"; next}
             {af=\$c; print (af<=0.5) ? af : 1-af}' ${geno_key}/region_freq.afreq > ${geno_key}/maf.txt

        printf 'product_type\\tregion\\twindow_size\\tmin_maf\\tmax_maf\\tvcf_filter\\tgeno_key\\tbed_path\\tbim_path\\tfam_path\\tmaf_path\\tcreated_at\\n' > manifest/${geno_key}.tsv
        printf 'region_bfile\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\tregion_bfile/%s/region.bed\\tregion_bfile/%s/region.bim\\tregion_bfile/%s/region.fam\\tregion_bfile/%s/maf.txt\\t%s\\n' \\
            '${region}' '${params.window_size}' '${band.min}' '${band.max ?: ''}' '${params.vcf_filter ?: ''}' \\
            '${geno_key}' '${geno_key}' '${geno_key}' '${geno_key}' '${geno_key}' "\$(date -u +%FT%TZ)" >> manifest/${geno_key}.tsv
        """
    }.join("\n")
    """
    source ${CONDA_SH}
    conda activate ${BCFTOOLS_ENV}
    mkdir -p manifest

    ${perItem}
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process (geuvadis branch): GEUVADIS's own 462 individuals (from GD660's column
//          headers, same "<individual>.<lane>.<run>" convention workflow 05/02 use),
//          intersected with one chromosome's actual sample panel -- identical across
//          all 22 autosome files in this 1000G release, so computed once here, not
//          per-region. Lands on the same ~449-donor set workflow 02's geuvadis branch
//          and workflow 05 both already reach independently from the same source.
// ──────────────────────────────────────────────────────────────────────────── //

process geuvadis_donor_list {
    cpus 1
    memory { 2.GB * task.attempt }
    time { 10.min * task.attempt }
    errorStrategy 'retry'
    maxRetries 2

    publishDir { "${DATA_ROOT}" }, mode: 'copy'

    output:
    path("geuvadis_donors.txt")

    script:
    """
    source ${CONDA_SH}
    conda activate ${BCFTOOLS_ENV}

    zcat ${GEUVADIS_RPKM} | head -1 | tr '\\t' '\\n' | tail -n +5 \\
        | sed -E 's/\\..*//' | sort -u > all_donors.txt
    bcftools query -l ${ONEKG_VCF_DIR}/1kGP_high_coverage_Illumina.chr21.filtered.SNV_INDEL_SV_phased_panel.vcf.gz \\
        | sort -u > onekg_donors.txt
    comm -12 all_donors.txt onekg_donors.txt > geuvadis_donors.txt

    echo "GEUVADIS donors with 1000G genotype: \$(wc -l < geuvadis_donors.txt)/\$(wc -l < all_donors.txt)"
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
// Process (geuvadis branch): GEUVADIS's counterpart of region_bfile_batch -- same
//          batching/manifest shape, but one VCF per chromosome, --samples-file
//          restricted to geuvadis_donor_list's output, and no AF-band filtering (see
//          module docstring; every item reaching here has band == floor-0, enforced
//          above). maf.txt is still written via plink2 --freq on the extracted
//          (already donor-restricted) bfile -- cohort-correct regardless of the
//          missing AF-band step.
// ──────────────────────────────────────────────────────────────────────────── //

process geuvadis_region_bfile_batch {
    array 1000
    cpus 1
    memory { 8.GB * task.attempt }
    time { (10.min + 20.sec * params.batch_size) * task.attempt }   // see region_bfile_batch's own comment
    errorStrategy 'retry'
    maxRetries 2
    tag "batch of ${items.size()}"

    publishDir { "${DATA_ROOT}/region_bfile" }, mode: 'copy',
        pattern: "*/{region.bed,region.bim,region.fam,maf.txt}"
    publishDir { "${MANIFEST_DIR}/region_bfile" }, mode: 'copy',
        pattern: "manifest/*.tsv", saveAs: { it - 'manifest/' }

    input:
    val(items)   // list of [geno_key, region, chrom, win_start, win_end, band]
    path(donors)

    output:
    path("*/{region.bed,region.bim,region.fam,maf.txt}")
    path("manifest/*.tsv")

    script:
    def perItem = items.collect { item ->
        def (geno_key, region, chrom, win_start, win_end, band) = item
        def start1 = (win_start as long) + 1   // bcftools regions are 1-based inclusive
        def vcf = "${ONEKG_VCF_DIR}/1kGP_high_coverage_Illumina.${chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
        """
        echo "[geuvadis_region_bfile_batch] ${geno_key}"
        mkdir -p ${geno_key}
        bcftools view --regions ${chrom}:${start1}-${win_end} --samples-file ${donors} --force-samples \\
            -Oz -o ${geno_key}/region.vcf.gz ${vcf}
        bcftools index ${geno_key}/region.vcf.gz

        ${PLINK} --vcf ${geno_key}/region.vcf.gz --double-id --make-bed --out ${geno_key}/region
        echo "region variants: \$(wc -l < ${geno_key}/region.bim)"

        ${PLINK} --bfile ${geno_key}/region --freq --out ${geno_key}/region_freq
        awk 'NR==1 {for (i=1; i<=NF; i++) if (\$i == "ALT_FREQS") c=i; print "MAF"; next}
             {af=\$c; print (af<=0.5) ? af : 1-af}' ${geno_key}/region_freq.afreq > ${geno_key}/maf.txt

        printf 'product_type\\tregion\\twindow_size\\tmin_maf\\tmax_maf\\tvcf_filter\\tgeno_key\\tbed_path\\tbim_path\\tfam_path\\tmaf_path\\tcreated_at\\n' > manifest/${geno_key}.tsv
        printf 'region_bfile\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\tregion_bfile/%s/region.bed\\tregion_bfile/%s/region.bim\\tregion_bfile/%s/region.fam\\tregion_bfile/%s/maf.txt\\t%s\\n' \\
            '${region}' '${params.window_size}' '${band.min}' '${band.max ?: ''}' '${params.vcf_filter ?: ''}' \\
            '${geno_key}' '${geno_key}' '${geno_key}' '${geno_key}' '${geno_key}' "\$(date -u +%FT%TZ)" >> manifest/${geno_key}.tsv
        """
    }.join("\n")
    """
    source ${CONDA_SH}
    conda activate ${BCFTOOLS_ENV}
    mkdir -p manifest

    ${perItem}
    """
}


// ──────────────────────────────────────────────────────────────────────────── //
workflow {

    regions = Channel
        .fromPath(params.config_tsv)
        .splitCsv(header: true, sep: '\t')
        .map { r ->
            if (!r.region) error "config row missing 'region': ${r}"
            r.region.trim()
        }
        .unique()

    regionBand = regions
        .combine(Channel.fromList(bandList))
        .map { reg, band ->
            def (chrom, ws, we) = windowOf(reg)
            [genoKey(reg, band), reg, chrom, ws, we, band]
        }

    if (params.cohort == 'gtex') {
        region_bfile_batch(regionBand.collate(params.batch_size))
    } else {
        // .first() turns this single-emission output into a value channel, which
        // Nextflow then broadcasts/repeats across every batch -- NOT .combine(), which
        // would flatten regionBand.collate()'s list-of-lists at the top level instead
        // of pairing it as one element with the donors path (confirmed directly:
        // "Not a valid path value" from a 4-element flattened tuple).
        donors = geuvadis_donor_list().first()
        geuvadis_region_bfile_batch(regionBand.collate(params.batch_size), donors)
    }
}
