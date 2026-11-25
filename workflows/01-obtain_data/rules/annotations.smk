# Public annotation and reference resources used by the attribution (15) and gene-selection steps.
# CisBP-RNA has no stable download URL (web form): see the note at the end of this file.

HOCOMOCO_DIR = os.path.join(RAW_DIR, "HOCOMOCO", "hocomoco13", "H13CORE", "formatted_motifs")
PHYLOP_DIR = os.path.join(RAW_DIR, "phylop")
GNOMAD_DIR = os.path.join(RAW_DIR, "gnomad", "v4.1.1")
GNOMAD_BASE = "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1.1/vcf/genomes"
CHROMS = [str(i) for i in range(1, 23)]


rule download_hocomoco:
    # HOCOMOCO v13 CORE motifs (TFs), MEME format
    params:
        url = "https://hocomoco13.autosome.org/final_bundle/hocomoco13/H13CORE/formatted_motifs/H13CORE_meme_format.meme"
    output:
        meme = os.path.join(HOCOMOCO_DIR, "H13CORE_meme_format.meme")
    threads: 1
    resources:
        runtime = 10,
        memory = 1
    conda: "../../../envs/embher_env.yaml"
    shell:
        "wget --no-check-certificate {params.url} -O {output.meme}"


rule download_phylop:
    # 241-way mammalian (Zoonomia) phyloP, hg38, bigWig
    params:
        url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/cactus241way/cactus241way.phyloP.bw"
    output:
        bw = os.path.join(PHYLOP_DIR, "cactus241way.phyloP.bw")
    threads: 1
    resources:
        runtime = 4 * 60,
        memory = 1
    conda: "../../../envs/embher_env.yaml"
    shell:
        "wget -c --no-check-certificate {params.url} -O {output.bw}"


rule download_gnomad_chrom:
    # gnomAD v4.1.1 genomes sites VCF, one file per autosome (each several tens of GB)
    params:
        url = lambda wc: "{}/gnomad.genomes.v4.1.1.sites.chr{}.vcf.bgz".format(GNOMAD_BASE, wc.chrom)
    output:
        vcf = os.path.join(GNOMAD_DIR, "gnomad.genomes.v4.1.1.sites.chr{chrom}.vcf.bgz"),
        tbi = os.path.join(GNOMAD_DIR, "gnomad.genomes.v4.1.1.sites.chr{chrom}.vcf.bgz.tbi")
    wildcard_constraints:
        chrom = "|".join(CHROMS)
    threads: 1
    resources:
        runtime = 12 * 60,
        memory = 2
    conda: "../../../envs/embher_env.yaml"
    shell:
        """
        wget -c --no-check-certificate {params.url}     -O {output.vcf}
        wget -c --no-check-certificate {params.url}.tbi -O {output.tbi}
        """

# CisBP-RNA (RBP motifs, workflow 15): download "Homo sapiens" PWMs from http://cisbp-rna.ccbr.utoronto.ca
# (web form), unzip into data/raw/CISBP-RNA/ so that
# data/raw/CISBP-RNA/pwms_all_motifs/ and RBP_Information_all_motifs.txt exist, then build the DNA-alphabet
# MEME file read by workflow 15 with workflows/15-borzoi_attribution/scripts/prepare_motif_tables.py.
