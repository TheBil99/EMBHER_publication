UNPHASED_FTP = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_raw_GT_with_annot"
PHASED_FTP   = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"

UNPHASED_DIR = os.path.join(RAW_DIR, "1000genomes", "data_collections", "1000G_2504_high_coverage", "working", "20201028_3202_raw_GT_with_annot")
PHASED_DIR   = os.path.join(RAW_DIR, "1000genomes", "data_collections", "1000G_2504_high_coverage", "working", "20220422_3202_phased_SNV_INDEL_SV")

UNPHASED_CHROMS = ["chr" + str(c) for c in range(1, 23)] + ["chrX", "chrY", "others"]
PHASED_CHROMS   = ["chr" + str(c) for c in range(1, 23)]


rule download_1000genomes_unphased:
    wildcard_constraints:
        chrom = "|".join(UNPHASED_CHROMS)
    params:
        url = lambda wc: "{}/20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_{}.recalibrated_variants.vcf.gz".format(UNPHASED_FTP, wc.chrom)
    output:
        vcf = os.path.join(UNPHASED_DIR, "20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_{chrom}.recalibrated_variants.vcf.gz"),
        tbi = os.path.join(UNPHASED_DIR, "20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_{chrom}.recalibrated_variants.vcf.gz.tbi"),
    threads: 1
    resources:
        gres = "none",
        runtime = 12*60,
        memory = 4
    conda: "../../../envs/embher_env.yaml"
    shell:
        """
        wget --no-check-certificate {params.url}     -O {output.vcf}
        wget --no-check-certificate {params.url}.tbi -O {output.tbi}

        echo "Done!"
        """


rule download_1000genomes_phased:
    wildcard_constraints:
        chrom = "|".join(PHASED_CHROMS)
    params:
        url = lambda wc: "{}/1kGP_high_coverage_Illumina.{}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz".format(PHASED_FTP, wc.chrom)
    output:
        vcf = os.path.join(PHASED_DIR, "1kGP_high_coverage_Illumina.{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"),
        tbi = os.path.join(PHASED_DIR, "1kGP_high_coverage_Illumina.{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz.tbi"),
    threads: 1
    resources:
        gres = "none",
        runtime = 2*60,
        memory = 4
    conda: "../../../envs/embher_env.yaml"
    shell:
        """
        wget --no-check-certificate {params.url}     -O {output.vcf}
        wget --no-check-certificate {params.url}.tbi -O {output.tbi}

        echo "Done!"
        """


rule download_1000genomes_phased_chrX:
    params:
        url = "{}/1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.v2.vcf.gz".format(PHASED_FTP)
    output:
        vcf = os.path.join(PHASED_DIR, "1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.v2.vcf.gz"),
        tbi = os.path.join(PHASED_DIR, "1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.v2.vcf.gz.tbi"),
    threads: 1
    resources:
        gres = "none",
        runtime = 12*60,
        memory = 4
    conda: "../../../envs/embher_env.yaml"
    shell:
        """
        wget --no-check-certificate {params.url}     -O {output.vcf}
        wget --no-check-certificate {params.url}.tbi -O {output.tbi}

        echo "Done!"
        """
