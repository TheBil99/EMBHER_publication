GEUVADIS_FILES_BASE = "https://ftp.ebi.ac.uk/biostudies/fire/E-GEUV-/001/E-GEUV-1/Files"

GEUVADIS_DIR = os.path.join(RAW_DIR, "GEUVADIS")


rule download_geuvadis_gene_rpkm:
    params:
        url = "{}/E-GEUV-1/analysis_results/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz".format(GEUVADIS_FILES_BASE)
    output:
        rpkm = os.path.join(GEUVADIS_DIR, "GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz"),
    threads: 1
    resources:
        gres = "none",
        runtime = 1*60,
        memory = 4
    conda: "../../../envs/embher_env.yaml"
    shell:
        """
        wget --no-check-certificate {params.url} -O {output.rpkm}

        echo "Done!"
        """


rule download_geuvadis_gene_rpkm_660:
    params:
        url = "{}/E-GEUV-1/analysis_results/GD660.GeneQuantRPKM.txt.gz".format(GEUVADIS_FILES_BASE)
    output:
        rpkm = os.path.join(GEUVADIS_DIR, "GD660.GeneQuantRPKM.txt.gz"),
    threads: 1
    resources:
        gres = "none",
        runtime = 1*60,
        memory = 4
    conda: "../../../envs/embher_env.yaml"
    shell:
        """
        wget --no-check-certificate {params.url} -O {output.rpkm}

        echo "Done!"
        """


rule download_geuvadis_sample_metadata:
    params:
        url = "{}/E-GEUV-1.sdrf.txt".format(GEUVADIS_FILES_BASE)
    output:
        sdrf = os.path.join(GEUVADIS_DIR, "E-GEUV-1.sdrf.txt"),
    threads: 1
    resources:
        gres = "none",
        runtime = 1*60,
        memory = 4
    conda: "../../../envs/embher_env.yaml"
    shell:
        """
        wget --no-check-certificate {params.url} -O {output.sdrf}

        echo "Done!"
        """


rule download_geuvadis_readmes:
    params:
        readme_url = "{}/E-GEUV-1/analysis_results/README".format(GEUVADIS_FILES_BASE),
        files_readme_url = "{}/E-GEUV-1/analysis_results/GeuvadisRNASeqAnalysisFiles_README.txt".format(GEUVADIS_FILES_BASE)
    output:
        readme = os.path.join(GEUVADIS_DIR, "README"),
        files_readme = os.path.join(GEUVADIS_DIR, "GeuvadisRNASeqAnalysisFiles_README.txt"),
    threads: 1
    resources:
        gres = "none",
        runtime = 1*60,
        memory = 4
    conda: "../../../envs/embher_env.yaml"
    shell:
        """
        wget --no-check-certificate {params.readme_url}       -O {output.readme}
        wget --no-check-certificate {params.files_readme_url} -O {output.files_readme}

        echo "Done!"
        """
