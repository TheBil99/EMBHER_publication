rule download_genome_sequence:
    params:
        url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/GRCh38.primary_assembly.genome.fa.gz",
    output:
        fasta = os.path.join(RAW_DIR,"GENCODE","human","release_46","GRCh38.primary_assembly.genome.fa.gz"),
        index = os.path.join(RAW_DIR,"GENCODE","human","release_46","GRCh38.primary_assembly.genome.fa.gz.fai")
    threads: 1
    resources:
        gres = "none",
        runtime = 2*60,
        memory = 4
    conda:
        "../../../envs/embher_env.yaml"
    shell:
        """
        wget --user-agent="Chrome" --no-check-certificate {params.url} -O - | gunzip | bgzip -c > {output.fasta}

        samtools faidx {output.fasta}

        echo "Done!"
        """

rule download_genome_annotation:
    params:
        url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz",
    output:
        gtf = os.path.join(RAW_DIR,"GENCODE","human","release_46","gencode.v46.annotation.gtf.gz"),
    threads: 1
    resources:
        gres = "none",
        runtime = 1*60,
        memory = 4
    conda:
        "../../../envs/embher_env.yaml"
    shell:
        """
        wget --user-agent="Chrome" --no-check-certificate {params.url} -O {output.gtf}

        echo "Done!"
        """

rule gencode_gtf_to_parquet:
    # One row per feature, attributes as columns: the --gencode_parquet of embher-capture and workflows 06/15/03
    input:
        gtf = os.path.join(RAW_DIR,"GENCODE","human","release_46","gencode.v46.annotation.gtf.gz")
    output:
        parquet = os.path.join(RAW_DIR,"GENCODE","human","release_46","gencode.v46.annotation.gtf.parquet")
    threads: 1
    resources:
        runtime = 30,
        memory = 24
    conda: "../../../envs/embher_env.yaml"
    shell:
        "python workflows/01-obtain_data/scripts/gtf_to_parquet.py --gtf {input.gtf} --out {output.parquet}"
