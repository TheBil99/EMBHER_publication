GENOMI_ANCESTRY_PANEL_URL = "https://github.com/exon-research/genomi-ancestry-panel/releases/download/v1.0.0/panel-1000g-30x-grch38-1.0.0.tar.gz"

GENOMI_ANCESTRY_PANEL_DIR = os.path.join(RAW_DIR, "GenomiAncestryPanel", "panel-1000g-30x-grch38-1.0.0")


rule download_genomi_ancestry_panel:
    # Third-party PCA panel (not part of the official 1000 Genomes/IGSR release) built
    # from the same 1000G high-coverage 30x phased VCFs this project already uses (see
    # its own manifest.json's source_inputs) -- per-sample PC1-10 scores for all 3202
    # 1000G donors, covering every GEUVADIS individual this project's own 1000G-VCF
    # genotype-availability filter keeps (confirmed: the panel's 13 missing-vs-GD462
    # donors are exactly the same 13 workflows/02-expression_data_preparation's own
    # geuvadis branch already drops). Used as GEUVADIS's genotype-PC covariates
    # (workflow 05), since 1000G/IGSR itself publishes no such file -- unlike GTEx,
    # which ships its own.
    params:
        url = GENOMI_ANCESTRY_PANEL_URL
    output:
        reference_scores = os.path.join(GENOMI_ANCESTRY_PANEL_DIR, "reference_scores.tsv"),
        samples = os.path.join(GENOMI_ANCESTRY_PANEL_DIR, "samples.tsv"),
        manifest = os.path.join(GENOMI_ANCESTRY_PANEL_DIR, "manifest.json"),
    threads: 1
    resources:
        gres = "none",
        runtime = 20,
        memory = 2
    conda: "../../../envs/embher_env.yaml"
    shell:
        """
        mkdir -p $(dirname $(dirname {output.reference_scores}))
        wget --no-check-certificate {params.url} -O $(dirname $(dirname {output.reference_scores}))/panel-1000g-30x-grch38-1.0.0.tar.gz
        tar xzf $(dirname $(dirname {output.reference_scores}))/panel-1000g-30x-grch38-1.0.0.tar.gz -C $(dirname $(dirname {output.reference_scores}))
        rm $(dirname $(dirname {output.reference_scores}))/panel-1000g-30x-grch38-1.0.0.tar.gz

        echo "Done!"
        """
