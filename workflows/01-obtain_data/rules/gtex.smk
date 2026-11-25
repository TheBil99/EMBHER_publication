# Public GTEx v10 files (GTEx Portal). The controlled-access GTEx files (phased WGS VCF, subject phenotypes
# with self-reported race, genotype PCs) come from dbGaP phs000424 and must be placed by hand, see
# workflows/01-obtain_data/README.md. The eQTL covariates tar (GTEx_Analysis_v10_eQTL_covariates.tar) is on
# https://gtexportal.org/home/downloads/adult-gtex/qtl and is unpacked by hand into GTEX_DIR.

GTEX_DIR = os.path.join(RAW_DIR, "GTEx", "v10")
GTEX_BUCKET = "https://storage.googleapis.com/adult-gtex"

GTEX_FILES = {
    "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz":
        GTEX_BUCKET + "/bulk-gex/v10/rna-seq/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz",
    "GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt":
        GTEX_BUCKET + "/annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt",
    "GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt":
        GTEX_BUCKET + "/annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt",
}


rule download_gtex_public:
    params:
        url = lambda wc: GTEX_FILES[wc.name]
    output:
        os.path.join(GTEX_DIR, "{name}")
    wildcard_constraints:
        name = "|".join(GTEX_FILES)
    threads: 1
    resources:
        runtime = 30,
        memory = 2
    conda: "../../../envs/embher_env.yaml"
    shell:
        "wget -c --no-check-certificate {params.url} -O {output}"
