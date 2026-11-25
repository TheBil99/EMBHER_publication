rule download_borzoi_weights:
    params:
        weights = "johahi/borzoi-replicate-0"
    output:
        weights = directory(os.path.join(ARTICLES_DIR,"Hingerl2024","model_weights","borzoi-replicate-0"))
    conda: "../../../envs/embher_env.yaml"
    shell:
        """
        huggingface-cli download {params.weights} --local-dir {output.weights}

        echo "Done!"
        """

rule download_flashzoi_weights:
    params:
        weights = "johahi/flashzoi-replicate-0"
    output:
        weights = directory(os.path.join(ARTICLES_DIR,"Hingerl2024","model_weights","flashzoi-replicate-0"))
    conda: "../../../envs/embher_env.yaml"
    shell:
        """
        huggingface-cli download {params.weights} --local-dir {output.weights}

        echo "Done!"
        """

rule download_borzoi_intervals:
    params:
        intervals = "https://github.com/calico/borzoi/raw/5c9358222b5026abb733ed5fb84f3f6c77239b37/data/sequences_human.bed.gz"
    output:
        train = os.path.join(ARTICLES_DIR,"Hingerl2024","intervals","borzoi-replicate-0","train.bed.gz"),
        val = os.path.join(ARTICLES_DIR,"Hingerl2024","intervals","borzoi-replicate-0","val.bed.gz"),
        test = os.path.join(ARTICLES_DIR,"Hingerl2024","intervals","borzoi-replicate-0","test.bed.gz")
    run:
        import pandas as pd

        # load
        intervals = pd.read_table(params.intervals, header=None)
        intervals.columns = ["chrom","start","end","fold"]
        
        # prepare intervals
        train = intervals.loc[~intervals["fold"].isin(["fold3","fold4"])]
        val = intervals.loc[intervals["fold"].isin(["fold3"])]
        test = intervals.loc[intervals["fold"].isin(["fold4"])]
            
        # save
        train.to_csv(output.train, header=None, **SAVE_PARAMS)
        val.to_csv(output.val, header=None, **SAVE_PARAMS)
        test.to_csv(output.test, header=None, **SAVE_PARAMS)
            
        print("Done!")