#!/usr/bin/env python3
"""
Build the two expressed-gene panels: hi_q_gtex_genes and hi_q_geuvadis_genes.

A panel is every gene with TPM > --min-tpm in at least --min-pct % of the donors of workflow 02's
cached TPM matrix (GTEx whole blood / GEUVADIS_LCL), restricted to genes Borzoi's pipeline can run
("viable": autosomal, in GENCODE v46, gene body inside the +-97,304 bp window around the TSS -- the
rule of workflow 06's fits_borzoi_crop).

Per panel, in --out:
  panels/<panel>.tsv     gene_id, gene_symbol, borzoi_compatible
  gene_meta/<panel>.tsv  the genes in workflow 06's --gene_meta_tsv format

Usage (from the project root, embher_env env, via sbatch):
  python workflows/03-gene_panels/scripts/build_hi_q_panels.py --out results/03-gene_panels
"""
import argparse
from pathlib import Path

import pandas as pd

HALF = 6144 * 32 // 2 - 1000
AUTOSOMES = {f'chr{i}' for i in range(1, 23)}
GENCODE = 'data/raw/GENCODE/human/release_46/gencode.v46.annotation.gtf.parquet'   # built by workflow 01 (gencode_gtf_to_parquet)
EXPR = {'hi_q_gtex_genes': 'results/02-expression_data_preparation/gtex/tissue_matrix/Whole_Blood.parquet',
        'hi_q_geuvadis_genes': 'results/02-expression_data_preparation/geuvadis/tissue_matrix/GEUVADIS_LCL.parquet'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-tpm', type=float, default=0.1)
    ap.add_argument('--min-pct', type=int, default=50)
    ap.add_argument('--gencode', default=GENCODE)
    a = ap.parse_args()
    out = Path(a.out)
    (out / 'panels').mkdir(parents=True, exist_ok=True)
    (out / 'gene_meta').mkdir(parents=True, exist_ok=True)

    g = pd.read_parquet(a.gencode, columns=['Chromosome', 'Feature', 'Start', 'End', 'Strand', 'gene_id', 'gene_name'])
    g = g[g['Feature'] == 'gene'].copy()
    g['gene_id'] = g['gene_id'].str.split('.').str[0]
    g = g.drop_duplicates('gene_id')
    g['tss'] = g['Start'].where(g['Strand'] == '+', g['End'] - 1)
    g['viable'] = (g['Chromosome'].isin(AUTOSOMES) & (g['Start'] >= g['tss'] - HALF) & (g['End'] <= g['tss'] + HALF))
    g = g[g['viable']].set_index('gene_id', drop=False)

    for panel, path in EXPR.items():
        mat = pd.read_parquet(path)
        mat.index = mat.index.str.split('.').str[0]
        expressed = set(mat.index[(mat.values > a.min_tpm).mean(axis=1) >= a.min_pct / 100])
        p = g.loc[sorted(expressed & set(g.index))]
        pd.DataFrame({'gene_id': p['gene_id'], 'gene_symbol': p['gene_name'], 'borzoi_compatible': True}
                     ).to_csv(out / 'panels' / f'{panel}.tsv', sep='\t', index=False)
        pd.DataFrame({'gene_id': p['gene_id'], 'gene_symbol': p['gene_name'],
                      'chr_hg38': p['Chromosome'].str.removeprefix('chr'), 'start_hg38': p['Start'].astype('int64'),
                      'end_hg38': p['End'].astype('int64'), 'tss_hg38': p['tss'].astype('int64')}
                     ).to_csv(out / 'gene_meta' / f'{panel}.tsv', sep='\t', index=False)
        print(f'[done] {panel}: {len(p)} genes ({len(expressed)} expressed in {path})', flush=True)


if __name__ == '__main__':
    main()
