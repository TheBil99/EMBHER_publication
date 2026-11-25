#!/usr/bin/env python
"""manifest.tsv, layers.tsv and README.md of the seqlet_beds bundle (md5sums.txt is written by the zip step)."""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from bundle_common import LAYERS, MODELS, bed_name

README = """# Seqlets of Borzoi attributions along the architecture

Seqlet intervals (hg38) called on the input x gradient attribution of Borzoi (replicate 0) to the L2 norm of the
embedding of each of 24 layers and to the whole-blood expression prediction, for {n_genes} genes of the
`{source_run}` panel, with pretrained weights and with randomly initialised weights (same architecture and head,
seed {seed}). Made on {today} by workflow 99 of the embher_analysis project, from the seqlets of workflow 15.

## Files

`seqlet_beds/<model>_layer-<NN>_<layer_name>.bed.gz` (+ `.tbi`), 50 files: model is `pretrained` or `random_init`;
`<NN>` and `<layer_name>` are the `layer_index` and `layer_name` of `layers.tsv` (forward-pass order, as in the
paper's supplementary table of Borzoi layers). Layer 25, `expression`, is the attribution to the whole-blood
expression prediction, not a layer; it is the last point of the figures.

- `layers.tsv`: layer_index, layer_name, workflow-15 target, and the filename of each model.
- `manifest.tsv`: per gene, model and layer, the number of seqlets, chromosome, TSS and window start.
- `md5sums.txt`: checksums of every file of this archive.

## Columns of the BED files

Tab-separated, one header line starting with `#`, sorted by chromosome then start (bgzip + tabix, `tabix -p bed`).

| Column | Meaning |
|---|---|
| `chrom`, `start`, `end` | hg38, 0-based half-open |
| `name` | `<gene>\\|<seqlet_id>`; `seqlet_id` is the 0-based row order of the seqlet table of that gene and target |
| `score` | summed attribution of the seqlet |
| `strand` | `.` (seqlets are strand-less; the sign of the attribution is `pattern_type`) |
| `p_value` | seqlet p-value (tangermeme recursive seqlets, via Decima's `find_peaks`) |
| `pattern_type` | `pos` or `neg` attribution |
| `gene` | gene name |

## How they were made

- Window: 524,288 bp around the TSS of each gene on the forward strand of the chromosome (`manifest.tsv` has
  `win_start` and `tss`); genomic position = `win_start` + position in the window.
- Attribution: input x gradient, {precision}, to the L2 norm of the layer's output; seqlets called on the
  base-summed whole-window attribution with p < {threshold}, length {min_len} to {max_len}.
- Nothing was filtered by region or sequence. Seqlets of every target of one gene overlap freely.
- Every interval was checked against the window sequence stored with the attribution and against the GRCh38
  primary assembly FASTA (GENCODE release 46) before packing.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packed_dir", required=True)
    ap.add_argument("--genes", required=True)
    ap.add_argument("--genes_dir", required=True)
    ap.add_argument("--source_run", required=True)
    ap.add_argument("--out_dir", required=True, help="folder of manifest.tsv, layers.tsv, README.md")
    a = ap.parse_args()

    genes = [g.strip() for g in open(a.genes) if g.strip()]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.concat([pd.read_csv(Path(a.packed_dir) / g / "meta.tsv", sep="\t") for g in genes])
    meta["file"] = [f"seqlet_beds/{bed_name(m, i, n)}" for m, i, n in
                    zip(meta["model"], meta["layer_index"], meta["layer_name"])]
    meta.to_csv(out / "manifest.tsv", sep="\t", index=False)
    rows = [dict(layer_index=i, layer_name=n, target=t,
                 **{f"{m}_file": f"seqlet_beds/{bed_name(m, i, n)}" for m in MODELS.values()})
            for i, n, t in LAYERS]
    pd.DataFrame(rows).to_csv(out / "layers.tsv", sep="\t", index=False)

    g0 = Path(a.genes_dir) / genes[0]
    sl = json.loads(next((g0 / "seqlets" / "reference").glob("*/seqlets.json")).read_text())["settings"]
    rsum = json.loads(next((g0 / "attribution" / "random_init").glob("*/summary.json")).read_text())
    (out / "README.md").write_text(README.format(
        n_genes=len(genes), source_run=a.source_run, seed=rsum.get("seed"), today=date.today().isoformat(),
        precision="bf16", threshold=sl["threshold"], min_len=sl["min_len"], max_len=sl["max_len"]))
    print(f"[done] docs for {len(genes)} genes -> {out}", flush=True)


if __name__ == "__main__":
    main()
