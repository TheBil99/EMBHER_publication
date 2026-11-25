#!/usr/bin/env python
"""Enriched motifs along the Borzoi architecture: one figure per motif database, over many genes.

Reads <genes_dir>/<gene>/annotation/<db_name>/motif_layers.parquet (motif_layers.py) and plots, per condition
(pretrained = `reference`, `random_init`), the mean over genes of the number of enriched motifs of each layer's
attribution seqlets, with a 95% bootstrap CI over genes. The whole-blood expression target (pretrained only)
is the last point. All genes are used.

--normalize adjusts each gene's count for its number of seqlets (the power of the test grows with it): per layer,
n_enriched is regressed on log(n_seqlets) across genes (slope estimated within each role, so the roles' different
levels do not drive it, and shared between them so they stay comparable) and each value
is moved to the mean log seqlet count, n_adj = n_enriched - slope * (log n_seqlets - mean log n_seqlets). Files get
a `_normalized` suffix (the raw ones are untouched).

Writes <out_prefix>.png / .pdf, <out_prefix>_per_gene.parquet (enriched motifs per gene, role and layer, long)
and <out_prefix>_summary.parquet (value, CI and n genes per role and layer; `expression` is a label).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from layer_axes import draw_profile, save
from layer_order import LABELS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes_dir", required=True, help="<run>/genes; one folder per gene")
    ap.add_argument("--db_name", default="hocomoco_v13", help="motif database folder under each gene")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--unit", default="TFs", help="what the names are (TFs, RBPs), for the labels")
    ap.add_argument("--normalize", action="store_true",
                    help="adjust the enriched counts for the number of seqlets (see the docstring); "
                         "writes <out_prefix>_normalized.*")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(a.genes_dir).glob(f"*/annotation/{a.db_name}/motif_layers.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "scale" in df:                                    # whole-window scale only
        df = df[df["scale"] == "whole"]
    genes = sorted(df["gene"].unique())
    out_prefix = a.out_prefix + ("_normalized" if a.normalize else "")
    value_col, ylabel = "n_enriched", f"enriched {a.unit} per gene"
    if a.normalize:
        df = df.copy()
        df["n_enriched_raw"] = df["n_enriched"]
        df["log_seqlets"] = np.log(df["n_seqlets"].clip(lower=1))
        df["n_enriched"] = df["n_enriched"].astype(float)
        for lab, g in df.groupby("label"):
            x = g["log_seqlets"].to_numpy()
            y = g["n_enriched_raw"].to_numpy(float)
            xc = x - g.groupby("role")["log_seqlets"].transform("mean").to_numpy()      # within-role: the roles differ in level,
            yc = y - g.groupby("role")["n_enriched_raw"].transform("mean").to_numpy()   # which must not drive the slope
            if len(g) < 3 or xc.std() == 0:
                continue
            slope = (xc * yc).sum() / (xc ** 2).sum()
            df.loc[g.index, "n_enriched"] = y - slope * (x - x.mean())
            df.loc[g.index, "slope"] = slope
        ylabel = f"enriched {a.unit} per gene (adjusted to mean seqlet count)"
    cols = ["gene", "role", "label", "n_enriched"] + (["n_enriched_raw", "n_seqlets", "slope"] if a.normalize else [])
    df[cols].to_parquet(
        f"{out_prefix}_per_gene.parquet", compression="zstd", index=False)
    print(f"[info] {len(genes)} genes", flush=True)

    w = rng.multinomial(len(genes), np.full(len(genes), 1 / len(genes)), size=a.n_boot) / len(genes)
    rows = []
    for role in ("reference", "random_init"):
        for lab in LABELS + ["expression"]:
            t = df[(df["role"] == role) & (df["label"] == lab)].set_index("gene")["n_enriched"]
            if t.empty:
                continue
            v = t.reindex(genes).fillna(0).to_numpy(float)
            boot = w @ v
            rows.append(dict(role=role, label=lab, n_genes=int(t.reindex(genes).notna().sum()),
                             value=v.mean(), ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5)))
    s = pd.DataFrame(rows)
    s.to_parquet(f"{out_prefix}_summary.parquet", compression="zstd", index=False)

    fig = draw_profile(s, ylabel,
                       f"{a.db_name}: motifs in attribution seqlets along the architecture ({len(genes)} genes)")
    save(fig, out_prefix)
    print(f"[done] {out_prefix}.png / .pdf", flush=True)


if __name__ == "__main__":
    main()
