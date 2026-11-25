#!/usr/bin/env python
"""fig_s6: violin comparison (embher_plotting archetypes) of HE-regression
heritability across four output quantities for the same hi_q_random500 cohort --
  genotype            workflow 10's genotype-GRM h2 (he_mac_ld_6_covar)
  embedding            Borzoi's final-layer embedding h2 (embeddings_final, position
                       axis -- per-channel embedding)
  all RNA-seq preds    Borzoi's RNA-seq prediction head h2, all tracks pooled
                       (rna_predictions)
  whole blood          same prediction head, restricted to the whole-blood tracks
  predictions          alone (rna_predictions_whole_blood)

One violin per group, coloured with the palette's "genotype" colour (genotype),
"embedding" colour (embedding, all heads) or "prediction" colour (whole blood
predictions). Genotype is restricted to the same
genes the Borzoi violins use (the embedding-final-layer file list), via --gene-map's
symbol -> Ensembl join, same convention as every other genotype join in this workflow.

Reads each run's own already-published per-gene heritability parquets (workflow 12:
<gene>__embeddings_final_pos_heritability.parquet, <gene>__rna_predictions_heritability.parquet,
<gene>__rna_predictions_whole_blood_heritability.parquet; workflow 10: one parquet per
Ensembl gene, globbed by --genotype-h2-dir) -- nothing is recomputed, this only pools
the h2 column across genes.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection

HE_GENOTYPE_METHOD = "he_mac_ld_6_covar"


def load_h2(heritability_dir: str, variant: str) -> pd.Series:
    """One value per gene, indexed by gene symbol (parsed from the file's own
    <gene>__<variant>_heritability.parquet name)."""
    files = sorted(glob.glob(f"{heritability_dir}/*__{variant}_heritability.parquet"))
    if not files:
        raise SystemExit(f"[error] no {variant} heritability files under {heritability_dir}")
    rows = []
    for f in files:
        gene = Path(f).name.split(f"__{variant}_heritability.parquet")[0]
        rows.append((gene, pd.read_parquet(f)["h2"].iloc[0]))
    return pd.Series(dict(rows), name="h2").dropna()


def load_genotype(h2_dir: str, method: str, gene_map: pd.DataFrame, genes: list[str]) -> pd.Series:
    """Genotype h2 for exactly `genes` (gene symbols) -- joined to Ensembl ids via
    --gene-map, one parquet per gene (no combined summary table for this HE run), glob
    --genotype-h2-dir for each. Converged fits only."""
    symbol_to_ensg = dict(zip(gene_map["experiment"], gene_map["gene_id"].str.split(".").str[0]))
    rows, missing = {}, []
    for gene in genes:
        ensg = symbol_to_ensg.get(gene)
        if ensg is None:
            missing.append(gene)
            continue
        matches = sorted(Path(h2_dir).glob(f"{ensg}_*.parquet"))
        if not matches:
            missing.append(gene)
            continue
        df = pd.read_parquet(matches[0])
        df = df[(df["method"] == method) & (df["converged"] == True)]  # noqa: E712
        if df.empty:
            missing.append(gene)
            continue
        rows[gene] = df["h2"].iloc[0]
    if missing:
        print(f"[warn] {len(missing)} gene(s) missing/no converged {method} row in {h2_dir}: "
              f"{missing[:10]}{' ...' if len(missing) > 10 else ''}", flush=True)
    return pd.Series(rows, name="h2")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained-heritability-dir", required=True,
                     help="results/12-heritability_over_layers/hi_q_random500/heritability")
    ap.add_argument("--genotype-h2-dir", required=True,
                     help="results/10-genotype_heritability/gtex/hi_q_gtex_genes/h2/he_mac_ld_6")
    ap.add_argument("--genotype-method", default=HE_GENOTYPE_METHOD)
    ap.add_argument("--gene-map", required=True,
                     help="workflows/11-embedding_heritability/configs/hi_q_random500.tsv")
    ap.add_argument("--embher-plotting-src", required=True,
                     help="embher_analysis's src/ directory -- put on sys.path by explicit path "
                          "(not a __file__-relative lookup), so this still works when "
                          "Nextflow stages this driver alone into an isolated work directory")
    ap.add_argument("--ymin", type=float, default=-0.25, help="lower y limit (negative shows h2 < 0)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.embher_plotting_src)
    from embher_plotting import apply_style, archetypes as A, annotations as N, layout as L
    from embher_plotting.theme import P2F_COLORS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = {
        ("pretrained", "embedding"): load_h2(args.pretrained_heritability_dir, "embeddings_final_pos"),
        ("pretrained", "all heads"): load_h2(args.pretrained_heritability_dir, "rna_predictions"),
        ("pretrained", "whole blood"): load_h2(args.pretrained_heritability_dir, "rna_predictions_whole_blood"),
    }

    gene_map = pd.read_csv(args.gene_map, sep="\t", dtype=str)
    genes = sorted(series[("pretrained", "embedding")].index)
    genotype = load_genotype(args.genotype_h2_dir, args.genotype_method, gene_map, genes)

    rows = [{"model": model, "output": output, "n": len(h), "mean": h.mean(),
             "median": h.median(), "sd": h.std()} for (model, output), h in series.items()]
    rows.append({"model": None, "output": "genotype", "n": len(genotype),
                 "mean": genotype.mean(), "median": genotype.median(), "sd": genotype.std()})
    pd.DataFrame(rows).to_csv(out_dir / "fig_s6_summary.tsv", sep="\t", index=False)

    apply_style()
    fig, ax = L.panel_grid(1, ncols=1, panel=L.figsize_cm(9, 6.5))   # single-column figure
    ax = ax[0]

    # Violins carry the colour of what they measure (embedding / prediction);
    # legend entries are these roles.
    GROUPS = ["genotype", "embedding\n(final layer)", "all RNA-seq\npredictions", "whole blood\npredictions"]
    GROUP_KEYS = dict(zip(GROUPS[1:], ("embedding", "all heads", "whole blood")))
    GROUP_HUE = {"embedding\n(final layer)": "embedding", "all RNA-seq\npredictions": "embedding",
                 "whole blood\npredictions": "prediction"}
    HUES = ["genotype", "embedding", "prediction"]
    colors = {"genotype": P2F_COLORS["genotype"], "embedding": P2F_COLORS["embedding"],
              "prediction": P2F_COLORS["prediction"]}
    data = {("genotype", "genotype"): genotype.to_numpy()}
    for group, key in GROUP_KEYS.items():
        data[(group, GROUP_HUE[group])] = series[("pretrained", key)].to_numpy()

    A.grouped_violin(ax, data, GROUPS, HUES, colors=colors,
                     legend_kw=dict(loc="upper right"))
    for coll in [c for c in ax.collections if isinstance(c, LineCollection)]:
        coll.remove()                        # drop the IQR bars (ax.vlines)
    for coll in [c for c in ax.collections if isinstance(c, PolyCollection)]:
        coll.set_alpha(1.0)                  # opaque violin bodies (library default is 0.8)
    N.ref_line(ax, 0.0, "y", ls=N.DASH, color=P2F_COLORS["reference"])
    ax.set_ylim(args.ymin, 0.75)
    ax.set_ylabel("$h^2$")

    top = L.title(fig, "Borzoi heritability by output",
                  f"HE regression; genotype n={len(genotype)}, Borzoi n={len(genes)} genes (hi_q_random500)")
    fig.tight_layout(rect=(0, 0, 1, top))
    L.save(fig, out_dir / "fig_s6", formats=("pdf", "png"), png_dpi=300)

    print(f"[done] wrote {out_dir}/fig_s6_summary.tsv, "
          f"fig_s6.pdf", flush=True)


if __name__ == "__main__":
    main()
