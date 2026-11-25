#!/usr/bin/env python
"""fig_2: Figure 2 on an 18 cm canvas, laid out like fig_3 -- (a) genotype h2 (HE regression,
workflow 10) vs Borzoi head prediction h2 and vs embedding h2 (workflow 11), two square scatters;
(b) median h2_E / h2_G over the trunk's depth, pretrained vs random-init (workflow 12), with
bootstrap 95% CIs on the median (summarize_embedding_genotype_ratio.py's convention, imported by
path via --ratio-script). The data loaders are defined below.

Cohort of (a): results/11-embedding_heritability/gtex/hi_q_random500 (embeddings and head
predictions) joined to results/10-genotype_heritability/gtex/hi_q_gtex_genes (genotype) by an inner
join on the embedding/prediction subset; converged fits only, h2 < 0 kept. Workflow 11 is keyed by
gene SYMBOL, workflow 10 by Ensembl id (--gene-map is the join).

Same conventions as fig_3: one canvas from nested gridspecs, a draw_x(fig, spec, ...) function per
panel, panel letters, no titles (they go to the information page), square axes of AX cm, small line
markers, scatter dots rasterised. Writes fig_2.{pdf,png} into --out-dir.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

W, H = 18.0, 5.6   # cm: width, row height (a left, b right, one row)
AX = 3.8           # cm: side of every square axes (as in the 12 cm a-above-b version)
WSPACE_A = 0.7     # gap between the two scatters, in axes widths
TICKS = [1, 6, 12, 18, 24]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- data loaders (formerly plot_fig_2a.py / plot_fig_2b.py) ---------------------------
EMBEDDING_COL = "EREML"   # across_genes.parquet's own column name, regardless of estimator
HE_GENOTYPE_METHOD = "he_mac_ld_6_covar"


def load_genotype(h2_dir: str, method: str, gene_map: pd.DataFrame) -> pd.DataFrame:
    """One parquet per gene (no combined summary table for this HE run) -- glob
    --genotype-h2-dir for each gene_map Ensembl id rather than reading one file."""
    ensg_list = gene_map["gene_id"].str.split(".").str[0].tolist()
    rows, missing = [], []
    for ensg in ensg_list:
        matches = sorted(Path(h2_dir).glob(f"{ensg}_*.parquet"))
        if not matches:
            missing.append(ensg)
            continue
        df = pd.read_parquet(matches[0])
        df = df[df["method"] == method]
        if df.empty:
            missing.append(ensg)
            continue
        rows.append(df.iloc[0])
    if missing:
        print(f"[warn] {len(missing)} gene(s) missing/no {method} row in {h2_dir}: "
              f"{missing[:10]}{' ...' if len(missing) > 10 else ''}", flush=True)
    genotype = pd.DataFrame(rows).set_index("experiment")
    keep = genotype["converged"] == True  # noqa: E712
    dropped = genotype.index[~keep].tolist()
    if dropped:
        print(f"[warn] dropping {len(dropped)} gene(s) with a non-converged "
              f"{method} fit: {dropped}", flush=True)
    return genotype.loc[keep, ["h2"]].rename(columns={"h2": "h2_genotype"})


def load_prediction_h2(prediction_heritability_dir: str) -> pd.DataFrame:
    """One parquet per gene, keyed by gene SYMBOL directly (its own 'experiment'
    column) -- no gene-map join needed on this side."""
    files = sorted(Path(prediction_heritability_dir).glob("*_prediction_heritability.parquet"))
    if not files:
        raise ValueError(f"no *_prediction_heritability.parquet files in {prediction_heritability_dir}")
    rows = [pd.read_parquet(f).iloc[0] for f in files]
    pred = pd.DataFrame(rows).set_index("experiment")
    keep = pred["converged"] == True  # noqa: E712
    dropped = pred.index[~keep].tolist()
    if dropped:
        print(f"[warn] dropping {len(dropped)} gene(s) with a non-converged "
              f"prediction h2 fit: {dropped}", flush=True)
    return pred.loc[keep, ["h2"]].rename(columns={"h2": "h2_prediction"})


def build_cohort(embedding_parquet: str, gene_map_tsv: str, genotype: pd.DataFrame,
                 prediction_h2: pd.DataFrame):
    """Returns (df_genotype_embedding, df_genotype_prediction): genotype h2 (Ensembl-
    keyed) joined, via the gene map's symbol->ensg lookup, against embedding h2 and
    prediction h2 (both gene-symbol-keyed)."""
    embedding = pd.read_parquet(embedding_parquet)[[EMBEDDING_COL]]
    embedding = embedding.rename(columns={EMBEDDING_COL: "h2_embedding"})

    gene_map = pd.read_csv(gene_map_tsv, sep="\t", dtype=str)
    symbol_to_ensg = dict(zip(gene_map["experiment"], gene_map["gene_id"].str.split(".").str[0]))

    embedding_ensg = embedding.copy()
    embedding_ensg.index = embedding_ensg.index.map(symbol_to_ensg)
    unmapped = embedding_ensg.index.isna().sum()
    if unmapped:
        print(f"[warn] {unmapped} embedding experiment(s) not found in gene-map, dropped",
              flush=True)
        embedding_ensg = embedding_ensg[embedding_ensg.index.notna()]
    df_genotype_embedding = pd.concat(
        [embedding_ensg["h2_embedding"], genotype["h2_genotype"]], axis=1,
        join="inner").dropna(subset=["h2_embedding", "h2_genotype"])

    pred_h2_ensg = prediction_h2.copy()
    pred_h2_ensg.index = pred_h2_ensg.index.map(symbol_to_ensg)
    pred_h2_ensg = pred_h2_ensg[pred_h2_ensg.index.notna()]
    df_genotype_prediction = pd.concat(
        [pred_h2_ensg["h2_prediction"], genotype["h2_genotype"]], axis=1,
        join="inner").dropna(subset=["h2_prediction", "h2_genotype"])

    return df_genotype_embedding, df_genotype_prediction


AXIS_SUFFIX = "_pos"


def load_embedding_h2(run_dir: str) -> pd.DataFrame:
    """One row per (gene, layer): the real-phenotype h2 of every
    <gene>__<layer>_pos_heritability.parquet under <run_dir>/heritability. Output-head and
    channel-axis files do not match the pattern and are ignored."""
    files = sorted(Path(run_dir, "heritability").glob(f"*__*{AXIS_SUFFIX}_heritability.parquet"))
    if not files:
        raise SystemExit(f"[error] no *{AXIS_SUFFIX}_heritability.parquet files under {run_dir}/heritability")
    rows = []
    for f in files:
        gene, variant = f.name[: -len("_heritability.parquet")].split("__", 1)
        df = pd.read_parquet(f, columns=["phenotype", "h2"])
        df = df[df["phenotype"] == "pheno_expression"]
        if df.empty:
            continue
        rows.append((gene, variant[: -len(AXIS_SUFFIX)], float(df["h2"].iloc[0])))
    return pd.DataFrame(rows, columns=["gene", "layer", "h2"])


def load_genotype_h2(h2_dir: str, method: str, gene_map: pd.DataFrame, genes: list[str]) -> pd.Series:
    """Genotype h2 (gene symbol -> value) of converged `method` fits; <ENSG>_*.parquet per gene."""
    symbol_to_ensg = dict(zip(gene_map["experiment"], gene_map["gene_id"].str.split(".").str[0]))
    out, missing = {}, []
    for gene in genes:
        ensg = symbol_to_ensg.get(gene)
        matches = sorted(Path(h2_dir).glob(f"{ensg}_*.parquet")) if ensg else []
        if not matches:
            missing.append(gene)
            continue
        df = pd.read_parquet(matches[0])
        df = df[(df["method"] == method) & (df["converged"] == True)]  # noqa: E712
        if df.empty:
            missing.append(gene)
            continue
        out[gene] = float(df["h2"].iloc[0])
    if missing:
        print(f"[warn] {len(missing)} gene(s) without a converged {method} fit in {h2_dir}: "
              f"{missing[:10]}{' ...' if len(missing) > 10 else ''}", flush=True)
    return pd.Series(out, name="h2_genotype")


def ratio_curve(emb: pd.DataFrame, genotype: pd.Series, layers: list[str], ratio_mod) -> pd.DataFrame:
    """Median ratio and bootstrap 95% CI per layer (NaN where a layer has no gene)."""
    rng = np.random.default_rng(ratio_mod.BOOTSTRAP_SEED)
    emb = emb[emb["gene"].isin(genotype.index)].copy()
    emb["ratio"] = emb["h2"] / emb["gene"].map(genotype)
    rows = []
    for layer in layers:
        vals = emb.loc[emb["layer"] == layer, "ratio"].dropna().to_numpy()
        lo, hi = ratio_mod.median_bootstrap_ci(vals, rng)
        rows.append(dict(layer=layer, median=float(np.median(vals)) if len(vals) else float("nan"),
                         ci_lo=lo, ci_hi=hi, n_genes=len(vals)))
    return pd.DataFrame(rows)


# --- panels ---------------------------------------------------------------------------
def draw_a(fig, spec, df_pred, df_embed, mods, letter=False):
    """Two square scatters (genotype h2 vs prediction h2, vs embedding h2) in `spec`."""
    A, N, P2F = mods["A"], mods["N"], mods["P2F"]
    inner = spec.subgridspec(1, 2, wspace=WSPACE_A)
    panels = [(df_pred["h2_prediction"], df_pred["h2_genotype"], "$h^2_P$", P2F["prediction"]),
              (df_embed["h2_embedding"], df_embed["h2_genotype"], "$h^2_E$", P2F["embedding"])]
    axes = []
    for col, (x, y, xlabel, color) in enumerate(panels):
        ax = fig.add_subplot(inner[0, col])
        A.scatter(ax, x.to_numpy(float), y.to_numpy(float), color=color, stats="pearson")
        N.identity_line(ax, color=P2F["reference"])
        ax.set_box_aspect(1)
        ax.xaxis.set_major_locator(MultipleLocator(0.25))   # ticks at multiples of 0.25
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("$h^2_G$")
        axes.append(ax)
    if letter:
        N.panel_label(axes[0], "a", x=-0.32, y=1.05)
    return axes


STAGES = [("encoder", 1, 10), ("transformer", 11, 18), ("decoder", 19, 24)]


def _stage_bands(ax, mods):
    """Grey shade over the transformer layers and a 4 pt stage label above the axes."""
    ax.axvspan(10.5, 18.5, color=mods["NEUTRAL"], alpha=0.15, lw=0, zorder=0)
    for name, lo, hi in STAGES:
        ax.text((lo + hi) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=mods["INK2"], fontsize=4)


def draw_b(fig, spec, pre, ri, mods, letter=False):
    """Median h2_E / h2_G per layer, pretrained vs random init, in `spec`."""
    A, N, P2F = mods["A"], mods["N"], mods["P2F"]
    ax = fig.add_subplot(spec)
    _stage_bands(ax, mods)
    xs = np.arange(1, len(pre) + 1)
    N.ref_line(ax, 1.0, "y", color=P2F["reference"], ls=N.DASH)

    def curve(c, color, label):
        ok = c["median"].notna().to_numpy()
        for ln in A.line(ax, xs[ok], c["median"].to_numpy()[ok], c["ci_lo"].to_numpy()[ok],
                         c["ci_hi"].to_numpy()[ok], color=color, label=label):
            ln.set_markersize(2)

    # opaque pretrained colour (the palette entry carries its own alpha)
    curve(pre, mcolors.to_rgb(P2F["embedding"]), "pretrained")
    curve(ri, P2F["random"], "random init")
    ax.set_ylabel("$h^2_E$ / $h^2_G$")
    ax.set_xlabel("Layer")
    ax.set_xticks([t for t in TICKS if t <= len(pre)])
    ax.set_box_aspect(1)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0)
    if letter:
        N.panel_label(ax, "b", x=-0.32, y=1.05)
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding-parquet", required=True)
    ap.add_argument("--gene-map", required=True)
    ap.add_argument("--genotype-h2-dir", required=True,
                    help="results/10-genotype_heritability/gtex/hi_q_gtex_genes/h2/he_mac_ld_6")
    ap.add_argument("--genotype-method", default="he_mac_ld_6_covar")
    ap.add_argument("--prediction-heritability-dir", required=True)
    ap.add_argument("--pretrained-run-dir", required=True)
    ap.add_argument("--random-init-run-dir", required=True)
    ap.add_argument("--ratio-script", required=True)
    ap.add_argument("--embher-plotting-src", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.embher_plotting_src)
    from embher_plotting import apply_style, archetypes as A, annotations as N, layout as L
    from embher_plotting.theme import P2F_COLORS, INK2, NEUTRAL
    mods = {"A": A, "N": N, "P2F": P2F_COLORS, "INK2": INK2, "NEUTRAL": NEUTRAL}

    ratio_mod = _load_module("ratio_mod", Path(args.ratio_script))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # panel a data (plot_fig_2a.py's loaders)
    gene_map = pd.read_csv(args.gene_map, sep="\t", dtype=str)
    genotype = load_genotype(args.genotype_h2_dir, args.genotype_method, gene_map)
    pred_h2 = load_prediction_h2(args.prediction_heritability_dir)
    df_embed, df_pred = build_cohort(args.embedding_parquet, args.gene_map, genotype, pred_h2)

    # panel b data (plot_fig_2b.py's loaders)
    pretrained = load_embedding_h2(args.pretrained_run_dir)
    random_init = load_embedding_h2(args.random_init_run_dir)
    genes = sorted(set(pretrained["gene"]) | set(random_init["gene"]))
    geno_b = load_genotype_h2(Path(args.genotype_h2_dir), args.genotype_method, gene_map, genes)
    layers = sorted(set(pretrained["layer"]) | set(random_init["layer"]), key=ratio_mod.layer_sort_key)
    pre = ratio_curve(pretrained, geno_b, layers, ratio_mod)
    ri = ratio_curve(random_init, geno_b, layers, ratio_mod)

    apply_style()
    fig = plt.figure(figsize=L.figsize_cm(W, H))
    gap = (W * 0.94) - AX * (3 + WSPACE_A)      # cm left for the a-b gap (grid spans 94% of the width)
    outer = fig.add_gridspec(1, 3, width_ratios=[AX * (2 + WSPACE_A), gap, AX], wspace=0,
                             left=0.06, right=1.0)
    draw_a(fig, outer[0], df_pred, df_embed, mods, letter=True)
    draw_b(fig, outer[2], pre, ri, mods, letter=True)
    L.save(fig, out / "fig_2", rasterize=True, png_dpi=300)
    print(f"[done] n={len(df_embed)}/{len(df_pred)} genes (a), {len(layers)} layers (b) -> {out}", flush=True)


if __name__ == "__main__":
    main()
