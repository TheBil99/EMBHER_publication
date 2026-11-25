#!/usr/bin/env python
"""Signal the trained weights add over a random network, along the architecture, for several databases.

For each database, per layer: pretrained - random init, paired over genes. Reads the per-gene tables of the
per-database figures (summary/<db>/layers_per_gene.parquet; workflow 15):

  motif databases (cisbp_rna, hocomoco_v13)   mean enriched motifs per gene
  PhyloP track (phylop241)                    mean observed - mean expected PhyloP, pooled over genes
  gnomAD (gnomad)                             log odds ratio of unseen : seen positions in the seqlets vs
                                              the context-expected, pooled over genes

The 95% CI is a delta-method interval from the per-gene values (paired over genes), or, with --n_boot N, a
bootstrap over genes (the same resampled genes for both roles). Three figures, each 4.5 cm wide x 4 cm high, one hue per database, layers numbered 1..24 in the plotting order
(layer_order.py; the layer behind each number is in the summary table):

  <out_prefix>-normalized-<dbs>.{pdf,png}   each database's delta divided by its own maximum over the layers
                                            (0 = no signal over random, 1 = the largest gain of that database;
                                            the shapes are comparable, the magnitudes are not)
  <out_prefix>-raw-<dbs>.{pdf,png}          the deltas in their own units, one stacked panel per database
                                            (they do not share an axis)
  <out_prefix>-raw2x2-<dbs>.{pdf,png}       the same raw panels as 2 x 2 facets, same figure size

--adjusted_dbs lists motif databases whose per-gene counts are read from the seqlet-count-adjusted table instead
(summary/<db>/layers_normalized_per_gene.parquet, plot_motif_layers.py --normalize); the outputs then carry an
`_adjusted` suffix on the prefix, so the raw figures are not overwritten.

and <out_prefix>-<dbs>_summary.parquet (pretrained and random values, delta and CI, the normalizing constant).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from layer_order import LABELS  # noqa: E402

KIND = {"cisbp_rna": "motif", "hocomoco_v13": "motif", "phylop241": "phylop", "gnomad": "gnomad"}
COLORS = {"cisbp_rna": "#D55E00", "hocomoco_v13": "#0072B2", "phylop241": "#009E73", "gnomad": "#CC79A7"}   # Okabe-Ito
UNITS = {"cisbp_rna": "RBPs", "hocomoco_v13": "TFs", "phylop241": "PhyloP", "gnomad": "log OR"}
CM = 1 / 2.54


def role_values(df, kind, genes):
    """{role: {label: per-gene arrays}} for one database."""
    out = {}
    for role in ("reference", "random_init"):
        d = df[df["role"] == role]
        out[role] = {}
        for lab in LABELS:
            t = d[d["label"] == lab].set_index("gene").reindex(genes)
            if kind == "motif":
                out[role][lab] = (t["n_enriched"].fillna(0).to_numpy(float),)
            else:
                out[role][lab] = tuple(t[c].fillna(0).to_numpy(float) for c in ("observed", "expected", "n_positions"))
    return out


def stat(kind, w, arrs):
    if kind == "motif":
        return w @ arrs[0]
    o, e, n = (w @ a for a in arrs)
    if kind == "gnomad":                           # o = positions seen: log[(unseen:seen observed) / (unseen:seen expected)]
        return np.log((n - o) / o * e / (n - e))
    return (o - e) / n


def grad(kind, m):
    """d stat / d (per-gene mean arrays) at the means m (stat() is scale-free for phylop/gnomad)."""
    if kind == "motif":
        return np.array([1.0])
    o, e, n = m
    if kind == "gnomad":
        return np.array([-1 / (n - o) - 1 / o, 1 / e + 1 / (n - e), 1 / (n - o) - 1 / (n - e)])
    return np.array([1 / n, -1 / n, -(o - e) / n ** 2])


def delta_ci(kind, pre, rnd):
    """Delta-method 95% CI of stat(pretrained) - stat(random) from the per-gene values (paired over genes).

    pre / rnd: tuples of per-gene arrays. Returns (delta, ci_low, ci_high)."""
    G = len(pre[0])
    mp, mr = [a.mean() for a in pre], [a.mean() for a in rnd]
    d = stat(kind, np.full(G, 1 / G), pre) - stat(kind, np.full(G, 1 / G), rnd)
    gvec = np.concatenate([grad(kind, mp), -grad(kind, mr)])
    cov = np.atleast_2d(np.cov(np.vstack([*pre, *rnd]), ddof=1)) / G
    se = float(np.sqrt(gvec @ cov @ gvec))
    return d, d - 1.96 * se, d + 1.96 * se


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary_dir", required=True, help="<run>/summary; one folder per database")
    ap.add_argument("--dbs", default="cisbp_rna,hocomoco_v13,phylop241,gnomad")
    ap.add_argument("--out_prefix", default=None, help="default <summary_dir>/pretrained_vs_random")
    ap.add_argument("--adjusted_dbs", default="",
                    help="comma-separated motif databases read from layers_normalized_per_gene.parquet "
                         "(adjusted for the seqlet count); outputs get an `_adjusted` prefix suffix")
    ap.add_argument("--n_boot", type=int, default=None,
                    help="bootstrap draws over genes for the CI; default None = delta-method CI from the per-gene values")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    dbs = a.dbs.split(",")
    tag = "__".join(dbs)
    adjusted = [d for d in a.adjusted_dbs.split(",") if d]
    for d in adjusted:
        assert d in dbs and KIND[d] == "motif", f"--adjusted_dbs: {d} must be a motif database in --dbs"
    prefix = a.out_prefix or str(Path(a.summary_dir) / ("pretrained_vs_random_adjusted" if adjusted else "pretrained_vs_random"))

    rows = []
    for db in dbs:
        kind = KIND[db]
        df = pd.read_parquet(Path(a.summary_dir) / db / ("layers_normalized_per_gene.parquet" if db in adjusted else "layers_per_gene.parquet"))
        genes = sorted(df["gene"].unique())
        vals = role_values(df, kind, genes)
        G = len(genes)
        if a.n_boot:
            w = np.vstack([np.ones(G) / (G if kind == "motif" else 1),
                           rng.multinomial(G, np.full(G, 1 / G), size=a.n_boot) / (G if kind == "motif" else 1)])
        for i, lab in enumerate(LABELS):
            if a.n_boot:
                pre = np.array([stat(kind, wi, vals["reference"][lab]) for wi in w])
                rnd = np.array([stat(kind, wi, vals["random_init"][lab]) for wi in w])
                d = pre - rnd
                lo, hi = np.percentile(d[1:], 2.5), np.percentile(d[1:], 97.5)
                pre0, rnd0, d0 = pre[0], rnd[0], d[0]
            else:
                w0 = np.full(G, 1 / G)
                pre0, rnd0 = (stat(kind, w0, vals[r][lab]) for r in ("reference", "random_init"))
                d0, lo, hi = delta_ci(kind, vals["reference"][lab], vals["random_init"][lab])
            rows.append(dict(db=db, layer_index=i + 1, label=lab, n_genes=G, pretrained=pre0, random_init=rnd0,
                             delta=d0, ci_low=lo, ci_high=hi))
    s = pd.DataFrame(rows)
    const = s.groupby("db")["delta"].max()
    s["norm_const"] = s["db"].map(const)
    for c in ("delta", "ci_low", "ci_high"):
        s[f"{c}_norm"] = s[c] / s["norm_const"]
    s.to_parquet(f"{prefix}-{tag}_summary.parquet", compression="zstd", index=False)
    print(s.groupby("db")[["norm_const"]].first().to_string(), flush=True)

    plt.rcParams.update({"font.size": 3, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 1.5, "ytick.major.size": 1.5, "xtick.major.pad": 1.5,
                         "ytick.major.pad": 1.5, "pdf.fonttype": 42})
    x = np.arange(1, len(LABELS) + 1)
    ticks = [1, 6, 12, 18, 24]

    # normalized: one panel
    fig, ax = plt.subplots(figsize=(4.5 * CM, 4 * CM))
    for db in dbs:
        r = s[s["db"] == db].sort_values("layer_index")
        ax.fill_between(x, r["ci_low_norm"], r["ci_high_norm"], color=COLORS[db], alpha=0.25, lw=0)
        ax.plot(x, r["delta_norm"], color=COLORS[db], lw=0.8, marker="o", ms=0.8, label=db)
    ax.axhline(0, color="k", lw=0.5, ls="--", label="random init")           # delta 0 = the random network
    ax.set_xticks(ticks)
    ax.set_xlabel("layer", labelpad=1)
    ax.set_ylabel("pretrained - random\n(divided by its maximum)", labelpad=1)
    ax.legend(frameon=False, loc="lower right", handlelength=1, borderpad=0, labelspacing=0.2)
    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.17, top=0.97)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(f"{prefix}-normalized-{tag}.{ext}", **kw)
    plt.close(fig)

    # raw: one stacked panel per database
    fig, axes = plt.subplots(len(dbs), 1, figsize=(4.5 * CM, 4 * CM), sharex=True)
    for ax, db in zip(axes, dbs):
        r = s[s["db"] == db].sort_values("layer_index")
        ax.fill_between(x, r["ci_low"], r["ci_high"], color=COLORS[db], alpha=0.25, lw=0)
        ax.plot(x, r["delta"], color=COLORS[db], lw=0.8, marker="o", ms=0.8)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylabel(f"{db}\n(Δ {UNITS[db]})", labelpad=1)
        ax.set_xticks(ticks)
    axes[-1].set_xlabel("layer", labelpad=1)
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.14, top=0.98, hspace=0.25)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(f"{prefix}-raw-{tag}.{ext}", **kw)
    plt.close(fig)
    # raw, 2 x 2 facets: same data as the stacked figure, same size
    ncol = 2
    nrow = -(-len(dbs) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * CM, 4 * CM), sharex=True, squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        if k >= len(dbs):
            ax.axis("off")
            continue
        db = dbs[k]
        r = s[s["db"] == db].sort_values("layer_index")
        ax.fill_between(x, r["ci_low"], r["ci_high"], color=COLORS[db], alpha=0.25, lw=0)
        ax.plot(x, r["delta"], color=COLORS[db], lw=0.8, marker="o", ms=0.8)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(db, color=COLORS[db], pad=1.5)
        ax.set_ylabel(f"Δ {UNITS[db]}", labelpad=1)
        ax.set_xticks(ticks)
        if k // ncol == nrow - 1 or k + ncol >= len(dbs):
            ax.set_xlabel("layer", labelpad=1)
            ax.tick_params(labelbottom=True)
    fig.subplots_adjust(left=0.15, right=0.94, bottom=0.15, top=0.92, wspace=0.85, hspace=0.55)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(f"{prefix}-raw2x2-{tag}.{ext}", **kw)
    plt.close(fig)
    print(f"[done] {prefix}-{{normalized,raw,raw2x2}}-{tag}.{{pdf,png}}", flush=True)


if __name__ == "__main__":
    main()
