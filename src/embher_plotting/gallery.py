"""Render one example of every archetype (synthetic data) to check the style.

    PYTHONPATH=src python -m embher_plotting.gallery [out_stem]   # default: showcase.pdf in this folder
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from . import annotations as N
from . import archetypes as A
from . import layout as L
from . import theme as T
from .theme import AQUA, BLUE, ORANGE, apply_style, cm, shades


def palette_figure():
    """Swatches of every colour the style defines: neutrals, categorical, quantity/ancestry
    (P2F) colours, and the continuous maps. Reads theme.py, so it never goes stale."""
    from matplotlib.patches import Rectangle

    groups = [
        ("neutrals", {"BG": T.BG, "INK": T.INK, "INK2": T.INK2, "MUTED": T.MUTED,
                      "GRID": T.GRID, "NEUTRAL": T.NEUTRAL}),
        ("categorical", dict(zip(["BLUE", "ORANGE", "AQUA", "PINK", "GOLD"], T.CATEGORICAL))),
        ("quantity / ancestry / track (P2F)", T.P2F_COLORS),
    ]
    sw, gap, per_row = 1.0, 0.25, 6          # swatch width, gap (swatch units), swatches per row
    rows = [(t, list(d.items())) for t, d in groups]
    n_rows = sum(-(-len(v) // per_row) for _, v in rows)
    height = 0.7 * len(rows) + 1.35 * n_rows + 2.5   # in swatch rows, 1 unit = 1 cm
    fig = matplotlib.pyplot.figure(figsize=L.figsize_cm(18, height + 0.4))
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.96))
    ax.axis("off")
    y = height
    for title_, items in rows:
        y -= 0.5
        ax.text(0, y, title_, fontsize=T.FONT_LABEL, fontweight="bold", va="center")
        y -= 0.2
        for i in range(0, len(items), per_row):
            y -= 1.0
            for j, (name, col) in enumerate(items[i:i + per_row]):
                x0 = j * (sw + gap)
                ax.add_patch(Rectangle((x0, y + 0.3), sw, 0.6, fc=col, ec=T.GRID, lw=T.LINEWIDTH))
                ax.text(x0, y + 0.2, name, fontsize=T.FONT_SMALL, va="top")
                ax.text(x0, y + 0.0, col, fontsize=T.FONT_SMALL, va="top", color=T.INK2)
            y -= 0.35
    y -= 0.5
    ax.text(0, y, "continuous", fontsize=T.FONT_LABEL, fontweight="bold", va="center")
    grad = np.linspace(0, 1, 256)[None, :]
    for name, cmap in [("sequential (Blues)", T.SEQUENTIAL),
                       ("diverging (embher_diverging)", "embher_diverging")]:
        y -= 1.0
        ax.imshow(grad, cmap=cmap, extent=(0, 3 * (sw + gap), y + 0.3, y + 0.9), aspect="auto")
        ax.text(3 * (sw + gap) + 0.15, y + 0.6, name, fontsize=T.FONT_SMALL, va="center")
    ax.set_xlim(0, per_row * (sw + gap))
    ax.set_ylim(0, height)
    return fig


def build():
    apply_style()
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 0.15, 150)
    y = x * 0.8 + rng.normal(0, 0.06, 150)
    groups = ["rare", "low", "mid", "common"]

    fig, ax = L.panel_grid(12, ncols=4, panel=(6.8 * cm, 5.4 * cm))
    A.scatter(ax[0], x, y, identity=True, zero_lines=True, stats="spearman", trend="linear")
    ax[0].set(title="scatter", xlabel="embedding $h^2$", ylabel="genotype $h^2$")

    A.violin(ax[1], [x, y, x + 0.2], labels=["a", "b", "c"], colors=[BLUE, ORANGE, AQUA])
    N.bracket(ax[1], 0, 1, "p=0.01")
    ax[1].set(title="violin", ylabel="$h^2$")

    A.grouped_violin(ax[2], {(g, h): rng.normal(i * 0.1, 0.1, 80) + j * 0.1
                             for i, g in enumerate(groups) for j, h in enumerate(["HE", "REML"])},
                     groups, ["HE", "REML"])
    N.ref_line(ax[2], 0)
    ax[2].set(title="grouped violin", ylabel="bias")

    A.box(ax[3], [rng.normal(m, 0.1, 40) for m in (0.2, 0.3, 0.5)], labels=["a", "b", "c"],
          colors=[BLUE, ORANGE, AQUA])
    ax[3].set(title="box + points", ylabel="$h^2$")

    A.bar(ax[4], groups, [0.1, 0.25, 0.4, 0.3], err=[0.03] * 4, colors=shades(BLUE, 4),
          value_labels=True)
    ax[4].set(title="bar", ylabel="$h^2$ per bin")

    A.grouped_bar(ax[5], rng.uniform(0.1, 0.5, (4, 2)), groups, ["pretrained", "random"],
                  err=np.full((4, 2), 0.03))
    ax[5].set(title="grouped bar", ylabel="$h^2$")

    layers = np.arange(10)
    m = 0.1 + 0.4 * np.sin(layers / 6)
    A.line(ax[6], layers, m, m - 0.08, m + 0.08, color=BLUE, label="pretrained")
    A.line(ax[6], layers, m * 0.4, m * 0.4 - 0.05, m * 0.4 + 0.05, color=ORANGE, label="random")
    N.ref_line(ax[6], 0.3, ls=N.DASH, label="genotype")
    ax[6].legend()
    ax[6].set(title="line + band", xlabel="layer", ylabel="median $h^2$")

    A.point_range(ax[7], groups, [0.1, 0.3, 0.5, 0.4], [0.0, 0.2, 0.4, 0.2], [0.2, 0.4, 0.6, 0.6])
    ax[7].set(title="point range", xlabel="effect")

    A.heatmap(ax[8], rng.uniform(0, 1, (4, 5)), xlabels=list("abcde"), ylabels=groups,
              cbar_label="fail %")
    ax[8].set(title="heatmap (sequential)")

    c = np.corrcoef(rng.normal(size=(5, 30)))
    A.heatmap(ax[9], c, xlabels=list("abcde"), ylabels=list("abcde"), center=0,
              cbar_label="correlation")
    ax[9].set(title="heatmap (diverging)")

    A.histogram(ax[10], x, bins=20, density=True)
    A.density(ax[10], y, color=ORANGE, fill=False)
    ax[10].set(title="histogram + density", xlabel="$h^2$")

    A.ecdf(ax[11], x, color=BLUE, label="x")
    A.ecdf(ax[11], y, color=ORANGE, label="y")
    ax[11].legend()
    ax[11].set(title="ecdf", xlabel="$h^2$", ylabel="fraction")

    top = L.title(fig, "embher plotting gallery", "synthetic data")
    fig.tight_layout(rect=(0, 0, 1, top))
    N.panel_letters(ax)

    fig2 = matplotlib.pyplot.figure(figsize=L.figsize_cm(9, 8))
    A.scatter_marginals(fig2, {"EUR": (x, y), "other": (x + 0.2, y - 0.1)})
    fig2.suptitle("scatter with marginals", x=0.01, ha="left")
    return [fig, fig2, palette_figure()]


if __name__ == "__main__":
    stem = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "showcase")
    figs = build()
    with L.PdfBook(f"{stem}.pdf") as book:
        for f in figs:
            book.add(f)
    print(f"[done] {stem}.pdf")
