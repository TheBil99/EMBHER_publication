"""Figure sizing, grids, saving and style checks.

Figures are built at their final physical size (figutils): use `figsize_cm`,
`single_column_figsize` / `double_column_figsize`, or `panel_grid` (panel sizes in cm).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .theme import (CNS_FONT_SIZE, DPI, FONT, FONT_FAMILY, FORMATS, INK, INK2, NATURE_DOUBLE_COLUMN_CM,
                    NATURE_SINGLE_COLUMN_CM, PAD_INCHES, PANEL, PANEL_MM, cm)


# --- sizes ----------------------------------------------------------------------
def figsize_cm(width_cm: float, height_cm: float) -> tuple[float, float]:
    """Centimetres -> matplotlib inches: `plt.subplots(figsize=figsize_cm(8.5, 6))`."""
    return width_cm * cm, height_cm * cm


def single_column_figsize(height_cm: float):
    """Nature single-column width (90 mm)."""
    return figsize_cm(NATURE_SINGLE_COLUMN_CM, height_cm)


def double_column_figsize(height_cm: float):
    """Nature double-column width (180 mm)."""
    return figsize_cm(NATURE_DOUBLE_COLUMN_CM, height_cm)


# --- grids ----------------------------------------------------------------------
def panel_grid(n, ncols=4, panel=PANEL, **kw):
    """Grid sized from the panel count (`panel` = (width, height) in inches; default 4.5 x 4 cm,
    so 4 columns fill a double column). Spare cells are hidden. Returns (fig, list of n axes)."""
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
                             squeeze=False, **kw)
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


def title(fig, text, subtitle=None):
    """Record a figure title (and optional subtitle). Nothing is drawn: figures are always
    clean, and the text is written to the information page that `save` / `PdfBook` add after
    the figure pages. Returns 1.0, the top of the free area (so `fig.tight_layout(rect=(0, 0, 1,
    top))` keeps working unchanged)."""
    info = fig.__dict__.setdefault("_embher_info", {})
    info["title"], info["subtitle"] = text, subtitle or ""
    return 1.0


def strip_titles(fig) -> dict:
    """Remove every title from `fig` -- axes titles (left, centre, right), a `suptitle`, and
    whatever `title()` recorded -- and return them as {"title", "subtitle", "panels"}.
    `panels` is a list of (label, text): the panel letter if the axes has one, else its
    position. Annotations, legends, panel letters and stat text are left alone."""
    info = dict(fig.__dict__.get("_embher_info", {}))
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text().strip():
        info.setdefault("title", suptitle.get_text())
        suptitle.set_text("")
    panels = []
    for i, ax in enumerate(fig.axes, start=1):
        texts = [t for t in (ax.get_title("left"), ax.get_title("center"), ax.get_title("right")) if t.strip()]
        if texts:
            panels.append((getattr(ax, "_embher_letter", f"axes {i}"), " | ".join(texts)))
            for loc in ("left", "center", "right"):
                ax.set_title("", loc=loc)
    return {"title": info.get("title", ""), "subtitle": info.get("subtitle", ""), "panels": panels}


def _has_info(info) -> bool:
    return bool(info["title"] or info["subtitle"] or info["panels"])


def info_figure(entries):
    """Text-only page listing, for each figure, its title, subtitle and panel titles.
    entries: list of (heading, info) with info as returned by `strip_titles`."""
    width_cm, wrap = 18.0, 100
    lines = [("Figure information", "head")]
    for heading, info in entries:
        lines.append(("", "gap"))
        lines.append((heading, "bold"))
        if info["title"]:
            lines += [(l, "text") for l in textwrap.wrap(f"Title: {info['title']}", wrap)]
        if info["subtitle"]:
            lines += [(l, "text") for l in textwrap.wrap(f"Subtitle: {info['subtitle']}", wrap)]
        for label, text in info["panels"]:
            lines += [("    " + l, "text") for l in textwrap.wrap(f"{label}: {text}", wrap - 4)]
    line_pt = FONT["tick"] * 1.5
    height_cm = max(4.0, (len(lines) * line_pt + 40) / 72 / cm)
    fig = plt.figure(figsize=figsize_cm(width_cm, height_cm))
    y = 1 - 20 / (height_cm * cm * 72)
    for text, kind in lines:
        fig.text(0.04, y, text, ha="left", va="top", color=INK,
                 fontsize=FONT["title"] if kind == "head" else FONT["tick"],
                 fontweight="bold" if kind in ("head", "bold") else "normal")
        y -= line_pt / (height_cm * cm * 72)
    return fig


def facet_panels(n_rows, n_cols, width=PANEL_MM, height=PANEL_MM, max_width=None,
                 margin_right=14, margin_bottom=16, pad_left=None, last_margin_right=None,
                 **panel_kwargs):
    """Grid of equally sized panels (axes area in mm) in one cnsplots figure. The axes keep
    `width` x `height` mm; only the margins change the overall size. `pad_left` (mm) reserves
    room for long y tick labels on the first column, `last_margin_right` (mm) a wider margin
    after the last column (a legend). Needs cnsplots. Returns (multipanel, axes[n_rows][n_cols])."""
    import cnsplots as cns

    if max_width is None:
        max_width = n_cols * (width + 18) + (last_margin_right - margin_right if last_margin_right else 0)
    mp = cns.multipanel(max_width=max_width, unit="mm")
    labels = [[chr(65 + i * n_cols + j) for j in range(n_cols)] for i in range(n_rows)]
    axes = [[None] * n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i][j] = mp.panel(
                labels[i][j], width=width, height=height, unit="mm",
                below=labels[i - 1][j] if i > 0 else None,
                margin_right=last_margin_right if (last_margin_right and j == n_cols - 1) else margin_right,
                margin_bottom=margin_bottom,
                **({"pad_left": pad_left} if pad_left is not None and j == 0 else {}),
                **panel_kwargs)
    return mp, axes


# --- saving ---------------------------------------------------------------------
def rasterize_points(fig):
    """Rasterise every scatter/strip point collection in `fig` so PDF/SVG stay small; axes,
    text, lines and bands stay vector."""
    from matplotlib.collections import PathCollection

    for ax in fig.axes:
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                coll.set_rasterized(True)


def save(fig, stem, formats=FORMATS, dpi=DPI, transparent=False, rasterize=False,
         info_page=True, png_dpi=None):
    """Write `<stem>.<fmt>` for each format (tight bbox, 0.02 in pad), creating the folder,
    then close the figure. The figure is saved without titles (see `strip_titles`); a PDF gets a
    second page with its title, subtitle and panel titles unless `info_page=False` or there are
    none. PDF/SVG are preferred for line art; `rasterize=True` rasterises points. `png_dpi` sets
    the resolution of the PNG alone (default `dpi`), so a PDF's rasterised points keep `dpi`."""
    from matplotlib.backends.backend_pdf import PdfPages

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    info = strip_titles(fig)
    if rasterize:
        rasterize_points(fig)
    for fmt in formats:
        kw = dict(bbox_inches="tight", pad_inches=PAD_INCHES, transparent=transparent)
        if fmt == "pdf":
            with PdfPages(f"{stem}.pdf") as pdf:
                pdf.savefig(fig, **kw)
                if info_page and _has_info(info):
                    page = info_figure([(stem.name, info)])
                    pdf.savefig(page, bbox_inches="tight", pad_inches=PAD_INCHES)
                    plt.close(page)
        else:
            fig.savefig(f"{stem}.{fmt}", dpi=png_dpi or dpi, **kw)
    plt.close(fig)


def audit_figure(fig):
    """Check a figure against the style rules: the set of font sizes and families used by
    every text, and each axes area in mm (width, height)."""
    fig.canvas.draw()
    texts = [t for t in fig.findobj(mpl.text.Text) if t.get_text().strip()]
    fig_w, fig_h = fig.get_size_inches() * 25.4
    axes_mm = [(round(ax.get_position().width * fig_w, 1), round(ax.get_position().height * fig_h, 1))
               for ax in fig.axes if ax.get_visible() and ax.get_position().width > 0]
    return {"font_sizes": {round(t.get_fontsize(), 2) for t in texts},
            "font_families": {t.get_fontfamily()[0] for t in texts}, "axes_mm": axes_mm}


def save_cns(panels, path):
    """cnsplots export: rasterise points, require Arial at one 5 pt size, write with cnsplots."""
    import cnsplots as cns

    fig = getattr(panels, "fig", panels)   # a cnsplots multipanel or a plain figure
    strip_titles(fig)
    rasterize_points(fig)
    audit = audit_figure(fig)
    assert audit["font_sizes"] == {float(CNS_FONT_SIZE)} and audit["font_families"] == {FONT_FAMILY}, audit
    cns.savefig(path, fig=fig)


class PdfBook:
    """Multi-page PDF: `with PdfBook(path) as book: book.add(fig)` (closes each figure).
    Figure pages are clean (titles stripped); on exit one information page is appended listing
    each page's title, subtitle and panel titles. `png_dpi`: also write the first page as
    `<path stem>.png` at that resolution (the information page never)."""

    def __init__(self, path, info_page=True, png_dpi=None):
        from matplotlib.backends.backend_pdf import PdfPages
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._pdf = PdfPages(path)
        self._path = Path(path)
        self._png_dpi = png_dpi
        self._info_page = info_page
        self._entries = []

    def add(self, fig):
        info = strip_titles(fig)
        self._entries.append((f"Page {len(self._entries) + 1}", info))
        self._pdf.savefig(fig, bbox_inches="tight", pad_inches=PAD_INCHES)
        if self._png_dpi and len(self._entries) == 1:
            fig.savefig(self._path.with_suffix(".png"), dpi=self._png_dpi,
                        bbox_inches="tight", pad_inches=PAD_INCHES)
        plt.close(fig)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        entries = [e for e in self._entries if _has_info(e[1])]
        if self._info_page and entries:
            page = info_figure(entries)
            self._pdf.savefig(page, bbox_inches="tight", pad_inches=PAD_INCHES)
            plt.close(page)
        self._pdf.close()
