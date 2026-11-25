"""Units, typography, colours and rcParams shared by every embher figure.

Typography, line widths, tick geometry, units and output settings are adapted from an
internal figure-style helper, figutils (Nature-style figures built at their final
physical size, Arial, thin axes, editable vector text). Where figutils defines a
palette that competes with a embher one, embher's is the default; figutils' remaining
palettes are kept below for reuse. Call `apply_style()` once per script.
"""
from __future__ import annotations

import colorsys
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from cycler import cycler

# --- units --------------------------------------------------------------------
cm = 1 / 2.54                      # centimetres -> matplotlib inches
NATURE_SINGLE_COLUMN_CM = 9.0
NATURE_DOUBLE_COLUMN_CM = 18.0
NATURE_MAX_HEIGHT_CM = 17.0

# --- typography / graphical constants (figutils) --------------------------------
FONT_FAMILY = "Arial"
# Arial is proprietary and not installed; Liberation Sans and Arimo (in ~/.local/share/fonts) are
# metric-identical open substitutes, Nimbus Sans a close one. `_alias_arial` registers the first
# available one under the name "Arial".
FONT_FALLBACKS = [FONT_FAMILY, "Liberation Sans", "Nimbus Sans", "Helvetica", "DejaVu Sans"]

FONT_PANEL = 8
FONT_TITLE = 7
FONT_LABEL = 7
FONT_TEXT = 6
FONT_SMALL = 5.5
FONT = {"panel": FONT_PANEL, "title": FONT_TITLE, "label": FONT_LABEL, "tick": FONT_TEXT,
        "legend": FONT_TEXT, "annotation": FONT_TEXT, "small": FONT_SMALL}

LINEWIDTH = 0.6
ERRORBAR_LINEWIDTH = 0.7
VIOLIN_EDGE_LINEWIDTH = 0.3        # outline of violin bodies (thinner than axes/box lines)
TICK_WIDTH = 0.2
TICK_LENGTH = 3.5
SPINE_WIDTH = 0.2                    # axes lines (spines)
MARKERSIZE = 3.5                   # points; scatter `s` is MARKERSIZE ** 2

CNS_FONT_SIZE = 5                  # uniform font size for cnsplots figures
PANEL_MM = 30                      # side of a square cnsplots panel (axes area), mm

DPI = 600                          # savefig dpi
PAD_INCHES = 0.02
FORMATS = ("pdf", "png")
PANEL = (4.5 * cm, 4.0 * cm)       # one panel of a grid: 4 columns = one 18 cm double column

# --- neutrals (embher) ----------------------------------------------------------
BG = "#FFFFFF"      # figure and axes background
INK = "#0b0b0b"     # text, data-bearing marks
INK2 = "#52514e"    # secondary annotations
MUTED = "#8b8a85"   # reference lines, whiskers
GRID = "#e6e5e1"    # light rules, gridlines
NEUTRAL = "#a3a29c" # "other" / background series

# --- categorical (embher): first three are the CVD-safe triple used across the project
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PINK, GOLD = "#d0549a", "#c9a227"
CATEGORICAL = [BLUE, ORANGE, AQUA, PINK, GOLD]

# P2F palette: one colour per quantity / ancestry / annotation track
P2F_COLORS = {
    "genotype": "#B85F6B",
    "embedding": "#2D6BD1",
    "prediction": "#FEBA52",
    "random": "#8E9299",
    "EUR": "#8A8635",
    "AFR": "#824691",
    "TF": "#4F0000",
    "RBP": "#B24718",
    "phyloP": "#DE8520",
    "gnomAD": "#F6BD22",
    "reference": "#8E9299",
    "grid": "#D9DDE2",
}

# --- continuous (embher) ----------------------------------------------------------
SEQUENTIAL = "Blues"
DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "embher_diverging", [ORANGE, "#ffffff", BLUE])
if "embher_diverging" not in mpl.colormaps:
    mpl.colormaps.register(DIVERGING)

# --- figutils palettes with no embher counterpart (kept as defined there) ----------
COL_GAIN, COL_LOSS, COL_HIGHLIGHT = "#D95F02", "#1B9E9A", "#6A51A3"
COL_GREY, COL_LIGHT_GREY, COL_DARK_GREY = "#BDBDBD", "#E6E6E6", "#4D4D4D"
COL_REF, COL_ALT = "#ABA194", "#8C271E"

PAL_METRIC = {"AUROC": "#2A9D8F", "AUPRC": "#94D2BD"}
PAL_MODEL = {"Tabulazoi": "#2A9D8F", "AlphaGenome": "#E76F51", "Borzoi": "#7A6F9B", "PWM": "#999999"}
PAL_GENE_SETS = {"all": "#66C2A5", "pLI": "#FC8D62", "OMIM_Disease": "#8DA0CB",
                 "OMIM_Expanded": "#E78AC3", "panelapp_all": "#A6D854",
                 "panelapp_biallelic": "#FFD92F", "panelapp_monoallelic": "#E5C494"}
PAL_ELEMENTS = {"greendb_enhancer": "#66C2A5", "greendb_promoter": "#E78AC3",
                "greendb_silencer": "#FC8D62", "greendb_bivalent": "#8DA0CB",
                "postar_rbpsite": "#A6D854"}
PAL_VARIANT_FREQ = {"rare (MAF < 1%)": "#8C271E", "common (MAF > 5%)": "#ABA194"}
PAL_REFALT = {"alt": COL_ALT, "ref": COL_REF}
PAL_BOOL = {"True": "#5995ED", "False": "lightgray"}
PAL_VARIANT_TYPE = {"substitution": "#4D9DE0", "insertion": "#E15554", "deletion": "#E1BC29"}
PAL_DATA_TYPE = {"random": "#999999", "real": "#D95F02"}
PAL_OKABE_ITO = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
                 "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
TF_PAL = {"ETS": "#976e98ff", "ETV": "#6e987dff"}
# one colour per benchmarked model; ensembles blend their members' colours
PAL_BENCH_MODELS = {
    "Borzoi_all": "#2ca02c", "Borzoi_ENCODE": "#1f77b4", "Borzoi_ENCODEembryo": "#ff7f0e",
    "Borzoi_GTEx": "#ffbb78", "Borzoi_ENCODEadult": "#aec7e8",
    "Tabulazoi_all": "#ff9896", "Tabulazoi_TabulaSapiens": "#9467bd",
    "Tabulazoi_CardosoMoreira": "#98df8a",
    "AlphaGenome_all": "#17becf", "AlphaGenome_ENCODE": "#9edae5",
    "AlphaGenome_ENCODEembryo": "#8c564b", "AlphaGenome_GTEx": "#c49c94",
    "AlphaGenome_ENCODEadult": "#e377c2", "inverse_tss_distance": "#d62728",
}


# Shared, gitignored, group-readable copy next to the project data (override: EMBHER_ARIAL_TTF)
ARIAL_TTF = Path(os.environ.get("EMBHER_ARIAL_TTF", Path(__file__).resolve().parents[2] / "data/fonts/arial.ttf"))


def _alias_arial() -> None:
    """Make the name "Arial" resolve (matplotlib-wide, so cnsplots' hard-coded Arial works).
    If a real Arial file is at ARIAL_TTF (e.g. Arial Unicode MS, regular only) it is the regular
    face and Liberation Sans (metric-identical) supplies bold/italic; otherwise Liberation Sans,
    Arimo or Nimbus Sans stand in for all faces. Nothing happens if Arial is installed."""
    import dataclasses

    from matplotlib import font_manager as fm

    entries = fm.fontManager.ttflist
    if any(f.name == FONT_FAMILY for f in entries):
        return
    have_file = ARIAL_TTF.is_file()
    if have_file:
        fm.fontManager.addfont(str(ARIAL_TTF))
        entries[-1] = dataclasses.replace(entries[-1], name=FONT_FAMILY)
    for source in ("Liberation Sans", "Arimo", "Nimbus Sans"):
        match = [f for f in entries if f.name == source]
        if have_file:
            match = [f for f in match if f.weight != 400 or f.style != "normal"]
        if match:
            entries.extend(dataclasses.replace(f, name=FONT_FAMILY) for f in match)
            break
    if hasattr(fm.fontManager, "_findfont_cached"):
        fm.fontManager._findfont_cached.cache_clear()


_alias_arial()


def _available_fonts() -> list[str]:
    """FONT_FALLBACKS that are installed, in order (avoids one findfont warning per missing font)."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    found = [f for f in FONT_FALLBACKS if f in installed]
    return found or ["DejaVu Sans"]


def apply_style() -> None:
    """Set the global rcParams (figutils settings). Font sizes are final physical sizes,
    so build figures at their intended size (see `layout.figsize_cm`)."""
    mpl.rcParams.update({
        # typography
        "font.family": _available_fonts(), "font.size": FONT_TEXT, "text.color": INK,
        # maths text (h^2 etc.) in the same family instead of matplotlib's DejaVu
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.cal": "Arial", "mathtext.sf": "Arial", "mathtext.tt": "Arial",
        "mathtext.default": "regular",
        "axes.titlesize": FONT_TITLE, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.labelsize": FONT_LABEL, "axes.labelweight": "normal",
        "xtick.labelsize": FONT_TEXT, "ytick.labelsize": FONT_TEXT,
        "legend.fontsize": FONT_TEXT, "legend.title_fontsize": FONT_TEXT,
        "legend.labelcolor": "black",
        # axes
        "axes.linewidth": SPINE_WIDTH, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "black", "axes.labelcolor": "black",
        "axes.facecolor": BG, "figure.facecolor": BG, "axes.axisbelow": True,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": LINEWIDTH,
        "axes.prop_cycle": cycler(color=CATEGORICAL), "axes.unicode_minus": True,
        # ticks
        "xtick.major.width": TICK_WIDTH, "ytick.major.width": TICK_WIDTH,
        "xtick.major.size": TICK_LENGTH, "ytick.major.size": TICK_LENGTH,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.color": "black", "ytick.color": "black",
        # lines
        "lines.linewidth": LINEWIDTH, "lines.markersize": MARKERSIZE,
        # legends
        "legend.frameon": False, "legend.borderaxespad": 0.3, "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4, "legend.labelspacing": 0.3,
        # figure / output
        "figure.dpi": 150, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "savefig.pad_inches": PAD_INCHES, "savefig.facecolor": BG,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",   # editable text
    })


def apply_cns_style(font_size: float = CNS_FONT_SIZE):
    """`apply_style()` plus figutils' cnsplots style: every text element at `font_size`
    pt, regular-weight titles, cnsplots settings aligned. Returns the cnsplots module
    (imported here, so cnsplots is only needed when this is called)."""
    import cnsplots as cns

    apply_style()
    mpl.rcParams.update({
        "font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size,
        "xtick.labelsize": font_size, "ytick.labelsize": font_size,
        "legend.fontsize": font_size, "legend.title_fontsize": font_size,
        "figure.titlesize": font_size, "axes.titleweight": "normal",
    })
    cns.settings.font_family = FONT_FAMILY
    cns.settings.font_sans_serif = (FONT_FAMILY,)
    cns.settings.title_fontsize = font_size
    cns.settings.title_fontweight = "normal"
    cns.settings.legend_fontsize = font_size
    cns.settings.legend_title_fontsize = font_size
    cns.settings.panel_label_fontname = FONT_FAMILY
    cns.settings.pdf_fonttype = 42
    cns.settings.svg_fonttype = "none"
    return cns


# --- colour helpers -------------------------------------------------------------
def tint(hex_color: str, t: float):
    """t in [0, 1]: 0 -> white, 1 -> the full colour."""
    base = np.array(mcolors.to_rgb(hex_color))
    return tuple(1.0 - t * (1.0 - base))


def lighten_color(color, amount: float = 0.6):
    """Blend `color` toward white by `amount` (0 unchanged, 1 white): derives a paler
    companion (e.g. AUPRC from AUROC) from a series' own colour."""
    return tint(color, 1.0 - amount)


def shades(hex_color: str, n: int, lo: float = 0.45, hi: float = 1.0):
    """n tints of one hue, light -> dark: an ordinal series (e.g. bin count, layer depth)."""
    ts = [hi] if n == 1 else np.linspace(lo, hi, n)
    return [tint(hex_color, t) for t in ts]


def blend_colors(colors, min_saturation: float = 0.55) -> str:
    """Mean hue (on the colour wheel) and lightness of `colors`, with a saturation floor
    so blends do not turn grey (e.g. an ensemble of models)."""
    hls = [colorsys.rgb_to_hls(*mcolors.to_rgb(c)) for c in colors]
    hue = np.angle(np.mean([np.exp(2j * np.pi * h) for h, _, _ in hls])) / (2 * np.pi) % 1
    lightness = float(np.clip(np.mean([l for _, l, _ in hls]), 0.4, 0.7))
    saturation = max(np.mean([s for _, _, s in hls]), min_saturation)
    return mcolors.to_hex(colorsys.hls_to_rgb(hue, lightness, saturation))


def colors_for(keys, palette=None) -> dict:
    """Stable key -> colour map, in the order given (cycles if there are more keys than colours)."""
    palette = palette or CATEGORICAL
    return {k: palette[i % len(palette)] for i, k in enumerate(keys)}
