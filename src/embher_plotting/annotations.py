"""Marks and axis tweaks that go on top of any archetype: reference lines, significance
brackets, statistic text, panel letters, legend placement, tick-label helpers."""
from __future__ import annotations

import numpy as np

from .theme import (FONT, FONT_PANEL, LINEWIDTH, MUTED, SPINE_WIDTH, TICK_LENGTH, TICK_WIDTH, INK, INK2)

DASH = (0, (4, 3))
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


# --- statistics text ------------------------------------------------------------
def format_pvalue(p: float) -> str:
    """Manuscript P value: 0.034 -> 'P = 0.034'; 0.00042 -> 'P = 4.2 × 10⁻⁴'."""
    if np.isnan(p):
        return "P = NA"
    if p >= 0.001:
        return f"P = {p:.3f}"
    if p == 0:
        return "P < 10⁻³⁰⁰"
    exponent = int(np.floor(np.log10(abs(p))))
    return f"P = {p / 10 ** exponent:.1f} × 10{str(exponent).translate(SUPERSCRIPT)}"


def correlation_label(r: float, p: float, n: int, method: str = "pearson") -> str:
    """'ρ = 0.86' / 'r = 0.86', P value and n, one per line."""
    symbol = "ρ" if method == "spearman" else "r"
    stat = f"{symbol} = NA" if np.isnan(r) else f"{symbol} = {r:.2f}"
    return f"{stat}\n{format_pvalue(p)}\nn = {n:,}"


def stat_text(ax, text, loc="upper left", fontsize=FONT["annotation"], color=INK):
    """Short statistic (r, P, n) pinned inside the axes."""
    pos = {"upper left": (0.05, 0.95, "left", "top"), "upper right": (0.95, 0.95, "right", "top"),
           "lower left": (0.05, 0.05, "left", "bottom"), "lower right": (0.95, 0.05, "right", "bottom")}
    x, y, ha, va = pos[loc]
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, color=color)


# --- reference marks --------------------------------------------------------------
def ref_line(ax, value=0.0, axis="y", color=MUTED, lw=LINEWIDTH, ls="-", label=None, zorder=1):
    """Horizontal (axis='y') or vertical (axis='x') reference line, e.g. zero or a mean."""
    fn = ax.axhline if axis == "y" else ax.axvline
    return fn(value, color=color, lw=lw, ls=ls, label=label, zorder=zorder)


def chance_line(ax, value=0.5, axis="x", color="#BDBDBD", lw=LINEWIDTH, ls="--", zorder=0):
    """Chance-performance line (AUROC = 0.5 by default; vertical, as in figutils).
    Not appropriate for AUPRC unless prevalence is 0.5."""
    return ref_line(ax, value, axis, color=color, lw=lw, ls=ls, zorder=zorder)


def ref_band(ax, lo, hi, axis="y", color=MUTED, alpha=0.15):
    """Shaded span over the whole axes width (axis='y') or height (axis='x')."""
    fn = ax.axhspan if axis == "y" else ax.axvspan
    return fn(lo, hi, color=color, alpha=alpha, lw=0, zorder=0)


def identity_line(ax, label="y = x", color=MUTED, lw=LINEWIDTH):
    """x = y diagonal across the wider of the two axis ranges. Limits are restored
    afterwards, so the line never changes the autoscaling."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
    line, = ax.plot([lo, hi], [lo, hi], color=color, lw=lw, ls=DASH, zorder=1, label=label)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return line


def bracket(ax, x1, x2, text, y=None, color=INK2):
    """Significance bracket between two x positions, `text` (e.g. a P value) above it.
    `y` defaults to just below the top of the axes; the y-limit is raised to fit."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    y = hi - 0.1 * span if y is None else y
    tick = 0.02 * span
    ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color=color, lw=LINEWIDTH, clip_on=False)
    ax.text((x1 + x2) / 2, y + tick * 0.5, text, ha="center", va="bottom",
            fontsize=FONT["annotation"], color=INK)
    ax.set_ylim(top=max(hi, y + 6 * tick))


# --- panel letters ------------------------------------------------------------------
def panel_label(ax, label, x=-0.12, y=1.05, fontsize=FONT_PANEL):
    """Bold panel label ('a', 'b', ...) in axes coordinates; adjust x, y per panel."""
    ax._embher_letter = label        # lets the information page name this panel
    return ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
                   fontsize=fontsize, fontweight="bold", clip_on=False)


def panel_letters(axes, start="a", **kw):
    """Label every visible axes in order: a, b, c, ..."""
    for i, ax in enumerate(a for a in np.ravel(axes) if a.axison):
        panel_label(ax, chr(ord(start) + i), **kw)


# --- axis helpers ---------------------------------------------------------------------
def clean_axis(ax, *, top=False, right=False, left=True, bottom=True):
    """Standardise spine visibility, spine linewidth and tick geometry."""
    for name, shown in (("top", top), ("right", right), ("left", left), ("bottom", bottom)):
        ax.spines[name].set_visible(shown)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    ax.tick_params(width=TICK_WIDTH, length=TICK_LENGTH, labelsize=FONT["tick"])
    return ax


def legend_out(ax, title="", **kwargs):
    """Legend of the artists drawn on `ax`, to the right of the axes."""
    kwargs.setdefault("bbox_to_anchor", (1.02, 0.5))
    return ax.legend(title=title, frameon=False, loc="center left", **kwargs)


def rotate_xticklabels(ax, angle=45):
    for label in ax.get_xticklabels():
        label.set_rotation(angle)
        label.set_ha("right")


def color_yticklabels(ax, colors):
    """Colour the y tick labels of `ax`, in order, with `colors`."""
    for tick, color in zip(ax.get_yticklabels(), colors):
        tick.set_color(color)


def share_axis(axes, axis="y", hide_labels=True):
    """Give the axes the same x or y limits (the union); with `hide_labels`, drop the tick
    labels of all but the first. Keeps an inverted first axis inverted."""
    axes = list(axes)
    getlim = (lambda a: a.get_ylim()) if axis == "y" else (lambda a: a.get_xlim())
    setlim = (lambda a, lim: a.set_ylim(lim)) if axis == "y" else (lambda a, lim: a.set_xlim(lim))
    lims = [getlim(a) for a in axes]
    lo, hi = min(min(l) for l in lims), max(max(l) for l in lims)
    inverted = getlim(axes[0])[0] > getlim(axes[0])[1]
    for a in axes:
        setlim(a, (hi, lo) if inverted else (lo, hi))
    if hide_labels:
        for a in axes[1:]:
            a.tick_params(**{"label" + ("left" if axis == "y" else "bottom"): False})


def stat_cor(g, regplot=True, correlation="spearman", annotation_xy=(0.05, 0.95)):
    """Annotate every facet of a seaborn FacetGrid with its correlation (rho or r, P, n),
    plus an optional dashed black linear trend line. Needs seaborn."""
    import seaborn as sns
    from scipy import stats

    correlation = correlation.lower()
    if correlation not in {"spearman", "pearson"}:
        raise ValueError("correlation must be either 'spearman' or 'pearson'")
    x, y, row_var, col_var = g._x_var, g._y_var, g._row_var, g._col_var

    for key, ax in g.axes_dict.items():
        if row_var and col_var:
            row_val, col_val = key
            subset = g.data[(g.data[row_var] == row_val) & (g.data[col_var] == col_val)]
        elif col_var:
            subset = g.data[g.data[col_var] == key]
        elif row_var:
            subset = g.data[g.data[row_var] == key]
        else:
            subset = g.data
        subset = subset.dropna(subset=[x, y])
        if len(subset) < 2:
            continue
        if regplot:
            sns.regplot(data=subset, x=x, y=y, scatter=False, ax=ax, ci=None,
                        line_kws={"linestyle": "--", "color": "black", "linewidth": LINEWIDTH})
        try:
            fn = stats.spearmanr if correlation == "spearman" else stats.pearsonr
            r, p = fn(subset[x], subset[y])
        except Exception:
            r, p = np.nan, np.nan
        ax.text(*annotation_xy, correlation_label(r, p, len(subset), correlation),
                transform=ax.transAxes, ha="left", va="top", fontsize=FONT["annotation"])
