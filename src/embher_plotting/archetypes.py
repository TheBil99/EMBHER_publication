"""The plot archetypes. Each takes an existing `ax`, draws on it in house style and
returns the artist(s) (or stats) so a caller can still adjust. Axis labels and titles
are left to the caller. Call `theme.apply_style()` first.

Archetype          Use                                             Seen in (scripts)
-----------------  ----------------------------------------------  ------------------
scatter            paired x-vs-y, identity line, trend, r / P / n ~22 (99, 10, 11, 12)
violin             distribution per group, dodged by hue           ~13 (06, 07, 94, 99)
box                distribution per group + jittered points        ~3 (13)
bar                one value (+ error) per category                ~7 (95, 98, 150)
line               value along an ordered axis with a CI band      ~10 (13, 14, 15, 92, 99)
point_range        estimate + interval per category (forest)       ~4 (150)
heatmap            matrix of values, annotated cells + colourbar   ~3 (06, 14)
histogram          counts / density of one variable                ~6
density, ecdf      smooth density / cumulative curve               ~3
scatter_marginals  scatter with a density on each axis             1 (94)
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch
from scipy import stats as sps
from scipy.interpolate import PchipInterpolator

from . import annotations as ann
from .theme import (BG, BLUE, ERRORBAR_LINEWIDTH, FONT, INK, LINEWIDTH, MARKERSIZE, MUTED, VIOLIN_EDGE_LINEWIDTH,
                    SEQUENTIAL, SPINE_WIDTH, TICK_LENGTH, TICK_WIDTH, colors_for)


def _clean(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def _per_item(colors, n):
    if colors is None:
        return [BLUE] * n
    return [colors] * n if isinstance(colors, (str, tuple)) else list(colors)


def dodge(n_groups, n_hue, width=0.8):
    """Slot centres, per-hue offsets and per-item width for side-by-side marks.
    Returns (centres[n_groups], offsets[n_hue], item_width)."""
    item = width / n_hue
    offsets = (np.arange(n_hue) - (n_hue - 1) / 2) * item
    return np.arange(n_groups, dtype=float), offsets, item * 0.92


def legend_patches(ax, colors, edge_linewidth=LINEWIDTH, **kw):
    """Legend of filled patches with the house edge. colors: {label: colour}; kwargs go to
    `ax.legend` (e.g. loc="upper right")."""
    return ax.legend(handles=[Patch(facecolor=c, edgecolor=INK, linewidth=edge_linewidth, label=str(k))
                              for k, c in colors.items()], **kw)


def _legend(ax, hues, colors, edge_linewidth=LINEWIDTH):
    legend_patches(ax, {h: colors[h] for h in hues}, edge_linewidth=edge_linewidth,
                   loc="upper left", bbox_to_anchor=(1.01, 1.0))


# --- scatter ------------------------------------------------------------------
def lowess_line(ax, x, y, color="black", ls="--", lw=LINEWIDTH, **kwargs):
    """LOWESS trend line (the smoother seaborn's `lowess=True` uses). Needs statsmodels."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    fit = lowess(y, x, return_sorted=True)
    return ax.plot(fit[:, 0], fit[:, 1], color=color, ls=ls, lw=lw, **kwargs)


def scatter(ax, x, y, color=BLUE, label=None, s=None, alpha=None, identity=False,
            zero_lines=False, stats=None, stat_loc="upper left", trend=None,
            rasterize_over=5000):
    """Opaque points (alpha=1) with a black edge. Large clouds (> rasterize_over) get smaller,
    fainter, edgeless, rasterised markers.
    stats: None, "pearson" or "spearman" -> r / rho, P and n written inside the axes.
    trend: None, "linear" or "lowess" -> dashed black trend line.
    Returns {"r", "p", "n"} (r and p are NaN without stats)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    dense = len(x) > rasterize_over
    ax.scatter(x, y, s=s or (MARKERSIZE ** 2 / 4 if dense else MARKERSIZE ** 2), color=color,
               alpha=alpha or (0.4 if dense else 1.0),
               edgecolors="none" if dense else INK, linewidths=0 if dense else 0.3,
               rasterized=dense, zorder=3, label=label)
    if zero_lines:
        ann.ref_line(ax, 0, "y")
        ann.ref_line(ax, 0, "x")
    if identity:
        ann.identity_line(ax)
    if trend == "linear" and len(x) >= 2:
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color="black", ls="--", lw=LINEWIDTH, zorder=2)
    elif trend == "lowess" and len(x) >= 2:
        lowess_line(ax, x, y, zorder=2)
    out = {"r": np.nan, "p": np.nan, "n": len(x)}
    if stats and len(x) >= 3:
        r, p = {"pearson": sps.pearsonr, "spearman": sps.spearmanr}[stats](x, y)
        out.update(r=float(r), p=float(p))
        ann.stat_text(ax, ann.correlation_label(r, p, len(x), stats), stat_loc)
    return out


# --- violin -------------------------------------------------------------------
def violin(ax, data, positions=None, width=0.8, colors=None, labels=None, show_mean=False):
    """One violin per array in `data`, with an IQR bar and median dot.
    NaNs are dropped; a group with < 2 points or zero spread (no KDE possible) is drawn
    as a short horizontal tick instead of vanishing. `colors`: one colour or one per group."""
    data = [_clean(d) for d in data]
    n = len(data)
    positions = np.arange(n, dtype=float) if positions is None else np.asarray(positions, float)
    colors = _per_item(colors, n)
    ok = [i for i, d in enumerate(data) if len(d) >= 2 and np.ptp(d) > 0]
    if ok:
        parts = ax.violinplot([data[i] for i in ok], positions=positions[ok], widths=width,
                              showextrema=False)
        for body, i in zip(parts["bodies"], ok):
            body.set_facecolor(colors[i])
            body.set_edgecolor(INK)
            body.set_linewidth(VIOLIN_EDGE_LINEWIDTH)
            body.set_alpha(0.8)
    for i, d in enumerate(data):
        if not len(d):
            continue
        if i in ok:
            q1, q3 = np.percentile(d, [25, 75])
            ax.vlines(positions[i], q1, q3, color=INK, lw=1.6, zorder=3)
            ax.plot(positions[i], np.median(d), "o", ms=2.2, color=BG, zorder=4)
            if show_mean:
                ax.plot(positions[i], d.mean(), "D", ms=2.8, color=INK, zorder=4)
        else:
            ax.plot([positions[i] - width / 4, positions[i] + width / 4], [d[0], d[0]],
                    color=INK, lw=1.0, zorder=4)
    if labels is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    return positions


def grouped_violin(ax, data, groups, hues, colors=None, width=0.8, show_mean=False,
                   legend=True, legend_kw=None):
    """Violins dodged by hue within each group. data[(group, hue)] -> array. A group that
    has only some of the hues dodges just those, centred on its slot (so a single-violin
    group sits in the middle); every violin has the same width. Adds the hue legend
    (outside the axes by default; `legend_kw` goes to `ax.legend`). Returns the group centres."""
    colors = colors or colors_for(hues)
    centres = np.arange(len(groups), dtype=float)
    present = [[h for h in hues if (g, h) in data] for g in groups]
    n_max = max((len(p) for p in present), default=1)
    step = width / n_max
    for h in hues:
        idx = [(i, p.index(h) - (len(p) - 1) / 2) for i, p in enumerate(present) if h in p]
        if idx:
            violin(ax, [data[(groups[i], h)] for i, _ in idx],
                   [centres[i] + k * step for i, k in idx], step * 0.92, colors[h],
                   show_mean=show_mean)
    ax.set_xticks(centres)
    ax.set_xticklabels(groups)
    if legend:
        used = [h for h in hues if any(h in p for p in present)]
        if legend_kw is None:
            _legend(ax, used, colors, edge_linewidth=VIOLIN_EDGE_LINEWIDTH)
        else:
            legend_patches(ax, {h: colors[h] for h in used},
                           edge_linewidth=VIOLIN_EDGE_LINEWIDTH, **legend_kw)
    return centres


# --- box ----------------------------------------------------------------------
def box(ax, data, positions=None, width=0.6, colors=None, labels=None, jitter=True, seed=0):
    """Box (median, IQR, 1.5 IQR whiskers) per array, over jittered raw points when
    `jitter` is set. Outlier markers are off; the points show every observation."""
    data = [_clean(d) for d in data]
    n = len(data)
    positions = np.arange(n, dtype=float) if positions is None else np.asarray(positions, float)
    colors = _per_item(colors, n)
    bp = ax.boxplot(data, positions=positions, widths=width, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=MUTED, lw=LINEWIDTH),
                    capprops=dict(color=MUTED, lw=LINEWIDTH), boxprops=dict(edgecolor=INK, lw=LINEWIDTH))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    if jitter:
        rng = np.random.default_rng(seed)
        for p, d in zip(positions, data):
            ax.scatter(p + rng.uniform(-width / 4, width / 4, len(d)), d, s=2, color=INK,
                       alpha=0.25, linewidths=0, zorder=3)
    if labels is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    return bp


# --- bar ----------------------------------------------------------------------
def bar(ax, labels, values, err=None, colors=None, horizontal=False, width=0.7, ref=None,
        value_labels=False, fmt="{:.2f}"):
    """One bar per category with optional `err` (symmetric, or a (2, n) lower/upper pair),
    a zero line, optional value text at the tip, and `ref`: value for a dashed reference line."""
    values = np.asarray(values, float)
    pos = np.arange(len(values))
    kw = dict(color=_per_item(colors, len(values)), edgecolor="none",
              error_kw=dict(elinewidth=SPINE_WIDTH, ecolor=INK, capsize=2, capthick=SPINE_WIDTH))
    err = None if err is None else np.asarray(err, float)
    axis = "x" if horizontal else "y"
    if horizontal:
        bars = ax.barh(pos, values, height=width, xerr=err, **kw)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    else:
        bars = ax.bar(pos, values, width=width, yerr=err, **kw)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
    ann.ref_line(ax, 0, axis)
    if ref is not None:
        ann.ref_line(ax, ref, axis, ls=ann.DASH)
    if value_labels:
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], fontsize=FONT["small"],
                     color=INK, padding=2)
    return bars


def grouped_bar(ax, values, groups, hues, err=None, colors=None, width=0.8):
    """values[n_groups, n_hue] bars dodged by hue; err the same shape or None. Adds the hue legend."""
    values = np.asarray(values, float)
    colors = colors or colors_for(hues)
    centres, offs, item = dodge(len(groups), len(hues), width)
    for j, h in enumerate(hues):
        ax.bar(centres + offs[j], values[:, j], width=item, color=colors[h], edgecolor="none",
               yerr=None if err is None else np.asarray(err, float)[:, j],
               error_kw=dict(elinewidth=SPINE_WIDTH, ecolor=INK, capsize=2, capthick=SPINE_WIDTH))
    ax.set_xticks(centres)
    ax.set_xticklabels(groups)
    ann.ref_line(ax, 0, "y")
    _legend(ax, hues, colors)
    return centres


# --- line ---------------------------------------------------------------------
def smooth_band(x, y, points_per_segment=8):
    """Shape-preserving (PCHIP) oversampling of a band edge: rounds the corners of a
    fill_between without overshooting the original values. Needs strictly increasing x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    fine = np.linspace(x[0], x[-1], (len(x) - 1) * points_per_segment + 1)
    return fine, PchipInterpolator(x, y)(fine)


def line(ax, x, y, lo=None, hi=None, color=BLUE, label=None, marker="o", ls="-",
         smooth=True, band_alpha=0.2):
    """y along an ordered x, with an optional lo/hi band (CI, IQR). The band is smoothed
    when x is strictly increasing; the line and markers never are."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = ax.plot(x, y, color=color, ls=ls, marker=marker or "", ms=MARKERSIZE, label=label, zorder=3)
    if lo is not None and hi is not None:
        bx, lo, hi = x, np.asarray(lo, float), np.asarray(hi, float)
        if smooth and len(x) >= 3 and np.all(np.diff(x) > 0):
            bx, lo = smooth_band(x, lo)
            _, hi = smooth_band(x, hi)
        ax.fill_between(bx, lo, hi, color=color, alpha=band_alpha, lw=0, zorder=2)
    return out


# --- point + interval ---------------------------------------------------------
def point_range(ax, labels, est, lo, hi, color=BLUE, horizontal=True, offset=0.0, label=None):
    """Estimate with its interval per category (forest plot). `offset` dodges one series
    against another drawn on the same axes."""
    est, lo, hi = (np.asarray(a, float) for a in (est, lo, hi))
    idx = np.arange(len(est))
    err = [est - lo, hi - est]
    kw = dict(fmt="o", color=color, ecolor=color, ms=MARKERSIZE, mec=INK, mew=0.3,
              elinewidth=ERRORBAR_LINEWIDTH, capsize=2, capthick=ERRORBAR_LINEWIDTH,
              label=label, zorder=3)
    if horizontal:
        ax.errorbar(est, idx + offset, xerr=err, **kw)
        ax.set_yticks(idx)
        ax.set_yticklabels(labels)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
    else:
        ax.errorbar(idx + offset, est, yerr=err, **kw)
        ax.set_xticks(idx)
        ax.set_xticklabels(labels)


# --- heatmap ------------------------------------------------------------------
def _is_dark(rgba):
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45


def heatmap(ax, matrix, xlabels=None, ylabels=None, cmap=SEQUENTIAL, vmin=None, vmax=None,
            center=None, annotate=True, fmt="{:.2f}", cbar_label=None, aspect="auto"):
    """Matrix as coloured cells. center=<value> uses the diverging map, symmetric about it.
    Cell text turns white on dark cells. Returns the image."""
    m = np.asarray(matrix, float)
    if center is not None:
        cmap = "embher_diverging" if cmap == SEQUENTIAL else cmap
        half = (max(abs(np.nanmin(m) - center), abs(np.nanmax(m) - center))
                if vmax is None else vmax - center)
        vmin, vmax = center - half, center + half
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if xlabels is not None:
        ax.set_xticks(range(m.shape[1]))
        ax.set_xticklabels(xlabels)
    if ylabels is not None:
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels(ylabels)
    if annotate:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                            fontsize=FONT["small"],
                            color="white" if _is_dark(im.cmap(im.norm(m[i, j]))) else INK)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.8, pad=0.02, aspect=25)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK, labelsize=FONT["small"], length=TICK_LENGTH * 0.8, width=TICK_WIDTH)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=FONT["label"], color=INK)
    return im


# --- distributions ------------------------------------------------------------
def histogram(ax, values, bins=30, color=BLUE, density=False, label=None, alpha=0.8):
    return ax.hist(_clean(values), bins=bins, density=density, color=color, alpha=alpha,
                   edgecolor=BG, linewidth=LINEWIDTH, label=label)


def density(ax, values, color=BLUE, label=None, fill=True, vertical=False):
    """Gaussian KDE curve, optionally filled. vertical=True puts the density on x
    (for a right-hand marginal)."""
    v = _clean(values)
    g = np.linspace(v.min(), v.max(), 200)
    d = sps.gaussian_kde(v)(g)
    ax.plot(*((d, g) if vertical else (g, d)), color=color, label=label)
    if fill:
        if vertical:
            ax.fill_betweenx(g, 0, d, color=color, alpha=0.25, lw=0)
        else:
            ax.fill_between(g, 0, d, color=color, alpha=0.25, lw=0)


def ecdf(ax, values, color=BLUE, label=None):
    v = np.sort(_clean(values))
    return ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, label=label)


# --- composite ----------------------------------------------------------------
def scatter_marginals(fig, groups, colors=None, subplot_spec=None, ratio=4):
    """Scatter with a density strip on top and on the right. groups: {name: (x, y)}.
    Builds its own axes in `fig` (or inside `subplot_spec`). Returns (main, top, right)."""
    colors = colors or colors_for(list(groups))
    gs = (fig.add_gridspec(2, 2, width_ratios=[ratio, 1], height_ratios=[1, ratio],
                           hspace=0.05, wspace=0.05) if subplot_spec is None else
          subplot_spec.subgridspec(2, 2, width_ratios=[ratio, 1], height_ratios=[1, ratio],
                                   hspace=0.05, wspace=0.05))
    main = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=main)
    right = fig.add_subplot(gs[1, 1], sharey=main)
    for name, (x, y) in groups.items():
        scatter(main, x, y, color=colors[name], label=name)
        density(top, x, colors[name])
        density(right, y, colors[name], vertical=True)
    top.axis("off")
    right.axis("off")
    main.legend()
    return main, top, right
