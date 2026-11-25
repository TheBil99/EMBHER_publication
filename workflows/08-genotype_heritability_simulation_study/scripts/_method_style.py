"""Shared REML/HE visual encoding for plot_calibration.py and plot_bin_bias.py.

Two estimator families (exact REML, exact_reml_heritability.py: "exact_reml_" prefix,
"exact_reml_nonneg_" for its --nonneg fit; HE regression, he_heritability.py: "he_" prefix)
x several bin-count sub-methods (mac_maf_bins_4, mac_maf_ld_bins_6, mac_maf_ld_bins_8, ...)
is a nested categorical-x-ordinal structure: hue carries the family (categorical, the
thing that's easy to conflate at a glance), lightness carries the bin count within a
family (ordinal).

Hues are the validated palette's slots 1 (blue) and 2 (orange), a CVD-safe pair.
"""
from __future__ import annotations

import re

import matplotlib.colors as mcolors
import numpy as np

REML_HUE = "#2a78d6"    # palette slot 1 (blue)
HE_HUE = "#eb6834"      # palette slot 2 (orange)


def is_he(method):
    return method.startswith("he_")


def is_exact_reml(method):
    return method.startswith("exact_reml_")


def base_method(method):
    """Strip the family prefix so REML/HE variants of the same bin
    config share one identity for ranking/lightness purposes."""
    if is_he(method):
        return method[len("he_"):]
    if method.startswith("exact_reml_nonneg_"):   # exact REML with variance components >= 0
        return method[len("exact_reml_nonneg_"):]
    if is_exact_reml(method):
        return method[len("exact_reml_"):]
    return method


def method_nbins(method):
    """0 for single_grm, else the trailing bin count (maf_bins_2 -> 2)."""
    base = base_method(method)
    if base in ("single_grm",):
        return 0
    m = re.search(r"\d+$", base)
    return int(m.group()) if m else 0


def method_strategy(method):
    """Bin-definition label a method's true_bin_h2 rows carry, matching
    compute_true_bin_h2.py's own 'strategy' column. plot_bin_bias joins a method
    to its truth on (nbins, strategy), so this has to distinguish bin definitions
    that can produce the SAME bin count: mac_maf_bins_3 and a pure-MAF 3-bin split
    are both nbins=3 but partition the variants completely differently, and
    joining one against the other's truth would silently report nonsense bias.

    mac_maf_bins_N -> "mac_cohort", mac_maf_ld_bins_N / mac_ld_N -> "mac_ld_cohort" (workflow
    06's per-arm strategy, written by its run_true_bin_h2_mac process); everything else is a
    fixed-width MAF bin config -> "width"."""
    base = base_method(method)
    if base.startswith(("mac_maf_ld_bins", "mac_ld_")):
        return "mac_ld_cohort"
    if base.startswith("mac_maf_bins"):
        return "mac_cohort"
    return "width"


def _shade(hue_hex, t):
    """t in [0, 1]: 0 -> pale tint, 1 -> the full base hue."""
    base = np.array(mcolors.to_rgb(hue_hex))
    white = np.array([1.0, 1.0, 1.0])
    return tuple(white * (1 - t) + base * t)


def method_style(method, all_bases):
    """Return (facecolor, edge_linestyle) for one method.

    all_bases: every distinct base_method() across ALL methods being plotted
    together (both families), sorted by method_nbins -- so single_grm/
    he_single_grm land on the same lightness step, maf_bins_2/he_maf_bins_2
    the next step, etc. Keeps the two families visually parallel.
    """
    base = base_method(method)
    n = len(all_bases)
    rank = all_bases.index(base) if base in all_bases else 0
    t = 0.45 + 0.55 * (rank / max(n - 1, 1))
    if is_he(method):
        hue = HE_HUE
    else:
        hue = REML_HUE
    facecolor = _shade(hue, t)
    linestyle = "--" if is_he(method) else "-"
    return facecolor, linestyle


def sorted_bases(methods):
    return sorted({base_method(m) for m in methods}, key=method_nbins)


def draw_violin(ax, data, positions, width, facecolor, ink, muted):
    """One violin per group in `data` (a list of 1-D arrays, same length as
    `positions`), styled consistently across both plotting scripts. Shows the
    full density and the true min/max (via showextrema) -- unlike a boxplot
    with showfliers=False, nothing is ever silently clipped or hidden.

    A group with <2 points or zero variance can't support a KDE (violinplot
    raises on it), so it's drawn as a thin marker at that constant value
    instead of being dropped.

    NaNs (e.g. a replicate an estimator crashed on, recorded as a failed
    result rather than dropped) are stripped from
    each group first: np.ptp() on an array containing even one NaN returns
    NaN, and NaN > 0 is False, so a single crashed replicate out of (say) 100
    would otherwise misclassify the WHOLE group as degenerate and silently
    draw nothing for the other 99 valid points.
    """
    data = [d[~np.isnan(d)] for d in data]
    ok_idx = [i for i, d in enumerate(data) if len(d) >= 2 and np.ptp(d) > 0]
    degenerate_idx = [i for i in range(len(data)) if i not in ok_idx]

    if ok_idx:
        vp = ax.violinplot([data[i] for i in ok_idx],
                            positions=[positions[i] for i in ok_idx],
                            widths=width, showmeans=False, showmedians=True,
                            showextrema=True)
        for body in vp["bodies"]:
            body.set_facecolor(facecolor)
            body.set_edgecolor(ink)
            body.set_linewidth(0.9)
            body.set_alpha(0.9)
        vp["cmedians"].set_color(ink)
        vp["cmedians"].set_linewidth(1.2)
        for key in ("cmins", "cmaxes", "cbars"):
            vp[key].set_color(muted)
            vp[key].set_linewidth(0.9)

    for i in degenerate_idx:
        if len(data[i]):
            ax.plot([positions[i] - width / 2, positions[i] + width / 2],
                    [data[i][0], data[i][0]], color=ink, lw=1.2, zorder=4)
