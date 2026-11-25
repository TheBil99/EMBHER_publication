"""One profile figure along the architecture, shared by workflow 15's per-database plots.

The x axis is the 24 layers (layer_order.LABELS) followed by `expression`, the whole-blood expression target:
the last point of each role's line, drawn like the layers. `draw_profile` takes the summary table (role, label, value, ci_low,
ci_high) and returns the figure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from layer_order import LABELS  # noqa: E402

XLABELS = LABELS + ["expression"]
COLORS = {"reference": "#1f77b4", "random_init": "#888888"}
NAMES = {"reference": "pretrained", "random_init": "random init"}


def draw_profile(summary, ylabel, title, hline=None, hline_label=None):
    """Each role is one line over XLABELS: the layers and then its expression value (the last point, drawn like the rest).
    A role without an `expression` row (random init, which has no expression readout) stops at the last layer."""
    x = np.arange(len(XLABELS))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for role in ("reference", "random_init"):
        r = summary[summary["role"] == role].set_index("label").reindex(XLABELS)
        ls = "-" if role == "reference" else "--"
        ax.fill_between(x, r["ci_low"], r["ci_high"], color=COLORS[role], alpha=0.25, lw=0)
        ax.plot(x, r["value"], color=COLORS[role], marker="o", ms=3, lw=1.4, ls=ls, label=NAMES[role])
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=0.8, label=hline_label)
    ax.set_xticks(x, XLABELS, rotation=90, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def save(fig, out_prefix):
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=200)
    plt.close(fig)
