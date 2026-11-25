"""embher plotting: one visual style and a small set of plot archetypes.

Usage (put `src` on sys.path, as for `elasticnet_fit`):

    from embher_plotting import apply_style, archetypes as A, annotations as N, layout as L
    apply_style()
    fig, axes = L.panel_grid(2)
    A.scatter(axes[0], x, y, identity=True, stats="pearson")
    A.violin(axes[1], [a, b], labels=["rare", "common"])
    L.save(fig, "results/<workflow>/plots/figure")      # .pdf + .png

Figures are always clean: `L.title(fig, title, subtitle)` and any axes title or suptitle are
removed on `L.save` / `PdfBook.add` and written to an information page after the figure pages
(PDF only); the figure pages hold just the figure, its legends and annotations.

`archetypes` holds the catalogue; `python -m embher_plotting.gallery` renders one
example of each.
"""
from . import annotations, archetypes, layout, theme
from .theme import apply_style

__all__ = ["annotations", "archetypes", "layout", "theme", "apply_style"]
