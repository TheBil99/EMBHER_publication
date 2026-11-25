"""Shared definitions of the intermediate-file bundles (layer names / order, file names, models).

Layer names and order are Table S2's (workflow 15's layer_order.LABELS, imported by path), not retyped.
The expression target is not one of the 24 layers: it gets index 25, the last point of the figures.
"""
import importlib.util
from pathlib import Path

LAYER_ORDER_PY = Path(__file__).resolve().parents[2] / "15-borzoi_attribution" / "scripts" / "layer_order.py"
# workflow 15's role folder -> the name the supplementary tables use
MODELS = {"reference": "pretrained", "random_init": "random_init"}


def _labels():
    spec = importlib.util.spec_from_file_location("layer_order", LAYER_ORDER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.LABELS)


# (layer_index, layer_name, workflow-15 target folder)
LAYERS = [(i, name, f"embedding_norm_{name}") for i, name in enumerate(_labels(), start=1)]
LAYERS.append((len(LAYERS) + 1, "expression", "expression"))


def bed_name(model, index, layer_name):
    return f"{model}_layer-{index:02d}_{layer_name}.bed.gz"
