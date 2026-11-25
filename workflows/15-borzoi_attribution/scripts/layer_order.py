"""Order of the 24 capture layers on the x axis of workflow 15's plots.

The library's CAPTURE_LAYERS (shared with workflows 12 and 17) lists the two skip convolutions after the
transformer. Here each sits after the encoder layer it reads: horizontal_conv0 the res-tower output
(encoder_res_tower_4) and horizontal_conv1 unet1 (encoder_unet1).
"""
# capture_layers is torch-free, so this import does not load the model stack
from embher.embedding.attribution.capture_layers import CAPTURE_LAYERS, capture_label

AFTER = {"encoder_res_tower_4": "skip_horizontal_conv0", "encoder_unet1": "skip_horizontal_conv1"}
LABELS = []
for _l in CAPTURE_LAYERS:
    if _l.startswith("skip/"):
        continue
    LABELS.append(capture_label(_l))
    if LABELS[-1] in AFTER:
        LABELS.append(AFTER[LABELS[-1]])
assert len(LABELS) == len(CAPTURE_LAYERS) and set(LABELS) == {capture_label(l) for l in CAPTURE_LAYERS}
