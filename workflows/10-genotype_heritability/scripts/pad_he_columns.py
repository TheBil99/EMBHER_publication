#!/usr/bin/env python
"""Add the inference columns exact_reml_heritability.py --nonneg writes and he_heritability.py
does not (lrt, lrt_p), as NaN, so --estimator he output has the same columns as the REML run's.
Edits the parquet in place; an existing column is left untouched."""
import sys

import numpy as np
import pandas as pd

path = sys.argv[1]
df = pd.read_parquet(path)
for c in ("lrt", "lrt_p"):
    if c not in df.columns:
        df[c] = np.nan
df.to_parquet(path, index=False)
