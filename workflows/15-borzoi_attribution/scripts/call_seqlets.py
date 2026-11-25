#!/usr/bin/env python
"""Seqlets of the reference-window attributions, called once per target and shared by every annotation.

For --gene, reads <genes_dir>/<gene>/attribution/<role>/<target>/attribution.h5 (embher-attribute) and
calls seqlets on the base-summed whole-window attribution (Decima's Attribution.find_peaks, tangermeme
recursive_seqlets, p < --threshold, length --min_len to --max_len). Nothing is filtered by region or
sequence here: the annotation steps (motif_layers.py, constraint) choose what they keep.

Writes <genes_dir>/<gene>/seqlets/<role>/<target>/seqlets.parquet (zstd; Decima's peak table: start / end
are indices in the window, row order is the seqlet id the annotations refer to) and, last, seqlets.json
(settings). A target is skipped when seqlets.json is newer than its attribution.h5 and has the same
settings (--force recomputes).
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    from decima.core.attribution import Attribution

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes_dir", required=True, help="<run>/genes; one folder per gene")
    ap.add_argument("--gene", required=True, help="gene name (the folder under --genes_dir)")
    ap.add_argument("--threshold", type=float, default=5e-4, help="seqlet p-value (Decima's)")
    ap.add_argument("--min_len", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=25)
    ap.add_argument("--force", action="store_true", help="recompute targets already done")
    a = ap.parse_args()

    gdir = Path(a.genes_dir) / a.gene
    files = sorted(gdir.glob("attribution/*/*/attribution.h5"))
    if not files:
        raise SystemExit(f"[error] {a.gene}: no attributions under {gdir / 'attribution'}")
    settings = dict(threshold=a.threshold, min_len=a.min_len, max_len=a.max_len)
    n_done = 0
    for f in files:
        role, target = f.parent.parent.name, f.parent.name
        out = gdir / "seqlets" / role / target
        marker = out / "seqlets.json"
        if not a.force and marker.exists() and marker.stat().st_mtime >= f.stat().st_mtime \
                and json.loads(marker.read_text())["settings"] == settings:
            continue
        with h5py.File(f, "r") as h:
            attr = h["attribution"][...].astype(np.float32)
        pk = Attribution.find_peaks(attr, a.threshold, a.min_len, a.max_len, 0, "both").reset_index(drop=True)
        out.mkdir(parents=True, exist_ok=True)
        marker.unlink(missing_ok=True)                       # seqlets.json last = marks it done
        tmp = out / "seqlets.tmp.parquet"
        pk.to_parquet(tmp, compression="zstd", index=False)
        tmp.replace(out / "seqlets.parquet")
        marker.write_text(json.dumps(dict(settings=settings, n_seqlets=len(pk))))
        n_done += 1
        print(f"[done] {a.gene} {role} {target}: {len(pk)} seqlets", flush=True)
    print(f"[info] {a.gene}: {n_done} of {len(files)} targets computed", flush=True)


if __name__ == "__main__":
    main()
