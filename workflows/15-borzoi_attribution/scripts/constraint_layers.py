#!/usr/bin/env python
"""Constraint counts of the seqlets of every target of one gene, by sequence context (torch-free).

For --gene, takes the seqlets <genes_dir>/<gene>/seqlets/<role>/<target>/seqlets.parquet (call_seqlets.py,
start / end are half-open indices in the window), the reference window sequence (attribution.h5) and the
prepared track --track (prepare_constraint_track.py), and counts, per target, separately for the positions
inside seqlets and for every other position of the window (the background), by context (trinucleotide of the
reference, reverse-complement collapsed, 32 classes; CpG is part of it):

  --kind gnomad   n_positions and n_seen (positions with a PASS SNV, AC > 0): seen vs unseen. Positions
                  whose local coverage looks low are dropped from both groups: the AN of the nearest
                  variant (a proxy for the called alleles at the position) must be at least --min_an_frac
                  times the 95th percentile of the window's AN.
  --kind phylop   n_positions (positions with a score), sum_phylop

Nothing is excluded for biology; the only filter is callability (a context and, for phylop, a score; for
gnomad, the coverage proxy above). No expectation
is computed here (that needs the background of all genes: plot_constraint_ratio.py).

Writes <genes_dir>/<gene>/annotation/<db_name>/constraint_counts.parquet (long: role, target, label, group,
context, counts) and, last, constraint_counts.json (settings). Skipped when that file is newer than every
seqlets.json of the gene and the settings agree (--force recomputes).
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def contexts(seq):
    """Canonical trinucleotide code (0..63) of every position, -1 where undefined (ends, N)."""
    s = seq.astype(np.int64)
    ctx = np.full(len(s), -1, np.int64)
    a, b, c = s[:-2], s[1:-1], s[2:]
    code = 16 * a + 4 * b + c
    rc = 16 * (3 - c) + 4 * (3 - b) + (3 - a)
    ctx[1:-1] = np.where((a < 4) & (b < 4) & (c < 4), np.minimum(code, rc), -1)
    return ctx


def seqlet_mask(start, end, L):
    start, end = np.asarray(start, np.int64), np.asarray(end, np.int64)    # empty seqlet tables come back as float/object
    d = np.zeros(L + 1, np.int64)
    np.add.at(d, np.clip(start, 0, L), 1)
    np.add.at(d, np.clip(end, 0, L), -1)
    return np.cumsum(d)[:L] > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes_dir", required=True, help="<run>/genes; one folder per gene")
    ap.add_argument("--gene", required=True)
    ap.add_argument("--db_name", required=True, help="folder under annotation/ (gnomad, phylop, ...)")
    ap.add_argument("--kind", required=True, choices=["gnomad", "phylop"])
    ap.add_argument("--track", required=True, help="<gene>.parquet from prepare_constraint_track.py")
    ap.add_argument("--min_an_frac", type=float, default=0.9,
                    help="gnomad: minimum local AN (nearest variant) as a fraction of the window's 95th percentile")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    gdir = Path(a.genes_dir) / a.gene
    out = gdir / "annotation" / a.db_name
    seqlet_files = sorted(gdir.glob("seqlets/*/*/seqlets.parquet"))
    if not seqlet_files:
        raise SystemExit(f"[error] {a.gene}: no seqlets under {gdir / 'seqlets'}")
    settings = dict(kind=a.kind, track=Path(a.track).name,
                    stat="seen_by_context_an_proxy" if a.kind == "gnomad" else "mean_by_context")
    if a.kind == "gnomad":
        settings["min_an_frac"] = a.min_an_frac
    marker = out / "constraint_counts.json"
    newest = max(f.with_name("seqlets.json").stat().st_mtime for f in seqlet_files)
    if not a.force and marker.exists() and marker.stat().st_mtime >= newest \
            and json.loads(marker.read_text())["settings"] == settings:
        print(f"[skip] {a.gene} {a.db_name}: up to date")
        return

    with h5py.File(gdir / "attribution" / "reference" / "expression" / "attribution.h5", "r") as h:
        seq = h["sequence"][...]
    L = len(seq)
    ctx = contexts(seq)
    track = pd.read_parquet(a.track)
    if a.kind == "gnomad":
        seen = np.zeros(L, bool)
        seen[track["idx"].to_numpy()] = True
        pos = np.unique(track["idx"].to_numpy())
        an = track.groupby("idx")["an"].max().reindex(pos).to_numpy()
        k = np.clip(np.searchsorted(pos, np.arange(L)), 1, len(pos) - 1)       # nearest variant, either side
        near = np.where(np.arange(L) - pos[k - 1] <= pos[k] - np.arange(L), k - 1, k)
        local_an = an[near]
        covered = local_an >= a.min_an_frac * np.percentile(an, 95)
        callable_ = (ctx >= 0) & covered
    else:
        score = np.full(L, np.nan)
        score[track["idx"].to_numpy()] = track["phylop"].to_numpy()
        callable_ = (ctx >= 0) & np.isfinite(score)
        score = np.nan_to_num(score)

    rows = []
    for f in seqlet_files:
        role, target = f.parent.parent.name, f.parent.name
        sl = pd.read_parquet(f, columns=["start", "end"])
        mask = seqlet_mask(sl["start"].to_numpy(), sl["end"].to_numpy(), L)
        for group, inside in (("seqlet", True), ("background", False)):
            sel = callable_ & (mask == inside)
            c = ctx[sel]
            d = dict(n_positions=np.bincount(c, minlength=64))
            if a.kind == "gnomad":
                d["n_seen"] = np.bincount(c[seen[sel]], minlength=64)
            else:
                d["sum_phylop"] = np.bincount(c, weights=score[sel], minlength=64)
            t = pd.DataFrame(d).rename_axis("context").reset_index()
            t = t[t["n_positions"] > 0]
            t.insert(0, "group", group)
            t.insert(0, "label", target.replace("embedding_norm_", ""))
            t.insert(0, "target", target)
            t.insert(0, "role", role)
            t.insert(0, "gene", a.gene)
            t["n_seqlets"] = len(sl)
            rows.append(t)
    out.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)                                  # json last = marks it done
    tmp = out / "constraint_counts.tmp.parquet"
    pd.concat(rows, ignore_index=True).to_parquet(tmp, compression="zstd", index=False)
    tmp.replace(out / "constraint_counts.parquet")
    marker.write_text(json.dumps(dict(settings=settings, n_targets=len(seqlet_files))))
    print(f"[done] {a.gene} {a.db_name}: {len(seqlet_files)} targets", flush=True)


if __name__ == "__main__":
    main()
