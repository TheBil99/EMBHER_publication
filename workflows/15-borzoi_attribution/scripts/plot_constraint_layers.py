#!/usr/bin/env python
"""Constraint of the attributed sequence along the Borzoi architecture: one figure per constraint track.

Reads <genes_dir>/<gene>/annotation/<db_name>/constraint_counts.parquet (constraint_layers.py) of all genes.

Expected values: per role, the background (outside the seqlets) of every gene and target is pooled by
context; its fraction of seen positions (gnomad) or mean PhyloP (phylop) is the expected value of a position
of that context. Per gene, target and layer: expected = sum over contexts of the seqlet positions times that
rate. Observed and expected are pooled over genes (genes weighted by their seqlet positions) and the value is

  gnomad   log odds ratio of unseen : seen positions, observed against expected (a position is "seen" when
           gnomAD has a PASS SNV with AC > 0 there; positive = more unseen than expected = constraint;
           low-coverage positions are dropped by constraint_layers.py)
  phylop   --phylop_stat difference (default): mean observed - mean expected PhyloP
           --phylop_stat ratio:                mean observed / mean expected PhyloP (1 = as expected). The
           pooled expected mean is close to 0 and changes sign between targets, so the ratio is unstable
           there; check the printed expected means before using it.

The 95% CI is a bootstrap over genes (resampled together; the context rates are kept fixed). The whole-blood
expression target (pretrained only) is the last point of the figure.

Writes <out_prefix>.png / .pdf, <out_prefix>_per_gene.parquet (observed, expected and positions per gene, role
and target) and <out_prefix>_summary.parquet (value and CI, plus the pooled observed and expected sums, the
means and their difference and ratio, per role and layer; `expression` is a label).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from layer_axes import draw_profile, save
from layer_order import LABELS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes_dir", required=True, help="<run>/genes; one folder per gene")
    ap.add_argument("--db_name", required=True)
    ap.add_argument("--kind", required=True, choices=["gnomad", "phylop"])
    ap.add_argument("--phylop_stat", choices=["difference", "ratio"], default="difference")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(a.genes_dir).glob(f"*/annotation/{a.db_name}/constraint_counts.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    genes = sorted(df["gene"].unique())
    obs_col = "n_seen" if a.kind == "gnomad" else "sum_phylop"

    # rate per context and role, from the background of all genes and targets
    bg = df[df["group"] == "background"].groupby(["role", "context"])[["n_positions", obs_col]].sum()
    rate = (bg[obs_col] / bg["n_positions"]).rename("rate")
    sl = df[df["group"] == "seqlet"].join(rate, on=["role", "context"]).dropna(subset=["rate"])
    sl["observed"] = sl[obs_col]
    sl["expected"] = sl["n_positions"] * sl["rate"]
    per = (sl.groupby(["gene", "role", "target", "label"])[["observed", "expected", "n_positions"]].sum()
             .reset_index())
    per.to_parquet(f"{a.out_prefix}_per_gene.parquet", compression="zstd", index=False)

    def matrix(role, label):
        t = per[(per["role"] == role) & (per["label"] == label)].set_index("gene")
        return [t[c].reindex(genes).fillna(0).to_numpy(float) for c in ("observed", "expected", "n_positions")]

    def stat(w, o, e, n):                 # pooled value of one (role, label) under gene weights w
        O, E, N = w @ o, w @ e, w @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            if a.kind == "gnomad":        # log of (unseen : seen observed) / (unseen : seen expected)
                return np.log((N - O) / O * E / (N - E))
            return O / E if a.phylop_stat == "ratio" else (O - E) / N

    G = len(genes)
    w = np.vstack([np.ones(G), rng.multinomial(G, np.full(G, 1 / G), size=a.n_boot).astype(float)])
    rows = []
    for role in ("reference", "random_init"):
        for lab in LABELS + ["expression"]:
            m = matrix(role, lab)
            if not m[2].any():
                continue
            v = np.array([stat(wi, *m) for wi in w])                       # row 0 = the data
            O, E, N = (x.sum() for x in m)
            rows.append(dict(role=role, label=lab, n_genes=int((m[2] > 0).sum()), value=v[0],
                             ci_low=np.nanpercentile(v[1:], 2.5), ci_high=np.nanpercentile(v[1:], 97.5),
                             observed=O, expected=E, n_positions=N, observed_mean=O / N, expected_mean=E / N,
                             difference=(O - E) / N, ratio=O / E))
    s = pd.DataFrame(rows)
    s.to_parquet(f"{a.out_prefix}_summary.parquet", compression="zstd", index=False)
    if a.kind == "phylop":
        print(f"[info] pooled expected mean PhyloP: min {s['expected_mean'].min():.3f}, "
              f"max {s['expected_mean'].max():.3f}", flush=True)

    if a.kind == "gnomad":
        ylabel, hline, hl = "log odds ratio of unseen : seen positions", None, None
    elif a.phylop_stat == "ratio":
        ylabel, hline, hl = "mean PhyloP, observed / expected", 1, "as expected"
    else:
        ylabel, hline, hl = "mean PhyloP, observed - expected", None, None
    fig = draw_profile(s, ylabel, f"{a.db_name}: constraint of attribution seqlets along the architecture "
                                  f"({G} genes, pooled)", hline, hl)
    save(fig, a.out_prefix)
    print(f"[done] {a.out_prefix}.png / .pdf", flush=True)


if __name__ == "__main__":
    main()
