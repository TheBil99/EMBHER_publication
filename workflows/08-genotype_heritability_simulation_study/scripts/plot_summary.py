#!/usr/bin/env python
"""Combined calibration summary: one PDF.

Pages 1-7 (summarize_gene_calibration.build): every method's calibration, error
distribution, QQ-plot, coverage, over-high fraction, bias and convergence -- one line per
GENE, gridded rows=p_causal, cols=method -- see that module's docstring. --no-se drops the
three pages built on h2_se (QQ-plot, coverage, over-high).

Last page (plot_bin_bias.build): every binned method's PER-BIN h2_k1, h2_k2, ... bias
against the matched per-replicate true per-bin h2 from compute_true_bin_h2.py.

Two tables (different granularity, so they stay separate files even though
the PDF is now one): bias_summary.parquet (method/true_h2/p_causal, genes and
replicates pooled) and bin_bias_summary.parquet (method/bin/true_h2/p_causal).
convergence_summary.parquet is still written for backward compatibility with
anything reading it directly, derived from bias_summary's own columns (page 7
above -- the convergence page -- gets its numbers straight from the raw
per-replicate data, not from this table).

--split-causal-band
-------------------
A run may mix simulations built under DIFFERENT causal-variant MAF partitions
(workflow 07's causal_min_maf/causal_max_maf): the same genes, genotype panel,
h2 grid and seeds, differing only in which variants were allowed to carry an
effect. Pooling those into one summary is meaningless -- the whole point of the
comparison is that a MAF-binned estimator behaves differently when a rare bin's
TRUE h2 is genuinely non-zero versus exactly zero (the boundary case real
expression sits in). With this flag the inputs are partitioned by band and the
pages are built once per band, giving one PDF per partition
(<out-pdf stem>_<band>.pdf) instead of one pooled PDF.

The band is read off each input FILE's name, which workflow 07 already encodes
via makeExpName: a '_cmaf<lo>_<hi>_' infix means that band, and its absence
means the default whole-spectrum draw, labelled cmaf0_0.5 so both partitions
carry the same self-describing form. The three tables stay single files, gaining
a causal_band column rather than being split into one set per band.

Without the flag nothing is partitioned: a single pooled PDF at --out-pdf.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import plot_bin_bias
import summarize_gene_calibration

# '_cmaf<lo>_<hi>_' as workflow 07's makeExpName writes it, e.g.
# CXCR1_h2_0.05_pc_0.01_rep_46_cmaf0.01_0.5_he_mac_maf_bins_4.parquet. The trailing
# underscore keeps the match anchored to the band and off the method suffix.
BAND_RE = re.compile(r"_cmaf([0-9.]+)_([0-9.]+)_")
DEFAULT_BAND = "cmaf0_0.5"


def band_of(path):
    """Causal-MAF band a published per-replicate parquet belongs to, from its name."""
    m = BAND_RE.search(Path(path).name)
    return f"cmaf{m[1]}_{m[2]}" if m else DEFAULT_BAND


def build_one(h2_parquets, true_bin_parquets, out_pdf, no_se=False):
    """The single-band pipeline: build every page, write one PDF, and return the two
    tables it produced."""
    calib_df = summarize_gene_calibration.load(h2_parquets)
    figs, bias_table = summarize_gene_calibration.build(calib_df, no_se=no_se)

    est = plot_bin_bias.load(h2_parquets)
    truth = plot_bin_bias.load(true_bin_parquets)
    est["p_causal"] = est.experiment.str.extract(r"pc_([0-9.eE+-]+)_rep")[0].astype(float)
    fig_bins, bin_bias_table = plot_bin_bias.build(est, truth)

    with PdfPages(out_pdf) as pdf:
        for fig in figs:
            pdf.savefig(fig, facecolor=fig.get_facecolor())
        pdf.savefig(fig_bins, facecolor=fig_bins.get_facecolor())
    print(f"[done] -> {out_pdf} ({len(figs) + 1} pages)")
    return bias_table, bin_bias_table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h2-parquets", nargs="+", required=True,
                     help="every per-condition estimator output (*_<estimator>_<arm>.parquet)")
    ap.add_argument("--true-bin-parquets", nargs="+", required=True,
                     help="every *_true_bins_N.parquet (compute_true_bin_h2.py output)")
    ap.add_argument("--out-pdf", required=True)
    ap.add_argument("--out-bias-table", required=True)
    ap.add_argument("--out-bin-bias-table", required=True)
    ap.add_argument("--out-convergence-table", required=True)
    ap.add_argument("--split-causal-band", action="store_true",
                     help="one PDF per causal-MAF partition present in the inputs "
                          "(see module docstring); off = one pooled PDF, as before")
    ap.add_argument("--no-se", action="store_true",
                     help="skip the pages built on h2_se (H0 QQ-plot, CI coverage, confidently too high)")
    args = ap.parse_args()

    if args.split_causal_band:
        bands = sorted({band_of(p) for p in args.h2_parquets + args.true_bin_parquets})
        print(f"[split] causal-MAF partitions present: {', '.join(bands)}")
        stem = Path(args.out_pdf)
        stem = stem.with_suffix("")
        bias_parts, bin_bias_parts = [], []
        for band in bands:
            sel = lambda ps: [p for p in ps if band_of(p) == band]
            bias, bin_bias = build_one(sel(args.h2_parquets), sel(args.true_bin_parquets),
                                       f"{stem}_{band}.pdf", no_se=args.no_se)
            bias.insert(0, "causal_band", band)
            bin_bias.insert(0, "causal_band", band)
            bias_parts.append(bias)
            bin_bias_parts.append(bin_bias)
        bias_table = pd.concat(bias_parts, ignore_index=True)
        bin_bias_table = pd.concat(bin_bias_parts, ignore_index=True)
        group_cols = ["causal_band", "method", "true_h2", "p_causal"]
    else:
        bias_table, bin_bias_table = build_one(
            args.h2_parquets, args.true_bin_parquets, args.out_pdf, no_se=args.no_se)
        group_cols = ["method", "true_h2", "p_causal"]

    convergence_table = bias_table[group_cols + ["n", "n_unconverged"]].copy()
    convergence_table["fail_pct"] = 100 * convergence_table.n_unconverged / convergence_table.n

    bias_table.to_parquet(args.out_bias_table, index=False)
    bin_bias_table.to_parquet(args.out_bin_bias_table, index=False)
    convergence_table.to_parquet(args.out_convergence_table, index=False)
    print(f"[done] tables -> {args.out_bias_table}, {args.out_bin_bias_table}, "
          f"{args.out_convergence_table}")


if __name__ == "__main__":
    main()
