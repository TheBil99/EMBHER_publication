#!/usr/bin/env python
"""Run the prediction-vs-measured correlation for every gene in one task.

Replaces one ~30 s task per gene -- conda activation plus matplotlib import on every
one dominated the runtime for a fit that itself takes a fraction of a second. Kept as a
subprocess call into correlate_predictions.py so that script stays independently
usable, and for per-gene failure isolation: a gene that errors is recorded and skipped
rather than taking the whole task down.

Genes are paired from the config: the phenotype is <EXPERIMENT>_pheno.parquet and the
prediction is the cached file whose name contains the slugged gene id (predKey embeds
the gene id with dots replaced by underscores).
"""
from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import pandas as pd


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", s)


def load_worker(path):
    spec = importlib.util.spec_from_file_location("corr_worker", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-tsv", required=True)
    ap.add_argument("--pheno", nargs="+", required=True)
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--worker", required=True, help="correlate_predictions.py")
    ap.add_argument("--figures", action="store_true",
                    help="also write a per-gene figure; off by default (slow, unused "
                         "downstream -- the cross-gene figure is built by summarize)")
    ap.add_argument("--covar", default=None,
                    help="optional TSV of fixed-effect covariates, passed through to "
                         "correlate_predictions.py's --covar")
    args = ap.parse_args()

    worker = load_worker(args.worker)          # imports paid once for the whole run
    cfg = pd.read_csv(args.config_tsv, sep="\t")
    pheno_by_name = {Path(p).name: str(p) for p in args.pheno}
    preds = [str(p) for p in args.predictions]

    done, failed = [], {}
    for i, r in enumerate(cfg.itertuples(index=False), 1):
        exp, gid = str(r.experiment), str(r.gene_id)
        ph = pheno_by_name.get(f"{slug(exp)}_pheno.parquet")   # resolve_expression slugs the name
        hit = [p for p in preds if slug(gid) in Path(p).name]
        if ph is None or not hit:
            failed[exp] = f"missing {'phenotype' if ph is None else 'prediction'}"
            continue
        out = f"{exp}_prediction_correlation.parquet"
        try:
            worker.analyse(hit[0], ph, gid, exp, out,
                           f"{exp}_prediction_correlation.pdf" if args.figures else None,
                           quiet=True, covar_path=args.covar)
            done.append(out)
        except Exception as e:                     # one bad gene must not sink the task
            failed[exp] = f"{type(e).__name__}: {e}"
        if i % 50 == 0 or i == len(cfg):
            print(f"[info] {i}/{len(cfg)} genes ({len(failed)} failed)", flush=True)

    if failed:
        print(f"[warn] {len(failed)} gene(s) failed:", flush=True)
        for g, msg in list(failed.items())[:20]:
            print(f"    {g}: {msg}", flush=True)
    if not done:
        raise SystemExit("[error] no gene produced a correlation summary")
    print(f"[done] {len(done)} correlation summaries", flush=True)


if __name__ == "__main__":
    main()
