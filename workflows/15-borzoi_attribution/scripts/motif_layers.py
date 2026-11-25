#!/usr/bin/env python
"""Motif enrichment of reference-window attributions along the Borzoi architecture.

For --gene, reads the seqlets in --seqlets_dir ({reference,random_init}/<target>/seqlets.parquet, the gene's
folder <genes_dir>/<gene>/seqlets; call_seqlets.py, on the attributions of embher-attribute; targets: expression, embedding_norm_<layer>;
--threshold must be the one they were called with) and, per target:

  1. keeps the seqlets whose scanned window lies in the sequence (and in the region);
  2. scans +-18 bp around each seqlet with FIMO against a motif database (grelu scan_sequences, as
     Decima's scan_motifs), at --pthresh;
  3. tests, per TF / RBP, whether more seqlet windows carry it than a pool of --n_null random windows
     of the same region predicts (motif_enrichment.py: one-sided binomial, BH).

A window counts once per TF however many hits it has. The null pool depends only on the sequence, so
it is scanned once per gene and shared by every target, trained or random_init.

Databases: --motif_db is a MEME file (or a gReLU database name) and --motif_names a TSV (motif_id, name)
of the motifs to scan and the TF / RBP each belongs to (scripts/prepare_motif_tables.py); without it every
motif is scanned and the TF is the id's first field.

Region and strand (for RNA-binding proteins, which act on the gene's own transcript):
  --region gene_body   keep only seqlets inside the gene's full span (start to end, introns included),
                       and draw the null windows and the "motifs with a hit" scan from the same span;
  --strand sense       scan the gene's sense strand only: the window of a minus-strand gene is
                       reverse-complemented and scanned with rc=False (so hit coordinates are in that
                       oriented window); the default `both` scans both strands.
Both need --gene_meta (gene start / end / TSS; the strand follows from where the TSS sits).

Targets already done with the same settings are skipped (--force recomputes); --no_plot writes only
the table; --save_hits also writes every motif hit of each target (motif_hits.parquet).

Writes, under <genes_dir>/<gene>/annotation/<db_name>/, motif_layers.parquet (zstd; one row per role and
target) and, in <role>/<target>/, motif_enrichment.parquet, motif_rows.json (and motif_hits.parquet, whose
seqlet_idx is the row of the shared seqlets.parquet). The plot has the ratio of enriched TFs (layer / the trained expression target),
the raw enriched counts with the expression count and the number of TFs with any hit in the region (a
grey reference that saturates at about the database size), and the seqlet counts.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from layer_order import LABELS  # noqa: E402  (workflow 15's x-axis order)
from embher.embedding.attribution.motif_enrichment import (  # noqa: E402
    enrichment, tf_name, window_tf_counts)

W = 18                       # half-width of the scanned window (Decima's scan_motifs default)
MOTIF_DB = "hocomoco_v13"      # set from --motif_db
MOTIF_NAMES = None             # motif id -> TF / RBP name, from --motif_names (None: all motifs, TF = id prefix)
COMP = str.maketrans("ACGTN", "TGCAN")


def seq_string(idx):
    return "".join(np.array(list("ACGTN"))[idx])


def scan(seqs, ids, pthresh, rc=True):
    from grelu.interpret.motifs import scan_sequences
    if not seqs:
        return pd.DataFrame(columns=["sequence", "motif"])
    hits = scan_sequences(seqs=seqs, seq_ids=ids, motifs=MOTIF_DB, names=list(MOTIF_NAMES) if MOTIF_NAMES else None,
                          pthresh=pthresh, rc=rc)
    if "motif" not in hits:                                      # gReLU returns no columns when nothing matched
        return pd.DataFrame(columns=["sequence", "motif", "motif_id"])
    if MOTIF_NAMES:
        hits["motif_id"] = hits["motif"]
        hits["motif"] = hits["motif"].map(MOTIF_NAMES)           # one name per TF / RBP, as tf_name gives
    return hits


def oriented(seq, flip):
    """The sequence, or its reverse complement when `flip` (a minus-strand gene scanned on its sense strand)."""
    return seq.translate(COMP)[::-1] if flip else seq


def valid_mids(mids, S):
    ok = [m for m in mids if m - W >= 0 and m + W <= len(S) and "N" not in S[m - W:m + W]]
    return np.array(ok, dtype=int)


def null_pool(S, lo, hi, n, pthresh, rng, flip, rc):
    """(counts per TF of windows carrying it, n windows, TFs with a hit anywhere in [lo, hi))."""
    mids = valid_mids(rng.integers(lo, hi, int(n * 1.05)), S)[:n]
    hits = scan([oriented(S[m - W:m + W], flip) for m in mids], [str(i) for i in range(len(mids))],
                pthresh, rc)
    step = min(1000, hi - lo)
    chunks = [oriented(S[s:s + step], flip) for s in range(lo, hi - step + 1, step)]
    chunks = [c for c in chunks if "N" not in c]
    seq_hits = scan(chunks, [str(i) for i in range(len(chunks))], pthresh, rc)
    seq_tfs = int(seq_hits["motif"].map(tf_name).nunique()) if len(seq_hits) else 0
    return window_tf_counts(hits, len(mids)), len(mids), seq_tfs


def gene_span(gene_meta_path, gene):
    """(start, end, strand) of the gene from the table the windows were built with. Coordinates are
    GENCODE's (1-based, inclusive); the TSS is `start` for a plus-strand gene and `end - 1` for a minus one."""
    from embher.embedding.utils.data import load_gene_metadata
    meta = load_gene_metadata(gene_meta_path, filter_chromosomes=None)
    sym = meta["gene_symbol"] if "gene_symbol" in meta else meta["ensg"]
    hit = meta[(sym == gene) | (meta["ensg"] == gene)]
    if hit.empty:
        raise SystemExit(f"[error] {gene} not in {gene_meta_path}")
    r = hit.iloc[0]
    start, end, tss = int(r["start"]), int(r["end"]), int(r["tss"])
    if tss == start:
        strand = "+"
    elif tss in (end, end - 1):
        strand = "-"
    else:
        raise SystemExit(f"[error] {gene}: TSS {tss} is neither the start {start} nor the end {end} of the gene")
    return start, end, strand


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes_dir", required=True, help="<run>/genes; one folder per gene")
    ap.add_argument("--threshold", type=float, default=5e-4,
                    help="seqlet p-value the seqlets were called with (checked against seqlets.json)")
    ap.add_argument("--pthresh", type=float, default=1e-4, help="FIMO p-value")
    ap.add_argument("--gene", required=True, help="gene name (the folder under --genes_dir)")
    ap.add_argument("--seqlets_dir", required=True,
                    help="the gene's seqlets folder (<genes_dir>/<gene>/seqlets, from call_seqlets.py)")
    ap.add_argument("--motif_db", default="hocomoco_v13", help="gReLU database name or a MEME file")
    ap.add_argument("--motif_names", help="TSV (motif_id, name): the motifs to scan and the TF / RBP each "
                                          "belongs to (default: every motif, TF = the id's first field)")
    ap.add_argument("--db_name", help="folder under the gene for this database's results "
                                      "(default: the database's name / file stem)")
    ap.add_argument("--gene_meta", help="gene table (start / end / TSS), needed for --region gene_body "
                                        "and --strand sense")
    ap.add_argument("--region", choices=["whole", "gene_body"], default="whole")
    ap.add_argument("--strand", choices=["both", "sense"], default="both")
    ap.add_argument("--save_hits", action="store_true", help="also write each target's motif hits")
    ap.add_argument("--n_null", type=int, default=20000, help="random windows in the null pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unit", default="TFs", help="what the names are, for the plot labels (TFs, RBPs)")
    ap.add_argument("--no_plot", action="store_true", help="only write the table, no per-gene figure")
    ap.add_argument("--force", action="store_true", help="recompute targets already done")
    a = ap.parse_args()
    global MOTIF_DB, MOTIF_NAMES
    MOTIF_DB = a.motif_db
    if a.motif_names:
        tab = pd.read_csv(a.motif_names, sep="\t", dtype=str)
        MOTIF_NAMES = dict(zip(tab["motif_id"], tab["name"]))
        assert not any("." in n for n in MOTIF_NAMES.values()), "names must not contain '.'"
        print(f"[info] {len(MOTIF_NAMES)} motifs of {tab['name'].nunique()} names from {a.motif_names}", flush=True)
    db_name = a.db_name or Path(a.motif_db).stem
    run = Path(a.genes_dir)
    rng = np.random.default_rng(a.seed)

    rows = []
    for gene in [a.gene]:
        gdir = run / gene
        files = sorted(gdir.glob("attribution/*/*/attribution.h5"))
        files = [f for f in files if f.parent.parent.name in ("reference", "random_init")]
        if not files:
            print(f"[skip] {gene}: no layer attributions")
            continue
        with h5py.File(files[0], "r") as h:
            S = seq_string(h["sequence"][...])
            win_start = int(h.attrs["win_start"])
        lo, hi, flip = W, len(S) - W, False
        if a.region == "gene_body" or a.strand == "sense":
            if not a.gene_meta:
                raise SystemExit("[error] --region gene_body / --strand sense need --gene_meta")
            start, end, strand = gene_span(a.gene_meta, gene)
            if a.region == "gene_body":
                lo, hi = max(start - 1 - win_start, W), min(end - win_start, len(S) - W)
            flip = a.strand == "sense" and strand == "-"
            print(f"[info] {gene}: {strand} strand, span {start}-{end}, region {a.region} "
                  f"[{lo}, {hi}) of {len(S)}, strand {a.strand}", flush=True)
        rc = a.strand == "both"

        def out_dir(f):                       # <genes_dir>/<gene>/annotation/<db_name>/<role>/<target>
            return run / gene / "annotation" / db_name / f.parent.parent.name / f.parent.name

        def seqlets_dir(f):                   # <seqlets_dir>/<role>/<target>
            return Path(a.seqlets_dir) / f.parent.parent.name / f.parent.name

        settings = dict(threshold=a.threshold, pthresh=a.pthresh, n_null=a.n_null, seed=a.seed)
        if a.region != "whole" or a.strand != "both":
            settings.update(region=a.region, strand=a.strand)       # the defaults keep older runs valid
        done = {}                                        # target dir -> rows saved by an earlier run
        for f in files:
            saved = out_dir(f) / "motif_rows.json"
            if not a.force and saved.exists() and saved.stat().st_mtime >= f.stat().st_mtime \
                    and (not a.save_hits or (out_dir(f) / "motif_hits.parquet").exists()):
                d = json.loads(saved.read_text())
                if d["settings"] == settings:            # older runs also had a tss_10kb scale: drop it
                    done[f] = [{k: v for k, v in r.items() if k != "scale"}
                               for r in d["rows"] if r.get("scale", "whole") == "whole"]
        todo = [f for f in files if f not in done]
        print(f"[info] {gene}: {len(done)} targets already done, {len(todo)} to do", flush=True)
        if todo:
            t0 = time.perf_counter()
            null_counts, n_null, seq_tfs = null_pool(S, lo, hi, a.n_null, a.pthresh, rng, flip, rc)
            print(f"[info] {gene}: null pool + sequence scan {time.perf_counter() - t0:.0f}s "
                  f"({seq_tfs} TFs with a hit)", flush=True)

        for f in files:
            if f in done:
                rows += done[f]
                continue
            role, name = f.parent.parent.name, f.parent.name
            label = name.replace("embedding_norm_", "")
            t0 = time.perf_counter()
            marker = seqlets_dir(f) / "seqlets.json"
            if not marker.exists():
                raise SystemExit(f"[error] {gene} {role} {name}: no seqlets (run call_seqlets.py first)")
            if json.loads(marker.read_text())["settings"]["threshold"] != a.threshold:
                raise SystemExit(f"[error] {gene} {role} {name}: seqlets were called at another --threshold")
            pk = pd.read_parquet(seqlets_dir(f) / "seqlets.parquet")
            mids_all = ((pk["start"] + pk["end"]) // 2).to_numpy()
            keep = np.isin(mids_all, valid_mids(mids_all, S)) & (mids_all >= lo) & (mids_all < hi)
            sel, pk, mids = np.flatnonzero(keep), pk[keep].reset_index(drop=True), mids_all[keep]
            hits = scan([oriented(S[m - W:m + W], flip) for m in mids],
                        [str(i) for i in range(len(mids))], a.pthresh, rc)
            out_dir(f).mkdir(parents=True, exist_ok=True)
            if a.save_hits:                  # `sequence` numbers the kept seqlets; seqlet_idx is the shared file's row
                cols = [c for c in ["motif", "motif_id", "start", "end", "strand", "score", "pval",
                                    "matched_seq"] if c in hits]
                out = hits[cols].copy()
                out.insert(0, "seqlet_idx", sel[hits["sequence"].astype(int).to_numpy()])
                out.to_parquet(out_dir(f) / "motif_hits.parquet", compression="zstd", index=False)
            enr = enrichment(window_tf_counts(hits, len(mids)), len(mids), null_counts, n_null)
            target_rows = [dict(gene=gene, role=role, target=name, label=label, n_seqlets=len(mids),
                                n_tested=len(enr), n_enriched=int(enr["enriched"].sum()) if len(enr) else 0,
                                n_seq_tfs=int(seq_tfs))]
            rows += target_rows
            enr.to_parquet(out_dir(f) / "motif_enrichment.parquet", compression="zstd", index=False)
            (out_dir(f) / "motif_rows.json").write_text(
                json.dumps(dict(settings=settings, rows=target_rows)))      # last = marks it done
            print(f"[done] {gene} {role} {name}: {len(pk)} seqlets "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    for gene, g in df.groupby("gene"):
        g.to_parquet(run / gene / "annotation" / db_name / "motif_layers.parquet", compression="zstd", index=False)
        if not a.no_plot:
            plot_gene(gene, g, run / gene / "annotation" / db_name, a.unit)
    print(f"[done] {len(df)} rows")


def plot_gene(gene, g, out, unit="TFs"):
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 9), sharex=True)
    x = np.arange(len(LABELS))
    colors = {"reference": "#1f77b4", "random_init": "#888888"}
    expr = g[(g["role"] == "reference") & (g["target"] == "expression")]
    n_expr = float(expr["n_enriched"].iloc[0]) if len(expr) else np.nan
    for role in ("reference", "random_init"):
        r = g[(g["role"] == role) & (g["target"] != "expression")].set_index("label").reindex(LABELS)
        style = dict(color=colors[role], marker="o", ms=3, lw=1.2,
                     ls="-" if role == "reference" else "--", label=role.replace("_", " "))
        ratio = r["n_enriched"] / n_expr if n_expr and n_expr > 0 else r["n_enriched"] * np.nan
        axes[0].plot(x, ratio, **style)
        axes[1].plot(x, r["n_enriched"], **style)
        axes[2].plot(x, r["n_seqlets"], **style)
    if len(expr):
        axes[1].axhline(n_expr, color="#d95f02", lw=1, label="expression (trained)")
        axes[1].axhline(float(expr["n_seq_tfs"].iloc[0]), color="#bbbbbb", ls=":", lw=1,
                        label=f"{unit} with a hit in the region")
        axes[2].axhline(float(expr["n_seqlets"].iloc[0]), color="#d95f02", lw=1)
    axes[0].axhline(1, color="k", lw=0.4)
    if np.isnan(n_expr) or n_expr == 0:
        axes[0].text(0.5, 0.5, f"expression has no enriched {unit}\n(ratio undefined)",
                     transform=axes[0].transAxes, ha="center", fontsize=8)
    axes[0].set_ylabel(f"enriched {unit} / expression")
    axes[1].set_ylabel(f"enriched {unit} (q < 0.05)")
    axes[2].set_ylabel("seqlets")
    axes[2].set_xticks(x, LABELS, rotation=90, fontsize=6)
    axes[1].legend(fontsize=6)
    fig.suptitle(f"{gene}: motifs in attribution seqlets along the architecture (norm of each layer)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "motif_layers.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
