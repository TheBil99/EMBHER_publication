#!/usr/bin/env python
"""Motif tables for the motif step: which motifs of a database to scan and the TF / RBP each one is.

Writes TSVs (motif_id, name) next to the processed data; the databases themselves stay untouched under
data/raw and are read in place by motif_layers.py (--motif_db <MEME file> --motif_names <table>).

  hocomoco_v13    every motif of HOCOMOCO v13 CORE (H13CORE_meme_format.meme); name = the TF symbol of the
                  annotation (H13CORE_annotation.jsonl)
  cisbp_rna       the human CISBP-RNA motifs with a PWM in pwms_all_motifs.as_dna.meme whose RBP has
                  RBP_Status == 'D' (the motif was determined directly for that RBP by an experiment, not
                  inferred from a similar RNA-binding domain) and Motif_Type != 'JPLE' (a predicted motif);
                  name = RBP_Name
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def meme_ids(path):
    return [l.split()[1] for l in open(path) if l.startswith("MOTIF")]


def hocomoco(out):
    base = ROOT / "data/raw/HOCOMOCO/hocomoco13/H13CORE"
    ids = set(meme_ids(base / "formatted_motifs/H13CORE_meme_format.meme"))
    rows = []
    for line in open(base / "H13CORE_annotation.jsonl"):
        d = json.loads(line)
        if d["name"] in ids:
            rows.append(dict(motif_id=d["name"], name=d["masterlist_info"]["tf"], quality=d["quality"]))
    tab = pd.DataFrame(rows)
    assert len(tab) == len(ids), (len(tab), len(ids))
    print(f"[hocomoco_v13] name differs from the id prefix for {(tab['name'] != tab['motif_id'].str.split('.').str[0]).sum()} motifs")
    tab.to_csv(out, sep="\t", index=False)
    print(f"[hocomoco_v13] {len(tab)} motifs, {tab['name'].nunique()} TFs -> {out}")


def cisbp_rna(out):
    base = ROOT / "data/raw/CISBP-RNA"
    ids = set(meme_ids(base / "pwms_all_motifs.as_dna.meme"))
    info = pd.read_csv(base / "RBP_Information_all_motifs.txt", sep="\t", dtype=str)
    keep = info[(info["RBP_Status"] == "D") & (info["Motif_ID"] != ".") & info["Motif_ID"].isin(ids)
                & (info["Motif_Type"] != "JPLE")]
    assert keep["Motif_ID"].is_unique, "a direct motif belongs to more than one RBP"
    tab = keep.rename(columns={"Motif_ID": "motif_id", "RBP_Name": "name"})[
        ["motif_id", "name", "Motif_Type", "DBID"]].rename(columns={"DBID": "ensembl_gene"})
    tab.to_csv(out, sep="\t", index=False)
    print(f"[cisbp_rna] {len(tab)} motifs, {tab['name'].nunique()} RBPs -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(ROOT / "data/prep/motif_tables"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hocomoco(out / "hocomoco_v13.tsv")
    cisbp_rna(out / "cisbp_rna.tsv")


if __name__ == "__main__":
    main()
