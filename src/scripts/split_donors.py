#!/usr/bin/env python
"""Split the GTEx WGS donor panel into train/val/test, once, for PrediXcan.

Same donors underlie every gene (workflow 09's per-gene donor list is this set
intersected with that gene's expression/covariates), so the split is made once here
and intersected per gene downstream -- every gene's elastic net sees the same donors
in the same role, which is what makes val-selected hyperparameters and the held-out
test correlation comparable across genes.
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--donors", required=True, help="wgs_donors.txt, one donor per line")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--out-test", required=True)
    args = ap.parse_args()

    donors = np.array([d.strip() for d in open(args.donors) if d.strip()])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(donors))

    n_train = int(round(args.train_frac * len(donors)))
    n_val = int(round(args.val_frac * len(donors)))
    train = sorted(donors[perm[:n_train]])
    val = sorted(donors[perm[n_train:n_train + n_val]])
    test = sorted(donors[perm[n_train + n_val:]])

    for path, subs in ((args.out_train, train), (args.out_val, val), (args.out_test, test)):
        with open(path, "w") as fh:
            fh.write(",".join(subs))

    print(f"[done] {len(donors)} donors -> train {len(train)}, val {len(val)}, "
          f"test {len(test)}", flush=True)


if __name__ == "__main__":
    main()
