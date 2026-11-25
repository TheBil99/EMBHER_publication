#!/usr/bin/env python
"""Per-donor mean Euclidean distance to the own group (intra) vs the other group (inter).

For one gene and two ancestry groups (EUR and one other, from the comparison's groups
file), for each donor:
  intra  mean distance to every OTHER donor of the same group
  inter  mean distance to every donor of the other group
in genotypes (0/1/2 non-reference counts) and raw Borzoi embeddings (haplotypes summed,
not whitened). Donors outside the two groups are left out.

Outputs (prefix <gene>_intra_inter_distances): per-donor .tsv, per-group _summary.tsv
(medians, inter/intra ratio, % of donors closer on average to the other group) and the
violin plot (.png/.pdf).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"intra": "#2a78d6", "inter": "#eb6834"}


def euclidean(X):
    sq = (X ** 2).sum(1)
    return np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0, None))


def intra_inter(X, labels):
    X = X - X.mean(0)                                   # numerics only; distances unchanged
    D = euclidean(X)
    out = pd.DataFrame(index=range(len(labels)))
    for g in np.unique(labels):
        same, other = labels == g, labels != g
        # own row excluded from intra (its zero self-distance)
        out.loc[same, "intra"] = D[np.ix_(same, same)].sum(1) / (same.sum() - 1)
        out.loc[same, "inter"] = D[np.ix_(same, other)].mean(1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene", required=True, help="label used in titles and file names")
    ap.add_argument("--groups", required=True, help="IID / GROUP TSV (make_ancestry_groups.py)")
    ap.add_argument("--genotype", required=True, help="donor x variant parquet (genotype_features.py)")
    ap.add_argument("--embedding", required=True, help="donor x feature parquet (combine_haplotypes.py)")
    a = ap.parse_args()

    groups = pd.read_csv(a.groups, sep="\t").set_index("IID").GROUP
    order = ["EUR", groups[groups != "EUR"].iloc[0]]
    reps = {"Genotype": a.genotype, "Borzoi embedding": a.embedding}

    tables = []
    for name, path in reps.items():
        df = pd.read_parquet(path)
        df = df[df.index.astype(str).isin(groups.index)]
        labels = groups.loc[df.index.astype(str)].to_numpy()
        t = intra_inter(df.to_numpy(np.float64), labels)
        t.insert(0, "donor", df.index.astype(str)); t.insert(1, "group", labels)
        t.insert(2, "representation", name)
        tables.append(t)
    res = pd.concat(tables, ignore_index=True)
    res["inter_minus_intra"] = res.inter - res.intra
    prefix = f"{a.gene}_intra_inter_distances"
    res.to_csv(f"{prefix}.tsv", sep="\t", index=False)

    summary = (res.groupby(["representation", "group"], sort=False)
                  .agg(n=("donor", "size"), intra_median=("intra", "median"),
                       inter_median=("inter", "median"),
                       pct_closer_to_other_group=("inter_minus_intra", lambda d: 100 * (d < 0).mean()))
                  .reset_index())
    summary["inter_over_intra_median"] = summary.inter_median / summary.intra_median
    summary.insert(0, "gene", a.gene)
    summary.to_csv(f"{prefix}_summary.tsv", sep="\t", index=False)
    print(summary.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, len(reps), figsize=(4.6 * len(reps), 4.4), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, name in zip(axes, reps):
        r = res[res.representation == name]
        for gi, g in enumerate(order):
            for j, kind in enumerate(("intra", "inter")):
                x = gi * 2.6 + j
                d = r.loc[r.group == g, kind].to_numpy()
                body = ax.violinplot([d], positions=[x], widths=0.85, showextrema=False)["bodies"][0]
                body.set_facecolor(COLORS[kind]); body.set_edgecolor(COLORS[kind]); body.set_alpha(0.35)
                ax.scatter(x + rng.uniform(-0.13, 0.13, len(d)), d, s=4, color=COLORS[kind],
                           alpha=0.55, linewidths=0, rasterized=True)
                q1, med, q3 = np.percentile(d, [25, 50, 75])
                ax.vlines(x, q1, q3, color="#0b0b0b", lw=3)
                ax.scatter(x, med, s=28, color="white", edgecolor="#0b0b0b", zorder=3)
        n = r.groupby("group").size()
        ax.set_xticks([0.5, 3.1], [f"{g} donors\n(n = {n[g]})" for g in order])
        ax.set_ylabel("Mean Euclidean distance"); ax.set_title(name, fontsize=10, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[k], label=lab) for k, lab in
               (("intra", "to own group (intra)"), ("inter", "to other group (inter)"))]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{a.gene}: per-donor mean distance to own vs other group, {' vs '.join(order)}",
                 fontsize=11, x=0.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(f"{prefix}_violin.{ext}", dpi=200)


if __name__ == "__main__":
    main()
