#!/usr/bin/env python
"""Ancestry groups for one comparison, from GTEx self-reported race (dbGaP pht002742).

  eur_afr     EUR = White (RACE 3), AFR = Black or African American (RACE 2); every
              other donor is left out of the comparison
  eur_noneur  EUR = White, nonEUR = every other donor

Output: TSV with columns IID, GROUP, listing only the comparison's donors (all GTEx
subjects, not only those with WGS -- consumers intersect with their own donors).
"""
import argparse

import pandas as pd

LABELS = {
    "eur_afr": {3: "EUR", 2: "AFR"},
    "eur_noneur": None,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phenotypes", required=True, help="dbGaP GTEx_Subject_Phenotypes .txt.gz")
    ap.add_argument("--comparison", required=True, choices=sorted(LABELS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    race = pd.read_csv(a.phenotypes, sep="\t", comment="#", usecols=["SUBJID", "RACE"])
    labels = LABELS[a.comparison]
    if labels is None:
        group = race.RACE.map(lambda r: "EUR" if r == 3 else "nonEUR")
    else:
        group = race.RACE.map(labels)
    out = pd.DataFrame({"IID": race.SUBJID, "GROUP": group}).dropna()
    out.to_csv(a.out, sep="\t", index=False)
    print(out.GROUP.value_counts().to_string())


if __name__ == "__main__":
    main()
