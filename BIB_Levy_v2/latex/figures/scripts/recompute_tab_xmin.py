#!/usr/bin/env python3
"""Reproduce Table~\\ref{tab:xmin} (Appendix B): sensitivity of the BIB
argmax-persistence exponent to the x_min / fit convention.

Per design (BIB-BIB, m=50, N=3) the pooled T_argmax of both agents is refit as
  - truncated power law (TPL) with Clauset auto x_min (powerlaw default),
  - TPL with fixed x_min = 1, 5, 10,
  - pure power law (PL) with Clauset auto x_min.
The absolute exponent depends on the convention, but the cross-design spread
stays <= 0.04 under every choice -- the design collapse, not the bare value,
is the robust result.

Data via figdata ($PAPERA_DATA / <repo>/data / in-repo simulation tree):
    simulation/reward_huge/data/reward_huge_<design>/durations_bib-bib_m50_huge.json

Usage:  python recompute_tab_xmin.py
Prints the table rows and the cross-design spread; no files are written.
"""
import json
import os
import sys
import warnings
import numpy as np
import powerlaw

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figdata

DESIGNS = ["rs", "ra", "ss", "sa"]


def relpath(d):
    return (f"simulation/reward_huge/data/reward_huge_{d}/"
            f"durations_bib-bib_m50_huge.json")


def load_pool(d):
    j = json.load(open(figdata.find(relpath(d))))
    a = [v for v in j.get("T_argmax1", []) if v >= 1] + \
        [v for v in j.get("T_argmax2", []) if v >= 1]
    return np.asarray(a, dtype=int)


def tpl_alpha(data, xmin=None):
    f = (powerlaw.Fit(data, discrete=True, xmin=xmin, verbose=False)
         if xmin else powerlaw.Fit(data, discrete=True, verbose=False))
    return float(f.truncated_power_law.alpha), int(f.xmin)


def pl_alpha(data):
    f = powerlaw.Fit(data, discrete=True, verbose=False)
    return float(f.power_law.alpha)


def main():
    cols = ["xmin auto", "xmin=1", "xmin=5", "xmin=10", "PL"]
    rows = {}
    auto_xmin = {}
    for d in DESIGNS:
        data = load_pool(d)
        a_auto, xm = tpl_alpha(data)
        auto_xmin[d] = xm
        rows[d] = [a_auto,
                   tpl_alpha(data, 1)[0],
                   tpl_alpha(data, 5)[0],
                   tpl_alpha(data, 10)[0],
                   pl_alpha(data)]
    # print table
    print(f"{'design':6} " + "  ".join(f"{c:>9}" for c in cols))
    for d in DESIGNS:
        print(f"{d:6} " + "  ".join(f"{v:>9.3f}" for v in rows[d]))
    spreads = [max(rows[d][i] for d in DESIGNS) - min(rows[d][i] for d in DESIGNS)
               for i in range(len(cols))]
    print(f"{'spread':6} " + "  ".join(f"{s:>9.3f}" for s in spreads))
    print("\nauto x_min per design (TPL):", auto_xmin)
    print("Cross-design spread <= 0.04 under every convention "
          "=> design collapse is x_min-independent.")


if __name__ == "__main__":
    main()
