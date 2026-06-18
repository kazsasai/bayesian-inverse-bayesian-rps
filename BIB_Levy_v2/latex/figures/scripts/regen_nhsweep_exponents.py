#!/usr/bin/env python3
r"""Regenerate the BIB-BIB and BO-BO argmax Nh-sweep TPL exponents that back
the manuscript's boundary-Nh descriptions, from the DEPOSITED data, using the
same powerlaw truncated-power-law / Clauset auto-x_min pipeline as
regen_bo_exponents.py (both agents pooled). Makes the boundary numbers
reproducible from deposited code rather than hard-coded constants.

Reproduces (manuscript Sec.~IV.A, $\Nh$-sweep):
  BIB-BIB: at Nh=3 ~1.62; Nh=15 ~1.40-1.44; Nh=20 rs~1.78, others ~1.49-1.57
           -> span ~[1.40, 1.78]
  BO-BO  : design-conditional, no universality -> span ~[1.18, 2.81]

Data via figdata ($PAPERA_DATA -> <repo>/data -> in-repo simulation tree):
  Nh=10 : simulation/reward_huge/data/reward_huge_<d>/durations_<pair>_m50_huge.json
  Nh!=10: simulation/reward_huge/data/reward_huge_v3_<d>_h<nh>/durations_<pair>_m50_huge.json

Usage:  PAPERA_DATA=/path/to/data python regen_nhsweep_exponents.py
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import powerlaw
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figdata

DES = ["rs", "ra", "ss", "sa"]
NH = [3, 6, 10, 15, 20]


def _subdir(d, nh):
    return f"reward_huge_{d}" if nh == 10 else f"reward_huge_v3_{d}_h{nh}"


def argmax_pool(d, nh, pair):
    p = figdata.find(f"simulation/reward_huge/data/{_subdir(d,nh)}/durations_{pair}_m50_huge.json")
    j = json.load(open(p))
    return np.asarray([v for k in ("T_argmax1", "T_argmax2") for v in j.get(k, []) if v >= 1], int)


def tpl(a):
    a = np.asarray(a, float); a = a[a >= 1]
    return float(powerlaw.Fit(a, discrete=True, verbose=False).truncated_power_law.alpha)


if __name__ == "__main__":
    for pair in ["bib-bib", "bo-bo"]:
        print(f"=== {pair} argmax Nh-sweep (powerlaw TPL, Clauset auto x_min) ===")
        allv = []
        for d in DES:
            row = []
            for nh in NH:
                try:
                    a = round(tpl(argmax_pool(d, nh, pair)), 2); row.append((nh, a)); allv.append(a)
                except FileNotFoundError:
                    row.append((nh, None))
            print(f"  {d}: {row}")
        allv = [v for v in allv if v is not None]
        print(f"  -> span [{min(allv):.2f}, {max(allv):.2f}]")
