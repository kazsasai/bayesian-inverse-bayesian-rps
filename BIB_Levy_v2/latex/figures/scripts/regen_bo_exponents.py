#!/usr/bin/env python3
r"""Regenerate the BO-BO exponents quoted in the manuscript from the DEPOSITED
data, using the SAME pipeline as the BIB quartets (powerlaw truncated power law,
Clauset auto x_min, pooled over both agents). This makes the BO numbers
reproducible from deposited code rather than hard-coded constants.

Reproduces:
  * argmax quartet at Nh=10, m=50, N=3   -> manuscript {2.04, 2.15, 1.51, 1.50}
  * laminar quartet (max_h P(h) > 0.4)   -> manuscript {2.13, 2.19, 1.44, 1.23}
  * argmax Nh-sweep (NH_SWEEP_BO)        -> span ~[1.18, 2.81]

Data via figdata ($PAPERA_DATA -> <repo>/data -> in-repo simulation tree):
  argmax  Nh=10 : simulation/reward_huge/data/reward_huge_<d>/durations_bo-bo_m50_huge.json
  argmax  Nh!=10: simulation/reward_huge/data/reward_huge_v3_<d>_h<nh>/durations_bo-bo_m50_huge.json
  laminar       : simulation_tie_mode_ablation/data/scheme_ablation/huge_laminar/pmax_<d>_bo.npz

Usage:  PAPERA_DATA=/path/to/data python regen_bo_exponents.py
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
THETA = 0.4
LAM_DIR = "simulation_tie_mode_ablation/data/scheme_ablation/huge_laminar"


def _subdir(d, nh):
    return f"reward_huge_{d}" if nh == 10 else f"reward_huge_v3_{d}_h{nh}"


def argmax_pool(d, nh):
    p = figdata.find(f"simulation/reward_huge/data/{_subdir(d,nh)}/durations_bo-bo_m50_huge.json")
    j = json.load(open(p))
    return np.asarray([v for k in ("T_argmax1", "T_argmax2") for v in j.get(k, []) if v >= 1], int)


def laminar_pool(d):
    z = np.load(figdata.find(f"{LAM_DIR}/pmax_{d}_bo.npz"))
    out = []
    for key in ("pmax1", "pmax2"):
        for run in z[key]:
            il = run > THETA
            b = np.concatenate([[0], np.flatnonzero(np.diff(il.astype(np.int8))) + 1, [il.size]])
            rl = np.diff(b)
            out += rl[il[b[:-1]]].tolist()
    return np.asarray(out, float)


def tpl(a):
    a = np.asarray(a, float); a = a[a >= 1]
    return float(powerlaw.Fit(a, discrete=True, verbose=False).truncated_power_law.alpha)


if __name__ == "__main__":
    print("=== BO-BO regenerated exponents (powerlaw TPL, Clauset auto x_min; uniform with BIB) ===")
    arg = [round(tpl(argmax_pool(d, 10)), 2) for d in DES]
    lam = [round(tpl(laminar_pool(d)), 2) for d in DES]
    print(f"argmax  Nh=10 (rs,ra,ss,sa) = {arg}   [manuscript {{2.04,2.15,1.51,1.50}}]")
    print(f"laminar       (rs,ra,ss,sa) = {lam}   [manuscript {{2.13,2.19,1.44,1.23}}]")
    print("\nargmax Nh-sweep (NH_SWEEP_BO):")
    allv = []
    for d in DES:
        row = []
        for nh in NH:
            try:
                a = round(tpl(argmax_pool(d, nh)), 2); row.append((nh, a)); allv.append(a)
            except FileNotFoundError:
                row.append((nh, None))
        print(f"  {d}: {row}")
    allv = [v for v in allv if v is not None]
    print(f"  -> span [{min(allv):.2f}, {max(allv):.2f}]   [manuscript ~1.2 to 2.8]")
