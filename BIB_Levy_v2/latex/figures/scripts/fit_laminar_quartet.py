#!/usr/bin/env python3
r"""Fit the BIB-BIB laminar-phase exponent per design (the main-text quartet),
LIVE from the deposited pmax data (no hard-coded constants).

Laminar phase = maximal run of steps with max_h P(h) > theta (theta = 0.4),
pooled over both agents (pmax1, pmax2) and the 20 runs; fitted with a
truncated power law (powerlaw, Clauset auto-x_min).

Source: simulation_tie_mode_ablation/data/scheme_ablation/huge_laminar/
        pmax_<design>_eq3.npz  (arrays pmax1, pmax2 of shape (n_runs, 1e5))
resolved via figdata ($PAPERA_DATA -> <repo>/data -> <repo>).

Manuscript reports {1.338, 1.340, 1.339, 1.342}, mean 1.340 +/- 0.002,
cross-design range 0.004.

Usage:  PAPERA_DATA=/path/to/data python fit_laminar_quartet.py
"""
import os, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import powerlaw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figdata

THETA = 0.4
LAM_DIR = "simulation_tie_mode_ablation/data/scheme_ablation/huge_laminar"
DESIGNS = ["rs", "ra", "ss", "sa"]


def laminar_lengths(design, theta=THETA):
    z = np.load(figdata.find(f"{LAM_DIR}/pmax_{design}_eq3.npz"))
    out = []
    for key in ("pmax1", "pmax2"):
        for run in z[key]:
            il = run > theta
            d = np.diff(il.astype(np.int8))
            b = np.concatenate([[0], np.flatnonzero(d) + 1, [il.size]])
            rl = np.diff(b)
            st = il[b[:-1]]
            out += rl[st].tolist()
    return np.asarray(out, float)


def main():
    print(f"=== Live BIB-BIB laminar exponent per design (theta={THETA}, TPL/Clauset) ===")
    alphas = []
    for d in DESIGNS:
        L = laminar_lengths(d)
        L = L[L >= 1]
        fit = powerlaw.Fit(L, discrete=True, verbose=False)
        a = float(fit.truncated_power_law.alpha)
        alphas.append(a)
        print(f"  {d}: n={L.size:>7d}  x_min={int(fit.xmin):>3d}  alpha_TPL={a:.3f}")
    a = np.array(alphas)
    print(f"\n  quartet = {[round(x,3) for x in alphas]}")
    print(f"  mean = {a.mean():.3f} +/- {a.std(ddof=1):.4f}   cross-design range = {a.max()-a.min():.4f}")
    print("  Manuscript: {1.338,1.340,1.339,1.342}, mean 1.340 +/- 0.002, range 0.004")


if __name__ == "__main__":
    main()
