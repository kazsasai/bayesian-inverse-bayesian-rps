#!/usr/bin/env python3
r"""Regenerate Table~\ref{tab:sharpness} (the $P(d\mid h)$-sharpness sweep
exponent statistics) from the DEPOSITED sharpness-sweep npz, using the same
theta=0.4 + powerlaw truncated-power-law / Clauset auto-x_min pipeline as
regen_bo_exponents.py (both agents pooled per condition). The sweep is over
the two structured designs (ss, sa) and alpha_init in {0.4..0.9}.

Reproduces (manuscript Table~\ref{tab:sharpness}):
  Argmax persistence  BIB 1.48 +/- 0.02  [1.44, 1.51]
                      BO  1.74 +/- 0.19  [1.43, 2.08]
  Laminar phase       BIB 1.34 +/- 0.02  [1.30, 1.37]
                      BO  1.77 +/- 0.37  [1.48, 2.75]

Data via figdata ($PAPERA_DATA -> <repo>/data -> in-repo simulation tree):
  simulation/analyze_sharpness_plateau/data/sharpness_plateau/<d>_a<aa>_<pair>.npz
  arrays: argmax1/argmax2 (runs x steps), P1/P2 (runs x steps x Nh)

Usage:  PAPERA_DATA=/path/to/data python regen_sharpness_table.py
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import powerlaw
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figdata

REL = "simulation/analyze_sharpness_plateau/data/sharpness_plateau"
THETA = 0.4
SHARP = [40, 50, 60, 70, 80, 90]
DES = ["ss", "sa"]


def runs_of(seq):
    seq = np.asarray(seq)
    if seq.size == 0:
        return np.array([], int)
    b = np.concatenate([[0], np.flatnonzero(np.diff(seq)) + 1, [seq.size]])
    return np.diff(b)


def argmax_pool(z):
    out = []
    for key in ("argmax1", "argmax2"):
        for run in z[key]:
            out += runs_of(run).tolist()
    return np.asarray([v for v in out if v >= 1], float)


def laminar_pool(z):
    out = []
    for key in ("P1", "P2"):
        for run in z[key]:
            il = run.max(axis=1) > THETA
            b = np.concatenate([[0], np.flatnonzero(np.diff(il.astype(np.int8))) + 1, [il.size]])
            rl = np.diff(b)
            out += rl[il[b[:-1]]].tolist()
    return np.asarray(out, float)


def tpl(a):
    a = np.asarray(a, float); a = a[a >= 1]
    return float(powerlaw.Fit(a, discrete=True, verbose=False).truncated_power_law.alpha)


def stats(v):
    v = np.array(v)
    return f"mean {v.mean():.2f}  SD {v.std(ddof=1):.2f}  range [{v.min():.2f}, {v.max():.2f}]"


if __name__ == "__main__":
    for pair, scheme in [("bib-bib", "BIB"), ("bo-bo", "BO")]:
        arg, lam = [], []
        for d in DES:
            for a in SHARP:
                z = np.load(figdata.find(f"{REL}/{d}_a{a}_{pair}.npz"))
                arg.append(tpl(argmax_pool(z)))
                lam.append(tpl(laminar_pool(z)))
        print(f"=== {scheme} ({pair}) ===")
        print(f"  Argmax  : {stats(arg)}")
        print(f"  Laminar : {stats(lam)}")
