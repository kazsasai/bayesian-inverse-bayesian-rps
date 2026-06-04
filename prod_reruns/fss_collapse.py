#!/usr/bin/env python3
"""
Method-matched finite-size-scaling (FSS) collapse exponent z for the argmax-persistence
and laminar-phase observables, to test whether they share z (close seam B) using ONE
identical, reproducible pipeline.

Model (manuscript form): P(T|Nh) = T^{-alpha} F(T Nh^z)  =>  for the CCDF,
    C(t|Nh) * Nh^{-w}  collapses as a function of  t * Nh^z .
We find (z, w) that minimise the across-curve spread of the rescaled CCDFs over a fixed
scaling window, by 2-D grid + local refine. Same window/method for both observables.

Inputs (all already in the repo / produced earlier):
  argmax  : simulation/reward_huge/data/reward_huge_v2_<d>/durations_bib-bib_m50_huge.json (Nh=10)
            simulation/reward_huge/data/reward_huge_v3_<d>_h{3,6,15,20}/durations_bib-bib_m50_huge.json
  laminar : prod_reruns/laminar_nhsweep_cache.json  (Nh sweep re-sim, max_h P(h)>0.4, both agents)
Reproducible: deterministic; pooled over the 4 designs; CCDF window C in [CLO, CHI].
"""
import json, os, sys, numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "simulation").is_dir())
NH = [3, 6, 10, 15, 20]
DES = ["rs", "ra", "ss", "sa"]
CLO, CHI = 1e-3, 3e-1          # CCDF scaling window (knee/tail region; excludes top plateau & noisy floor)


def ccdf(arr):
    a = np.sort(np.asarray([x for x in arr if x >= 1], float))
    n = len(a)
    ux, idx = np.unique(a, return_index=True)
    C = 1.0 - idx / n           # P(X >= ux)
    return ux, C


def argmax_pool(nh):
    out = []
    for d in DES:
        p = (ROOT / "simulation/reward_huge/data" /
             (f"reward_huge_v2_{d}" if nh == 10 else f"reward_huge_v3_{d}_h{nh}") /
             "durations_bib-bib_m50_huge.json")
        if p.exists():
            j = json.load(open(p))
            for k in ("T_argmax1", "T_argmax2"):
                out += j.get(k, [])
    return out


def laminar_pool(nh, cache):
    return [v for k, r in cache.items() if int(k.split("|")[1]) == nh for v in r["lengths"]]


def curves(pool_fn, nhs):
    cs = {}
    for nh in nhs:
        t, C = ccdf(pool_fn(nh))
        m = (C >= CLO) & (C <= CHI) & (t >= 2)
        if m.sum() >= 8:
            cs[nh] = (np.log10(t[m]), np.log10(C[m]))   # (logt, logC) in the window
    return cs


def collapse_residual(z, w, cs):
    """across-curve variance of rescaled logC on a common rescaled-logt grid."""
    nhs = sorted(cs)
    X = {nh: cs[nh][0] + z * np.log10(nh) for nh in nhs}      # logt + z logNh
    Y = {nh: cs[nh][1] + w * np.log10(nh) for nh in nhs}      # logC + w logNh
    lo = max(x.min() for x in X.values()); hi = min(x.max() for x in X.values())
    if hi - lo < 0.3:
        return np.inf
    grid = np.linspace(lo, hi, 40)
    ys = []
    for nh in nhs:
        order = np.argsort(X[nh])
        ys.append(np.interp(grid, X[nh][order], Y[nh][order]))
    ys = np.array(ys)
    return float(np.mean(np.var(ys, axis=0)))


def optimise(cs):
    zg = np.arange(0.3, 3.21, 0.05); wg = np.arange(-1.5, 3.51, 0.1)
    best = (np.inf, None, None)
    for z in zg:
        for w in wg:
            r = collapse_residual(z, w, cs)
            if r < best[0]:
                best = (r, z, w)
    _, z0, w0 = best
    # local refine
    for z in np.arange(z0 - 0.05, z0 + 0.051, 0.01):
        for w in np.arange(w0 - 0.1, w0 + 0.101, 0.02):
            r = collapse_residual(z, w, cs)
            if r < best[0]:
                best = (r, z, w)
    return best  # (resid, z, w)


def z_se(cs, zbest, wbest):
    """1-sigma on z from the curvature of residual(z) near the optimum (w re-min at each z)."""
    def rz(z):
        return min(collapse_residual(z, w, cs) for w in np.arange(wbest - 0.4, wbest + 0.41, 0.05))
    r0 = rz(zbest); dz = 0.05
    rp = rz(zbest + dz); rm = rz(zbest - dz)
    curv = (rp - 2 * r0 + rm) / dz**2
    if curv <= 0:
        return np.nan
    # residual ~ variance; scale so that delta-resid = r0/dof at 1 sigma (heuristic)
    npts = 40 * len(cs)
    sigma2 = r0 / max(npts - 2, 1)
    return float(np.sqrt(2 * sigma2 / curv))


def run(label, pool_fn, nhs):
    cs = curves(pool_fn, nhs)
    if len(cs) < 3:
        print(f"{label}: too few usable Nh ({sorted(cs)})"); return None
    resid, z, w = optimise(cs)
    se = z_se(cs, z, w)
    print(f"{label:8s} Nh={sorted(cs)}: z={z:.3f} +/- {se:.3f}  (w={w:.2f}, collapse resid={resid:.4f})")
    return z, se, resid, sorted(cs)


if __name__ == "__main__":
    cache = json.load(open(HERE / "laminar_nhsweep_cache.json"))
    print("=== FSS collapse exponent z (method-matched; manuscript form P=T^-a F(T Nh^z)) ===")
    print(f"CCDF window C in [{CLO},{CHI}], pooled over designs {DES}\n")
    for nhs, tag in [([3, 6, 10, 15], "excl-boundary {3,6,10,15}"),
                     ([3, 6, 10, 15, 20], "all {3,6,10,15,20}")]:
        print(f"--- {tag} ---")
        run("argmax", argmax_pool, nhs)
        run("laminar", lambda nh: laminar_pool(nh, cache), nhs)
        print()
