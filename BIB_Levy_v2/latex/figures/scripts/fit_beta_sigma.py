#!/usr/bin/env python3
r"""Fit the finite-size exponent beta of the time-averaged posterior spread,
    sigma(P(h)) ~ N_h^{-beta},
LIVE from the deposited per-run sigma data (no hard-coded constants).

For each design (rs/ra/ss/sa) and pair (bib-bib, bo-bo) we read
    simulation/reward_huge/data/reward_huge_v3_<design>_h<Nh>/sigmas_<pair>_m50_huge.json
(each file = list of runs; per run we use the time-average sig{1,2}_mean of
std_i P(h_i), pooled over both agents), average over the 20 runs to get
sigma(N_h), then regress log sigma on log N_h. beta = -slope.

Data resolution: figdata searches $PAPERA_DATA, then <repo>/data, then <repo>.
The Zenodo *figure-only* archive contains N_h in {3,6,15,20}; N_h=10 sigma is in
the full archive only, so this script fits over whatever N_h are present and
reports the count. Manuscript values: beta_BIB = 1.067 +/- 0.008,
beta_BO = 1.278 +/- 0.020 (benchmark sigma ~ 1/N_h is beta = 1).

Usage:  PAPERA_DATA=/path/to/data python fit_beta_sigma.py
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figdata

DESIGNS = ["rs", "ra", "ss", "sa"]
PAIRS = {"bib-bib": "BIB", "bo-bo": "BO"}
NH_ALL = [3, 6, 10, 15, 20]


def sigma_for(design, pair, nh):
    """Mean over runs of the per-run time-averaged posterior spread (both agents)."""
    if nh == 10:
        rel = f"simulation/reward_huge/data/reward_huge_v2_{design}/sigmas_{pair}_m50_huge.json"
    else:
        rel = f"simulation/reward_huge/data/reward_huge_v3_{design}_h{nh}/sigmas_{pair}_m50_huge.json"
    try:
        path = figdata.find(rel)
    except (FileNotFoundError, Exception):
        return None
    runs = json.load(open(path))
    per_run = [0.5 * (r["sig1_mean"] + r["sig2_mean"]) for r in runs]
    return float(np.mean(per_run))


def fit_beta(nhs, sigs):
    x, y = np.log(np.asarray(nhs, float)), np.log(np.asarray(sigs, float))
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # standard error of the slope
    se = float(np.sqrt(ss_res / max(n - 2, 1) / np.sum((x - x.mean()) ** 2))) if n > 2 else float("nan")
    return -slope, se, r2


def main():
    print("=== Live beta fit: sigma(P(h)) ~ N_h^{-beta} (per-run sigma data) ===")
    results = {"BIB": [], "BO": []}
    for pair, tag in PAIRS.items():
        print(f"\n--- {tag} ({pair}) ---")
        for design in DESIGNS:
            pts = [(nh, sigma_for(design, pair, nh)) for nh in NH_ALL]
            pts = [(nh, s) for nh, s in pts if s is not None]
            if len(pts) < 3:
                print(f"  {design}: too few N_h with data ({[p[0] for p in pts]})")
                continue
            nhs, sigs = zip(*pts)
            beta, se, r2 = fit_beta(nhs, sigs)
            results[tag].append(beta)
            print(f"  {design}: N_h={list(nhs)}  beta={beta:.3f} +/- {se:.3f}  (R^2={r2:.3f})"
                  f"   sigma={[round(s,4) for s in sigs]}")
    for tag in ("BIB", "BO"):
        b = np.array(results[tag])
        if b.size:
            print(f"\n{tag}: pooled beta = {b.mean():.3f} +/- {b.std(ddof=1):.3f}  (n={b.size} designs)")
    print("\nManuscript: beta_BIB = 1.067 +/- 0.008 ; beta_BO = 1.278 +/- 0.020 "
          "(benchmark sigma~1/N_h => beta=1).")


if __name__ == "__main__":
    main()
