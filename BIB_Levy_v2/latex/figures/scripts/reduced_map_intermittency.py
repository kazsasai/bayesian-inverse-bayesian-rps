#!/usr/bin/env python3
r"""Reduced-map check of the on-off origin of the BIB argmax-persistence
exponent (Appendix \ref{app:heuristic}).

Two-hypothesis reduction.  With p = P(h_1), the Bayesian update is a Mobius
map p -> L1 p / (L1 p + L2 (1-p)); in the log-odds variable u = log[p/(1-p)]
it is purely additive,
        u_{t+1} = u_t + eta_t ,
with eta_t the per-step log-likelihood ratio (the driving).  At the symmetric
fixed point <eta>=0, so u is an unbiased random walk and the argmax (sign of u)
changes when u first returns to 0.  By the Sparre-Andersen theorem the
first-return time then has the universal on-off exponent P(T) ~ T^{-3/2}.

The conditional Jelinek-Mercer smoothing (fires only when min_i P(h_i) < 0.002)
acts as a reinjection at the simplex boundary: for N=2 the loser mass is reset
from <0.002 to ~0.016, i.e. |u| is reflected from the wall u_wall=log(0.998/
0.002)~6.21 to u_reinj=log(0.984/0.016)~4.12.  This caps the longest laminar
phases (the truncated-power-law cutoff) but leaves the bulk random walk -- and
hence the 3/2 law -- intact.  An *always-on* smoothing instead contracts u
toward 0 at every step (mean reversion), turning the walk into an
Ornstein-Uhlenbeck-like process whose return times are exponential: the heavy
tail is destroyed, reproducing the ablation of Appendix \ref{app:smoothing}.

Outputs (printed): power-law and truncated-power-law alpha for the conditional
model (expected ~3/2 for PL), and a truncated-power-law-vs-exponential
log-likelihood ratio for the conditional vs always-on models.

Usage:  python reduced_map_intermittency.py
Requires: numpy, powerlaw.  No external data.
"""
import numpy as np
import warnings
import powerlaw

warnings.simplefilter("ignore")

WALL = np.log(0.998 / 0.002)     # ~6.21 : min P(h)=0.002 boundary trigger
REINJ = np.log(0.984 / 0.016)    # ~4.12 : loser mass reset by conditional JM


def laminar_runs(mode, sigma=0.3, rho=0.97, T=2_000_000, seed=0):
    """Return the laminar-phase lengths (constant-argmax run lengths) of the
    reduced log-odds walk under 'conditional' or 'always-on' smoothing."""
    rng = np.random.default_rng(seed)
    u = 0.0
    sign = 1
    runs = []
    cur = 0
    eta = rng.normal(0.0, sigma, size=T)     # symmetric, zero-mean driving
    for t in range(T):
        u += eta[t]
        if mode == "conditional":
            if u > WALL:
                u = REINJ
            elif u < -WALL:
                u = -REINJ
        else:                                # always-on: contract every step
            u *= rho
        s = 1 if u >= 0 else -1
        if s == sign:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
            sign = s
    return np.array([r for r in runs if r >= 1], dtype=int)


def main():
    print("Reduced two-hypothesis log-odds walk (on-off mechanism)\n")
    for sigma in (0.2, 0.3, 0.5):
        L = laminar_runs("conditional", sigma=sigma)
        fit = powerlaw.Fit(L, discrete=True, verbose=False)
        print(f"  conditional, sigma={sigma}: n={L.size:>6}  xmin={int(fit.xmin):>2}"
              f"  PL_alpha={fit.power_law.alpha:.3f}"
              f"  TPL_alpha={fit.truncated_power_law.alpha:.3f}"
              f"  max={L.max()}")
    print("\n  => PL exponent ~ 3/2 (Sparre-Andersen on-off); the TPL convention\n"
          "     is pulled below 3/2 by the reinjection cutoff.\n")

    print("Ablation: conditional vs always-on smoothing")
    for mode in ("conditional", "always-on"):
        L = laminar_runs(mode, sigma=0.3, seed=1)
        fit = powerlaw.Fit(L, discrete=True, verbose=False)
        R, p = fit.distribution_compare("truncated_power_law", "exponential",
                                        normalized_ratio=True)
        print(f"  {mode:11}: mean={L.mean():.1f}  max={L.max():>5}"
              f"  PL_alpha={fit.power_law.alpha:.3f}"
              f"  TPLvsEXP R={R:+.1f} ({'heavy-tailed' if R > 5 else 'exponential-like'})")
    print("\n  => conditional keeps the heavy tail; always-on collapses to\n"
          "     exponential-like statistics (cf. Appendix on conditional smoothing).")


if __name__ == "__main__":
    main()
