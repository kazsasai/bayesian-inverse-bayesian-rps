#!/usr/bin/env python3
r"""Regenerate prong2_part1_reduced.pdf (SI Sec. 1, Fig. \ref{si:fig-driftscaling}).

Reduced log-posterior walk  x_{t+1} = x_t + eta_t  with Gaussian increments of
tunable mean drift d = <eta> and unit scale.  We measure the FIRST-RETURN time
to the origin (the time until x first crosses back through 0 after the opening
step) -- the reduced-model analogue of the argmax-persistence / laminar time.

  (A) First-return CCDFs P(T >= tau) follow the Sparre-Andersen tau^{-3/2} pdf
      (CCDF slope -1/2) with an exponential cutoff tau* ~ d^{-2} that retreats
      to infinity as d -> 0, so d = 0 is the critical line.
  (B) Rescaling tau -> tau d^2 collapses the d > 0 curves onto a single scaling
      function, confirming  P(tau; d) = tau^{-3/2} G(tau d^2).

Self-contained: numpy + matplotlib only, no external data.
Usage:  python make_prong2_part1_reduced.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.constrained_layout.use"] = True
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE / "prong2_part1_reduced.pdf"
OUT_PNG = HERE / "prong2_part1_reduced.png"

DRIFTS = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16]
SIGMA = 1.0           # increment scale; sets tau* ~ (SIGMA/d)^2
M = 200_000           # independent walks per drift
T_MAX = 30_000        # cap on the first-return time we track
BARRIER = 250.0       # absorb walks that escape this far (return then negligible)
SEED = 11

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.3,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "axes.spines.top": True, "axes.spines.right": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def first_return_times(d, rng):
    """First-return-to-origin times for M biased Gaussian walks.

    Each walk starts at 0, takes one opening step (fixing the sign s0), then
    runs until x crosses back through 0 (x*s0 <= 0) -> return, or escapes past
    +BARRIER in the s0 direction (return thereafter negligible), or hits T_MAX.
    Non-returning walks (escaped or censored at T_MAX) keep an infinite return
    time: at finite drift a fraction of walks escape to infinity and never
    return, so the CCDF plateaus at the escape probability P_esc(d) -- this is
    the 'cutoff' of the tau^{-3/2} law, and P_esc ~ d is what makes the curves
    collapse under tau d^2.  Returns the full length-M array (with +inf)."""
    x = rng.normal(d, SIGMA, M)            # opening step away from 0
    s0 = np.where(x >= 0, 1.0, -1.0)
    ret = np.full(M, np.inf)               # +inf == never returned
    active = np.ones(M, dtype=bool)
    for t in range(2, T_MAX + 1):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        x[idx] += rng.normal(d, SIGMA, idx.size)
        proj = x[idx] * s0[idx]
        returned = proj <= 0.0
        escaped = proj > BARRIER
        ret[idx[returned]] = t
        active[idx[returned]] = False
        active[idx[escaped]] = False       # leaves ret = +inf (no finite return)
    return ret


def ccdf(times, grid):
    """P(T >= tau) over ALL walks (length-M array; +inf entries count as >= any
    finite tau, producing the escape-probability plateau)."""
    times = np.sort(times)
    n = times.size
    # number of samples >= tau  ==  n - searchsorted(times, tau, 'left')
    ge = n - np.searchsorted(times, grid, side="left")
    return ge / n


def main():
    rng = np.random.default_rng(SEED)
    grid = np.unique(np.round(np.logspace(0, np.log10(T_MAX), 60)).astype(int))
    grid = grid[grid >= 1]

    results = {}
    for d in DRIFTS:
        rt = first_return_times(d, rng)
        c = ccdf(rt, grid)
        results[d] = (grid.astype(float), c, rt.size)
        fin = rt[np.isfinite(rt)]
        p_esc = float(np.mean(~np.isfinite(rt)))
        print(f"  d={d:<5}: P_esc={p_esc:.4f}  finite median={np.median(fin):.0f}  "
              f"finite max={int(fin.max())}")

    # slope check on the d=0 power-law band (expect CCDF ~ -1/2).
    g0, c0, _ = results[0.0]
    band = (g0 >= 3) & (g0 <= 300) & (c0 > 0)
    slope = np.polyfit(np.log10(g0[band]), np.log10(c0[band]), 1)[0]
    print(f"  d=0 CCDF log-log slope on [3,300] = {slope:.3f}  (target -0.5)")

    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(DRIFTS)))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.05))

    # ---- Panel A: raw first-return CCDFs ----
    for d, col in zip(DRIFTS, colors):
        g, c, _ = results[d]
        m = c > 0
        axA.loglog(g[m], c[m], color=col, label=f"$d={d:g}$")
    ref = g0[(g0 >= 1) & (g0 <= 3e4)]
    axA.loglog(ref, 0.9 * ref ** -0.5, "--", color="black", lw=1.1,
               label=r"slope $-1/2$" "\n" r"(pdf $\tau^{-3/2}$)")
    axA.set_xlabel(r"First-return time  $\tau$")
    axA.set_ylabel(r"CCDF  $P(T \geq \tau)$")
    axA.set_ylim(1e-4, 1.4)
    axA.legend(loc="lower left", frameon=False, fontsize=7.5, ncol=1)
    axA.text(-0.16, 1.02, "(a)", transform=axA.transAxes,
             fontsize=11, fontweight="bold", va="bottom", ha="left")

    # ---- Panel B: tau d^2 collapse (d > 0) ----
    # Only show the scaling regime tau >= XMIN_B: at the very smallest tau the
    # discrete startup (P~1, so P*tau^{1/2}~tau^{1/2}) makes a small upward kink
    # before the plateau; dropping it leaves the clean collapse.
    XMIN_B = 12
    for d, col in zip(DRIFTS, colors):
        if d == 0.0:
            continue
        g, c, _ = results[d]
        m = (c > 0) & (g >= XMIN_B)
        axB.loglog(g[m] * d ** 2, c[m] * np.sqrt(g[m]), color=col, label=f"$d={d:g}$")
    axB.set_xlabel(r"Scaled time  $\tau d^{2}$")
    axB.set_ylabel(r"$P(T \geq \tau)\,\tau^{1/2}$")
    axB.legend(loc="upper left", frameon=False, fontsize=7.5)
    axB.text(0.97, 0.05, r"collapse onto a single" "\n" r"scaling function $G(\tau d^{2})$",
             transform=axB.transAxes, fontsize=8, va="bottom", ha="right")
    axB.text(-0.16, 1.02, "(b)", transform=axB.transAxes,
             fontsize=11, fontweight="bold", va="bottom", ha="left")

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print("wrote", OUT_PDF.name, "/", OUT_PNG.name)


if __name__ == "__main__":
    main()
