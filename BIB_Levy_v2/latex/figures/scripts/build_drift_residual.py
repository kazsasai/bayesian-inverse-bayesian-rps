#!/usr/bin/env python3
r"""Build fig_drift_residual.pdf -- the finite-sample drift-residual
extrapolation (Appendix, on-off-intermittency heuristic).

The inverse-Bayesian renewal re-injects the renewed hypothesis at the
finite-sample divergence delta = E[D_KL(p||phat)] ~ (N-1)/(2m), a residual
that biases the log-posterior walk away from the ideal driftless limit and
vanishes only as m->infinity. This figure shows the measured BIB-BIB
argmax-persistence exponent alpha as a function of that single residual,
reached either by enlarging the window m (RPS, N=3) or by halving the
alphabet (matching pennies, N=2): both collapse onto one trend that
extrapolates to the Sparre-Andersen 3/2 as delta -> 0.

Reproduced from deposited data (powerlaw TPL on the [5,10000] power-law
region, same as the window-sweep figure; both agents pooled):
  RPS  N=3  m=10/20/50/100  durations_bib-bib_m{m}_huge.json (4 designs)
  MP   N=2  m=50            prod_reruns/mp_bib_cache.json
First run computes + caches the points; subsequent runs read the cache.

Usage:  PAPERA_DATA=/path/to/data python build_drift_residual.py
"""
import json, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.constrained_layout.use"] = True
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figdata
FIG_DIR = str(figdata.FIG_DIR)
OUT_PDF = os.path.join(FIG_DIR, "fig_drift_residual.pdf")
OUT_PNG = os.path.join(FIG_DIR, "fig_drift_residual.png")
CACHE = os.path.join(HERE, "data", "drift_residual_cache.json")

DES = ["rs", "ra", "ss", "sa"]


def tpl_alpha(arr):
    import powerlaw
    arr = np.asarray(arr, float)
    arr = arr[(arr >= 5) & (arr <= 10000)]
    if arr.size < 30:
        return float("nan")
    return float(powerlaw.Fit(arr, discrete=True, verbose=False).truncated_power_law.alpha)


def compute_points():
    pts = []
    # RPS N=3 window sweep
    for m in [10, 20, 50, 100]:
        per, pooled = [], []
        for d in DES:
            p = figdata.find(f"simulation/reward_huge/data/reward_huge_{d}/durations_bib-bib_m{m}_huge.json")
            j = json.load(open(p))
            arr = [v for k in ("T_argmax1", "T_argmax2") for v in j.get(k, []) if v >= 1]
            per.append(tpl_alpha(arr)); pooled += arr
        per = [x for x in per if np.isfinite(x)]
        pts.append(dict(label=f"RPS $N=3$", N=3, m=m, delta=2.0 / (2 * m),
                        mean=float(np.mean(per)), sd=float(np.std(per, ddof=1)),
                        pooled=float(tpl_alpha(pooled))))
    # Matching pennies N=2, m=50
    mp = json.load(open(os.path.join(figdata.REPO, "prod_reruns", "mp_bib_cache.json")))
    per = [tpl_alpha([v for v in mp[k].get("argmax", []) if v >= 1]) for k in mp]
    per = [x for x in per if np.isfinite(x)]
    pts.append(dict(label="matching pennies $N=2$", N=2, m=50, delta=1.0 / (2 * 50),
                    mean=float(np.mean(per)), sd=float(np.std(per, ddof=1)),
                    pooled=float(np.mean(per))))
    return pts


def get_points():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    pts = compute_points()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(pts, open(CACHE, "w"), indent=2)
    return pts


def main():
    pts = get_points()
    rps = [p for p in pts if p["N"] == 3]
    mp = [p for p in pts if p["N"] == 2][0]
    conv = [p for p in rps if p["m"] >= 20] + [mp]          # converged-window points
    unconv = [p for p in rps if p["m"] < 20]                # m<=10: tail not converged

    plt.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.8, "lines.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8, "axes.spines.top": True,
        "axes.spines.right": True, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.375, 2.95))

    # 3/2 reference
    ax.axhline(1.5, ls="--", color="0.45", lw=1.2, zorder=1)
    ax.text(0.103, 1.505, r"$3/2$ (Sparre--Andersen)", fontsize=6.5,
            color="0.35", va="bottom", ha="right")

    # linear fit over the converged points, extrapolated to delta=0
    xc = np.array([p["delta"] for p in conv]); yc = np.array([p["mean"] for p in conv])
    slope, intc = np.polyfit(xc, yc, 1)
    xf = np.array([0.0, 0.055])
    ax.plot(xf, slope * xf + intc, "-", color="#1f77b4", lw=1.3, zorder=2,
            label=fr"linear fit ($\alpha_0={intc:.2f}$)")
    ax.plot([0.0], [intc], "*", color="#1f77b4", ms=9, mec="black",
            mew=0.4, zorder=5)

    # RPS converged points (filled circles + per-design SD)
    xr = [p["delta"] for p in rps if p["m"] >= 20]
    yr = [p["mean"] for p in rps if p["m"] >= 20]
    er = [p["sd"] for p in rps if p["m"] >= 20]
    ax.errorbar(xr, yr, yerr=er, fmt="o", color="#1f77b4", ms=5,
                mec="black", mew=0.4, capsize=2.5, lw=0, elinewidth=0.9,
                zorder=4, label=r"RPS $N=3$ (window $m$)")
    for p in [q for q in rps if q["m"] >= 20]:
        ax.annotate(fr"$m={p['m']}$", (p["delta"], p["mean"]),
                    textcoords="offset points", xytext=(0, -15), va="top",
                    ha="center", fontsize=6, color="0.3")

    # MP point (filled square)
    ax.errorbar([mp["delta"]], [mp["mean"]], yerr=[mp["sd"]], fmt="s",
                color="#d62728", ms=5.5, mec="black", mew=0.4, capsize=2.5,
                lw=0, elinewidth=0.9, zorder=4, label=r"matching pennies $N=2$ ($m=50$)")

    # unconverged m=10 (faded open marker, pooled value, excluded from fit)
    for p in unconv:
        ax.plot([p["delta"]], [p["pooled"]], "o", mfc="none", mec="0.6",
                mew=1.0, ms=5, zorder=3)
        ax.annotate(fr"$m={p['m']}$ (not conv.)", (p["delta"], p["pooled"]),
                    textcoords="offset points", xytext=(-3, 6), fontsize=6,
                    color="0.55", ha="right")

    ax.set_xlabel(r"drift residual $(N-1)/(2m)$")
    ax.set_ylabel(r"argmax exponent $\alpha_{\mathrm{TPL}}$")
    ax.set_xlim(-0.006, 0.108)
    ax.set_ylim(1.20, 1.56)
    ax.legend(loc="lower left", frameon=False, fontsize=6.3)
    ax.grid(False)

    fig.savefig(OUT_PDF); fig.savefig(OUT_PNG, dpi=300)
    print(f"alpha_0 (intercept) = {intc:.3f}   slope = {slope:.2f}")
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
