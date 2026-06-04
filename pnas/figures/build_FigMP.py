#!/usr/bin/env python3
"""Main-text generality figure: BIB on MATCHING PENNIES reproduces the RPS critical class.
(A) argmax-persistence CCDF, (B) laminar-phase CCDF, both designs (random init; sample/argmax),
with the 3/2-class reference. Reads prod_reruns/mp_bib_cache.json (re-fill at full scale to lock).
PNAS house style (despine, 8pt body, 11pt bold panel labels, frameon=False, no grid)."""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import powerlaw, warnings
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve()
ROOT = next(p for p in HERE.parents if (p / "prod_reruns").is_dir())
CACHE = ROOT / "prod_reruns" / "mp_bib_cache.json"
OUT_PDF = HERE.parent / "fig_mp_generality.pdf"
OUT_PNG = HERE.parent / "fig_mp_generality.png"

DES = {"randinit_sample_h10": ("rand+sample", "#1f77b4", "o"),
       "randinit_argmax_h10": ("rand+argmax", "#d62728", "D")}


def ccdf(arr, npts=300):
    a = np.sort(np.asarray([x for x in arr if x >= 1], float)); n = len(a)
    ux, idx = np.unique(a, return_index=True); p = (n - idx) / n
    if len(ux) > npts:
        sel = np.unique(np.round(np.logspace(0, np.log10(len(ux) - 1), npts)).astype(int))
        ux, p = ux[sel], p[sel]
    return ux, p


def alpha(arr):
    a = np.array([x for x in arr if x >= 1]); f = powerlaw.Fit(a, discrete=True, verbose=False)
    return f.truncated_power_law.alpha


def main():
    cache = json.load(open(CACHE))
    plt.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.8, "lines.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight"})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.05),
                                   gridspec_kw={"wspace": 0.26})
    for obs, ax, lab in [("argmax", axA, r"$T_{\mathrm{argmax}}$ (steps)"),
                         ("laminar", axB, r"$L_{\mathrm{laminar}}$ (steps)")]:
        for key, (name, col, mk) in DES.items():
            if key not in cache:
                continue
            x, p = ccdf(cache[key][obs])
            a = alpha(cache[key][obs])
            ax.loglog(x, p, color=col, lw=1.1, marker=mk, markersize=3.0, markevery=0.14,
                      markeredgecolor="black", markeredgewidth=0.3, alpha=0.95,
                      label=fr"{name} ($\alpha$={a:.2f})")
        xr = np.logspace(0.3, np.log10(2e3), 40); yr = 0.6 * (xr / xr[0]) ** (-0.5)
        ax.loglog(xr, yr, ":", color="0.45", lw=1.4, label=r"on-off ref $\alpha=3/2$")
        ax.set_xlabel(lab); ax.set_ylim(2e-5, 1.5)
        ax.legend(loc="lower left", frameon=False, fontsize=6.5)
    axA.set_ylabel(r"CCDF $P(T\geq t)$"); axB.set_ylabel(r"CCDF $P(L\geq t)$")
    for ax, L in [(axA, "A"), (axB, "B")]:
        ax.text(-0.17, 1.04, L, transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom", ha="left")
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight"); fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
