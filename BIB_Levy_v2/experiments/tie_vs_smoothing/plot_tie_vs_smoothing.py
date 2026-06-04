#!/usr/bin/env python3
"""
plot_tie_vs_smoothing.py
Summary figure for the 2x2 (tie_mode x JM-smoothing) ablation.
Reads combined_small.csv, draws:
  (a) argmax-persistence exponent alpha across the 4 designs x 4 conditions
  (b) mean posterior spread sigma_bar (collapse indicator)
Frozen / undefined-alpha cells (posterior pinned, argmax never switches) are
marked with a hatch and a 'frozen' tag.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "combined_small.csv")

plt.rcParams.update({
    "font.size": 11, "font.family": "sans-serif", "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

rows = list(csv.DictReader(open(CSV)))
designs = ["rs", "ra", "ss", "sa"]
# condition key -> (tie_mode, jm) ; label ; color
conds = [
    ("skip", "1",    "eq3 (skip), JM on",      "#2c5f8a"),
    ("uniform", "1", "caseA (uniform), JM on",  "#6e8b3d"),
    ("skip", "0",    "eq3 (skip), JM OFF",      "#c0392b"),
    ("uniform", "0", "caseA (uniform), JM OFF", "#e08e3c"),
]

def get(design, tie, jm, field):
    for r in rows:
        if r["design"] == design and r["tie_mode"] == tie and r["jm_smoothing"] == jm:
            return r[field]
    return None

def fval(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 4.8))
x = np.arange(len(designs))
nb = len(conds)
w = 0.80 / nb

for k, (tie, jm, lab, col) in enumerate(conds):
    alphas = [fval(get(d, tie, jm, "alpha")) for d in designs]
    ns = [int(float(get(d, tie, jm, "n_durations"))) for d in designs]
    xpos = x + (k - (nb - 1) / 2) * w
    # frozen cell: alpha nan/<=1 boundary AND very few durations
    heights = [a if (np.isfinite(a) and a > 0.01) else 0.0 for a in alphas]
    bars = axA.bar(xpos, heights, w, color=col, edgecolor="black", lw=0.5, label=lab)
    for xi, (a, n) in enumerate(zip(alphas, ns)):
        if (not np.isfinite(a)) or a <= 0.01 or n <= 110:
            # mark frozen
            axA.bar(xpos[xi], 3.0, w, color="none", edgecolor=col,
                    lw=0.8, hatch="////", zorder=0)
            axA.text(xpos[xi], 0.06, "frozen", rotation=90, ha="center",
                     va="bottom", fontsize=7, color=col)

axA.axhline(1.0, ls="--", color="#b8860b", lw=1.0, alpha=0.7)
axA.axhline(1.5, ls=":", color="0.5", lw=1.0, alpha=0.7)
axA.axhspan(1.30, 1.45, color="0.85", alpha=0.5, zorder=-1)
axA.set_xticks(x); axA.set_xticklabels(designs)
axA.set_ylim(0, 3.0)
axA.set_ylabel(r"argmax-persistence exponent $\alpha$")
axA.set_title(r"(a) $\alpha$: JM-on stays L\'evy ($\sim$1.3--1.4); "
              "JM-off collapses", fontsize=10.5)
axA.legend(fontsize=8, loc="upper right", ncol=1, framealpha=0.95)

for k, (tie, jm, lab, col) in enumerate(conds):
    sig = [fval(get(d, tie, jm, "sigma_bar")) for d in designs]
    xpos = x + (k - (nb - 1) / 2) * w
    axB.bar(xpos, sig, w, color=col, edgecolor="black", lw=0.5, label=lab)
axB.axhspan(0.12, 0.16, color="#9fc5e8", alpha=0.45, zorder=-1)
axB.text(3.45, 0.14, "healthy\non-off band", fontsize=7.5, va="center", color="#2c5f8a")
axB.set_xticks(x); axB.set_xticklabels(designs)
axB.set_ylim(0, 0.34)
axB.set_ylabel(r"mean posterior spread $\bar\sigma(P(h))$")
axB.set_title(r"(b) $\bar\sigma$: JM-off over-concentrates "
              r"($\bar\sigma\!\to\!0.30$, frozen)", fontsize=10.5)

fig.suptitle(
    "Does tie$\\rightarrow$uniform replace the conditional JM smoothing? "
    "(2$\\times$2 ablation, bib--bib, $N_h=10$, $T=2000$, 100 runs)\n"
    "Answer: no --- without JM the posterior pins to one hypothesis and the "
    "argmax freezes, even with tie$\\rightarrow$uniform.",
    fontsize=10.5, y=1.08)
fig.subplots_adjust(wspace=0.22)
out = os.path.join(HERE, "fig_tie_vs_smoothing_small.png")
fig.savefig(out, dpi=200)
fig.savefig(out.replace(".png", ".pdf"))
print("wrote:", out)
