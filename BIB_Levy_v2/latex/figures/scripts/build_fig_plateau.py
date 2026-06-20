#!/usr/bin/env python3
r"""Rebuild fig_plateau.pdf (Stage-1 reframe, single panel): BIB-BIB plateau-length
CCDF collapsed onto one heavy-tailed family (pooled alpha~1.23) plus BO-BO as two
initialization families (random rs+ra, structured ss+sa; no exponent).  Gray
dashed on-off alpha=3/2 reference.  Plateau: maximal argmax-stable intervals with
P(h*)>theta=0.3 for >= f=0.8 of the run, pooled over 6 sharpness values, agent 1.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.constrained_layout.use"] = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
from build_Fig3_universality import ccdf_ds

OUT = figdata.FIG_DIR / "fig_plateau.pdf"
DATA = "simulation/analyze_sharpness_plateau/data/sharpness_plateau"
THETA, FRAC = 0.3, 0.8
ALPHAS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
C_BIB, C_BO_R, C_BO_S = "#c1272d", "#2c5f8a", "#7fadd1"

def plat(argmax, P):
    pa = P[np.arange(P.shape[0]), argmax]
    b = np.concatenate([[0], np.flatnonzero(np.diff(argmax)) + 1, [len(argmax)]])
    out = []
    for i in range(len(b) - 1):
        seg = pa[b[i]:b[i + 1]]
        if (seg > THETA).mean() >= FRAC and (b[i + 1] - b[i]) >= 1:
            out.append(b[i + 1] - b[i])
    return out

def pool(designs, pair):
    o = []
    for d in designs:
        for s in ALPHAS:
            f = figdata.find(f"{DATA}/{d}_a{int(s*100):02d}_{pair}.npz")
            if not f or not os.path.exists(f):
                continue
            z = np.load(f); am, P = z["argmax1"], z["P1"]
            for r in range(am.shape[0]):
                o += plat(am[r], P[r])
    return np.asarray(o, float)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.8, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "axes.spines.top": True, "axes.spines.right": True,
    "pdf.fonttype": 42, "ps.fonttype": 42})

fig, ax = plt.subplots(figsize=(3.7, 3.1))
x, p = ccdf_ds(pool(["rs", "ra", "ss", "sa"], "bib-bib"))
ax.loglog(x, p, color=C_BIB, lw=1.6, label=r"BIB-BIB ($\alpha\approx 1.23$)")
for grp, col, lab in ((("rs", "ra"), C_BO_R, "BO-BO (random init)"),
                      (("ss", "sa"), C_BO_S, "BO-BO (structured init)")):
    x, p = ccdf_ds(pool(list(grp), "bo-bo"))
    ax.loglog(x, p, color=col, lw=1.4, ls="--", label=lab)
xr = np.logspace(np.log10(3), np.log10(1.0e3), 50)
yr = 0.5 * (xr / xr[0]) ** (-0.5)
ax.loglog(xr, yr, ":", color="0.4", lw=1.3, label=r"on-off $\alpha=3/2$", zorder=0)
ax.set_xlabel(r"plateau length $T_{\mathrm{pl}}$ (steps)")
ax.set_ylabel(r"CCDF $P(T_{\mathrm{pl}} \geq t)$")
ax.set_ylim(1e-5, 1.5)
ax.legend(loc="lower left", frameon=False, fontsize=6.8)
ax.grid(False)
fig.savefig(OUT)
fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=160)
print("Saved:", OUT)
