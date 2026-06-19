#!/usr/bin/env python3
r"""Rebuild the universality figures after the Sec. III / Sec. IV.B split (case 3):
two single-panel figures instead of the old 2-panel fig_universality.

  fig_universality.pdf  (Sec. III)    -- argmax-persistence CCDFs
  fig_laminar.pdf       (Sec. IV.B)   -- laminar-phase CCDFs (theta=0.4)

Each panel overlays the four BIB-BIB designs (red, collapsed) and BO-BO as two
initialization families (random = rs+ra, structured = ss+sa; blue dashed, raw
CCDFs, no exponent).  Gray dotted on-off alpha=3/2 reference.  Both panels are
single-column sized so the two sibling CCDF figures match.  Reuses the data
loaders of build_Fig3_universality.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.constrained_layout.use"] = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
from build_Fig3_universality import argmax_durations, laminar_lengths, ccdf_ds, DES, THETA

C_BIB = "#c1272d"
C_BO_R, C_BO_S = "#2c5f8a", "#7fadd1"   # BO random / structured init
MK = {"rs": "o", "ra": "s", "ss": "^", "sa": "D"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.6, "axes.linewidth": 0.8, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "axes.spines.top": True, "axes.spines.right": True, "pdf.fonttype": 42, "ps.fonttype": 42})


def panel(ax, getter, xlabel, ylab, ylim):
    # BIB: four designs, red, collapsed
    for i, d in enumerate(DES):
        x, p = ccdf_ds(getter(d, "bib-bib"))
        ax.loglog(x, p, color=C_BIB, lw=1.1, marker=MK[d], markersize=3.0,
                  markevery=0.14, markeredgecolor="black", markeredgewidth=0.35,
                  alpha=0.85, label="BIB-BIB" if i == 0 else None)
    # BO: two initialization families (raw CCDFs, dashed)
    for grp, col, lab in ((("rs", "ra"), C_BO_R, "BO-BO (random init)"),
                          (("ss", "sa"), C_BO_S, "BO-BO (structured init)")):
        pool = np.concatenate([getter(d, "bo-bo") for d in grp])
        x, p = ccdf_ds(pool)
        ax.loglog(x, p, color=col, lw=1.4, ls="--", alpha=0.95, label=lab)
    # on-off 3/2 reference (CCDF slope -1/2)
    xr = np.logspace(np.log10(4), np.log10(np.max([l.get_xdata().max() for l in ax.lines])), 50)
    yr = 0.55 * (xr / xr[0]) ** (-0.5)
    ax.loglog(xr, yr, ":", color="0.4", lw=1.3, label=r"on-off $\alpha=3/2$")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylab); ax.set_ylim(*ylim); ax.grid(False)
    ax.legend(loc="lower left", frameon=False, fontsize=6.6)


def emit(name, getter, xlabel, ylab, ylim):
    fig, ax = plt.subplots(figsize=(3.4, 3.05))
    panel(ax, getter, xlabel, ylab, ylim)
    out = figdata.FIG_DIR / name
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print("Saved:", out)


# fig_universality.pdf : argmax persistence (Sec. III)
emit("fig_universality.pdf", argmax_durations,
     r"$T_{\mathrm{argmax}}$ (steps)", r"CCDF $P(T \geq t)$", (1e-6, 1.5))

# fig_laminar.pdf : laminar phase (Sec. IV.B)
emit("fig_laminar.pdf", laminar_lengths,
     r"$L_{\mathrm{laminar}}$ (steps, $\max_h P(h)>%.1f$)" % THETA,
     r"CCDF $P(L \geq t)$", (1e-5, 1.5))
