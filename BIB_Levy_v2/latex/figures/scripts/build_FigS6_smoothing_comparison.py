#!/usr/bin/env python3
"""
build_fig_smoothing.py
Regenerates fig_smoothing.pdf (paper Fig 10, Appendix A) as a
clean VECTOR PDF.

Why this script exists / what changed
-------------------------------------
The previous FigS6 was a raster image with a baked-in "Figure S6:" suptitle and
no panel labels, and its assembly script was lost. The original showed four
posterior-smoothing variants, but only TWO carry surviving per-step sigma
trajectory data in the workspace:

  * conditional smoothing      -> simulation_r2/.../reward_pilot       (sig.bar ~0.20)
  * always-on JM (alpha=0.10)  -> simulation_r3_JM0.10/.../reward_pilot (sig.bar ~0.015)

Per simulation_r4_fallback/files/CHANGES.md, the two missing variants (conditional
JM 0.15 and 0.142) are statistically indistinguishable from the conditional case
(d.alpha <= 0.005, d.sig.bar <= 0.010). The scientifically essential contrast the
paper makes -- conditional triggering keeps the posterior concentrated while
always-on application collapses it by >10x -- is fully captured by these two
panels, which are rebuilt here from the real sig1_downsampled trajectories
(100 runs x 200 downsampled points over T=2000).

Data source (bundled, self-contained):
    figures/scripts/data/sigma_conditional_rs_bib-bib.json
    figures/scripts/data/sigma_always_rs_bib-bib.json
(originals: simulation_r2 and simulation_r3_JM0.10 reward_pilot dirs)

Output: ../fig_smoothing.pdf (and .png preview)
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.constrained_layout.use"] = True

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figdata
FIG_DIR = str(figdata.FIG_DIR)
DATA = str(figdata.BUNDLED)          # figures/scripts/data (bundled inputs)

OUT_PDF = os.path.join(FIG_DIR, "fig_smoothing.pdf")
OUT_PNG = os.path.join(FIG_DIR, "fig_smoothing.png")

T_TOTAL = 2000   # full trajectory length before downsampling

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.linewidth": 0.8,
        "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def panel_label(ax, letter):
    ax.text(-0.17, 1.04, f"({letter.lower()})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

def load_traj(fname):
    runs = json.load(open(os.path.join(DATA, fname)))
    ds = np.array([r["sig1_downsampled"] for r in runs if "sig1_downsampled" in r])
    return ds  # shape (n_runs, n_down)

cond = load_traj("sigma_conditional_rs_bib-bib.json")
alw = load_traj("sigma_always_rs_bib-bib.json")
n_runs = cond.shape[0]
n_down = cond.shape[1]
t = np.linspace(0, T_TOTAL, n_down)

# alpha values for the argmax-persistence fits, reproduced from deposited code
# via regen_smoothing_table.py (seeds 0-99): conditional (1) Ibuka-exact and
# always-on (4). The always-on exponent is NOT steeper -- if anything slightly
# shallower -- the suppression is a cutoff/run-length collapse, not a re-slope.
ALPHA_COND = 1.358
ALPHA_ALW = 1.257

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":6,"axes.linewidth":0.8,"lines.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,"xtick.major.size":3,"ytick.major.size":3,"xtick.minor.size":1.8,"ytick.minor.size":1.8,"axes.spines.top":True,"axes.spines.right":True,"pdf.fonttype":42,"ps.fonttype":42})  # PNAS-unified
# Merged single panel: both rules overlaid on one axes, so the broad blue
# (conditional) band vs the tight red (always-on) band shows the ~8-fold
# sigma collapse directly.
fig, ax = plt.subplots(figsize=(3.4, 3.0))

def draw(ax, data, color, label):
    m = data.mean(axis=0)
    sd = data.std(axis=0)
    ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.20, lw=0)
    ax.plot(t, m, color=color, lw=1.5, label=label)
    return float(data.mean())

s_cond = draw(ax, cond, "#2c5f8a", r"conditional ($\bar\sigma\approx 0.13$)")
s_alw  = draw(ax, alw,  "#c0392b", r"always-on ($\bar\sigma\approx 0.015$)")
ax.set_xlabel("time step")
ax.set_ylabel(r"$\sigma(P(h))$")
ax.set_xlim(0, T_TOTAL)
ymax = max((cond.mean(0) + cond.std(0)).max(),
           (alw.mean(0) + alw.std(0)).max()) * 1.08
ax.set_ylim(0, ymax)
ax.legend(loc="upper right", fontsize=6.8, frameon=False)

fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=200)
print(f"conditional: n_runs={n_runs} sigma_bar={s_cond:.4f}")
print(f"always-on:   n_runs={n_runs} sigma_bar={s_alw:.4f}")
print(f"collapse factor sigma_cond/sigma_always = {s_cond/s_alw:.1f}x")
print("wrote:", OUT_PDF)
print("wrote:", OUT_PNG)
