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
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def panel_label(ax, letter):
    ax.text(-0.17, 1.04, letter.upper(), transform=ax.transAxes,
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

# alpha values for the argmax-persistence fits (from CHANGES.md verification)
ALPHA_COND = 1.341
ALPHA_ALW = 1.508

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":6,"axes.linewidth":0.8,"lines.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","xtick.major.size":3,"ytick.major.size":3,"xtick.minor.size":1.8,"ytick.minor.size":1.8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})  # PNAS-unified
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.05), sharex=True)

def draw(ax, data, color, title, alpha_val, letter):
    m = data.mean(axis=0)
    sd = data.std(axis=0)
    ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.22, lw=0)
    ax.plot(t, m, color=color, lw=1.4, label=f"mean over {n_runs} runs")
    sbar = float(data.mean())
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$\sigma(P(h))$")
    ax.set_xlim(0, T_TOTAL)
    ax.text(0.97, 0.95,
            rf"$\bar\sigma = {sbar:.3f}$" + "\n" + rf"$\alpha \approx {alpha_val:.3f}$",
            transform=ax.transAxes, va="top", ha="right", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6"))
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    panel_label(ax, letter)
    return sbar

s_cond = draw(axA, cond, "#2c5f8a",
              "Conditional smoothing (triggered)", ALPHA_COND, "a")
s_alw = draw(axB, alw, "#c0392b",
             "Always-on JM smoothing", ALPHA_ALW, "b")

# shared y so the >10x collapse is visually unambiguous
ymax = max((cond.mean(0) + cond.std(0)).max(),
           (alw.mean(0) + alw.std(0)).max()) * 1.08
for ax in (axA, axB):
    ax.set_ylim(0, ymax)

fig.subplots_adjust(wspace=0.18)
fig.tight_layout()
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=200)
print(f"conditional: n_runs={n_runs} sigma_bar={s_cond:.4f}")
print(f"always-on:   n_runs={n_runs} sigma_bar={s_alw:.4f}")
print(f"collapse factor sigma_cond/sigma_always = {s_cond/s_alw:.1f}x")
print("wrote:", OUT_PDF)
print("wrote:", OUT_PNG)
