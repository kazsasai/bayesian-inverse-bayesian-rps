#!/usr/bin/env python3
"""
build_Fig1_bc.py -- PNAS Fig 1 as a single matplotlib composite.

Panel A: the BIB cycle schematic (fig_soc_mechanism.pdf), rasterised at
high DPI and imported, so its panel label is drawn by matplotlib in the
SAME font/size/position as B and C (fixes the mixed-font label issue).
Panels B, C: BIB and BO posterior P(h) heatmaps with the argmax trajectory,
from one representative run (rs design, Nh=10, m=50, seed=42), faithful to
panels (b)/(e) of the PRE fig_dynamics_demo.

Panel labels follow the PNAS convention: bold uppercase letters at the
upper-left of each panel (shared helper `panel_label`, also used by the
other figure scripts so labelling is uniform across figures).

Requires numpy, matplotlib, poppler (pdftoppm), and the reward_huge_v2
simulator. rpsgame_reward.py imports `powerlaw` at load (unused here);
install it or put an empty powerlaw.py on PYTHONPATH if missing.
"""
import sys
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = Path(__file__).resolve()
FIGDIR = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "simulation").is_dir())
sys.path.insert(0, str(ROOT / "simulation" / "reward_huge_v2"))
from rpsgame_reward import AgentReward  # noqa: E402

# ---- shared PNAS panel-label style: bold uppercase, sans, upper-left ----
PANEL_LABEL_KW = dict(fontsize=16, fontweight="bold", va="bottom", ha="left")


def panel_label(ax, letter, x=-0.04, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, **PANEL_LABEL_KW)


# ---- rasterise the concept schematic PDF and import it ----
tmp = Path(tempfile.mkdtemp())
subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                str(FIGDIR / "fig_soc_mechanism.pdf"), str(tmp / "concept")],
               check=True)
concept = mpimg.imread(str(tmp / "concept.png"))
concept_aspect = concept.shape[1] / concept.shape[0]  # width / height

# ---- representative run for B, C ----
T, SEED = 1500, 42


def run(a1, a2, T=T, seed=SEED, h_length=50, init="random", pred="sample"):
    g1 = AgentReward(a1, h_length=h_length, init_mode=init, predict_mode=pred, seed=seed)
    g2 = AgentReward(a2, h_length=h_length, init_mode=init, predict_mode=pred, seed=seed + 100000)
    Nh = g1.bayes.h_num
    P = np.empty((T, Nh)); am = np.empty(T, int)
    for t in range(T):
        h1 = g1.choice(); h2 = g2.choice()
        am[t] = g1.argmax_h(); P[t] = g1.bayes.h_prov.copy()
        g1.update_from_outcome(h1, h2); g2.update_from_outcome(h2, h1)
    return P, am, Nh


Pb, amb, Nh = run("bib", "bib")
Po, amo, _ = run("bo", "bo")

# ---- compose: A on top, B | C below; A aligned to the B+C heatmap span ----
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig = plt.figure(figsize=(10, 10 / concept_aspect + 3.4))
gs = fig.add_gridspec(2, 2, height_ratios=[10 / concept_aspect, 3.4],
                      left=0.075, right=0.965, top=0.975, bottom=0.11,
                      hspace=0.45, wspace=0.30)


def heat(ax, P, am, letter, title):
    im = ax.imshow(P.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, T, -0.5, Nh - 0.5],
                   vmin=0, vmax=min(1.0, float(P.max()) * 1.05))
    ax.plot(np.arange(T), am, color="white", lw=1.0)
    ax.set_yticks([0, 3, 6, 9])
    ax.set_ylabel("hypothesis $h$"); ax.set_xlabel("step $t$")
    ax.set_title(title)
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    panel_label(ax, letter)


axB = fig.add_subplot(gs[1, 0]); heat(axB, Pb, amb, "B", "BIB")
axC = fig.add_subplot(gs[1, 1]); heat(axC, Po, amo, "C", "BO")

axA = fig.add_subplot(gs[0, :])
axA.imshow(concept, aspect="auto")
axA.axis("off")
panel_label(axA, "A", x=0.0, y=1.0)

# Align A's box to span exactly the B and C heatmap boxes (equal width),
# preserving the schematic aspect ratio so it is not distorted.
fig.canvas.draw()
bx0 = axB.get_position().x0
cx1 = axC.get_position().x1
figW, figH = fig.get_size_inches()
w = cx1 - bx0
h = (w * figW / concept_aspect) / figH
pa = axA.get_position()
axA.set_position([bx0, pa.y1 - h, w, h])

fig.savefig(FIGDIR / "fig1_composite.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "fig1_composite.png", dpi=150, bbox_inches="tight")
print("wrote", FIGDIR / "fig1_composite.pdf",
      "| A box x=[%.3f..%.3f] | BIB <maxP>=%.3f BO <maxP>=%.3f"
      % (bx0, cx1, float(Pb.max(1).mean()), float(Po.max(1).mean())))
