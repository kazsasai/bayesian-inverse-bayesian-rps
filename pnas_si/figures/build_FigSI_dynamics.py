#!/usr/bin/env python3
"""
build_FigSI_dynamics.py -- complementary BIB/BO dynamics demo for the SI.

Shows only the panels that main-text Fig. 1 does NOT carry (Fig. 1 keeps
the two posterior P(h) heatmaps): for BIB-BIB (top) and BO-BO (bottom),
the hand sequence and the top-three P(h) trajectories.  Representative run,
rs design, Nh=10, m=50, seed=42 (the same run as Fig. 1), so the two
figures are complementary with no duplicated panel.  Panel labels follow
the unified PNAS style (bold uppercase, upper-left).

Requires numpy, matplotlib, and the reward_huge_v2 simulator (powerlaw
must be importable; install it or stub it on PYTHONPATH).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
FIGDIR = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "simulation").is_dir())
sys.path.insert(0, str(ROOT / "simulation" / "reward_huge_v2"))
from rpsgame_reward import AgentReward  # noqa: E402

T, SEED = 1500, 42
SYM2COL = {"r": "#e6c700", "p": "#1f77b4", "s": "#2ca02c"}
SYM_IDX = {"r": 0, "p": 1, "s": 2}
PANEL_KW = dict(fontsize=11, fontweight="bold", va="bottom", ha="left")


def panel_label(ax, letter, x=-0.17, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, **PANEL_KW)


def run(a1, a2, T=T, seed=SEED, h_length=50, init="random", pred="sample"):
    g1 = AgentReward(a1, h_length=h_length, init_mode=init, predict_mode=pred, seed=seed)
    g2 = AgentReward(a2, h_length=h_length, init_mode=init, predict_mode=pred, seed=seed + 100000)
    Nh = g1.bayes.h_num
    P = np.empty((T, Nh)); H = np.empty(T, dtype="<U1")
    for t in range(T):
        h1 = g1.choice(); h2 = g2.choice()
        P[t] = g1.bayes.h_prov.copy(); H[t] = h1
        g1.update_from_outcome(h1, h2); g2.update_from_outcome(h2, h1)
    return P, H, Nh


Pb, Hb, Nh = run("bib", "bib")
Po, Ho, _ = run("bo", "bo")

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "font.family": "sans-serif"})
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":6,"axes.linewidth":0.8,"lines.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","xtick.major.size":3,"ytick.major.size":3,"xtick.minor.size":1.8,"ytick.minor.size":1.8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})  # PNAS-unified
fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.1),
                         gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.26, "hspace": 0.36})


def panel_hands(ax, hands, label):
    ax.set_xlim(0, T); ax.set_ylim(-0.5, 2.5)
    hidx = np.array([SYM_IDX[h] for h in hands])
    for s, col in SYM2COL.items():
        idx = np.where(hidx == SYM_IDX[s])[0]
        ax.vlines(idx, SYM_IDX[s] - 0.4, SYM_IDX[s] + 0.4, color=col, lw=0.7)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["R", "P", "S"]); ax.set_xlabel("step $t$")
    panel_label(ax, label)


def panel_top3(ax, P, label):
    order = np.argsort(-P.mean(axis=0))[:3]
    for h in order:
        ax.plot(np.arange(T), P[:, h], lw=1.5, label=f"h={h}")
    ax.axhline(1.0 / Nh, ls=":", color="grey", label=f"$1/N_h={1/Nh:.2f}$")
    ax.set_ylim(0, 1); ax.set_xlim(0, T); ax.set_xlabel("step $t$"); ax.set_ylabel(r"$P(h)$")
    ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=2)
    panel_label(ax, label)


# Row 0: BIB-BIB; Row 1: BO-BO. Left: hand sequence; right: top-three P(h).
panel_hands(axes[0, 0], Hb, "A"); panel_top3(axes[0, 1], Pb, "B")
panel_hands(axes[1, 0], Ho, "C"); panel_top3(axes[1, 1], Po, "D")

fig.tight_layout()
fig.savefig(FIGDIR / "fig_dynamics_demo_si.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "fig_dynamics_demo_si.png", dpi=130, bbox_inches="tight")
print("wrote", FIGDIR / "fig_dynamics_demo_si.pdf",
      "| BIB <maxP>=%.3f BO <maxP>=%.3f" % (float(Pb.max(1).mean()), float(Po.max(1).mean())))
