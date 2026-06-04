#!/usr/bin/env python3
r"""Build the 4-panel main-text Fig. 5 (fig_scope.pdf): internal-state
criticality vs behavioural expression (A,B) + the Brockbank & Vul human-play
re-analysis (C,D), promoted from the SI into the main text.

Panels
------
(A) BIB-BIB at the uniform-Nash fixed point: the heavy tail is internal only
    (argmax-persistence TPL) while behavioural streaks are exponential.
(B) Against a fixed biased opponent the hand-run tail crosses over to a power
    law (behavioural Levy re-appears).
(C) Brockbank & Vul: human transition-run-length CCDF, heavier against fixed
    (exploitable) bots than adaptive bots.
(D) Mean transition-run length increases with exploitability (Spearman rho).

Self-contained: rebuilds from bundled data + the small behaviour cache, with
numpy / matplotlib / powerlaw / scipy only.
  scope (A,B): BIB_Levy_v2/latex/figures/scripts/data/data_ivb_{equil,biased}.npz
  behaviour (C,D): pnas_si/figures/figS_behaviour_cache.json
      (produced by pnas_si/figures/make_figS_behaviour.py from the public
       github.com/erik-brockbank/rps dataset; see fetch_bv_data.sh)

Usage:  python build_Fig5_scope_4panel.py
"""
import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import powerlaw

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # pnas/figures -> repo root
SCOPE_DATA = os.path.join(REPO, "BIB_Levy_v2", "latex", "figures", "scripts", "data")
BV_CACHE = os.path.join(REPO, "pnas_si", "figures", "figS_behaviour_cache.json")
OUT_PDF = os.path.join(HERE, "fig_scope.pdf")
OUT_PNG = os.path.join(HERE, "fig_scope.png")

# Okabe-Ito for the behaviour panels (match make_figS_behaviour.py)
C_FIX, C_ADA = "#D55E00", "#0072B2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def panel_label(ax, letter):
    ax.text(-0.17, 1.04, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


# ---------- helpers for the scope panels (inlined from build_Fig_scope) ----------
def ccdf(data, xmin=1):
    d = np.sort(np.asarray(data, float)); d = d[d >= xmin]; n = d.size
    x = np.unique(d)
    return x, 1.0 - np.searchsorted(d, x, side="left") / n


def plot_ccdf(ax, data, color, marker, label):
    x, cc = ccdf(data)
    ax.loglog(x, cc, marker=marker, ls="none", ms=3.2, mfc="none",
              mec=color, alpha=0.8, label=label)


def fit_best(data):
    data = np.asarray(data); data = data[data >= 1]
    if len(data) < 30 or len(np.unique(data)) < 5:
        return "err", float("nan")
    fit = powerlaw.Fit(data, discrete=True, verbose=False)
    return "tpl", float(fit.truncated_power_law.alpha)


def overlay_tpl_fit(ax, data, color):
    d = np.asarray(data, float); d = d[d >= 1]; n_total = d.size
    fit = powerlaw.Fit(d, discrete=True, verbose=False)
    tpl = fit.truncated_power_law
    alpha = float(tpl.alpha)
    lam = float(getattr(tpl, "Lambda", getattr(tpl, "parameter2", 0.0)))
    xmin = int(round(fit.xmin)); xmax = int(d.max())
    kk = np.arange(xmin, xmax + 1, dtype=float)
    w = kk ** (-alpha) * np.exp(-lam * kk); w /= w.sum()
    surv = np.cumsum(w[::-1])[::-1]
    cc_at_xmin = (d >= xmin).sum() / n_total
    ax.loglog(kk, surv * cc_at_xmin, "--", color=color, lw=1.5, alpha=0.95, zorder=5)
    return alpha


# ============================ figure ============================
equil = np.load(os.path.join(SCOPE_DATA, "data_ivb_equil.npz"))
biased = np.load(os.path.join(SCOPE_DATA, "data_ivb_biased.npz"))
with open(BV_CACHE) as fh:
    C = json.load(fh)
(cx2, cy2) = map(np.array, C["curveA"]["v2"])
(cx3, cy3) = map(np.array, C["curveA"]["v3"])
rows = C["rows"]
wr = np.array([r["wr"] for r in rows]); mt = np.array([r["mean_tran"] for r in rows])
rho_t, p_t = spearmanr(wr, mt)

fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
(axA, axB), (axC, axD) = axes
fig.subplots_adjust(wspace=0.28, hspace=0.34)

# ---- (A) equilibrium: internal heavy tail vs exponential behaviour ----
al_a = overlay_tpl_fit(axA, equil["argmax_persist"], "#b2182b")
plot_ccdf(axA, equil["argmax_persist"], "#b2182b", "o", "internal: argmax")
axA.plot([], [], "--", color="#b2182b", lw=1.4, label=rf"TPL fit ($\alpha$={al_a:.2f})")
plot_ccdf(axA, equil["hand_runs"], "#2166ac", "s", "hand-repeat (exp.)")
plot_ccdf(axA, equil["win_streaks"], "#4d9221", "^", "win streaks (exp.)")
axA.set_xlabel("run length"); axA.set_ylabel(r"CCDF  $P(X \geq x)$")
axA.set_ylim(1e-5, 1.5); axA.legend(loc="upper right", frameon=False)
panel_label(axA, "A")

# ---- (B) biased opponent: hand-run exponential -> power law ----
plot_ccdf(axB, equil["hand_runs"], "#2166ac", "s", r"BIB--BIB (pinned, exp.)")
al_b = overlay_tpl_fit(axB, biased["hand_runs"], "#ee7733")
plot_ccdf(axB, biased["hand_runs"], "#ee7733", "o", "vs fixed-biased opp.")
axB.plot([], [], "--", color="#ee7733", lw=1.4, label=rf"TPL fit ($\alpha$={al_b:.2f})")
hf = biased["hand_freq"]
axB.set_xlabel("hand-repeat run length"); axB.set_ylabel(r"CCDF  $P(X \geq x)$")
axB.set_ylim(1e-5, 1.5); axB.legend(loc="upper right", frameon=False)
axB.text(0.04, 0.04, f"hand bias r/p/s\n= {hf[0]:.2f}/{hf[1]:.2f}/{hf[2]:.2f}\n(uniform when pinned)",
         transform=axB.transAxes, ha="left", va="bottom", fontsize=6.5,
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6"))
panel_label(axB, "B")

# ---- (C) Brockbank & Vul: transition-run CCDF, fixed vs adaptive ----
for (x, y), lab, col, mk in [((cx2, cy2), "fixed bots (exploitable)", C_FIX, "o"),
                             ((cx3, cy3), "adaptive bots", C_ADA, "s")]:
    axC.loglog(x, y, marker=mk, ms=3.0, color=col, mfc=col, mec=col, lw=1.0, label=lab)
axC.set_xlabel(r"Human transition-run length  $\ell$")
axC.set_ylabel(r"CCDF   $P(L \geq \ell)$")
axC.set_ylim(2e-4, 1.3); axC.legend(loc="upper right", frameon=False, handlelength=1.4)
panel_label(axC, "C")

# ---- (D) exploitability vs behavioural persistence (gradient) ----
for r in rows:
    col, mk = (C_FIX, "o") if r["ver"] == "v2" else (C_ADA, "s")
    axD.scatter(r["wr"], r["mean_tran"], color=col, marker=mk, s=30,
                edgecolor="black", linewidth=0.4, zorder=3)
b, a0 = np.polyfit(wr, mt, 1)
xs = np.linspace(wr.min(), wr.max(), 50)
axD.plot(xs, a0 + b * xs, ls=(0, (5, 3)), color="0.45", lw=1.0, zorder=2)
axD.axvline(1 / 3, color="0.3", ls=":", lw=0.8, zorder=1)
axD.text(1 / 3 + 0.006, mt.min(), "Nash 1/3", fontsize=6.5, color="0.3", va="bottom", ha="left")
axD.set_xlabel("Exploitability  (human win rate)")
axD.set_ylabel("Mean transition-run length")
axD.text(0.04, 0.93, rf"Spearman $\rho = {rho_t:+.2f}$", transform=axD.transAxes, fontsize=7.5)
axD.legend(handles=[Line2D([], [], marker="o", ls="", mfc=C_FIX, mec="black", mew=0.4, ms=5, label="fixed (exploitable)"),
                    Line2D([], [], marker="s", ls="", mfc=C_ADA, mec="black", mew=0.4, ms=5, label="adaptive")],
           loc="lower right", frameon=False, handlelength=1.0)
panel_label(axD, "D")

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"A argmax TPL alpha={al_a:.2f} | B biased TPL alpha={al_b:.2f} "
      f"| C/D rows={len(rows)} Spearman rho={rho_t:+.3f} p={p_t:.1e}")
print("wrote", OUT_PDF)
