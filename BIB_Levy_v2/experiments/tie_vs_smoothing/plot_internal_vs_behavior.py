#!/usr/bin/env python3
"""
plot_internal_vs_behavior.py
Two-panel CCDF figure (paper format, bold lower-case (a)/(b) panel labels):

(a) bib-bib at the uniform-Nash fixed point: the heavy tail lives ONLY in the
    internal argmax-persistence (power law); behavioural streaks (hand-repeat,
    win) are exponential. Internal-state criticality is masked behaviourally
    by the adversarial pinning of the action marginal to uniform Nash.

(b) The same behavioural observable (hand-repeat runs) when the adversarial
    pinning is removed: against a FIXED biased opponent the bib agent locks
    onto an exploiting hypothesis, P(d|h*) concentrates, and the hand-run tail
    turns from exponential into a power law -- behavioural Levy re-appears.

Reads data_ivb_equil.npz and data_ivb_biased.npz (from gen_internal_vs_behavior.py).
"""
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import powerlaw
import engine_tie_smoothing as sim

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size": 11, "font.family": "sans-serif", "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def panel_label(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="right")


def ccdf(data, xmin=1):
    d = np.sort(np.asarray(data, float))
    d = d[d >= xmin]
    n = d.size
    x = np.unique(d)
    cc = 1.0 - (np.searchsorted(d, x, side="left")) / n
    return x, cc


def best_alpha(data):
    f = sim.fit_distributions(np.asarray(data, int))
    return f.get("best", "err"), f.get("alpha_best", np.nan)


def plot_ccdf(ax, data, color, marker, label):
    x, cc = ccdf(data)
    ax.loglog(x, cc, marker=marker, ls="none", ms=3.2, mfc="none",
              mec=color, alpha=0.8, label=label)
    return x, cc


def overlay_tpl_fit(ax, data, color):
    """Overlay the fitted truncated-power-law CCDF as a curve that follows the
    cutoff. Uses the SAME fit as the engine (powerlaw auto-selected xmin), and
    anchors the curve to the empirical CCDF at xmin. Returns (alpha, xmin)."""
    d = np.asarray(data, float)
    d = d[d >= 1]
    n_total = d.size
    fit = powerlaw.Fit(d, discrete=True, verbose=False)   # auto xmin (== engine)
    tpl = fit.truncated_power_law
    alpha = float(tpl.alpha)
    lam = float(getattr(tpl, "Lambda", getattr(tpl, "parameter2", 0.0)))
    xmin = int(round(fit.xmin))
    xmax = int(d.max())
    kk = np.arange(xmin, xmax + 1, dtype=float)
    w = kk ** (-alpha) * np.exp(-lam * kk)
    w /= w.sum()
    surv = np.cumsum(w[::-1])[::-1]                 # P(X>=k | X>=xmin), =1 at xmin
    cc_at_xmin = (d >= xmin).sum() / n_total        # empirical CCDF at xmin
    ax.loglog(kk, surv * cc_at_xmin, "--", color=color, lw=1.5,
              alpha=0.95, zorder=5)
    return alpha, xmin


equil = np.load(os.path.join(HERE, "data_ivb_equil.npz"))
biased = np.load(os.path.join(HERE, "data_ivb_biased.npz"))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.0))

# ---- (a) equilibrium: internal vs behavioural ----
b_am, a_am = best_alpha(equil["argmax_persist"])
b_hr, a_hr = best_alpha(equil["hand_runs"])
b_ws, a_ws = best_alpha(equil["win_streaks"])
plot_ccdf(axA, equil["argmax_persist"], "#b2182b", "o",
          r"internal: argmax persistence")
al_a, _ = overlay_tpl_fit(axA, equil["argmax_persist"], "#b2182b")
axA.plot([], [], "--", color="#b2182b", lw=1.4,
         label=rf"truncated power-law fit ($\alpha$={al_a:.2f})")
plot_ccdf(axA, equil["hand_runs"], "#2166ac", "s",
          r"behaviour: hand-repeat runs (exp.)")
plot_ccdf(axA, equil["win_streaks"], "#4d9221", "^",
          r"behaviour: win streaks (exp.)")
axA.set_xlabel("run length")
axA.set_ylabel(r"CCDF  $P(X \geq x)$")
axA.legend(fontsize=8.2, loc="lower left", framealpha=0.95)
axA.set_ylim(1e-5, 1.5)
panel_label(axA, "a")

# ---- (b) loosening: hand-runs exponential -> power law vs fixed biased ----
plot_ccdf(axB, equil["hand_runs"], "#2166ac", "s",
          r"hand runs: bib--bib (pinned, exp.)")
b_hrb, a_hrb = best_alpha(biased["hand_runs"])
plot_ccdf(axB, biased["hand_runs"], "#ee7733", "o",
          r"hand runs: vs fixed-biased opponent")
al_b, _ = overlay_tpl_fit(axB, biased["hand_runs"], "#ee7733")
axB.plot([], [], "--", color="#ee7733", lw=1.4,
         label=rf"truncated power-law fit ($\alpha$={al_b:.2f})")
hf = biased["hand_freq"]
axB.set_xlabel("hand-repeat run length")
axB.set_ylabel(r"CCDF  $P(X \geq x)$")
axB.legend(fontsize=8.2, loc="lower left", framealpha=0.95)
axB.set_ylim(1e-5, 1.5)
axB.text(0.97, 0.95,
         f"hand bias r/p/s\n= {hf[0]:.2f}/{hf[1]:.2f}/{hf[2]:.2f}\n(uniform when pinned)",
         transform=axB.transAxes, ha="right", va="top", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6"))
panel_label(axB, "b")

fig.subplots_adjust(wspace=0.22)
out = os.path.join(HERE, "fig_internal_vs_behavior.png")
fig.savefig(out); fig.savefig(out.replace(".png", ".pdf"))
print("panel (a) equil:  argmax", b_am, round(a_am, 2),
      "| hand_runs", b_hr, "| win", b_ws)
print("panel (b) biased: hand_runs", b_hrb, round(a_hrb, 2),
      "| hand bias", np.round(hf, 3))
print("wrote:", out)
