#!/usr/bin/env python3
r"""Build figS_baseline_control.pdf -- SI control: standard RL baselines vs the
BIB internal-state criticality (handoff v2).

Two panels, internal argmax-persistence CCDFs:
  (A) vs a uniform-random opponent -- BIB and regret matching are both 3/2-class;
      Q-learning and WSLS are non-critical.
  (B) self-play -- only BIB stays critical; regret matching collapses to
      exponential, Q-learning non-critical, WSLS degenerate.

Style matches the paper's figures (build_Fig3_universality / build_Fig5_scope):
Helvetica, despined log-log CCDFs, FILLED markers with black edge, dashed
TPL-fit overlay, on-off 3/2 reference, bold uppercase corner panel labels.

DATA
  BIB (validated local data; T_argmax1-only guard for the non-learning opponent):
    PAPERA_DATA/simulation/reward_huge/data/reward_huge_v2_{rs,ra,ss,sa}/
      durations_bib-random_m50_huge.json  (panel A: T_argmax1 ONLY)
      durations_bib-bib_m50_huge.json      (panel B: pool T_argmax1+T_argmax2)
  Baselines: ../../pnas_rl_comparison/data/baseline_dwells.npz
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.constrained_layout.use"] = True
import warnings
warnings.filterwarnings("ignore")
import powerlaw

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BIB_DIR = os.path.join(os.environ.get("PAPERA_DATA",
                       os.path.expanduser("~/paperA_data_full")),
                       "simulation", "reward_huge", "data")
BASE_NPZ = os.path.join(REPO, "pnas_rl_comparison", "data",
                        "baseline_dwells.npz")
OUT_PDF = os.path.join(HERE, "fig_control_ab.pdf")
OUT_PNG = os.path.join(HERE, "fig_control_ab.png")
DESIGNS = ("rs", "ra", "ss", "sa")

# ---- rcParams: paper-unified (build_Fig5_scope_4panel.py) ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 6.3, "axes.linewidth": 0.8,
    "lines.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8, "axes.spines.top": True, "axes.spines.right": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# method -> (color, marker).  BIB/RM colors match build_Fig5_scope; distinct
# shapes carry the distinction (legible in grayscale).
STYLE = {
    "bib":  ("#b2182b", "o"),
    "rm":   ("#ee7733", "s"),
    "q":    ("#2166ac", "^"),
    "wsls": ("#4d9221", "D"),
}
REF = "0.45"


def panel_label(ax, letter):
    ax.text(-0.17, 1.04, f"({letter.lower()})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


def corner_tag(ax, text):
    ax.text(0.97, 0.96, text, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="right", color="0.2")


def ccdf_ds(arr, npts=140):
    """Downsampled CCDF (log-spaced unique x), paper convention."""
    a = np.sort(np.asarray(arr, float)); a = a[a >= 1]; n = a.size
    if n == 0:
        return np.array([]), np.array([])
    x = np.unique(a)
    cc = 1.0 - np.searchsorted(a, x, side="left") / n
    if x.size > npts:
        sel = np.unique(np.round(np.logspace(0, np.log10(x.size - 1), npts)).astype(int))
        x, cc = x[sel], cc[sel]
    return x, cc


def markers_ccdf(ax, data, key, label, markevery=1):
    color, marker = STYLE[key]
    x, cc = ccdf_ds(data)
    m = cc > 0
    ax.loglog(x[m], cc[m], color=color, lw=0.0, marker=marker, ms=3.4,
              mfc=color, mec="black", mew=0.4, alpha=0.95, markevery=markevery,
              label=label, zorder=4)


def overlay_tpl(ax, data, key):
    color = STYLE[key][0]
    d = np.asarray(data, float); d = d[d >= 1]; n = d.size
    fit = powerlaw.Fit(d, discrete=True, verbose=False)
    tpl = fit.truncated_power_law
    alpha = float(tpl.alpha)
    lam = float(getattr(tpl, "Lambda", getattr(tpl, "parameter2", 0.0)))
    xmin = int(round(fit.xmin)); xmax = int(d.max())
    kk = np.arange(xmin, xmax + 1, dtype=float)
    w = kk ** (-alpha) * np.exp(-lam * kk); w /= w.sum()
    surv = np.cumsum(w[::-1])[::-1]
    cc_at = (d >= xmin).sum() / n
    ax.loglog(kk, surv * cc_at, "--", color=color, lw=1.3, alpha=0.95, zorder=5)
    return alpha


def _bib_path(d, pair):
    """Paper's RHF path reward_huge_{d}; v2_ duplicate is incomplete (fallback)."""
    for sub in (f"reward_huge_{d}", f"reward_huge_v2_{d}"):
        p = os.path.join(BIB_DIR, sub, f"durations_{pair}_m50_huge.json")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"durations_{pair}_m50_huge.json (design {d})")


def bib_pool(pair, agent1_only):
    """per-design list + pooled array (T_argmax1 only for non-learning opp)."""
    per = []
    for d in DESIGNS:
        j = json.load(open(_bib_path(d, pair)))
        v = [t for t in j["T_argmax1"] if t >= 1]
        if not agent1_only:
            v += [t for t in j["T_argmax2"] if t >= 1]
        per.append(np.asarray(v, float))
    return per, np.concatenate(per)


def bib_band(ax, per):
    mx = int(max(p.max() for p in per))
    grid = np.unique(np.round(np.logspace(0, np.log10(mx), 200)).astype(int))
    grid = grid[grid >= 1].astype(float)

    def surv(p):
        p = np.sort(p); return 1.0 - np.searchsorted(p, grid, side="left") / p.size
    cur = np.array([surv(p) for p in per])
    ax.fill_between(grid, cur.min(0), cur.max(0), color=STYLE["bib"][0],
                    alpha=0.13, lw=0, zorder=2)


def ref_32(ax, x0=5.0, x1=4000.0, y0=0.16):
    xs = np.array([x0, x1])
    ax.loglog(xs, y0 * (xs / x0) ** -0.5, ls=":", color=REF, lw=1.5, zorder=1,
              label=r"on-off ref $\alpha=3/2$")


def main():
    z = np.load(BASE_NPZ)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.05),
                                   )

    # ===== (A) vs uniform-random =====
    perR, poolR = bib_pool("bib-random", agent1_only=True)
    bib_band(axA, perR)
    markers_ccdf(axA, poolR, "bib", "BIB (argmax hyp.)")
    overlay_tpl(axA, poolR, "bib")
    markers_ccdf(axA, z["RegretMatching__random__argmax"], "rm", r"regret match. (argmax $R^+$)")
    overlay_tpl(axA, z["RegretMatching__random__argmax"], "rm")
    markers_ccdf(axA, z["Qlearning__random__argmax"], "q", "Q-learning (action)")
    markers_ccdf(axA, z["WSLS__random__argmax"], "wsls", "WSLS (action)")
    ref_32(axA)
    axA.set_xlabel("argmax-persistence length (steps)")
    axA.set_ylabel(r"CCDF  $P(X \geq x)$")
    axA.set_xlim(1, 3e5); axA.set_ylim(1e-5, 1.5)
    corner_tag(axA, "vs uniform-random")
    panel_label(axA, "A")

    # ===== (B) self-play =====
    perS, poolS = bib_pool("bib-bib", agent1_only=False)
    bib_band(axB, perS)
    markers_ccdf(axB, poolS, "bib", "BIB (argmax hyp.)")
    overlay_tpl(axB, poolS, "bib")
    markers_ccdf(axB, z["RegretMatching__self_play__argmax"], "rm", r"regret match. (argmax $R^+$)")
    markers_ccdf(axB, z["Qlearning__self_play__argmax"], "q", "Q-learning (action)")
    markers_ccdf(axB, z["WSLS__self_play__argmax"], "wsls", "WSLS (action)")
    ref_32(axB)
    axB.set_xlabel("argmax-persistence length (steps)")
    axB.set_xlim(1, 3e5); axB.set_ylim(1e-5, 1.5)
    corner_tag(axB, "self-play")
    panel_label(axB, "B")

    # shared horizontal legend below both panels (out of the data area), with a
    # clear gap below the x-axis labels (more-negative y = lower / bigger gap).
    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.13), handlelength=1.4,
               columnspacing=1.4)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
