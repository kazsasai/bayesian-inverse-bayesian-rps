"""Rebuild Fig 3 (file fig_universality.pdf): the BIB universality
contrast.  (a) argmax-persistence CCDFs and (b) laminar-phase-length CCDFs, each
overlaying the four designs (rs/ra/ss/sa) for BIB-BIB (solid) and BO-BO (dashed).
BIB collapses onto a single heavy tail; BO fans out / decays faster.

Data via figdata:
  (a) argmax: simulation/reward_huge/data/reward_huge_<design>/durations_<pair>_m50_huge.json
  (b) laminar (max_h P(h) > theta=0.4 run lengths), from per-design pmax npz:
        BIB : .../huge_laminar/pmax_<design>_eq3.npz   (published rule)
        BO  : .../huge_laminar/pmax_<design>_bo.npz
House style: bold lower-case (a)/(b) panel labels, no titles.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.constrained_layout.use"] = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata

OUT_PDF = figdata.FIG_DIR / "fig_universality.pdf"
OUT_PNG = figdata.FIG_DIR / "fig_universality.png"

DES = ["rs", "ra", "ss", "sa"]
DES_NAME = {"rs": "rs (rand+sample)", "ra": "ra (rand+argmax)",
            "ss": "ss (struct+sample)", "sa": "sa (struct+argmax)"}
COL = {"rs": "#1f77b4", "ra": "#ff7f0e", "ss": "#2ca02c", "sa": "#d62728"}
RHF = {"rs": "reward_huge_rs", "ra": "reward_huge_ra",
       "ss": "reward_huge_ss", "sa": "reward_huge_sa"}
LAM_DIR = "simulation_tie_mode_ablation/data/scheme_ablation/huge_laminar"
THETA = 0.4


def ccdf_ds(arr, npts=350):
    a = np.sort(np.asarray(arr, float))
    n = len(a)
    if n == 0:
        return np.array([]), np.array([])
    ux, idx = np.unique(a, return_index=True)
    p = (n - idx) / n
    if len(ux) > npts:
        sel = np.unique(np.round(np.logspace(0, np.log10(len(ux) - 1), npts)).astype(int))
        ux, p = ux[sel], p[sel]
    return ux, p


def argmax_durations(design, pair):
    j = json.load(open(figdata.find(
        f"simulation/reward_huge/data/{RHF[design]}/durations_{pair}_m50_huge.json")))
    out = []
    for k in ("T_argmax1", "T_argmax2"):
        out += [v for v in j.get(k, []) if v >= 1]
    return np.asarray(out, float)


def laminar_lengths(design, pair, theta=THETA):
    tag = "eq3" if pair == "bib-bib" else "bo"
    z = np.load(figdata.find(f"{LAM_DIR}/pmax_{design}_{tag}.npz"))
    out = []
    for key in ("pmax1", "pmax2"):
        for run in z[key]:
            il = run > theta
            d = np.diff(il.astype(np.int8))
            b = np.concatenate([[0], np.flatnonzero(d) + 1, [il.size]])
            rl = np.diff(b); st = il[b[:-1]]
            out += rl[st].tolist()
    return np.asarray(out, float)


MK = {"rs": "o", "ra": "s", "ss": "^", "sa": "D"}


def _panel(ax, getter, pair, ylim):
    """Plot the four design CCDFs (one scheme) in a single panel."""
    for d in DES:
        x, p = ccdf_ds(getter(d, pair))
        ax.loglog(x, p, color=COL[d], lw=1.2, marker=MK[d], markersize=3.2,
                  markevery=0.12, markeredgecolor="black", markeredgewidth=0.4,
                  alpha=0.95, label=DES_NAME[d])
    ax.set_ylim(*ylim)
    ax.grid(False)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "axes.spines.top": True, "axes.spines.right": True,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    # rows = observable (argmax / laminar); columns = scheme (BIB / BO)
    fig, ((axAB, axAO), (axLB, axLO)) = plt.subplots(
        2, 2, figsize=(7.0, 6.1),
        )

    # --- top row: argmax persistence ---
    _panel(axAB, argmax_durations, "bib-bib", (1e-6, 1.5))
    _panel(axAO, argmax_durations, "bo-bo", (1e-6, 1.5))
    for ax in (axAB, axAO):
        ax.set_xlabel(r"$T_{\mathrm{argmax}}$ (steps)")
    axAB.set_ylabel(r"CCDF $P(T \geq t)$")

    # --- bottom row: laminar-phase length, with on-off alpha=3/2 reference ---
    _panel(axLB, laminar_lengths, "bib-bib", (1e-5, 1.5))
    _panel(axLO, laminar_lengths, "bo-bo", (1e-5, 1.5))
    xr = np.logspace(np.log10(3), np.log10(1.2e3), 50)
    yr = 0.45 * (xr / xr[0]) ** (-0.5)
    for ax in (axLB, axLO):
        ax.loglog(xr, yr, ":", color="0.4", lw=1.5,
                  label=r"on-off ref $\alpha=3/2$")
        ax.set_xlabel(r"$L_{\mathrm{laminar}}$ (steps, $\max_h P(h)>%.1f$)" % THETA)
    axLB.set_ylabel(r"CCDF $P(L \geq t)$")

    # design legend once (top-left); on-off ref legend on a laminar panel
    axAB.legend(loc="lower left", fontsize=6.8, frameon=False)
    axLB.legend(loc="lower left", fontsize=6.8, frameon=False)

    # panel labels (house style) + a compact scheme tag per panel
    tags = {axAB: ("A", "BIB-BIB"), axAO: ("B", "BO-BO"),
            axLB: ("C", "BIB-BIB"), axLO: ("D", "BO-BO")}
    for ax, (t, scheme) in tags.items():
        ax.text(-0.17, 1.04, f"({t.lower()})", transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="left")
        ax.text(0.97, 0.95, scheme, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top", ha="right", color="0.2")

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=160)
    print("Saved:", OUT_PDF)


if __name__ == "__main__":
    main()
