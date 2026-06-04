#!/usr/bin/env python3
"""
plot_timeseries.py
Single-run time-series diagnostics comparing JM smoothing ON vs OFF.

For a chosen (design, tie_mode), runs one trajectory with jm_smoothing=True and
one with jm_smoothing=False (same seed) and draws, for agent 1:
  row 1  argmax hypothesis index over time, with laminar phases (consecutive-
         argmax runs = T_argmax persistence) shaded as alternating bands.
  row 2  hand played (r/p/s) as a colored event strip.
  row 3  sigma(P(h)) posterior spread over time.

Laminar = a maximal interval on which the argmax index is constant (the system
"trusts" one hypothesis); its length is the argmax-persistence time analysed in
the paper. JM-on -> many laminar phases of heavy-tailed length (on-off
intermittency); JM-off -> the argmax freezes into one giant laminar phase.

Usage: python3 plot_timeseries.py --design rs --tie uniform --T 3000 --seed 1
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import engine_tie_smoothing as sim

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN_TAG = {"rs": ("random", "sample"), "ra": ("random", "argmax"),
              "ss": ("structured", "sample"), "sa": ("structured", "argmax")}
HAND_COLORS = ListedColormap(["#d1495b", "#edae49", "#3a6ea5"])  # r, p, s
HAND_NAMES = ["r", "p", "s"]

plt.rcParams.update({
    "font.size": 10, "font.family": "sans-serif", "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "savefig.dpi": 200,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def run(design, tie, jm, T, seed):
    init_mode, predict_mode = DESIGN_TAG[design]
    return sim.run_pair("bib", "bib", T, h_length=50, h_num=10,
                        init_mode=init_mode, predict_mode=predict_mode,
                        tie_mode=tie, defeat_mode="random_other",
                        seed=seed, jm_smoothing=jm)


def shade_laminar(ax, argmax, T):
    """Shade alternating laminar (constant-argmax) runs."""
    runs = sim.consecutive_runs(argmax)
    edges = np.concatenate([[0], np.cumsum(runs)])
    for i in range(len(runs)):
        if i % 2 == 0:
            ax.axvspan(edges[i], edges[i + 1], color="0.90", lw=0, zorder=0)
    return runs


def draw_column(axes, df, title, h_num):
    t = df["t"].values
    am = df["argmax1"].values
    T = len(t)

    # row 1: argmax with laminar shading
    ax0 = axes[0]
    runs = shade_laminar(ax0, am, T)
    ax0.step(t, am, where="post", color="#222", lw=0.8)
    ax0.set_ylim(-0.5, h_num - 0.5)
    ax0.set_ylabel("argmax\nhypothesis")
    ax0.set_title(title, fontsize=10.5)
    ax0.text(0.99, 0.94,
             f"laminar phases: {len(runs)}\nlongest: {int(runs.max())} steps",
             transform=ax0.transAxes, ha="right", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6"))

    # row 2: hand strip
    ax1 = axes[1]
    hand_idx = np.array([HAND_NAMES.index(h) for h in df["h1"].values])
    ax1.imshow(hand_idx[None, :], aspect="auto", cmap=HAND_COLORS,
               norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3),
               extent=[0, T, 0, 1], interpolation="nearest")
    ax1.set_yticks([]); ax1.set_ylabel("hand\n(r/p/s)")

    # row 3: sigma
    ax2 = axes[2]
    ax2.plot(t, df["sig1"].values, color="#6e8b3d", lw=0.7)
    ax2.axhspan(0.12, 0.16, color="#9fc5e8", alpha=0.4, zorder=0)
    ax2.set_ylim(0, 0.34)
    ax2.set_ylabel(r"$\sigma(P(h))$")
    ax2.set_xlabel("time step")
    for ax in axes:
        ax.set_xlim(0, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="rs", choices=list(DESIGN_TAG))
    ap.add_argument("--tie", default="uniform", choices=["skip", "uniform"])
    ap.add_argument("--T", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    h_num = 10
    scheme = "eq3" if args.tie == "skip" else "caseA"

    df_on = run(args.design, args.tie, True, args.T, args.seed)
    df_off = run(args.design, args.tie, False, args.T, args.seed)

    fig, axes = plt.subplots(3, 2, figsize=(14, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 0.6, 1.4],
                                          "hspace": 0.18, "wspace": 0.16})
    draw_column(axes[:, 0], df_on, f"JM smoothing ON  ({scheme}, {args.design})", h_num)
    draw_column(axes[:, 1], df_off, f"JM smoothing OFF  ({scheme}, {args.design})", h_num)

    fig.suptitle(
        f"Hands / argmax / laminar structure under JM-on vs JM-off "
        f"(bib--bib, {args.design}, tie={args.tie}, $N_h$={h_num}, $T$={args.T}, seed={args.seed})",
        fontsize=11, y=0.975)
    out = os.path.join(HERE, f"fig_timeseries_{args.design}_{args.tie}_seed{args.seed}.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    # quick stats
    for tag, df in [("ON", df_on), ("OFF", df_off)]:
        runs = sim.consecutive_runs(df["argmax1"].values)
        print(f"{tag}: laminar_phases={len(runs)} longest={int(runs.max())} "
              f"sigma_bar={df['sig1'].mean():.3f}")
    print("wrote:", out)


if __name__ == "__main__":
    main()
