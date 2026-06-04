#!/usr/bin/env python3
"""Build fig_bib_vs_bo.pdf: the combined BIB-vs-BO functional comparison
(merges the former reward figure and BO-tournament figure into one
four-panel, full-width figure).

Panels
------
(a) BIB-vs-BO net advantage per step (r_win - r_defeat) by design, with a
    one-sample z-score of the win rate against the Nash value 1/3 (n=20 runs).
(b) Cumulative reward of BIB vs BO over the normalised post-burn-in window
    (mean +/- SD across runs; final cumulative reward in the legend).
(c) 5x5 cross-design tournament: z-score of player-1's win rate against the
    chance value 1/3 for every pairing (BO designs + Nash random).
(d) BO-internal tournament ranking against the BO field (random excluded).

The former scatter panel (BO-internal strength vs BIB-vs-BO z) was dropped:
it was simply panel (a) plotted against panel (d), and that contrast is now
stated in the caption / body text.

Data (resolved via figdata):
  reward : simulation/reward_huge/data/reward_huge_v2_<design>/
           rewards_bib-bo_m50_huge.json
  tourn. : simulation/analyze_sharpness_plateau/data/bo_tournament/
           bo_tournament_results.json  (bundled fallback)

House style: bold lower-case (a)-(d) panel labels, no per-panel titles.
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

OUT_PDF = str(figdata.FIG_DIR / "fig_bib_vs_bo.pdf")
OUT_PNG = str(figdata.FIG_DIR / "fig_bib_vs_bo.png")

# tournament JSON (in-repo / Zenodo via figdata, bundled fallback)
_rel = ("simulation/analyze_sharpness_plateau/data/bo_tournament/"
        "bo_tournament_results.json")
JSON_PATH = str(figdata.find(_rel)) if figdata.exists(_rel) \
    else str(figdata.bundled("bo_tournament_results.json"))

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11, "font.family": "sans-serif", "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

DES = ["rs", "ra", "ss", "sa"]
DES_LONG = {"rs": "rs\n(rank 1)", "ra": "ra\n(rank 2)",
            "ss": "ss\n(rank 3)", "sa": "sa\n(rank 4)"}
COL = {"rs": "#1f77b4", "ra": "#ff7f0e", "ss": "#2ca02c", "sa": "#d62728"}
BLUE = "#3a6ea5"
STRATEGIES = ["rs", "ss", "ra", "sa", "random"]
BO = ["rs", "ra", "ss", "sa"]
ZCRIT = 1.96


def panel_label(ax, letter):
    ax.text(-0.17, 1.04, letter.upper(), transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def stars(z):
    a = abs(z)
    return "***" if a > 3.29 else "**" if a > 2.58 else "*" if a > 1.96 else "ns"


def reward_relpath(design):
    return (f"simulation/reward_huge/data/reward_huge_v2_{design}/"
            f"rewards_bib-bo_m50_huge.json")


def load_reward(design):
    return json.load(open(figdata.find(reward_relpath(design))))


def main():
    # ---- load tournament data ----
    with open(JSON_PATH) as f:
        agg = json.load(f)
    Z = np.full((len(STRATEGIES), len(STRATEGIES)), np.nan)
    for i, s1 in enumerate(STRATEGIES):
        for j, s2 in enumerate(STRATEGIES):
            Z[i, j] = agg[f"{s1}_vs_{s2}"]["z_score_1_vs_chance"]
    summary = []
    for s1 in BO:
        wrs = [agg[f"{s1}_vs_{s2}"]["win_rate_1"] for s2 in BO]
        zsum = [agg[f"{s1}_vs_{s2}"]["z_score_1_vs_chance"] for s2 in BO]
        summary.append((s1, float(np.mean(wrs)), float(sum(zsum))))
    summary.sort(key=lambda x: x[1], reverse=True)

    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":6,"axes.linewidth":0.8,"lines.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","xtick.major.size":3,"ytick.major.size":3,"xtick.minor.size":1.8,"ytick.minor.size":1.8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})  # PNAS-unified
    fig, ((axA, axB), (axC, axD)) = plt.subplots(
        2, 2, figsize=(7.0, 6.1),
        gridspec_kw={"wspace": 0.55, "hspace": 0.42})

    # ============================ (a) net advantage ============================
    means, sds, zs = {}, {}, {}
    for d in DES:
        r = load_reward(d)
        net = np.array([(x["win_count"] - x["defeat_count"]) / x["n_post"]
                        for x in r])
        wr = np.array([x["win_count"] / x["n_post"] for x in r])
        n = len(net)
        means[d] = net.mean()
        sds[d] = net.std(ddof=1)
        zs[d] = (wr.mean() - 1.0 / 3.0) / (wr.std(ddof=1) / np.sqrt(n))
    xpos = np.arange(len(DES))
    axA.bar(xpos, [means[d] for d in DES], yerr=[sds[d] for d in DES],
            color=[COL[d] for d in DES], edgecolor="black", lw=0.7,
            width=0.62, capsize=4, error_kw=dict(lw=1.0))
    axA.axhline(0, color="black", lw=0.8)
    tops = [means[d] + sds[d] for d in DES] + [0.0]
    bots = [means[d] - sds[d] for d in DES] + [0.0]
    span = max(tops) - min(bots)
    axA.set_ylim(min(bots) - 0.30 * span, max(tops) + 0.30 * span)
    for i, d in enumerate(DES):
        up = means[d] >= 0
        y = means[d] + (sds[d] if up else -sds[d])
        axA.annotate(f"$z={zs[d]:+.2f}$\n{stars(zs[d])}", (i, y),
                     textcoords="offset points", xytext=(0, 10) if up else (0, -10),
                     ha="center", va="bottom" if up else "top",
                     annotation_clip=False, fontsize=6.5,
                     fontweight="bold" if stars(zs[d]) != "ns" else "normal")
    axA.set_xticks(xpos)
    axA.set_xticklabels([DES_LONG[d] for d in DES], fontsize=7)
    axA.set_xlabel("Design ordered by Nash-distance hardness  "
                   "(rank 1 = near, 4 = far)", fontsize=6.5)
    axA.set_ylabel(r"Net advantage of BIB per step  $(r_{\rm win}-r_{\rm lose})$")
    panel_label(axA, "a")

    # ============================ (b) cumulative reward ============================
    for d in DES:
        r = load_reward(d)
        t = np.array(r[0]["sample_indices"], float) / r[0]["n_post"]
        C = np.array([x["cumR_downsampled"] for x in r], float)
        m, sd = C.mean(0), C.std(0, ddof=1)
        cf = np.array([x["cumR_final"] for x in r])
        axB.plot(t, m, color=COL[d], lw=1.8,
                 label=fr"{d}: cumR $={cf.mean():+.0f} \pm {cf.std(ddof=1):.0f}$")
        axB.fill_between(t, m - sd, m + sd, color=COL[d], alpha=0.15, lw=0)
    axB.axhline(0, color="black", lw=0.8)
    axB.set_xlabel(r"Normalised post-burn-in time  $(t/n_{\rm post})$")
    axB.set_ylabel(r"Cumulative reward of BIB (vs BO)")
    axB.legend(loc="upper left", fontsize=6.5, frameon=False)
    panel_label(axB, "b")

    # ============================ (c) tournament heat-map ============================
    vmax = np.nanmax(np.abs(Z))
    im = axC.imshow(Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axC.set_xticks(range(len(STRATEGIES)))
    axC.set_yticks(range(len(STRATEGIES)))
    axC.set_xticklabels(STRATEGIES)
    axC.set_yticklabels(STRATEGIES)
    axC.set_xlabel("opponent (player 2)")
    axC.set_ylabel("player 1")
    for i in range(len(STRATEGIES)):
        for j in range(len(STRATEGIES)):
            if i == j:
                axC.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True,
                                            color="0.85", hatch="///",
                                            ec="0.6", lw=0.5))
                continue
            z = Z[i, j]
            sig = abs(z) >= ZCRIT
            axC.text(j, i, f"{z:+.1f}", ha="center", va="center", fontsize=7,
                     fontweight=("bold" if sig else "normal"),
                     color=("white" if abs(z) > 0.6 * vmax else "black"))
    cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
    cb.set_label("z-score", fontsize=7)
    panel_label(axC, "c")

    # ============================ (d) BO-internal ranking ============================
    order = [d for d, _, _ in summary]
    wr_vals = [wr for _, wr, _ in summary]
    x = np.arange(len(order))
    axD.bar(x, wr_vals, color=BLUE, edgecolor="black", lw=0.7, width=0.6)
    axD.axhline(1 / 3, ls="--", color="0.5", lw=1.0)
    axD.text(len(order) - 0.55, 1 / 3, "1/3 (Nash)", va="bottom", ha="right",
             color="0.4", fontsize=6.5)
    for xi, (d, wr, sz) in enumerate(summary):
        axD.text(xi, wr + 0.0002, f"rank {xi+1}\nsum z\n{sz:+.1f}", ha="center",
                 va="bottom", fontsize=6.5)
    axD.set_xticks(x)
    axD.set_xticklabels(order)
    axD.set_ylabel("mean win rate vs other BO designs")
    axD.set_ylim(min(wr_vals) - 0.0015, max(wr_vals) + 0.0020)
    panel_label(axD, "d")

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print("reward z:", {d: round(zs[d], 2) for d in DES})
    print("BO-internal order:", order)
    print("wrote:", OUT_PDF)


if __name__ == "__main__":
    main()
