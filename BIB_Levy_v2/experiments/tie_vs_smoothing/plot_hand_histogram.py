#!/usr/bin/env python3
"""
plot_hand_histogram.py
Shows WHY the hand stream looks like noise even when the BIB agent's internal
state is frozen: the hand is an i.i.d. sample from the predictive distribution
P(d) = sum_h P(h) P(d|h). When the posterior is pinned to one hypothesis h*,
P(d) collapses to that hypothesis's likelihood vector P(d|h*) -- a fixed, non-
degenerate categorical over {r,p,s} -- so the actions stay stochastic.

For the BIB agent (agent 1) of a bib-bib game, per design x JM-smoothing, we
overlay three r/p/s distributions over the analysed (2nd-half) window:
  - empirical hand frequency           (what was actually played)
  - P(d|h*) of the dominant hypothesis (the "contents" of the frozen belief)
  - mean predictive mixture <P(d)>_t   (theoretical readout, = P(d|h*) when frozen)

Usage: python3 plot_hand_histogram.py --designs rs ss --tie uniform --T 3000 --seed 1
"""
import argparse
import os
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import engine_tie_smoothing as sim

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN_TAG = {"rs": ("random", "sample"), "ra": ("random", "argmax"),
              "ss": ("structured", "sample"), "sa": ("structured", "argmax")}
HANDS = ["r", "p", "s"]

plt.rcParams.update({
    "font.size": 10, "font.family": "sans-serif", "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "savefig.dpi": 200,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def run_instrumented(design, tie, jm, T, seed):
    """Run a bib-bib game; record agent-1 hands, argmax, and per-step
    predictive distribution P(d)=sum_h P(h)P(d|h)."""
    init_mode, predict_mode = DESIGN_TAG[design]
    ag1 = sim.AgentReward("bib", 50, 10, init_mode, predict_mode, tie_mode=tie,
                          defeat_mode="random_other", seed=seed, jm_smoothing=jm)
    ag2 = sim.AgentReward("bib", 50, 10, init_mode, predict_mode, tie_mode=tie,
                          defeat_mode="random_other", seed=seed + 100000, jm_smoothing=jm)
    a0 = T // 2
    hands, argm, pred = [], [], []
    for t in range(T):
        # predictive distribution of agent 1 BEFORE acting
        b = ag1.bayes
        pd = b.h_prov @ b.likelihood            # mixture over hypotheses
        h1 = ag1.choice(); h2 = ag2.choice()
        if t >= a0:
            hands.append(h1); argm.append(ag1.argmax_h()); pred.append(pd.copy())
        ag1.update_from_outcome(h1, h2); ag2.update_from_outcome(h2, h1)
    argm = np.array(argm)
    dom = Counter(argm).most_common(1)[0]
    Lstar = ag1.bayes.likelihood[dom[0]].copy()      # P(d|h*) of dominant hyp
    freq = Counter(hands); n = len(hands)
    emp = np.array([freq[h] / n for h in HANDS])
    pred_mean = np.mean(pred, axis=0)
    held_frac = dom[1] / n
    return dict(emp=emp, Lstar=Lstar, pred_mean=pred_mean,
                hstar=dom[0], held_frac=held_frac,
                sigma_bar=float(ag1.bayes.h_prov.std()))


def draw(ax, r, title):
    x = np.arange(3); w = 0.27
    ax.bar(x - w, r["emp"], w, color="#444", label="empirical hands")
    ax.bar(x,     r["Lstar"], w, color="#c0392b",
           label=r"$P(d|h^*)$ (frozen belief)")
    ax.bar(x + w, r["pred_mean"], w, color="#3a6ea5",
           label=r"$\langle P(d)\rangle_t$ (mixture)")
    ax.axhline(1/3, ls=":", color="0.6", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(HANDS)
    ax.set_ylim(0, 0.55); ax.set_ylabel("probability")
    ax.set_title(title, fontsize=10)
    ax.text(0.98, 0.96,
            f"dominant $h^*$={r['hstar']}\nheld {r['held_frac']*100:.0f}% of window",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="*", default=["rs", "ss"])
    ap.add_argument("--tie", default="uniform", choices=["skip", "uniform"])
    ap.add_argument("--T", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    scheme = "eq3" if args.tie == "skip" else "caseA"

    nd = len(args.designs)
    fig, axes = plt.subplots(nd, 2, figsize=(11, 3.4 * nd), squeeze=False)
    for i, design in enumerate(args.designs):
        for j, jm in enumerate([True, False]):
            r = run_instrumented(design, args.tie, jm, args.T, args.seed)
            tag = "JM ON" if jm else "JM OFF"
            draw(axes[i][j], r,
                 f"{design} ({scheme}) — {tag}  "
                 + (r"($\bar\sigma$=%.2f)" % r["sigma_bar"]))
            if i == 0 and j == 0:
                axes[i][j].legend(fontsize=8, loc="upper left", framealpha=0.95)
            print(f"{design} {tag}: hands={np.round(r['emp'],3)} "
                  f"P(d|h*)={np.round(r['Lstar'],3)} "
                  f"<P(d)>={np.round(r['pred_mean'],3)} h*={r['hstar']} "
                  f"held={r['held_frac']:.2f}")

    fig.suptitle(
        "BIB hands are samples from the predictive distribution "
        r"$P(d)=\sum_h P(h)\,P(d|h)$" "\n"
        "When JM-off freezes the belief, "
        r"$P(d)\!\to\!P(d|h^*)$ — a fixed categorical, so hands stay noisy "
        f"(bib--bib, tie={args.tie}, $T$={args.T}, 2nd half, seed={args.seed})",
        fontsize=10.5, y=1.02)
    fig.subplots_adjust(hspace=0.32, wspace=0.18)
    out = os.path.join(HERE, f"fig_hand_histogram_{args.tie}_seed{args.seed}.png")
    fig.savefig(out); fig.savefig(out.replace(".png", ".pdf"))
    print("wrote:", out)


if __name__ == "__main__":
    main()
