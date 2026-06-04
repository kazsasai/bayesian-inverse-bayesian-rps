#!/usr/bin/env python3
r"""Regenerate item1_drift_identity.pdf (SI Sec. 1, Fig. \ref{si:fig-drift}).

Numerical check of the drift identity used in the first-return argument:
the per-step drift of the log-posterior ratio x = log[P(h_i)/P(h_j)] equals a
difference of Kullback--Leibler divergences to the data,

    <eta> = E_{delta~p}[ log L_i(delta) - log L_j(delta) ]
          = D_KL(p || L_j) - D_KL(p || L_i).

For 60 random triples (p, L_i, L_j) on the 3-outcome reward simplex
delta in {+1, 0, -1} we plot the MEASURED drift (sample average of the
per-step log-likelihood-ratio increment under i.i.d. delta ~ p) against the
PREDICTED KL difference (computed in closed form).  The points lie on y = x;
the residual scatter is Monte-Carlo error and vanishes as the sample size
grows (Pearson r ~ 0.99999 at SAMPLES = 2e5).

Self-contained: requires only numpy + matplotlib, no external data.
Usage:  python make_item1_drift_identity.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE / "item1_drift_identity.pdf"
OUT_PNG = HERE / "item1_drift_identity.png"

N_TRIPLES = 60          # matches the SI caption ("over 60 random triples")
SAMPLES = 200_000       # i.i.d. draws delta ~ p per triple (sets the MC error)
SEED = 7

# PNAS-unified style (same rcParams as the other SI figure builders).
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def random_dist(rng, n=3, alpha=0.85, floor=0.02):
    """A random categorical distribution on n outcomes (Dirichlet, with a
    small floor so the log-likelihoods stay finite).  alpha/floor are chosen so
    the KL gaps stay in the moderate +/-2 range typical of the reward
    likelihoods, rather than producing rare near-degenerate triples."""
    q = rng.dirichlet(np.full(n, alpha))
    q = q + floor
    return q / q.sum()


def main():
    rng = np.random.default_rng(SEED)
    outcomes = np.arange(3)            # delta in {+1, 0, -1} indexed 0,1,2
    predicted = np.empty(N_TRIPLES)
    measured = np.empty(N_TRIPLES)

    for k in range(N_TRIPLES):
        p = random_dist(rng)           # data distribution
        Li = random_dist(rng)          # likelihood of hypothesis i
        Lj = random_dist(rng)          # likelihood of hypothesis j
        logratio = np.log(Li) - np.log(Lj)          # per-outcome increment
        # Predicted drift = D_KL(p||Lj) - D_KL(p||Li) = sum_d p (logLi - logLj).
        predicted[k] = float(np.sum(p * logratio))
        # Measured drift = sample mean of the increment under delta ~ p.
        draws = rng.choice(outcomes, size=SAMPLES, p=p)
        measured[k] = float(np.mean(logratio[draws]))

    r = float(np.corrcoef(predicted, measured)[0, 1])
    print(f"item1 drift identity: {N_TRIPLES} triples, SAMPLES={SAMPLES}, "
          f"Pearson r = {r:.6f}, range=[{predicted.min():.2f}, {predicted.max():.2f}]")

    lo = min(predicted.min(), measured.min())
    hi = max(predicted.max(), measured.max())
    pad = 0.12 * (hi - lo)
    line = np.array([lo - pad, hi + pad])

    fig, ax = plt.subplots(figsize=(3.5, 3.25))
    ax.plot(line, line, "--", color="black", lw=1.2, label="$y = x$", zorder=1)
    ax.scatter(predicted, measured, s=44, facecolor="#1f77b4",
               edgecolor="black", linewidth=0.6, zorder=2)
    ax.set_xlim(line); ax.set_ylim(line)
    ax.set_aspect("equal")
    ax.set_xlabel(r"predicted  $D_{\mathrm{KL}}(p\,\|\,L_j) - D_{\mathrm{KL}}(p\,\|\,L_i)$")
    ax.set_ylabel(r"measured drift  $\langle\eta\rangle$")
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.04, 0.84, "log-posterior-ratio drift\n= KL gap to the data",
            transform=ax.transAxes, fontsize=8.5, va="top")

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print("wrote", OUT_PDF.name, "/", OUT_PNG.name)


if __name__ == "__main__":
    main()
