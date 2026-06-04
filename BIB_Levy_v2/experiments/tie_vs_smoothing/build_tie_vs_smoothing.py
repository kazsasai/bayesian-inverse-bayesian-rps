#!/usr/bin/env python3
"""
build_tie_vs_smoothing.py

Hypothesis (Sasai, 2026-05-21)
------------------------------
Does the tie->uniform observation rule (Fig 7 "caseA") substitute for the
conditional Jelinek-Mercer smoothing (Eq. 5), making the smoothing unnecessary?

Rationale: at the uniform-play fixed point a tie occurs ~1/N of the time
(~33% for N=3). With tie_mode='uniform' each tie injects a uniformly drawn
observation -- a zero-mean stochastic perturbation (the engine's own comment) --
which could keep the posterior from collapsing, the role the conditional JM
smoothing plays for tie_mode='skip'.

Design: 2x2 factorial
    tie_mode     in {skip, uniform}      (skip = Eq.3 / eq3; uniform = caseA)
    jm_smoothing in {True, False}        (Eq. 5 on/off)
defeat_mode fixed = random_other (so eq3 and caseA are the only tie variants).
Pair = bib-bib. Designs swept (default: all four rs/ra/ss/sa).

Metrics per cell:
    alpha       argmax-persistence best-fit exponent (Levy if 1<alpha<=3)
    best        pl / tpl / exp
    sigma_bar   mean posterior spread sigma(P(h)) (collapse indicator)
    quits_rate  fraction of ties (sanity)
    collapse    fraction of runs whose min sigma ~ 0 (posterior pinned)

Prediction if the hypothesis holds:
    (uniform, OFF) stays Levy (alpha~1.4) with healthy sigma_bar and no collapse
    -> tie->uniform makes the smoothing redundant.
    (skip,   OFF) collapses / loses the power law -> smoothing needed there.
"""
import argparse
import csv
import os
import numpy as np
import engine_tie_smoothing as sim

HERE = os.path.dirname(os.path.abspath(__file__))

DESIGN_TAG = {"rs": ("random", "sample"), "ra": ("random", "argmax"),
              "ss": ("structured", "sample"), "sa": ("structured", "argmax")}

SCALES = {
    "small":  dict(n_steps=2000,  analysis_start=1000,  n_runs=100),
    "medium": dict(n_steps=10000, analysis_start=5000,  n_runs=200),
}


def run_cell(design, tie_mode, jm_smoothing, cfg, n_workers):
    init_mode, predict_mode = DESIGN_TAG[design]
    agg, rstats, sstats = sim.parallel_runs(
        "bib", "bib",
        n_steps=cfg["n_steps"], n_runs=cfg["n_runs"],
        analysis_start=cfg["analysis_start"],
        h_length=50, h_num=10,
        init_mode=init_mode, predict_mode=predict_mode,
        tie_mode=tie_mode, defeat_mode="random_other",
        n_workers=n_workers, seed_base=0, jm_smoothing=jm_smoothing,
    )
    fit = sim.fit_distributions(agg["T_argmax1"])
    sig_means = np.array([s["sig1_mean"] for s in sstats])
    sig_mins = np.array([s["sig1_min"] for s in sstats])
    qrs = [r["quits_count"] / r["n_post"] for r in rstats if r["n_post"] > 0]
    # collapse: posterior pinned to a vertex -> sigma close to its max possible
    # (for N_h=10 uniform sigma~0; a single-vertex posterior has sigma~0.3).
    # We flag *low* sigma_bar (over-concentration prevented) separately; the
    # always-on collapse signature is a *tight low* band, so we report sigma_bar.
    return {
        "design": design,
        "tie_mode": tie_mode,
        "jm_smoothing": int(jm_smoothing),
        "scheme": "eq3" if tie_mode == "skip" else "caseA",
        "alpha": fit.get("alpha_best", np.nan),
        "best": fit.get("best", "error"),
        "levy": fit.get("levy_region", False),
        "n_durations": fit.get("n", 0),
        "sigma_bar": float(sig_means.mean()),
        "sigma_bar_sd": float(sig_means.std()),
        "sigma_min_med": float(np.median(sig_mins)),
        "quits_rate": float(np.mean(qrs)) if qrs else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES), default="small")
    ap.add_argument("--designs", nargs="*", default=["rs", "ra", "ss", "sa"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cfg = SCALES[args.scale]

    rows = []
    print(f"2x2 tie x smoothing | scale={args.scale} "
          f"(T={cfg['n_steps']}, runs={cfg['n_runs']}) | designs={args.designs}\n")
    header = (f"{'design':>6} {'scheme':>6} {'tie':>8} {'JM':>4} "
              f"{'alpha':>7} {'best':>5} {'levy':>5} {'sigma_bar':>10} "
              f"{'quits':>7} {'n':>7}")
    print(header)
    print("-" * len(header))
    for design in args.designs:
        for tie_mode in ("skip", "uniform"):
            for jm in (True, False):
                r = run_cell(design, tie_mode, jm, cfg, args.workers)
                rows.append(r)
                print(f"{r['design']:>6} {r['scheme']:>6} {r['tie_mode']:>8} "
                      f"{r['jm_smoothing']:>4} {r['alpha']:>7.3f} {r['best']:>5} "
                      f"{str(r['levy']):>5} {r['sigma_bar']:>10.4f} "
                      f"{r['quits_rate']:>7.3f} {r['n_durations']:>7}")
        print()

    out_csv = os.path.join(HERE, f"results_tie_vs_smoothing_{args.scale}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote:", out_csv)


if __name__ == "__main__":
    main()
