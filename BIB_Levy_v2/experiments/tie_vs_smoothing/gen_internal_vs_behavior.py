#!/usr/bin/env python3
"""
gen_internal_vs_behavior.py
Generate duration arrays for the internal-vs-behavioral tail comparison.

--mode equil   : bib-bib at the uniform-Nash fixed point (adversarially pinned)
--mode biased  : bib vs a FIXED biased opponent (0.6,0.2,0.2) -> pinning removed

Saves <mode>.npz with arrays: argmax_persist, hand_runs, win_streaks, hand_freq.
"""
import argparse, os
from multiprocessing import Pool
import numpy as np
import engine_tie_smoothing as sim

HERE = os.path.dirname(os.path.abspath(__file__))
HANDS = sim.HANDS


def gen_equil(n_runs, T, astart, seed_base=0):
    agg, _, _ = sim.parallel_runs("bib", "bib", n_steps=T, n_runs=n_runs,
        analysis_start=astart, h_length=50, h_num=10,
        init_mode="random", predict_mode="sample",
        tie_mode="skip", defeat_mode="random_other",
        n_workers=6, seed_base=seed_base, jm_smoothing=True)
    # hand_freq from a few representative runs not tracked here; set uniform-ish via None
    return (np.array(agg["T_argmax1"]), np.array(agg["hand1_runs"]),
            np.array(agg["win_runs"]), None)


def _biased_one(args):
    s, T, astart, bias = args
    ag = sim.AgentReward("bib", 50, 10, "random", "sample", tie_mode="skip",
                         defeat_mode="random_other", seed=s, jm_smoothing=True)
    orng = np.random.default_rng(10000 + s)
    am = np.empty(T, dtype=int); hd = np.empty(T, dtype='<U1'); res = np.empty(T, dtype='<U10')
    for t in range(T):
        h1 = ag.choice()
        h2 = orng.choice(HANDS, p=bias)
        am[t] = ag.argmax_h(); hd[t] = h1; res[t] = sim.rps(h1, h2)
        ag.update_from_outcome(h1, h2)
    hb = np.array([(hd[astart:] == h).sum() for h in HANDS], float)
    return (sim.consecutive_runs(am[astart:]),
            sim.consecutive_runs(hd[astart:]),
            sim.streak_lengths(res[astart:] == 'win'), hb, T - astart)


def gen_biased(n_runs, T, astart, bias=(0.6, 0.2, 0.2), n_workers=6, seed_base=0):
    jobs = [(s, T, astart, bias) for s in range(seed_base, seed_base + n_runs)]
    with Pool(processes=n_workers) as pool:
        outs = pool.map(_biased_one, jobs)
    AM = [o[0] for o in outs]; HR = [o[1] for o in outs]; WR = [o[2] for o in outs]
    hb = sum(o[3] for o in outs); nb = sum(o[4] for o in outs)
    return (np.concatenate(AM), np.concatenate(HR),
            np.concatenate(WR), hb / nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["equil", "biased"], required=True)
    ap.add_argument("--n_runs", type=int, default=150)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--astart", type=int, default=1000)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--append", action="store_true",
                    help="concatenate onto the existing npz (accumulate runs)")
    a = ap.parse_args()
    if a.mode == "equil":
        am, hr, wr, hf = gen_equil(a.n_runs, a.T, a.astart, seed_base=a.seed_base)
    else:
        am, hr, wr, hf = gen_biased(a.n_runs, a.T, a.astart, seed_base=a.seed_base)
    out = os.path.join(HERE, f"data_ivb_{a.mode}.npz")
    n_runs_total = a.n_runs
    if a.append and os.path.exists(out):
        prev = np.load(out)
        am = np.concatenate([prev["argmax_persist"], am])
        hr = np.concatenate([prev["hand_runs"], hr])
        wr = np.concatenate([prev["win_streaks"], wr])
        n_runs_total = int(prev["n_runs"]) + a.n_runs if "n_runs" in prev.files else a.n_runs
        if hf is not None and "hand_freq" in prev.files and not np.isnan(prev["hand_freq"]).any():
            hf = 0.5 * (hf + prev["hand_freq"])   # approx pooled (equal chunks)
    np.savez(out, argmax_persist=am, hand_runs=hr, win_streaks=wr,
             hand_freq=(hf if hf is not None else np.array([np.nan]*3)),
             T=a.T, n_runs=n_runs_total, astart=a.astart)
    for k, v in [("argmax_persist", am), ("hand_runs", hr), ("win_streaks", wr)]:
        f = sim.fit_distributions(v)
        b = (f.get("best", "err"), f.get("alpha_best", float("nan")))
        print(f"  {k:16s} n={len(v):>7} best={b[0]} alpha={b[1]:.2f}")
    if hf is not None:
        print("  hand_freq r/p/s =", np.round(hf, 3))
    print("wrote:", out)


if __name__ == "__main__":
    main()
