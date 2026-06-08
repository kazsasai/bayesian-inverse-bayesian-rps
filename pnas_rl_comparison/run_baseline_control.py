#!/usr/bin/env python3
r"""Driver for the RL-baseline control (SI Appendix), built on the BIB harness.

Runs each baseline (WSLS, Q-learning, regret matching) through
``rps_baselines.run_pair_objs`` -- the same loop as ``rpsgame_reward.run_pair``,
producing identical DataFrame columns -- against three opponents (uniform-random,
self-play, fixed-biased 0.6/0.2/0.2) over many seeds at T=2x10^5, and pools two
run-length observables via the repo's ``consecutive_runs``:

  * __argmax : persistence of the dominant internal preference (argmax1 column).
               regret matching -> argmax cumulative regret; Q/WSLS -> played
               action (no separate internal preference).  This is the figure
               observable.  Self-play pools agent1+agent2 (both are the learner);
               vs a non-learning opponent pools agent1 ONLY (the opponent's
               argmax is degenerate: argmax2 == -1 -> a single length-T run).
  * __action : played-action run length (h1 column); behaviour-table observable.

Output: data/baseline_dwells.npz {<method>__<opponent>__<obs>} + baseline_meta.json.
BIB curves are NOT produced here (validated durations_*.json; bib-random -> T1).

Usage:  python run_baseline_control.py [--seeds 40] [--T 200000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import rps_baselines as B

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "simulation", "reward_huge_v2"))
from rpsgame_reward import consecutive_runs  # canonical run-length extraction

METHOD_KIND = {"WSLS": "wsls", "Qlearning": "ql", "RegretMatching": "rm"}
OPP_KIND = {"random": "random", "fixed_biased": "biased"}  # self_play handled separately


def run_condition(method, opponent, T, seeds):
    kind = METHOD_KIND[method]
    selfplay = (opponent == "self_play")
    a2 = kind if selfplay else OPP_KIND[opponent]
    p_arg, p_act = [], []
    for s in seeds:
        df = B.run_pair_objs(kind, a2, T, seed=s)
        p_arg.append(consecutive_runs(df["argmax1"].values))
        p_act.append(consecutive_runs(df["h1"].values))
        if selfplay:                       # agent2 is also the learner -> pool it
            p_arg.append(consecutive_runs(df["argmax2"].values))
            p_act.append(consecutive_runs(df["h2"].values))
    return {f"{method}__{opponent}__argmax": np.concatenate(p_arg),
            f"{method}__{opponent}__action": np.concatenate(p_act)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--methods", nargs="+",
                    default=["WSLS", "Qlearning", "RegretMatching"])
    ap.add_argument("--opponents", nargs="+",
                    default=["random", "self_play", "fixed_biased"])
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(args.seeds))

    pooled, summary = {}, []
    t0 = time.time()
    for opp in args.opponents:
        for m in args.methods:
            tic = time.time()
            cond = run_condition(m, opp, args.T, seeds)
            pooled.update(cond)
            for k, v in cond.items():
                vv = v[v >= 1]
                summary.append({"key": k, "n": int(vv.size),
                                "max": int(vv.max()) if vv.size else 0,
                                "mean": round(float(vv.mean()), 2) if vv.size else 0.0})
            print(f"[{time.time()-tic:5.1f}s] {opp:12s} {m:15s} done", flush=True)

    np.savez_compressed(os.path.join(args.out, "baseline_dwells.npz"), **pooled)
    meta = {
        "T": args.T, "n_seeds": args.seeds, "seeds": seeds,
        "methods": args.methods, "opponents": args.opponents,
        "harness": "rps_baselines.run_pair_objs (mirrors rpsgame_reward.run_pair); "
                   "HANDS/rps imported from rpsgame_reward (identical encoding)",
        "observables": {"__argmax": "consecutive_runs(argmax1) [+argmax2 self-play]; "
                                    "RM=argmax regret, Q/WSLS=played action",
                        "__action": "consecutive_runs(h1) [+h2 self-play]; played action"},
        "pooling": "self-play pools both agents; vs non-learning opponent agent-1 only",
        "params": {"WSLS": "deterministic win-stay/lose-or-tie-shift to BR",
                   "Qlearning": "tabular recall-1; lr=0.1 gamma=0.9 eps=0.1",
                   "RegretMatching": "vanilla; sigma=R+/sum R+; argmax cumulative regret"},
        "biased_p": [0.6, 0.2, 0.2],
        "elapsed_sec": round(time.time() - t0, 1), "summary": summary,
    }
    with open(os.path.join(args.out, "baseline_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.out}/baseline_dwells.npz ({len(pooled)} arrays) "
          f"+ baseline_meta.json   [{meta['elapsed_sec']}s]")


if __name__ == "__main__":
    main()
