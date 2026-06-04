#!/usr/bin/env python3
r"""Verify the SI Sec. 13 claim "a truncated power law is preferred over an
exponential in every [bot] condition" under the PROPER Clauset auto-x_min
convention (the prior session's per-condition fits used a fixed x_min=1).

For each of the 15 Brockbank & Vul bot conditions we pool the players'
transition-run lengths, fit with powerlaw.Fit (discrete, Clauset KS-optimal
x_min), and compare truncated-power-law (TPL) vs exponential (EXP) by the
normalised log-likelihood ratio R (R > 0 => TPL preferred).

Raw data: ./rps/data/{v2,v3}/*.json  (run fetch_bv_data.sh first), or $RPS_DATA.
Usage:  python verify_figS_behaviour_clauset.py
"""
import os, json, glob, collections, warnings
import numpy as np
import powerlaw
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
_cand = [os.environ.get("RPS_DATA"), os.path.join(HERE, "rps", "data"), "rps/data"]
DATA_DIR = next((d for d in _cand if d and os.path.isdir(os.path.join(d, "v2"))), None)
if DATA_DIR is None:
    raise SystemExit("No raw B&V data; run fetch_bv_data.sh first.")
MOVE = {"rock": 0, "paper": 1, "scissors": 2}


def runs(symbols):
    out, cur, prev = [], 0, object()
    for s in symbols:
        if s == prev:
            cur += 1
        else:
            if cur:
                out.append(cur)
            cur, prev = 1, s
    if cur:
        out.append(cur)
    return out


def load(ver, by_cond):
    for f in glob.glob(os.path.join(DATA_DIR, ver, "*.json")):
        if "freeResp" in f or "sliderData" in f:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        strat, rounds = d.get("player2_bot_strategy"), d.get("rounds")
        if not strat or not rounds:
            continue
        mv = [MOVE[r["player1_move"]] for r in rounds
              if r.get("player1_move") in MOVE
              and r.get("player1_outcome") in ("win", "loss", "tie")]
        if len(mv) < 50:
            continue
        by_cond[(ver, strat)].extend(runs((np.diff(mv) % 3).tolist()))


def main():
    by_cond = collections.defaultdict(list)
    load("v2", by_cond)
    load("v3", by_cond)

    print(f"{'condition':28s} {'ver':3s} {'n':>6s} {'xmin':>4s} "
          f"{'TPL_a':>6s} {'R(TPLvsEXP)':>12s} {'p':>7s}  verdict")
    npref = 0
    rows = sorted(by_cond.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    for (ver, strat), data in rows:
        arr = np.array(data, dtype=int)
        fit = powerlaw.Fit(arr, discrete=True, verbose=False)
        R, p = fit.distribution_compare("truncated_power_law", "exponential",
                                        normalized_ratio=True)
        pref = R > 0
        npref += pref
        print(f"{strat[:28]:28s} {ver:3s} {arr.size:6d} {int(fit.xmin):4d} "
              f"{fit.truncated_power_law.alpha:6.2f} {R:12.2f} {p:7.3f}  "
              f"{'TPL' if pref else 'EXP'}")
    print(f"\nTPL preferred (Clauset auto-x_min) in {npref}/{len(rows)} conditions.")
    print("SI claim: 'a truncated power law preferred over an exponential in "
          "every condition.'")


if __name__ == "__main__":
    main()
