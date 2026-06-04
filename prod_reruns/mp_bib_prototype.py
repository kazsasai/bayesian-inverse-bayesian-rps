#!/usr/bin/env python3
"""
TASK 4 (generality probe): does the BIB inference self-organise to the SAME critical
class off rock-paper-scissors? We reuse the paper's exact inference core (BayesReward:
Bayesian update + conditional JM smoothing + inverse-Bayesian argmin-renewal from the
recent-m histogram) and swap ONLY the game to MATCHING PENNIES (2 actions, Nash = uniform,
zero-sum, no draw). Reward-based observation rule (Eq. 3), 2-symbol version:
    win    -> kansoku = my own action      (reinforce)
    defeat -> kansoku = the other symbol    ("look elsewhere"; = random_other for d_num=2)
Roles: agent1 = matcher (win iff same), agent2 = mismatcher (win iff different).

Observables (identical to the RPS paper): argmax-persistence T_argmax (consecutive argmax_h)
and laminar phases (consecutive steps with max_h P(h) > theta=0.4), pooled over both agents.
Fit truncated power law (powerlaw, Clauset x_min). Compare alpha to RPS ~1.43 (argmax) /
~1.34 (laminar).

Checkpointed; env: NR (runs, default 10), T (default 40000), BURN (default 8000),
PREDICT (sample|argmax, default sample), HNUM (default 10), BUDGET (s), WORKERS.
Run repeatedly until "complete", then: python mp_bib_prototype.py --fit
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np

def _find_engine():
    env = os.environ.get("ENGINE_DIR")
    here = Path(__file__).resolve()
    cands = ([Path(env)] if env else []) + [
        b / sub for b in [here.parent, *here.parents]
        for sub in ("simulation_tie_mode_ablation/files",
                    "BIB_Levy_github/simulation_tie_mode_ablation/files")]
    for c in cands:
        if (c / "rpsgame_reward_tie.py").exists():
            return str(c)
    raise SystemExit("rpsgame_reward_tie.py not found; set ENGINE_DIR")

sys.path.insert(0, _find_engine())
from rpsgame_reward_tie import BayesReward   # the paper's exact inference core

THETA = 0.4
SYMS = np.array(["0", "1"])
CACHE = Path(__file__).resolve().parent / "mp_bib_cache.json"


def runlengths(mask):
    out, cur = [], 0
    for v in mask:
        if v: cur += 1
        else:
            if cur: out.append(cur)
            cur = 0
    if cur: out.append(cur)
    return out


def argmax_runs(seq):
    out, cur, prev = [], 0, None
    for x in seq:
        if x == prev: cur += 1
        else:
            if cur: out.append(cur)
            cur, prev = 1, x
    if cur: out.append(cur)
    return out


def mp_win(me, opp, role):
    # role 'match': win iff equal; 'mismatch': win iff different
    return (me == opp) if role == "match" else (me != opp)


def step(bayes, role, my, opp):
    win = mp_win(my, opp, role)
    kansoku = my if win else SYMS[1 - int(my)]      # Eq.3 (2-symbol): win->own, defeat->other
    bayes.inference(kansoku)
    bayes.update_history(kansoku)
    bayes.inverse()                                  # BIB renewal (argmin-hypothesis)


def run_one(args):
    seed, T, burn, predict, hnum = args
    b1 = BayesReward(h_num=hnum, d_num=2, d_type=SYMS, h_length=50,
                     init_mode="random", predict_mode=predict, seed=seed)
    b2 = BayesReward(h_num=hnum, d_num=2, d_type=SYMS, h_length=50,
                     init_mode="random", predict_mode=predict, seed=seed + 100000)
    am1 = np.empty(T, np.int32); am2 = np.empty(T, np.int32)
    lm1 = np.empty(T, bool); lm2 = np.empty(T, bool)
    for t in range(T):
        h1 = b1.expect(); h2 = b2.expect()
        am1[t] = b1.argmax_h(); am2[t] = b2.argmax_h()
        lm1[t] = b1.h_prov.max() > THETA; lm2[t] = b2.h_prov.max() > THETA
        step(b1, "match", h1, h2); step(b2, "mismatch", h2, h1)
    arg = argmax_runs(am1[burn:].tolist()) + argmax_runs(am2[burn:].tolist())
    lam = runlengths(lm1[burn:]) + runlengths(lm2[burn:])
    return arg, lam


def fit():
    import powerlaw, warnings
    warnings.filterwarnings("ignore")
    d = json.load(open(CACHE))
    for design, rec in d.items():
        for obs in ("argmax", "laminar"):
            a = np.array([x for x in rec[obs] if x >= 1])
            if len(a) < 100:
                print(f"{design}/{obs}: too few ({len(a)})"); continue
            f = powerlaw.Fit(a, discrete=True, verbose=False)
            R, p = f.distribution_compare("truncated_power_law", "exponential",
                                          normalized_ratio=True)
            print(f"{design:18s} {obs:8s}: n={len(a):6d}  alpha_TPL={f.truncated_power_law.alpha:.3f} "
                  f" alpha_PL={f.power_law.alpha:.3f}  xmin={f.xmin:.0f}  "
                  f"1/Lambda={1/f.truncated_power_law.Lambda:.0f}  TPL>EXP R={R:.1f}(p={p:.2f})  "
                  f"runs={rec['runs_done']}")
    print("\n(RPS reference: argmax alpha ~ 1.43, laminar alpha ~ 1.34)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fit", action="store_true")
    if ap.parse_args().fit:
        fit(); return
    from multiprocessing import Pool
    NR = int(os.environ.get("NR", "10")); T = int(os.environ.get("T", "40000"))
    BURN = int(os.environ.get("BURN", "8000")); PRED = os.environ.get("PREDICT", "sample")
    HNUM = int(os.environ.get("HNUM", "10")); BUDGET = float(os.environ.get("BUDGET", "40"))
    W = int(os.environ.get("WORKERS", "4"))
    design = f"randinit_{PRED}_h{HNUM}"
    d = json.load(open(CACHE)) if CACHE.exists() else {}
    rec = d.get(design, {"runs_done": 0, "argmax": [], "laminar": []})
    t0 = time.time()
    with Pool(W) as pool:
        while rec["runs_done"] < NR and time.time() - t0 < BUDGET:
            need = min(W, NR - rec["runs_done"])
            jobs = [(5000 + 7 * rec["runs_done"] + i, T, BURN, PRED, HNUM) for i in range(need)]
            for arg, lam in pool.imap_unordered(run_one, jobs):
                rec["argmax"] += arg; rec["laminar"] += lam; rec["runs_done"] += 1
            d[design] = rec; json.dump(d, open(CACHE, "w"))
            print(f"{design}: runs={rec['runs_done']}/{NR} argmax_n={len(rec['argmax'])} "
                  f"lam_n={len(rec['laminar'])} ({time.time()-t0:.0f}s)", flush=True)
    print("complete" if rec["runs_done"] >= NR else "partial (re-run to continue)")


if __name__ == "__main__":
    main()
