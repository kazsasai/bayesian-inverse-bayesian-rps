"""Incremental BO-BO laminar runner (max_h P(h) trajectories) for Fig 3(b).

Mirror of run_eq3_laminar_chunked.py but with BO agents (no inverse step) under
the same published reward observation rule (tie=skip, defeat=random_other).
Saves each run's analysis-window pmax to tmp_bo/, then --assemble builds
pmax_<design>_bo.npz (hybrid/eq3-compatible layout: keys pmax1, pmax2).

Re-invoke until "ALL DONE", then run with --assemble.
"""
import argparse, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rpsgame_reward_tie import AgentReward

OUT_DIR = HERE.parent / 'data' / 'scheme_ablation' / 'huge_laminar'
TMP = OUT_DIR / 'tmp_bo'
SCHEME = ('skip', 'random_other')          # published observation rule
DESIGNS = [('random', 'sample'), ('random', 'argmax'),
           ('structured', 'sample'), ('structured', 'argmax')]
DESIGN_TAG = {('random', 'sample'): 'rs', ('random', 'argmax'): 'ra',
              ('structured', 'sample'): 'ss', ('structured', 'argmax'): 'sa'}
N_STEPS = 200000
ANALYSIS_START = 100000
H_LEN, H_NUM = 50, 10
N_RUNS = 20


def run_one(args):
    init_mode, predict_mode, seed = args
    tie_mode, defeat_mode = SCHEME
    ag1 = AgentReward('bo', H_LEN, H_NUM, init_mode, predict_mode,
                      tie_mode=tie_mode, defeat_mode=defeat_mode, seed=seed)
    ag2 = AgentReward('bo', H_LEN, H_NUM, init_mode, predict_mode,
                      tie_mode=tie_mode, defeat_mode=defeat_mode,
                      seed=seed + 100000)
    p1 = np.empty(N_STEPS, dtype=np.float32)
    p2 = np.empty(N_STEPS, dtype=np.float32)
    for t in range(N_STEPS):
        h1 = ag1.choice(); h2 = ag2.choice()
        p1[t] = float(ag1.bayes.h_prov.max())
        p2[t] = float(ag2.bayes.h_prov.max())
        ag1.update_from_outcome(h1, h2); ag2.update_from_outcome(h2, h1)
    a = ANALYSIS_START
    return init_mode, predict_mode, seed, p1[a:].copy(), p2[a:].copy()


def pending():
    jobs = []
    for d in DESIGNS:
        tag = DESIGN_TAG[d]
        for r in range(N_RUNS):
            if not (TMP / f'{tag}_run{r:02d}.npz').exists():
                jobs.append((d[0], d[1], r))
    return jobs


def assemble():
    a = ANALYSIS_START; win = N_STEPS - a
    for d in DESIGNS:
        tag = DESIGN_TAG[d]
        p1 = np.empty((N_RUNS, win), dtype=np.float32)
        p2 = np.empty((N_RUNS, win), dtype=np.float32)
        for r in range(N_RUNS):
            dd = np.load(TMP / f'{tag}_run{r:02d}.npz')
            p1[r] = dd['p1']; p2[r] = dd['p2']
        out = OUT_DIR / f'pmax_{tag}_bo.npz'
        np.savez_compressed(out, pmax1=p1, pmax2=p2, analysis_start=a,
                            n_steps=N_STEPS, n_runs=N_RUNS)
        print(f'assembled {out.name} ({out.stat().st_size/1e6:.1f} MB)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=36.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--assemble', action='store_true')
    args = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    if args.assemble:
        assemble(); return
    jobs = pending()
    print(f'pending: {len(jobs)}', flush=True)
    if not jobs:
        print('ALL DONE'); return
    t0 = time.time(); done = 0
    with Pool(args.workers) as pool:
        i = 0
        while i < len(jobs) and time.time() - t0 < args.budget:
            batch = jobs[i:i + args.workers]
            for im, pm, seed, p1, p2 in pool.imap_unordered(run_one, batch):
                np.savez_compressed(TMP / f'{DESIGN_TAG[(im, pm)]}_run{seed:02d}.npz',
                                    p1=p1, p2=p2)
                done += 1
            i += args.workers
    print(f'saved {done} this call; {len(pending())} remain', flush=True)


if __name__ == '__main__':
    main()
