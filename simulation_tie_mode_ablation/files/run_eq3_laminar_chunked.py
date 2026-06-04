"""
Incremental eq3 laminar runner. Runs as many (design, run) jobs as fit in a
wall-clock budget, saving each run's analysis-window pmax to its own file under
tmp_eq3/. Re-invoke until all 4 designs x 20 runs = 80 jobs are done, then run
with --assemble to build pmax_<design>_eq3.npz (hybrid-compatible layout).

eq3 = (tie=skip, defeat=random_other).  Same engine/scale as run_hybrid_laminar.
"""
import argparse, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rpsgame_reward_tie import AgentReward

OUT_DIR = HERE.parent / 'data' / 'scheme_ablation' / 'huge_laminar'
TMP = OUT_DIR / 'tmp_eq3'
SCHEME_EQ3 = ('skip', 'random_other')
DESIGNS = [('random', 'sample'), ('random', 'argmax'),
           ('structured', 'sample'), ('structured', 'argmax')]
DESIGN_TAG = {('random', 'sample'): 'rs', ('random', 'argmax'): 'ra',
              ('structured', 'sample'): 'ss', ('structured', 'argmax'): 'sa'}
N_STEPS = 200000
ANALYSIS_START = 100000
H_LEN, H_NUM = 50, 10


def run_one(args):
    init_mode, predict_mode, seed = args
    tie_mode, defeat_mode = SCHEME_EQ3
    ag1 = AgentReward('bib', H_LEN, H_NUM, init_mode, predict_mode,
                      tie_mode=tie_mode, defeat_mode=defeat_mode, seed=seed)
    ag2 = AgentReward('bib', H_LEN, H_NUM, init_mode, predict_mode,
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


def pending_jobs():
    jobs = []
    for d in DESIGNS:
        tag = DESIGN_TAG[d]
        for run_idx in range(20):
            f = TMP / f'{tag}_run{run_idx:02d}.npz'
            if not f.exists():
                jobs.append((d[0], d[1], run_idx))
    return jobs


def assemble():
    a = ANALYSIS_START
    win = N_STEPS - a
    for d in DESIGNS:
        tag = DESIGN_TAG[d]
        out = OUT_DIR / f'pmax_{tag}_eq3.npz'
        p1 = np.empty((20, win), dtype=np.float32)
        p2 = np.empty((20, win), dtype=np.float32)
        for run_idx in range(20):
            f = TMP / f'{tag}_run{run_idx:02d}.npz'
            dd = np.load(f)
            p1[run_idx] = dd['p1']; p2[run_idx] = dd['p2']
        np.savez_compressed(out, pmax1=p1, pmax2=p2,
                            analysis_start=a, n_steps=N_STEPS, n_runs=20)
        print(f'assembled {out.name} ({out.stat().st_size/1e6:.1f} MB)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--budget', type=float, default=35.0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--assemble', action='store_true')
    args = p.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)

    if args.assemble:
        assemble(); return

    jobs = pending_jobs()
    print(f'pending jobs: {len(jobs)}', flush=True)
    if not jobs:
        print('ALL DONE'); return
    t0 = time.time(); done = 0
    with Pool(args.workers) as pool:
        # submit in batches of `workers`, stop when budget exceeded
        i = 0
        while i < len(jobs) and (time.time() - t0) < args.budget:
            batch = jobs[i:i + args.workers]
            for init_mode, predict_mode, seed, p1, p2 in pool.imap_unordered(run_one, batch):
                tag = DESIGN_TAG[(init_mode, predict_mode)]
                np.savez_compressed(TMP / f'{tag}_run{seed:02d}.npz', p1=p1, p2=p2)
                done += 1
            i += args.workers
            print(f'  batch done, total saved this call={done}, '
                  f'elapsed={time.time()-t0:.0f}s', flush=True)
    remaining = len(pending_jobs())
    print(f'saved {done} this call; {remaining} still pending', flush=True)


if __name__ == '__main__':
    main()
