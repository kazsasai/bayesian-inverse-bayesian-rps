"""
Run eq3 (published) scheme at huge scale capturing max_h P(h) trajectories
per step, for like-for-like comparison with the hybrid laminar run
(run_hybrid_laminar.py). Same engine, same scale, same output layout.

eq3 = (tie=skip, defeat=random_other)  -- published Eq. (3).

Output: per-design npz pmax_<design>_eq3.npz with pmax1, pmax2 over the
analysis window. Analyse with: analyze_laminar_hybrid.py adapted via --scheme.
"""
import argparse, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rpsgame_reward_tie import AgentReward

OUT_DIR = HERE.parent / 'data' / 'scheme_ablation' / 'huge_laminar'

SCHEME_EQ3 = ('skip', 'random_other')
DESIGNS = [('random', 'sample'), ('random', 'argmax'),
           ('structured', 'sample'), ('structured', 'argmax')]
DESIGN_TAG = {('random', 'sample'): 'rs', ('random', 'argmax'): 'ra',
              ('structured', 'sample'): 'ss', ('structured', 'argmax'): 'sa'}


def run_one(args):
    init_mode, predict_mode, n_steps, h_length, h_num, seed = args
    tie_mode, defeat_mode = SCHEME_EQ3
    ag1 = AgentReward('bib', h_length, h_num, init_mode, predict_mode,
                      tie_mode=tie_mode, defeat_mode=defeat_mode, seed=seed)
    ag2 = AgentReward('bib', h_length, h_num, init_mode, predict_mode,
                      tie_mode=tie_mode, defeat_mode=defeat_mode,
                      seed=seed + 100000)
    pmax1 = np.empty(n_steps, dtype=np.float32)
    pmax2 = np.empty(n_steps, dtype=np.float32)
    for t in range(n_steps):
        h1 = ag1.choice()
        h2 = ag2.choice()
        pmax1[t] = float(ag1.bayes.h_prov.max())
        pmax2[t] = float(ag2.bayes.h_prov.max())
        ag1.update_from_outcome(h1, h2)
        ag2.update_from_outcome(h2, h1)
    return seed, pmax1, pmax2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-steps', type=int, default=200000)
    p.add_argument('--n-runs', type=int, default=20)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--analysis-start', type=int, default=100000)
    p.add_argument('--h-length', type=int, default=50)
    p.add_argument('--h-num', type=int, default=10)
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for design in DESIGNS:
        tag = DESIGN_TAG[design]
        out_path = OUT_DIR / f'pmax_{tag}_eq3.npz'
        if out_path.exists():
            print(f'[skip] {out_path.name} exists')
            continue
        t0 = time.time()
        job_args = [(design[0], design[1], args.n_steps,
                     args.h_length, args.h_num, run_idx)
                    for run_idx in range(args.n_runs)]
        pmax1_all = np.empty((args.n_runs, args.n_steps), dtype=np.float32)
        pmax2_all = np.empty((args.n_runs, args.n_steps), dtype=np.float32)
        with Pool(args.workers) as pool:
            for i, (seed, p1, p2) in enumerate(
                    pool.imap_unordered(run_one, job_args)):
                pmax1_all[i] = p1
                pmax2_all[i] = p2
        a = args.analysis_start
        np.savez_compressed(out_path,
                            pmax1=pmax1_all[:, a:],
                            pmax2=pmax2_all[:, a:],
                            analysis_start=a,
                            n_steps=args.n_steps,
                            n_runs=args.n_runs)
        print(f'[{tag}] {time.time()-t0:.0f}s -> {out_path.name} '
              f'({out_path.stat().st_size/1e6:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
