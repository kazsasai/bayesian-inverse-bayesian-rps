"""
Run hybrid scheme at huge scale capturing max_h P(h) trajectories per step.
Output: per-design npz with pmax1, pmax2 over the analysis window, for
laminar-phase R7 verification (compare to published alpha^lam = 1.34).

Lightweight: stores only max(P) (1 float per step per agent), not the full
posterior matrix. ~16 MB per cell at huge scale.
"""
import argparse, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rpsgame_reward_tie import AgentReward, rps

OUT_DIR = HERE.parent / 'data' / 'scheme_ablation' / 'huge_laminar'

SCHEME_HYBRID = ('uniform', 'opponent')
DESIGNS = [('random', 'sample'), ('random', 'argmax'),
           ('structured', 'sample'), ('structured', 'argmax')]
DESIGN_TAG = {('random', 'sample'): 'rs', ('random', 'argmax'): 'ra',
              ('structured', 'sample'): 'ss', ('structured', 'argmax'): 'sa'}


def run_one(args):
    init_mode, predict_mode, n_steps, h_length, h_num, seed = args
    tie_mode, defeat_mode = SCHEME_HYBRID
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
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--analysis-start', type=int, default=100000)
    p.add_argument('--h-length', type=int, default=50)
    p.add_argument('--h-num', type=int, default=10)
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for design in DESIGNS:
        tag = DESIGN_TAG[design]
        out_path = OUT_DIR / f'pmax_{tag}_hybrid.npz'
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
        # Save only the analysis window to halve disk
        a = args.analysis_start
        np.savez_compressed(out_path,
                            pmax1=pmax1_all[:, a:],
                            pmax2=pmax2_all[:, a:],
                            analysis_start=a,
                            n_steps=args.n_steps,
                            n_runs=args.n_runs)
        print(f'[{tag}] {time.time()-t0:.0f}s -> {out_path.name} '
              f'({out_path.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
