"""
run_sharpness_plateau.py — Generate huge data for Paper A §3.9 (plateau)
                           and §3.10 (P(d|h) sharpness sweep).

Extends rpsgame_reward.py with:
  - Configurable P(d|h) sharpness parameter alpha for structured init
    templates (single peak: [alpha, (1-alpha)/2, (1-alpha)/2] and cyclic
    permutations + uniform)
  - Full posterior P(h) trajectory storage for plateau analysis

Records per run:
  - argmax1, argmax2 trajectories (full)
  - P(h) trajectories (full, both players)
  - hand sequences h1, h2
  - outcomes (win/defeat/quits)

For T = 10,000 × 20 runs × 96 conditions (4 design × 6 sharpness × 2 pair
[bib-bib, bo-bo]):
  - Each condition: ~30-60 sec on M1 with 13 workers
  - Total: ~30-60 min
  - Per-condition data size: ~10 MB (npz compressed)
  - Total data size: ~1 GB

Usage:
  python3 run_sharpness_plateau.py --workers 13

Place this file in the same directory as rpsgame_reward.py.
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rpsgame_reward import BayesReward, AgentReward, rps

OUT_DIR = Path('./data/sharpness_plateau')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Sharpness-parameterized structured init
# ============================================================
def make_sharpness_templates(h_num, d_num, alpha):
    """Make P(d|h) templates with controllable sharpness alpha.

    For d_num = 3:
      - 1 uniform template
      - 3 single-peak templates: [alpha, (1-alpha)/2, (1-alpha)/2] cyclic
      - 3 mild-peak templates (alpha mixed toward uniform)
      - 3 anti-peak templates: [(1-anti)/2, (1-anti)/2, anti] cyclic
                               where anti = max(alpha-0.3, 1/d_num)
    For h_num=10, returns 10 templates (extra padded with noise).
    """
    if d_num != 3:
        raise NotImplementedError(f"sharpness templates only for d_num=3 (got {d_num})")

    a = float(alpha)
    rest = (1.0 - a) / 2.0
    # mild: halfway between alpha and uniform
    a_mild = (a + 1.0/3) / 2.0
    rest_mild = (1.0 - a_mild) / 2.0
    # anti-peak: weight on two opposing hands
    anti = (1.0 - a) / 2.0  # smaller weight on the "preferred" hand
    rest_anti = (1.0 - anti) / 2.0

    base_templates = [
        [1/3, 1/3, 1/3],                       # 0: uniform
        [a, rest, rest],                       # 1: prefer hand 0
        [rest, a, rest],                       # 2: prefer hand 1
        [rest, rest, a],                       # 3: prefer hand 2
        [a_mild, rest_mild, rest_mild],        # 4: mild prefer 0
        [rest_mild, a_mild, rest_mild],        # 5: mild prefer 1
        [rest_mild, rest_mild, a_mild],        # 6: mild prefer 2
        [anti, rest_anti, rest_anti],          # 7: anti-prefer 0
        [rest_anti, anti, rest_anti],          # 8: anti-prefer 1
        [rest_anti, rest_anti, anti],          # 9: anti-prefer 2
    ]

    if h_num <= len(base_templates):
        return np.array(base_templates[:h_num])

    out = list(base_templates)
    rng_pad = np.random.default_rng(12345)
    while len(out) < h_num:
        base = base_templates[len(out) % len(base_templates)]
        noise = rng_pad.standard_normal(d_num) * 0.05
        v = np.clip(np.array(base) + noise, 1e-3, None)
        out.append((v / v.sum()).tolist())
    return np.array(out[:h_num])


class BayesRewardSharp(BayesReward):
    """Bayesian agent with controllable P(d|h) sharpness."""

    def __init__(self, alpha=0.8, h_num=10, d_num=3, d_type=None,
                 h_length=50, init_mode='random', predict_mode='sample',
                 seed=None):
        # Skip parent init for likelihood; reproduce manually with sharpness
        from rpsgame_reward import HANDS
        self.rng = np.random.default_rng(seed)
        self.h_num = h_num
        self.d_num = d_num
        self.d_type = np.array(d_type if d_type is not None else HANDS)
        self.h_length = h_length
        self.init_mode = init_mode
        self.predict_mode = predict_mode
        self.alpha = alpha

        if init_mode == 'random':
            self.likelihood = np.abs(self.rng.standard_normal((h_num, d_num)))
            self.likelihood /= self.likelihood.sum(axis=1, keepdims=True)
        elif init_mode == 'structured':
            self.likelihood = make_sharpness_templates(h_num, d_num, alpha)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        self.h_prov = np.ones(h_num) / h_num
        self.history = []


class AgentRewardSharp(AgentReward):
    """Agent with sharpness-aware Bayes."""

    def __init__(self, agent_type='bib', h_length=50, h_num=10,
                 init_mode='random', predict_mode='sample',
                 alpha=0.8, seed=None):
        from rpsgame_reward import HANDS
        self.type = agent_type
        self.rng = np.random.default_rng(seed)
        self.h_num = h_num
        self.N = 3
        if agent_type == 'random':
            self.bayes = None
        else:
            self.bayes = BayesRewardSharp(
                alpha=alpha, h_num=h_num, d_num=3, h_length=h_length,
                init_mode=init_mode, predict_mode=predict_mode, seed=seed
            )


# ============================================================
# Single condition runner with full posterior storage
# ============================================================
def run_pair_full(a1_type, a2_type, n_steps, alpha,
                  h_length=50, h_num=10,
                  init_mode='random', predict_mode='sample',
                  seed=42):
    """Run one pair, store all key trajectories."""
    seed1 = seed
    seed2 = seed + 100000
    ag1 = AgentRewardSharp(a1_type, h_length, h_num, init_mode, predict_mode,
                            alpha=alpha, seed=seed1)
    ag2 = AgentRewardSharp(a2_type, h_length, h_num, init_mode, predict_mode,
                            alpha=alpha, seed=seed2)

    h1_arr = np.empty(n_steps, dtype='<U1')
    h2_arr = np.empty(n_steps, dtype='<U1')
    res_arr = np.empty(n_steps, dtype='<U10')
    am1_arr = np.full(n_steps, -1, dtype=np.int32)
    am2_arr = np.full(n_steps, -1, dtype=np.int32)
    P1 = np.zeros((n_steps, h_num), dtype=np.float32) if a1_type != 'random' else None
    P2 = np.zeros((n_steps, h_num), dtype=np.float32) if a2_type != 'random' else None

    for t in range(n_steps):
        h1 = ag1.choice()
        h2 = ag2.choice()
        am1_arr[t] = ag1.argmax_h()
        am2_arr[t] = ag2.argmax_h()
        if P1 is not None:
            P1[t] = ag1.bayes.h_prov.copy()
        if P2 is not None:
            P2[t] = ag2.bayes.h_prov.copy()
        h1_arr[t] = h1
        h2_arr[t] = h2
        res_arr[t] = rps(h1, h2)
        ag1.update_from_outcome(h1, h2)
        ag2.update_from_outcome(h2, h1)

    return {
        'h1': h1_arr, 'h2': h2_arr, 'res': res_arr,
        'argmax1': am1_arr, 'argmax2': am2_arr,
        'P1': P1, 'P2': P2,
    }


def run_one_condition(args):
    """Worker: one (design, alpha, pair, run_idx) condition."""
    (design, alpha, pair, run_idx, n_steps, h_length, h_num, base_seed) = args
    init_mode = {'r': 'random', 's': 'structured'}[design[0]]
    predict_mode = {'s': 'sample', 'a': 'argmax'}[design[1]]
    a1, a2 = pair.split('-')
    seed = base_seed + run_idx * 1000 + int(alpha * 1000)
    result = run_pair_full(a1, a2, n_steps, alpha, h_length, h_num,
                            init_mode, predict_mode, seed=seed)
    return (design, alpha, pair, run_idx, result)


# ============================================================
# Main sweep
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=13)
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--n_runs', type=int, default=20)
    parser.add_argument('--h_length', type=int, default=50)
    parser.add_argument('--h_num', type=int, default=10)
    args = parser.parse_args()

    designs = ['rs', 'ra', 'ss', 'sa']
    alphas = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pairs = ['bib-bib', 'bo-bo']

    print(f'Sharpness + plateau sweep')
    print(f'  designs:  {designs}')
    print(f'  alphas:   {alphas}')
    print(f'  pairs:    {pairs}')
    print(f'  n_steps:  {args.n_steps}')
    print(f'  n_runs:   {args.n_runs}')
    print(f'  Total: {len(designs)} × {len(alphas)} × {len(pairs)} × {args.n_runs} '
          f'= {len(designs)*len(alphas)*len(pairs)*args.n_runs} sub-runs')
    print()

    # Build job list
    jobs = []
    for design in designs:
        for alpha in alphas:
            for pair in pairs:
                for run_idx in range(args.n_runs):
                    jobs.append((design, alpha, pair, run_idx,
                                 args.n_steps, args.h_length, args.h_num, 42))

    t0 = time.time()
    # Group by (design, alpha, pair) for output structure
    results_by_cond = {}
    print(f'Running {len(jobs)} sub-runs with {args.workers} workers...')
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one_condition, jobs)):
            design, alpha, pair, run_idx, data = res
            key = (design, alpha, pair)
            if key not in results_by_cond:
                results_by_cond[key] = []
            results_by_cond[key].append((run_idx, data))
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed * (len(jobs) - i - 1) / (i + 1)
                print(f'  done {i+1}/{len(jobs)}  '
                      f'elapsed {elapsed:.0f}s  ETA {eta:.0f}s')

    elapsed = time.time() - t0
    print(f'\nAll runs complete in {elapsed:.0f}s')
    print()

    # Save per condition
    print('Saving aggregated data per condition...')
    for (design, alpha, pair), runs in results_by_cond.items():
        runs.sort(key=lambda x: x[0])
        # Stack arrays: each result has shape (n_steps,) or (n_steps, h_num)
        argmax1 = np.stack([r[1]['argmax1'] for r in runs], axis=0)  # (n_runs, n_steps)
        argmax2 = np.stack([r[1]['argmax2'] for r in runs], axis=0)
        # P1, P2 may be None if random agent
        if runs[0][1]['P1'] is not None:
            P1 = np.stack([r[1]['P1'] for r in runs], axis=0)  # (n_runs, n_steps, h_num)
        else:
            P1 = None
        if runs[0][1]['P2'] is not None:
            P2 = np.stack([r[1]['P2'] for r in runs], axis=0)
        else:
            P2 = None

        # Don't save raw hand/result arrays here to keep size down;
        # those are not needed for plateau / sharpness analysis
        out_path = OUT_DIR / f'{design}_a{int(alpha*100):02d}_{pair}.npz'
        save_dict = {'argmax1': argmax1, 'argmax2': argmax2}
        if P1 is not None:
            save_dict['P1'] = P1
        if P2 is not None:
            save_dict['P2'] = P2
        np.savez_compressed(out_path, **save_dict)
        print(f'  saved {out_path.name}: '
              f'argmax shape {argmax1.shape}, '
              f'P1 shape {P1.shape if P1 is not None else "None"}')

    print(f'\nAll saved to {OUT_DIR}')
    print()
    print('Next step:')
    print('  python3 analyze_sharpness_plateau.py')


if __name__ == '__main__':
    main()
