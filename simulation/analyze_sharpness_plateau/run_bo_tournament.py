"""
run_bo_tournament.py — Cross-design BO tournament + Nash baseline

Tests Sasai-sensei's Nash-distance hypothesis as an internal benchmark:
  Does BO's "best" design correspond to Nash-near (rs) or Nash-far (sa)?
  How does each BO design fare against the Nash mixed strategy (uniform random)?

If BO's best design is rs (Nash-near, like the BIB-vs-BO result), then
the win-rate ordering is universal across inference type → BIB's
"Nash-near advantage" finding (R3 in the paper) is not BIB-specific.

If BO's best design is sa (Nash-far, deterministic), or some other
non-Nash-near design, then BIB's Nash-near advantage is BIB-specific.

The 'random' (Nash) baseline is added as a fifth strategy. By Nash
equilibrium theory, ALL non-trivial RPS strategies achieve win rate
1/3 against the Nash strategy in expectation. Deviation from 1/3 in
any cell involving 'random' is therefore a finite-T statistical
fluctuation, NOT a strategic difference. This serves as:
  (i) sanity check on the BO designs (none should beat random)
  (ii) measure of how far each BO design drifts from Nash
       (via fluctuations or systematic biases)

Strategies (5):
  rs, ra, ss, sa  — BO designs (4)
  random          — uniform Nash mixed strategy

Tournament: 5 × 5 = 25 cells.

Setup:
  T = 200,000 steps per match
  20 independent runs per cell
  Burn-in: discard first T/2 = 100,000 steps
  N_h = 10, m = 50, alpha_init = 0.8 (canonical, BO designs only)

Compute time on M1 with 13 workers: ~45-90 min for full tournament.

Place this file alongside rpsgame_reward.py.

Usage:
  python3 run_bo_tournament.py --workers 13
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rpsgame_reward import BayesReward, AgentReward, rps, HANDS

OUT_DIR = Path('./data/bo_tournament')
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ['rs', 'ra', 'ss', 'sa', 'random']
DESIGN_TO_PARAMS = {
    'rs': ('random', 'sample'),
    'ra': ('random', 'argmax'),
    'ss': ('structured', 'sample'),
    'sa': ('structured', 'argmax'),
    # 'random' handled separately
}


def make_agent(strategy, h_length, h_num, seed):
    """Build either a BO agent (with given design) or a random Nash agent."""
    if strategy == 'random':
        return AgentReward('random', h_length, h_num,
                            init_mode='random', predict_mode='sample',
                            seed=seed)
    init, predict = DESIGN_TO_PARAMS[strategy]
    return AgentReward('bo', h_length, h_num, init, predict, seed=seed)


def run_match(strategy1, strategy2, n_steps, h_length, h_num, seed):
    """Single match: strategy1 vs strategy2, recording win/loss/tie counts
    after burn-in."""
    ag1 = make_agent(strategy1, h_length, h_num, seed=seed)
    ag2 = make_agent(strategy2, h_length, h_num, seed=seed + 100000)

    burn_in = n_steps // 2
    n_wins_1 = 0
    n_wins_2 = 0
    n_ties = 0

    for t in range(n_steps):
        h1 = ag1.choice()
        h2 = ag2.choice()
        outcome = rps(h1, h2)
        ag1.update_from_outcome(h1, h2)
        ag2.update_from_outcome(h2, h1)
        if t >= burn_in:
            if outcome == 'win':
                n_wins_1 += 1
            elif outcome == 'defeat':
                n_wins_2 += 1
            else:  # 'quits' (tie)
                n_ties += 1

    n_recorded = n_steps - burn_in
    return {
        'strategy1': strategy1,
        'strategy2': strategy2,
        'n_recorded': n_recorded,
        'wins_1': n_wins_1,
        'wins_2': n_wins_2,
        'ties': n_ties,
        'win_rate_1': n_wins_1 / n_recorded,
        'win_rate_2': n_wins_2 / n_recorded,
        'tie_rate': n_ties / n_recorded,
    }


def run_one(args):
    (s1, s2, run_idx, n_steps, h_length, h_num, base_seed) = args
    seed = base_seed + run_idx * 1000
    result = run_match(s1, s2, n_steps, h_length, h_num, seed)
    result['run_idx'] = run_idx
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=13)
    parser.add_argument('--n_steps', type=int, default=200_000)
    parser.add_argument('--n_runs', type=int, default=20)
    parser.add_argument('--h_length', type=int, default=50)
    parser.add_argument('--h_num', type=int, default=10)
    args = parser.parse_args()

    cells = [(s1, s2) for s1 in STRATEGIES for s2 in STRATEGIES]

    print(f'BO cross-design tournament + Nash baseline')
    print(f'  strategies: {STRATEGIES}')
    print(f'  cells:    {len(cells)} ({len(STRATEGIES)} × {len(STRATEGIES)})')
    print(f'  n_steps:  {args.n_steps}')
    print(f'  n_runs:   {args.n_runs}')
    print(f'  total runs: {len(cells) * args.n_runs}')
    print()

    jobs = []
    for s1, s2 in cells:
        for run_idx in range(args.n_runs):
            jobs.append((s1, s2, run_idx,
                         args.n_steps, args.h_length, args.h_num, 42))

    t0 = time.time()
    print(f'Running {len(jobs)} matches with {args.workers} workers...')
    all_results = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, jobs)):
            all_results.append(res)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed * (len(jobs) - i - 1) / (i + 1)
                print(f'  done {i+1}/{len(jobs)}  '
                      f'elapsed {elapsed:.0f}s  ETA {eta:.0f}s')

    elapsed = time.time() - t0
    print(f'\nAll matches complete in {elapsed:.0f}s')
    print()

    aggregated = {}
    for s1, s2 in cells:
        cell_results = [r for r in all_results
                         if r['strategy1'] == s1 and r['strategy2'] == s2]
        total_recorded = sum(r['n_recorded'] for r in cell_results)
        total_wins_1 = sum(r['wins_1'] for r in cell_results)
        total_wins_2 = sum(r['wins_2'] for r in cell_results)
        total_ties = sum(r['ties'] for r in cell_results)

        win_rate_1 = total_wins_1 / total_recorded
        win_rate_2 = total_wins_2 / total_recorded
        tie_rate = total_ties / total_recorded

        n = total_recorded
        p_null = 1.0 / 3.0
        se = np.sqrt(p_null * (1 - p_null) / n)
        z = (win_rate_1 - p_null) / se

        per_run_wr1 = [r['win_rate_1'] for r in cell_results]

        aggregated[f'{s1}_vs_{s2}'] = {
            'strategy1': s1, 'strategy2': s2,
            'total_recorded': total_recorded,
            'win_rate_1': win_rate_1,
            'win_rate_2': win_rate_2,
            'tie_rate': tie_rate,
            'z_score_1_vs_chance': z,
            'mean_wr1_per_run': float(np.mean(per_run_wr1)),
            'std_wr1_per_run': float(np.std(per_run_wr1, ddof=1)),
            'n_runs': len(cell_results),
        }

    out_path = OUT_DIR / 'bo_tournament_results.json'
    with open(out_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f'Saved: {out_path}')

    # Print 5×5 win-rate matrix
    print()
    print('=' * 75)
    print('Win-rate matrix (rows = strategy1, cols = strategy2):')
    print('Cell value = strategy1 win rate (against strategy2 in column)')
    print('=' * 75)
    print()
    print(f"{'':>8}", end=' ')
    for s2 in STRATEGIES:
        print(f"{s2:>10}", end=' ')
    print()
    for s1 in STRATEGIES:
        print(f"{s1:>8}", end=' ')
        for s2 in STRATEGIES:
            wr = aggregated[f'{s1}_vs_{s2}']['win_rate_1']
            print(f"{wr:>10.4f}", end=' ')
        print()

    print()
    print('=' * 75)
    print('Z-score matrix vs chance (1/3) for strategy1:')
    print('=' * 75)
    print()
    print(f"{'':>8}", end=' ')
    for s2 in STRATEGIES:
        print(f"{s2:>10}", end=' ')
    print()
    for s1 in STRATEGIES:
        print(f"{s1:>8}", end=' ')
        for s2 in STRATEGIES:
            z = aggregated[f'{s1}_vs_{s2}']['z_score_1_vs_chance']
            print(f"{z:>+10.2f}", end=' ')
        print()

    # Tournament summary excluding random opponent
    print()
    print('=' * 75)
    print('Tournament summary (BO-internal: each BO design vs BO field)')
    print('Excludes the random (Nash) opponent for clean BO ranking')
    print('=' * 75)
    print()
    print(f"{'design':>8} {'mean wr':>10} {'sum z':>10} {'rank':>6}")
    print('-' * 40)
    bo_designs = ['rs', 'ra', 'ss', 'sa']
    bo_summary = []
    for s1 in bo_designs:
        wrs = [aggregated[f'{s1}_vs_{s2}']['win_rate_1'] for s2 in bo_designs]
        zs = [aggregated[f'{s1}_vs_{s2}']['z_score_1_vs_chance'] for s2 in bo_designs]
        mean_wr = np.mean(wrs)
        sum_z = sum(zs)
        bo_summary.append((s1, mean_wr, sum_z))

    bo_summary.sort(key=lambda x: x[1], reverse=True)
    for rank, (d, wr, z) in enumerate(bo_summary, 1):
        print(f"{d:>8} {wr:>10.4f} {z:>+10.2f} {rank:>6}")

    # Vs random (Nash distance check)
    print()
    print('=' * 75)
    print('BO vs Nash random — drift from Nash equilibrium')
    print('Each cell shows the BO design as player 1, random as player 2.')
    print('All cells should be statistically near 1/3 (Nash invariance);')
    print('any systematic deviation reveals the design\'s drift from Nash.')
    print('=' * 75)
    print()
    print(f"{'design':>8} {'wr vs random':>14} {'z':>10} {'note':>20}")
    print('-' * 60)
    for s1 in bo_designs:
        cell = aggregated[f'{s1}_vs_random']
        wr = cell['win_rate_1']
        z = cell['z_score_1_vs_chance']
        note = ('above chance' if z > 2 else
                'below chance' if z < -2 else 'near chance')
        print(f"{s1:>8} {wr:>14.4f} {z:>+10.2f} {note:>20}")

    # Random vs each BO (which BO best exploits random?)
    print()
    print('=' * 75)
    print('Nash random vs BO — which BO design random exploits most/least')
    print('Same data, opposite perspective.')
    print('=' * 75)
    print()
    print(f"{'design':>8} {'random wr':>14} {'z':>10} {'note':>20}")
    print('-' * 60)
    for s2 in bo_designs:
        cell = aggregated[f'random_vs_{s2}']
        wr = cell['win_rate_1']
        z = cell['z_score_1_vs_chance']
        note = ('random > BO' if z > 2 else
                'random < BO' if z < -2 else 'near chance')
        print(f"{s2:>8} {wr:>14.4f} {z:>+10.2f} {note:>20}")


if __name__ == '__main__':
    main()
