"""
Tie-mode ablation runner
========================

Compare the published Eq. (3) "skip on tie" rule against the case-A
"uniform random observation on tie" rule across the 4 designs of paper A.

For each (init_mode, predict_mode, tie_mode) cell, runs the standard
6-pair sweep (BIB-BIB, BO-BO, random-random, BIB-BO, BIB-random,
BO-random) at the requested scale (default: small) and emits:
  - per-condition duration JSONs (with tie_mode in filename)
  - per-condition reward/sigma JSONs
  - summary CSV `summary_ablation_<scale>.csv` with all rows
  - CCDF PNGs for the headline T_argmax1 of each condition
  - a side-by-side BIB-BIB CCDF comparison plot per design

Usage
-----
    # quick pilot (T=2000 x 100 runs, ~few minutes)
    python run_tie_mode_ablation.py --scale small

    # paper-grade comparison (T=10000 x 200 runs, ~hours)
    python run_tie_mode_ablation.py --scale medium

    # full publication-grade (T=200000 x 20 runs, ~half day)
    python run_tie_mode_ablation.py --scale huge

Reversion
---------
The entire experiment lives inside simulation_tie_mode_ablation/.
Delete that directory to revert. The published simulator at
simulation_r4_fallback/files/rpsgame_reward.py is never touched.
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

# Use the local ablation variant; do NOT touch the original simulator
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rpsgame_reward_tie as sim


SCALE_CONFIGS = {
    'small':  dict(n_steps=2000,   analysis_start=1000,   n_runs=100),
    'medium': dict(n_steps=10000,  analysis_start=5000,   n_runs=200),
    'large':  dict(n_steps=50000,  analysis_start=25000,  n_runs=50),
    'huge':   dict(n_steps=200000, analysis_start=100000, n_runs=20),
}

DESIGNS = [
    ('random',     'sample'),   # rs
    ('random',     'argmax'),   # ra
    ('structured', 'sample'),   # ss
    ('structured', 'argmax'),   # sa
]
DESIGN_TAG = {
    ('random',     'sample'):    'rs',
    ('random',     'argmax'):    'ra',
    ('structured', 'sample'):    'ss',
    ('structured', 'argmax'):    'sa',
}

PAIRS = [('bib', 'bib'), ('bo', 'bo'), ('random', 'random'),
         ('bib', 'bo'),  ('bib', 'random'), ('bo', 'random')]
TIE_MODES = ['skip', 'uniform']
DEFEAT_MODES = ['random_other', 'opponent']

# Named schemes built from (tie_mode, defeat_mode):
#   eq3       = (skip,    random_other)   - published Eq. (3)
#   caseA     = (uniform, random_other)   - tie-noise only
#   hybrid    = (uniform, opponent)       - tie-noise + ground-truth defeat
#   defeatGT  = (skip,    opponent)       - defeat ground-truth only
SCHEMES = {
    'eq3':       ('skip',    'random_other'),
    'caseA':     ('uniform', 'random_other'),
    'hybrid':    ('uniform', 'opponent'),
    'defeatGT':  ('skip',    'opponent'),
}


def run_one_cell(out_dir, a1, a2, init_mode, predict_mode, tie_mode,
                 cfg, n_workers, seed_base):
    """Run a single (pair, design, tie_mode) cell and persist outputs."""
    design_tag = DESIGN_TAG[(init_mode, predict_mode)]
    cond_id = f"{design_tag}_{a1}-{a2}_tie{tie_mode}"
    durations_path = out_dir / f'durations_{cond_id}.json'
    rewards_path = out_dir / f'rewards_{cond_id}.json'
    sigmas_path = out_dir / f'sigmas_{cond_id}.json'

    cache_hit = (durations_path.exists()
                 and rewards_path.exists() and sigmas_path.exists())

    if cache_hit:
        try:
            with open(durations_path) as f:
                agg = json.load(f)
            with open(rewards_path) as f:
                reward_stats = json.load(f)
            with open(sigmas_path) as f:
                sigma_stats = json.load(f)
            return agg, reward_stats, sigma_stats, 0.0, True
        except Exception:
            pass  # fall through to re-run

    t0 = time.time()
    agg, reward_stats, sigma_stats = sim.parallel_runs(
        a1, a2,
        n_steps=cfg['n_steps'],
        n_runs=cfg['n_runs'],
        analysis_start=cfg['analysis_start'],
        h_length=50, h_num=10,
        init_mode=init_mode, predict_mode=predict_mode,
        tie_mode=tie_mode,
        n_workers=n_workers, seed_base=seed_base,
    )
    elapsed = time.time() - t0

    # Persist immediately
    with open(durations_path, 'w') as f:
        json.dump({k: list(v) for k, v in agg.items()}, f)
    with open(rewards_path, 'w') as f:
        json.dump(reward_stats, f)
    with open(sigmas_path, 'w') as f:
        json.dump(sigma_stats, f)

    return agg, reward_stats, sigma_stats, elapsed, False


def build_summary_row(a1, a2, init_mode, predict_mode, tie_mode,
                      cfg, agg, reward_stats, sigma_stats):
    row = {
        'a1': a1, 'a2': a2,
        'design': DESIGN_TAG[(init_mode, predict_mode)],
        'init_mode': init_mode, 'predict_mode': predict_mode,
        'tie_mode': tie_mode,
        'n_steps': cfg['n_steps'], 'n_runs': cfg['n_runs'],
    }
    # Reward + sigma summary
    if reward_stats:
        wrs = [r['win_count'] / r['n_post']
               for r in reward_stats if r['n_post'] > 0]
        qrs = [r['quits_count'] / r['n_post']
               for r in reward_stats if r['n_post'] > 0]
        crs = [r['cumR_final'] / r['n_post']
               for r in reward_stats if r['n_post'] > 0]
        row['win_rate_mean'] = float(np.mean(wrs)) if wrs else np.nan
        row['win_rate_std'] = float(np.std(wrs)) if wrs else np.nan
        row['quits_rate_mean'] = float(np.mean(qrs)) if qrs else np.nan
        row['cumR_per_step_mean'] = float(np.mean(crs)) if crs else np.nan
    if sigma_stats:
        row['sig1_mean'] = float(np.mean([s['sig1_mean']
                                          for s in sigma_stats]))
        row['sig2_mean'] = float(np.mean([s['sig2_mean']
                                          for s in sigma_stats]))

    # Power-law fits on all duration metrics
    metric_keys = ['T_argmax1', 'T_argmax2', 'hand1_runs', 'hand2_runs',
                   'result_runs', 'match_runs',
                   'win_runs', 'win_or_draw_runs', 'defeat_runs']
    for k in metric_keys:
        try:
            fit = sim.fit_distributions(agg[k])
        except Exception as e:
            fit = {'error': f'fit_exception: {e}'}
        if 'error' not in fit:
            row[f'{k}_n'] = fit['n']
            row[f'{k}_alpha'] = fit['alpha_best']
            row[f'{k}_best'] = fit['best']
            row[f'{k}_w_pl'] = fit['w_pl']
            row[f'{k}_w_tpl'] = fit['w_tpl']
            row[f'{k}_w_exp'] = fit['w_exp']
            row[f'{k}_levy'] = fit['levy_region']
        else:
            row[f'{k}_n'] = fit.get('n', 0)
            row[f'{k}_alpha'] = np.nan
            row[f'{k}_best'] = 'error'
            row[f'{k}_w_pl'] = np.nan
            row[f'{k}_w_tpl'] = np.nan
            row[f'{k}_w_exp'] = np.nan
            row[f'{k}_levy'] = False
    return row


def plot_bib_comparison(out_dir, agg_by_tie, design_tag):
    """CCDF of BIB-BIB T_argmax1 overlaid for whichever tie_modes are present."""
    import powerlaw
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {'skip': 'navy', 'uniform': 'crimson'}
    plotted = 0
    for tie_mode in TIE_MODES:
        if tie_mode not in agg_by_tie:
            continue
        durations = np.asarray(agg_by_tie[tie_mode].get('T_argmax1', []))
        durations = durations[durations >= 1]
        if len(durations) < 30 or len(np.unique(durations)) < 5:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = powerlaw.Fit(durations, discrete=True, verbose=False)
            fit.plot_ccdf(ax=ax, color=colors[tie_mode], marker='o',
                          markersize=3, linestyle='None',
                          label=f'tie={tie_mode} (α={fit.power_law.alpha:.2f})')
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel('Duration T_argmax1')
    ax.set_ylabel('P(X >= T)')
    ax.set_title(f'BIB-BIB T_argmax1 CCDF — design {design_tag}')
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f'compare_BIB_{design_tag}.png', dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale',
                        choices=list(SCALE_CONFIGS), default='small')
    parser.add_argument('--workers', type=int,
                        default=sim.DEFAULT_N_WORKERS)
    parser.add_argument('--output',
                        default=str(HERE.parent / 'data' / 'ablation'))
    parser.add_argument('--designs', nargs='*',
                        choices=['rs', 'ra', 'ss', 'sa'],
                        default=['rs', 'ra', 'ss', 'sa'])
    parser.add_argument('--tie-modes', dest='tie_modes', nargs='*',
                        choices=['skip', 'uniform'],
                        default=['skip', 'uniform'])
    parser.add_argument('--pairs', nargs='*', default=None,
                        help='Subset of pairs to run, e.g. "bib-bib" "bo-bo"')
    args = parser.parse_args()

    cfg = SCALE_CONFIGS[args.scale]
    out_dir = Path(args.output) / args.scale
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_designs = [d for d in DESIGNS
                        if DESIGN_TAG[d] in args.designs]
    selected_tie_modes = args.tie_modes

    if args.pairs:
        selected_pairs = []
        for p in args.pairs:
            try:
                a, b = p.split('-')
                selected_pairs.append((a, b))
            except ValueError:
                raise SystemExit(f"Bad --pairs entry: {p!r} (use 'a-b' form)")
    else:
        selected_pairs = PAIRS

    total = (len(selected_designs) * len(selected_tie_modes)
             * len(selected_pairs))
    print(f"Tie-mode ablation: scale={args.scale}, "
          f"designs={[DESIGN_TAG[d] for d in selected_designs]}, "
          f"tie_modes={selected_tie_modes}, "
          f"pairs={len(selected_pairs)}, total={total} cells")
    print(f"  T={cfg['n_steps']}, analyse [{cfg['analysis_start']}, "
          f"{cfg['n_steps']}), n_runs={cfg['n_runs']}, "
          f"workers={args.workers}")
    print(f"  Output: {out_dir.resolve()}")

    rows = []
    agg_cache = {}  # for BIB-BIB comparison plots
    grid_t0 = time.time()
    idx = 0
    for (init_mode, predict_mode) in selected_designs:
        design_tag = DESIGN_TAG[(init_mode, predict_mode)]
        for tie_mode in selected_tie_modes:
            for (a1, a2) in selected_pairs:
                idx += 1
                agg, rstats, sstats, elapsed, hit = run_one_cell(
                    out_dir, a1, a2, init_mode, predict_mode, tie_mode,
                    cfg, args.workers, seed_base=0)
                tag = (f"{design_tag} {a1}-{a2} tie={tie_mode}")
                tail = '(cache)' if hit else f'({elapsed:.1f}s)'
                print(f"[{idx}/{total}] {tag}  {tail}")

                row = build_summary_row(
                    a1, a2, init_mode, predict_mode, tie_mode,
                    cfg, agg, rstats, sstats)
                rows.append(row)

                if (a1, a2) == ('bib', 'bib'):
                    agg_cache.setdefault(design_tag, {})[tie_mode] = agg

                # Save partial summary after every cell (crash recovery)
                pd.DataFrame(rows).to_csv(
                    out_dir / f'summary_ablation_{args.scale}_partial.csv',
                    index=False,
                )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / f'summary_ablation_{args.scale}.csv'
    summary.to_csv(summary_path, index=False)

    # Generate BIB-BIB comparison plots (only if any data present)
    for design_tag, agg_by_tie in agg_cache.items():
        if agg_by_tie:
            plot_bib_comparison(out_dir, agg_by_tie, design_tag)

    print(f"\nAblation done in {(time.time() - grid_t0)/60:.1f} min")
    print(f"Summary: {summary_path}")

    # Brief table for BIB-BIB only
    bibbib = summary[(summary['a1'] == 'bib') & (summary['a2'] == 'bib')]
    cols = ['design', 'tie_mode',
            'T_argmax1_alpha', 'T_argmax1_best',
            'T_argmax1_w_pl', 'T_argmax1_w_tpl', 'T_argmax1_w_exp',
            'win_rate_mean', 'quits_rate_mean', 'sig1_mean']
    cols = [c for c in cols if c in bibbib.columns]
    print("\n=== BIB-BIB headline summary ===")
    print(bibbib[cols].to_string(index=False))


if __name__ == '__main__':
    main()
