"""Fast summary builder reading cached JSONs (no resimulation).
Fits T_argmax1 only for speed. Use for quick comparison plots."""

import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import powerlaw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rpsgame_reward_tie as sim

DATA = HERE.parent / 'data' / 'ablation' / 'small'

DESIGNS = ['rs', 'ra', 'ss', 'sa']
PAIRS = ['bib-bib', 'bo-bo']
TIES = ['skip', 'uniform']

rows = []
for design in DESIGNS:
    for pair in PAIRS:
        for tie in TIES:
            cond = f'{design}_{pair}_tie{tie}'
            dpath = DATA / f'durations_{cond}.json'
            rpath = DATA / f'rewards_{cond}.json'
            spath = DATA / f'sigmas_{cond}.json'
            if not dpath.exists():
                print(f'MISS: {cond}')
                continue
            with open(dpath) as f:
                agg = json.load(f)
            with open(rpath) as f:
                rstats = json.load(f)
            with open(spath) as f:
                sstats = json.load(f)

            wrs = [r['win_count']/r['n_post'] for r in rstats if r['n_post']>0]
            qrs = [r['quits_count']/r['n_post'] for r in rstats if r['n_post']>0]
            sigs = [s['sig1_mean'] for s in sstats]

            fit = sim.fit_distributions(agg['T_argmax1'])
            row = dict(design=design, pair=pair, tie_mode=tie,
                       n=fit.get('n', 0),
                       alpha=fit.get('alpha_best', np.nan),
                       best=fit.get('best', 'error'),
                       w_pl=fit.get('w_pl', np.nan),
                       w_tpl=fit.get('w_tpl', np.nan),
                       w_exp=fit.get('w_exp', np.nan),
                       win_rate=float(np.mean(wrs)) if wrs else np.nan,
                       quits_rate=float(np.mean(qrs)) if qrs else np.nan,
                       sigma_post=float(np.mean(sigs)) if sigs else np.nan)
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(DATA / 'quick_summary.csv', index=False)

# Pretty print
print('=== T_argmax1 across pair × design × tie_mode (small scale) ===')
for pair in PAIRS:
    sub = df[df['pair']==pair]
    print(f'\n--- {pair} ---')
    print(sub[['design','tie_mode','alpha','best','w_tpl','win_rate','quits_rate','sigma_post']]
          .to_string(index=False, float_format='%.4f'))

# Δα table per pair (uniform - skip)
print('\n=== Δα(uniform - skip) ===')
for pair in PAIRS:
    sub = df[df['pair']==pair]
    pivot = sub.pivot(index='design', columns='tie_mode', values='alpha')
    pivot['delta_alpha'] = pivot['uniform'] - pivot['skip']
    pivot = pivot.reindex(DESIGNS)
    print(f'\n[{pair}]')
    print(pivot.to_string(float_format='%.4f'))
