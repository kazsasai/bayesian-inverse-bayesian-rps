"""Fit T_argmax1 distributions for all cached scheme cells, save to _fit_cache.json.
Resumable: cells already in cache are skipped. Runs ONE batch of `--n` fits and exits."""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rpsgame_reward_tie as sim

p = argparse.ArgumentParser()
p.add_argument('--scale', default='small')
p.add_argument('--n', type=int, default=4, help='max fits this call')
args = p.parse_args()

DATA = HERE.parent / 'data' / 'scheme_ablation' / args.scale
FIT_CACHE = DATA / '_fit_cache.json'

SCHEMES = {'eq3':('skip','random_other'), 'caseA':('uniform','random_other'),
           'hybrid':('uniform','opponent'), 'defeatGT':('skip','opponent')}
DESIGNS = ['rs','ra','ss','sa']
PAIRS = ['bib-bib', 'bo-bo']

cache = {}
if FIT_CACHE.exists():
    try: cache = json.load(open(FIT_CACHE))
    except Exception: cache = {}

n_done = 0
for design in DESIGNS:
    for pair in PAIRS:
        for scheme, (tie, defeat) in SCHEMES.items():
            cond = f'{design}_{pair}_{scheme}_tie{tie}_def{defeat}'
            if cond in cache:
                continue
            dpath = DATA / f'durations_{cond}.json'
            if not dpath.exists():
                continue
            with open(dpath) as f: agg = json.load(f)
            print(f'  fitting {cond} (n={len(agg["T_argmax1"])}) ...', flush=True)
            fit = sim.fit_distributions(agg['T_argmax1'])
            # Convert numpy bool to python bool for JSON
            fit = {k: (bool(v) if isinstance(v, (np.bool_,)) else v)
                   for k, v in fit.items()}
            cache[cond] = fit
            with open(FIT_CACHE, 'w') as f:
                json.dump(cache, f, default=str)
            n_done += 1
            if n_done >= args.n:
                print(f'[batch done] {n_done} new fits, {len(cache)} total cached')
                sys.exit(0)
print(f'[all done] 0 new fits needed, {len(cache)} total cached')
