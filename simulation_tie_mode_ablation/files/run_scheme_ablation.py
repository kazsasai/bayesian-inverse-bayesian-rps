"""
Scheme-based ablation runner
============================

Sweeps named observation schemes built from (tie_mode, defeat_mode) and
emits a single tidy comparison table.

Schemes:
  eq3       = (tie=skip,    defeat=random_other)  - published Eq. (3)
  caseA     = (tie=uniform, defeat=random_other)  - Sasai-sensei case A
  hybrid    = (tie=uniform, defeat=opponent)      - tie noise + ground-
                                                    truth defeat (new
                                                    suggestion)
  defeatGT  = (tie=skip,    defeat=opponent)      - ground-truth defeat
                                                    only (orthogonal)

Output paths embed both knobs so caches don't collide with the original
tie-only runner (run_tie_mode_ablation.py).

Usage:
    python run_scheme_ablation.py --scale small --pairs bib-bib bo-bo
    python run_scheme_ablation.py --scale medium --schemes eq3 caseA hybrid
"""

import argparse, json, sys, time, warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import powerlaw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rpsgame_reward_tie as sim

SCALE_CONFIGS = {
    'small':  dict(n_steps=2000,   analysis_start=1000,   n_runs=100),
    'medium': dict(n_steps=10000,  analysis_start=5000,   n_runs=200),
    'large':  dict(n_steps=50000,  analysis_start=25000,  n_runs=50),
    'huge':   dict(n_steps=200000, analysis_start=100000, n_runs=20),
}

DESIGN_TAG = {('random','sample'):'rs', ('random','argmax'):'ra',
              ('structured','sample'):'ss', ('structured','argmax'):'sa'}
DESIGNS = list(DESIGN_TAG.keys())

SCHEMES = {
    'eq3':      ('skip',    'random_other'),
    'caseA':    ('uniform', 'random_other'),
    'hybrid':   ('uniform', 'opponent'),
    'defeatGT': ('skip',    'opponent'),
}

PAIRS = [('bib','bib'), ('bo','bo'), ('random','random'),
         ('bib','bo'), ('bib','random'), ('bo','random')]


def run_one(out_dir, scheme, a1, a2, init_mode, predict_mode, cfg,
            n_workers, seed_base=0):
    tie_mode, defeat_mode = SCHEMES[scheme]
    design = DESIGN_TAG[(init_mode, predict_mode)]
    cond = f"{design}_{a1}-{a2}_{scheme}_tie{tie_mode}_def{defeat_mode}"
    dpath = out_dir / f'durations_{cond}.json'
    rpath = out_dir / f'rewards_{cond}.json'
    spath = out_dir / f'sigmas_{cond}.json'

    if dpath.exists() and rpath.exists() and spath.exists():
        try:
            agg = json.load(open(dpath))
            rstats = json.load(open(rpath))
            sstats = json.load(open(spath))
            return agg, rstats, sstats, 0.0, True
        except Exception:
            pass

    t0 = time.time()
    agg, rstats, sstats = sim.parallel_runs(
        a1, a2, n_steps=cfg['n_steps'], n_runs=cfg['n_runs'],
        analysis_start=cfg['analysis_start'],
        h_length=50, h_num=10,
        init_mode=init_mode, predict_mode=predict_mode,
        tie_mode=tie_mode, defeat_mode=defeat_mode,
        n_workers=n_workers, seed_base=seed_base,
    )
    elapsed = time.time() - t0
    json.dump({k: list(v) for k,v in agg.items()}, open(dpath, 'w'))
    json.dump(rstats, open(rpath, 'w'))
    json.dump(sstats, open(spath, 'w'))
    return agg, rstats, sstats, elapsed, False


def build_row(scheme, a1, a2, init_mode, predict_mode, cfg,
              agg, rstats, sstats):
    tie_mode, defeat_mode = SCHEMES[scheme]
    design = DESIGN_TAG[(init_mode, predict_mode)]
    row = dict(scheme=scheme, tie_mode=tie_mode, defeat_mode=defeat_mode,
               design=design, a1=a1, a2=a2,
               n_steps=cfg['n_steps'], n_runs=cfg['n_runs'])

    wrs = [r['win_count']/r['n_post'] for r in rstats if r['n_post']>0]
    qrs = [r['quits_count']/r['n_post'] for r in rstats if r['n_post']>0]
    crs = [r['cumR_final']/r['n_post'] for r in rstats if r['n_post']>0]
    row['win_rate_mean']     = float(np.mean(wrs)) if wrs else np.nan
    row['win_rate_std']      = float(np.std(wrs)) if wrs else np.nan
    row['quits_rate_mean']   = float(np.mean(qrs)) if qrs else np.nan
    row['cumR_per_step_mean']= float(np.mean(crs)) if crs else np.nan
    if sstats:
        row['sig1_mean'] = float(np.mean([s['sig1_mean'] for s in sstats]))
        row['sig2_mean'] = float(np.mean([s['sig2_mean'] for s in sstats]))

    fit = sim.fit_distributions(agg['T_argmax1'])
    if 'error' not in fit:
        row['T_argmax1_n']     = fit['n']
        row['T_argmax1_alpha'] = fit['alpha_best']
        row['T_argmax1_best']  = fit['best']
        row['T_argmax1_w_pl']  = fit['w_pl']
        row['T_argmax1_w_tpl'] = fit['w_tpl']
        row['T_argmax1_w_exp'] = fit['w_exp']
        row['T_argmax1_levy']  = fit['levy_region']
    else:
        for k in ['n','alpha','best','w_pl','w_tpl','w_exp','levy']:
            row[f'T_argmax1_{k}'] = np.nan if k!='best' else 'error'
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale', choices=list(SCALE_CONFIGS), default='small')
    p.add_argument('--workers', type=int, default=sim.DEFAULT_N_WORKERS)
    p.add_argument('--output',
                   default=str(HERE.parent / 'data' / 'scheme_ablation'))
    p.add_argument('--designs', nargs='*',
                   choices=['rs','ra','ss','sa'],
                   default=['rs','ra','ss','sa'])
    p.add_argument('--schemes', nargs='*',
                   choices=list(SCHEMES), default=list(SCHEMES))
    p.add_argument('--pairs', nargs='*', default=['bib-bib', 'bo-bo'])
    args = p.parse_args()

    cfg = SCALE_CONFIGS[args.scale]
    out_dir = Path(args.output) / args.scale
    out_dir.mkdir(parents=True, exist_ok=True)

    sel_designs = [d for d in DESIGNS if DESIGN_TAG[d] in args.designs]
    sel_schemes = args.schemes
    sel_pairs = [tuple(p.split('-')) for p in args.pairs]

    total = len(sel_designs) * len(sel_schemes) * len(sel_pairs)
    print(f"Scheme ablation: scale={args.scale}, schemes={sel_schemes}, "
          f"designs={[DESIGN_TAG[d] for d in sel_designs]}, "
          f"pairs={[f'{a}-{b}' for a,b in sel_pairs]}, total={total}")
    print(f"  Output: {out_dir.resolve()}")

    rows = []
    idx = 0
    t0_grid = time.time()
    for (init_mode, predict_mode) in sel_designs:
        for scheme in sel_schemes:
            for (a1, a2) in sel_pairs:
                idx += 1
                agg, rs, ss, el, hit = run_one(
                    out_dir, scheme, a1, a2, init_mode, predict_mode,
                    cfg, args.workers)
                tag = (f"{DESIGN_TAG[(init_mode,predict_mode)]} "
                       f"{a1}-{a2} {scheme}")
                tail = '(cache)' if hit else f'({el:.1f}s)'
                print(f"[{idx}/{total}] {tag}  {tail}")
                rows.append(build_row(scheme, a1, a2, init_mode,
                                      predict_mode, cfg, agg, rs, ss))
                pd.DataFrame(rows).to_csv(
                    out_dir / f'summary_scheme_{args.scale}_partial.csv',
                    index=False,
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f'summary_scheme_{args.scale}.csv', index=False)
    print(f"\nDone in {(time.time()-t0_grid)/60:.1f} min")

    # Compact print of BIB-BIB only
    bb = df[(df.a1=='bib') & (df.a2=='bib')]
    cols = ['design','scheme','T_argmax1_alpha','T_argmax1_best',
            'T_argmax1_w_tpl','win_rate_mean','quits_rate_mean','sig1_mean']
    cols = [c for c in cols if c in bb.columns]
    print('\n=== BIB-BIB summary ===')
    print(bb[cols].to_string(index=False, float_format='%.4f'))

    bo = df[(df.a1=='bo') & (df.a2=='bo')]
    if not bo.empty:
        print('\n=== BO-BO summary ===')
        print(bo[cols].to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
