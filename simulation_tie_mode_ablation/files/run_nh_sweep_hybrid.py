"""
N_h sweep under the hybrid observation scheme (tie=uniform, defeat=opponent),
medium scale (T=10000 x 200 runs).

Purpose: verify Task 4 from HANDOFF — that R5 (alpha invariance in core
central regime N_h in {6,10}) and R6 (sigma ~ N_h^{-beta} with beta
invariant across 4 designs) survive under the hybrid observation rule.

Output: data/scheme_ablation/medium_nh_hybrid/<cell>.json per condition.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rpsgame_reward_tie as sim

OUT_BASE = HERE.parent / 'data' / 'scheme_ablation'

SCHEME_HYBRID = ('uniform', 'opponent')  # tie_mode, defeat_mode
DESIGNS = [('random', 'sample'), ('random', 'argmax'),
           ('structured', 'sample'), ('structured', 'argmax')]
DESIGN_TAG = {('random', 'sample'): 'rs', ('random', 'argmax'): 'ra',
              ('structured', 'sample'): 'ss', ('structured', 'argmax'): 'sa'}

NH_VALUES = [3, 6, 10, 15, 20]

SCALE_CONFIGS = {
    'small':  dict(n_steps=2000,   analysis_start=1000,   n_runs=100),
    'medium': dict(n_steps=10000,  analysis_start=5000,   n_runs=200),
    'huge':   dict(n_steps=200000, analysis_start=100000, n_runs=20),
}


def run_one(out_dir, h_num, init_mode, predict_mode, cfg, n_workers,
            seed_base=0):
    tie_mode, defeat_mode = SCHEME_HYBRID
    design = DESIGN_TAG[(init_mode, predict_mode)]
    cond = f"{design}_h{h_num:02d}_hybrid_tie{tie_mode}_def{defeat_mode}"
    dpath = out_dir / f'durations_{cond}.json'
    spath = out_dir / f'sigmas_{cond}.json'

    if dpath.exists() and spath.exists():
        try:
            agg = json.load(open(dpath))
            sstats = json.load(open(spath))
            return agg, sstats, 0.0, True
        except Exception:
            pass

    t0 = time.time()
    agg, rstats, sstats = sim.parallel_runs(
        'bib', 'bib', n_steps=cfg['n_steps'], n_runs=cfg['n_runs'],
        analysis_start=cfg['analysis_start'],
        h_length=50, h_num=h_num,
        init_mode=init_mode, predict_mode=predict_mode,
        tie_mode=tie_mode, defeat_mode=defeat_mode,
        n_workers=n_workers, seed_base=seed_base,
    )
    elapsed = time.time() - t0
    json.dump({k: list(v) for k, v in agg.items()}, open(dpath, 'w'))
    json.dump(sstats, open(spath, 'w'))
    return agg, sstats, elapsed, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale', choices=list(SCALE_CONFIGS), default='medium')
    p.add_argument('--workers', type=int, default=sim.DEFAULT_N_WORKERS)
    p.add_argument('--nh-values', nargs='*', type=int, default=NH_VALUES)
    p.add_argument('--designs', nargs='*',
                   choices=['rs', 'ra', 'ss', 'sa'],
                   default=['rs', 'ra', 'ss', 'sa'])
    args = p.parse_args()

    cfg = SCALE_CONFIGS[args.scale]
    out_dir = OUT_BASE / f'{args.scale}_nh_hybrid'
    out_dir.mkdir(parents=True, exist_ok=True)

    sel_designs = [d for d in DESIGNS if DESIGN_TAG[d] in args.designs]
    total = len(sel_designs) * len(args.nh_values)
    print(f"Hybrid N_h sweep: scale={args.scale}, n_h={args.nh_values}, "
          f"designs={[DESIGN_TAG[d] for d in sel_designs]}, total={total}")
    print(f"  Output: {out_dir.resolve()}")
    print(f"  Workers: {args.workers}")

    idx = 0
    t0_grid = time.time()
    for h_num in args.nh_values:
        for (init_mode, predict_mode) in sel_designs:
            idx += 1
            tag = DESIGN_TAG[(init_mode, predict_mode)]
            agg, sstats, el, hit = run_one(
                out_dir, h_num, init_mode, predict_mode,
                cfg, args.workers)
            status = "cache" if hit else f"{el:.0f}s"
            print(f"  [{idx}/{total}] h_num={h_num:2d} {tag}: {status}")
    elapsed = time.time() - t0_grid
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == '__main__':
    main()
