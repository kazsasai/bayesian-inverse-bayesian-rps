"""Bootstrap CI for the powerlaw alpha of T_argmax1 distributions.

For each huge BIB-BIB cell (3 schemes x 4 designs = 12 cells), draw B
bootstrap resamples with replacement and fit truncated power law; report
alpha median, IQR, and 2.5/97.5 percentiles.

Resumable per-cell. Runs at most --max-cells cells per invocation to fit
inside short timeouts; cached to _bootstrap_cache.json.

Usage:
    python3 bootstrap_alpha.py --scale huge --B 200 --workers 7
"""
import argparse, json, sys, time, warnings
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import powerlaw

SCHEMES = {
    'eq3':      ('skip',    'random_other'),
    'caseA':    ('uniform', 'random_other'),
    'hybrid':   ('uniform', 'opponent'),
    'defeatGT': ('skip',    'opponent'),
}
DESIGNS = ['rs', 'ra', 'ss', 'sa']


def fit_one(data):
    """Fit truncated power law to one bootstrap resample.
    Returns (alpha, xmin) or (nan, nan)."""
    data = np.asarray(data)
    data = data[data >= 1]
    if data.size < 30 or np.unique(data).size < 5:
        return float('nan'), float('nan')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = powerlaw.Fit(data, discrete=True, verbose=False)
            tpl = fit.truncated_power_law
            return float(tpl.alpha), float(tpl.xmin)
    except Exception:
        return float('nan'), float('nan')


def _boot_worker(packed):
    seed, data = packed
    rng = np.random.default_rng(seed)
    resample = rng.choice(data, size=data.size, replace=True)
    return fit_one(resample)


def bootstrap_cell(data, B, workers, base_seed):
    args = [(base_seed + i, data) for i in range(B)]
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(_boot_worker, args)
    else:
        results = [_boot_worker(a) for a in args]
    alphas = np.array([r[0] for r in results], dtype=float)
    xmins = np.array([r[1] for r in results], dtype=float)
    valid = ~np.isnan(alphas)
    a = alphas[valid]
    if a.size == 0:
        return None
    return {
        'B': int(B),
        'B_valid': int(a.size),
        'alpha_median': float(np.median(a)),
        'alpha_mean':   float(np.mean(a)),
        'alpha_std':    float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        'alpha_p025':   float(np.percentile(a, 2.5)),
        'alpha_p25':    float(np.percentile(a, 25)),
        'alpha_p75':    float(np.percentile(a, 75)),
        'alpha_p975':   float(np.percentile(a, 97.5)),
        'xmin_median':  float(np.median(xmins[valid])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale', default='huge')
    p.add_argument('--B', type=int, default=200,
                   help='bootstrap resamples per cell')
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--max-cells', type=int, default=12,
                   help='max cells per invocation')
    p.add_argument('--schemes', nargs='+',
                   default=['eq3', 'caseA', 'hybrid'])
    p.add_argument('--pairs', nargs='+', default=['bib-bib'])
    p.add_argument('--seed-base', type=int, default=12345)
    args = p.parse_args()

    DATA = HERE.parent / 'data' / 'scheme_ablation' / args.scale
    CACHE = DATA / '_bootstrap_cache.json'
    cache = {}
    if CACHE.exists():
        try:
            cache = json.load(open(CACHE))
        except Exception:
            cache = {}

    n_done = 0
    for design in DESIGNS:
        for pair in args.pairs:
            for scheme in args.schemes:
                tie, defeat = SCHEMES[scheme]
                cond = (f'{design}_{pair}_{scheme}'
                        f'_tie{tie}_def{defeat}')
                key = f'B{args.B}_{cond}'
                if key in cache:
                    continue
                dpath = DATA / f'durations_{cond}.json'
                if not dpath.exists():
                    continue
                with open(dpath) as f:
                    agg = json.load(f)
                data = np.asarray(agg['T_argmax1'], dtype=int)
                if data.size < 30:
                    cache[key] = {'error': 'too_few', 'n': int(data.size)}
                    continue
                t0 = time.time()
                print(f'  bootstrap {cond} (n={data.size}, B={args.B})',
                      flush=True)
                res = bootstrap_cell(data, args.B, args.workers,
                                     args.seed_base + n_done * args.B)
                if res is None:
                    cache[key] = {'error': 'all_nan', 'n': int(data.size)}
                else:
                    res['n'] = int(data.size)
                    cache[key] = res
                    print(f'    alpha = {res["alpha_median"]:.4f} '
                          f'[{res["alpha_p025"]:.4f}, '
                          f'{res["alpha_p975"]:.4f}] '
                          f'({time.time()-t0:.0f}s)', flush=True)
                with open(CACHE, 'w') as f:
                    json.dump(cache, f, indent=2)
                n_done += 1
                if n_done >= args.max_cells:
                    print(f'[batch done] {n_done} cells, '
                          f'{len(cache)} total cached')
                    return
    print(f'[all done] {n_done} new cells, {len(cache)} total cached')


if __name__ == '__main__':
    main()
