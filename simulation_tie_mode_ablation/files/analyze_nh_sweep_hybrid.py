"""
Analyze N_h sweep results under hybrid scheme.
Computes:
  R5 test: alpha(T_argmax1) as a function of N_h, per design.
           Published BIB: alpha invariant at 1.433 +/- 0.014 in {6, 10}.
  R6 test: sigma(P(h)) mean ~ N_h^{-beta} fit per design.
           Published BIB: beta = 1.067 +/- 0.008 across designs.
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rpsgame_reward_tie as sim  # use the same fit_distributions as published

DATA_BASE = HERE.parent / 'data' / 'scheme_ablation'
DESIGNS = ['rs', 'ra', 'ss', 'sa']
NH_VALUES = [3, 6, 10, 15, 20]
SCHEME = 'hybrid_tieuniform_defopponent'


def fit_alpha(durations):
    """Use the same fit_distributions wrapper as run_scheme_ablation.py,
    returning alpha_best (best of PL / TPL / exp) for apples-to-apples
    comparison with published R5/R6 numbers."""
    arr = np.asarray(durations)
    arr = arr[arr >= 1].astype(int).tolist()
    if len(arr) < 50:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = sim.fit_distributions(arr)
    if 'error' in fit:
        return None
    return {
        'alpha': float(fit['alpha_best']),
        'best':  fit['best'],
        'n':     int(fit['n']),
    }


def load_cell(h_num, design, data_dir):
    cond = f'{design}_h{h_num:02d}_{SCHEME}'
    dpath = data_dir / f'durations_{cond}.json'
    spath = data_dir / f'sigmas_{cond}.json'
    if not dpath.exists() or not spath.exists():
        return None
    agg = json.load(open(dpath))
    sstats = json.load(open(spath))
    return agg, sstats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale', choices=['medium', 'huge'], default='medium')
    p.add_argument('--core-nh', nargs='*', type=int, default=[6, 10],
                   help='core central regime values (for R5 invariance check)')
    args = p.parse_args()
    data_dir = DATA_BASE / f'{args.scale}_nh_hybrid'
    print(f'Scale: {args.scale}   data: {data_dir.name}')

    # ----- R5: alpha(N_h, design) -----
    print('=' * 60)
    print('R5 test: alpha(T_argmax1) by N_h x design (hybrid)')
    print('=' * 60)
    alpha_table = {}  # (h_num, design) -> alpha
    sigma_table = {}  # (h_num, design) -> sigma_mean
    header = f'{"N_h":>5}  ' + '  '.join(f'{d:>7}' for d in DESIGNS)
    print(header)
    print('-' * len(header))
    for h in NH_VALUES:
        row = [f'{h:>5}']
        for d in DESIGNS:
            cell = load_cell(h, d, data_dir)
            if cell is None:
                alpha_table[(h, d)] = None
                sigma_table[(h, d)] = None
                row.append('   n/a ')
                continue
            agg, sstats = cell
            fit = fit_alpha(agg.get('T_argmax1', []))
            alpha = fit['alpha'] if fit else None
            best  = fit.get('best', '?') if fit else '?'
            alpha_table[(h, d)] = alpha
            # Sigma: average sig1_mean across runs (sig1 = sig2 since BIB-BIB symmetric pair)
            sig_means = [s['sig1_mean'] for s in sstats if 'sig1_mean' in s]
            sigma_table[(h, d)] = float(np.mean(sig_means)) if sig_means else None
            row.append(f'{alpha:>7.3f}' if alpha is not None else '   nan ')
        print('  '.join(row))

    # ----- R5 summary: alpha invariance in core regime -----
    print()
    print(f'Core central regime N_h in {args.core_nh}:')
    print(f'{"design":>8}  {"core mean":>10}  {"core SD":>9}')
    core_alphas_all = []
    for d in DESIGNS:
        core_alphas = [alpha_table[(h, d)] for h in args.core_nh
                       if alpha_table.get((h, d)) is not None]
        if core_alphas:
            core_alphas_all.extend(core_alphas)
            print(f'{d:>8}  {np.mean(core_alphas):>10.4f}  '
                  f'{np.std(core_alphas):>9.4f}')
    if core_alphas_all:
        print(f'{"pooled":>8}  {np.mean(core_alphas_all):>10.4f}  '
              f'{np.std(core_alphas_all):>9.4f}  '
              f'(n={len(core_alphas_all)})')
        print(f'Published (eq3): pooled mean = 1.433, SD = 0.014, n=8')

    # ----- R6: sigma ~ N_h^{-beta} per design -----
    print()
    print('=' * 60)
    print('R6 test: sigma(P(h)) ~ N_h^{-beta} per design')
    print('=' * 60)
    print(f'{"design":>8}  {"beta":>8}  {"R^2":>6}  {"sigma at N_h=10":>15}')
    betas = {}
    for d in DESIGNS:
        xs = np.array([h for h in NH_VALUES
                       if sigma_table.get((h, d)) is not None])
        ys = np.array([sigma_table[(h, d)] for h in xs])
        if len(xs) < 3:
            print(f'{d:>8}  insufficient data')
            continue
        # log-log fit: log(sigma) = -beta * log(N_h) + c
        logx, logy = np.log(xs), np.log(ys)
        slope, intercept = np.polyfit(logx, logy, 1)
        beta = -float(slope)
        # R^2
        y_pred = slope * logx + intercept
        ss_res = np.sum((logy - y_pred) ** 2)
        ss_tot = np.sum((logy - np.mean(logy)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        sigma_at_10 = sigma_table.get((10, d), float('nan'))
        betas[d] = beta
        print(f'{d:>8}  {beta:>8.3f}  {r2:>6.3f}  {sigma_at_10:>15.4f}')

    if betas:
        bvals = list(betas.values())
        print(f'\nCross-design beta: mean = {np.mean(bvals):.4f}, '
              f'SD = {np.std(bvals):.4f}, '
              f'spread = {max(bvals)-min(bvals):.4f}')
        print(f'Published (eq3, BIB): beta = 1.067 +/- 0.008')

    # Save summary
    summary = {
        'alpha_table': {f'h{h}_{d}': alpha_table[(h, d)]
                        for h in NH_VALUES for d in DESIGNS},
        'sigma_table': {f'h{h}_{d}': sigma_table[(h, d)]
                        for h in NH_VALUES for d in DESIGNS},
        'beta_per_design': betas,
        'core_nh': args.core_nh,
    }
    out = data_dir / 'nh_sweep_summary.json'
    json.dump(summary, open(out, 'w'), indent=2)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
