"""Extract laminar phases from pmax trajectories (hybrid scheme) and fit
power law per design. Compare to published R7 alpha^lam = 1.34 (eq3)."""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import powerlaw

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data' / 'scheme_ablation' / 'huge_laminar'
DESIGNS = ['rs', 'ra', 'ss', 'sa']


def laminar_lengths(pmax_track, theta):
    is_laminar = pmax_track > theta
    if is_laminar.size == 0:
        return np.array([], dtype=int)
    diffs = np.diff(is_laminar.astype(np.int8))
    boundaries = np.concatenate(
        [[0], np.flatnonzero(diffs) + 1, [is_laminar.size]])
    run_lengths = np.diff(boundaries)
    states = is_laminar[boundaries[:-1]]
    return run_lengths[states]


def fit_powerlaw(data, xmin=None):
    if len(data) < 50 or len(np.unique(data)) < 5:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if xmin is not None:
            fit = powerlaw.Fit(data, discrete=True, xmin=xmin,
                               verbose=False)
        else:
            fit = powerlaw.Fit(data, discrete=True, verbose=False)
    return {
        'alpha': float(fit.power_law.alpha),
        'xmin':  float(fit.power_law.xmin),
        'n':     int(len(data)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--theta', type=float, default=0.4,
                   help='laminar threshold on max_h P(h)')
    p.add_argument('--xmin', type=float, default=None,
                   help='force xmin (e.g. 1) instead of powerlaw scan')
    args = p.parse_args()

    print(f'Laminar phase analysis (hybrid scheme, theta={args.theta})')
    print(f'{"design":<6} {"n_lam":>10} {"alpha":>8} {"xmin":>6} '
          f'{"pmax_mn":>8}  {"frac_lam":>9}')
    results = {}
    for design in DESIGNS:
        f = DATA / f'pmax_{design}_hybrid.npz'
        if not f.exists():
            print(f'{design:<6} (no data)')
            continue
        d = np.load(f)
        p1 = d['pmax1']  # (n_runs, n_post)
        p2 = d['pmax2']
        pooled = np.concatenate([p1, p2], axis=0)
        all_lengths = []
        frac_lam = 0.0
        for run in pooled:
            ll = laminar_lengths(run, args.theta)
            all_lengths.append(ll)
            frac_lam += (run > args.theta).mean()
        frac_lam /= len(pooled)
        lengths = (np.concatenate(all_lengths) if all_lengths
                   else np.array([], dtype=int))
        fit = fit_powerlaw(lengths, xmin=args.xmin) or {'alpha': float('nan'),
                                        'xmin': float('nan'),
                                        'n': int(lengths.size)}
        results[design] = {**fit, 'pmax_mean': float(pooled.mean()),
                            'frac_laminar': float(frac_lam)}
        print(f'{design:<6} {fit["n"]:>10d} {fit["alpha"]:>8.3f} '
              f'{fit["xmin"]:>6.0f} {pooled.mean():>8.3f} '
              f'{frac_lam:>9.3f}')

    valid = [r['alpha'] for r in results.values()
             if not (r['alpha'] != r['alpha'])]  # NaN check
    if valid:
        print(f'\nCross-design (n={len(valid)}): '
              f'mean alpha = {np.mean(valid):.4f}, '
              f'SD = {np.std(valid):.4f}, '
              f'spread = {max(valid)-min(valid):.4f}')
        print(f'Published R7 (eq3, theta=0.4): alpha^lam = 1.340 +/- 0.002')

    tag = f'theta{int(round(args.theta*100)):02d}'
    if args.xmin is not None:
        tag += f'_xmin{int(args.xmin)}'
    out = DATA / f'laminar_summary_{tag}.json'
    json.dump({'theta': args.theta, 'xmin_forced': args.xmin,
               'results': results}, open(out, 'w'), indent=2)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
