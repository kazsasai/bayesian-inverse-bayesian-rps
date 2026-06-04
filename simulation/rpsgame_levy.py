"""
Gunji-type BIB Levy walk verification (1v1, discrete RPS / imitation)
=====================================================================

Tests whether Gunji-type Bayesian-Inverse-Bayesian inference gives rise to
power-law / Levy-walk-like distributions in a minimal 1v1 setting with
3 discrete observations.

Reference targets:
- Gunji et al. 2021 (CSBJ): power-law in N=1000 SPP swarm
- Shinohara et al. 2021 (Symmetry): power-law in 2-agent imitation game
  with continuous outputs and learning/forgetting parameters

Our question: does N=2 + discrete 3-valued obs still produce power laws
with Gunji's specific BIB algorithm?

Metrics tracked:
- T_argmax: duration of argmax P(h)         (Shinohara-style main metric)
- hand_runs: consecutive same-hand runs       (auxiliary)
- result_runs: consecutive same-result runs   (auxiliary, RPS only)
- match_runs: consecutive matched runs        (auxiliary, imitation only)

Usage:
    # Quick pilot (one condition, ~minute)
    python rpsgame_levy.py pilot

    # Full grid (small scale: ~hours)
    python rpsgame_levy.py grid --scale small

    # Full grid (large scale: ~half day)
    python rpsgame_levy.py grid --scale large
"""

import numpy as np
import pandas as pd
import json
import warnings
import os
from pathlib import Path
from itertools import product
from multiprocessing import Pool, cpu_count
import argparse

import powerlaw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)

HANDS = ['r', 'p', 's']

# Default number of worker processes for parallel runs.
# Use CPU count - 1 to keep one core free for the OS.
DEFAULT_N_WORKERS = max(1, cpu_count() - 1)


# ============================================================
# Core: Bayes / BIB with argmax_h tracking
# ============================================================

class Bayes:
    def __init__(self, h_num=10, d_num=3, d_type=None, h_length=10,
                 init_mode='random', predict_mode='sample', seed=None):
        """
        init_mode : 'random' | 'structured'
            'random'     : likelihood from |N(0,1)|, normalized (Gunji 2021 style)
            'structured' : hypotheses placed on a grid in the simplex
                           (Shinohara 2021 analog for discrete obs)
        predict_mode : 'sample' | 'argmax'
            'sample' : sample h ~ P(h), then d ~ P(d|h) (Gunji 2021 style)
            'argmax' : pick h_max = argmax P(h), then d ~ P(d|h_max) (Shinohara 2021 style)
        """
        self.rng = np.random.default_rng(seed)
        if d_type is None:
            d_type = HANDS
        self.h_num = h_num
        self.d_type = np.array(d_type)
        self.init_mode = init_mode
        self.predict_mode = predict_mode

        if init_mode == 'random':
            self.likelihood = np.abs(self.rng.standard_normal((h_num, d_num)))
            self.likelihood /= self.likelihood.sum(axis=1, keepdims=True)
        elif init_mode == 'structured':
            self.likelihood = self._structured_init(h_num, d_num)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        self.h_prov = np.ones(h_num) / h_num
        self.history = self.rng.choice(self.d_type, h_length)

    @staticmethod
    def _structured_init(h_num, d_num):
        """Structured grid in the simplex (analog of Shinohara 2021's equally-spaced
        means in the continuous case). For d_num=3 we provide 10 canonical templates."""
        if d_num != 3:
            raise NotImplementedError(
                f"structured init currently only defined for d_num=3 (got {d_num})")
        # Templates spread across the 3-simplex: center, strong corners,
        # mild corners, anti-corners (suppress one hand).
        templates = np.array([
            [1/3, 1/3, 1/3],   # 0  uniform center
            [0.8, 0.1, 0.1],   # 1  strong r
            [0.1, 0.8, 0.1],   # 2  strong p
            [0.1, 0.1, 0.8],   # 3  strong s
            [0.50, 0.25, 0.25],  # 4  mild r
            [0.25, 0.50, 0.25],  # 5  mild p
            [0.25, 0.25, 0.50],  # 6  mild s
            [0.10, 0.45, 0.45],  # 7  anti-r
            [0.45, 0.10, 0.45],  # 8  anti-p
            [0.45, 0.45, 0.10],  # 9  anti-s
        ])
        if h_num <= len(templates):
            return templates[:h_num].copy()
        # Pad with deterministic perturbations of the templates if h_num > 10
        out = list(templates)
        rng_pad = np.random.default_rng(12345)
        while len(out) < h_num:
            base = templates[len(out) % len(templates)]
            noise = rng_pad.standard_normal(d_num) * 0.05
            v = np.clip(base + noise, 1e-3, None)
            out.append(v / v.sum())
        return np.array(out[:h_num])

    def inference(self, data):
        self.history = np.roll(self.history, -1)
        self.history[-1] = data
        obs_num = np.nonzero(self.d_type == data)[0][0]
        post = self.likelihood.T[obs_num] * self.h_prov
        post /= post.sum()
        self.h_prov = post

    def argmax_h(self):
        """Index of currently most-believed hypothesis (Shinohara-style metric)."""
        return int(np.argmax(self.h_prov))

    def expect(self):
        """Predict an observation. Mode controls whether h is sampled or argmax'd."""
        if self.predict_mode == 'sample':
            like = self.likelihood[self.rng.choice(self.h_num, p=self.h_prov)]
        elif self.predict_mode == 'argmax':
            like = self.likelihood[np.argmax(self.h_prov)]
        else:
            raise ValueError(f"Unknown predict_mode: {self.predict_mode}")
        return self.rng.choice(self.d_type, p=like)

    def inverse(self):
        """BIB step: replace one low-posterior hypothesis with empirical
        observation distribution from the rolling window."""
        new_hypo = np.array([np.sum(self.history == d) for d in self.d_type]) + 1
        prov = (1.0 - self.h_prov) / (self.h_num - 1)
        prov = np.clip(prov, 0, None)
        prov /= prov.sum()
        select = self.rng.choice(self.h_num, p=prov)
        self.likelihood[select] = new_hypo / new_hypo.sum()


class Agent:
    """Two game-modes:
    - 'rps':     output the winning counter to predicted opponent hand
    - 'imitate': output the predicted opponent hand itself (matching)
    """
    COUNTER = {'r': 'p', 'p': 's', 's': 'r'}

    def __init__(self, a_type, game_mode='rps', window_size=10,
                 init_mode='random', predict_mode='sample', seed=None):
        self.type = a_type            # 'random' | 'bayes' | 'bib'
        self.mode = game_mode         # 'rps' | 'imitate'
        self.bayes = Bayes(10, 3, HANDS, window_size,
                           init_mode=init_mode, predict_mode=predict_mode,
                           seed=seed)
        self.rng = np.random.default_rng(seed)

    def choice(self):
        if self.type == 'random':
            return self.rng.choice(HANDS)
        elif self.type in ('bayes', 'bib'):
            predicted = self.bayes.expect()
            return self.COUNTER[predicted] if self.mode == 'rps' else predicted
        else:
            raise ValueError(f"Unknown agent type: {self.type}")

    def learn(self, opp_hand):
        if self.type in ('bayes', 'bib'):
            self.bayes.inference(opp_hand)
            if self.type == 'bib':
                self.bayes.inverse()

    def argmax_h(self):
        return self.bayes.argmax_h() if self.type in ('bayes', 'bib') else -1


def rps(me, rival):
    if me == rival:
        return 'quits'
    wins = {('r', 's'), ('p', 'r'), ('s', 'p')}
    return 'win' if (me, rival) in wins else 'defeat'


# ============================================================
# Pair simulation with full logging
# ============================================================

def run_pair(a_type_1, a_type_2, n_steps, game_mode='rps',
             window_size=10, init_mode='random', predict_mode='sample',
             seed=None):
    """One pair simulation. Returns DataFrame with full step-by-step log."""
    seed1 = seed
    seed2 = (seed + 100000) if seed is not None else None
    ag1 = Agent(a_type_1, game_mode, window_size,
                init_mode=init_mode, predict_mode=predict_mode, seed=seed1)
    ag2 = Agent(a_type_2, game_mode, window_size,
                init_mode=init_mode, predict_mode=predict_mode, seed=seed2)

    n_records = n_steps
    h1_arr = np.empty(n_records, dtype='<U1')
    h2_arr = np.empty(n_records, dtype='<U1')
    res_arr = np.empty(n_records, dtype='<U10')
    am1_arr = np.empty(n_records, dtype=np.int32)
    am2_arr = np.empty(n_records, dtype=np.int32)

    for t in range(n_steps):
        h1 = ag1.choice()
        h2 = ag2.choice()
        res = rps(h1, h2)
        # Log argmax_h BEFORE learning (state that produced this choice)
        am1_arr[t] = ag1.argmax_h()
        am2_arr[t] = ag2.argmax_h()
        h1_arr[t] = h1
        h2_arr[t] = h2
        res_arr[t] = res
        ag1.learn(h2)
        ag2.learn(h1)

    df = pd.DataFrame({
        't': np.arange(n_steps),
        'h1': h1_arr, 'h2': h2_arr, 'res': res_arr,
        'argmax1': am1_arr, 'argmax2': am2_arr,
        'match': h1_arr == h2_arr,
    })
    return df


# ============================================================
# Run-length / duration extraction
# ============================================================

def consecutive_runs(values):
    """Return run-lengths of consecutive equal values (>=1)."""
    arr = np.asarray(values)
    if len(arr) == 0:
        return np.array([], dtype=int)
    if arr.dtype.kind in ('U', 'O'):
        same = arr[1:] == arr[:-1]
    else:
        same = np.diff(arr) == 0
    changes = np.flatnonzero(~same) + 1
    boundaries = np.concatenate([[0], changes, [len(arr)]])
    return np.diff(boundaries)


def extract_metrics(df):
    """Extract all duration-type metrics from one pair-game DataFrame."""
    return {
        'T_argmax1':   consecutive_runs(df['argmax1'].values),
        'T_argmax2':   consecutive_runs(df['argmax2'].values),
        'hand1_runs':  consecutive_runs(df['h1'].values),
        'hand2_runs':  consecutive_runs(df['h2'].values),
        'result_runs': consecutive_runs(df['res'].values),
        'match_runs':  consecutive_runs(df['match'].values),
    }


# ============================================================
# Parallel run helpers
# ============================================================

def _run_and_extract(args):
    """Worker for parallel runs. Runs one pair simulation, returns metrics dict
    of duration arrays (no DataFrame, to minimize IPC payload).

    args = (a1, a2, n_steps, analysis_start, game_mode, window_size,
            init_mode, predict_mode, seed)
    """
    (a1, a2, n_steps, analysis_start, game_mode, window_size,
     init_mode, predict_mode, seed) = args
    df = run_pair(a1, a2, n_steps, game_mode=game_mode,
                  window_size=window_size, init_mode=init_mode,
                  predict_mode=predict_mode, seed=seed)
    df_ana = df.iloc[analysis_start:]
    metrics = extract_metrics(df_ana)
    # Convert numpy arrays to lists for cheaper IPC pickling
    return {k: v.tolist() for k, v in metrics.items()}


def parallel_runs(a1, a2, n_steps, n_runs, analysis_start,
                  game_mode='rps', window_size=10,
                  init_mode='random', predict_mode='sample',
                  n_workers=None, seed_base=0):
    """Run n_runs simulations in parallel. Returns aggregated metrics dict.

    Falls back to serial execution when n_workers <= 1.
    """
    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS

    metric_keys = ['T_argmax1', 'T_argmax2', 'hand1_runs',
                   'hand2_runs', 'result_runs', 'match_runs']
    agg = {k: [] for k in metric_keys}

    job_args = [
        (a1, a2, n_steps, analysis_start, game_mode, window_size,
         init_mode, predict_mode, seed_base + run)
        for run in range(n_runs)
    ]

    if n_workers <= 1:
        # Serial path (useful for debugging or small jobs)
        for ja in job_args:
            res = _run_and_extract(ja)
            for k in metric_keys:
                agg[k].extend(res[k])
    else:
        # Parallel path
        with Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(_run_and_extract, job_args, chunksize=1):
                for k in metric_keys:
                    agg[k].extend(res[k])

    return agg


# ============================================================
# Distribution fitting (pure PL vs truncated PL vs exponential)
# ============================================================

def fit_distributions(data, discrete=True):
    """Fit three candidates and compare via AIC weights:
    - pure power-law:       p(l) ∝ l^(-α)                  [1 param: α]
    - truncated power-law:  p(l) ∝ l^(-α) exp(-Λl)         [2 params: α, Λ]
    - exponential:          p(l) ∝ exp(-λl)                [1 param: λ]

    Returns w_pl, w_tpl, w_exp (AIC weights summing to 1).
    The 'best' field indicates which model has highest weight.
    """
    data = np.asarray(data)
    data = data[data >= 1]
    if len(data) < 30:
        return {'error': 'too_few_samples', 'n': int(len(data))}

    # Guard against fixed-point case: if data has near-zero variance,
    # fitting will hang or be meaningless.
    unique_vals = np.unique(data)
    if len(unique_vals) < 5:
        return {
            'error': 'degenerate_distribution',
            'n': int(len(data)),
            'n_unique': int(len(unique_vals)),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'note': 'likely fixed-point convergence',
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = powerlaw.Fit(data, discrete=discrete, verbose=False)
    except Exception as e:
        return {'error': str(e), 'n': int(len(data))}

    # Compute log-likelihoods from pdf (robust against powerlaw package quirks)
    data_fit = data[data >= fit.xmin]
    if len(data_fit) < 10:
        return {'error': 'too_few_above_xmin', 'n': int(len(data)),
                'xmin': float(fit.xmin)}

    def _ll_from_pdf(dist):
        """Sum log-pdf, with safety against zeros."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p = dist.pdf(data_fit)
            p = np.asarray(p, dtype=float)
            p = np.where(p > 0, p, 1e-300)
            return float(np.sum(np.log(p)))
        except Exception:
            return None

    ll_pl = _ll_from_pdf(fit.power_law)
    ll_tpl = _ll_from_pdf(fit.truncated_power_law)
    ll_exp = _ll_from_pdf(fit.exponential)

    if ll_pl is None or ll_tpl is None or ll_exp is None:
        return {'error': 'pdf_compute_failed', 'n': int(len(data)),
                'xmin': float(fit.xmin)}

    # AIC: pure-PL has 1 param (α), trunc-PL has 2 (α, Λ), exp has 1 (λ)
    aic = {
        'pl':  2 * 1 - 2 * ll_pl,
        'tpl': 2 * 2 - 2 * ll_tpl,
        'exp': 2 * 1 - 2 * ll_exp,
    }
    aic_min = min(aic.values())
    raw_weights = {k: np.exp(-(v - aic_min) / 2) for k, v in aic.items()}
    total = sum(raw_weights.values())
    weights = {k: v / total for k, v in raw_weights.items()}
    best = max(weights, key=weights.get)

    return {
        'n': int(len(data)),
        'xmin': float(fit.xmin),
        # Parameters for each candidate
        'alpha_pl':  float(fit.power_law.alpha),
        'alpha_tpl': float(fit.truncated_power_law.alpha),
        'Lambda_tpl': float(getattr(fit.truncated_power_law, 'Lambda',
                                    fit.truncated_power_law.parameter2)),
        'lambda_exp': float(getattr(fit.exponential, 'Lambda',
                                    fit.exponential.parameter1)),
        # Log-likelihoods
        'll_pl':  float(ll_pl),
        'll_tpl': float(ll_tpl),
        'll_exp': float(ll_exp),
        # AIC and weights
        'aic_pl':  float(aic['pl']),
        'aic_tpl': float(aic['tpl']),
        'aic_exp': float(aic['exp']),
        'w_pl':  float(weights['pl']),
        'w_tpl': float(weights['tpl']),
        'w_exp': float(weights['exp']),
        'best': best,
        # Levy region check (use whichever PL-family is best)
        'alpha_best': float(fit.power_law.alpha if best == 'pl'
                            else fit.truncated_power_law.alpha),
        'levy_region': bool(1.0 < (fit.power_law.alpha if best == 'pl'
                                   else fit.truncated_power_law.alpha) <= 3.0),
    }


# ============================================================
# Plotting
# ============================================================

def plot_cdf(durations, title='', save_path=None):
    """Log-log CCDF plot with three candidate fits overlaid:
    pure power-law, truncated power-law, and exponential."""
    data = np.asarray(durations)
    data = data[data >= 1]
    if len(data) < 30:
        print(f"  [plot_cdf] too few samples ({len(data)}), skipping")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = powerlaw.Fit(data, discrete=True, verbose=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        fit.plot_ccdf(ax=ax, color='purple', marker='x', linestyle='None',
                      markersize=4, label='data')
        fit.power_law.plot_ccdf(
            ax=ax, color='blue', linewidth=2,
            label=f'pure PL (α={fit.power_law.alpha:.2f})')
        fit.truncated_power_law.plot_ccdf(
            ax=ax, color='green', linewidth=2, linestyle=':',
            label=f'trunc PL (α={fit.truncated_power_law.alpha:.2f})')
        fit.exponential.plot_ccdf(
            ax=ax, color='red', linewidth=2, linestyle='--',
            label=f'exp (λ={getattr(fit.exponential, "Lambda", fit.exponential.parameter1):.3f})')
    except Exception as e:
        print(f"  [plot_cdf] plotting error: {e}")
        plt.close(fig)
        return None

    ax.set_xlabel('Duration T')
    ax.set_ylabel('P(X ≥ T)')
    ax.set_title(title)
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


# ============================================================
# Pilot: single condition quick-look
# ============================================================

def run_pilot(output_dir='./data/pilot', n_workers=None):
    """Single condition: BIB-BIB, RPS, m=10. Confirms shape of distribution."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS

    print("=" * 60)
    print("Pilot: BIB vs BIB, RPS game, m=10")
    print(f"  T=2000 x 100 runs (analyse last 1000 steps of each)")
    print(f"  Using {n_workers} worker process(es)")
    print("=" * 60)

    import time
    t0 = time.time()
    agg = parallel_runs('bib', 'bib', n_steps=2000, n_runs=100,
                        analysis_start=1000, game_mode='rps', window_size=10,
                        init_mode='random', predict_mode='sample',
                        n_workers=n_workers)
    print(f"  simulation done in {time.time() - t0:.1f}s "
          f"(|T_argmax1|={len(agg['T_argmax1'])})")

    # Fit and plot each metric
    print("\nFitting distributions (3 candidates: pure PL, trunc PL, exp)...")
    print(f"{'metric':16s} {'n':>6s} {'best':>5s} {'α':>6s} "
          f"{'w_pl':>6s} {'w_tpl':>6s} {'w_exp':>6s} {'levy':>5s}")
    for metric_name, durations in agg.items():
        fit = fit_distributions(durations)
        if 'error' in fit:
            print(f"{metric_name:16s} {fit.get('n', 0):6d}  error: {fit['error']}")
            continue
        levy = '✓' if fit['levy_region'] else '✗'
        print(f"{metric_name:16s} {fit['n']:6d} {fit['best']:>5s} "
              f"{fit['alpha_best']:6.2f} {fit['w_pl']:6.3f} "
              f"{fit['w_tpl']:6.3f} {fit['w_exp']:6.3f} {levy:>5s}")

        plot_path = out / f'pilot_cdf_{metric_name}.png'
        plot_cdf(durations,
                 title=f'BIB-BIB RPS m=10  ({metric_name}, n={fit["n"]})',
                 save_path=plot_path)

    # Save raw durations
    with open(out / 'pilot_durations.json', 'w') as f:
        json.dump({k: list(v) for k, v in agg.items()}, f)

    print(f"\nResults saved to: {out.resolve()}")


# ============================================================
# Full grid experiment
# ============================================================

def run_grid(output_dir='./data/levy_grid', scale='small',
             init_mode='random', predict_mode='sample',
             n_workers=None):
    """Run full grid: game_mode x match x window x scale."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS

    if scale == 'small':
        n_steps, analysis_start, n_runs = 2000, 1000, 1000
    elif scale == 'medium':
        n_steps, analysis_start, n_runs = 10000, 5000, 200
    elif scale == 'large':
        n_steps, analysis_start, n_runs = 50000, 25000, 50
    elif scale == 'huge':
        n_steps, analysis_start, n_runs = 200000, 100000, 20
    else:
        raise ValueError(scale)

    game_modes = ['rps', 'imitate']
    matches = [
        ('bib', 'bib'), ('bayes', 'bayes'), ('random', 'random'),
        ('bib', 'bayes'), ('bib', 'random'), ('bayes', 'random'),
    ]
    windows = [5, 10, 20, 40]

    metric_keys = ['T_argmax1', 'T_argmax2', 'hand1_runs',
                   'hand2_runs', 'result_runs', 'match_runs']

    rows = []
    total = len(game_modes) * len(matches) * len(windows)
    idx = 0

    print(f"Grid: scale={scale}, init={init_mode}, predict={predict_mode}, "
          f"workers={n_workers}")
    print(f"  T={n_steps}, analyse [{analysis_start}, {n_steps}), "
          f"n_runs={n_runs}, total conditions={total}")

    import time
    grid_t0 = time.time()
    for game_mode, (a1, a2), m in product(game_modes, matches, windows):
        idx += 1
        cond_id = f"{game_mode}_{a1}-{a2}_m{m}_{scale}"
        t0 = time.time()
        agg = parallel_runs(a1, a2, n_steps=n_steps, n_runs=n_runs,
                            analysis_start=analysis_start,
                            game_mode=game_mode, window_size=m,
                            init_mode=init_mode, predict_mode=predict_mode,
                            n_workers=n_workers)
        elapsed = time.time() - t0
        eta = (total - idx) * elapsed
        print(f"[{idx}/{total}] {cond_id}  ({elapsed:.1f}s, ETA {eta/60:.1f}m)")

        with open(out / f'durations_{cond_id}.json', 'w') as f:
            json.dump({k: list(v) for k, v in agg.items()}, f)

        row = {'game_mode': game_mode, 'a1': a1, 'a2': a2, 'window': m,
               'scale': scale, 'n_steps': n_steps, 'n_runs': n_runs,
               'init_mode': init_mode, 'predict_mode': predict_mode}
        for k in metric_keys:
            fit = fit_distributions(agg[k])
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

        rows.append(row)

        if (a1, a2) == ('bib', 'bib') and m == 10:
            plot_cdf(agg['T_argmax1'],
                     title=f'{cond_id} (T_argmax1)',
                     save_path=out / f'cdf_{cond_id}.png')

    summary = pd.DataFrame(rows)
    summary_path = out / f'summary_{scale}.csv'
    summary.to_csv(summary_path, index=False)
    grid_elapsed = time.time() - grid_t0
    print(f"\nGrid done in {grid_elapsed/60:.1f} min")
    print(f"Summary: {summary_path}")
    print(summary[['game_mode', 'a1', 'a2', 'window',
                   'T_argmax1_best', 'T_argmax1_alpha',
                   'T_argmax1_w_pl', 'T_argmax1_levy']].to_string(index=False))
    return summary


# ============================================================
# Design comparison: init_mode x predict_mode x agent type
# ============================================================

def run_design_comparison(output_dir='./data/design_compare', n_runs=100,
                          n_workers=None):
    """Compare 4 design combinations (init x predict) across BIB/BO/random pairs.

    This addresses the question: is the power-law shape we see due to
    BIB's inverse step, or is it an artefact of (random init + sample predict)
    introducing pseudo-non-stationarity into BO too?

    Shinohara 2021 used (structured init + argmax predict) and reported
    BO -> exponential, BIB -> power-law. We test all 4 combinations.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS

    init_modes = ['random', 'structured']
    predict_modes = ['sample', 'argmax']
    pairs = [('bib', 'bib'), ('bayes', 'bayes')]

    n_steps, analysis_start = 2000, 1000

    rows = []
    total = len(init_modes) * len(predict_modes) * len(pairs)
    idx = 0

    print(f"Design comparison: {total} conditions × {n_runs} runs each")
    print(f"  T={n_steps}, analyse [{analysis_start}, {n_steps}), m=10, RPS mode")
    print(f"  Using {n_workers} worker process(es)")
    print()

    import time
    for init_mode, predict_mode, (a1, a2) in product(init_modes, predict_modes, pairs):
        idx += 1
        cond_id = f"init={init_mode}_pred={predict_mode}_{a1}-{a2}"
        t0 = time.time()
        ag_full = parallel_runs(a1, a2, n_steps=n_steps, n_runs=n_runs,
                                analysis_start=analysis_start,
                                game_mode='rps', window_size=10,
                                init_mode=init_mode, predict_mode=predict_mode,
                                n_workers=n_workers)
        agg = ag_full['T_argmax1']
        elapsed = time.time() - t0
        print(f"[{idx}/{total}] {cond_id}  ({elapsed:.1f}s)")

        fit = fit_distributions(agg)
        row = {
            'init_mode': init_mode,
            'predict_mode': predict_mode,
            'pair': f'{a1}-{a2}',
            'n_durations': len(agg),
        }
        if 'error' not in fit:
            row.update({
                'best': fit['best'],
                'alpha': fit['alpha_best'],
                'w_pl': fit['w_pl'],
                'w_tpl': fit['w_tpl'],
                'w_exp': fit['w_exp'],
                'levy': fit['levy_region'],
                'xmin': fit['xmin'],
            })
        else:
            row.update({'error': fit.get('error')})
        rows.append(row)

        plot_cdf(agg, title=cond_id, save_path=out / f'{cond_id}.png')

    df = pd.DataFrame(rows)
    df.to_csv(out / 'design_comparison.csv', index=False)

    print("\n" + "=" * 78)
    print("Design Comparison Results (T_argmax1)")
    print("=" * 78)
    cols = ['init_mode', 'predict_mode', 'pair', 'n_durations']
    if 'best' in df.columns:
        cols += ['best', 'alpha', 'w_pl', 'w_tpl', 'w_exp', 'levy']
    print(df[cols].to_string(index=False))

    # Highlight the key contrast
    print("\nKey contrast (init=structured, predict=argmax — Shinohara-like):")
    sub = df[(df['init_mode'] == 'structured') & (df['predict_mode'] == 'argmax')]
    if 'best' in sub.columns:
        print(sub[cols].to_string(index=False))

    print(f"\nResults saved to: {out.resolve()}")
    return df


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=DEFAULT_N_WORKERS,
                        help=f'Number of parallel worker processes '
                             f'(default: {DEFAULT_N_WORKERS}, set to 1 for serial)')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_pilot = sub.add_parser('pilot', help='Quick pilot (one condition)')
    p_pilot.add_argument('--output', default='./data/pilot')

    p_grid = sub.add_parser('grid', help='Full grid (48 conditions per scale)')
    p_grid.add_argument('--scale',
                        choices=['small', 'medium', 'large', 'huge'],
                        default='small',
                        help='small=2k×1000, medium=10k×200, '
                             'large=50k×50, huge=200k×20 (steps × runs)')
    p_grid.add_argument('--init', dest='init_mode',
                        choices=['random', 'structured'], default='random')
    p_grid.add_argument('--predict', dest='predict_mode',
                        choices=['sample', 'argmax'], default='sample')
    p_grid.add_argument('--output', default='./data/levy_grid')

    p_design = sub.add_parser('design',
        help='Design comparison: init_mode x predict_mode')
    p_design.add_argument('--output', default='./data/design_compare')
    p_design.add_argument('--runs', type=int, default=100)

    args = parser.parse_args()
    if args.cmd == 'pilot':
        run_pilot(args.output, n_workers=args.workers)
    elif args.cmd == 'grid':
        run_grid(args.output, scale=args.scale,
                 init_mode=args.init_mode, predict_mode=args.predict_mode,
                 n_workers=args.workers)
    elif args.cmd == 'design':
        run_design_comparison(args.output, n_runs=args.runs,
                              n_workers=args.workers)


if __name__ == '__main__':
    main()
