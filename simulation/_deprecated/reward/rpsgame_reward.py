# =============================================================================
# PROVENANCE  (see ../README.md for the full engine map)
#   VERSION : earliest pilot -- DEPRECATED
#   Faithful first translation of Ibuka & Sasai 2024. Lacks likelihood_spread()
#   and streak_lengths(). SUPERSEDED by reward_huge/; NOT used for any paper
#   figure or Zenodo dataset. Kept only for lineage.
# =============================================================================
"""
Reward-based BIB inference for Rock-Paper-Scissors
====================================================

Faithful translation of Ibuka & Sasai 2024 (ICA conference) Daihinmin
implementation to the 1v1 RPS setting.

Key difference from rpsgame_levy.py (Gunji-type, observation-driven):
  - Observation d^t is reward-based, not opponent's hand directly
  - win   -> d^t = my_hand               (my hand was good)
  - quits -> no update                   (ambiguous, skip)
  - defeat -> d^t = uniformly random     (my hand was bad,
              from {hands} - {my_hand}    look elsewhere)

Hypotheses h_k (k=0..9) and observations d (in {r,p,s}) are decoupled
in size: N_h=10, N_d=3, following the existing Gunji-type setup
in rpsgame_levy.py for direct comparability.

Action selection (Daihinmin-style):
  h ~ P(h)              sample hypothesis
  d ~ P(d|h)            sample observation
  action = d            output the sampled hand directly
                        (no counter mapping like rpsgame_levy.py:rps mode)

Inverse Bayesian step (Gunji-type, m=50 ring buffer):
  min_h = argmin(P(h))
  P(d|min_h) <- empirical histogram of last m observations / m

References:
  Ibuka & Sasai (2024) ICA. "Inverse Bayesian Inference for Player Agents
  in Computer Daihinmin Game"

Usage:
    python rpsgame_reward.py pilot
    python rpsgame_reward.py grid --scale small
    python rpsgame_reward.py grid --scale huge
"""

import numpy as np
import pandas as pd
import json
import warnings
import time
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
DEFAULT_N_WORKERS = max(1, cpu_count() - 1)


# ============================================================
# Reward-based Bayes / BIB
# ============================================================

class BayesReward:
    """Reward-based BIB agent (Ibuka 2024 style).

    Hypotheses h_k (k=0..N_h-1) are categorical distributions over hands.
    Observation d^t is constructed from game outcome, not from opponent's hand.
    """

    def __init__(self, h_num=10, d_num=3, d_type=None, h_length=50,
                 init_mode='random', predict_mode='sample', seed=None):
        self.rng = np.random.default_rng(seed)
        self.h_num = h_num
        self.d_num = d_num
        self.d_type = np.array(d_type if d_type is not None else HANDS)
        self.h_length = h_length    # m: history window size
        self.init_mode = init_mode
        self.predict_mode = predict_mode

        # Likelihood C(d|h_k): shape (h_num, d_num)
        if init_mode == 'random':
            self.likelihood = np.abs(self.rng.standard_normal((h_num, d_num)))
            self.likelihood /= self.likelihood.sum(axis=1, keepdims=True)
        elif init_mode == 'structured':
            self.likelihood = self._structured_init(h_num, d_num)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        # Hypothesis prior P(h): uniform start
        self.h_prov = np.ones(h_num) / h_num

        # History buffer for last m observations (Daihinmin style)
        self.history = []

    @staticmethod
    def _structured_init(h_num, d_num):
        """Same templates as rpsgame_levy.py for direct comparability."""
        if d_num != 3:
            raise NotImplementedError(
                f"structured init only defined for d_num=3 (got {d_num})")
        templates = np.array([
            [1/3, 1/3, 1/3],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.50, 0.25, 0.25],
            [0.25, 0.50, 0.25],
            [0.25, 0.25, 0.50],
            [0.10, 0.45, 0.45],
            [0.45, 0.10, 0.45],
            [0.45, 0.45, 0.10],
        ])
        if h_num <= len(templates):
            return templates[:h_num].copy()
        out = list(templates)
        rng_pad = np.random.default_rng(12345)
        while len(out) < h_num:
            base = templates[len(out) % len(templates)]
            noise = rng_pad.standard_normal(d_num) * 0.05
            v = np.clip(base + noise, 1e-3, None)
            out.append(v / v.sum())
        return np.array(out[:h_num])

    # ------------------------------------------------------------
    # Standard Bayesian update
    # ------------------------------------------------------------
    def inference(self, data):
        """Standard Bayes: P(h|d) ∝ P(d|h) P(h)."""
        obs_idx = int(np.where(self.d_type == data)[0][0])
        likelihoods = self.likelihood[:, obs_idx]
        marginal = float(np.dot(self.h_prov, likelihoods))
        if marginal <= 0.0:
            return  # numerical safeguard
        post = self.h_prov * likelihoods / marginal
        s = post.sum()
        if s > 0 and np.isfinite(s):
            self.h_prov = post / s

        # Daihinmin-style smoothing: prevent any P(h) from collapsing to 0
        if (self.h_prov < 0.002).any():
            self.h_prov = self.h_prov * 0.91 + 0.015
            self.h_prov /= self.h_prov.sum()

    # ------------------------------------------------------------
    # Inverse Bayesian: replace lowest-posterior hypothesis with
    # empirical histogram of last m observations (Gunji-type, Ibuka 2024)
    # ------------------------------------------------------------
    def inverse(self):
        """If history is full, replace P(d|min_h) with observation histogram."""
        if len(self.history) < self.h_length:
            return  # not enough data yet

        # Compute histogram of last m observations
        hist = np.zeros(self.d_num)
        for obs in self.history[-self.h_length:]:
            obs_idx = int(np.where(self.d_type == obs)[0][0])
            hist[obs_idx] += 1
        hist = hist / self.h_length

        # Find true minimum-posterior hypothesis
        min_h = int(np.argmin(self.h_prov))

        # Replace likelihood of min_h
        self.likelihood[min_h] = hist
        # Smoothing to prevent zero entries (Laplace +1 style)
        if (self.likelihood[min_h] < 1e-6).any():
            self.likelihood[min_h] = self.likelihood[min_h] * 0.95 + 0.05/self.d_num

    def update_history(self, observation):
        """Append observation to history buffer (ring buffer of size m)."""
        self.history.append(observation)
        if len(self.history) > self.h_length:
            self.history.pop(0)

    def argmax_h(self):
        return int(np.argmax(self.h_prov))

    def expect(self):
        """Daihinmin-style action selection: sample h ~ P(h), then d ~ P(d|h).
        Output the sampled d directly (no counter mapping)."""
        if self.predict_mode == 'sample':
            h = self.rng.choice(self.h_num, p=self.h_prov)
        elif self.predict_mode == 'argmax':
            h = int(np.argmax(self.h_prov))
        else:
            raise ValueError(f"Unknown predict_mode: {self.predict_mode}")
        return self.rng.choice(self.d_type, p=self.likelihood[h])


# ============================================================
# Agent wrapper: reward-based update logic
# ============================================================

class AgentReward:
    """RPS agent with reward-based BIB update (Ibuka 2024 style)."""

    def __init__(self, a_type, h_length=50,
                 init_mode='random', predict_mode='sample', seed=None):
        """
        a_type:
          'random' -- uniform random hand, no learning
          'bo'     -- Bayes-only: standard Bayes update, no inverse step
          'bib'    -- BIB: standard Bayes + inverse step (Gunji-type replacement)
        """
        self.type = a_type
        self.rng = np.random.default_rng(seed)
        if a_type == 'random':
            self.bayes = None
        else:
            self.bayes = BayesReward(
                h_num=10, d_num=3, h_length=h_length,
                init_mode=init_mode, predict_mode=predict_mode, seed=seed)

    def choice(self):
        if self.type == 'random':
            return self.rng.choice(HANDS)
        return self.bayes.expect()

    def update_from_outcome(self, my_hand, opp_hand):
        """Reward-based update.
        win   -> kansoku = my_hand
        quits -> no update
        defeat -> kansoku = uniformly chosen from {hands} - {my_hand}
        """
        if self.bayes is None:
            return

        result = rps(my_hand, opp_hand)
        if result == 'quits':
            return  # no update on draw

        if result == 'win':
            kansoku = my_hand
        else:  # defeat
            others = [h for h in HANDS if h != my_hand]
            kansoku = self.rng.choice(others)

        # Standard Bayesian update with kansoku as observation
        self.bayes.inference(kansoku)
        # Update history buffer
        self.bayes.update_history(kansoku)
        # Inverse Bayesian step (BIB only)
        if self.type == 'bib':
            self.bayes.inverse()

    def argmax_h(self):
        return -1 if self.bayes is None else self.bayes.argmax_h()


def rps(me, rival):
    if me == rival:
        return 'quits'
    wins = {('r', 's'), ('p', 'r'), ('s', 'p')}
    return 'win' if (me, rival) in wins else 'defeat'


# ============================================================
# Pair simulation
# ============================================================

def run_pair(a1, a2, n_steps,
             h_length=50, init_mode='random', predict_mode='sample',
             seed=None):
    seed1 = seed
    seed2 = (seed + 100000) if seed is not None else None
    ag1 = AgentReward(a1, h_length, init_mode, predict_mode, seed=seed1)
    ag2 = AgentReward(a2, h_length, init_mode, predict_mode, seed=seed2)

    h1_arr = np.empty(n_steps, dtype='<U1')
    h2_arr = np.empty(n_steps, dtype='<U1')
    res_arr = np.empty(n_steps, dtype='<U10')
    am1_arr = np.empty(n_steps, dtype=np.int32)
    am2_arr = np.empty(n_steps, dtype=np.int32)

    for t in range(n_steps):
        h1 = ag1.choice()
        h2 = ag2.choice()
        am1_arr[t] = ag1.argmax_h()
        am2_arr[t] = ag2.argmax_h()
        h1_arr[t] = h1
        h2_arr[t] = h2
        res_arr[t] = rps(h1, h2)
        # Reward-based update for each agent
        ag1.update_from_outcome(h1, h2)
        ag2.update_from_outcome(h2, h1)

    return pd.DataFrame({
        't': np.arange(n_steps),
        'h1': h1_arr, 'h2': h2_arr, 'res': res_arr,
        'argmax1': am1_arr, 'argmax2': am2_arr,
        'match': h1_arr == h2_arr,
    })


# ============================================================
# Metrics & fitting (reused from rpsgame_levy.py)
# ============================================================

def consecutive_runs(values):
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
    return {
        'T_argmax1':   consecutive_runs(df['argmax1'].values),
        'T_argmax2':   consecutive_runs(df['argmax2'].values),
        'hand1_runs':  consecutive_runs(df['h1'].values),
        'hand2_runs':  consecutive_runs(df['h2'].values),
        'result_runs': consecutive_runs(df['res'].values),
        'match_runs':  consecutive_runs(df['match'].values),
    }


def fit_distributions(data, discrete=True):
    data = np.asarray(data)
    data = data[data >= 1]
    if len(data) < 30:
        return {'error': 'too_few_samples', 'n': int(len(data))}
    unique_vals = np.unique(data)
    if len(unique_vals) < 5:
        return {'error': 'degenerate_distribution', 'n': int(len(data)),
                'n_unique': int(len(unique_vals))}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = powerlaw.Fit(data, discrete=discrete, verbose=False)
    except Exception as e:
        return {'error': str(e), 'n': int(len(data))}

    data_fit = data[data >= fit.xmin]
    if len(data_fit) < 10:
        return {'error': 'too_few_above_xmin', 'n': int(len(data))}

    def _ll(dist):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p = np.asarray(dist.pdf(data_fit), dtype=float)
            p = np.where(p > 0, p, 1e-300)
            return float(np.sum(np.log(p)))
        except Exception:
            return None

    ll_pl = _ll(fit.power_law)
    ll_tpl = _ll(fit.truncated_power_law)
    ll_exp = _ll(fit.exponential)
    if ll_pl is None or ll_tpl is None or ll_exp is None:
        return {'error': 'pdf_compute_failed', 'n': int(len(data))}

    aic = {'pl': 2 - 2 * ll_pl, 'tpl': 4 - 2 * ll_tpl, 'exp': 2 - 2 * ll_exp}
    aic_min = min(aic.values())
    raw = {k: np.exp(-(v - aic_min) / 2) for k, v in aic.items()}
    total = sum(raw.values())
    weights = {k: v / total for k, v in raw.items()}
    best = max(weights, key=weights.get)

    alpha_best = float(fit.power_law.alpha if best == 'pl'
                       else fit.truncated_power_law.alpha)
    return {
        'n': int(len(data)),
        'xmin': float(fit.xmin),
        'alpha_pl': float(fit.power_law.alpha),
        'alpha_tpl': float(fit.truncated_power_law.alpha),
        'Lambda_tpl': float(getattr(fit.truncated_power_law, 'Lambda',
                                    fit.truncated_power_law.parameter2)),
        'lambda_exp': float(getattr(fit.exponential, 'Lambda',
                                    fit.exponential.parameter1)),
        'll_pl': float(ll_pl), 'll_tpl': float(ll_tpl), 'll_exp': float(ll_exp),
        'w_pl': float(weights['pl']), 'w_tpl': float(weights['tpl']),
        'w_exp': float(weights['exp']),
        'best': best,
        'alpha_best': alpha_best,
        'levy_region': bool(1.0 < alpha_best <= 3.0),
    }


def plot_cdf(durations, title='', save_path=None, ymin_padding=0.5):
    """Log-log CCDF with three candidate fits.

    The y-axis is clipped to roughly match the data's CCDF range so that
    rapidly-decaying fits (e.g. exponential) don't stretch the plot
    downward and squash the data points.

    Parameters
    ----------
    ymin_padding : float
        The y-axis lower bound is set to (min CCDF of data) * ymin_padding.
        Smaller value -> a bit more room below data; default 0.5.
    """
    data = np.asarray(durations)
    data = data[data >= 1]
    if len(data) < 30 or len(np.unique(data)) < 5:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = powerlaw.Fit(data, discrete=True, verbose=False)

    # Compute data CCDF range so that the figure shows data clearly,
    # and fit curves that drop far below the data are clipped at the bottom.
    n = len(data)
    data_above_xmin = data[data >= fit.xmin]
    n_above = max(len(data_above_xmin), 1)
    # Smallest positive CCDF value in the fitted range is 1/n_above
    data_ccdf_min = 1.0 / n_above
    y_lower = data_ccdf_min * ymin_padding
    y_upper = 1.5

    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        fit.plot_ccdf(ax=ax, color='purple', marker='x', linestyle='None',
                      markersize=4, label='data')
        fit.power_law.plot_ccdf(ax=ax, color='blue', linewidth=2,
                                label=f'pure PL (α={fit.power_law.alpha:.2f})')
        fit.truncated_power_law.plot_ccdf(
            ax=ax, color='green', linewidth=2, linestyle=':',
            label=f'trunc PL (α={fit.truncated_power_law.alpha:.2f})')
        fit.exponential.plot_ccdf(
            ax=ax, color='red', linewidth=2, linestyle='--',
            label=f'exp (λ={getattr(fit.exponential, "Lambda", fit.exponential.parameter1):.3f})')
    except Exception:
        plt.close(fig); return None

    ax.set_xlabel('Duration T'); ax.set_ylabel('P(X ≥ T)')
    ax.set_title(title); ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(y_lower, y_upper)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


# ============================================================
# Parallel runs
# ============================================================

def _run_and_extract(args):
    (a1, a2, n_steps, analysis_start,
     h_length, init_mode, predict_mode, seed) = args
    df = run_pair(a1, a2, n_steps, h_length=h_length,
                  init_mode=init_mode, predict_mode=predict_mode, seed=seed)
    df_ana = df.iloc[analysis_start:]
    metrics = extract_metrics(df_ana)
    return {k: v.tolist() for k, v in metrics.items()}


def parallel_runs(a1, a2, n_steps, n_runs, analysis_start,
                  h_length=50, init_mode='random', predict_mode='sample',
                  n_workers=None, seed_base=0):
    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS
    metric_keys = ['T_argmax1', 'T_argmax2', 'hand1_runs',
                   'hand2_runs', 'result_runs', 'match_runs']
    agg = {k: [] for k in metric_keys}
    job_args = [
        (a1, a2, n_steps, analysis_start,
         h_length, init_mode, predict_mode, seed_base + run)
        for run in range(n_runs)
    ]
    if n_workers <= 1:
        for ja in job_args:
            res = _run_and_extract(ja)
            for k in metric_keys:
                agg[k].extend(res[k])
    else:
        with Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(_run_and_extract, job_args, chunksize=1):
                for k in metric_keys:
                    agg[k].extend(res[k])
    return agg


# ============================================================
# Pilot: replicate Ibuka 2024 conditions on RPS
# ============================================================

def run_pilot(output_dir='./data/reward_pilot', n_workers=None):
    """Replicate Ibuka 2024 main result on RPS:
    - BO  (Bayes only) :  expected exponential
    - BIB (Bayes + inverse) :  expected truncated power-law (η ≈ 2)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS

    print("=" * 70)
    print("Reward-based BIB pilot (Ibuka 2024 style on RPS)")
    print(f"  T=2000 × 100 runs, analyse last 1000")
    print(f"  N_h=10, N_d=3, m=50, random init, sample predict")
    print(f"  Workers: {n_workers}")
    print("=" * 70)

    print(f"\n{'pair':18s} {'n':>6s} {'best':>5s} {'α':>6s} "
          f"{'w_pl':>6s} {'w_tpl':>6s} {'w_exp':>6s} {'levy':>5s}")
    print("-" * 75)

    pairs = [('bib', 'bib'), ('bo', 'bo'), ('bib', 'bo'),
             ('bib', 'random'), ('bo', 'random')]

    for a1, a2 in pairs:
        cond_id = f"{a1}-{a2}"
        t0 = time.time()
        agg = parallel_runs(a1, a2, n_steps=2000, n_runs=100,
                            analysis_start=1000,
                            h_length=50, init_mode='random',
                            predict_mode='sample',
                            n_workers=n_workers)
        elapsed = time.time() - t0

        fit = fit_distributions(agg['T_argmax1'])
        if 'error' not in fit:
            levy = '✓' if fit['levy_region'] else '✗'
            print(f"{cond_id:18s} {fit['n']:6d} {fit['best']:>5s} "
                  f"{fit['alpha_best']:6.2f} {fit['w_pl']:6.3f} "
                  f"{fit['w_tpl']:6.3f} {fit['w_exp']:6.3f} {levy:>5s}  "
                  f"({elapsed:.0f}s)")
        else:
            print(f"{cond_id:18s}  error: {fit.get('error')} "
                  f"(n={fit.get('n', 0)}, {elapsed:.0f}s)")

        with open(out / f'durations_{cond_id}.json', 'w') as f:
            json.dump({k: list(v) for k, v in agg.items()}, f)
        plot_cdf(agg['T_argmax1'],
                 title=f'Reward-BIB RPS: {cond_id}',
                 save_path=out / f'cdf_{cond_id}.png')

    print(f"\nResults: {out.resolve()}")


# ============================================================
# Grid: full window x pair x scale sweep
# ============================================================

def run_grid(output_dir='./data/reward_grid', scale='small',
             init_mode='random', predict_mode='sample',
             n_workers=None):
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

    pairs = [('bib', 'bib'), ('bo', 'bo'), ('random', 'random'),
             ('bib', 'bo'), ('bib', 'random'), ('bo', 'random')]
    windows = [10, 20, 50, 100]   # m=50 is Ibuka 2024 default

    metric_keys = ['T_argmax1', 'T_argmax2', 'hand1_runs',
                   'hand2_runs', 'result_runs', 'match_runs']

    rows = []
    total = len(pairs) * len(windows)
    idx = 0

    print(f"Reward-based BIB grid: scale={scale}, init={init_mode}, "
          f"predict={predict_mode}, workers={n_workers}")
    print(f"  T={n_steps}, analyse [{analysis_start}, {n_steps}), "
          f"n_runs={n_runs}, total={total}")

    grid_t0 = time.time()
    for (a1, a2), m in product(pairs, windows):
        idx += 1
        cond_id = f"{a1}-{a2}_m{m}_{scale}"
        t0 = time.time()
        agg = parallel_runs(a1, a2, n_steps=n_steps, n_runs=n_runs,
                            analysis_start=analysis_start,
                            h_length=m, init_mode=init_mode,
                            predict_mode=predict_mode,
                            n_workers=n_workers)
        elapsed = time.time() - t0
        eta = (total - idx) * elapsed
        print(f"[{idx}/{total}] {cond_id}  ({elapsed:.1f}s, ETA {eta/60:.1f}m)")

        with open(out / f'durations_{cond_id}.json', 'w') as f:
            json.dump({k: list(v) for k, v in agg.items()}, f)

        row = {'a1': a1, 'a2': a2, 'window': m, 'scale': scale,
               'n_steps': n_steps, 'n_runs': n_runs,
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

    summary = pd.DataFrame(rows)
    summary_path = out / f'summary_{scale}.csv'
    summary.to_csv(summary_path, index=False)
    print(f"\nGrid done in {(time.time() - grid_t0)/60:.1f} min")
    print(f"Summary: {summary_path}")
    print(summary[['a1', 'a2', 'window',
                   'T_argmax1_n', 'T_argmax1_alpha', 'T_argmax1_best',
                   'T_argmax1_levy']].to_string(index=False))


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=DEFAULT_N_WORKERS)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_pilot = sub.add_parser('pilot', help='Replicate Ibuka 2024 on RPS')
    p_pilot.add_argument('--output', default='./data/reward_pilot')

    p_grid = sub.add_parser('grid', help='Full window×pair sweep')
    p_grid.add_argument('--scale',
                        choices=['small', 'medium', 'large', 'huge'],
                        default='small')
    p_grid.add_argument('--init', dest='init_mode',
                        choices=['random', 'structured'], default='random')
    p_grid.add_argument('--predict', dest='predict_mode',
                        choices=['sample', 'argmax'], default='sample')
    p_grid.add_argument('--output', default='./data/reward_grid')

    args = parser.parse_args()
    if args.cmd == 'pilot':
        run_pilot(args.output, n_workers=args.workers)
    elif args.cmd == 'grid':
        run_grid(args.output, scale=args.scale,
                 init_mode=args.init_mode, predict_mode=args.predict_mode,
                 n_workers=args.workers)


if __name__ == '__main__':
    main()
