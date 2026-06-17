# =============================================================================
# PROVENANCE  (see ../README.md for the full engine map)
#   VERSION : production engine, gen-1   |   1019 lines   |   md5(head) fd4b6e
#   Canonical reward engine: adds likelihood_spread() (posterior spread sigma ->
#   beta exponent) and streak_lengths(); emits durations_* and sigmas_*.json.
#   BYTE-IDENTICAL to analyze_sharpness_plateau/rpsgame_reward.py (same md5).
#   Ships the production runners run_phase1/2_nh_sweep.sh, run_all_huge_v2/v3.sh.
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

    def likelihood_spread(self):
        """Std deviation of posterior across hypotheses; expected to scale
        as 1/N_h under naive uniform-difference assumption."""
        return float(np.std(self.h_prov))

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

    def __init__(self, a_type, h_length=50, h_num=10,
                 init_mode='random', predict_mode='sample', seed=None):
        """
        a_type:
          'random' -- uniform random hand, no learning
          'bo'     -- Bayes-only: standard Bayes update, no inverse step
          'bib'    -- BIB: standard Bayes + inverse step (Gunji-type replacement)
        h_num: number of hypotheses (default 10, Ibuka-style)
        """
        self.type = a_type
        self.rng = np.random.default_rng(seed)
        if a_type == 'random':
            self.bayes = None
        else:
            self.bayes = BayesReward(
                h_num=h_num, d_num=3, h_length=h_length,
                init_mode=init_mode, predict_mode=predict_mode, seed=seed)

    def likelihood_spread(self):
        """Wrapper: 0 for random agent, σ(P(h)) for bayes/bib agents."""
        if self.bayes is None:
            return 0.0
        return self.bayes.likelihood_spread()

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
             h_length=50, h_num=10,
             init_mode='random', predict_mode='sample',
             seed=None):
    seed1 = seed
    seed2 = (seed + 100000) if seed is not None else None
    ag1 = AgentReward(a1, h_length, h_num, init_mode, predict_mode, seed=seed1)
    ag2 = AgentReward(a2, h_length, h_num, init_mode, predict_mode, seed=seed2)

    h1_arr = np.empty(n_steps, dtype='<U1')
    h2_arr = np.empty(n_steps, dtype='<U1')
    res_arr = np.empty(n_steps, dtype='<U10')
    am1_arr = np.empty(n_steps, dtype=np.int32)
    am2_arr = np.empty(n_steps, dtype=np.int32)
    sig1_arr = np.empty(n_steps, dtype=np.float32)
    sig2_arr = np.empty(n_steps, dtype=np.float32)

    for t in range(n_steps):
        h1 = ag1.choice()
        h2 = ag2.choice()
        am1_arr[t] = ag1.argmax_h()
        am2_arr[t] = ag2.argmax_h()
        sig1_arr[t] = ag1.likelihood_spread()
        sig2_arr[t] = ag2.likelihood_spread()
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
        'sig1': sig1_arr, 'sig2': sig2_arr,
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


def streak_lengths(boolean_array):
    """Length of consecutive True runs in a boolean array.
    Returns empty array if no True values."""
    arr = np.asarray(boolean_array, dtype=bool)
    if not arr.any():
        return np.array([], dtype=int)
    diffs = np.diff(arr.astype(np.int8))
    starts = np.flatnonzero(diffs == 1) + 1
    ends = np.flatnonzero(diffs == -1) + 1
    if arr[0]:
        starts = np.r_[0, starts]
    if arr[-1]:
        ends = np.r_[ends, len(arr)]
    return ends - starts


def extract_metrics(df):
    """Extract all duration-type metrics for power-law analysis.

    Categories:
      - Hypothesis dynamics:  T_argmax* (internal state switching)
      - Action dynamics:      hand*_runs (output streaks)
      - Outcome dynamics:     result_runs, match_runs (3-state outcome)
      - Reward dynamics:      win_runs, win_or_draw_runs, defeat_runs
                              (Ibuka 2024 style binary outcome streaks)

    The reward-dynamics metrics test whether 'consecutive wins' (Ibuka 2024)
    show power-law scaling. Three variants distinguish how draws are treated:
      - win_runs:          strictly consecutive wins (broken by draw or defeat)
      - win_or_draw_runs:  not-defeated streaks (draws don't break the run)
      - defeat_runs:       strictly consecutive defeats
    """
    res = df['res'].values
    return {
        # Internal state
        'T_argmax1':         consecutive_runs(df['argmax1'].values),
        'T_argmax2':         consecutive_runs(df['argmax2'].values),
        # Action level
        'hand1_runs':        consecutive_runs(df['h1'].values),
        'hand2_runs':        consecutive_runs(df['h2'].values),
        # Outcome level (3-state, includes draws)
        'result_runs':       consecutive_runs(res),
        'match_runs':        consecutive_runs(df['match'].values),
        # Reward level (binary, Ibuka 2024 style)
        'win_runs':          streak_lengths(res == 'win'),
        'win_or_draw_runs':  streak_lengths(res != 'defeat'),
        'defeat_runs':       streak_lengths(res == 'defeat'),
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

    def _safe_attr(name):
        """Get a powerlaw distribution attribute. Trigger the internal MLE
        fit and catch ZeroDivisionError / ValueError that the powerlaw
        library can raise on degenerate data."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return getattr(fit, name)
        except (ZeroDivisionError, ValueError, OverflowError, AttributeError):
            return None
        except Exception:
            return None

    def _ll(dist):
        if dist is None:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p = np.asarray(dist.pdf(data_fit), dtype=float)
            p = np.where(p > 0, p, 1e-300)
            return float(np.sum(np.log(p)))
        except (ZeroDivisionError, ValueError, OverflowError):
            return None
        except Exception:
            return None

    pl_dist = _safe_attr('power_law')
    tpl_dist = _safe_attr('truncated_power_law')
    exp_dist = _safe_attr('exponential')

    ll_pl = _ll(pl_dist)
    ll_tpl = _ll(tpl_dist)
    ll_exp = _ll(exp_dist)

    # If TPL fitting failed, fall back to comparing only PL vs exp.
    # This is robust to ZeroDivisionError from the powerlaw library that
    # can occur with very long durations (e.g. huge-scale experiments).
    available = {}
    if ll_pl is not None:
        available['pl'] = ll_pl
    if ll_tpl is not None:
        available['tpl'] = ll_tpl
    if ll_exp is not None:
        available['exp'] = ll_exp

    if 'pl' not in available and 'exp' not in available:
        return {'error': 'all_fits_failed', 'n': int(len(data))}

    # AIC: each PL/exp/TPL has 1, 1, 2 free params respectively
    n_params = {'pl': 1, 'exp': 1, 'tpl': 2}
    aic = {k: 2 * n_params[k] - 2 * ll for k, ll in available.items()}
    aic_min = min(aic.values())
    raw = {k: np.exp(-(v - aic_min) / 2) for k, v in aic.items()}
    total = sum(raw.values())
    weights = {k: v / total for k, v in raw.items()}
    # Fill in missing models with weight 0 for downstream consistency
    for k in ('pl', 'tpl', 'exp'):
        weights.setdefault(k, 0.0)
    best = max(weights, key=weights.get)

    # Get alphas (some may be None if fitting failed)
    alpha_pl = float(pl_dist.alpha) if pl_dist is not None else np.nan
    alpha_tpl = float(tpl_dist.alpha) if tpl_dist is not None else np.nan
    Lambda_tpl = (float(getattr(tpl_dist, 'Lambda',
                                getattr(tpl_dist, 'parameter2', np.nan)))
                  if tpl_dist is not None else np.nan)
    lambda_exp = (float(getattr(exp_dist, 'Lambda',
                                getattr(exp_dist, 'parameter1', np.nan)))
                  if exp_dist is not None else np.nan)

    if best == 'pl' and not np.isnan(alpha_pl):
        alpha_best = alpha_pl
    elif best == 'tpl' and not np.isnan(alpha_tpl):
        alpha_best = alpha_tpl
    elif not np.isnan(alpha_tpl):
        alpha_best = alpha_tpl
    elif not np.isnan(alpha_pl):
        alpha_best = alpha_pl
    else:
        alpha_best = np.nan

    return {
        'n': int(len(data)),
        'xmin': float(fit.xmin),
        'alpha_pl': alpha_pl,
        'alpha_tpl': alpha_tpl,
        'Lambda_tpl': Lambda_tpl,
        'lambda_exp': lambda_exp,
        'll_pl': float(ll_pl) if ll_pl is not None else np.nan,
        'll_tpl': float(ll_tpl) if ll_tpl is not None else np.nan,
        'll_exp': float(ll_exp) if ll_exp is not None else np.nan,
        'w_pl': float(weights['pl']),
        'w_tpl': float(weights['tpl']),
        'w_exp': float(weights['exp']),
        'best': best,
        'alpha_best': float(alpha_best) if not np.isnan(alpha_best) else np.nan,
        'levy_region': bool(1.0 < alpha_best <= 3.0)
                        if not np.isnan(alpha_best) else False,
        'tpl_fit_failed': tpl_dist is None,
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

    # Plot data points (always works)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit.plot_ccdf(ax=ax, color='purple', marker='x', linestyle='None',
                          markersize=4, label='data')
    except Exception:
        plt.close(fig); return None

    # Each fit overlay is independently optional; skip if fitting fails.
    def _try_plot(name, color, linestyle, linewidth, label_fn):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dist = getattr(fit, name)
                dist.plot_ccdf(ax=ax, color=color, linewidth=linewidth,
                               linestyle=linestyle, label=label_fn(dist))
        except (ZeroDivisionError, ValueError, OverflowError, AttributeError):
            pass
        except Exception:
            pass

    _try_plot('power_law', 'blue', '-', 2,
              lambda d: f'pure PL (α={d.alpha:.2f})')
    _try_plot('truncated_power_law', 'green', ':', 2,
              lambda d: f'trunc PL (α={d.alpha:.2f})')
    _try_plot('exponential', 'red', '--', 2,
              lambda d: f'exp (λ={getattr(d, "Lambda", d.parameter1):.3f})')

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
    """Worker: run one simulation and return:
      - duration metrics (existing 9 indicators)
      - reward statistics: per-run win/defeat/quits counts and
        downsampled cumulative reward / running win rate time series.
      - sigma statistics: per-run mean/std of σ(P(h)) for both agents,
        plus downsampled time series.
    """
    (a1, a2, n_steps, analysis_start,
     h_length, h_num, init_mode, predict_mode, seed) = args
    df = run_pair(a1, a2, n_steps, h_length=h_length, h_num=h_num,
                  init_mode=init_mode, predict_mode=predict_mode, seed=seed)
    df_ana = df.iloc[analysis_start:]
    metrics = extract_metrics(df_ana)

    # ---- Reward statistics (over the analyzed window) ----
    res = df_ana['res'].values
    n_post = len(res)
    win_count = int((res == 'win').sum())
    defeat_count = int((res == 'defeat').sum())
    quits_count = int((res == 'quits').sum())

    # Cumulative reward, downsampled to ~200 points to keep JSON small
    rewards = np.where(res == 'win', 1, np.where(res == 'defeat', -1, 0))
    cumR = np.cumsum(rewards)
    n_keep = 200
    if n_post >= n_keep:
        idx_ds = np.linspace(0, n_post - 1, n_keep).astype(int)
        cumR_ds = cumR[idx_ds].tolist()
        # Running win rate (window=200), downsampled
        K = max(50, n_post // 200)
        is_win = (res == 'win').astype(float)
        cs = np.cumsum(np.insert(is_win, 0, 0))
        rw = np.full(n_post, np.nan)
        if n_post >= K:
            rw[K - 1:] = (cs[K:] - cs[:-K]) / K
        rw_ds = rw[idx_ds].tolist()
        # Replace NaN with None for JSON
        rw_ds = [None if (v != v) else float(v) for v in rw_ds]
    else:
        cumR_ds = cumR.tolist()
        rw_ds = []
        idx_ds = np.arange(n_post)

    reward_stats = {
        'n_post': n_post,
        'win_count': win_count,
        'defeat_count': defeat_count,
        'quits_count': quits_count,
        'cumR_final': int(cumR[-1]) if n_post > 0 else 0,
        'cumR_downsampled': cumR_ds,
        'win_rate_downsampled': rw_ds,
        'sample_indices': idx_ds.tolist() if hasattr(idx_ds, 'tolist') else list(idx_ds),
        'seed': seed,
    }

    # ---- Sigma statistics (likelihood spread σ(P(h)) for each agent) ----
    sig1 = df_ana['sig1'].values
    sig2 = df_ana['sig2'].values
    if n_post >= n_keep:
        sig1_ds = sig1[idx_ds].astype(float).tolist()
        sig2_ds = sig2[idx_ds].astype(float).tolist()
    else:
        sig1_ds = sig1.astype(float).tolist()
        sig2_ds = sig2.astype(float).tolist()

    sigma_stats = {
        'h_num': int(h_num),
        'sig1_mean': float(np.mean(sig1)),
        'sig1_std': float(np.std(sig1)),
        'sig1_max': float(np.max(sig1)),
        'sig1_min': float(np.min(sig1)),
        'sig1_downsampled': sig1_ds,
        'sig2_mean': float(np.mean(sig2)),
        'sig2_std': float(np.std(sig2)),
        'sig2_max': float(np.max(sig2)),
        'sig2_min': float(np.min(sig2)),
        'sig2_downsampled': sig2_ds,
        'seed': seed,
    }

    return {
        'metrics': {k: v.tolist() for k, v in metrics.items()},
        'reward_stats': reward_stats,
        'sigma_stats': sigma_stats,
    }


def parallel_runs(a1, a2, n_steps, n_runs, analysis_start,
                  h_length=50, h_num=10,
                  init_mode='random', predict_mode='sample',
                  n_workers=None, seed_base=0):
    """Run n_runs simulations in parallel.

    Returns:
      agg : dict of metric_name -> aggregated list (across all runs)
      reward_stats_list : list of per-run reward_stats dicts
      sigma_stats_list : list of per-run sigma_stats dicts
    """
    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS
    metric_keys = ['T_argmax1', 'T_argmax2', 'hand1_runs', 'hand2_runs',
                   'result_runs', 'match_runs',
                   'win_runs', 'win_or_draw_runs', 'defeat_runs']
    agg = {k: [] for k in metric_keys}
    reward_stats_list = []
    sigma_stats_list = []

    job_args = [
        (a1, a2, n_steps, analysis_start,
         h_length, h_num, init_mode, predict_mode, seed_base + run)
        for run in range(n_runs)
    ]
    if n_workers <= 1:
        for ja in job_args:
            out = _run_and_extract(ja)
            for k in metric_keys:
                agg[k].extend(out['metrics'][k])
            reward_stats_list.append(out['reward_stats'])
            sigma_stats_list.append(out['sigma_stats'])
    else:
        with Pool(processes=n_workers) as pool:
            for out in pool.imap_unordered(_run_and_extract, job_args, chunksize=1):
                for k in metric_keys:
                    agg[k].extend(out['metrics'][k])
                reward_stats_list.append(out['reward_stats'])
                sigma_stats_list.append(out['sigma_stats'])

    return agg, reward_stats_list, sigma_stats_list


# ============================================================
# Pilot: replicate Ibuka 2024 conditions on RPS
# ============================================================

def run_pilot(output_dir='./data/reward_pilot', n_workers=None,
              full_design=False, h_num=10):
    """Replicate Ibuka 2024 main result on RPS.

    By default runs the random+sample design (matches earlier rpsgame_levy.py
    pilot). Pass full_design=True (or --full from CLI) to sweep all 4
    combinations of (init_mode, predict_mode), which is the appropriate
    comparison for paper-level analysis.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS

    if full_design:
        designs = [(init, pred)
                   for init in ['random', 'structured']
                   for pred in ['sample', 'argmax']]
    else:
        designs = [('random', 'sample')]

    print("=" * 75)
    print("Reward-based BIB pilot (Ibuka 2024 style on RPS)")
    print(f"  T=2000 × 100 runs, analyse last 1000")
    print(f"  N_h={h_num}, N_d=3, m=50")
    print(f"  Designs: {len(designs)} ({'full 2x2' if full_design else 'random+sample only'})")
    print(f"  Workers: {n_workers}")
    print("=" * 75)

    pairs = [('bib', 'bib'), ('bo', 'bo'), ('bib', 'bo'),
             ('bib', 'random'), ('bo', 'random')]

    for init_mode, predict_mode in designs:
        design_tag = f"init={init_mode}_pred={predict_mode}"
        print(f"\n--- {design_tag} ---")
        print(f"{'pair':18s} {'n':>6s} {'best':>5s} {'α':>6s} "
              f"{'w_pl':>6s} {'w_tpl':>6s} {'w_exp':>6s} {'levy':>5s}")
        print("-" * 75)

        for a1, a2 in pairs:
            cond_id = f"{design_tag}_{a1}-{a2}"
            t0 = time.time()
            agg, reward_stats, sigma_stats = parallel_runs(
                a1, a2, n_steps=2000, n_runs=100,
                analysis_start=1000,
                h_length=50, h_num=h_num,
                init_mode=init_mode, predict_mode=predict_mode,
                n_workers=n_workers)
            elapsed = time.time() - t0

            # Aggregate reward stats across runs
            avg_winrate = np.mean([r['win_count'] / r['n_post']
                                   for r in reward_stats if r['n_post'] > 0])
            avg_cumR_per_step = np.mean([r['cumR_final'] / r['n_post']
                                         for r in reward_stats
                                         if r['n_post'] > 0])
            avg_sig1 = np.mean([s['sig1_mean'] for s in sigma_stats])

            fit = fit_distributions(agg['T_argmax1'])
            if 'error' not in fit:
                levy = '✓' if fit['levy_region'] else '✗'
                print(f"{a1+'-'+a2:18s} {fit['n']:6d} {fit['best']:>5s} "
                      f"{fit['alpha_best']:6.2f} "
                      f"WR={avg_winrate:.4f} R/t={avg_cumR_per_step:+.4f} "
                      f"σ̄={avg_sig1:.4f} {levy:>3s}  ({elapsed:.0f}s)")
            else:
                print(f"{a1+'-'+a2:18s}  error: {fit.get('error')} "
                      f"(n={fit.get('n', 0)}, {elapsed:.0f}s)")

            with open(out / f'durations_{cond_id}.json', 'w') as f:
                json.dump({k: list(v) for k, v in agg.items()}, f)
            with open(out / f'rewards_{cond_id}.json', 'w') as f:
                json.dump(reward_stats, f)
            with open(out / f'sigmas_{cond_id}.json', 'w') as f:
                json.dump(sigma_stats, f)
            plot_cdf(agg['T_argmax1'],
                     title=f'Reward-BIB: {cond_id}',
                     save_path=out / f'cdf_{cond_id}.png')

    print(f"\nResults: {out.resolve()}")


# ============================================================
# Grid: full window x pair x scale sweep
# ============================================================

def run_grid(output_dir='./data/reward_grid', scale='small',
             init_mode='random', predict_mode='sample',
             h_num=10, n_workers=None):
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

    metric_keys = ['T_argmax1', 'T_argmax2', 'hand1_runs', 'hand2_runs',
                   'result_runs', 'match_runs',
                   'win_runs', 'win_or_draw_runs', 'defeat_runs']

    rows = []
    total = len(pairs) * len(windows)
    idx = 0

    print(f"Reward-based BIB grid: scale={scale}, init={init_mode}, "
          f"predict={predict_mode}, h_num={h_num}, workers={n_workers}")
    print(f"  T={n_steps}, analyse [{analysis_start}, {n_steps}), "
          f"n_runs={n_runs}, total={total}")

    grid_t0 = time.time()
    for (a1, a2), m in product(pairs, windows):
        idx += 1
        cond_id = f"{a1}-{a2}_m{m}_{scale}"
        durations_path = out / f'durations_{cond_id}.json'
        rewards_path = out / f'rewards_{cond_id}.json'
        sigmas_path = out / f'sigmas_{cond_id}.json'

        # ---- Resume support ----
        # All three files (durations, rewards, sigmas) must exist for cache hit;
        # otherwise re-run the simulation. Caches missing sigmas will trigger
        # re-run (this is the path for grids run with the v3 σ-logging code).
        agg = None
        reward_stats = None
        sigma_stats = None
        if (durations_path.exists() and rewards_path.exists()
                and sigmas_path.exists()):
            try:
                with open(durations_path) as f:
                    agg_loaded = json.load(f)
                with open(rewards_path) as f:
                    reward_stats = json.load(f)
                with open(sigmas_path) as f:
                    sigma_stats = json.load(f)
                if all(k in agg_loaded for k in metric_keys):
                    agg = agg_loaded
                    print(f"[{idx}/{total}] {cond_id}  (loaded from cache)")
                else:
                    raise ValueError("incomplete cache")
            except Exception as e:
                print(f"[{idx}/{total}] {cond_id}  (cache invalid: {e}, re-running)")
                agg = None; reward_stats = None; sigma_stats = None
        elif durations_path.exists() and not (
                rewards_path.exists() and sigmas_path.exists()):
            print(f"[{idx}/{total}] {cond_id}  "
                  f"(re-running: rewards or sigmas data missing)")
            agg = None; reward_stats = None; sigma_stats = None

        if agg is None:
            t0 = time.time()
            agg, reward_stats, sigma_stats = parallel_runs(
                a1, a2, n_steps=n_steps, n_runs=n_runs,
                analysis_start=analysis_start,
                h_length=m, h_num=h_num,
                init_mode=init_mode, predict_mode=predict_mode,
                n_workers=n_workers)
            elapsed = time.time() - t0
            eta = (total - idx) * elapsed
            # Save IMMEDIATELY after simulation, before fitting.
            with open(durations_path, 'w') as f:
                json.dump({k: list(v) for k, v in agg.items()}, f)
            with open(rewards_path, 'w') as f:
                json.dump(reward_stats, f)
            with open(sigmas_path, 'w') as f:
                json.dump(sigma_stats, f)
            print(f"[{idx}/{total}] {cond_id}  ({elapsed:.1f}s, ETA {eta/60:.1f}m)")

        # ---- Fitting (separate from simulation; failures are tolerated) ----
        row = {'a1': a1, 'a2': a2, 'window': m, 'scale': scale,
               'n_steps': n_steps, 'n_runs': n_runs,
               'init_mode': init_mode, 'predict_mode': predict_mode,
               'h_num': h_num}

        # Reward summary statistics (across runs)
        if reward_stats:
            wrs = [r['win_count'] / r['n_post']
                   for r in reward_stats if r['n_post'] > 0]
            crs = [r['cumR_final'] / r['n_post']
                   for r in reward_stats if r['n_post'] > 0]
            row['win_rate_mean'] = float(np.mean(wrs)) if wrs else np.nan
            row['win_rate_std'] = float(np.std(wrs)) if wrs else np.nan
            row['cumR_per_step_mean'] = float(np.mean(crs)) if crs else np.nan
            row['cumR_per_step_std'] = float(np.std(crs)) if crs else np.nan
            # z-score against chance (1/3)
            if wrs and len(wrs) >= 5:
                from scipy import stats as _stats
                t_stat, p_val = _stats.ttest_1samp(wrs, 1/3)
                row['t_stat_vs_chance'] = float(t_stat)
                row['p_val_vs_chance'] = float(p_val)
            else:
                row['t_stat_vs_chance'] = np.nan
                row['p_val_vs_chance'] = np.nan

        # Sigma summary statistics (across runs)
        if sigma_stats:
            row['sig1_mean'] = float(np.mean([s['sig1_mean']
                                              for s in sigma_stats]))
            row['sig1_std_run'] = float(np.std([s['sig1_mean']
                                                for s in sigma_stats]))
            row['sig2_mean'] = float(np.mean([s['sig2_mean']
                                              for s in sigma_stats]))
            row['sig2_std_run'] = float(np.std([s['sig2_mean']
                                                for s in sigma_stats]))

        for k in metric_keys:
            try:
                fit = fit_distributions(agg[k])
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
                row[f'{k}_tpl_failed'] = fit.get('tpl_fit_failed', False)
            else:
                row[f'{k}_n'] = fit.get('n', 0)
                row[f'{k}_alpha'] = np.nan
                row[f'{k}_best'] = 'error'
                row[f'{k}_w_pl'] = np.nan
                row[f'{k}_w_tpl'] = np.nan
                row[f'{k}_w_exp'] = np.nan
                row[f'{k}_levy'] = False
                row[f'{k}_tpl_failed'] = False
                row[f'{k}_error'] = fit.get('error', 'unknown')
        rows.append(row)

        # Save partial summary after each condition (for crash recovery)
        partial = pd.DataFrame(rows)
        partial.to_csv(out / f'summary_{scale}_partial.csv', index=False)

    summary = pd.DataFrame(rows)
    summary_path = out / f'summary_{scale}.csv'
    summary.to_csv(summary_path, index=False)
    print(f"\nGrid done in {(time.time() - grid_t0)/60:.1f} min")
    print(f"Summary: {summary_path}")

    # Display key columns including reward + sigma summary
    cols_show = ['a1', 'a2', 'window', 'h_num']
    if 'win_rate_mean' in summary.columns:
        cols_show += ['win_rate_mean', 'cumR_per_step_mean']
    if 'sig1_mean' in summary.columns:
        cols_show += ['sig1_mean', 'sig2_mean']
    cols_show += ['T_argmax1_n', 'T_argmax1_alpha', 'T_argmax1_best',
                  'T_argmax1_levy']
    cols_avail = [c for c in cols_show if c in summary.columns]
    print(summary[cols_avail].to_string(index=False))


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=DEFAULT_N_WORKERS)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_pilot = sub.add_parser('pilot', help='Replicate Ibuka 2024 on RPS')
    p_pilot.add_argument('--output', default='./data/reward_pilot')
    p_pilot.add_argument('--full', action='store_true',
                         help='Sweep all 4 (init x predict) designs')
    p_pilot.add_argument('--h_num', type=int, default=10,
                         help='Number of hypotheses (default 10, '
                              'Gunji-style minimum = 3 for RPS)')

    p_grid = sub.add_parser('grid', help='Full window×pair sweep')
    p_grid.add_argument('--scale',
                        choices=['small', 'medium', 'large', 'huge'],
                        default='small')
    p_grid.add_argument('--init', dest='init_mode',
                        choices=['random', 'structured'], default='random')
    p_grid.add_argument('--predict', dest='predict_mode',
                        choices=['sample', 'argmax'], default='sample')
    p_grid.add_argument('--h_num', type=int, default=10,
                        help='Number of hypotheses (default 10). '
                             'For SOC test, sweep N_h ∈ {3, 6, 10, 15, 20}')
    p_grid.add_argument('--output', default='./data/reward_grid')

    args = parser.parse_args()
    if args.cmd == 'pilot':
        run_pilot(args.output, n_workers=args.workers,
                  full_design=args.full, h_num=args.h_num)
    elif args.cmd == 'grid':
        run_grid(args.output, scale=args.scale,
                 init_mode=args.init_mode, predict_mode=args.predict_mode,
                 h_num=args.h_num, n_workers=args.workers)


if __name__ == '__main__':
    main()
