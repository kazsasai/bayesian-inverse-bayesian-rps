"""
Non-stationary Reward-based BIB on RPS with Cycle reversal
===========================================================

Extends rpsgame_reward.py by introducing a time-varying win rule:
  - Normal:   r ≻ s ≻ p ≻ r  (cycle clockwise)
  - Reversed: r ≻ p ≻ s ≻ r  (cycle counterclockwise)

The two rules represent time-reversal of the heteroclinic cycle in the
underlying replicator dynamics, providing a clean non-stationary
environment that retains all symmetry of the RPS game.

Agents do NOT observe the rule directly. They only observe the win/lose
outcome under the *current* rule, and must adapt purely through learning.
This is where BIB is expected to outperform BO.

Key design choices:
  - Reward-based observation (Ibuka 2024 style): kept identical to
    rpsgame_reward.py.
  - Agents do not know rule changes; only outcome changes.
  - Two switching modes:
      * periodic: deterministic switch every T_change steps
      * stochastic: each step switches with probability 1/T_change

Usage:
    # Quick check: BIB vs BO under cycle reversal
    python rpsgame_nonstationary.py pilot

    # Full sweep over T_change values
    python rpsgame_nonstationary.py sweep --output ./data/nonstationary_sweep
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
# Win rule (encoded as set of winning ordered pairs)
# ============================================================

# Normal: r beats s, s beats p, p beats r  (counter-clockwise: rsp)
WINS_NORMAL = frozenset([('r', 's'), ('s', 'p'), ('p', 'r')])

# Reversed: r beats p, p beats s, s beats r  (clockwise: rps)
WINS_REVERSED = frozenset([('r', 'p'), ('p', 's'), ('s', 'r')])


def rps_with_rule(me, rival, wins_set):
    """RPS judgement under the given rule.
    `wins_set` is the set of ordered pairs (me, rival) where 'me' wins."""
    if me == rival:
        return 'quits'
    return 'win' if (me, rival) in wins_set else 'defeat'


# ============================================================
# Reward-based BIB agent (re-implemented here for self-containedness)
# ============================================================

class BayesReward:
    """Same as rpsgame_reward.BayesReward; replicated for self-contained file."""

    def __init__(self, h_num=10, d_num=3, d_type=None, h_length=50,
                 init_mode='random', predict_mode='sample', seed=None):
        self.rng = np.random.default_rng(seed)
        self.h_num = h_num
        self.d_num = d_num
        self.d_type = np.array(d_type if d_type is not None else HANDS)
        self.h_length = h_length
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
        self.history = []

    @staticmethod
    def _structured_init(h_num, d_num):
        if d_num != 3:
            raise NotImplementedError("structured init only for d_num=3")
        templates = np.array([
            [1/3, 1/3, 1/3],
            [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
            [0.50, 0.25, 0.25], [0.25, 0.50, 0.25], [0.25, 0.25, 0.50],
            [0.10, 0.45, 0.45], [0.45, 0.10, 0.45], [0.45, 0.45, 0.10],
        ])
        if h_num <= len(templates):
            return templates[:h_num].copy()
        out = list(templates)
        rng_pad = np.random.default_rng(12345)
        while len(out) < h_num:
            base = templates[len(out) % len(templates)]
            v = np.clip(base + rng_pad.standard_normal(d_num) * 0.05, 1e-3, None)
            out.append(v / v.sum())
        return np.array(out[:h_num])

    def inference(self, data):
        obs_idx = int(np.where(self.d_type == data)[0][0])
        likelihoods = self.likelihood[:, obs_idx]
        marginal = float(np.dot(self.h_prov, likelihoods))
        if marginal <= 0.0:
            return
        post = self.h_prov * likelihoods / marginal
        s = post.sum()
        if s > 0 and np.isfinite(s):
            self.h_prov = post / s
        if (self.h_prov < 0.002).any():
            self.h_prov = self.h_prov * 0.91 + 0.015
            self.h_prov /= self.h_prov.sum()

    def inverse(self):
        if len(self.history) < self.h_length:
            return
        hist = np.zeros(self.d_num)
        for obs in self.history[-self.h_length:]:
            hist[int(np.where(self.d_type == obs)[0][0])] += 1
        hist = hist / self.h_length
        min_h = int(np.argmin(self.h_prov))
        self.likelihood[min_h] = hist
        if (self.likelihood[min_h] < 1e-6).any():
            self.likelihood[min_h] = self.likelihood[min_h] * 0.95 + 0.05/self.d_num

    def update_history(self, observation):
        self.history.append(observation)
        if len(self.history) > self.h_length:
            self.history.pop(0)

    def argmax_h(self):
        return int(np.argmax(self.h_prov))

    def expect(self):
        if self.predict_mode == 'sample':
            h = self.rng.choice(self.h_num, p=self.h_prov)
        elif self.predict_mode == 'argmax':
            h = int(np.argmax(self.h_prov))
        else:
            raise ValueError(f"Unknown predict_mode: {self.predict_mode}")
        return self.rng.choice(self.d_type, p=self.likelihood[h])


class AgentReward:
    """Reward-based BIB agent. Crucially, the agent NEVER directly knows the
    rule — it only observes win/defeat outcomes from `update_from_outcome()`."""

    def __init__(self, a_type, h_length=50,
                 init_mode='random', predict_mode='sample', seed=None):
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

    def update_from_outcome(self, my_hand, opp_hand, current_wins):
        """Reward-based update under the *current* (possibly varying) rule."""
        if self.bayes is None:
            return
        result = rps_with_rule(my_hand, opp_hand, current_wins)
        if result == 'quits':
            return
        if result == 'win':
            kansoku = my_hand
        else:  # defeat
            others = [h for h in HANDS if h != my_hand]
            kansoku = self.rng.choice(others)
        self.bayes.inference(kansoku)
        self.bayes.update_history(kansoku)
        if self.type == 'bib':
            self.bayes.inverse()

    def argmax_h(self):
        return -1 if self.bayes is None else self.bayes.argmax_h()


# ============================================================
# Rule schedule
# ============================================================

def make_rule_schedule(n_steps, T_change, mode='periodic', seed=None):
    """Return an array of length n_steps where each entry is the active
    win-set at that step.

    mode: 'periodic'   -> deterministic switch every T_change steps
          'stochastic' -> each step independently switches with prob 1/T_change
          'static'     -> never switches (control condition)
    """
    if mode == 'static':
        return [WINS_NORMAL] * n_steps
    rng = np.random.default_rng(seed)
    rule_seq = [WINS_NORMAL] * n_steps
    if mode == 'periodic':
        current = WINS_NORMAL
        for t in range(n_steps):
            if t > 0 and t % T_change == 0:
                current = WINS_REVERSED if current == WINS_NORMAL else WINS_NORMAL
            rule_seq[t] = current
    elif mode == 'stochastic':
        current = WINS_NORMAL
        p_switch = 1.0 / T_change if T_change > 0 else 0.0
        for t in range(n_steps):
            if t > 0 and rng.random() < p_switch:
                current = WINS_REVERSED if current == WINS_NORMAL else WINS_NORMAL
            rule_seq[t] = current
    else:
        raise ValueError(f"Unknown rule mode: {mode}")
    return rule_seq


# ============================================================
# Pair simulation under non-stationary rule
# ============================================================

def run_pair(a1, a2, n_steps,
             T_change=1000, switch_mode='periodic',
             h_length=50, init_mode='random', predict_mode='sample',
             seed=None):
    """Run a single pair-simulation with time-varying win rule.

    Returns DataFrame with all step-by-step state including the active rule.
    """
    seed1 = seed
    seed2 = (seed + 100000) if seed is not None else None
    seed_rule = (seed + 200000) if seed is not None else None

    ag1 = AgentReward(a1, h_length, init_mode, predict_mode, seed=seed1)
    ag2 = AgentReward(a2, h_length, init_mode, predict_mode, seed=seed2)
    rule_seq = make_rule_schedule(n_steps, T_change, switch_mode, seed=seed_rule)

    h1_arr = np.empty(n_steps, dtype='<U1')
    h2_arr = np.empty(n_steps, dtype='<U1')
    res_arr = np.empty(n_steps, dtype='<U10')
    am1_arr = np.empty(n_steps, dtype=np.int32)
    am2_arr = np.empty(n_steps, dtype=np.int32)
    rule_arr = np.empty(n_steps, dtype=np.int8)  # 0=normal, 1=reversed

    for t in range(n_steps):
        h1 = ag1.choice()
        h2 = ag2.choice()
        am1_arr[t] = ag1.argmax_h()
        am2_arr[t] = ag2.argmax_h()
        h1_arr[t] = h1
        h2_arr[t] = h2

        current_wins = rule_seq[t]
        rule_arr[t] = 0 if current_wins == WINS_NORMAL else 1
        res_arr[t] = rps_with_rule(h1, h2, current_wins)

        ag1.update_from_outcome(h1, h2, current_wins)
        # opponent sees mirrored game: same rule, but their hand is h2,
        # opponent is h1 — pass current_wins (which judges from agent-1's
        # perspective). For agent 2 we need to invert the result.
        # rps_with_rule(h2, h1, current_wins) gives ag2's outcome correctly
        # since the rule is symmetric under swap as long as we pass (me, rival).
        ag2.update_from_outcome(h2, h1, current_wins)

    return pd.DataFrame({
        't': np.arange(n_steps),
        'h1': h1_arr, 'h2': h2_arr, 'res': res_arr,
        'argmax1': am1_arr, 'argmax2': am2_arr,
        'rule': rule_arr,
        'match': h1_arr == h2_arr,
    })


# ============================================================
# Performance analysis: how fast does each agent adapt?
# ============================================================

def analyze_adaptation(df, T_change, switch_mode):
    """Analyze adaptation speed after rule changes.

    Returns dict with:
      - win_rate_overall: agent1's overall win rate
      - win_rate_pre_window: win rate in last K steps before each switch
      - win_rate_post_window: win rate in first K steps after each switch
      - argmax_change_after_switch: time until argmax_h changes after switch
    """
    K = 50  # window size for pre/post analysis
    res = df['res'].values
    rule = df['rule'].values
    am1 = df['argmax1'].values
    n = len(df)

    is_win = (res == 'win').astype(np.float32)
    win_rate_overall = float(is_win.mean())

    # Identify switch points (where rule changes)
    rule_changes = np.flatnonzero(np.diff(rule) != 0) + 1

    pre_rates = []
    post_rates = []
    adapt_times = []
    for t_switch in rule_changes:
        if t_switch >= K and t_switch + K < n:
            pre = is_win[t_switch - K:t_switch].mean()
            post = is_win[t_switch:t_switch + K].mean()
            pre_rates.append(pre)
            post_rates.append(post)
            # Time until argmax_h changes after switch
            am_at_switch = am1[t_switch]
            relative = am1[t_switch:t_switch + min(K * 4, n - t_switch)]
            changed = np.flatnonzero(relative != am_at_switch)
            adapt_times.append(int(changed[0]) if len(changed) > 0 else -1)

    return {
        'win_rate_overall': win_rate_overall,
        'win_rate_pre_mean': float(np.mean(pre_rates)) if pre_rates else np.nan,
        'win_rate_post_mean': float(np.mean(post_rates)) if post_rates else np.nan,
        'adapt_time_median': float(np.median(adapt_times)) if adapt_times else np.nan,
        'adapt_time_mean': float(np.mean([t for t in adapt_times if t >= 0]))
                           if adapt_times else np.nan,
        'n_rule_changes': len(rule_changes),
    }


# ============================================================
# Metrics & fitting (same as rpsgame_reward.py)
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
    if len(np.unique(data)) < 5:
        return {'error': 'degenerate_distribution', 'n': int(len(data))}
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

    ll_pl = _ll(fit.power_law); ll_tpl = _ll(fit.truncated_power_law)
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
        'w_pl': float(weights['pl']), 'w_tpl': float(weights['tpl']),
        'w_exp': float(weights['exp']),
        'best': best,
        'alpha_best': alpha_best,
        'levy_region': bool(1.0 < alpha_best <= 3.0),
    }


# ============================================================
# Plotting: time series with rule changes marked
# ============================================================

def plot_time_series(df, title='', save_path=None, max_points=5000):
    """Plot win rate (running mean) and argmax_h over time, with rule
    transitions marked. This is the key visualization for showing BIB
    adaptation."""
    n = len(df)
    if n > max_points:
        # Subsample for plotting clarity
        idx = np.linspace(0, n - 1, max_points).astype(int)
        sub = df.iloc[idx]
    else:
        sub = df

    # Running win rate (sliding window)
    is_win = (df['res'].values == 'win').astype(float)
    K = max(50, n // 200)
    cs = np.cumsum(np.insert(is_win, 0, 0))  # length n+1
    # running_winrate[t] = mean of is_win[t-K+1 ... t]  for t >= K-1
    running_winrate = np.full(n, np.nan)
    # Use indices that produce equal-length slices
    if n >= K:
        running_winrate[K - 1:] = (cs[K:] - cs[:-K]) / K

    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    # Top: running win rate
    axs[0].plot(np.arange(n), running_winrate, color='C0', linewidth=0.7)
    axs[0].axhline(0.333, color='gray', linestyle=':', linewidth=0.8,
                   label='chance (1/3)')
    axs[0].set_ylabel('win rate (window=200)')
    axs[0].set_ylim(0, 1)
    axs[0].legend(loc='upper right', fontsize=8)

    # Middle: argmax_h
    axs[1].plot(sub['t'], sub['argmax1'], color='C1', linewidth=0.4)
    axs[1].set_ylabel('argmax_h (agent 1)')
    axs[1].set_ylim(-0.5, 9.5)

    # Bottom: rule indicator (filled regions)
    rule = df['rule'].values
    rule_changes = np.flatnonzero(np.diff(rule) != 0) + 1
    boundaries = np.concatenate([[0], rule_changes, [n]])
    for i in range(len(boundaries) - 1):
        t0, t1 = boundaries[i], boundaries[i + 1]
        if rule[t0] == 0:
            axs[2].axhspan(0, 1, xmin=t0/n, xmax=t1/n,
                           color='lightblue', alpha=0.5)
        else:
            axs[2].axhspan(0, 1, xmin=t0/n, xmax=t1/n,
                           color='lightyellow', alpha=0.5)
    axs[2].set_ylim(0, 1); axs[2].set_yticks([])
    axs[2].set_xlabel('step t')
    axs[2].set_ylabel('rule')

    # Mark rule changes on all panels
    for t_change in rule_changes:
        for ax in axs[:2]:
            ax.axvline(t_change, color='red', linewidth=0.3, alpha=0.5)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_cdf(durations, title='', save_path=None, ymin_padding=0.5):
    data = np.asarray(durations); data = data[data >= 1]
    if len(data) < 30 or len(np.unique(data)) < 5:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = powerlaw.Fit(data, discrete=True, verbose=False)
    n_above = max(len(data[data >= fit.xmin]), 1)
    y_lower = (1.0 / n_above) * ymin_padding
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
    ax.set_ylim(y_lower, 1.5)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


# ============================================================
# Parallel runs
# ============================================================

def _run_and_extract(args):
    (a1, a2, n_steps, T_change, switch_mode, analysis_start,
     h_length, init_mode, predict_mode, seed) = args
    df = run_pair(a1, a2, n_steps, T_change=T_change, switch_mode=switch_mode,
                  h_length=h_length, init_mode=init_mode,
                  predict_mode=predict_mode, seed=seed)
    df_ana = df.iloc[analysis_start:]
    metrics = extract_metrics(df_ana)
    adapt = analyze_adaptation(df, T_change, switch_mode)
    return {
        'metrics': {k: v.tolist() for k, v in metrics.items()},
        'adapt': adapt,
    }


def parallel_runs(a1, a2, n_steps, n_runs, analysis_start,
                  T_change=1000, switch_mode='periodic',
                  h_length=50, init_mode='random', predict_mode='sample',
                  n_workers=None, seed_base=0):
    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS
    metric_keys = ['T_argmax1', 'T_argmax2', 'hand1_runs',
                   'hand2_runs', 'result_runs', 'match_runs']
    agg_metrics = {k: [] for k in metric_keys}
    agg_adapt = []
    job_args = [
        (a1, a2, n_steps, T_change, switch_mode, analysis_start,
         h_length, init_mode, predict_mode, seed_base + run)
        for run in range(n_runs)
    ]
    if n_workers <= 1:
        for ja in job_args:
            res = _run_and_extract(ja)
            for k in metric_keys:
                agg_metrics[k].extend(res['metrics'][k])
            agg_adapt.append(res['adapt'])
    else:
        with Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(_run_and_extract, job_args, chunksize=1):
                for k in metric_keys:
                    agg_metrics[k].extend(res['metrics'][k])
                agg_adapt.append(res['adapt'])
    return agg_metrics, agg_adapt


# ============================================================
# Pilot: BIB vs BO under fixed T_change=1000
# ============================================================

def run_pilot(output_dir='./data/nonstationary_pilot',
              T_change=1000, switch_mode='periodic',
              n_workers=None):
    """Compare BIB and BO under a single T_change condition.

    Saves:
    - One example time-series plot per condition (showing rule changes)
    - CDF plot of T_argmax1
    - JSON of duration distributions and adaptation summary
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS

    n_steps = 10000
    analysis_start = 0  # use full sequence for time-varying analysis
    n_runs = 50

    pairs = [('bib', 'bib'), ('bo', 'bo'), ('bib', 'bo'),
             ('bib', 'random'), ('bo', 'random')]

    print("=" * 78)
    print(f"Non-stationary RPS pilot (cycle reversal, {switch_mode}, "
          f"T_change={T_change})")
    print(f"  T={n_steps} × {n_runs} runs, m=50, random+sample")
    print(f"  Workers: {n_workers}")
    print("=" * 78)
    print(f"\n{'pair':16s} {'win_rate':>9s} {'adapt(med)':>11s} "
          f"{'n_dur':>7s} {'α':>6s} {'best':>5s} {'levy':>5s}")
    print("-" * 70)

    summary_rows = []

    # Also run one representative example for time-series plotting
    for a1, a2 in pairs:
        cond_id = f"{a1}-{a2}_{switch_mode}_T{T_change}"
        t0 = time.time()
        agg_metrics, agg_adapt = parallel_runs(
            a1, a2, n_steps=n_steps, n_runs=n_runs,
            analysis_start=analysis_start,
            T_change=T_change, switch_mode=switch_mode,
            h_length=50, init_mode='random', predict_mode='sample',
            n_workers=n_workers)
        elapsed = time.time() - t0

        # Aggregate adaptation stats across runs
        win_rates = [a['win_rate_overall'] for a in agg_adapt]
        adapt_meds = [a['adapt_time_median'] for a in agg_adapt
                      if not np.isnan(a['adapt_time_median'])]

        fit = fit_distributions(agg_metrics['T_argmax1'])
        alpha = fit.get('alpha_best', np.nan)
        best = fit.get('best', 'err')
        levy = '✓' if fit.get('levy_region', False) else '✗'
        n_dur = fit.get('n', 0)

        print(f"{cond_id[:16]:16s} {np.mean(win_rates):>9.3f} "
              f"{np.median(adapt_meds) if adapt_meds else np.nan:>11.1f} "
              f"{n_dur:>7d} {alpha:>6.2f} {best:>5s} {levy:>5s}  ({elapsed:.0f}s)")

        # Save data
        with open(out / f'durations_{cond_id}.json', 'w') as f:
            json.dump({k: list(v) for k, v in agg_metrics.items()}, f)
        with open(out / f'adapt_{cond_id}.json', 'w') as f:
            json.dump(agg_adapt, f)

        # Plot CDF
        plot_cdf(agg_metrics['T_argmax1'],
                 title=f'{cond_id} (T_argmax1)',
                 save_path=out / f'cdf_{cond_id}.png')

        # Plot one example time-series (seed=0 run)
        df_example = run_pair(a1, a2, n_steps,
                              T_change=T_change, switch_mode=switch_mode,
                              h_length=50, init_mode='random',
                              predict_mode='sample', seed=0)
        plot_time_series(df_example,
                         title=f'{cond_id} (example, seed=0)',
                         save_path=out / f'timeseries_{cond_id}.png')

        summary_rows.append({
            'pair': f'{a1}-{a2}',
            'switch_mode': switch_mode,
            'T_change': T_change,
            'win_rate_mean': float(np.mean(win_rates)),
            'win_rate_std': float(np.std(win_rates)),
            'adapt_median': float(np.median(adapt_meds)) if adapt_meds else np.nan,
            'n_duration': n_dur,
            'alpha_best': float(alpha) if not np.isnan(alpha) else np.nan,
            'best_model': best,
            'levy_region': bool(fit.get('levy_region', False)),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / f'summary_{switch_mode}_T{T_change}.csv', index=False)
    print(f"\nResults: {out.resolve()}")


# ============================================================
# Sweep: vary T_change to map out the m vs T_change relationship
# ============================================================

def run_sweep(output_dir='./data/nonstationary_sweep',
              switch_mode='periodic',
              n_workers=None):
    """Sweep T_change values to study how adaptation scales.

    Particularly interesting: T_change near m=50 (history window),
    where BIB's memory and environment dynamics interact strongly.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS

    T_changes = [50, 100, 200, 500, 1000, 2000, 5000]
    pairs = [('bib', 'bib'), ('bo', 'bo')]
    n_steps = 10000
    n_runs = 50

    print(f"Sweep: T_change in {T_changes}, switch={switch_mode}")
    print(f"  T={n_steps} × {n_runs} runs, m=50")

    rows = []
    for T_change, (a1, a2) in product(T_changes, pairs):
        t0 = time.time()
        agg_metrics, agg_adapt = parallel_runs(
            a1, a2, n_steps=n_steps, n_runs=n_runs,
            analysis_start=0, T_change=T_change, switch_mode=switch_mode,
            n_workers=n_workers)
        elapsed = time.time() - t0
        wr = [a['win_rate_overall'] for a in agg_adapt]
        ad = [a['adapt_time_median'] for a in agg_adapt
              if not np.isnan(a['adapt_time_median'])]
        fit = fit_distributions(agg_metrics['T_argmax1'])
        print(f"  T_change={T_change:5d} {a1}-{a2}: "
              f"win={np.mean(wr):.3f} adapt={np.median(ad) if ad else np.nan:.1f} "
              f"α={fit.get('alpha_best', np.nan):.2f} ({elapsed:.0f}s)")

        rows.append({
            'T_change': T_change, 'pair': f'{a1}-{a2}',
            'switch_mode': switch_mode,
            'win_rate_mean': float(np.mean(wr)),
            'adapt_median': float(np.median(ad)) if ad else np.nan,
            'alpha': float(fit.get('alpha_best', np.nan)),
            'best': fit.get('best', 'err'),
            'levy': bool(fit.get('levy_region', False)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / f'sweep_{switch_mode}.csv', index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nResults: {out.resolve()}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=DEFAULT_N_WORKERS)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_pilot = sub.add_parser('pilot', help='BIB vs BO at fixed T_change')
    p_pilot.add_argument('--output', default='./data/nonstationary_pilot')
    p_pilot.add_argument('--T_change', type=int, default=1000)
    p_pilot.add_argument('--switch_mode',
                         choices=['periodic', 'stochastic', 'static'],
                         default='periodic')

    p_sweep = sub.add_parser('sweep', help='Sweep T_change values')
    p_sweep.add_argument('--output', default='./data/nonstationary_sweep')
    p_sweep.add_argument('--switch_mode',
                         choices=['periodic', 'stochastic'],
                         default='periodic')

    args = parser.parse_args()
    if args.cmd == 'pilot':
        run_pilot(args.output,
                  T_change=args.T_change, switch_mode=args.switch_mode,
                  n_workers=args.workers)
    elif args.cmd == 'sweep':
        run_sweep(args.output, switch_mode=args.switch_mode,
                  n_workers=args.workers)


if __name__ == '__main__':
    main()
