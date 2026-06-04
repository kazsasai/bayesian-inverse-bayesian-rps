"""
N-hand cyclic-dominance Rock-Paper-Scissors with reward-based BIB.

Generalizes the standard 3-hand RPS to N=3, 5, 7, 9, ... where each hand
beats exactly k = (N-1)/2 other hands in a cyclic dominance pattern.

Examples:
  N=3 (standard RPS):     hand i beats hand (i+1) mod 3
  N=5 (RPS-Lizard-Spock): hand i beats hands (i+1) and (i+2) mod 5
  N=7:                    hand i beats hands (i+1), (i+2), (i+3) mod 7

Each hand wins against k hands and loses against k hands,
maintaining a symmetric cyclic structure.

The state space is the (N-1)-simplex. The interior corresponds to
'all hands present' (draw), vertices to 'single hand' (draw), and faces
of dimension d correspond to '(d+1) hands present' configurations. In
particular, 1-faces (edges) where exactly 2 hands are present yield
deterministic outcomes — this is the boundary-driven dynamics aspect.

Reward-based observation (Ibuka 2024 style):
  win   -> kansoku = my_hand
  quits -> no update
  defeat -> kansoku = uniformly random from {hands} - {my_hand}

Usage:
    python rpsgame_nhand.py pilot --N 3       # standard RPS
    python rpsgame_nhand.py pilot --N 5       # RPSLS
    python rpsgame_nhand.py pilot --N 7
    python rpsgame_nhand.py grid --N 5 --scale small
    python rpsgame_nhand.py grid --N 5 --scale huge
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

DEFAULT_N_WORKERS = max(1, cpu_count() - 1)


# ============================================================
# N-hand cyclic dominance ruleset
# ============================================================

def make_wins(N, k=None):
    """Build win set for N-hand cyclic RPS.

    Hand i beats hands (i+1), (i+2), ..., (i+k) mod N.
    Default k = (N-1)//2 gives symmetric structure (each hand wins
    against and loses to the same number of hands).

    Returns:
      wins: set of (winner, loser) pairs of integer hand indices.
    """
    if k is None:
        k = (N - 1) // 2
    if not (1 <= k <= N - 1):
        raise ValueError(f"k must be in [1, N-1], got k={k}")
    if 2 * k >= N and N % 2 == 1:
        # For symmetric structure, need 2k+1 = N (each hand wins k, loses k, ties 0)
        pass  # allow asymmetric configurations
    wins = set()
    for i in range(N):
        for j in range(1, k + 1):
            wins.add((i, (i + j) % N))
    return frozenset(wins)


def hand_labels(N):
    """Return display labels for hands. For N=3 use traditional, else integers."""
    if N == 3:
        return ['r', 'p', 's']
    if N == 5:
        return ['r', 'p', 's', 'l', 'k']  # rock, paper, scissors, lizard, spock
    return [f'h{i}' for i in range(N)]


def rps_n(me, rival, wins_set):
    """N-hand RPS judgement. me, rival are integer hand indices."""
    if me == rival:
        return 'quits'
    return 'win' if (me, rival) in wins_set else 'defeat'


# ============================================================
# Reward-based BIB agent (N-hand version)
# ============================================================

class BayesRewardN:
    """Reward-based BIB agent for N-hand RPS.

    Hypotheses h_k (k=0..h_num-1) are categorical distributions over N hands.
    Observation d in {0, 1, ..., N-1} is constructed from game outcome.
    """

    def __init__(self, h_num=10, N=3, h_length=50,
                 init_mode='random', predict_mode='sample', seed=None):
        self.rng = np.random.default_rng(seed)
        self.h_num = h_num
        self.N = N  # number of hands
        self.h_length = h_length
        self.init_mode = init_mode
        self.predict_mode = predict_mode

        if init_mode == 'random':
            self.likelihood = np.abs(self.rng.standard_normal((h_num, N)))
            self.likelihood /= self.likelihood.sum(axis=1, keepdims=True)
        elif init_mode == 'structured':
            self.likelihood = self._structured_init(h_num, N)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        self.h_prov = np.ones(h_num) / h_num
        self.history = []  # ints in 0..N-1

    @staticmethod
    def _structured_init(h_num, N):
        """Structured init: uniform + N hand-favoring + N anti-hand templates.

        For arbitrary N, generate templates with varying degrees of
        concentration on specific hands.
        """
        templates = []
        # 1: uniform
        templates.append(np.ones(N) / N)
        # N: strong concentration on each hand (0.7 on one, 0.3/(N-1) elsewhere)
        for i in range(N):
            v = np.full(N, 0.3 / (N - 1)) if N > 1 else np.array([1.0])
            v[i] = 0.7
            templates.append(v)
        # N: mild concentration (0.5 on one, 0.5/(N-1) elsewhere)
        for i in range(N):
            v = np.full(N, 0.5 / (N - 1)) if N > 1 else np.array([1.0])
            v[i] = 0.5
            templates.append(v)
        # N: anti-hand (0.1 on one, 0.9/(N-1) elsewhere)
        for i in range(N):
            v = np.full(N, 0.9 / (N - 1)) if N > 1 else np.array([1.0])
            v[i] = 0.1
            templates.append(v)
        templates = np.array(templates)  # shape (1+3N, N)

        if h_num <= len(templates):
            return templates[:h_num].copy()
        # Pad with noisy versions if h_num > 1+3N
        out = list(templates)
        rng_pad = np.random.default_rng(12345)
        while len(out) < h_num:
            base = templates[len(out) % len(templates)]
            v = np.clip(base + rng_pad.standard_normal(N) * 0.05, 1e-3, None)
            out.append(v / v.sum())
        return np.array(out[:h_num])

    def inference(self, data_idx):
        """Standard Bayes update; data_idx is integer hand index."""
        likelihoods = self.likelihood[:, data_idx]
        marginal = float(np.dot(self.h_prov, likelihoods))
        if marginal <= 0.0:
            return
        post = self.h_prov * likelihoods / marginal
        s = post.sum()
        if s > 0 and np.isfinite(s):
            self.h_prov = post / s
        # Smoothing to prevent collapse
        if (self.h_prov < 0.002).any():
            self.h_prov = self.h_prov * 0.91 + 0.015
            self.h_prov /= self.h_prov.sum()

    def inverse(self):
        """Replace lowest-posterior hypothesis with empirical histogram."""
        if len(self.history) < self.h_length:
            return
        hist = np.zeros(self.N)
        for obs_idx in self.history[-self.h_length:]:
            hist[obs_idx] += 1
        hist = hist / self.h_length
        min_h = int(np.argmin(self.h_prov))
        self.likelihood[min_h] = hist
        if (self.likelihood[min_h] < 1e-6).any():
            self.likelihood[min_h] = self.likelihood[min_h] * 0.95 + 0.05 / self.N

    def update_history(self, observation_idx):
        self.history.append(int(observation_idx))
        if len(self.history) > self.h_length:
            self.history.pop(0)

    def argmax_h(self):
        return int(np.argmax(self.h_prov))

    def expect(self):
        """Sample h ~ P(h), then d ~ P(d|h). Return integer hand index."""
        if self.predict_mode == 'sample':
            h = self.rng.choice(self.h_num, p=self.h_prov)
        elif self.predict_mode == 'argmax':
            h = int(np.argmax(self.h_prov))
        else:
            raise ValueError(f"Unknown predict_mode: {self.predict_mode}")
        return int(self.rng.choice(self.N, p=self.likelihood[h]))


class AgentRewardN:
    """N-hand RPS agent with reward-based BIB."""

    def __init__(self, a_type, N=3, h_length=50,
                 init_mode='random', predict_mode='sample', seed=None):
        self.type = a_type
        self.N = N
        self.rng = np.random.default_rng(seed)
        if a_type == 'random':
            self.bayes = None
        else:
            self.bayes = BayesRewardN(
                h_num=10, N=N, h_length=h_length,
                init_mode=init_mode, predict_mode=predict_mode, seed=seed)

    def choice(self):
        if self.type == 'random':
            return int(self.rng.integers(0, self.N))
        return self.bayes.expect()

    def update_from_outcome(self, my_hand, opp_hand, wins_set):
        if self.bayes is None:
            return
        result = rps_n(my_hand, opp_hand, wins_set)
        if result == 'quits':
            return
        if result == 'win':
            kansoku = my_hand
        else:  # defeat
            others = [i for i in range(self.N) if i != my_hand]
            kansoku = int(self.rng.choice(others))
        self.bayes.inference(kansoku)
        self.bayes.update_history(kansoku)
        if self.type == 'bib':
            self.bayes.inverse()

    def argmax_h(self):
        return -1 if self.bayes is None else self.bayes.argmax_h()


# ============================================================
# Pair simulation
# ============================================================

def run_pair(a1, a2, n_steps, N=3, k=None,
             h_length=50, init_mode='random', predict_mode='sample',
             seed=None):
    if k is None:
        k = (N - 1) // 2
    wins_set = make_wins(N, k)
    seed1 = seed
    seed2 = (seed + 100000) if seed is not None else None
    ag1 = AgentRewardN(a1, N, h_length, init_mode, predict_mode, seed=seed1)
    ag2 = AgentRewardN(a2, N, h_length, init_mode, predict_mode, seed=seed2)

    h1_arr = np.empty(n_steps, dtype=np.int32)
    h2_arr = np.empty(n_steps, dtype=np.int32)
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
        res_arr[t] = rps_n(h1, h2, wins_set)
        ag1.update_from_outcome(h1, h2, wins_set)
        ag2.update_from_outcome(h2, h1, wins_set)

    return pd.DataFrame({
        't': np.arange(n_steps),
        'h1': h1_arr, 'h2': h2_arr, 'res': res_arr,
        'argmax1': am1_arr, 'argmax2': am2_arr,
        'match': h1_arr == h2_arr,
    })


# ============================================================
# Metrics & fitting (reused from rpsgame_reward)
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
    arr = np.asarray(boolean_array, dtype=bool)
    if not arr.any():
        return np.array([], dtype=int)
    diffs = np.diff(arr.astype(np.int8))
    starts = np.flatnonzero(diffs == 1) + 1
    ends = np.flatnonzero(diffs == -1) + 1
    if arr[0]: starts = np.r_[0, starts]
    if arr[-1]: ends = np.r_[ends, len(arr)]
    return ends - starts


def extract_metrics(df):
    res = df['res'].values
    return {
        'T_argmax1':         consecutive_runs(df['argmax1'].values),
        'T_argmax2':         consecutive_runs(df['argmax2'].values),
        'hand1_runs':        consecutive_runs(df['h1'].values),
        'hand2_runs':        consecutive_runs(df['h2'].values),
        'result_runs':       consecutive_runs(res),
        'match_runs':        consecutive_runs(df['match'].values),
        'win_runs':          streak_lengths(res == 'win'),
        'win_or_draw_runs':  streak_lengths(res != 'defeat'),
        'defeat_runs':       streak_lengths(res == 'defeat'),
    }


def fit_distributions(data, discrete=True):
    data = np.asarray(data); data = data[data >= 1]
    if len(data) < 30:
        return {'error': 'too_few_samples', 'n': int(len(data))}
    if len(np.unique(data)) < 5:
        return {'error': 'degenerate_distribution', 'n': int(len(data))}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = powerlaw.Fit(data, discrete=discrete, verbose=False)
    except Exception as e:
        return {'error': str(e)[:50], 'n': int(len(data))}
    data_fit = data[data >= fit.xmin]
    if len(data_fit) < 10:
        return {'error': 'few_above_xmin', 'n': int(len(data))}

    def safe(name):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return getattr(fit, name)
        except Exception:
            return None

    def ll(d):
        if d is None: return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p = np.asarray(d.pdf(data_fit), dtype=float)
            p = np.where(p > 0, p, 1e-300)
            return float(np.sum(np.log(p)))
        except Exception:
            return None

    pl = safe('power_law'); tpl = safe('truncated_power_law'); ex = safe('exponential')
    ll_pl = ll(pl); ll_tpl = ll(tpl); ll_ex = ll(ex)
    avail = {}
    if ll_pl is not None: avail['pl'] = ll_pl
    if ll_tpl is not None: avail['tpl'] = ll_tpl
    if ll_ex is not None: avail['exp'] = ll_ex
    if not avail:
        return {'error': 'all_failed', 'n': int(len(data))}
    nparam = {'pl':1,'exp':1,'tpl':2}
    aic = {k: 2*nparam[k] - 2*v for k, v in avail.items()}
    am = min(aic.values())
    raw = {k: np.exp(-(v-am)/2) for k, v in aic.items()}
    tot = sum(raw.values())
    w = {k: v/tot for k, v in raw.items()}
    for k in ('pl','tpl','exp'): w.setdefault(k, 0.0)
    best = max(w, key=w.get)
    a_pl = float(pl.alpha) if pl is not None else np.nan
    a_tpl = float(tpl.alpha) if tpl is not None else np.nan
    if best == 'pl' and not np.isnan(a_pl): a_best = a_pl
    elif best == 'tpl' and not np.isnan(a_tpl): a_best = a_tpl
    elif not np.isnan(a_tpl): a_best = a_tpl
    elif not np.isnan(a_pl): a_best = a_pl
    else: a_best = np.nan
    return {
        'n': int(len(data)), 'xmin': float(fit.xmin),
        'alpha_pl': a_pl, 'alpha_tpl': a_tpl,
        'Lambda_tpl': float(getattr(tpl,'Lambda',np.nan)) if tpl is not None else np.nan,
        'lambda_exp': float(getattr(ex,'Lambda',np.nan)) if ex is not None else np.nan,
        'w_pl': w['pl'], 'w_tpl': w['tpl'], 'w_exp': w['exp'],
        'best': best,
        'alpha_best': float(a_best) if not np.isnan(a_best) else np.nan,
        'levy_region': bool(1.0 < a_best <= 3.0) if not np.isnan(a_best) else False,
    }


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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit.plot_ccdf(ax=ax, color='purple', marker='x', linestyle='None',
                          markersize=4, label='data')
    except Exception:
        plt.close(fig); return None

    def _try(name, color, ls, lw, fn):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                d = getattr(fit, name)
                d.plot_ccdf(ax=ax, color=color, linestyle=ls, linewidth=lw, label=fn(d))
        except Exception:
            pass

    _try('power_law', 'blue', '-', 2, lambda d: f'PL α={d.alpha:.2f}')
    _try('truncated_power_law', 'green', ':', 2, lambda d: f'TPL α={d.alpha:.2f}')
    _try('exponential', 'red', '--', 2,
         lambda d: f'exp λ={getattr(d,"Lambda",d.parameter1):.3f}')

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
    (a1, a2, n_steps, analysis_start, N, k,
     h_length, init_mode, predict_mode, seed) = args
    df = run_pair(a1, a2, n_steps, N=N, k=k,
                  h_length=h_length, init_mode=init_mode,
                  predict_mode=predict_mode, seed=seed)
    df_ana = df.iloc[analysis_start:]
    metrics = extract_metrics(df_ana)
    return {k_: v.tolist() for k_, v in metrics.items()}


def parallel_runs(a1, a2, n_steps, n_runs, analysis_start,
                  N=3, k=None, h_length=50,
                  init_mode='random', predict_mode='sample',
                  n_workers=None, seed_base=0):
    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS
    metric_keys = ['T_argmax1', 'T_argmax2', 'hand1_runs', 'hand2_runs',
                   'result_runs', 'match_runs',
                   'win_runs', 'win_or_draw_runs', 'defeat_runs']
    agg = {kk: [] for kk in metric_keys}
    job_args = [
        (a1, a2, n_steps, analysis_start, N, k,
         h_length, init_mode, predict_mode, seed_base + run)
        for run in range(n_runs)
    ]
    if n_workers <= 1:
        for ja in job_args:
            res = _run_and_extract(ja)
            for kk in metric_keys:
                agg[kk].extend(res[kk])
    else:
        with Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(_run_and_extract, job_args, chunksize=1):
                for kk in metric_keys:
                    agg[kk].extend(res[kk])
    return agg


# ============================================================
# Pilot run
# ============================================================

def run_pilot(output_dir, N=3, k=None, n_workers=None):
    """Pilot for one N value, all 5 pair conditions × 4 designs."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS
    if k is None:
        k = (N - 1) // 2

    pairs = [('bib','bib'), ('bo','bo'), ('bib','bo'),
             ('bib','random'), ('bo','random')]
    designs = [(im, pm) for im in ['random','structured']
                        for pm in ['sample','argmax']]

    print("=" * 78)
    print(f"N-hand RPS pilot: N={N}, k={k}")
    print(f"  T=2000 × 100 runs, m=50, all 4 designs × 5 pairs = 20 conditions")
    print(f"  Workers: {n_workers}")
    print("=" * 78)
    print(f"\n{'design':12s} {'pair':16s} {'n':>6s} {'best':>5s} {'α':>6s} {'levy':>5s}")
    print("-" * 60)

    rows = []
    for init_mode, predict_mode in designs:
        design_tag = f"{init_mode[0]}+{predict_mode[0]}"
        for a1, a2 in pairs:
            cond_id = f"N{N}_{init_mode[0]}{predict_mode[0]}_{a1}-{a2}"
            t0 = time.time()
            agg = parallel_runs(a1, a2, n_steps=2000, n_runs=100,
                                analysis_start=1000, N=N, k=k,
                                h_length=50, init_mode=init_mode,
                                predict_mode=predict_mode, n_workers=n_workers)
            elapsed = time.time() - t0
            fit = fit_distributions(agg['T_argmax1'])
            if 'error' not in fit:
                levy = '✓' if fit['levy_region'] else '✗'
                print(f"{design_tag:12s} {a1+'-'+a2:16s} {fit['n']:6d} "
                      f"{fit['best']:>5s} {fit['alpha_best']:6.2f} {levy:>5s}  "
                      f"({elapsed:.0f}s)")
                rows.append({
                    'N': N, 'k': k, 'init_mode': init_mode,
                    'predict_mode': predict_mode, 'a1': a1, 'a2': a2,
                    'n': fit['n'], 'alpha_pl': fit['alpha_pl'],
                    'alpha_tpl': fit['alpha_tpl'],
                    'alpha_best': fit['alpha_best'], 'best': fit['best'],
                    'w_pl': fit['w_pl'], 'w_tpl': fit['w_tpl'],
                    'w_exp': fit['w_exp'], 'levy': fit['levy_region'],
                })
            else:
                print(f"{design_tag:12s} {a1+'-'+a2:16s}  error: {fit.get('error')}")
                rows.append({
                    'N': N, 'k': k, 'init_mode': init_mode,
                    'predict_mode': predict_mode, 'a1': a1, 'a2': a2,
                    'n': fit.get('n', 0), 'best': 'error',
                })
            with open(out / f'durations_{cond_id}.json', 'w') as f:
                json.dump({kk: list(v) for kk, v in agg.items()}, f)
            plot_cdf(agg['T_argmax1'], title=f'N={N}: {cond_id}',
                     save_path=out / f'cdf_{cond_id}.png')

    df = pd.DataFrame(rows)
    df.to_csv(out / f'pilot_summary_N{N}.csv', index=False)
    print(f"\nSummary: {out / f'pilot_summary_N{N}.csv'}")


# ============================================================
# Grid run with resume support
# ============================================================

def run_grid(output_dir, N=3, k=None, scale='small',
             init_mode='random', predict_mode='sample', n_workers=None):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = DEFAULT_N_WORKERS
    if k is None:
        k = (N - 1) // 2

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

    pairs = [('bib','bib'), ('bo','bo'), ('random','random'),
             ('bib','bo'), ('bib','random'), ('bo','random')]
    windows = [10, 20, 50, 100]
    metric_keys = ['T_argmax1', 'T_argmax2', 'hand1_runs', 'hand2_runs',
                   'result_runs', 'match_runs',
                   'win_runs', 'win_or_draw_runs', 'defeat_runs']

    rows = []
    total = len(pairs) * len(windows)
    idx = 0

    print(f"N-hand grid: N={N}, k={k}, scale={scale}, "
          f"init={init_mode}, predict={predict_mode}, workers={n_workers}")
    print(f"  T={n_steps}, n_runs={n_runs}, total conditions={total}")

    grid_t0 = time.time()
    for (a1, a2), m in product(pairs, windows):
        idx += 1
        cond_id = f"N{N}_{a1}-{a2}_m{m}_{scale}"
        durations_path = out / f'durations_{cond_id}.json'

        if durations_path.exists():
            try:
                with open(durations_path) as f:
                    agg = json.load(f)
                if all(kk in agg for kk in metric_keys):
                    print(f"[{idx}/{total}] {cond_id}  (cached)")
                    elapsed = 0.0
                else:
                    raise ValueError("incomplete cache")
            except Exception as e:
                print(f"[{idx}/{total}] {cond_id}  (cache invalid: {e}, re-running)")
                agg = None
        else:
            agg = None

        if agg is None:
            t0 = time.time()
            agg = parallel_runs(a1, a2, n_steps=n_steps, n_runs=n_runs,
                                analysis_start=analysis_start,
                                N=N, k=k, h_length=m,
                                init_mode=init_mode, predict_mode=predict_mode,
                                n_workers=n_workers)
            elapsed = time.time() - t0
            eta = (total - idx) * elapsed
            with open(durations_path, 'w') as f:
                json.dump({kk: list(v) for kk, v in agg.items()}, f)
            print(f"[{idx}/{total}] {cond_id}  ({elapsed:.1f}s, ETA {eta/60:.1f}m)")

        row = {'N': N, 'k': k, 'a1': a1, 'a2': a2, 'window': m, 'scale': scale,
               'n_steps': n_steps, 'n_runs': n_runs,
               'init_mode': init_mode, 'predict_mode': predict_mode}
        for kk in metric_keys:
            try:
                fit = fit_distributions(agg[kk])
            except Exception as e:
                fit = {'error': f'fit_exception: {str(e)[:50]}'}
            if 'error' not in fit:
                row[f'{kk}_n'] = fit['n']
                row[f'{kk}_alpha_pl'] = fit['alpha_pl']
                row[f'{kk}_alpha_tpl'] = fit['alpha_tpl']
                row[f'{kk}_alpha_best'] = fit['alpha_best']
                row[f'{kk}_best'] = fit['best']
                row[f'{kk}_w_pl'] = fit['w_pl']
                row[f'{kk}_w_tpl'] = fit['w_tpl']
                row[f'{kk}_w_exp'] = fit['w_exp']
                row[f'{kk}_levy'] = fit['levy_region']
            else:
                row[f'{kk}_n'] = fit.get('n', 0)
                for c in ['alpha_pl','alpha_tpl','alpha_best','w_pl','w_tpl','w_exp']:
                    row[f'{kk}_{c}'] = np.nan
                row[f'{kk}_best'] = 'error'
                row[f'{kk}_levy'] = False
                row[f'{kk}_error'] = fit.get('error', 'unknown')
        rows.append(row)
        partial = pd.DataFrame(rows)
        partial.to_csv(out / f'summary_N{N}_{scale}_partial.csv', index=False)

    summary = pd.DataFrame(rows)
    summary_path = out / f'summary_N{N}_{scale}.csv'
    summary.to_csv(summary_path, index=False)
    print(f"\nGrid done in {(time.time() - grid_t0)/60:.1f} min")
    print(f"Summary: {summary_path}")
    print(summary[['a1','a2','window','T_argmax1_n',
                   'T_argmax1_alpha_tpl','T_argmax1_best',
                   'T_argmax1_levy']].to_string(index=False))


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=DEFAULT_N_WORKERS)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_pilot = sub.add_parser('pilot', help='Pilot for one N value')
    p_pilot.add_argument('--N', type=int, default=3,
                         help='Number of hands (3, 5, 7, ...)')
    p_pilot.add_argument('--k', type=int, default=None,
                         help='Number of hands each hand beats '
                              '(default: (N-1)//2)')
    p_pilot.add_argument('--output', default=None)

    p_grid = sub.add_parser('grid', help='Grid for one N')
    p_grid.add_argument('--N', type=int, default=3)
    p_grid.add_argument('--k', type=int, default=None)
    p_grid.add_argument('--scale', choices=['small','medium','large','huge'],
                        default='small')
    p_grid.add_argument('--init', dest='init_mode',
                        choices=['random','structured'], default='random')
    p_grid.add_argument('--predict', dest='predict_mode',
                        choices=['sample','argmax'], default='sample')
    p_grid.add_argument('--output', default=None)

    args = parser.parse_args()
    if args.cmd == 'pilot':
        out = args.output or f'./data/nhand_pilot_N{args.N}'
        run_pilot(out, N=args.N, k=args.k, n_workers=args.workers)
    elif args.cmd == 'grid':
        out = args.output or (f'./data/nhand_grid_N{args.N}_'
                              f'{args.init_mode[0]}{args.predict_mode[0]}')
        run_grid(out, N=args.N, k=args.k, scale=args.scale,
                 init_mode=args.init_mode, predict_mode=args.predict_mode,
                 n_workers=args.workers)


if __name__ == '__main__':
    main()
