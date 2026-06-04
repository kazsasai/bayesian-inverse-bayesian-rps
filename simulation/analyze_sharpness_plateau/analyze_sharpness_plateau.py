"""
analyze_sharpness_plateau.py — Analyze and visualize results from
                                run_sharpness_plateau.py.

Two analyses:

  (A) PLATEAU DETECTION:
      For each (design, alpha, pair) condition, classify each timestep as:
        - 'plateau': argmax stable AND P(h_argmax) > theta
        - 'transit': otherwise
      Compute plateau-run lengths and compare BIB vs BO.

  (B) SHARPNESS DEPENDENCE:
      For each (design, pair, alpha), fit truncated power law to argmax
      persistence and plot alpha(sharpness).

Outputs:
  figures/fig_plateau_detection.png    — plateau identification visual + stats
  figures/fig_sharpness_dependence.png — alpha vs sharpness alpha (TPL exp)

Usage:
  python3 analyze_sharpness_plateau.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import powerlaw

DATA_DIR = Path('./data/sharpness_plateau')
OUT_DIR = Path('./figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DESIGNS = ['rs', 'ra', 'ss', 'sa']
DESIGN_NAMES = {'rs': 'random + sample', 'ra': 'random + argmax',
                'ss': 'structured + sample', 'sa': 'structured + argmax'}
DESIGN_COLORS = {'rs': '#1f77b4', 'ra': '#ff7f0e',
                 'ss': '#2ca02c', 'sa': '#d62728'}
DESIGN_MARKERS = {'rs': 'o', 'ra': 's', 'ss': '^', 'sa': 'D'}
ALPHAS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ============================================================
# Helpers
# ============================================================
def load_condition(design, alpha, pair):
    fname = f'{design}_a{int(alpha*100):02d}_{pair}.npz'
    path = DATA_DIR / fname
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


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


# ============================================================
# Laminar / burst phase detection (on-off intermittency style)
# ============================================================
def laminar_phases(P, theta=0.2):
    """Return run lengths of laminar phases (max_h P(h) > theta).

    Following the on-off intermittency literature [Platt, Spiegel & Tresser
    1993; Heagy, Platt & Hammel 1994], a laminar phase is a continuous time
    interval during which the system is "near an invariant manifold",
    operationalized here as the time interval during which the maximum
    posterior probability max_h P(h) exceeds threshold theta. The argmax
    identifier may switch within a laminar phase as long as the new argmax
    immediately satisfies P > theta. A laminar phase ends when max_h P(h)
    drops at or below theta (i.e. enters a "burst" of competing-hypothesis
    state).

    Args:
        P:     (T, h_num) float array of posterior trajectories
        theta: threshold for laminar (default 0.2 — well above 1/N_h=0.1)

    Returns:
        laminar_lengths: int array of laminar phase lengths
        burst_lengths:   int array of burst phase lengths
        laminar_frac:    fraction of timesteps in laminar state
    """
    T = P.shape[0]
    p_max = P.max(axis=1)  # (T,)
    is_laminar = p_max > theta  # boolean (T,)

    # Run-length encode boolean array
    # Laminar runs: consecutive True
    # Burst runs:   consecutive False
    if T == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0.0

    diffs = np.diff(is_laminar.astype(np.int8))
    boundaries = np.concatenate([[0], np.flatnonzero(diffs) + 1, [T]])
    run_lengths = np.diff(boundaries)
    # Determine state of each run from the first element
    states = is_laminar[boundaries[:-1]]

    laminar_lengths = run_lengths[states]
    burst_lengths = run_lengths[~states]

    return laminar_lengths, burst_lengths, is_laminar.mean()


# ============================================================
# Plateau detection (argmax-stable AND posterior dominant)
# ============================================================
def plateau_runs(argmax, P, theta=0.3, frac_required=0.8):
    """Return run lengths of segments where argmax is stable AND
    P(h_argmax) > theta for at least frac_required of the run.

    Args:
        argmax: (T,) int array of argmax indices
        P:      (T, h_num) float array of posterior
        theta:  threshold for "dominant" (default 0.3, well above 1/N_h=0.1)
        frac_required: minimum fraction of run with p > theta (default 0.8)

    Returns:
        plateau_runs: int array of run lengths satisfying the plateau condition
        all_runs:     int array of all argmax-stable run lengths
        dominant_frac: fraction of timesteps where P(argmax) > theta
    """
    T = len(argmax)
    p_argmax = P[np.arange(T), argmax]
    is_dominant = p_argmax > theta

    am_runs = consecutive_runs(argmax)
    boundaries = np.concatenate([[0], np.cumsum(am_runs)])

    plateau_lengths = []
    for i in range(len(am_runs)):
        s = boundaries[i]
        e = boundaries[i+1]
        seg = p_argmax[s:e]
        # Plateau if argmax-stable AND posterior dominantly above theta
        if (seg > theta).mean() >= frac_required and (e - s) >= 1:
            plateau_lengths.append(e - s)
    plateau_lengths = np.array(plateau_lengths, dtype=int)

    return plateau_lengths, am_runs, is_dominant.mean()


# ============================================================
# Plateau analysis: BIB vs BO across designs
# ============================================================
def analyze_plateau_stats(theta=0.5):
    """Build statistics for plateau detection across designs and alphas."""
    stats = []
    for design in DESIGNS:
        for alpha in ALPHAS:
            for pair in ['bib-bib', 'bo-bo']:
                d = load_condition(design, alpha, pair)
                if d is None or 'P1' not in d.files:
                    continue
                argmax = d['argmax1']  # (n_runs, T)
                P = d['P1']            # (n_runs, T, h_num)

                all_plateau = []
                all_runs = []
                dom_fracs = []
                for r in range(argmax.shape[0]):
                    pl, ar, df = plateau_runs(argmax[r], P[r], theta=theta)
                    all_plateau.extend(pl.tolist())
                    all_runs.extend(ar.tolist())
                    dom_fracs.append(df)

                stats.append({
                    'design': design, 'alpha': alpha, 'pair': pair,
                    'plateau_lengths': np.array(all_plateau),
                    'argmax_runs': np.array(all_runs),
                    'dominant_frac': np.mean(dom_fracs),
                })
    return stats


def fig_laminar_detection(theta=0.4):
    """Visual demonstration: laminar/burst detection in BIB vs BO."""
    design = 'sa'
    alpha = 0.7
    bib = load_condition(design, alpha, 'bib-bib')
    bo = load_condition(design, alpha, 'bo-bo')

    fig, axes = plt.subplots(2, 2, figsize=(13, 7),
                              sharex=True,
                              gridspec_kw={'height_ratios': [1.5, 1.0],
                                            'hspace': 0.20, 'wspace': 0.18})

    for col, (label, d) in enumerate([('(a)  BIB-BIB', bib),
                                        ('(b)  BO-BO',  bo)]):
        argmax = d['argmax1'][0]
        P = d['P1'][0]
        T = len(argmax)
        t = np.arange(T)
        p_max = P.max(axis=1)

        # Top: max_h P(h) trajectory + threshold + laminar highlights
        ax_top = axes[0, col]
        ax_top.plot(t, p_max, color='black', linewidth=0.8, alpha=0.7)
        ax_top.axhline(theta, color='red', linestyle='--', linewidth=1.2,
                       alpha=0.7, label=f'$\\theta = {theta}$')

        # Highlight laminar runs (continuous max_h P > theta)
        is_laminar = p_max > theta
        # Find boundaries
        if T > 0:
            diffs = np.diff(is_laminar.astype(np.int8))
            boundaries = np.concatenate([[0], np.flatnonzero(diffs) + 1, [T]])
            states = is_laminar[boundaries[:-1]]
            laminar_lengths = []
            for i in range(len(boundaries) - 1):
                s, e = boundaries[i], boundaries[i+1]
                if states[i]:
                    ax_top.axvspan(s, e, alpha=0.25, color='gold',
                                    linewidth=0)
                    laminar_lengths.append(e - s)

            n_laminar = len(laminar_lengths)
            max_laminar = max(laminar_lengths) if laminar_lengths else 0
            total_laminar = sum(laminar_lengths) if laminar_lengths else 0

        ax_top.set_xlim(0, T)
        ax_top.set_ylim(0, 1)
        ax_top.set_title(label, fontsize=13, fontweight='bold', pad=4)
        if col == 0:
            ax_top.set_ylabel(r'$\max_h P(h)$', fontsize=11)
        ax_top.text(0.98, 0.96,
                     f'# laminar: {n_laminar}\n'
                     f'longest: {max_laminar}\n'
                     f'total: {total_laminar} ({total_laminar*100/T:.0f}\\%)',
                     transform=ax_top.transAxes, ha='right', va='top',
                     fontsize=9.5,
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor='gray', linewidth=0.6, alpha=0.92))
        ax_top.legend(loc='upper left', fontsize=9, frameon=False)
        ax_top.grid(True, alpha=0.3)

        # Bottom: argmax index trajectory
        ax_bot = axes[1, col]
        ax_bot.step(t, argmax, where='post', color='red', linewidth=0.7,
                     alpha=0.6)
        ax_bot.set_xlim(0, T)
        ax_bot.set_ylim(-0.5, 9.5)
        ax_bot.set_yticks([0, 5, 9])
        ax_bot.set_xlabel(r'step  $t$', fontsize=10)
        if col == 0:
            ax_bot.set_ylabel('argmax $h$', fontsize=11)
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / 'fig_laminar_detection.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def fig_laminar_length_distribution(theta=0.4):
    """Plot laminar-phase length CCDFs and fit truncated power law.

    Direct test of the on-off intermittency prediction [Platt, Spiegel &
    Tresser 1993]: laminar phase lengths should follow a -3/2 power law.
    """
    lp_by_pair = {'bib-bib': [], 'bo-bo': []}
    lp_by_design_pair = {(d, p): [] for d in DESIGNS
                          for p in ['bib-bib', 'bo-bo']}

    for design in DESIGNS:
        for alpha in ALPHAS:
            for pair in ['bib-bib', 'bo-bo']:
                d = load_condition(design, alpha, pair)
                if d is None or 'P1' not in d.files:
                    continue
                P = d['P1']
                for r in range(P.shape[0]):
                    lp, _, _ = laminar_phases(P[r], theta=theta)
                    lp_by_pair[pair].extend(lp.tolist())
                    lp_by_design_pair[(design, pair)].extend(lp.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    pooled_alphas = {}
    per_design_alphas_all = {}

    for ax_idx, (pair, title) in enumerate([
        ('bib-bib', '(a)  BIB-BIB laminar-phase CCDF'),
        ('bo-bo',   '(b)  BO-BO laminar-phase CCDF'),
    ]):
        ax = axes[ax_idx]
        all_lp = np.array(lp_by_pair[pair], dtype=int)
        all_lp = all_lp[all_lp >= 1]

        a_pooled = np.nan
        if len(all_lp) >= 30:
            try:
                fit = powerlaw.Fit(all_lp, discrete=True, verbose=False)
                a_pooled = float(fit.truncated_power_law.alpha)
                fit.plot_ccdf(ax=ax, color='black', marker='.', linestyle='None',
                              markersize=3, alpha=0.4)
                fit.truncated_power_law.plot_ccdf(
                    ax=ax, color='black', linestyle='-', linewidth=2.0,
                    alpha=0.95)
            except Exception:
                pass
        pooled_alphas[pair] = a_pooled

        per_design = {}
        for design in DESIGNS:
            data = np.array(lp_by_design_pair[(design, pair)], dtype=int)
            data = data[data >= 1]
            if len(data) < 30:
                continue
            try:
                fit = powerlaw.Fit(data, discrete=True, verbose=False)
                a_d = float(fit.truncated_power_law.alpha)
                per_design[design] = a_d
                fit.plot_ccdf(ax=ax, color=DESIGN_COLORS[design],
                              marker='.', linestyle='None', markersize=3,
                              alpha=0.55)
                fit.truncated_power_law.plot_ccdf(
                    ax=ax, color=DESIGN_COLORS[design],
                    linestyle='-', linewidth=1.4, alpha=0.85)
            except Exception:
                pass
        per_design_alphas_all[pair] = per_design

        # On-off intermittency reference line: alpha = 3/2 -> CCDF slope -1/2
        if len(all_lp) >= 30:
            t_ref = np.logspace(0, np.log10(all_lp.max()), 50)
            ccdf_ref = (t_ref / t_ref[0]) ** (-0.5)
            ax.plot(t_ref, ccdf_ref * 0.9, '--', color='gray',
                    linewidth=1.5, alpha=0.7,
                    label=r'on-off intermittency $\alpha = 3/2$')

        # Annotation
        text_lines = [f'pooled: $\\alpha = {a_pooled:.3f}$']
        for d in DESIGNS:
            if d in per_design:
                text_lines.append(
                    f'  {DESIGN_NAMES[d]}: $\\alpha = {per_design[d]:.3f}$')
        ax.text(0.04, 0.04, '\n'.join(text_lines),
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='gray', linewidth=0.6, alpha=0.92))

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(r'laminar phase length  $\tau$', fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel(r'$P(\tau > T)$', fontsize=11)
        ax.set_ylim(1e-4, 1.5)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    out = OUT_DIR / 'fig_laminar_length_dist.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    print()
    print('=== Laminar-phase TPL exponents (theta = {:.2f}) ==='.format(theta))
    for pair in ['bib-bib', 'bo-bo']:
        a = pooled_alphas.get(pair, np.nan)
        print(f'  {pair} pooled: α = {a:.3f}  '
              f'(on-off intermittency reference: 1.5)')
        for d in DESIGNS:
            ad = per_design_alphas_all.get(pair, {}).get(d, np.nan)
            print(f'    {d}: α = {ad:.3f}')

    return pooled_alphas, per_design_alphas_all


def fig_plateau_detection(theta=0.3, frac_required=0.8):
    """Visual demonstration: plateau detection in BIB vs BO."""
    # Show a representative single run for each pair (sa design, alpha=0.7)
    design = 'sa'
    alpha = 0.7
    bib = load_condition(design, alpha, 'bib-bib')
    bo = load_condition(design, alpha, 'bo-bo')

    fig, axes = plt.subplots(2, 2, figsize=(13, 7),
                              sharex=True,
                              gridspec_kw={'height_ratios': [1.5, 1.0],
                                            'hspace': 0.20, 'wspace': 0.18})

    for col, (label, d) in enumerate([('(a)  BIB-BIB', bib),
                                        ('(b)  BO-BO',  bo)]):
        argmax = d['argmax1'][0]
        P = d['P1'][0]
        T = len(argmax)
        t = np.arange(T)
        p_argmax = P[np.arange(T), argmax]

        am_runs = consecutive_runs(argmax)
        boundaries = np.concatenate([[0], np.cumsum(am_runs)])

        # Top: P(h_argmax) trajectory + threshold + plateau highlights
        ax_top = axes[0, col]
        ax_top.plot(t, p_argmax, color='black', linewidth=0.8, alpha=0.7)
        ax_top.axhline(theta, color='red', linestyle='--', linewidth=1.2,
                       alpha=0.7, label=f'$\\theta = {theta}$')

        plateau_segments = []
        for i in range(len(am_runs)):
            s = boundaries[i]
            e = boundaries[i+1]
            seg = p_argmax[s:e]
            if (seg > theta).mean() >= frac_required and (e - s) >= 1:
                ax_top.axvspan(s, e, alpha=0.25, color='gold', linewidth=0)
                plateau_segments.append(e - s)

        ax_top.set_xlim(0, T)
        ax_top.set_ylim(0, 1)
        ax_top.set_title(label, fontsize=13, fontweight='bold', pad=4)
        if col == 0:
            ax_top.set_ylabel(r'$P(h_{\mathrm{argmax}})$', fontsize=11)
        n_plateau = len(plateau_segments)
        max_plateau = max(plateau_segments) if plateau_segments else 0
        total_plateau = sum(plateau_segments) if plateau_segments else 0
        ax_top.text(0.98, 0.96,
                     f'# plateau: {n_plateau}\n'
                     f'longest: {max_plateau}\n'
                     f'total: {total_plateau} ({total_plateau*100/T:.0f}\\%)',
                     transform=ax_top.transAxes, ha='right', va='top',
                     fontsize=9.5,
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor='gray', linewidth=0.6, alpha=0.92))
        ax_top.legend(loc='upper left', fontsize=9, frameon=False)
        ax_top.grid(True, alpha=0.3)

        # Bottom: argmax index trajectory
        ax_bot = axes[1, col]
        ax_bot.step(t, argmax, where='post', color='red', linewidth=0.7,
                     alpha=0.6)
        ax_bot.set_xlim(0, T)
        ax_bot.set_ylim(-0.5, 9.5)
        ax_bot.set_yticks([0, 5, 9])
        ax_bot.set_xlabel(r'step  $t$', fontsize=10)
        if col == 0:
            ax_bot.set_ylabel('argmax $h$', fontsize=11)
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / 'fig_plateau_detection.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def fig_plateau_summary(theta=0.5):
    """Summary plot: plateau frequency and size across designs/alphas."""
    stats = analyze_plateau_stats(theta=theta)

    # Compute summary metrics: mean plateau length, plateau rate
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel (a): plateau time-fraction (sum of plateau lengths / T_total) per condition
    # Panel (b): mean plateau length

    for ax_idx, (metric, ylabel, title) in enumerate([
        ('plateau_time_frac',
         'Plateau time fraction',
         '(a) Time-share of plateau states across designs'),
        ('mean_plateau_length',
         'Mean plateau length',
         '(b) Mean plateau length (when plateau exists)'),
    ]):
        ax = axes[ax_idx]
        for pair, ls, marker_size in [('bib-bib', '-', 9), ('bo-bo', '--', 7)]:
            for design in DESIGNS:
                xs, ys = [], []
                for s in stats:
                    if s['design'] != design or s['pair'] != pair:
                        continue
                    if metric == 'plateau_time_frac':
                        if len(s['argmax_runs']) > 0:
                            tot_plateau = s['plateau_lengths'].sum()
                            tot_time = s['argmax_runs'].sum()
                            ys.append(tot_plateau / tot_time if tot_time > 0 else 0)
                        else:
                            ys.append(0)
                    elif metric == 'mean_plateau_length':
                        if len(s['plateau_lengths']) > 0:
                            ys.append(s['plateau_lengths'].mean())
                        else:
                            ys.append(0)
                    xs.append(s['alpha'])
                if pair == 'bib-bib':
                    label = f'BIB ({design})'
                else:
                    label = f'BO ({design})'
                ax.plot(xs, ys, marker=DESIGN_MARKERS[design],
                         linestyle=ls,
                         color=DESIGN_COLORS[design],
                         markersize=marker_size,
                         linewidth=1.5, alpha=0.8 if pair == 'bib-bib' else 0.5,
                         label=label)
        ax.set_xlabel(r'sharpness $\alpha$', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        if ax_idx == 1:
            ax.set_yscale('log')

    # Combined legend (BIB designs solid, BO dashed)
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h[:4], [f'{DESIGN_NAMES[d]} (BIB)' for d in DESIGNS],
                    fontsize=9, loc='upper left')
    axes[1].legend(h[4:], [f'{DESIGN_NAMES[d]} (BO)' for d in DESIGNS],
                    fontsize=9, loc='upper left')

    plt.tight_layout()
    out = OUT_DIR / 'fig_plateau_summary.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ============================================================
# Plateau LENGTH DISTRIBUTION analysis
# ============================================================
def fig_plateau_length_distribution(theta=0.3, frac_required=0.8):
    """Plot plateau length CCDFs and fit truncated power law.

    The on-off intermittency literature [Platt et al. 1993; Heagy et al. 1994]
    predicts a universal -3/2 power law for laminar phase distributions in
    systems with stochastically driven invariant manifolds. We test whether
    our BIB plateau lengths exhibit this signature.
    """
    # Aggregate plateau lengths across all designs / alphas for BIB and BO
    pl_by_pair = {'bib-bib': [], 'bo-bo': []}
    pl_by_design_pair = {}
    for design in DESIGNS:
        for pair in ['bib-bib', 'bo-bo']:
            key = (design, pair)
            pl_by_design_pair[key] = []
        for alpha in ALPHAS:
            for pair in ['bib-bib', 'bo-bo']:
                d = load_condition(design, alpha, pair)
                if d is None or 'P1' not in d.files:
                    continue
                argmax = d['argmax1']
                P = d['P1']
                for r in range(argmax.shape[0]):
                    pl, _, _ = plateau_runs(argmax[r], P[r],
                                            theta=theta,
                                            frac_required=frac_required)
                    pl_by_pair[pair].extend(pl.tolist())
                    pl_by_design_pair[(design, pair)].extend(pl.tolist())

    # Two-panel plot: (a) BIB-BIB CCDF, (b) BO-BO CCDF
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax_idx, (pair, title) in enumerate([
        ('bib-bib', '(a)  BIB-BIB plateau-length CCDF'),
        ('bo-bo',   '(b)  BO-BO plateau-length CCDF'),
    ]):
        ax = axes[ax_idx]
        # All-design pooled CCDF in black + per-design CCDFs in colors
        all_pl = np.array(pl_by_pair[pair], dtype=int)
        all_pl = all_pl[all_pl >= 1]

        if len(all_pl) >= 30:
            try:
                fit = powerlaw.Fit(all_pl, discrete=True, verbose=False)
                a_pooled = float(fit.truncated_power_law.alpha)
                # Pooled fit overlay
                fit.plot_ccdf(ax=ax, color='black', marker='.', linestyle='None',
                              markersize=3, alpha=0.4, label='_pooled')
                fit.truncated_power_law.plot_ccdf(
                    ax=ax, color='black', linestyle='-', linewidth=2.0,
                    alpha=0.95)
            except Exception as e:
                a_pooled = np.nan
        else:
            a_pooled = np.nan

        # Per-design CCDFs
        per_design_alphas = {}
        for design in DESIGNS:
            data = np.array(pl_by_design_pair[(design, pair)], dtype=int)
            data = data[data >= 1]
            if len(data) < 30:
                continue
            try:
                fit = powerlaw.Fit(data, discrete=True, verbose=False)
                a_d = float(fit.truncated_power_law.alpha)
                per_design_alphas[design] = a_d
                fit.plot_ccdf(ax=ax, color=DESIGN_COLORS[design],
                              marker='.', linestyle='None', markersize=3,
                              alpha=0.55, label='_data')
                fit.truncated_power_law.plot_ccdf(
                    ax=ax, color=DESIGN_COLORS[design],
                    linestyle='-', linewidth=1.4, alpha=0.85)
            except Exception:
                pass

        # On-off intermittency reference: α = 3/2
        # Reference line: P(T>T_0) ~ T^(-(α-1)) for TPL → for α=1.5, slope = -0.5
        # Plot reference -3/2 power law on log-log
        if len(all_pl) >= 30:
            t_ref = np.logspace(0, np.log10(all_pl.max()), 50)
            # CCDF ~ T^(1-α) for power law T^(-α); for α=3/2, CCDF ~ T^(-0.5)
            # Anchor at first data point
            ccdf_ref = (t_ref / t_ref[0]) ** (-0.5)
            # Scale to be visible
            ax.plot(t_ref, ccdf_ref * 0.9, '--', color='gray',
                    linewidth=1.5, alpha=0.7,
                    label=r'on-off intermittency reference $\alpha = 3/2$')

        # Annotation: pooled and per-design alphas
        if not np.isnan(a_pooled):
            text_lines = [f'pooled all-design: $\\alpha = {a_pooled:.3f}$']
            for d in DESIGNS:
                if d in per_design_alphas:
                    text_lines.append(
                        f'  {DESIGN_NAMES[d]}: $\\alpha = {per_design_alphas[d]:.3f}$')
            ax.text(0.04, 0.04, '\n'.join(text_lines),
                    transform=ax.transAxes, ha='left', va='bottom',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                              edgecolor='gray', linewidth=0.6, alpha=0.92))

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(r'plateau length  $T_{\mathrm{pl}}$', fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel(r'$P(T_{\mathrm{pl}} > T)$', fontsize=11)
        ax.set_ylim(1e-4, 1.5)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    out = OUT_DIR / 'fig_plateau_length_dist.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    # Print summary
    print()
    print('=== Plateau-length TPL exponents ===')
    print(f'BIB-BIB pooled: α = {a_pooled:.3f}  (on-off ref: 1.5)')
    print(f'(see figure for per-design values)')


# ============================================================
# Sharpness dependence: alpha (TPL exponent) vs sharpness alpha
# ============================================================
def fit_tpl_alpha(durations):
    if durations is None or len(durations) < 30:
        return np.nan
    try:
        fit = powerlaw.Fit(durations, discrete=True, verbose=False)
        return float(fit.truncated_power_law.alpha)
    except Exception:
        return np.nan


def fig_sharpness_dependence():
    """Plot fitted alpha (TPL exponent) vs sharpness alpha."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax_idx, (pair, title) in enumerate([
        ('bib-bib', '(a)  BIB-BIB: $\\alpha_{\\mathrm{TPL}}$ vs sharpness'),
        ('bo-bo',   '(b)  BO-BO: $\\alpha_{\\mathrm{TPL}}$ vs sharpness'),
    ]):
        ax = axes[ax_idx]
        for design in DESIGNS:
            xs, ys = [], []
            for sharpness in ALPHAS:
                d = load_condition(design, sharpness, pair)
                if d is None:
                    continue
                argmax = d['argmax1']
                # Pool runs: collect all argmax-stable run lengths
                all_dur = []
                for r in range(argmax.shape[0]):
                    runs = consecutive_runs(argmax[r])
                    all_dur.extend(runs.tolist())
                all_dur = np.array(all_dur)
                a_tpl = fit_tpl_alpha(all_dur)
                xs.append(sharpness)
                ys.append(a_tpl)
            ax.plot(xs, ys,
                     marker=DESIGN_MARKERS[design],
                     color=DESIGN_COLORS[design],
                     markersize=10, linewidth=1.7, alpha=0.85,
                     label=DESIGN_NAMES[design])

        if pair == 'bib-bib':
            ax.axhline(1.43, color='gray', linestyle=':', linewidth=1.2,
                        alpha=0.7, label=r'$\alpha = 1.43$ (canonical)')
        ax.set_xlabel(r'$P(d|h)$ sharpness  $\alpha_{\mathrm{init}}$',
                      fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel(r'fitted exponent  $\alpha_{\mathrm{TPL}}$',
                          fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(ALPHAS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9.5, loc='best')

    plt.tight_layout()
    out = OUT_DIR / 'fig_sharpness_dependence.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ============================================================
# Main
# ============================================================
def main():
    print('=== Laminar phase detection (theta = 0.2) ===')
    fig_laminar_detection(theta=0.4)

    print('\n=== Laminar phase length distribution ===')
    fig_laminar_length_distribution(theta=0.4)

    print('\n=== Plateau detection (theta = 0.3, frac_required = 0.8) ===')
    fig_plateau_detection(theta=0.3, frac_required=0.8)

    print('\n=== Plateau summary across designs/alphas ===')
    fig_plateau_summary(theta=0.3)

    print('\n=== Plateau-length distribution ===')
    fig_plateau_length_distribution(theta=0.3, frac_required=0.8)

    print('\n=== Sharpness dependence ===')
    fig_sharpness_dependence()


if __name__ == '__main__':
    main()
