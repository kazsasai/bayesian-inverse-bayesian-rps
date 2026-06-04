"""Generate BIB-BIB CCDF comparison and α-bar-chart from cached small-scale data."""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import powerlaw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rpsgame_reward_tie as sim

DATA = HERE.parent / 'data' / 'ablation' / 'small'
DESIGNS = ['rs', 'ra', 'ss', 'sa']
COLORS = {'skip': '#1f4e79', 'uniform': '#c00000'}

# --- Plot 1: 4-panel CCDF (one per design), BIB-BIB only, both tie modes ---
fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
for ax, design in zip(axes.flat, DESIGNS):
    for tie in ['skip', 'uniform']:
        cond = f'{design}_bib-bib_tie{tie}'
        with open(DATA / f'durations_{cond}.json') as f:
            agg = json.load(f)
        data = np.asarray(agg['T_argmax1'])
        data = data[data >= 1]
        if len(data) < 30 or len(np.unique(data)) < 5:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = powerlaw.Fit(data, discrete=True, verbose=False)
            fit.plot_ccdf(ax=ax, color=COLORS[tie], marker='o',
                          markersize=3, linestyle='None',
                          label=f'tie={tie}  α={fit.power_law.alpha:.2f}  n={len(data)}')
    ax.set_xlabel('T_argmax1')
    ax.set_ylabel('P(X ≥ T)')
    ax.set_title(f'BIB-BIB, design = {design}')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle('Tie-mode ablation (case A): BIB argmax persistence CCDF — small scale '
             '(T=2000 × 100 runs)', fontsize=12, y=1.00)
fig.tight_layout()
fig.savefig(DATA / 'fig_BIB_ccdf_4panel.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'Wrote: {DATA / "fig_BIB_ccdf_4panel.png"}')

# --- Plot 2: α bar chart, BIB-BIB + BO-BO ---
quick = pd.read_csv(DATA / 'quick_summary.csv')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, pair in zip(axes, ['bib-bib', 'bo-bo']):
    sub = quick[quick['pair'] == pair]
    x = np.arange(len(DESIGNS))
    w = 0.36
    for j, tie in enumerate(['skip', 'uniform']):
        sub2 = sub[sub['tie_mode'] == tie].set_index('design').reindex(DESIGNS)
        ax.bar(x + (j - 0.5) * w, sub2['alpha'].values, width=w,
               color=COLORS[tie], label=f'tie={tie}',
               edgecolor='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(DESIGNS)
    ax.set_ylabel('α (T_argmax1, best fit)')
    ax.set_title(f'{pair.upper()}: α across designs × tie modes')
    ax.axhspan(1.0, 3.0, color='gold', alpha=0.10, label='Lévy region')
    ax.axhline(1.0, color='gray', lw=0.6, ls='--')
    ax.axhline(3.0, color='gray', lw=0.6, ls='--')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('Tie-mode ablation: α(T_argmax1) — small scale pilot',
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(DATA / 'fig_alpha_bars.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'Wrote: {DATA / "fig_alpha_bars.png"}')

# --- Plot 3: spread (universality) visualization ---
fig, ax = plt.subplots(figsize=(7, 4.5))
for j, pair in enumerate(['bib-bib', 'bo-bo']):
    sub = quick[quick['pair'] == pair]
    for tie in ['skip', 'uniform']:
        sub2 = sub[sub['tie_mode'] == tie]
        alphas = sub2['alpha'].values
        ax.scatter([f'{pair}\n{tie}'] * len(alphas), alphas,
                   color=COLORS[tie], s=90, edgecolor='black',
                   linewidth=0.8, alpha=0.85, zorder=3)
        ax.scatter([f'{pair}\n{tie}'], [np.mean(alphas)],
                   color='black', marker='_', s=400, linewidth=2.4,
                   zorder=4)
ax.axhspan(1.0, 3.0, color='gold', alpha=0.10)
ax.axhline(1.0, color='gray', lw=0.6, ls='--')
ax.set_ylabel('α (T_argmax1, best fit)')
ax.set_title('Spread of α across 4 designs — universality check\n'
             '(tight cluster = universal; wide = design-dependent)')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(DATA / 'fig_alpha_spread.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'Wrote: {DATA / "fig_alpha_spread.png"}')

print('Done.')
