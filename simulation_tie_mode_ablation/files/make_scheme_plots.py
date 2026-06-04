"""Generate 4-scheme comparison summary and plots.

Usage:
    python make_scheme_plots.py             # default: small scale
    python make_scheme_plots.py --scale medium
    python make_scheme_plots.py --scale huge
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import powerlaw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rpsgame_reward_tie as sim

_parser = argparse.ArgumentParser()
_parser.add_argument('--scale',
                     choices=['small', 'medium', 'large', 'huge'],
                     default='small')
_args = _parser.parse_args()
SCALE = _args.scale
DATA = HERE.parent / 'data' / 'scheme_ablation' / SCALE
N_STEPS_TAG = {'small': 'T=2000×100', 'medium': 'T=10000×200',
               'large': 'T=50000×50', 'huge': 'T=200000×20'}.get(SCALE, SCALE)
print(f'Scale: {SCALE}  ({N_STEPS_TAG})')
print(f'Data:  {DATA}')
SCHEMES = {'eq3':('skip','random_other'), 'caseA':('uniform','random_other'),
           'hybrid':('uniform','opponent'), 'defeatGT':('skip','opponent')}
DESIGNS = ['rs','ra','ss','sa']
PAIRS = ['bib-bib', 'bo-bo']
COLORS = {'eq3':'#1f4e79', 'caseA':'#c00000', 'hybrid':'#548235',
          'defeatGT':'#7030a0'}

# --- Build tidy summary (with fit cache) ---
FIT_CACHE = DATA / '_fit_cache.json'
fit_cache = {}
if FIT_CACHE.exists():
    try:
        fit_cache = json.load(open(FIT_CACHE))
        print(f'Loaded fit cache: {len(fit_cache)} entries')
    except Exception:
        fit_cache = {}

rows = []
for design in DESIGNS:
    for pair in PAIRS:
        for scheme, (tie, defeat) in SCHEMES.items():
            cond = f'{design}_{pair}_{scheme}_tie{tie}_def{defeat}'
            dpath = DATA / f'durations_{cond}.json'
            if not dpath.exists():
                continue
            with open(dpath) as f: agg = json.load(f)
            with open(DATA / f'rewards_{cond}.json') as f: rstats = json.load(f)
            with open(DATA / f'sigmas_{cond}.json') as f: sstats = json.load(f)
            wrs = [r['win_count']/r['n_post'] for r in rstats if r['n_post']>0]
            qrs = [r['quits_count']/r['n_post'] for r in rstats if r['n_post']>0]
            sigs = [s['sig1_mean'] for s in sstats]
            # Fit cache keyed by cond_id
            if cond in fit_cache:
                fit = fit_cache[cond]
            else:
                print(f'  fitting {cond} ...')
                fit = sim.fit_distributions(agg['T_argmax1'])
                # Strip non-serializable values
                fit_cache[cond] = {k: (v if not isinstance(v, np.bool_) else bool(v))
                                   for k, v in fit.items()}
                with open(FIT_CACHE, 'w') as f:
                    json.dump(fit_cache, f, default=str)
            rows.append(dict(design=design, pair=pair, scheme=scheme,
                tie_mode=tie, defeat_mode=defeat,
                alpha=fit.get('alpha_best', np.nan),
                best=fit.get('best','error'),
                w_pl=fit.get('w_pl', np.nan),
                w_tpl=fit.get('w_tpl', np.nan),
                w_exp=fit.get('w_exp', np.nan),
                win_rate=float(np.mean(wrs)) if wrs else np.nan,
                quits_rate=float(np.mean(qrs)) if qrs else np.nan,
                sigma_post=float(np.mean(sigs)) if sigs else np.nan,
                n=fit.get('n', 0)))

df = pd.DataFrame(rows)
# Restrict SCHEMES/PAIRS to those actually present in this run's data.
SCHEMES = {k: v for k, v in SCHEMES.items() if k in set(df['scheme'])}
PAIRS = [p for p in PAIRS if p in set(df['pair'])]
df.to_csv(DATA / 'scheme_summary.csv', index=False)

# --- Pretty print ---
print('=== α(T_argmax1) by pair × design × scheme ===')
for pair in PAIRS:
    pivot = df[df['pair']==pair].pivot(index='design', columns='scheme', values='alpha')
    pivot = pivot.reindex(DESIGNS)[list(SCHEMES)]
    pivot.loc['mean'] = pivot.mean()
    pivot.loc['spread (max-min)'] = pivot.iloc[:4].max() - pivot.iloc[:4].min()
    print(f'\n[{pair}]')
    print(pivot.to_string(float_format='%.3f'))

print('\n=== win_rate × scheme (BIB-BIB) ===')
pivot = df[df['pair']=='bib-bib'].pivot(index='design', columns='scheme', values='win_rate')
print(pivot.reindex(DESIGNS)[list(SCHEMES)].to_string(float_format='%.4f'))

print('\n=== quits_rate × scheme (BIB-BIB) ===')
pivot = df[df['pair']=='bib-bib'].pivot(index='design', columns='scheme', values='quits_rate')
print(pivot.reindex(DESIGNS)[list(SCHEMES)].to_string(float_format='%.4f'))

# --- Plot 1: alpha bar chart, 4 schemes side by side ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, pair in zip(axes, PAIRS):
    sub = df[df['pair']==pair]
    x = np.arange(len(DESIGNS))
    w = 0.20
    for j, scheme in enumerate(SCHEMES):
        sub2 = sub[sub['scheme']==scheme].set_index('design').reindex(DESIGNS)
        ax.bar(x + (j - (len(SCHEMES)-1)/2)*w, sub2['alpha'].values, width=w,
               color=COLORS[scheme], label=scheme,
               edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(DESIGNS)
    ax.set_ylabel('α (T_argmax1, best fit)')
    ax.set_title(f'{pair.upper()}: α across designs × schemes')
    ax.axhspan(1.0, 3.0, color='gold', alpha=0.10)
    ax.axhline(1.0, color='gray', lw=0.6, ls='--')
    ax.axhline(3.0, color='gray', lw=0.6, ls='--')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 3.2)
fig.suptitle(f'Scheme ablation: α(T_argmax1) — {SCALE} scale ({N_STEPS_TAG})\n'
             'eq3 = Eq.(3), caseA = tie→uniform, hybrid = caseA + defeat→opp, '
             'defeatGT = defeat→opp only',
             fontsize=11, y=1.04)
fig.tight_layout()
fig.savefig(DATA / 'fig_alpha_4schemes.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# --- Plot 2: 4-panel BIB-BIB CCDFs ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
for ax, design in zip(axes.flat, DESIGNS):
    for scheme, (tie, defeat) in SCHEMES.items():
        cond = f'{design}_bib-bib_{scheme}_tie{tie}_def{defeat}'
        with open(DATA / f'durations_{cond}.json') as f:
            agg = json.load(f)
        data = np.asarray(agg['T_argmax1'])
        data = data[data >= 1]
        if len(data) < 30 or len(np.unique(data)) < 5: continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = powerlaw.Fit(data, discrete=True, verbose=False)
            fit.plot_ccdf(ax=ax, color=COLORS[scheme], marker='o',
                          markersize=3, linestyle='None',
                          label=f'{scheme}  α={fit.power_law.alpha:.2f}')
    ax.set_xlabel('T_argmax1'); ax.set_ylabel('P(X ≥ T)')
    ax.set_title(f'BIB-BIB, design = {design}')
    ax.legend(loc='lower left', fontsize=9); ax.grid(alpha=0.3)
fig.suptitle(f'Scheme ablation: BIB-BIB CCDF — {SCALE} scale ({N_STEPS_TAG})',
             fontsize=12, y=1.00)
fig.tight_layout()
fig.savefig(DATA / 'fig_BIB_ccdf_4schemes.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# --- Plot 3: spread per scheme (universality check) ---
fig, ax = plt.subplots(figsize=(9, 4.8))
for j, scheme in enumerate(SCHEMES):
    sub = df[(df['pair']=='bib-bib') & (df['scheme']==scheme)]
    alphas = sub.set_index('design').reindex(DESIGNS)['alpha'].values
    ax.scatter([scheme]*len(alphas), alphas, color=COLORS[scheme],
               s=110, edgecolor='black', linewidth=0.8, zorder=3,
               label=f'spread={alphas.max()-alphas.min():.3f}')
    ax.scatter([scheme], [np.mean(alphas)], color='black', marker='_',
               s=500, linewidth=2.4, zorder=4)
ax.axhspan(1.0, 3.0, color='gold', alpha=0.10)
ax.axhline(1.0, color='gray', lw=0.6, ls='--')
ax.set_ylabel('α(T_argmax1) — one point per design')
ax.set_title(f'Universality check (BIB-BIB): α spread across 4 designs ({SCALE})')
ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=9, loc='upper left')
fig.tight_layout()
fig.savefig(DATA / 'fig_spread_4schemes.png', dpi=140, bbox_inches='tight')
plt.close(fig)

print('\nWrote:')
for p in DATA.glob('fig_*4schemes*.png'):
    print(f'  {p}')
print(f'  {DATA / "scheme_summary.csv"}')
print(f'  {DATA / "fig_spread_4schemes.png"}')
