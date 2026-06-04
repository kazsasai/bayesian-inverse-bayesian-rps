"""Rebuild Figure S5 (fig_plateau.pdf): plateau-length CCDFs
for BIB-BIB and BO-BO, with per-design + pooled truncated-power-law fits.

Regenerated with the current `powerlaw` so the figure's exponents match the
body text. Panel labels follow the house style: lower-case (a)/(b) at the
top-left, no descriptive panel title.

Data: sharpness-sweep posteriors (4 designs x 6 sharpness x {bib-bib,bo-bo}),
plateau = maximal argmax-stable run with P(h*)>theta for >= frac of the run
(theta=0.3, frac=0.8); agent 1 only, pooled across the 6 sharpness values.
"""
from pathlib import Path
import sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import powerlaw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figdata
OUT = figdata.FIG_DIR / 'fig_plateau.pdf'

DESIGNS = ['rs', 'ra', 'ss', 'sa']
DESIGN_NAMES = {'rs': 'random + sample', 'ra': 'random + argmax',
                'ss': 'structured + sample', 'sa': 'structured + argmax'}
DESIGN_COLORS = {'rs': '#1f77b4', 'ra': '#ff7f0e',
                 'ss': '#2ca02c', 'sa': '#d62728'}
DESIGN_MARKERS = {'rs': 'o', 'ra': 's', 'ss': '^', 'sa': 'D'}
ALPHAS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
THETA, FRAC = 0.3, 0.8


def find_data_dir():
    """Sharpness-sweep data via figdata ($PAPERA_DATA / <repo>/data / in-repo),
    falling back to the bundled copy."""
    rel = 'simulation/analyze_sharpness_plateau/data/sharpness_plateau'
    if figdata.exists(rel):
        return figdata.find(rel)
    return figdata.bundled('sharpness_plateau')


DATA = find_data_dir()


def consec_runs(a):
    if a.size == 0:
        return np.array([], dtype=int)
    idx = np.flatnonzero(np.diff(a)) + 1
    b = np.concatenate([[0], idx, [a.size]])
    return np.diff(b)


def plateau_lengths(argmax, P, theta=THETA, frac=FRAC):
    p_argmax = P[np.arange(P.shape[0]), argmax]
    runs = consec_runs(argmax)
    b = np.concatenate([[0], np.cumsum(runs)])
    out = []
    for i in range(len(runs)):
        seg = p_argmax[b[i]:b[i + 1]]
        if (seg > theta).mean() >= frac and (b[i + 1] - b[i]) >= 1:
            out.append(int(b[i + 1] - b[i]))
    return out


def collect():
    by_pair = {'bib-bib': [], 'bo-bo': []}
    by_design_pair = {(d, p): [] for d in DESIGNS for p in ('bib-bib', 'bo-bo')}
    for design in DESIGNS:
        for pair in ('bib-bib', 'bo-bo'):
            for s in ALPHAS:
                f = DATA / f'{design}_a{int(s * 100):02d}_{pair}.npz'
                if not f.exists():
                    continue
                z = np.load(f)
                argmax, P = z['argmax1'], z['P1']
                for r in range(argmax.shape[0]):
                    pl = plateau_lengths(argmax[r], P[r])
                    by_pair[pair].extend(pl)
                    by_design_pair[(design, pair)].extend(pl)
    return by_pair, by_design_pair


def main():
    warnings.simplefilter('ignore')
    by_pair, by_design_pair = collect()
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":6,"axes.linewidth":0.8,"lines.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","xtick.major.size":3,"ytick.major.size":3,"xtick.minor.size":1.8,"ytick.minor.size":1.8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})  # PNAS-unified
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05))
    panel_tag = {0: 'A', 1: 'B'}

    for ax_idx, pair in enumerate(['bib-bib', 'bo-bo']):
        ax = axes[ax_idx]
        all_pl = np.array(by_pair[pair], dtype=int)
        all_pl = all_pl[all_pl >= 1]

        a_pooled = np.nan
        if all_pl.size >= 30:
            fit = powerlaw.Fit(all_pl, discrete=True, verbose=False)
            a_pooled = float(fit.truncated_power_law.alpha)
            # pooled data points (small, neutral) + bold fit line w/ legend entry
            fit.plot_ccdf(ax=ax, color='0.35', marker='.', linestyle='None',
                          markersize=3, alpha=0.30)
            fit.truncated_power_law.plot_ccdf(
                ax=ax, color='black', linestyle='-', linewidth=2.2, alpha=0.95,
                label=fr'pooled: $\alpha={a_pooled:.2f}$')

        per_design = {}
        for d in DESIGNS:
            data = np.array(by_design_pair[(d, pair)], dtype=int)
            data = data[data >= 1]
            if data.size < 30:
                continue
            fit = powerlaw.Fit(data, discrete=True, verbose=False)
            per_design[d] = float(fit.truncated_power_law.alpha)
            # larger, distinctly-shaped markers (thinned) so the four designs
            # are separable by both colour AND shape; fit line carries legend
            fit.plot_ccdf(ax=ax, color=DESIGN_COLORS[d],
                          marker=DESIGN_MARKERS[d], linestyle='None',
                          markersize=5.5, markevery=0.10, alpha=0.75,
                          markeredgecolor='black', markeredgewidth=0.4)
            fit.truncated_power_law.plot_ccdf(
                ax=ax, color=DESIGN_COLORS[d], linestyle='-', linewidth=1.6,
                alpha=0.9,
                label=fr'{d}: $\alpha={per_design[d]:.2f}$')

        if all_pl.size >= 30:
            t_ref = np.logspace(0, np.log10(all_pl.max()), 50)
            ccdf_ref = (t_ref / t_ref[0]) ** (-0.5)
            ax.plot(t_ref, ccdf_ref * 0.9, '--', color='gray', linewidth=1.5,
                    alpha=0.7,
                    label=r'on-off ref $\alpha=3/2$')

        # House-style panel label: bold lower-case tag, top-left just above
        # the axes (matches build_FigS1/FigS6/FigS7 panel_label), no title
        ax.text(-0.17, 1.04, panel_tag[ax_idx], transform=ax.transAxes,
                ha='left', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xlabel(r'plateau length  $T_{\mathrm{pl}}$', fontsize=7.5)
        if ax_idx == 0:
            ax.set_ylabel(r'$P(T_{\mathrm{pl}} > T)$', fontsize=7.5)
        ax.set_ylim(1e-4, 1.5)
        ax.grid(False)
        # legend now carries colour+marker -> design (with per-design alpha),
        # pooled, and the on-off reference in one place (no separate text box)
        ax.legend(fontsize=6.5, loc='lower left', frameon=False)
        print(f'{pair}: pooled={a_pooled:.3f}  per-design='
              f'{ {d: round(per_design[d], 3) for d in per_design} }', flush=True)

    plt.tight_layout()
    fig.savefig(OUT, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
