#!/usr/bin/env python3
"""
regenerate_fig2_fig3cd_real.py
==============================

Regenerates two figures for the PRE paper with REAL simulator output
(replacing synthetic placeholders):

    fig_dynamics_demo.png     -- BIB-BIB and BO-BO dynamics at T=1500
    Fig3_universality_4panel_PROD.png
        (a, b)  reused from existing JSON durations (grid_huge / reward_huge_v2)
        (c, d)  laminar phase length CCDFs from a fresh 8-condition x 5-seed
                simulator sweep (BIB-BIB, BO-BO across rs/ra/ss/sa designs)

Imports the simulator from
    /sessions/.../BIB_Analyze/simulation/reward_huge_v2/rpsgame_reward.py
via sys.path injection (also resolves the host path for local testing).

Outputs / caches written to the figures/ directory:
    fig_dynamics_demo.png        (atomic os.replace)
    Fig3_universality_4panel_PROD.png  (atomic os.replace)
    _laminar_cache.npz                 (pooled laminar lengths per condition)
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths -- auto-detect sandbox vs host
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
FIG_DIR = _HERE.parent                      # .../BIB_Levy_v2/latex/figures
REPO = _HERE.parents[3]                     # figures -> latex -> BIB_Levy_v2 -> <repo>

# Data root: $PAPERA_DATA, else the repo itself (same policy as figdata.py).
DATA_ROOT = Path(os.environ.get("PAPERA_DATA", REPO))

# Simulator path: always the version-controlled engine inside the repo.
SIM_DIR = REPO / "simulation" / "reward_huge_v2"
sys.path.insert(0, str(SIM_DIR))

from rpsgame_reward import AgentReward, rps  # noqa: E402

print(f"DATA_ROOT = {DATA_ROOT}")
print(f"FIG_DIR   = {FIG_DIR}")
print(f"SIM_DIR   = {SIM_DIR}")

# ---------------------------------------------------------------------------
# rcParams (match regenerate_figures.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.6,
    "lines.markersize": 5.0,
    "font.family": "DejaVu Sans",
    "mathtext.default": "regular",
})

# ---------------------------------------------------------------------------
# Constants (mirror regenerate_figures.py)
# ---------------------------------------------------------------------------
DESIGN_DIRS = {
    "rs": ("grid_huge",              "reward_huge/data/reward_huge_v2_rs"),
    "ra": ("grid_huge_rand_argmax",  "reward_huge/data/reward_huge_v2_ra"),
    "ss": ("grid_huge_struct_sample","reward_huge/data/reward_huge_v2_ss"),
    "sa": ("grid_huge_struct_argmax","reward_huge/data/reward_huge_v2_sa"),
}
DESIGN_LABEL = {
    "rs": "rs (rand+sample)",
    "ra": "ra (rand+argmax)",
    "ss": "ss (struct+sample)",
    "sa": "sa (struct+argmax)",
}
DESIGN_COLOR = {
    "rs": "#1f77b4",
    "ra": "#ff7f0e",
    "ss": "#2ca02c",
    "sa": "#d62728",
}
ALPHA_ARG_BIB = {"rs": 1.43, "ra": 1.43, "ss": 1.45, "sa": 1.41}
ALPHA_ARG_BO  = {"rs": 3.42, "ra": 3.58, "ss": 2.24, "sa": 1.89}

INIT_MAP = {"r": "random", "s": "structured"}
PRED_MAP = {"s": "sample",  "a": "argmax"}
HANDS = ['r', 'p', 's']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json_with_retry(path, retries=8, sleep_s=2.0):
    last = None
    for _ in range(retries):
        try:
            with open(path) as f:
                return json.load(f)
        except OSError as e:
            last = e
            time.sleep(sleep_s)
    raise last


def ccdf(values, min_val=1):
    a = np.sort(np.asarray([v for v in values if v >= min_val]))
    n = len(a)
    if n == 0:
        return np.array([]), np.array([])
    unique_x, idx = np.unique(a, return_index=True)
    p = (n - idx) / n
    return unique_x, p


def safe_save(fig, name):
    """Save figure to BOTH .png (preview) and .pdf (vector) atomically."""
    base = name[:-4] if name.endswith(".png") else name
    out_png = FIG_DIR / f"{base}.png"
    tmp_png = FIG_DIR / f"{base}.tmp.png"
    fig.savefig(tmp_png, dpi=300, bbox_inches="tight")
    os.replace(tmp_png, out_png)
    out_pdf = FIG_DIR / f"{base}.pdf"
    tmp_pdf = FIG_DIR / f"{base}.tmp.pdf"
    fig.savefig(tmp_pdf, bbox_inches="tight")
    os.replace(tmp_pdf, out_pdf)
    plt.close(fig)
    sz_png = out_png.stat().st_size
    sz_pdf = out_pdf.stat().st_size
    print(f"  wrote {out_png.name} ({sz_png/1024:.1f} KB)"
          f"  +  {out_pdf.name} ({sz_pdf/1024:.1f} KB)")


def dur_path_v2(design, pair, m):
    _, folder = DESIGN_DIRS[design]
    return DATA_ROOT / "simulation" / folder / \
        f"durations_{pair}_m{m}_huge.json"


def load_argmax_durations(path):
    d = load_json_with_retry(path)
    out = list(d.get("T_argmax1", [])) + list(d.get("T_argmax2", []))
    return [int(x) for x in out]


# ---------------------------------------------------------------------------
# Simulator wrappers (own loop -- run_pair does not snapshot h_prov)
# ---------------------------------------------------------------------------

def simulate_pair_with_snapshots(a1_type, a2_type, T,
                                  h_length=50,
                                  init_mode='random', predict_mode='sample',
                                  seed=0, snapshot=True):
    """Run one BIB-BIB or BO-BO pair for T steps.
    If snapshot=True, captures hand sequences, per-step posterior P(h), and
    per-step argmax for both agents.
    Returns dict.
    """
    ag1 = AgentReward(a1_type, h_length=h_length,
                      init_mode=init_mode, predict_mode=predict_mode,
                      seed=seed)
    ag2 = AgentReward(a2_type, h_length=h_length,
                      init_mode=init_mode, predict_mode=predict_mode,
                      seed=seed + 100000)
    Nh = ag1.bayes.h_num if ag1.bayes is not None else 10

    h1_arr = np.empty(T, dtype='<U1')
    h2_arr = np.empty(T, dtype='<U1')
    am1 = np.empty(T, dtype=np.int32)
    am2 = np.empty(T, dtype=np.int32)
    if snapshot:
        p1 = np.empty((T, Nh), dtype=np.float64)
        p2 = np.empty((T, Nh), dtype=np.float64)
    else:
        p1 = p2 = None

    for t in range(T):
        h1 = ag1.choice()
        h2 = ag2.choice()
        am1[t] = ag1.argmax_h()
        am2[t] = ag2.argmax_h()
        h1_arr[t] = h1
        h2_arr[t] = h2
        if snapshot:
            p1[t] = ag1.bayes.h_prov.copy()
            p2[t] = ag2.bayes.h_prov.copy()
        ag1.update_from_outcome(h1, h2)
        ag2.update_from_outcome(h2, h1)

    return {
        'h1': h1_arr, 'h2': h2_arr,
        'argmax1': am1, 'argmax2': am2,
        'p1': p1, 'p2': p2,
    }


def laminar_lengths_from_posterior(P, theta=0.4):
    """Per-step boolean: max_h P(h) > theta. Return lengths of consecutive
    True runs."""
    mask = P.max(axis=1) > theta
    if not mask.any():
        return np.array([], dtype=int)
    diffs = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(diffs == 1) + 1
    ends = np.flatnonzero(diffs == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    return ends - starts


# Worker for multiprocessing: returns laminar lengths for both agents in a run
def _laminar_worker(args):
    (pair_type, design, T, h_length, seed, burn_frac, theta) = args
    init_mode = INIT_MAP[design[0]]
    pred_mode = PRED_MAP[design[1]]
    a1, a2 = pair_type.split('-')
    out = simulate_pair_with_snapshots(
        a1, a2, T, h_length=h_length,
        init_mode=init_mode, predict_mode=pred_mode,
        seed=seed, snapshot=True)
    bi = int(T * burn_frac)
    P1 = out['p1'][bi:]
    P2 = out['p2'][bi:]
    L1 = laminar_lengths_from_posterior(P1, theta=theta)
    L2 = laminar_lengths_from_posterior(P2, theta=theta)
    return (pair_type, design, np.concatenate([L1, L2]))


# ---------------------------------------------------------------------------
# Power-law alpha estimation (truncated power law)
# ---------------------------------------------------------------------------

def fit_alpha_powerlaw(data, xmin=None, prefer="auto"):
    """Fit power-law alpha. prefer:
       'auto'  -- pick PL vs TPL by AIC weight
       'pl'    -- pure power-law alpha
       'tpl'   -- truncated power-law alpha
       'mle'   -- closed-form MLE (fallback)
    Returns (alpha, xmin, best_model_tag)."""
    data = np.asarray(data, dtype=float)
    data = data[data >= 1]
    if len(data) < 50:
        return np.nan, np.nan, 'none'
    try:
        import powerlaw as pl
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = pl.Fit(data, discrete=True, verbose=False)
        xmin_used = float(fit.xmin)
        a_pl = float(fit.power_law.alpha)
        try:
            a_tpl = float(fit.truncated_power_law.alpha)
        except Exception:
            a_tpl = np.nan

        if prefer == 'pl':
            return a_pl, xmin_used, 'pl'
        if prefer == 'tpl' and not np.isnan(a_tpl):
            return a_tpl, xmin_used, 'tpl'
        if prefer == 'auto':
            # AIC comparison via log-likelihoods on data >= xmin
            data_fit = data[data >= fit.xmin]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p_pl = np.asarray(fit.power_law.pdf(data_fit), dtype=float)
                    p_pl = np.where(p_pl > 0, p_pl, 1e-300)
                    ll_pl = float(np.sum(np.log(p_pl)))
                    if not np.isnan(a_tpl):
                        p_tpl = np.asarray(fit.truncated_power_law.pdf(data_fit), dtype=float)
                        p_tpl = np.where(p_tpl > 0, p_tpl, 1e-300)
                        ll_tpl = float(np.sum(np.log(p_tpl)))
                    else:
                        ll_tpl = -np.inf
            except Exception:
                ll_pl = -np.inf
                ll_tpl = -np.inf
            # AIC: PL has 1 param, TPL has 2 params
            aic_pl = 2 * 1 - 2 * ll_pl
            aic_tpl = 2 * 2 - 2 * ll_tpl
            if aic_pl <= aic_tpl or np.isnan(a_tpl):
                return a_pl, xmin_used, 'pl'
            else:
                return a_tpl, xmin_used, 'tpl'
        # default fallback
        if not np.isnan(a_tpl):
            return a_tpl, xmin_used, 'tpl'
        return a_pl, xmin_used, 'pl'
    except Exception as e:
        print(f"  powerlaw failed: {e}")

    # MLE fallback
    if xmin is None:
        xmin = max(1.0, np.percentile(data, 10))
    a = data[data >= xmin]
    if len(a) < 30:
        return np.nan, xmin, 'none'
    alpha = 1.0 + len(a) / np.sum(np.log(a / xmin))
    return float(alpha), float(xmin), 'mle'


# ===========================================================================
# Fig 2 -- dynamics demo from real simulator
# ===========================================================================

def figure2_real(seed=42, T=1500):
    print("\n--- Fig2 dynamics demo (REAL simulator) ---")
    bib = simulate_pair_with_snapshots(
        'bib', 'bib', T,
        h_length=50, init_mode='random', predict_mode='sample',
        seed=seed)
    bo = simulate_pair_with_snapshots(
        'bo', 'bo', T,
        h_length=50, init_mode='random', predict_mode='sample',
        seed=seed)

    Nh = 10
    sym2col = {'r': "#e6c700", 'p': "#1f77b4", 's': "#2ca02c"}
    sym_idx = {'r': 0, 'p': 1, 's': 2}

    # Uniform font-scale spec (matches regenerate_figures.py): double-col
    # figures use figsize_W = 16.4 inch with width=\linewidth in LaTeX
    # (scale 7.05/16.4 = 0.430).  figsize_H = 9.3 keeps panel (b)/(e)
    # aspect ~1.31:1 (5.94 x 4.53 inch inner), the reference for all
    # other figures in this paper.
    fig, axes = plt.subplots(2, 3, figsize=(16.4, 9.3),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.1, 1.3, 1.1]})
    # Widen the inter-column gap a bit so the (b)/(e) colorbar does not
    # crowd the (c)/(f) ylabel.  (default w_pad ~= 0.04167 in inches)
    fig.get_layout_engine().set(w_pad=0.12, h_pad=0.06)

    def panel_hands(ax, hands, label):
        ax.set_xlim(0, T)
        ax.set_ylim(-0.5, 2.5)
        hidx = np.array([sym_idx[h] for h in hands])
        for s, col in sym2col.items():
            idx = np.where(hidx == sym_idx[s])[0]
            ax.vlines(idx, sym_idx[s] - 0.4, sym_idx[s] + 0.4,
                      color=col, lw=0.7)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["R", "P", "S"])
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xlabel("step $t$")

    def panel_post(ax, P, argmax, label):
        # P shape (T, Nh) -- transpose to (Nh, T) for imshow
        im = ax.imshow(P.T, aspect="auto", origin="lower",
                       cmap="magma",
                       extent=[0, T, -0.5, Nh - 0.5],
                       vmin=0, vmax=min(1.0, float(P.max()) * 1.05))
        ax.plot(np.arange(T), argmax, color="white", lw=1.2)
        ax.set_ylabel("hypothesis $h$")
        ax.set_xlabel("step $t$")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_yticks([0, 3, 6, 9])
        # No text label on the colorbar: the colormap + the panel
        # description in the caption are self-explanatory, and the text
        # collided with the (c)/(f) ylabel.  Keep a slightly larger pad
        # too so the colorbar does not crowd the next column.
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    def panel_top3(ax, P, label):
        order = np.argsort(-P.mean(axis=0))[:3]
        for h in order:
            ax.plot(np.arange(T), P[:, h], lw=1.5, label=f"h={h}")
        ax.axhline(1.0 / Nh, ls=":", color="grey",
                   label=f"$1/N_h={1/Nh:.2f}$")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, T)
        ax.set_xlabel("step $t$")
        ax.set_ylabel(r"$P(h)$")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.legend(loc="upper right", frameon=False, fontsize=10, ncol=2)

    # Row 1: BIB-BIB (agent 1) -- row identity carried by caption
    panel_hands(axes[0, 0], bib['h1'], "(a)")
    panel_post(axes[0, 1], bib['p1'], bib['argmax1'], "(b)")
    panel_top3(axes[0, 2], bib['p1'], "(c)")

    # Row 2: BO-BO (agent 1)
    panel_hands(axes[1, 0], bo['h1'], "(d)")
    panel_post(axes[1, 1], bo['p1'], bo['argmax1'], "(e)")
    panel_top3(axes[1, 2], bo['p1'], "(f)")

    # No suptitle -- per PRE convention all descriptive content lives
    # in the caption; the agent-type / parameter info is given there.

    safe_save(fig, "fig_dynamics_demo.png")
    return bib, bo


# ===========================================================================
# Fig 3 -- universality 4-panel (a,b real durations, c,d real laminar)
# ===========================================================================

def run_laminar_sweep(T=20000, n_seeds=5, h_length=50,
                       burn_frac=0.5, theta=0.4,
                       n_workers=None, cache_path=None):
    """Run the 8-condition x n_seeds laminar sweep. Returns dict[(pair,design)]
    -> 1D array of laminar lengths."""
    from multiprocessing import Pool, cpu_count

    if cache_path is not None and Path(cache_path).exists():
        print(f"  loading laminar cache: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as z:
            return {tuple(k.split("|")): z[k] for k in z.files}

    designs = ["rs", "ra", "ss", "sa"]
    pairs = ["bib-bib", "bo-bo"]
    jobs = []
    for pair in pairs:
        for design in designs:
            for s in range(n_seeds):
                seed = 1000 * (designs.index(design) + 1) + \
                       100 * pairs.index(pair) + s
                jobs.append((pair, design, T, h_length, seed,
                             burn_frac, theta))

    nw = n_workers or min(8, max(1, cpu_count()))
    print(f"  laminar sweep: {len(jobs)} jobs, "
          f"workers={nw}, T={T}, n_seeds={n_seeds}, theta={theta}")

    t0 = time.time()
    out = {}
    with Pool(processes=nw) as pool:
        for pair, design, lengths in pool.imap_unordered(
                _laminar_worker, jobs, chunksize=1):
            key = (pair, design)
            if key not in out:
                out[key] = []
            out[key].append(lengths)
    elapsed = time.time() - t0
    print(f"  sweep wall time: {elapsed:.1f}s")

    # Concatenate per (pair, design)
    out_concat = {k: np.concatenate(v) if v else np.array([], dtype=int)
                  for k, v in out.items()}
    for k, arr in out_concat.items():
        print(f"    {k}: n_events={len(arr)}, "
              f"max={arr.max() if len(arr) else 'n/a'}")

    if cache_path is not None:
        save_dict = {f"{k[0]}|{k[1]}": v for k, v in out_concat.items()}
        np.savez_compressed(cache_path, **save_dict)
        print(f"  cached -> {cache_path}")

    return out_concat


def figure3_combined(laminar_data, alphas_out=None):
    """Build 4-panel Fig3.
    (a, b) reuse argmax durations from JSON.
    (c, d) use the laminar_data dict produced by run_laminar_sweep."""
    print("\n--- Fig3 universality 4-panel (REAL panels c, d) ---")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    # (a) BIB-BIB argmax across 4 designs, m=50
    ax = axes[0, 0]
    for design in ["rs", "ra", "ss", "sa"]:
        try:
            path = dur_path_v2(design, "bib-bib", 50)
            data = load_argmax_durations(path)
            x, p = ccdf(data, min_val=1)
            ax.loglog(x, p, color=DESIGN_COLOR[design],
                      label=f"{design} (α≈{ALPHA_ARG_BIB[design]:.2f})",
                      lw=1.7)
        except Exception as e:
            print(f"  Fig3a {design}: {e}")
    ax.set_xlabel(r"$T_{\rm argmax}$")
    ax.set_ylabel(r"CCDF $P(T \geq t)$")
    ax.set_title("(a) BIB-BIB argmax, $m{=}50$", loc="left")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    # (b) BO-BO argmax across 4 designs, m=50
    ax = axes[0, 1]
    for design in ["rs", "ra", "ss", "sa"]:
        try:
            path = dur_path_v2(design, "bo-bo", 50)
            data = load_argmax_durations(path)
            x, p = ccdf(data, min_val=1)
            ax.loglog(x, p, color=DESIGN_COLOR[design],
                      label=f"{design} (α≈{ALPHA_ARG_BO[design]:.2f})",
                      lw=1.7)
        except Exception as e:
            print(f"  Fig3b {design}: {e}")
    ax.set_xlabel(r"$T_{\rm argmax}$")
    ax.set_ylabel(r"CCDF $P(T \geq t)$")
    ax.set_title("(b) BO-BO argmax, $m{=}50$", loc="left")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    # (c) BIB-BIB laminar -- TPL alpha (heavy-tail regime)
    ax = axes[1, 0]
    alphas_bib = {}
    for design in ["rs", "ra", "ss", "sa"]:
        data = laminar_data.get(("bib-bib", design), np.array([]))
        if len(data) == 0:
            print(f"  Fig3c {design}: no data")
            continue
        x, p = ccdf(data, min_val=1)
        alpha, xmin, tag = fit_alpha_powerlaw(data, prefer='tpl')
        alphas_bib[design] = alpha
        label = (f"{design} (α={alpha:.3f})" if not np.isnan(alpha)
                 else f"{design}")
        ax.loglog(x, p, color=DESIGN_COLOR[design], label=label, lw=1.7)
    # 3/2 reference
    xref = np.logspace(0, 3.0, 50)
    pref = xref ** (-(1.5 - 1))
    pref /= pref[0]
    ax.loglog(xref, pref, ls="--", color="grey", lw=1.6,
              label=r"$\alpha=3/2$ (on-off)")
    ax.set_xlabel(r"laminar length")
    ax.set_ylabel(r"CCDF")
    ax.set_title(r"(c) BIB-BIB laminar, $\theta{=}0.4$", loc="left")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    # (d) BO-BO laminar -- pure PL alpha (short / light-tail regime)
    ax = axes[1, 1]
    alphas_bo = {}
    for design in ["rs", "ra", "ss", "sa"]:
        data = laminar_data.get(("bo-bo", design), np.array([]))
        if len(data) == 0:
            print(f"  Fig3d {design}: no data")
            continue
        x, p = ccdf(data, min_val=1)
        alpha, xmin, tag = fit_alpha_powerlaw(data, prefer='pl')
        alphas_bo[design] = alpha
        label = (f"{design} (α={alpha:.2f})" if not np.isnan(alpha)
                 else f"{design}")
        ax.loglog(x, p, color=DESIGN_COLOR[design], label=label, lw=1.7)
    ax.set_xlabel(r"laminar length")
    ax.set_ylabel(r"CCDF")
    ax.set_title(r"(d) BO-BO laminar, $\theta{=}0.4$", loc="left")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    safe_save(fig, "Fig3_universality_4panel_PROD.png")
    if alphas_out is not None:
        alphas_out['bib'] = alphas_bib
        alphas_out['bo'] = alphas_bo
    return alphas_bib, alphas_bo


# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()

    # Pre-flight time check
    print("\n=== Pre-flight timing trial (T=5000, BIB-BIB rs) ===")
    pf_t0 = time.time()
    simulate_pair_with_snapshots(
        'bib', 'bib', 5000, h_length=50,
        init_mode='random', predict_mode='sample',
        seed=999, snapshot=True)
    pf_el = time.time() - pf_t0
    T_sweep, n_seeds = 20000, 5
    n_jobs = 8 * n_seeds
    from multiprocessing import cpu_count
    nw = min(8, max(1, cpu_count()))
    projected = pf_el * (T_sweep / 5000) * n_jobs / nw
    print(f"  pre-flight: {pf_el:.2f}s ({pf_el/5000*1e6:.1f} us/step)")
    print(f"  projected sweep wall time: {projected:.1f}s "
          f"({n_jobs} jobs, {nw} workers, T={T_sweep})")
    if projected > 600:
        print(f"ABORT: projected {projected:.0f}s exceeds 10-minute budget")
        sys.exit(2)

    # Task A -- Fig2 dynamics demo
    figure2_real(seed=42, T=1500)

    # Task B -- laminar sweep (cached) + Fig3 4-panel
    cache_path = FIG_DIR / "_laminar_cache.npz"
    laminar = run_laminar_sweep(
        T=T_sweep, n_seeds=n_seeds, h_length=50,
        burn_frac=0.5, theta=0.4,
        n_workers=nw, cache_path=cache_path)

    alphas = {}
    figure3_combined(laminar, alphas_out=alphas)

    # Report
    print("\n=== FITTED ALPHAS ===")
    print("BIB-BIB laminar:")
    for d in ["rs", "ra", "ss", "sa"]:
        a = alphas['bib'].get(d, np.nan)
        print(f"  {d}: alpha = {a:.4f}")
    print("BO-BO laminar:")
    for d in ["rs", "ra", "ss", "sa"]:
        a = alphas['bo'].get(d, np.nan)
        print(f"  {d}: alpha = {a:.4f}")

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
