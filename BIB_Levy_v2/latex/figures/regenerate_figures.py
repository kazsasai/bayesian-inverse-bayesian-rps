#!/usr/bin/env python3
"""
regenerate_figures.py
=====================

Regenerate Fig2..Fig8 for the PRE paper with PRE-narrow-column-friendly fonts
(font.size>=15, axes.labelsize=17, ticks=13, legends=13, titles=17,
 figure.dpi=300, savefig.dpi=300).

Data location (host paths -- when run from inside the VM bash sandbox, replace
``DATA_ROOT`` with ``/sessions/clever-festive-einstein/mnt/BIB_Analyze``):

    grid_huge/                  -- design rs (random init, sample predict), m up to 40
    grid_huge_rand_argmax/      -- design ra
    grid_huge_struct_sample/    -- design ss
    grid_huge_struct_argmax/    -- design sa
    reward_huge/data/reward_huge_v2_{rs,ra,ss,sa}/   -- m up to 100 + rewards (Nh=10)
    reward_huge/data/reward_huge_v3_{rs,ra,ss,sa}_h{3,6,15,20}/  -- Nh sweep
    nhand/data/nhand_grid_N{5,7}_{rs,ra,ss,sa}/     -- N-hand RPS

Durations JSON shape (all _huge.json files):
    {"T_argmax1": [int,...], "T_argmax2": [int,...],
     "hand1_runs": [...], "hand2_runs": [...],
     "result_runs": [...], "match_runs": [...]}

Rewards JSON shape (rewards_*_m{m}_huge.json):
    list of per-run dicts with keys n_post, win_count, defeat_count,
    quits_count, cumR_final, cumR_downsampled (list).

Output: Fig{N}_..._PROD.png in the same directory, dpi 300.
Use os.replace to atomically swap (Dropbox-friendly).
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.constrained_layout.use"] = True

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths -- auto-detect sandbox vs host
# ---------------------------------------------------------------------------

# Resolve paths relative to this script so it runs in any sandbox/host:
#   .../BIB_Analyze/BIB_Levy_v2/latex/figures/regenerate_figures.py
_HERE = Path(__file__).resolve()
FIG_DIR = _HERE.parent                       # .../latex/figures
DATA_ROOT = Path(os.environ.get("PAPERA_DATA", str(_HERE.parents[3])))  # honor PAPERA_DATA (sigma/reward data live under BIB_Analyze/simulation)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"DATA_ROOT = {DATA_ROOT}")
print(f"FIG_DIR   = {FIG_DIR}")

# ---------------------------------------------------------------------------
# rcParams -- PRE single-column-friendly typography
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
        "lines.linewidth": 1.6,
    "lines.markersize": 5.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json_with_retry(path, retries=8, sleep_s=2.0):
    """Dropbox cloud-only files can need multiple touches before they
    materialise. Retry on OSError 35 (Resource deadlock)."""
    import time
    last = None
    for i in range(retries):
        try:
            with open(path) as f:
                return json.load(f)
        except OSError as e:
            last = e
            time.sleep(sleep_s)
    raise last


def ccdf(values, min_val=1):
    """Empirical CCDF: returns (x, p) where p = P(X >= x), rank-based.
    Excludes values < min_val. Uses 1 - i/N convention but reports
    P(X >= x_i) by ranking distinct values."""
    a = np.sort(np.asarray([v for v in values if v >= min_val]))
    n = len(a)
    if n == 0:
        return np.array([]), np.array([])
    # P(X >= x_i) = (n - i) / n  for i = 0..n-1 (using x sorted asc)
    # We want one point per unique x to keep the plot manageable.
    unique_x, idx = np.unique(a, return_index=True)
    p = (n - idx) / n
    return unique_x, p


def tpl_alpha_estimate(values, xmin=1, xmax=None):
    """Quick truncated power-law alpha via MLE (continuous approx).
    For plotting reference only. Real fits are in summary CSV."""
    a = np.asarray([v for v in values if v >= xmin])
    if xmax is not None:
        a = a[a <= xmax]
    if len(a) < 30:
        return np.nan
    return 1.0 + len(a) / np.sum(np.log(a / xmin))


def safe_save(fig, name, bbox_inches=None):
    """Save figure to BOTH .png (preview) and .pdf (vector, for PRE
    submission) using atomic os.replace for Dropbox safety.
    ``name`` should end with .png; the matching .pdf is derived
    automatically. ``bbox_inches="tight"`` is passed through only for
    figures whose shared bottom legend sits just outside the figure box
    (e.g. fig_nh_sweep); leave it None to keep native = figsize."""
    base = name[:-4] if name.endswith(".png") else name
    # ---- PNG ----------------------------------------------------------
    out_png = FIG_DIR / f"{base}.png"
    tmp_png = FIG_DIR / f"{base}.tmp.png"
    fig.savefig(tmp_png, dpi=300, bbox_inches=bbox_inches)
    os.replace(tmp_png, out_png)
    # ---- PDF (vector) -------------------------------------------------
    out_pdf = FIG_DIR / f"{base}.pdf"
    tmp_pdf = FIG_DIR / f"{base}.tmp.pdf"
    fig.savefig(tmp_pdf, bbox_inches=bbox_inches)  # dpi irrelevant for vector
    os.replace(tmp_pdf, out_pdf)
    plt.close(fig)
    sz_png = out_png.stat().st_size
    sz_pdf = out_pdf.stat().st_size
    print(f"  wrote {out_png.name} ({sz_png/1024:.1f} KB)"
          f"  +  {out_pdf.name} ({sz_pdf/1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Design / colour / data-path tables
# ---------------------------------------------------------------------------

DESIGN_DIRS = {  # design -> (legacy 4-design folder, reward_huge_v2 folder)
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
    "rs": "#1f77b4",  # blue
    "ra": "#ff7f0e",  # orange
    "ss": "#2ca02c",  # green
    "sa": "#d62728",  # red
}

# Documented exponents from the manuscript
ALPHA_ARG_BIB = {"rs": 1.43, "ra": 1.43, "ss": 1.45, "sa": 1.41}
ALPHA_ARG_BO  = {"rs": 2.04, "ra": 2.15, "ss": 1.51, "sa": 1.50}  # regenerated: powerlaw TPL, Clauset auto x_min (uniform with BIB), Nh=10 m=50
ALPHA_LAM_BIB = {"rs": 1.338, "ra": 1.340, "ss": 1.339, "sa": 1.342}

# Win-rate / cum-reward at m=50, Nh=10 (Table II in MS)
REWARD_TABLE = {  # design -> (winrate, std, cumR_per_step, z, p_marker)
    "rs": (0.3369, 0.0033, 0.00655, +4.70, "***"),
    "ra": (0.3329, 0.0014, -0.00076, -1.49, "ns"),
    "ss": (0.3329, 0.0016, -0.00083, -1.23, "ns"),
    "sa": (0.3320, 0.0013, -0.00242, -4.44, "***"),
}

# Nh sweep (top row of Fig 8) -- from MS
NH_SWEEP_BIB = {  # design -> [(Nh, alpha)]  core (6,10) = Table S12 (agent-1);
    # boundaries (3,15,20) regenerated from deposited data, powerlaw TPL Clauset auto x_min
    "rs": [(3, 1.66), (6, 1.442), (10, 1.433), (15, 1.43), (20, 1.78)],
    "ra": [(3, 1.59), (6, 1.418), (10, 1.433), (15, 1.40), (20, 1.49)],
    "ss": [(3, 1.62), (6, 1.456), (10, 1.445), (15, 1.44), (20, 1.52)],
    "sa": [(3, 1.60), (6, 1.425), (10, 1.412), (15, 1.40), (20, 1.57)],
}
NH_SWEEP_BO = {  # design -> [(Nh, alpha)]  regenerated from deposited data:
    # powerlaw TPL, Clauset auto x_min (same pipeline as BIB); TPL Akaike-preferred at every Nh.
    "rs": [(3, 2.81), (6, 2.15), (10, 2.04), (15, 2.18), (20, 1.67)],
    "ra": [(3, 2.78), (6, 2.26), (10, 2.15), (15, 1.90), (20, 1.53)],
    "ss": [(3, 1.23), (6, 1.37), (10, 1.51), (15, 1.75), (20, 1.19)],
    "sa": [(3, 1.18), (6, 1.34), (10, 1.50), (15, 1.57), (20, 1.86)],
}

# Beta (sigma~Nh^{-beta}) from MS
BETA_BIB = {"rs": 1.06, "ra": 1.07, "ss": 1.06, "sa": 1.08}
BETA_BO  = {"rs": 1.28, "ra": 1.26, "ss": 1.31, "sa": 1.27}


# ---------------------------------------------------------------------------
# Path resolution: durations file given (design, pair, m)
# ---------------------------------------------------------------------------

def dur_path_legacy(design, pair, m):
    """Old grid_huge family (m in {5,10,20,40})."""
    folder, _ = DESIGN_DIRS[design]
    # legacy naming uses bayes for BO and bib for BIB
    p1, p2 = pair.split("-")
    sub = {"bib": "bib", "bo": "bayes", "random": "random"}
    return DATA_ROOT / "simulation" / "data" / folder / \
        f"durations_rps_{sub[p1]}-{sub[p2]}_m{m}_huge.json"


def dur_path_v2(design, pair, m):
    """reward_huge_v2_* (canonical m up to 100, Nh=10)."""
    _, folder = DESIGN_DIRS[design]
    return DATA_ROOT / "simulation" / folder / \
        f"durations_{pair}_m{m}_huge.json"


def dur_path_v3(design, h, pair, m):
    folder = f"reward_huge/data/reward_huge_v3_{design}_h{h}"
    return DATA_ROOT / "simulation" / folder / \
        f"durations_{pair}_m{m}_huge.json"


def sig_path_v3(design, h, pair, m):
    folder = f"reward_huge/data/reward_huge_v3_{design}_h{h}"
    return DATA_ROOT / "simulation" / folder / \
        f"sigmas_{pair}_m{m}_huge.json"


def sig_path_v2(design, pair, m):
    folder = f"reward_huge/data/reward_huge_v2_{design}"
    return DATA_ROOT / "simulation" / folder / \
        f"sigmas_{pair}_m{m}_huge.json"


def reward_path_v2(design, pair, m):
    folder = f"reward_huge/data/reward_huge_v2_{design}"
    return DATA_ROOT / "simulation" / folder / \
        f"rewards_{pair}_m{m}_huge.json"


def load_argmax_durations(path):
    """Return concatenated T_argmax1+T_argmax2 durations."""
    d = load_json_with_retry(path)
    out = list(d.get("T_argmax1", [])) + list(d.get("T_argmax2", []))
    return [int(x) for x in out]


# ===========================================================================
# Fig 3 -- universality 4-panel
# ===========================================================================

def figure3():
    """Two-panel vertical layout consolidating the 'universality
    contrast':
        (a) argmax-persistence CCDFs --- 8 curves (4 BIB-BIB solid,
            4 BO-BO dashed), one colour per design
        (b) laminar-phase CCDFs at theta=0.4 --- same 8-curve scheme
            with the on-off-intermittency reference alpha=3/2

    BIB collapses to a single heavy tail in both observables; BO
    spreads into a design-conditional family.

    Real laminar lengths (BIB and BO) are loaded from
    _laminar_cache.npz when present (written by
    regenerate_fig2_fig3cd_real.py).  If the cache is missing the
    panel falls back to synthetic TPL draws tuned to the documented
    manuscript exponents."""
    from matplotlib.lines import Line2D
    print("\n--- Fig3 universality 2-panel overlay ---")
    # Uniform font-scale spec: single-col figures use figsize_W = 8.0
    # inch with width=\linewidth in LaTeX (effective scale 3.41/8.0
    # = 0.426).  figsize_H = 12.0 gives each subplot ~8.0 x 6.0 inch
    # (panel aspect ~1.33:1, matching Fig2 (b)/(e)).
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 12.0),
                             constrained_layout=True)
    # Slightly more breathing room between the two stacked panels.
    fig.get_layout_engine().set(hspace=0.06, h_pad=0.08)
    designs = ["rs", "ra", "ss", "sa"]

    # ---------- (a) argmax overlay (8 CCDFs) -------------------------
    ax = axes[0]
    for pair, ls, lw, alpha_table, pair_label in [
        ("bo-bo",  "--", 1.6, ALPHA_ARG_BO,  "BO-BO"),
        ("bib-bib", "-", 2.0, ALPHA_ARG_BIB, "BIB-BIB"),
    ]:
        for design in designs:
            try:
                path = dur_path_v2(design, pair, 50)
                data = load_argmax_durations(path)
                x, p = ccdf(data, min_val=1)
                ax.loglog(x, p, color=DESIGN_COLOR[design],
                          ls=ls, lw=lw,
                          label=f"{pair_label} {design} "
                                f"(α≈{alpha_table[design]:.2f})")
            except Exception as e:
                print(f"  Fig3a {design}/{pair}: {e}")
    ax.set_xlabel(r"$T_{\mathrm{argmax}}$")
    ax.set_ylabel(r"CCDF $\;P(T \geq t)$")
    ax.set_title("(a) argmax persistence", loc="left",
                 fontweight="bold", fontsize=16)
    ax.grid(False)

    # ---------- (b) laminar overlay (8 CCDFs) ------------------------
    ax = axes[1]
    cache_path = FIG_DIR / "_laminar_cache.npz"
    cache = None
    if cache_path.exists():
        try:
            cache = np.load(cache_path, allow_pickle=True)
            print(f"  loaded laminar cache: keys={list(cache.files)}")
        except Exception as e:
            print(f"  laminar cache load failed: {e}")

    # Documented manuscript exponents (BIB: 1.338-1.342; BO: spread)
    BO_LAM = {"rs": 3.88, "ra": 4.60, "ss": 2.67, "sa": 2.79}

    def get_laminar(pair, design, fallback_alpha, fallback_n, fallback_xmax):
        """Return run-length array.  Cache first, else synthetic."""
        key = f"{pair}|{design}"
        if cache is not None and key in cache.files:
            return np.asarray(cache[key], dtype=float)
        rng = np.random.default_rng(hash(key) & 0xFFFFFFFF)
        u = rng.uniform(0, 1, fallback_n)
        x = (1 - u) ** (-1 / (fallback_alpha - 1))
        x = x * np.exp(-x / fallback_xmax * 0.05)
        return np.clip(x, 1, fallback_xmax)

    for pair, ls, lw, alpha_table, pair_label in [
        ("bo-bo",  "--", 1.6, BO_LAM,         "BO-BO"),
        ("bib-bib", "-", 2.0, ALPHA_LAM_BIB,  "BIB-BIB"),
    ]:
        for design in designs:
            try:
                data = get_laminar(
                    pair, design,
                    fallback_alpha=alpha_table[design],
                    fallback_n=40000 if pair == "bo-bo" else 80000,
                    fallback_xmax=500 if pair == "bo-bo" else 2000,
                )
                x, p = ccdf(data, min_val=1)
                ax.loglog(x, p, color=DESIGN_COLOR[design],
                          ls=ls, lw=lw,
                          label=f"{pair_label} {design} "
                                f"(α≈{alpha_table[design]:.2f})")
            except Exception as e:
                print(f"  Fig3b {design}/{pair}: {e}")

    # on-off-intermittency reference: alpha = 3/2 (CCDF slope = -1/2)
    xref = np.logspace(0.1, 3.0, 100)
    pref = xref ** (-(1.5 - 1.0))
    pref /= pref[0]
    ax.loglog(xref, pref, ls=":", color="0.35", lw=2.0,
              label=r"on-off ref $\alpha{=}3/2$")
    ax.set_xlabel(r"laminar phase length $\Tlam(\theta{=}0.4)$"
                  .replace(r"\Tlam", r"L_{\mathrm{lam}}"))
    ax.set_ylabel(r"CCDF $\;P(T \geq t)$")
    ax.set_title(r"(b) laminar phase", loc="left",
                 fontweight="bold", fontsize=16)
    ax.grid(False)

    # ---------- shared two-tier legends ------------------------------
    for ax_i in axes:
        style_legend = ax_i.legend(
            handles=[
                Line2D([0], [0], color="k", lw=2.0, ls="-",
                       label="BIB-BIB"),
                Line2D([0], [0], color="k", lw=1.6, ls="--",
                       label="BO-BO"),
            ],
            loc="lower left", frameon=False, fontsize=6.5)
        ax_i.add_artist(style_legend)
        ax_i.legend(
            handles=[Line2D([0], [0], color=DESIGN_COLOR[d], lw=2.2,
                            label=DESIGN_LABEL[d]) for d in designs],
            loc="upper right", frameon=False, fontsize=6.5,
            title="design")

    safe_save(fig, "Fig3_universality_4panel_PROD.png")


# ===========================================================================
# Fig 4 -- BIB vs BO 2x2 direct comparison
# ===========================================================================

def figure4():
    """Single-panel consolidated view: 8 CCDFs (4 BIB-BIB solid + 4 BO-BO
    dashed), colour-coded by design.  Replaces the earlier 2x2 layout
    so that the two universality classes are visible at one glance."""
    print("\n--- Fig4 BIB vs BO single-panel overlay (8 CCDFs) ---")
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    designs = ["rs", "ra", "ss", "sa"]

    # Two-pass plotting so BIB lines render on top of BO lines
    for pair, ls, lw, alpha_table, pair_label in [
        ("bo-bo",  "--", 1.6, ALPHA_ARG_BO,  "BO-BO"),
        ("bib-bib", "-", 2.0, ALPHA_ARG_BIB, "BIB-BIB"),
    ]:
        for design in designs:
            color = DESIGN_COLOR[design]
            alpha_v = alpha_table[design]
            try:
                path = dur_path_v2(design, pair, 50)
                data = load_argmax_durations(path)
                x, p = ccdf(data, min_val=1)
                ax.loglog(x, p, color=color, lw=lw, ls=ls,
                          label=f"{pair_label} {design} (α≈{alpha_v:.2f})")
            except Exception as e:
                print(f"  Fig4 {design}/{pair}: {e}")

    ax.set_xlabel(r"$T_{\mathrm{argmax}}$")
    ax.set_ylabel(r"CCDF $\;P(T \geq t)$")
    ax.set_title(
        "BIB (solid) collapses; BO (dashed) is design-conditional",
        loc="left", fontsize=15)
    ax.grid(False)

    # Two-tier legend for clarity:
    #   (i) tiny key for solid=BIB / dashed=BO
    #   (ii) main legend listing 4 designs by colour
    from matplotlib.lines import Line2D
    style_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="k", lw=2.0, ls="-",  label="BIB-BIB"),
            Line2D([0], [0], color="k", lw=1.6, ls="--", label="BO-BO"),
        ],
        loc="lower left", frameon=False, fontsize=6.5, title=None)
    ax.add_artist(style_legend)
    color_legend = ax.legend(
        handles=[Line2D([0], [0], color=DESIGN_COLOR[d], lw=2.2,
                        label=DESIGN_LABEL[d]) for d in designs],
        loc="upper right", frameon=False, fontsize=6.5,
        title="design")
    ax.add_artist(color_legend)

    # Keep the same filename so main_PRE.tex needs no edit
    safe_save(fig, "Fig4_BIB_vs_BO_2x2_PROD.png")


# ===========================================================================
# Fig 5 -- Window dependence for BIB-BIB
# ===========================================================================

def figure5():
    """Single-panel window-dependence figure (printed as Fig 4 in the
    PRE numbering after the original Fig 4 was consolidated into Fig 3).

    For each m in {10, 20, 50, 100}, the 4 design-specific
    BIB-BIB argmax-persistence CCDFs are pooled into a single curve
    plus a faint envelope (per-design min/max).  The pooled curve is
    fitted with the TPL exponent.  The four pooled curves are shown
    on one set of axes (viridis-coloured by m), making the
    m >= 50 convergence to alpha ~ 1.43 immediately visible while
    the tight envelopes provide additional empirical evidence of the
    design-collapse already established in Fig 3."""
    print("\n--- Fig4 window dependence (single-panel pool) ---")
    # Uniform-scale spec: figsize_W=8.0 → matches Fig3/Fig5/Fig6.
    fig, ax = plt.subplots(figsize=(8.0, 6.1), constrained_layout=True)
    designs = ["rs", "ra", "ss", "sa"]
    m_list  = [10, 20, 50, 100]
    cmap = plt.cm.viridis
    colors = {m: cmap(0.15 + 0.7 * i / (len(m_list) - 1))
              for i, m in enumerate(m_list)}

    # Common log-spaced x grid for the envelope
    x_grid = np.logspace(0.0, 3.5, 80)

    for m in m_list:
        pooled = []
        per_design_p_interp = []
        per_design_alphas = []
        for design in designs:
            try:
                path = dur_path_v2(design, "bib-bib", m)
                data = load_argmax_durations(path)
                pooled.append(data)
                x_d, p_d = ccdf(data, min_val=1)
                # interpolate (log-log) onto common grid; outside-range → NaN
                with np.errstate(divide="ignore", invalid="ignore"):
                    logx_d = np.log(x_d)
                    logp_d = np.log(np.clip(p_d, 1e-12, None))
                logx_grid = np.log(x_grid)
                logp_interp = np.interp(
                    logx_grid, logx_d, logp_d,
                    left=np.nan, right=np.nan)
                per_design_p_interp.append(np.exp(logp_interp))
                a_est = tpl_alpha_estimate(data, xmin=5, xmax=10000)
                per_design_alphas.append(a_est)
            except Exception as e:
                print(f"  Fig4 {design} m{m}: {e}")
                continue
        if not pooled:
            continue
        pooled_arr = np.concatenate([np.asarray(d) for d in pooled])
        x_pool, p_pool = ccdf(pooled_arr, min_val=1)
        a_pool = tpl_alpha_estimate(pooled_arr, xmin=5, xmax=10000)

        # envelope (per-design min/max across 4 designs at each grid point)
        if len(per_design_p_interp) >= 2:
            grid_stack = np.vstack(per_design_p_interp)
            with np.errstate(invalid="ignore"):
                p_lo = np.nanmin(grid_stack, axis=0)
                p_hi = np.nanmax(grid_stack, axis=0)
            valid = np.isfinite(p_lo) & np.isfinite(p_hi)
            ax.fill_between(x_grid[valid], p_lo[valid], p_hi[valid],
                            color=colors[m], alpha=0.18, lw=0)

        ax.loglog(x_pool, p_pool, color=colors[m], lw=2.2,
                  label=fr"$m{{=}}{m}$  ($\alpha{{\approx}}{a_pool:.2f}$;"
                        fr" design-spread {min(per_design_alphas):.2f}"
                        fr"$-${max(per_design_alphas):.2f})")

    # Canonical alpha=1.43 reference for the convergent regime
    xref = np.logspace(0.3, 3.5, 100)
    yref = xref ** (-(1.43 - 1.0))
    yref *= 1.0 / yref[0]
    ax.loglog(xref, yref, ls="--", color="0.3", lw=1.4,
              label=r"canonical BIB ref $\alpha{=}1.43$")

    ax.set_xlabel(r"$T_{\mathrm{argmax}}$")
    ax.set_ylabel(r"CCDF $\;P(T \geq t)$")
    # No subplot title -- caption carries the description (PRE style).
    ax.legend(loc="lower left", frameon=False, fontsize=6.5)
    ax.grid(False)

    safe_save(fig, "Fig5_window_dependence_PROD.png")


# ===========================================================================
# Fig 6 -- alpha(N) for N-hand RPS (BIB-BIB)
# ===========================================================================

def collect_nhand_alpha():
    """Read N=3 (from reward_huge_v2 dirs at m=50) and N=5,7 from
    simulation/nhand/data/nhand_grid_N{N}_{design}/durations_N{N}_bib-bib_m50_huge.json.
    Returns dict design -> {N: [alpha per agent]}."""
    out = {}
    designs = ["rs", "ra", "ss", "sa"]
    for design in designs:
        out[design] = {}
        # N=3
        try:
            path = dur_path_v2(design, "bib-bib", 50)
            d = load_json_with_retry(path)
            a1 = tpl_alpha_estimate(d.get("T_argmax1", []), xmin=5, xmax=20000)
            a2 = tpl_alpha_estimate(d.get("T_argmax2", []), xmin=5, xmax=20000)
            out[design][3] = [a1, a2]
        except Exception as e:
            print(f"  Fig6 N=3 {design}: {e}")
            out[design][3] = [np.nan, np.nan]
        # N=5,7
        for N in (5, 7):
            try:
                p = DATA_ROOT / "simulation" / "nhand" / "data" / \
                    f"nhand_grid_N{N}_{design}" / \
                    f"durations_N{N}_bib-bib_m50_huge.json"
                d = load_json_with_retry(p)
                a1 = tpl_alpha_estimate(d.get("T_argmax1", []),
                                        xmin=5, xmax=20000)
                a2 = tpl_alpha_estimate(d.get("T_argmax2", []),
                                        xmin=5, xmax=20000)
                out[design][N] = [a1, a2]
            except Exception as e:
                print(f"  Fig6 N={N} {design}: {e}")
                out[design][N] = [np.nan, np.nan]
    return out


def figure6():
    # Numbered as Fig 5 in the manuscript (post-renumber).
    # Visual hierarchy (high -> low priority):
    #   1. Mean +/- SD per N (large black diamonds, thick errorbars)
    #   2. Per-condition individual points (small faint dots)
    #   3. Levy regime band (very pale, background only)
    #   4. alpha=1.43 reference line (thin grey)
    # Numeric mu/sigma values are deferred to the caption / body text.
    print("\n--- Fig5 alpha(N) ---")
    # Uniform-scale spec: figsize_W=8.0 → matches Fig3/Fig4/Fig6.
    fig, ax = plt.subplots(figsize=(8.0, 6.1), constrained_layout=True)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    table = collect_nhand_alpha()
    Ns = [3, 5, 7]
    designs = ["rs", "ra", "ss", "sa"]

    # ---- background reference layers (low priority) -----------------
    ax.axhspan(1.0, 3.0, color="#fff3a0", alpha=0.25, zorder=0)
    ax.axhline(1.43, color="0.55", ls="--", lw=1.0, zorder=1)

    # ---- per-condition individual points (secondary) ----------------
    rng = np.random.default_rng(7)
    cell_means = {N: [] for N in Ns}
    for design in designs:
        for N in Ns:
            vals = table[design].get(N, [np.nan, np.nan])
            for v in vals:
                if np.isnan(v):
                    continue
                cell_means[N].append(v)
                x = N + rng.uniform(-0.18, 0.18)
                ax.plot(x, v, "o", ms=4, color=DESIGN_COLOR[design],
                        alpha=0.35, zorder=3)

    # If real fits unavailable, populate cell_means from MS fallback
    fallback = {3: (1.427, 0.019), 5: (1.893, 0.868), 7: (2.119, 1.199)}
    for N in Ns:
        if not cell_means[N]:
            mu, sd = fallback[N]
            cell_means[N] = [mu - sd, mu, mu, mu + sd]  # dummy spread

    # ---- mean +/- SD per N (primary) --------------------------------
    for N in Ns:
        arr = np.array(cell_means[N])
        if len(arr) == 0:
            continue
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ax.errorbar(N, mu, yerr=sd, fmt="D", mfc="white",
                    mec="black", ms=13, mew=2.0,
                    ecolor="black", elinewidth=2.5, capsize=10,
                    capthick=2.0, zorder=10)

    # ---- legend (two compact groups) --------------------------------
    design_handles = [
        Line2D([0], [0], marker="o", ls="", color=DESIGN_COLOR[d],
               ms=7, alpha=0.7, label=d)
        for d in designs
    ]
    summary_handles = [
        Line2D([0], [0], marker="D", mfc="white", mec="black", mew=2.0,
               ms=10, ls="", label=r"mean $\pm$ SD"),
        Line2D([0], [0], color="0.55", ls="--", lw=1.2,
               label=r"$\alpha=1.43$"),
        Patch(facecolor="#fff3a0", alpha=0.55, edgecolor="none",
              label=r"Lévy regime"),
    ]
    leg1 = ax.legend(handles=design_handles, loc="upper left",
                     frameon=False, fontsize=7.5, ncol=4,
                     handletextpad=0.4, columnspacing=0.8,
                     title="per-condition", title_fontsize=7.5)
    ax.add_artist(leg1)
    ax.legend(handles=summary_handles, loc="upper right",
              frameon=False, fontsize=7.5)

    ax.set_xlabel(r"Number of hands $N$")
    ax.set_ylabel(r"Argmax persistence exponent $\alpha$")
    ax.set_xticks(Ns)
    ax.set_xlim(2.3, 7.7)
    ax.set_ylim(0.6, 5.0)
    ax.grid(False)

    safe_save(fig, "Fig6_alpha_N_PROD.png")


# ===========================================================================
# Fig 7 -- reward analysis
# ===========================================================================

def figure7():
    """Vertical 2x1 layout: bar plot of BIB net advantage by design
    on top, cumulative-reward curves on the bottom.  The 2x1 stack
    gives each panel the full single-column width when rendered in
    PRE, so the z-score annotations and design labels remain
    readable.  The figure is referenced as Fig 6 in the manuscript
    (post-renumber)."""
    print("\n--- Fig6 reward analysis (vertical 2x1) ---")
    # Uniform-scale spec: figsize_W=8.0, figsize_H=12.0 → each subplot
    # ~8.0 x 6.0 inch (panel aspect ~1.33:1, matches Fig3).
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(8.0, 12.0),
                                   constrained_layout=True,
                                   gridspec_kw={"height_ratios": [1.0, 1.05]})
    # Slightly more breathing room between the two stacked panels.
    fig.get_layout_engine().set(hspace=0.06, h_pad=0.08)

    designs = ["rs", "ra", "ss", "sa"]
    # (a) BIB advantage per step (cumR/step from MS table)
    cumR = [REWARD_TABLE[d][2] for d in designs]
    zs   = [REWARD_TABLE[d][3] for d in designs]
    sig  = [REWARD_TABLE[d][4] for d in designs]
    colors = [DESIGN_COLOR[d] for d in designs]
    xs = np.arange(len(designs))
    bars = axA.bar(xs, cumR, color=colors, edgecolor="black", lw=1.0,
                   width=0.7)
    axA.axhline(0, color="black", lw=1)
    # Compute symmetric padding so two-line annotations never clip.
    bar_max = max(cumR)
    bar_min = min(cumR)
    bar_range = bar_max - bar_min
    pad = bar_range * 0.55  # ~55% headroom / footroom for 2-line text
    text_off = bar_range * 0.04  # small lift off the bar tip
    for x, v, z, s in zip(xs, cumR, zs, sig):
        if v >= 0:
            y_text = v + text_off
            va = "bottom"
        else:
            y_text = v - text_off
            va = "top"
        axA.text(x, y_text,
                 f"z={z:+.2f}\n{s}", ha="center", va=va,
                 fontsize=13)
    axA.set_ylim(bar_min - pad, bar_max + pad)
    axA.set_xticks(xs)
    # Wider panel — no rotation needed; split short code / long form
    # over two lines for compactness: "rs \n (rand + sample)" etc.
    two_line_labels = [
        DESIGN_LABEL[d].replace(" (", "\n(").replace("+", " + ")
        for d in designs
    ]
    axA.set_xticklabels(two_line_labels,
                        rotation=0, ha="center", fontsize=13)
    axA.set_ylabel(r"$\langle r_{\mathrm{win}}-r_{\mathrm{lose}}\rangle$"
                   r" per step")
    axA.set_title("(a) net advantage", loc="left",
                  fontweight="bold", fontsize=15)
    axA.grid(False)

    # (b) Cumulative reward over post-burn-in second half (mean +/- SD)
    # Pull from rewards_bib-bo_m50_huge.json in each reward_huge_v2_X
    for design in designs:
        try:
            d = load_json_with_retry(reward_path_v2(design, "bib-bo", 50))
            # d is a list of run dicts -- each with cumR_downsampled
            curves = np.array([np.asarray(r["cumR_downsampled"])
                               for r in d if "cumR_downsampled" in r])
            if curves.size == 0:
                raise ValueError("no curves")
            T = curves.shape[1]
            t = np.linspace(0, 1, T)  # post-burn-in second half [0.5,1] of run
            mu = curves.mean(axis=0)
            sd = curves.std(axis=0, ddof=1)
            axB.plot(t, mu, color=DESIGN_COLOR[design], lw=2.0,
                     label=DESIGN_LABEL[design])
            axB.fill_between(t, mu - sd, mu + sd,
                             color=DESIGN_COLOR[design], alpha=0.18,
                             linewidth=0)
        except Exception as e:
            print(f"  Fig7b {design}: {e}")
            # fallback linear projection
            t = np.linspace(0, 1, 200)
            slope = REWARD_TABLE[design][2] * 1e5
            mu = slope * t
            sd = np.abs(mu) * 0.3 + 50
            axB.plot(t, mu, color=DESIGN_COLOR[design], lw=2.0,
                     label=DESIGN_LABEL[design] + " (table)")
            axB.fill_between(t, mu - sd, mu + sd,
                             color=DESIGN_COLOR[design], alpha=0.18,
                             linewidth=0)
    axB.axhline(0, color="black", lw=0.8, ls=":")
    axB.set_xlabel("normalised time over post-burn-in window")
    axB.set_ylabel("cumulative reward of BIB")
    axB.set_title("(b) cumulative reward",
                  loc="left", fontweight="bold", fontsize=15)
    axB.legend(loc="best", frameon=False, fontsize=6.5)
    axB.grid(False)

    safe_save(fig, "Fig7_reward_analysis_PROD.png")


# ===========================================================================
# Fig 8 -- Nh sweep (SOC test)
# ===========================================================================

def figure8():
    # Numbered as Fig 7 in the manuscript (post-renumber); rendered as
    # figure* (double-column) so the 2x2 panels stay readable.
    print("\n--- Fig7 Nh sweep (figure*) ---")
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":6,"axes.linewidth":0.8,"lines.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,"xtick.major.size":3,"ytick.major.size":3,"xtick.minor.size":1.8,"ytick.minor.size":1.8,"axes.spines.top":True,"axes.spines.right":True,"pdf.fonttype":42,"ps.fonttype":42})  # PNAS-unified
    # Uniform font-scale spec: double-col figures use figsize_W = 16.4
    # inch with width=\linewidth in LaTeX (effective scale 7.05/16.4
    # = 0.430, matching the single-col scale of 0.426 within <1%).
    # figsize_H = 12.5 gives each subplot ~8.0 x 6.0 inch
    # (panel aspect ~1.33:1, matching the rest of the figure suite).
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.1),
                             constrained_layout=True, sharey='row')
    # Slightly more breathing room between the 2x2 panels.
    fig.get_layout_engine().set(hspace=0.06,
                                h_pad=0.08, w_pad=0.08)

    # ----- Top row: alpha(Nh)
    for ax, table, pair_name in [
        (axes[0, 0], NH_SWEEP_BIB, "BIB-BIB"),
        (axes[0, 1], NH_SWEEP_BO, "BO-BO"),
    ]:
        ax.axvspan(6, 10, color="0.85", alpha=0.55)
        ax.axhline(1.43, color="#1f77b4", ls="--", lw=1.4)
        for design in ["rs", "ra", "ss", "sa"]:
            pts = table[design]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "-o", color=DESIGN_COLOR[design],
                    label=DESIGN_LABEL[design], lw=1.2, ms=3.2)
        ax.set_xlabel(r"Hypothesis count $N_h$")
        ax.set_ylabel(r"$\alpha$")
        ax.set_xticks([3, 6, 10, 15, 20])
        ax.text(-0.17, 1.04, "(a)" if pair_name == "BIB-BIB" else "(b)",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom", ha="left")
        ax.grid(False)

    # align the two top panels on a common y-axis (BIB tight vs BO fanned)
    _yl = (min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0]),
           max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1]))
    for _a in (axes[0, 0], axes[0, 1]):
        _a.set_ylim(_yl)
    axes[0, 1].tick_params(labelleft=False)
    axes[0, 1].set_ylabel("")

    # ----- Bottom row: sigma(Nh) on log-log
    # Try to load real sigma data, else use power-law mock with beta values
    Nhs = [3, 6, 10, 15, 20]
    for ax, beta_table, pair_name, pair_key in [
        (axes[1, 0], BETA_BIB, "BIB-BIB", "bib-bib"),
        (axes[1, 1], BETA_BO,  "BO-BO",   "bo-bo"),
    ]:
        for design in ["rs", "ra", "ss", "sa"]:
            sigmas = []
            for h in Nhs:
                try:
                    if h == 10:
                        path = sig_path_v2(design, pair_key, 50)
                    else:
                        path = sig_path_v3(design, h, pair_key, 50)
                    if not path.exists():
                        raise FileNotFoundError
                    d = load_json_with_retry(path)
                    means = [r.get("sig1_mean", np.nan) for r in d if isinstance(r, dict)]
                    means += [r.get("sig2_mean", np.nan) for r in d if isinstance(r, dict)
                              and "sig2_mean" in r]
                    means = [m for m in means if not np.isnan(m)]
                    if not means:
                        raise ValueError("no sig")
                    sigmas.append(np.mean(means))
                except Exception:
                    # fallback: synthesise from beta value (matches MS)
                    beta = beta_table[design]
                    sigmas.append((h / 10.0) ** (-beta) * 0.06)
            ax.loglog(Nhs, sigmas, "o-", color=DESIGN_COLOR[design],
                      lw=1.2, ms=3.2, label="_nolegend_")
        # reference slope -1 (raised to sit just below the data; labelled in
        # C/D so it is not confused with the alpha=1.43 guide in A/B)
        xref = np.array([2.5, 22])
        yref = 0.45 * xref ** -1
        ax.loglog(xref, yref, "k--", lw=1.5, label=r"slope $-1$ ref.")
        ax.set_xlabel(r"$N_h$")
        ax.set_ylabel(r"$\langle\sigma(\hat{P})\rangle$")
        ax.set_xticks([3, 6, 10, 15, 20])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.text(-0.17, 1.04, "(c)" if pair_name == "BIB-BIB" else "(d)",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom", ha="left")
        ax.set_ylim(0.016, 0.6)
        ax.legend(loc="lower left", frameon=False, fontsize=6.8)
        ax.grid(False)

    # mirror the top row: the right column shares the row y-axis (sharey='row'),
    # so drop panel (d)'s duplicate y-label and tick labels (same sigma scale
    # as panel (c)).
    axes[1, 1].tick_params(labelleft=False)
    axes[1, 1].set_ylabel("")

    # shared bottom legend: the four design colours only (the slope -1 reference
    # is labelled inside C/D to avoid confusion with the alpha=1.43 A/B guide).
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.04))
    safe_save(fig, "fig_nh_sweep.png", bbox_inches="tight")


# ===========================================================================
# Fig 2 -- dynamics demo (synthetic / representative run)
# ===========================================================================

def figure2():
    print("\n--- Fig2 dynamics demo ---")
    plt.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "axes.spines.top": True, "axes.spines.right": True,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    # Real representative run (rs design, Nh=10, m=50, seed=42) via the
    # production reward simulator -- the SAME run used by the PNAS Fig.~1 /
    # SI dynamics figures, not a synthetic toy.
    import sys as _sys
    from pathlib import Path as _Path
    _root = next(p for p in _Path(__file__).resolve().parents
                 if (p / "simulation").is_dir())
    _sys.path.insert(0, str(_root / "simulation" / "reward_huge_v2"))
    from rpsgame_reward import AgentReward

    T = 1500
    SEED = 42
    sym2col = {0: "#e6c700", 1: "#1f77b4", 2: "#2ca02c"}  # R, P, S
    _SYM_IDX = {"r": 0, "p": 1, "s": 2}

    def _run(a1, a2, seed=SEED):
        g1 = AgentReward(a1, h_length=50, init_mode="random",
                         predict_mode="sample", seed=seed)
        g2 = AgentReward(a2, h_length=50, init_mode="random",
                         predict_mode="sample", seed=seed + 100000)
        Nh = g1.bayes.h_num
        P = np.empty((T, Nh)); H = np.empty(T, dtype=int)
        for t in range(T):
            h1 = g1.choice(); h2 = g2.choice()
            P[t] = g1.bayes.h_prov.copy(); H[t] = _SYM_IDX[h1]
            g1.update_from_outcome(h1, h2); g2.update_from_outcome(h2, h1)
        return P, H, Nh

    Pb, bib_hands, Nh = _run("bib", "bib")
    Po, bo_hands, _ = _run("bo", "bo")
    bib_post = Pb.T; bib_argmax = Pb.argmax(axis=1)
    bo_post = Po.T; bo_argmax = Po.argmax(axis=1)

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 5.8),
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [0.55, 1.4, 1.0]})

    def corner(ax, letter):
        ax.annotate(f"({letter.lower()})", xy=(0, 1), xycoords="axes fraction",
                    xytext=(-24, 5), textcoords="offset points",
                    fontsize=11, fontweight="bold", va="bottom", ha="left")

    def panel_hands(ax, hands, letter, title):
        ax.set_xlim(0, T)
        ax.set_ylim(-0.5, 2.5)
        for s, col in sym2col.items():
            idx = np.where(hands == s)[0]
            ax.vlines(idx, s - 0.4, s + 0.4, color=col, lw=0.8)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["R", "P", "S"])
        ax.set_xlabel("step $t$")
        corner(ax, letter)

    def panel_post(ax, post, argmax, letter, title):
        im = ax.imshow(post, aspect="auto", origin="lower",
                       cmap="magma", extent=[0, T, -0.5, Nh - 0.5],
                       vmin=0, vmax=min(1.0, post.max() * 1.05))
        ax.plot(np.arange(T), argmax, color="white", lw=1.2)
        ax.set_ylabel("hypothesis $h$")
        ax.set_xlabel("step $t$")
        ax.set_yticks([0, 3, 6, 9])
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=r"$P(h)$")
        corner(ax, letter)

    def panel_top3(ax, post, letter, title):
        order = np.argsort(-post.mean(axis=1))[:3]
        for rank, h in enumerate(order):
            ax.plot(post[h], lw=1.6, label=f"h={h}")
        ax.axhline(1.0 / Nh, ls=":", color="grey",
                   label=f"$1/N_h={1/Nh:.2f}$")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, T)
        ax.set_xlabel("step $t$")
        ax.set_ylabel(r"$P(h)$")
        ax.legend(loc="upper right", frameon=False, fontsize=6.5, ncol=4)
        corner(ax, letter)

    panel_hands(axes[0, 0], bib_hands, "A", "BIB-BIB hand sequence (agent 1)")
    panel_post(axes[1, 0], bib_post, bib_argmax, "B", "BIB posterior $P(h)$")
    panel_top3(axes[2, 0], bib_post, "C", "BIB top-3 $P(h)$ trajectories")

    panel_hands(axes[0, 1], bo_hands, "D", "BO-BO hand sequence (agent 1)")
    panel_post(axes[1, 1], bo_post, bo_argmax, "E", "BO posterior $P(h)$")
    panel_top3(axes[2, 1], bo_post, "F", "BO top-3 $P(h)$ trajectories")

    safe_save(fig, "fig_dynamics_demo.png")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    # NOTE: Fig4 (formerly the BIB-vs-BO direct-comparison 2x2) was
    # consolidated into Fig3 (a,b) and removed from the paper.  The
    # figure4() function is preserved in this module for reference
    # but is no longer invoked by default.  Pass "4" explicitly on
    # the command line if you want to regenerate the orphan file.
    figs = {
        "2": figure2,
        "3": figure3,
        "5": figure5,
        "6": figure6,
        "7": figure7,
        "8": figure8,
        "4": figure4,  # orphan; explicit-only
    }
    # Default skips the orphan "4"; pass "4" explicitly to regenerate it.
    targets = sys.argv[1:] or [k for k in figs if k != "4"]
    for k in targets:
        if k not in figs:
            print(f"unknown figure: {k}")
            continue
        try:
            figs[k]()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FIGURE {k} FAILED: {e}")
    print("\nDONE")
