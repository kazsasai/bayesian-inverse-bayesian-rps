# Tie-mode ablation — small-scale pilot (Eq. 3 of main_PRE.tex)

**Date**: 2026-05-18
**Scale**: small (T=2000, analyse last 1000, n_runs=100, m=50, N_h=10)
**Status**: Pilot complete. Headline finding supports case-A robustness; ready to escalate to medium scale for paper-grade ablation.

## Question

What happens if Eq. (3) "no update on tie" is replaced with "δ ~ Uniform({r,p,s}) on tie" (case A — zero-mean noise injection)?

## Implementation

- New file: `simulation_tie_mode_ablation/files/rpsgame_reward_tie.py`
- Adds `tie_mode ∈ {'skip', 'uniform'}` parameter to `AgentReward`, `run_pair`, `parallel_runs`, `run_pilot`, `run_grid`.
- `tie_mode='skip'` is default — bit-identical to the original `simulation_r4_fallback/files/rpsgame_reward.py` (regression-tested on 120 seed × pair × design combinations: 0 failures).
- `tie_mode='uniform'` consumes one extra rng draw per draw, samples δ from {r,p,s} uniformly, then runs the standard Bayesian update + (BIB only) inverse step.

## Reversion path

The original simulator is untouched. Delete the directory `simulation_tie_mode_ablation/` to revert. All publication data in `simulation_r4_fallback/`, `simulation_r3_JM0.10/`, etc. is unaffected.

## Results (T_argmax1, headline metric)

### BIB-BIB

| Design | skip α | uniform α | Δα | best (skip) | best (uniform) | w_tpl |
|---|---:|---:|---:|---:|---:|---:|
| rs | 1.378 | 1.381 | +0.003 | tpl | tpl | 1.00 |
| ra | 1.318 | 1.344 | +0.026 | tpl | tpl | 1.00 |
| ss | 1.329 | 1.423 | +0.094 | tpl | tpl | 1.00 |
| sa | 1.311 | 1.366 | +0.055 | tpl | tpl | 1.00 |

- Mean: α(skip) ≈ 1.334, α(uniform) ≈ 1.378
- Spread across 4 designs: σ(α)_skip ≈ 0.030, σ(α)_uniform ≈ 0.039
- All 8 cells: truncated power-law beats both pure PL and exponential (w_tpl = 1.00)
- All 8 cells stay in the Lévy region 1 < α < 3

### BO-BO

| Design | skip α | uniform α | Δα |
|---|---:|---:|---:|
| rs | 1.908 | 1.799 | −0.108 |
| ra | 1.727 | 1.949 | +0.222 |
| ss | 1.490 | 1.499 | +0.009 |
| sa | 1.294 | 1.536 | +0.242 |

- BO α range much wider (1.29–1.95) — design-dependent, as expected (R2)
- BO Δα fluctuates in sign and magnitude (–0.11 to +0.24)
- Tie-noise destabilizes BO more than BIB

### Reward & posterior structure (BIB-BIB)

- Win rate stays at Nash (≈ 1/3) in both modes for all designs
- Tie rate stays at ≈ 1/3 (Nash) in both modes — the random injection doesn't perturb the macro outcome distribution
- Posterior spread σ(P(h)) is comparable between modes (≈ 0.13–0.16 in both)

## Interpretation

**Universality survives case-A perturbation.** All 4 designs give BIB α in the narrow range [1.31, 1.42] regardless of tie mode. Δα is small and consistently positive, with magnitude < 0.10 in all designs. The shape (truncated power-law, Lévy region) is preserved everywhere.

**BIB-BO contrast is preserved.** BO α remains design-dependent under both tie modes, and its Δα reacts erratically to tie-noise (sign flips, much larger magnitudes than BIB). This is the expected "non-critical absorbs noise into the exponent" behavior, vs BIB's "critical absorbs noise into the same exponent".

**SOC hypothesis supported.** If the BIB universality were a brittle artifact of the specific "skip on tie" rule, we would expect α to shift dramatically or the shape to change qualitatively under case-A. Instead, α shifts by < 7% on average, and the truncated-PL shape with Lévy α persists. This is consistent with internal-state SOC: an additive zero-mean perturbation drifts the operating point but does not destroy the critical attractor.

**Mild but real shift Δα > 0.** The +0.05 mean shift is small but consistent across 3/4 designs (rs is essentially zero). A plausible reading: the additional noise injection slightly accelerates posterior relaxation toward uniform, which shortens the long tail and increases α modestly. This is the SOC prediction with the relaxation rate slightly increased.

## Caveats

- Small scale: T=2000 × 100 runs ⇒ noisy α estimates (paper huge scale is T=200000 × 20 runs).
- Single seed-base (0). For tight CIs, would need 5–10 repeats with different seed bases.
- Only T_argmax1 fitted; laminar phase α and σ(P) scaling not yet verified under case A.

## Recommended next steps (in priority order)

1. **Medium-scale rerun** (T=10000 × 200 runs ≈ 5× more statistics) — run on Sasai-sensei's laptop overnight.
2. **Laminar phase α (R7)**: re-extract laminar durations with max P(h) > 0.4 threshold, compare across tie modes.
3. **JM trigger rate**: count how often `min P(h) < 0.002` fires per step in each mode. Hypothesis: tie-noise lowers trigger rate.
4. **BIB-random and BO-random pairs**: ablate against Nash baseline.
5. If medium results confirm: add a 1-paragraph robustness note to §3.2 or §4.2.5 of main_PRE.tex citing this as supporting evidence for SOC universality.

## File inventory

```
simulation_tie_mode_ablation/
├── FINDINGS_small_pilot.md              <- this file
├── files/
│   ├── rpsgame_reward_tie.py            <- ablation variant (extends rpsgame_reward.py)
│   ├── run_tie_mode_ablation.py         <- main runner (cached, resumable)
│   ├── quick_summary.py                 <- fast summary from cached JSONs
│   └── make_comparison_plots.py         <- generate 4-panel CCDF + α bars
└── data/ablation/small/
    ├── durations_<design>_<pair>_tie<mode>.json   (16 cells)
    ├── rewards_<design>_<pair>_tie<mode>.json
    ├── sigmas_<design>_<pair>_tie<mode>.json
    ├── quick_summary.csv
    ├── summary_ablation_small.csv
    ├── fig_BIB_ccdf_4panel.png         <- headline plot
    ├── fig_alpha_bars.png              <- α bar chart
    ├── fig_alpha_spread.png            <- universality visualization
    └── compare_BIB_<design>.png        <- per-design overlays
```

## Reproducibility

```bash
# Run from inside simulation_tie_mode_ablation/files/
python3 run_tie_mode_ablation.py --scale small \
    --pairs bib-bib bo-bo --workers 3

# Then summary + plots
python3 quick_summary.py
python3 make_comparison_plots.py
```

Re-running with `--scale medium` or `--scale huge` will reuse cached small-scale data and only run the new scale's cells.
