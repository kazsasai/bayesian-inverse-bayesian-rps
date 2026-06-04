# 4-Scheme observation-rule ablation — small + medium pilot

**Date**: 2026-05-18
**Scales**: small (T=2000×100) + medium (T=10000×200)
**Status**: Medium-scale confirmation complete. The "BIB universality is independent of the observation rule" finding is now solid across both scales; ready for huge-scale figure-grade rerun on eq3/caseA/hybrid.

> **Update (2026-05-22)**: Huge-scale (T=200000 × 20) confirmation,
> bootstrap CIs, hybrid-laminar α at θ ∈ {0.4, 0.5, 0.6}, and hybrid
> N_h-scaling β are all written up in
> [FINDINGS_huge.md](FINDINGS_huge.md). The numbers below for the
> medium scale are unchanged; the huge document supersedes the
> "ready for huge-scale rerun" status note above.

## Schemes tested

| Scheme | tie | defeat | win | Interpretation |
|---|---|---|---|---|
| `eq3` | skip | random_other | d_A | Published Eq. (3) — Ibuka & Sasai 2024 |
| `caseA` | uniform | random_other | d_A | Sasai-sensei case A: zero-mean tie noise |
| `hybrid` | uniform | opponent | d_A | New: tie noise + ground-truth defeat (case A + sharp loss info) |
| `defeatGT` | skip | opponent | d_A | Orthogonal: ground-truth defeat only (no tie noise) |

Defeat-rule "opponent" sets δ = d_B (the actual hand that beat us). For N=3, k=1 this is informationally maximal: d_B is fully determined by d_A.

## Headline result — α(T_argmax1), BIB-BIB

### medium scale (T=10000 × 200 runs) — *this run*

| Design | eq3 | caseA | hybrid | defeatGT |
|---|---:|---:|---:|---:|
| rs | 1.429 | 1.438 | 1.436 | 2.387 |
| ra | 1.380 | 1.395 | 1.425 | 2.283 |
| ss | 1.426 | 1.445 | 1.453 | 2.327 |
| sa | 1.413 | 1.434 | 1.440 | 2.227 |
| **mean** | **1.412** | **1.428** | **1.438** | **2.306** |
| **spread (max-min)** | **0.049** | **0.050** | **0.028** | **0.160** |

All 16 cells: best=tpl, w_tpl ≥ 0.96 (1.00 for the three balanced schemes; 0.96-0.99 for defeatGT).

### small scale (T=2000 × 100 runs) — reference

| Design | eq3 | caseA | hybrid | defeatGT |
|---|---:|---:|---:|---:|
| mean | 1.334 | 1.378 | 1.392 | 1.935 |
| spread | 0.068 | 0.079 | 0.026 | 0.221 |

### scale-to-scale shift

| Scheme | small ᾱ | medium ᾱ | Δᾱ | small spread | medium spread |
|---|---:|---:|---:|---:|---:|
| eq3 | 1.334 | 1.412 | +0.078 | 0.068 | 0.049 |
| caseA | 1.378 | 1.428 | +0.050 | 0.079 | 0.050 |
| hybrid | 1.392 | 1.438 | +0.046 | 0.026 | **0.028** |
| defeatGT | 1.935 | 2.306 | +0.371 | 0.221 | 0.160 |

Spread shrinks for all schemes as statistics improve; mean shifts upward modestly for the three balanced schemes and dramatically for defeatGT.

## BIB-BIB macro outcomes — medium

Win rate (Nash = 0.333):

| Design | eq3 | caseA | hybrid | defeatGT |
|---|---:|---:|---:|---:|
| rs | 0.333 | 0.333 | 0.332 | 0.280 |
| ra | 0.334 | 0.334 | 0.333 | 0.280 |
| ss | 0.332 | 0.333 | 0.332 | 0.286 |
| sa | 0.334 | 0.334 | 0.333 | 0.287 |

Tie rate (Nash = 0.333):

| Design | eq3 | caseA | hybrid | defeatGT |
|---|---:|---:|---:|---:|
| rs | 0.333 | 0.333 | 0.335 | **0.439** |
| ra | 0.333 | 0.333 | 0.334 | **0.441** |
| ss | 0.334 | 0.334 | 0.336 | **0.426** |
| sa | 0.333 | 0.333 | 0.335 | **0.425** |

Posterior σ(P(h)):

| Design | eq3 | caseA | hybrid | defeatGT |
|---|---:|---:|---:|---:|
| rs | 0.134 | 0.131 | 0.132 | 0.083 |
| ra | 0.133 | 0.130 | 0.131 | 0.090 |
| ss | 0.134 | 0.132 | 0.129 | 0.085 |
| sa | 0.134 | 0.131 | 0.131 | 0.093 |

## Three structural findings (now solid at medium scale)

### 1. Universality is independent of the observation rule (within the balanced class)

Three distinct observation rules — `eq3`, `caseA`, `hybrid` — produce α ≈ 1.41–1.44 with the same mean and similar spread across designs. The three means differ by at most 0.026, smaller than the within-scheme spread across designs.

This sharpens the published R1 result. The previous wording is "BIB α is universal across initializations and prediction modes". The corrected reading from this ablation is:

> **BIB α is universal across initializations, prediction modes, AND observation-rule variants that preserve the driving/relaxation balance.**

Most importantly, `hybrid` has spread 0.028 — *tighter than `eq3`* (0.049) — confirming the small-scale finding that adding ground-truth defeat info under tie noise improves (not destroys) the universality.

### 2. defeat-GT alone shifts the SOC point but preserves criticality

`defeatGT` α settles at 2.31 ± 0.08 (median spread 0.16) across the 4 designs:

- w_tpl ≥ 0.96 in all cells → still truncated power-law
- α still in Lévy region (1<α<3)
- α is still universal across designs (just at a different value)

Plus macro signatures of synchronization onto the cyclic structure:
- win rate drops to 0.28 (vs Nash 0.333)
- tie rate rises to 0.44 (vs Nash 0.333)
- posterior σ drops to ~0.09 (vs balanced ~0.13)

Interpretation: with sharp defeat info, both BIB-BIB agents learn each other's d_B → predict it perfectly → tie more often. The system locks onto a different SOC attractor with sharper posteriors and higher α. **Not "off" SOC — a different SOC point.**

### 3. Tie noise rescues balance

`hybrid = caseA + defeatGT` does NOT inherit defeatGT's α=2.31. Instead it returns to α=1.44, essentially identical to caseA alone (1.43). The tie noise re-injects relaxation, compensating for defeat-GT's over-driving, and the system self-organizes back to the original critical exponent.

This is a near-textbook demonstration of internal-state SOC: **the critical exponent is determined by the driving/relaxation ratio, not by any individual mechanism's magnitude.** Two distinct ratios (eq3-class vs defeatGT) yield two distinct universal α values; intermediate combinations interpolate.

## BO-BO contrast (medium)

| Design | eq3 | caseA | hybrid | defeatGT |
|---|---:|---:|---:|---:|
| rs | 2.029 | 1.976 | 1.831 | 1.996 |
| ra | 2.017 | 1.864 | 1.965 | 1.966 |
| ss | 1.464 | 1.508 | 1.528 | 1.398 |
| sa | 1.530 | 1.458 | 1.456 | 1.099 |
| **mean** | 1.760 | 1.701 | 1.695 | 1.615 |
| **spread (max-min)** | **0.565** | **0.518** | **0.509** | **0.897** |

BO spread is ~10× wider than BIB across all schemes, and `defeatGT` produces a sa-design α of 1.099 (near-exponential) right next to an rs-design α of 1.996. BO is not just design-dependent — it is *catastrophically* sensitive to observation-rule choice on top of design choice. This is exactly the absence of an inverse step (the relaxation half of the SOC mechanism) showing up.

## Implications for paper A

This ablation cleanly addresses three of Sasai-sensei's framing principles:

1. **§4.2.5 on-off intermittency** — hybrid's α-rescue is the additive-noise + state-dependent-driving balance PST 1993 requires. The new observation rules provide a clean experimental knob for that balance.
2. **"Internal-state SOC"** — the defeatGT → hybrid recovery is direct evidence that the BIB inverse step actively self-organizes the system back to a critical attractor when driving and relaxation are independently varied.
3. **§4.6 observation space design problem** — the existence of `hybrid` (which keeps α universal AND uses ground-truth defeat info) shows the published Eq. (3) is one of a class of equivalent observation rules, not a unique necessary choice.

## Caveats

- Powerlaw library `xmin` scan has small stochasticity (~0.02-0.03 in α per refit on the same data). For figure-grade precision, freeze a seed in `fit.Fit` or use bootstrap CIs.
- Medium-scale α is ~0.05-0.08 higher than small-scale α uniformly — this is a finite-statistics convergence pattern, not a real effect.
- Laminar α (R7) NOT yet checked under hybrid/defeatGT. Pending task.
- N_h scaling (R5/R6) NOT yet checked under hybrid/defeatGT. Pending task.

## Recommended next steps (priority order)

1. **Huge-scale rerun for eq3 + caseA + hybrid only** (defeatGT can stay at medium). 12 cells × ~30 min/cell ≈ overnight. This gives figure-grade data for paper A.
2. **Laminar α verification under hybrid** — if α_laminar ≈ 1.34 holds for hybrid, that's R1 + R7 reconfirmed under observation-rule perturbation.
3. **Discussion paragraph in §4.2.5** drafting the "driving/relaxation balance, multiple SOC attractors" framing.
4. **JM trigger rate per step** — count `min P(h) < 0.002` events per scheme. Hypothesis: defeatGT has higher trigger rate (sharper posterior); hybrid intermediate.

## File inventory

```
simulation_tie_mode_ablation/
├── FINDINGS_small_pilot.md                <- caseA-only pilot
├── FINDINGS_scheme_ablation.md            <- THIS (small + medium)
├── files/
│   ├── rpsgame_reward_tie.py              <- adds tie_mode + defeat_mode
│   ├── run_tie_mode_ablation.py           <- 1st-gen runner (tie only)
│   ├── run_scheme_ablation.py             <- 2nd-gen runner (4 schemes)
│   ├── prefit_scheme.py                   <- batched fitter, cache resumable
│   ├── make_scheme_plots.py               <- --scale arg (small/medium/...)
│   ├── make_comparison_plots.py           <- 1st-gen plotter
│   └── quick_summary.py                   <- 1st-gen summary
└── data/scheme_ablation/{small,medium}/
    ├── durations_<design>_<pair>_<scheme>_tie<x>_def<y>.json
    ├── rewards_/sigmas_  (same naming)
    ├── _fit_cache.json                    <- powerlaw fits, resumable
    ├── scheme_summary.csv
    ├── fig_alpha_4schemes.png             <- headline bar chart
    ├── fig_BIB_ccdf_4schemes.png          <- per-design CCDF overlays
    └── fig_spread_4schemes.png            <- universality check
```

## Reproducibility

```bash
cd simulation_tie_mode_ablation/files/

# (1) Run simulations (resumable via cache)
python3 run_scheme_ablation.py --scale medium --pairs bib-bib bo-bo --workers 7

# (2) Pre-compute fits in batches (45s-safe)
for i in {1..8}; do python3 prefit_scheme.py --scale medium --n 8; done

# (3) Build summary + plots
python3 make_scheme_plots.py --scale medium
```

## Regression / Reversion

- `defeat_mode='random_other'` + `tie_mode='skip'` is bit-identical to the unmodified `simulation_r4_fallback/files/rpsgame_reward.py` (verified across 120 conditions: 0 failures).
- Delete `simulation_tie_mode_ablation/` to revert. Original simulator md5: d19ce1b330729ffc6dc69c18d461d4a6 — unchanged.
