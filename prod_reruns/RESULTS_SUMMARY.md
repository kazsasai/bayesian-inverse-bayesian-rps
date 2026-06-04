# Production re-runs — results summary (FULL SCALE, LOCKED)

All on the real `reward_huge_v2` engine (windowed histogram, conditional smoothing,
exact reward-observation rule). **Full scale: T = 2e5 per run, 20 runs (40 = both agents
pooled).** These are the locked values; runners + JSON caches in this folder.
**Status: integrated into the manuscript (2026-06-01).**

## Prong 2 — self-organisation to zero drift (run_prong2_prod.py)
Within-phase drift <eta> = mean step change of log[P(top)/P(2nd)] while argmax unchanged,
vs renewal probability q (q=0 Bayes-only, q=1 full BIB), n=40:

| q | 0.0 (BO) | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 (BIB) |
|---|---|---|---|---|---|---|
| <eta> | **+0.00437 ± 0.00076** | +6.0e-5 | +2.8e-5 | +1.1e-5 | -1.1e-5 | **-1.4e-5 ± 0.6e-5** |

CONFIRMED: Bayes-only has a positive concentration drift (+0.0044); the renewal cancels it
to ~0 (<1e-4) for every q>=0.05. The presence of the renewal, not its rate, sets the
zero-drift line. -> integrated as one prose sentence in SI Sec 1 (no figure; step-like, not
the prototype's gradual curve).

## Prong 1 — invariance to the form of the inverse step (run_prong1_prod.py)
Laminar exponent (max_h P>0.4; truncated-power-law fit, Clauset auto-x_min), n=40:

Renewal-preserving forms:
| form | argmin (ref) | soft-min | two-lowest | 50%-blend | cadence-1/2 |
|---|---|---|---|---|---|
| alpha_lam | 1.324 | 1.296 | 1.180 | 1.340 | 1.340 |
| mean laminar len | 42 | 43 | 24 | 44 | 34 |

Controls (remove the directed recent-empirical renewal):
| control | Bayes-only | noise-injection |
|---|---|---|
| alpha_lam | 1.584 | 1.517 |
| mean laminar len | 5 | 6 |

CONFIRMED: renewal-preserving forms stay in a narrow low band alpha_lam in [1.18, 1.34]
with long laminar phases (24-44); controls separate (1.52-1.58) with short phases (5-6).
(Note: the two-lowest target is the most aggressive renewal and sits lowest at 1.18, still
in the on-off/Levy class.) -> integrated as SI Sec 3 paragraph + Table (si:tab-forms), and
one main-text Discussion sentence (Secs 3 and 12).

## Item 2 — distinction from standard resampling (run_item2_prod.py), n=40
| method | BIB (directed) | prior-restart | SIR (ESS resample) |
|---|---|---|---|
| mean laminar len | 40.5 | 5.6 | **0 (no phases)** |
| alpha_lam | 1.324 | 1.350 | — |

CONFIRMED: only the directed renewal sustains long on-off laminar phases; prior-restart only
short episodes; SIR sustains NONE (posterior stays mixed, like Bayes-only). The argmax tail
is heavy for all (not diagnostic); the laminar observable discriminates. -> integrated as a
new SI section "Distinction from standard resampling" (placed at END = SI Sec 12, no
renumbering) + Table (si:tab-resampling) + SIR ref (gordon1993).

## Manuscript impact
main now 7pp (the 7th page is ~4 references spilling; within PNAS 6-preferred/12-max);
SI now 14pp, 12 sections. Both build clean (0 errors/undefined). Laminar-primary framing kept.

## Brockbank & Vul behavioural re-analysis — DONE (integrated)
Condition-level gradient VERIFIED here from rps_bib_conditions.csv: Spearman
rho(win_rate, mean transition-run)=+0.854 (p=1e-4); fixed/exploitable (v2) condition mean
2.79 > adaptive (v3) 1.85; TPL preferred over EXP in all 15 conditions (R>0). Integrated as
new SI Sec 13 + figure (figS_bib_behaviour.pdf) + one main Discussion sentence; cited as
Brockbank & Vul (2024), Cognitive Psychology 151:101654, doi:10.1016/j.cogpsych.2024.101654.
NOTE: raw dataset not fetchable here (web restrictions); the prior session's per-condition
powerlaw fits used x_min=1 (not the paper's Clauset auto-x_min), so the SI text rests on the
convention-independent gradient/contrast. A full Clauset-convention re-fit + per-game stats
(Mann-Whitney etc.) would need the raw B&V data (their public repo's v2/v3 JSON game files).

## How to reproduce (full scale)
From this folder (engine auto-located; or set ENGINE_DIR):
    NR=20 T=200000 BURN=100000 python3 run_prong2_prod.py 0.0,0.05,0.1,0.25,0.5,1.0
    NR=20 T=200000 BURN=100000 python3 run_prong1_prod.py ref,softmin,bottom2,blend,cadence,bo,noise
    NR=20 T=200000 BURN=100000 python3 run_item2_prod.py bib,restart,sir
