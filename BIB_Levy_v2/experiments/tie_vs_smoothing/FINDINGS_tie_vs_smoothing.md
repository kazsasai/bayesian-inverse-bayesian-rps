# Does tie→uniform replace the conditional JM smoothing? — pilot finding (2026-05-21)

## Hypothesis (Sasai)

Fig 7's `caseA` makes ties produce a **uniformly drawn observation**
(`tie_mode='uniform'`) instead of "no update" (`skip`, Eq. 3). Since ties occur
~1/N ≈ 33% of the time at the uniform fixed point, this injects a large amount of
zero-mean stochastic perturbation into the Bayesian update. Conjecture: that
perturbation might keep the posterior from collapsing — the role the **conditional
Jelinek–Mercer smoothing (Eq. 5)** plays — making the smoothing unnecessary.

## Method

2×2 factorial, isolating exactly the two factors:

- `tie_mode ∈ {skip, uniform}` (eq3 vs caseA; `defeat_mode=random_other` fixed)
- `jm_smoothing ∈ {on, off}` (Eq. 5 gated by a new flag in the engine copy)

Pair = bib–bib; all four designs rs/ra/ss/sa; small scale (T=2000, analyse
[1000,2000), 100 runs, N_h=10, m=50). Code: `engine_tie_smoothing.py` (a copy of
`rpsgame_reward_tie.py` with a `jm_smoothing` flag gating the `min P(h)<0.002 →
P←0.91P+0.015` step) + `build_tie_vs_smoothing.py`. Results: `combined_small.csv`,
figure `fig_tie_vs_smoothing_small.pdf`.

## Result — the hypothesis is NOT supported

| design | scheme | JM | α | σ̄ | n(durations) | state |
|---|---|---|---:|---:|---:|---|
| rs | eq3 (skip) | on | 1.378 | 0.132 | 2845 | Lévy ✓ |
| rs | eq3 (skip) | **off** | 1.047 | 0.287 | 394 | collapsing |
| rs | caseA (uniform) | on | 1.381 | 0.140 | 3059 | Lévy ✓ |
| rs | caseA (uniform) | **off** | 1.189 | 0.288 | 503 | collapsing |
| ra | eq3 | on / off | 1.318 / 1.026 | 0.137 / 0.285 | 2370 / 370 | ✓ / collapsing |
| ra | caseA | on / off | 1.344 / 1.094 | 0.134 / 0.287 | 2966 / 398 | ✓ / collapsing |
| ss | eq3 | on / off | 1.329 / ~0 (exp) | 0.157 / 0.300 | 1662 / 103 | ✓ / **frozen** |
| ss | caseA | on / off | 1.423 / nan | 0.152 / 0.300 | 2281 / 100 | ✓ / **frozen** |
| sa | eq3 | on / off | 1.311 / nan | 0.152 / 0.300 | 1667 / 100 | ✓ / **frozen** |
| sa | caseA | on / off | 1.366 / nan | 0.151 / 0.300 | 1936 / 100 | ✓ / **frozen** |

With JM **on**, both tie rules give the universal Lévy regime (α≈1.31–1.42,
σ̄≈0.13–0.16). With JM **off**, the posterior **over-concentrates onto a single
hypothesis** (σ̄→0.29–0.30; for N_h=10 a single-vertex posterior has σ≈0.30) and
the argmax **freezes** — n collapses toward `n_runs` (≈1 duration/run), i.e. the
argmax index never changes. `tie→uniform` does **not** prevent this: it helps only
marginally for random-init designs (rs α 1.05→1.19, ra 1.03→1.09) and not at all
for structured-init designs (ss/sa freeze completely with or without tie→uniform).

## Interpretation — three regimes, one sweet spot

The conditional smoothing is a genuine **relaxation mechanism** sitting between two
distinct collapse modes:

- **JM off** → *over-concentration*: posterior pins to one vertex, σ̄→0.30, argmax
  frozen (this experiment).
- **JM always-on** → *under-concentration*: posterior smeared toward uniform,
  σ̄→0.015, heavy tail destroyed (Fig 10 / Appendix A).
- **JM conditional** → σ̄≈0.13–0.16, on-off intermittency, α≈1.4 ✓.

So the tie observation rule and the smoothing are **not interchangeable**: the
former perturbs the *driving*, the latter releases the system at the *edge of
collapse*. Removing the smoothing breaks the dynamics regardless of the tie rule.
This is a stronger statement than the paper currently makes and complements the
always-on result in Appendix A — together they bound the conditional smoothing
from both sides.

## Caveats / next step

Small scale only (1000 analysed steps); the JM-on α's run ~1.31–1.42, a bit below
the paper's huge-scale 1.43±0.02 (expected small-scale bias). The *qualitative*
contrast (frozen vs Lévy; σ̄ 0.30 vs 0.14) is unambiguous and decisive for the
hypothesis. To make it paper-grade, re-run the key cells (`uniform` × {on,off}) at
medium/huge scale to confirm the freeze persists and to pin the JM-on α at 1.43.
Run: `python3 build_tie_vs_smoothing.py --scale medium --designs rs ss`.
