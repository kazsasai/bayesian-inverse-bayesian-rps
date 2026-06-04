# Internal-state criticality vs behavioural expression (2026-05-21)

## The question (Sasai)

Prior empirical BIB work reports **behavioural Levy walks** (foraging/swarm
displacement), yet this RPS model shows no behavioural Levy. Intuitively, in
bib-bib play a small empirical bias ought to grow over time. Why doesn't it?

## Finding 1 — the heavy tail is internal-only at equilibrium

bib-bib, rs, JM-on, tie=skip, stationary window [1000,2000), 150 runs:

| observable | n | best fit | exponent |
|---|---:|---|---:|
| **internal: argmax persistence** | 4254 | **tpl (power law)** | α≈1.36 (Levy) |
| behaviour: hand-repeat runs | 99359 | exponential-like (tpl, α≈0.06, cutoff-dominated) | — |
| behaviour: outcome (w/t/l) runs | — | exponential | — |
| behaviour: win / win-or-draw / defeat streaks | — | exponential | — |

So the power-law/Levy signature lives **only** in the internal hypothesis
dynamics; every behavioural streak statistic is exponential. This is not a
measurement artefact — it is a clean decoupling. (Note: Shinohara's Symmetry
2021 also finds its power law in the *internal* max-confidence duration, not in
behaviour, so this is consistent with that lineage; the *behavioural* Levy of
the foraging/swarm papers comes from a different readout map.)

## Why — readout map + adversarial pinning

1. **Readout map.** Foraging/swarm: action = spatial displacement, a *low-
   entropy* readout of the persistent hypothesis (d∼N(μ*,Σ*)). Persistence ->
   short steps; switching -> long jumps; the internal tail is transcribed into
   displacement. RPS: action = a symbol from {r,p,s} sampled from P(d|h), which
   at the Nash point is near-uniform for every hypothesis -> a *high-entropy*,
   nearly hypothesis-independent readout that erases the temporal correlation.

2. **Adversarial pinning (cyclic dominance).** The intuition "small biases grow"
   holds in non-competitive settings. RPS is zero-sum with a unique uniform Nash:
   any persistent bias is exploitable, the opponent best-responds and erases it,
   and the fixed point of mutual adaptation is uniform play. Biases are competed
   away in the action *marginal*; the beliefs still churn (power law) but the
   behaviour is clamped to Nash.

## Finding 2 — remove the pinning and behavioural Levy re-appears

Pinning is loosened only when the opponent (i) does not compete away biases and
(ii) carries an exploitable bias. A uniform-random opponent fails (ii):
bib-random behaviour stays exponential. Against a **fixed biased opponent
(0.6,0.2,0.2)**:

| observable | best fit | exponent |
|---|---|---:|
| internal: argmax persistence | tpl | α≈1.34 (still Levy) |
| **behaviour: hand-repeat runs** | **tpl (power law)** | **α≈1.28** |
| hand frequency r/p/s | — | **0.22 / 0.62 / 0.16** (strongly biased; uniform when pinned) |

The bib agent locks onto an exploiting hypothesis, P(d|h*) concentrates, the
action marginal becomes biased, and the hand-run tail turns from exponential
into a power law. A periodically-switching (non-stationary) opponent likewise
makes hand-runs (α≈1.6) and win-streaks (α≈0.5) power-law while the internal
persistence becomes exponential (external forcing dominates the timescale).

## Take-away for the paper

Do **not** claim BIB produces behavioural Levy walks here. Claim: BIB produces
**SOC in the internal hypothesis dynamics**; whether it expresses behaviourally
depends on (a) the readout map (low- vs high-entropy) and (b) adversarial
pinning. Competitive RPS is the ideal minimal substrate precisely because it
*decouples* the two, isolating the inference-primitive criticality. This
resolves the apparent tension with the foraging/swarm Levy literature and
pre-empts the reviewer question "where is the behavioural Levy?".

Figure: `fig_internal_vs_behavior.pdf` (panels (a) equilibrium decoupling,
(b) pinning removed). Scripts: `gen_internal_vs_behavior.py`,
`plot_internal_vs_behavior.py`. Data: `data_ivb_{equil,biased}.npz`.
