# TASK 4: BIB generality off RPS — matching pennies (internal exponents)

Same inference core as the paper (BayesReward: Bayes update + conditional JM + inverse-Bayesian
argmin-renewal from recent-m histogram), ONLY the game swapped to matching pennies (2 actions,
Nash=uniform, zero-sum, no draw; reward-based Eq.3 obs: win->own symbol, defeat->other symbol;
agent1=matcher, agent2=mismatcher). Observables identical to RPS (argmax-persistence; laminar
max_h P(h)>0.4), pooled over both agents. MODERATE scale T=4e4, burn 8e3, 10 runs/design.

Results (truncated power law, Clauset x_min):
  random+sample : argmax alpha_TPL = 1.454 (n=11664), laminar alpha_TPL = 1.366 (n=5134)
  random+argmax : argmax alpha_TPL = 1.468 (n=13928), laminar alpha_TPL = 1.409 (n=4683)
  all TPL >> EXP (loglik-ratio R = 30-34, p~0).
RPS reference: argmax ~1.43, laminar ~1.34.

CONCLUSION: the BIB inference self-organises to the SAME critical class on a different
Nash-uniform zero-sum game with no cyclic structure. The exponent is NOT RPS-specific.
=> supports reframing the paper as a general self-organized-critical inference mechanism
(RPS as a controlled testbed), and pre-empts the "why RPS / RPS-specific" referee objection.

Reproduce: python mp_bib_prototype.py   (PREDICT=sample then argmax) ; python mp_bib_prototype.py --fit
Full-scale lock (recommended for the manuscript): set T=200000 NR=20 and re-run (resumable),
exactly as the laminar lock; the moderate-scale alphas above should sharpen but stay in-class.
