#!/usr/bin/env python3
"""Regenerate SI Table `tab-smoothing` (smoothing-implementation comparison)
from re-simulation, so the table is reproducible from deposited code rather
than from a lost cache.

Reuses the verified BayesReward/run_pair from engine_tie_smoothing.py and
monkey-patches ONLY the smoothing rule to the four tabled variants. Runs
serially (so the patch applies; no multiprocessing). Canonical small-scale
config: rs design (random init + sample), BIB-BIB, tie_mode=skip (eq3),
T=2000, burn=1000, 100 runs.

Variants (see Table `tab-smoothing-impl`):
  (1) Ibuka exact        P <- 0.91 P + 0.015           if min P(h) < 0.002
  (2) cond. JM 0.15      P <- 0.85 P + 0.15/Nh          if min P(h) < 0.002
  (3) cond. JM 0.142     P <- 0.858 P + 0.142/Nh        if min P(h) < 0.002
  (4) always-on JM 0.10  P <- 0.90 P + 0.10/Nh          every step

Usage:  python regen_smoothing_table.py
"""
import os, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_tie_smoothing as sim
import powerlaw

SM = {"a": 0.91, "b": 0.015, "always": False}


def patched_inference(self, data):
    obs_idx = int(np.where(self.d_type == data)[0][0])
    likelihoods = self.likelihood[:, obs_idx]
    marginal = float(np.dot(self.h_prov, likelihoods))
    if marginal <= 0.0:
        return
    post = self.h_prov * likelihoods / marginal
    s = post.sum()
    if s > 0 and np.isfinite(s):
        self.h_prov = post / s
    if SM["always"] or (self.h_prov < 0.002).any():
        self.h_prov = self.h_prov * SM["a"] + SM["b"]
        self.h_prov /= self.h_prov.sum()


sim.BayesReward.inference = patched_inference

# name, multiplier a, additive b (= weight/Nh at Nh=10), always-on?
VARIANTS = [
    ("(1) Ibuka exact",       0.91,  0.015,  False),
    ("(2) cond JM 0.15",      0.85,  0.015,  False),
    ("(3) cond JM 0.142",     0.858, 0.0142, False),
    ("(4) always-on JM 0.10", 0.90,  0.010,  True),
]
N_RUNS, N_STEPS, BURN = 100, 2000, 1000
POST = (N_STEPS - BURN) * 2 * N_RUNS  # total post-burn agent-steps (for mean run)

if __name__ == "__main__":
    print(f"re-sim: rs, BIB-BIB, tie_mode=skip, T={N_STEPS}, burn={BURN}, {N_RUNS} runs")
    print(f"{'variant':24s} {'alpha_TPL':>9} {'sigma_bar':>12} {'mean_run':>9} {'n_T':>7}")
    for name, a, b, always in VARIANTS:
        SM.update(a=a, b=b, always=always)
        Targ, sig_means = [], []
        for seed in range(N_RUNS):
            df = sim.run_pair('bib', 'bib', N_STEPS, h_length=50, h_num=10,
                              init_mode='random', predict_mode='sample',
                              tie_mode='skip', defeat_mode='random_other', seed=seed)
            d = df.iloc[BURN:]
            Targ += sim.consecutive_runs(d['argmax1'].values).tolist()
            Targ += sim.consecutive_runs(d['argmax2'].values).tolist()
            sig_means.append(float(np.mean(d['sig1'].values)))
            sig_means.append(float(np.mean(d['sig2'].values)))
        arr = np.array([x for x in Targ if x >= 1], float)
        aT = float(powerlaw.Fit(arr, discrete=True, verbose=False).truncated_power_law.alpha)
        print(f"{name:24s} {aT:9.3f} {np.mean(sig_means):8.4f}+/-{np.std(sig_means):.3f} "
              f"{POST/len(arr):9.1f} {len(arr):7d}")
