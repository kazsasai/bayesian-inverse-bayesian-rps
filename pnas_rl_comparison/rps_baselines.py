#!/usr/bin/env python3
"""
rps_baselines.py  --  standard RL baselines for the PNAS RL-comparison control.

Implements WSLS, tabular Q-learning, and regret matching with the SAME agent
interface as rpsgame_reward.AgentReward (choice / update_from_outcome / argmax_h /
likelihood_spread), and a harness run_pair_objs() that mirrors
rpsgame_reward.run_pair() exactly but takes agent OBJECTS. Encoding (HANDS) and
outcome rule (rps) are imported from rpsgame_reward so the baselines play the
identical game as the BIB agent.

Observable convention (argmax_h):
  - regret matching : argmax of cumulative regret  (internal preference)
  - Q-learning, WSLS: index of the played action    (no separate internal pref.)
So extract_metrics(df)['T_argmax1'] is the argmax-persistence used in the figure.
"""
import os
import sys

import numpy as np
import pandas as pd

# Make the BIB simulator importable so the baselines play the IDENTICAL game
# (HANDS encoding + rps outcome rule) as the published BIB agent.
_RW = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "simulation", "reward_huge_v2")
if _RW not in sys.path:
    sys.path.insert(0, _RW)
from rpsgame_reward import HANDS, rps  # identical encoding + outcome rule

PAYOFF = {"win": 1.0, "defeat": -1.0, "quits": 0.0}
BR = {"r": "p", "p": "s", "s": "r"}   # best response (the hand that beats key)
_IDX = {h: i for i, h in enumerate(HANDS)}


class WSLS:
    """Deterministic win-stay / lose-or-tie-shift to best response."""
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.last_result = None
        self.last_opp = None
        self.prev_hand = None
        self._last = None

    def choice(self):
        if self.last_result is None:
            self._last = HANDS[self.rng.integers(3)]
        elif self.last_result == "win":
            self._last = self.prev_hand
        else:  # defeat or quits
            self._last = BR[self.last_opp]
        return self._last

    def update_from_outcome(self, my_hand, opp_hand):
        self.last_result = rps(my_hand, opp_hand)
        self.last_opp = opp_hand
        self.prev_hand = my_hand

    def argmax_h(self):
        return _IDX[self._last]

    def likelihood_spread(self):
        return 0.0


class QLearning:
    """Tabular Q-learning, recall 1, state=(my_last,opp_last), epsilon-greedy."""
    def __init__(self, seed=None, lr=0.1, gamma=0.9, eps=0.1):
        self.rng = np.random.default_rng(seed)
        self.lr, self.gamma, self.eps = lr, gamma, eps
        self.Q = {}
        self.state = ("start", "start")
        self._zeros = np.zeros(3)
        self._a = 0
        self._last = None

    def choice(self):
        if self.rng.random() < self.eps:
            self._a = int(self.rng.integers(3))
        else:
            q = self.Q.get(self.state, self._zeros)
            self._a = int(np.argmax(q))
        self._last = HANDS[self._a]
        return self._last

    def update_from_outcome(self, my_hand, opp_hand):
        r = PAYOFF[rps(my_hand, opp_hand)]
        nxt = (my_hand, opp_hand)
        q = self.Q.setdefault(self.state, np.zeros(3))
        qn = self.Q.get(nxt, self._zeros)
        q[self._a] += self.lr * (r + self.gamma * float(np.max(qn)) - q[self._a])
        self.state = nxt

    def argmax_h(self):
        return self._a   # played-action proxy (no clean global internal pref.)

    def likelihood_spread(self):
        return 0.0


class RegretMatching:
    """Vanilla regret matching; play the CURRENT strategy sigma = R+/sum(R+)."""
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.R = np.zeros(3)
        self._last = None

    def choice(self):
        Rpos = np.maximum(self.R, 0.0)
        s = Rpos.sum()
        sigma = (Rpos / s) if s > 0 else np.ones(3) / 3.0
        idx = int(self.rng.choice(3, p=sigma))
        self._last = HANDS[idx]
        return self._last

    def update_from_outcome(self, my_hand, opp_hand):
        u = np.array([PAYOFF[rps(a, opp_hand)] for a in HANDS])
        self.R += u - u[_IDX[my_hand]]

    def argmax_h(self):
        return int(np.argmax(self.R))   # argmax cumulative (positive) regret

    def likelihood_spread(self):
        return 0.0


class FixedBiased:
    """Non-learning opponent: plays (r,p,s) with fixed probabilities."""
    def __init__(self, seed=None, p=(0.6, 0.2, 0.2)):
        self.rng = np.random.default_rng(seed)
        self.p = np.asarray(p, float)
        self._last = None

    def choice(self):
        self._last = HANDS[int(self.rng.choice(3, p=self.p))]
        return self._last

    def update_from_outcome(self, my_hand, opp_hand):
        pass

    def argmax_h(self):
        return -1

    def likelihood_spread(self):
        return 0.0


def make_agent(kind, seed=None):
    if kind == "wsls":
        return WSLS(seed)
    if kind == "ql":
        return QLearning(seed)
    if kind == "rm":
        return RegretMatching(seed)
    if kind == "biased":
        return FixedBiased(seed)
    if kind == "random":
        from rpsgame_reward import AgentReward
        return AgentReward("random", seed=seed)
    raise ValueError(kind)


def run_pair_objs(a1_kind, a2_kind, n_steps, seed=None):
    """Mirror rpsgame_reward.run_pair() loop, but for baseline agent kinds.
    seed2 = seed + 100000 (same convention as run_pair)."""
    seed2 = (seed + 100000) if seed is not None else None
    ag1 = make_agent(a1_kind, seed)
    ag2 = make_agent(a2_kind, seed2)
    h1 = np.empty(n_steps, dtype="<U1"); h2 = np.empty(n_steps, dtype="<U1")
    res = np.empty(n_steps, dtype="<U10")
    am1 = np.empty(n_steps, dtype=np.int32); am2 = np.empty(n_steps, dtype=np.int32)
    for t in range(n_steps):
        c1 = ag1.choice(); c2 = ag2.choice()
        am1[t] = ag1.argmax_h(); am2[t] = ag2.argmax_h()
        h1[t] = c1; h2[t] = c2
        res[t] = rps(c1, c2)
        ag1.update_from_outcome(c1, c2)
        ag2.update_from_outcome(c2, c1)
    return pd.DataFrame({"t": np.arange(n_steps), "h1": h1, "h2": h2, "res": res,
                         "argmax1": am1, "argmax2": am2,
                         "sig1": np.zeros(n_steps), "sig2": np.zeros(n_steps),
                         "match": h1 == h2})
