#!/usr/bin/env python3
r"""Unified fitting + validation for the RL-baseline control (SI Appendix).

Fits every cell of the control through ONE pipeline that matches the paper's
Methods (Clauset-Shalizi-Newman x_min + Akaike model comparison) and adds the
two guards the handoff requires:

  1. Model comparison includes STRETCHED-EXPONENTIAL and LOGNORMAL, not just
     PL-vs-EXP.  (A bare PL-vs-EXP test at small x_min mislabels a stretched
     exponential -- e.g. Q-learning -- as a "power law".)
  2. A "critical / 3/2-class" verdict requires BOTH
        (i)  a power-law range >= ~2 decades from x_min to the cutoff, AND
        (ii) a small gap between the pure-PL and TPL-interior exponents (<= 0.2).
     A TPL-interior fit over ~1 decade with a large PL/TPL gap is an artifact.

All exponents are reported on the TPL-interior basis; alpha_TPL is quoted only
where TPL (or PL) is Akaike-preferred over EXP/stretched/lognormal.

DATA SOURCES
  BIB (validated local data; do NOT reimplement BIB):
    self-play (vs another BIB) : durations_bib-bib_m50_huge.json -> pool
                                 T_argmax1 + T_argmax2 (both agents are BIB).
    vs uniform-random          : durations_bib-random_m50_huge.json -> T_argmax1
                                 ONLY.  T_argmax2 is the non-learning opponent's
                                 side (argmax never switches -> a degenerate
                                 20x10^5 spike that inflates the tail to a fake
                                 ~1.53 with a cutoff at the analysis window).
  Baselines: baseline_dwells.npz from run_baseline_control.py.

Writes validation_baseline_control.json next to the data.

Usage:
    python analyze_baseline_control.py \
        [--bib-dir ~/paperA_data_full/simulation/reward_huge/data] \
        [--baseline data/baseline_dwells.npz]
"""
import argparse
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import powerlaw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGNS = ("rs", "ra", "ss", "sa")
DEFAULT_BIB = os.path.expanduser("~/paperA_data_full/simulation/reward_huge/data")

# candidate distributions and their parameter counts (for AIC)
_CANDS = {"power_law": 1, "exponential": 1, "truncated_power_law": 2,
          "lognormal": 2, "stretched_exponential": 2}
_SHORT = {"power_law": "PL", "exponential": "EXP", "truncated_power_law": "TPL",
          "lognormal": "LOGN", "stretched_exponential": "STREXP"}


def _loglik(dist, x):
    if dist is None:
        return None
    try:
        p = np.asarray(dist.pdf(x), float)
        p = np.where(p > 0, p, 1e-300)
        return float(np.sum(np.log(p)))
    except Exception:
        return None


def fit_unified(data, n_boot=60, seed=0):
    """Full unified fit + criticality verdict for one 1-D run-length sample."""
    d = np.asarray(data, float)
    d = d[d >= 1]
    out = {"n": int(d.size), "max": int(d.max()) if d.size else 0}
    if d.size < 200 or np.unique(d).size < 5 or d.max() < 5:
        out["verdict"] = "insufficient range"
        return out

    # Subsample only very large samples so the CSN x_min sweep is tractable
    # (distribution shape preserved; true n kept for reporting).  Headline
    # critical cells (BIB ~1.5-3e5, RM-regret ~2e4) are below CAP -> full data;
    # only the multi-million-event non-critical action arrays are thinned.
    CAP = 400_000
    if d.size > CAP:
        d = np.random.default_rng(seed).choice(d, size=CAP, replace=False)
        out["n_fit"] = CAP

    fit = powerlaw.Fit(d, discrete=True, verbose=False)
    xmin = float(fit.xmin)
    tail = d[d >= xmin]
    xmax = float(d.max())
    decades = float(np.log10(xmax / xmin)) if xmin > 0 else 0.0

    # log-likelihoods over the common tail (data >= xmin)
    ll = {}
    for name in _CANDS:
        ll[name] = _loglik(getattr(fit, name, None), tail)
    ll = {k: v for k, v in ll.items() if v is not None}
    aic = {k: 2 * _CANDS[k] - 2 * v for k, v in ll.items()}
    amin = min(aic.values())
    raw = {k: np.exp(-(a - amin) / 2) for k, a in aic.items()}
    tot = sum(raw.values())
    weights = {k: v / tot for k, v in raw.items()}
    best = max(weights, key=weights.get)

    alpha_pl = float(fit.power_law.alpha)
    alpha_tpl = float(fit.truncated_power_law.alpha)
    gap = abs(alpha_pl - alpha_tpl)

    # pairwise discriminators (normalized loglik ratio; >0 favors first)
    def cmp(a, b):
        try:
            R, p = fit.distribution_compare(a, b, normalized_ratio=True)
            return round(float(R), 2), round(float(p), 3)
        except Exception:
            return None, None
    R_tpl_exp, p_tpl_exp = cmp("truncated_power_law", "exponential")
    R_tpl_str, _ = cmp("truncated_power_law", "stretched_exponential")
    R_tpl_logn, _ = cmp("truncated_power_law", "lognormal")
    R_pl_exp, _ = cmp("power_law", "exponential")

    # bootstrap CI on the TPL-interior exponent
    ci = None
    if tail.size >= 50:
        rng = np.random.default_rng(seed)
        bt = tail if tail.size <= 20_000 else rng.choice(tail, 20_000, replace=False)
        boots = []
        for _ in range(n_boot):
            bs = rng.choice(bt, size=bt.size, replace=True)
            try:
                fb = powerlaw.Fit(bs, discrete=True, xmin=xmin, verbose=False)
                boots.append(float(fb.truncated_power_law.alpha))
            except Exception:
                pass
        if len(boots) >= 20:
            ci = [round(float(np.percentile(boots, 2.5)), 3),
                  round(float(np.percentile(boots, 97.5)), 3)]

    # criticality verdict (handoff §5): heavy-tail-preferred AND >=2 decades AND gap<=0.2
    heavy = best in ("truncated_power_law", "power_law") and (R_tpl_exp or 0) > 0 \
        and (R_tpl_str is None or R_tpl_str > 0) and (R_tpl_logn is None or R_tpl_logn > 0)
    if best == "exponential":
        verdict = "non-critical (exponential)"
    elif best in ("stretched_exponential", "lognormal"):
        verdict = f"non-critical ({_SHORT[best]})"
    elif heavy and decades >= 2.0 and gap <= 0.2:
        verdict = "CRITICAL (3/2-class)"
    elif heavy:
        verdict = (f"power-law-like but NOT robust "
                   f"(decades={decades:.1f}, gap={gap:.2f})")
    else:
        verdict = f"non-critical ({_SHORT[best]})"

    out.update({
        "best": _SHORT[best], "alpha_TPL": round(alpha_tpl, 3),
        "alpha_PL": round(alpha_pl, 3), "gap": round(gap, 3),
        "alpha_TPL_CI95": ci, "xmin": int(round(xmin)), "xmax": int(xmax),
        "decades": round(decades, 2), "n_tail": int(tail.size),
        "weights": {_SHORT[k]: round(v, 3) for k, v in weights.items()},
        "R_TPLvsEXP": R_tpl_exp, "R_TPLvsSTR": R_tpl_str,
        "R_TPLvsLOGN": R_tpl_logn, "R_PLvsEXP": R_pl_exp,
        "verdict": verdict,
    })
    return out


# --- BIB loaders (validated local data) --------------------------------------
def _bib_path(bib_dir, d, pair):
    """Resolve the durations file.  The paper's RHF uses reward_huge_{d};
    the reward_huge_v2_{d} duplicate dirs are incomplete (fallback only)."""
    for sub in (f"reward_huge_{d}", f"reward_huge_v2_{d}"):
        p = os.path.join(bib_dir, sub, f"durations_{pair}_m50_huge.json")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"durations_{pair}_m50_huge.json (design {d}) not under {bib_dir}")


def _load_bib(bib_dir, pair, agent1_only):
    pool = []
    for d in DESIGNS:
        j = json.load(open(_bib_path(bib_dir, d, pair)))
        pool += [v for v in j["T_argmax1"] if v >= 1]
        if not agent1_only:
            pool += [v for v in j["T_argmax2"] if v >= 1]
    return np.asarray(pool, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib-dir", default=DEFAULT_BIB)
    ap.add_argument("--baseline", default=os.path.join(HERE, "data", "baseline_dwells.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "validation_baseline_control.json"))
    args = ap.parse_args()

    z = np.load(args.baseline)
    rows = []

    def add(method, opponent, observable, data):
        r = {"method": method, "opponent": opponent, "observable": observable}
        r.update(fit_unified(data))
        rows.append(r)
        a = r.get("alpha_TPL", "--"); v = r.get("verdict", "")
        print(f"  {method:14s} {opponent:12s} {observable:13s} "
              f"alpha_TPL={a!s:>6} decades={r.get('decades','--')!s:>4} "
              f"gap={r.get('gap','--')!s:>5}  {v}", flush=True)

    print("=== ARGMAX-PERSISTENCE (figure observable) ===")
    # BIB from validated data (guarded: bib-random uses T_argmax1 only)
    add("BIB", "uniform-random", "argmax-hyp",
        _load_bib(args.bib_dir, "bib-random", agent1_only=True))
    add("BIB", "self-play", "argmax-hyp",
        _load_bib(args.bib_dir, "bib-bib", agent1_only=False))
    # baselines
    add("RegretMatching", "uniform-random", "argmax-regret", z["RegretMatching__random__argmax"])
    add("RegretMatching", "self-play", "argmax-regret", z["RegretMatching__self_play__argmax"])
    add("Qlearning", "uniform-random", "action", z["Qlearning__random__argmax"])
    add("Qlearning", "self-play", "action", z["Qlearning__self_play__argmax"])
    add("WSLS", "uniform-random", "action", z["WSLS__random__argmax"])
    add("WSLS", "self-play", "action", z["WSLS__self_play__argmax"])

    print("\n=== BEHAVIOR (action run length; table) ===")
    for m in ("WSLS", "Qlearning", "RegretMatching"):
        for opp in ("self_play", "fixed_biased"):
            k = f"{m}__{opp}__action"
            if k in z.files:
                add(m, opp.replace("_", "-"), "action", z[k])

    meta = {"bib_dir": args.bib_dir, "baseline": os.path.basename(args.baseline),
            "method_note": "CSN xmin + Akaike over {PL,TPL,EXP,lognormal,stretched-exp}; "
                           "TPL-interior basis; verdict needs heavy-tail-preferred + >=2 decades + gap<=0.2",
            "bib_pooling": "bib-bib pools T_argmax1+T_argmax2; bib-random T_argmax1 ONLY (opponent side degenerate)",
            "rows": rows}
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.out}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
