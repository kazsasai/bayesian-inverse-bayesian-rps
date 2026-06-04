#!/usr/bin/env python3
"""
Lock the laminar finite-size numbers at full production scale (COWORK_FOLLOWUP_BRIEF).

Reads the FULL-scale laminar cache produced by run_laminar_nhsweep_prod.py
(CACHE_NAME=laminar_FULL_cache.json, all 4 designs x Nh{3,6,10,15,20} x 20 runs,
T=2e5, burn=1e5), and reports, by the IDENTICAL pipeline used for the argmax z:

  TASK A  z_lam : rs design, laminar 1/Lambda(Nh) (truncated power law, CSN x_min),
                  z_lam from Tmax_lam(Nh) ~ Nh^{-z}; + argmax-z consistency anchor
                  computed the SAME way on the existing full-scale argmax durations.
  TASK B  alpha_lam : per-design tail exponent alpha_lam(Nh); core-regime (Nh in {6,10})
                  pooled alpha_lam +/- SD (n = 4 designs x 2 Nh = 8), invariance vs 1.34.

Prints the decision-rule branch for each task and auto-fills the SI draft sentences.
Writes laminar_lock_results.json.

Run (on a machine that can do the full sweep):
  # 1) generate (resumable; set BUDGET huge so it runs to completion in one go):
  CACHE_NAME=laminar_FULL_cache.json NHS=3,6,10,15,20 NR=20 T=200000 BURN=100000 \
      BUDGET=100000000 WORKERS=$(nproc) python run_laminar_nhsweep_prod.py
  #    (repeat until it prints "complete combos: 20/20"; it checkpoints after each batch)
  # 2) analyse + lock:
  python analyze_laminar_lock.py
"""
import os, sys, json, glob
from pathlib import Path
import numpy as np
import powerlaw, warnings
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "simulation").is_dir())
CACHE = HERE / os.environ.get("CACHE_NAME", "laminar_FULL_cache.json")
NH = [3, 6, 10, 15, 20]
CORE_Z = [3, 6, 10, 15, 20]      # the argmax z=1.45 rs fit used all five (SI Sec 9)
CORE_ALPHA = [6, 10]             # argmax alpha invariance core regime (Table S12, n=8)
DES = ["rs", "ra", "ss", "sa"]
ARG_RS_PUBLISHED = {3: 6879, 6: 2751, 10: 1373, 15: 591, 20: 435}   # SI Sec 9, argmax rs 1/Lambda
ARGSUB = int(os.environ.get("ARGMAX_SUBSAMPLE", "0"))               # 0 = use all (recommended)


def tpl(lengths):
    a = np.array([x for x in lengths if x >= 1], float)
    f = powerlaw.Fit(a, discrete=True, verbose=False)
    lam = float(f.truncated_power_law.Lambda)
    return dict(alpha=float(f.truncated_power_law.alpha), Lambda=lam,
                cutoff=(1.0 / lam if lam > 0 else np.nan),
                xmin=float(f.xmin), n=int(len(a)), mean=float(a.mean()))


def zfit(nhs, cutoffs):
    x = np.log(np.array(nhs, float)); y = np.log(np.array(cutoffs, float)); n = len(x)
    sx, sy = x.mean(), y.mean(); sxx = ((x - sx) ** 2).sum()
    slope = ((x - sx) * (y - sy)).sum() / sxx
    yhat = sy + slope * (x - sx)
    ssr = ((y - yhat) ** 2).sum(); sst = ((y - sy) ** 2).sum()
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    se = np.sqrt(ssr / (n - 2) / sxx) if n > 2 else float("nan")
    return -slope, se, r2


def argmax_durations_rs(nh):
    sub = "reward_huge_v2_rs" if nh == 10 else f"reward_huge_v3_rs_h{nh}"
    p = ROOT / "simulation/reward_huge/data" / sub / "durations_bib-bib_m50_huge.json"
    if not p.exists():
        return None
    j = json.load(open(p))
    out = list(j.get("T_argmax1", [])) + list(j.get("T_argmax2", []))
    if ARGSUB and len(out) > ARGSUB:
        out = list(np.random.default_rng(0).choice(out, ARGSUB, replace=False))
    return out


def main():
    if not CACHE.exists():
        sys.exit(f"missing {CACHE.name}; run run_laminar_nhsweep_prod.py first")
    data = json.load(open(CACHE))
    pd_nh = {}
    for k, v in data.items():
        tag, nh = k.split("|"); pd_nh.setdefault(tag, {})[int(nh)] = v["lengths"]
    nr = {k: v["runs_done"] for k, v in data.items()}
    print("runs per combo:", nr, "\n")
    res = {}

    # ---------- TASK A: z_lam (rs) + argmax anchor (same pipeline) ----------
    print("== TASK A: laminar cutoff scaling z_lam (rs design, 1/Lambda, CSN x_min) ==")
    rs = pd_nh.get("rs", {})
    lam_rows = {nh: tpl(rs[nh]) for nh in NH if nh in rs}
    for nh in NH:
        if nh in lam_rows:
            r = lam_rows[nh]
            print(f"  Nh={nh:2d}: alpha_lam={r['alpha']:.3f}  1/Lambda={r['cutoff']:.0f}"
                  f"  xmin={r['xmin']:.0f}  mean={r['mean']:.1f}  n_phases={r['n']}")
    have = [nh for nh in CORE_Z if nh in lam_rows]
    zl, sel, r2l = zfit(have, [lam_rows[nh]["cutoff"] for nh in have])
    zl4, sel4, r2l4 = (zfit(have[:-1], [lam_rows[nh]["cutoff"] for nh in have[:-1]])
                       if len(have) >= 4 else (np.nan, np.nan, np.nan))
    print(f"  -> z_lam (all {have}) = {zl:.3f} +/- {sel:.3f}  (R^2={r2l:.3f})")
    print(f"  -> z_lam (excl-boundary {have[:-1]}) = {zl4:.3f} +/- {sel4:.3f}  (R^2={r2l4:.3f})")

    # argmax anchor via the identical 1/Lambda CSN-xmin pipeline on full-scale durations
    print("  -- argmax anchor (identical pipeline, full-scale durations) --")
    arg_cut = {}
    for nh in NH:
        d = argmax_durations_rs(nh)
        if d:
            arg_cut[nh] = tpl(d)["cutoff"]
            print(f"     Nh={nh:2d}: argmax 1/Lambda={arg_cut[nh]:.0f}  (published {ARG_RS_PUBLISHED[nh]})")
    if len(arg_cut) >= 4:
        za, sea, r2a = zfit(sorted(arg_cut), [arg_cut[nh] for nh in sorted(arg_cut)])
        print(f"     -> z_argmax (re-fit) = {za:.3f} +/- {sea:.3f}  (R^2={r2a:.3f})")
    zpub, sepub, r2pub = zfit(NH, [ARG_RS_PUBLISHED[nh] for nh in NH])
    print(f"     -> z_argmax (published 1/Lambda series) = {zpub:.3f} +/- {sepub:.3f}  (R^2={r2pub:.3f})  [paper: 1.45+/-0.06]")
    res["taskA"] = dict(z_lam_all=zl, se_all=sel, r2_all=r2l, nh_all=have,
                        z_lam_excl=zl4, se_excl=sel4, r2_excl=r2l4,
                        z_argmax_published=zpub, laminar_cutoffs={nh: lam_rows[nh]["cutoff"] for nh in have})

    # decision rule A
    gap = zl - zpub; comb = float(np.hypot(sel, max(sepub, 0.06)))
    if not np.isfinite(r2l) or r2l < 0.9:
        branchA = "NO-SINGLE-POWER (laminar cutoff has no clean Nh^-z; R^2<0.9) -> report, narrow shared-class to tail only. STOP."
    elif abs(gap) <= 2 * comb:
        branchA = f"CONSISTENT with argmax z ({gap:+.2f}, <=2*combinedSE={2*comb:.2f}) -> SEAM MAY CLOSE. STOP & report before rewriting."
    else:
        branchA = f"z_lam clearly ABOVE argmax z ({gap:+.2f} > 2*combinedSE={2*comb:.2f}) -> keep conservative framing; lock value."
    print("  DECISION A:", branchA)

    # ---------- TASK B: alpha_lam(Nh) per design + core invariance ----------
    print("\n== TASK B: laminar tail exponent alpha_lam(Nh) per design ==")
    alpha = {}
    for tag in DES:
        d = pd_nh.get(tag, {})
        row = []
        for nh in NH:
            if nh in d:
                a = tpl(d[nh])["alpha"]; alpha[(tag, nh)] = a; row.append(f"{nh}:{a:.3f}")
        print(f"  {tag}: " + "  ".join(row))
    core_vals = [alpha[(t, nh)] for t in DES for nh in CORE_ALPHA if (t, nh) in alpha]
    if core_vals:
        m, sd = float(np.mean(core_vals)), float(np.std(core_vals, ddof=1))
        inv = abs(m - 1.34) <= 0.05 and sd <= 0.05
        print(f"  -> core-regime (Nh in {CORE_ALPHA}) pooled alpha_lam = {m:.3f} +/- {sd:.3f}  (n={len(core_vals)})")
        print(f"     Nh-invariant within +/-0.05 of 1.34? {'YES' if inv else 'NO'}")
        res["taskB"] = dict(core_alpha_mean=m, core_alpha_sd=sd, n=len(core_vals), invariant=inv)
        branchB = ("alpha_lam core-invariant -> add SI sentence (dual-observable shared Nh-invariant tail)."
                   if inv else "alpha_lam DRIFTS with Nh -> STOP; soften dual-observable claim, lead on argmax invariance.")
    else:
        branchB = "insufficient core data"; res["taskB"] = {}
    print("  DECISION B:", branchB)

    # ---------- auto-filled draft text ----------
    print("\n== DRAFT SI Sec 9 replacement (fill if branches are the expected ones) ==")
    print(f'  [z_lam line] "..., z_lam = {zl:.2f} +/- {sel:.2f} (R^2 = {r2l:.2f}), well above the '
          f'argmax z = 1.45 +/- 0.06."')
    if res.get("taskB", {}).get("invariant"):
        m = res["taskB"]["core_alpha_mean"]; sd = res["taskB"]["core_alpha_sd"]
        print(f'  [alpha line] "The laminar tail exponent is likewise Nh-invariant in the core regime '
              f'(pooled alpha_lam = {m:.2f} +/- {sd:.2f}, n={res["taskB"]["n"]}), so the argmax and laminar '
              f'observables share an Nh-invariant 3/2-class tail and differ only in their finite-size cutoff '
              f'exponent (z = 1.45 for argmax, z_lam = {zl:.2f} for laminar)."')
    json.dump(res, open(HERE / "laminar_lock_results.json", "w"), indent=2)
    print("\nwrote laminar_lock_results.json")


if __name__ == "__main__":
    main()
