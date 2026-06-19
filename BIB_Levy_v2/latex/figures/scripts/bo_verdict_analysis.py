#!/usr/bin/env python3
r"""BO model-selection verdict + bootstrap exponent distributions.

Applies the appendix's own rigorous bar (Clauset x_min + Akaike over
{PL, TPL, exp, lognormal, stretched-exp}, heavy-tail-preferred over >=2 decades
with pure-/truncated gap <= 0.2; app:methods L2411-2414) to the headline BO-BO
argmax/laminar exponents, which the manuscript reports only as TPL fits.

Outputs (written to ./bo_verdict/):
  bo_verdict.csv / .json     -- Part 3 verdict table (Nh=10 all cells + BO argmax Nh-sweep)
  bootstrap_exponents.json   -- Part 5 per-cell B=1000 TPL-alpha arrays + summaries
  bo_dynamics_stats.csv      -- Part 6 laminar occupancy / max-P(h) stats
  per_run_passrate.csv       -- Part 4 per-run laminar pass-rate (argmax deferred: pooled-only data)

Fitting convention matches the paper: powerlaw (Alstott 2014), Clauset x_min
auto-scan, discrete=True, report truncated_power_law.alpha.  Bootstrap holds
x_min fixed per cell (selected once on the pooled set) so the B-array isolates
exponent variability from x_min-selection noise (Section 4 caveat); re-scanning
x_min each resample is computationally infeasible at B=1000 x 16 cells and would
inflate the spread by x_min-selection noise.
"""
import json, os, csv, warnings
import numpy as np
import powerlaw
import figdata
warnings.filterwarnings("ignore")
np.random.seed(20533918)  # deterministic (Zenodo DOI as seed)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bo_verdict")
os.makedirs(OUT, exist_ok=True)

DESIGNS = ["rs", "ra", "ss", "sa"]
PAIRS = ["bib-bib", "bo-bo"]
THETA = 0.4
LAM = "simulation_tie_mode_ablation/data/scheme_ablation/huge_laminar"
NPARAM = {"power_law": 1, "truncated_power_law": 2, "exponential": 1,
          "lognormal": 2, "stretched_exponential": 2}

# ----------------------------------------------------------------- loaders
def argmax_pool(d, pair, nh=10):
    sub = f"reward_huge_{d}" if nh == 10 else f"reward_huge_v3_{d}_h{nh}"
    j = json.load(open(figdata.find(
        f"simulation/reward_huge/data/{sub}/durations_{pair}_m50_huge.json")))
    o = []
    for k in ("T_argmax1", "T_argmax2"):
        o += [v for v in j.get(k, []) if v >= 1]
    return np.asarray(o, float)

def laminar_perrun(d, pair, theta=THETA):
    """Return list of per-run laminar-length arrays (both agents pooled per run)."""
    tag = "eq3" if pair == "bib-bib" else "bo"
    z = np.load(figdata.find(f"{LAM}/pmax_{d}_{tag}.npz"))
    out = []
    nruns = z["pmax1"].shape[0]
    for r in range(nruns):
        o = []
        for key in ("pmax1", "pmax2"):
            il = z[key][r] > theta
            dd = np.diff(il.astype(np.int8))
            b = np.concatenate([[0], np.flatnonzero(dd) + 1, [il.size]])
            rl = np.diff(b); st = il[b[:-1]]
            o += rl[st].tolist()
        out.append(np.asarray(o, float))
    return out

def laminar_pool(d, pair, theta=THETA):
    return np.concatenate(laminar_perrun(d, pair, theta))

def max_p_series(d, pair):
    tag = "eq3" if pair == "bib-bib" else "bo"
    z = np.load(figdata.find(f"{LAM}/pmax_{d}_{tag}.npz"))
    return np.concatenate([z["pmax1"].ravel(), z["pmax2"].ravel()])

# ----------------------------------------------------------------- verdict
def verdict(arr):
    F = powerlaw.Fit(arr, discrete=True, verbose=False)
    data = F.data
    aic = {}
    for name in NPARAM:
        ll = float(np.sum(getattr(F, name).loglikelihoods(data)))
        aic[name] = 2 * NPARAM[name] - 2 * ll
    amin = min(aic.values())
    w = {k: np.exp(-(aic[k] - amin) / 2) for k in aic}
    s = sum(w.values()); w = {k: v / s for k, v in w.items()}
    Rte, pte = F.distribution_compare("truncated_power_law", "exponential", normalized_ratio=True)
    Rpe, ppe = F.distribution_compare("power_law", "exponential", normalized_ratio=True)
    decades = float(np.log10(data.max() / F.xmin))
    gap = abs(F.power_law.alpha - F.truncated_power_law.alpha)
    winner = max(w, key=w.get)
    ht_pref = (w["power_law"] + w["truncated_power_law"]) > w["exponential"]
    PASS = bool(ht_pref and decades >= 2 and gap <= 0.2)
    return dict(n=int(len(arr)), xmin=float(F.xmin), winner=winner,
                w_PL=w["power_law"], w_TPL=w["truncated_power_law"], w_exp=w["exponential"],
                w_logn=w["lognormal"], w_stexp=w["stretched_exponential"],
                R_TPL_exp=float(Rte), p_TPL_exp=float(pte),
                R_PL_exp=float(Rpe), p_PL_exp=float(ppe),
                decades=decades, alpha_TPL=float(F.truncated_power_law.alpha),
                alpha_PL=float(F.power_law.alpha), gap=float(gap), PASS=PASS)

def bootstrap_tpl(arr, B=1000):
    F = powerlaw.Fit(arr, discrete=True, verbose=False)
    xmin = float(F.xmin)
    point = float(F.truncated_power_law.alpha)
    alphas = np.empty(B)
    n = len(arr)
    for b in range(B):
        rs = arr[np.random.randint(0, n, n)]
        Fb = powerlaw.Fit(rs, discrete=True, xmin=xmin, verbose=False)
        alphas[b] = Fb.truncated_power_law.alpha
    return alphas, xmin, point

# ----------------------------------------------------------------- run
rows = []
print("=== PART 3: model-selection verdict (Nh=10) ===")
hdr = f"{'pair':9} {'design':6} {'obs':8} {'Nh':>3} {'winner':>21} {'wTPL':>5} {'wexp':>5} {'R(TPL,e)':>8} {'dec':>5} {'aTPL':>6} {'gap':>6} PASS"
print(hdr)
for obs in ("argmax", "laminar"):
    for pair in PAIRS:
        for d in DESIGNS:
            arr = argmax_pool(d, pair, 10) if obs == "argmax" else laminar_pool(d, pair)
            v = verdict(arr); v.update(pair=pair, design=d, observable=obs, Nh=10)
            rows.append(v)
            print(f"{pair:9} {d:6} {obs:8} {10:>3} {v['winner']:>21} {v['w_TPL']:5.2f} "
                  f"{v['w_exp']:5.2f} {v['R_TPL_exp']:8.2f} {v['decades']:5.2f} "
                  f"{v['alpha_TPL']:6.3f} {v['gap']:6.3f} {v['PASS']}")

print("\n=== PART 3b: BO-BO argmax Nh-sweep ===")
for pair in ("bo-bo", "bib-bib"):
    for nh in (3, 6, 15, 20):  # 10 already above
        for d in DESIGNS:
            arr = argmax_pool(d, pair, nh)
            v = verdict(arr); v.update(pair=pair, design=d, observable="argmax", Nh=nh)
            rows.append(v)
        # compact per-Nh summary
        sub = [r for r in rows if r["pair"] == pair and r["Nh"] == nh and r["observable"] == "argmax"]
        npass = sum(r["PASS"] for r in sub)
        print(f"  {pair} Nh={nh:2}: PASS {npass}/4  "
              f"winners={[r['winner'][:4] for r in sub]}  dec={[round(r['decades'],2) for r in sub]}")

# write verdict table
cols = ["pair", "design", "observable", "Nh", "n", "xmin", "winner", "w_PL", "w_TPL",
        "w_exp", "w_logn", "w_stexp", "R_TPL_exp", "p_TPL_exp", "R_PL_exp", "p_PL_exp",
        "decades", "alpha_TPL", "alpha_PL", "gap", "PASS"]
with open(os.path.join(OUT, "bo_verdict.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
    for r in rows:
        wr.writerow({c: r[c] for c in cols})
json.dump(rows, open(os.path.join(OUT, "bo_verdict.json"), "w"), indent=2)

# ----------------------------------------------------------------- Part 5 bootstrap
print("\n=== PART 5: bootstrap TPL-alpha (B=1000, fixed x_min) ===")
boot = {}
for obs in ("argmax", "laminar"):
    for pair in PAIRS:
        for d in DESIGNS:
            arr = argmax_pool(d, pair, 10) if obs == "argmax" else laminar_pool(d, pair)
            a, xmin, point = bootstrap_tpl(arr, B=1000)
            key = f"{obs}|{pair}|{d}"
            boot[key] = dict(observable=obs, pair=pair, design=d, xmin=xmin,
                             point=point, B=1000, alphas=a.tolist(),
                             median=float(np.median(a)),
                             iqr=[float(np.percentile(a, 25)), float(np.percentile(a, 75))],
                             ci90=[float(np.percentile(a, 5)), float(np.percentile(a, 95))],
                             # B=200 summary to match paper's CI
                             iqr_B200=[float(np.percentile(a[:200], 25)), float(np.percentile(a[:200], 75))])
            print(f"  {key:24} point={point:6.3f} median={np.median(a):6.3f} "
                  f"IQR=[{np.percentile(a,25):.3f},{np.percentile(a,75):.3f}] "
                  f"90%=[{np.percentile(a,5):.3f},{np.percentile(a,95):.3f}]")
json.dump(boot, open(os.path.join(OUT, "bootstrap_exponents.json"), "w"))

# ----------------------------------------------------------------- Part 6 dynamics
print("\n=== PART 6: laminar occupancy / max-P(h) ===")
dyn = []
for pair in PAIRS:
    for d in DESIGNS:
        mp = max_p_series(d, pair)
        per = laminar_perrun(d, pair)
        lengths = np.concatenate(per)
        occ = float(np.mean(mp > THETA))
        row = dict(pair=pair, design=d, occupancy=occ, n_laminar_events=int(len(lengths)),
                   mean_laminar_len=float(lengths.mean()) if len(lengths) else 0.0,
                   median_laminar_len=float(np.median(lengths)) if len(lengths) else 0.0,
                   mean_maxP=float(mp.mean()), median_maxP=float(np.median(mp)))
        dyn.append(row)
        print(f"  {pair:9} {d:6} occ(maxP>0.4)={occ:6.3f} n_lam={len(lengths):7d} "
              f"meanLen={row['mean_laminar_len']:7.1f} meanMaxP={row['mean_maxP']:.3f}")
with open(os.path.join(OUT, "bo_dynamics_stats.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(dyn[0].keys())); wr.writeheader(); wr.writerows(dyn)

# ----------------------------------------------------------------- Part 4 per-run passrate (laminar)
print("\n=== PART 4: per-run laminar heavy-tail-over-exp pass-rate (caveat: see Section 4) ===")
pr = []
for pair in PAIRS:
    for d in DESIGNS:
        per = laminar_perrun(d, pair)
        npass = 0; nval = 0
        for arr in per:
            if len(arr) < 50:  # too few events for any fit
                continue
            nval += 1
            try:
                F = powerlaw.Fit(arr, discrete=True, verbose=False)
                R, p = F.distribution_compare("truncated_power_law", "exponential", normalized_ratio=True)
                if R > 0 and p < 0.05:
                    npass += 1
            except Exception:
                pass
        rate = npass / nval if nval else float("nan")
        pr.append(dict(pair=pair, design=d, observable="laminar",
                       runs_valid=nval, runs_pass=npass, pass_rate=rate))
        print(f"  {pair:9} {d:6} laminar per-run pass {npass}/{nval} = {rate:.2f}")
with open(os.path.join(OUT, "per_run_passrate.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(pr[0].keys())); wr.writeheader(); wr.writerows(pr)
print("\nNOTE: per-run ARGMAX pass-rate deferred -- argmax durations are cached pooled "
      "(no per-run labels); laminar uses the per-run pmax npz.")
print(f"\nAll outputs written to {OUT}")
