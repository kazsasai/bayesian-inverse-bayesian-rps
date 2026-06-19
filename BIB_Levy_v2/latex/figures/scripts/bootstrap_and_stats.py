#!/usr/bin/env python3
r"""Part 5 (bootstrap TPL-alpha), Part 6 (dynamics), Part 4 (per-run pass-rate).

Split out from bo_verdict_analysis.py because the discrete-TPL fit costs ~0.8 s
each, so the B-bootstrap must be parallelised (14 cores here).  x_min per cell is
read from the already-computed bo_verdict.json (Part 3), so no x_min re-scan.

Discrete fitting throughout (paper convention).  x_min held fixed per cell across
resamples (isolates exponent variability from x_min-selection noise; Section 4).
"""
import json, os, csv, warnings
import numpy as np
import powerlaw
import figdata
import multiprocessing as mp
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bo_verdict")
DESIGNS = ["rs", "ra", "ss", "sa"]
PAIRS = ["bib-bib", "bo-bo"]
THETA = 0.4
LAM = "simulation_tie_mode_ablation/data/scheme_ablation/huge_laminar"
B = 500
BASE_SEED = 20533918

def argmax_pool(d, pair, nh=10):
    sub = f"reward_huge_{d}" if nh == 10 else f"reward_huge_v3_{d}_h{nh}"
    j = json.load(open(figdata.find(
        f"simulation/reward_huge/data/{sub}/durations_{pair}_m50_huge.json")))
    return np.asarray([v for k in ("T_argmax1", "T_argmax2") for v in j.get(k, []) if v >= 1], float)

def laminar_perrun(d, pair, theta=THETA):
    tag = "eq3" if pair == "bib-bib" else "bo"
    z = np.load(figdata.find(f"{LAM}/pmax_{d}_{tag}.npz"))
    out = []
    for r in range(z["pmax1"].shape[0]):
        o = []
        for key in ("pmax1", "pmax2"):
            il = z[key][r] > theta
            dd = np.diff(il.astype(np.int8))
            b = np.concatenate([[0], np.flatnonzero(dd) + 1, [il.size]])
            o += np.diff(b)[il[b[:-1]]].tolist()
        out.append(np.asarray(o, float))
    return out

def laminar_pool(d, pair, theta=THETA):
    return np.concatenate(laminar_perrun(d, pair, theta))

def max_p_series(d, pair):
    tag = "eq3" if pair == "bib-bib" else "bo"
    z = np.load(figdata.find(f"{LAM}/pmax_{d}_{tag}.npz"))
    return np.concatenate([z["pmax1"].ravel(), z["pmax2"].ravel()])

# ---- worker globals ----
_DATA = {}   # key -> (arr, xmin)

def _init(data):
    global _DATA
    _DATA = data

def _boot_one(args):
    key, seed = args
    arr, xmin = _DATA[key]
    rng = np.random.default_rng(seed)
    rs = arr[rng.integers(0, len(arr), len(arr))]
    F = powerlaw.Fit(rs, discrete=True, xmin=xmin, verbose=False)
    return key, float(F.truncated_power_law.alpha)

def _perrun_one(args):
    # returns (cellkey, n_valid_increment, n_pass_increment)
    key, arr = args
    if len(arr) < 50:
        return key, 0, 0
    try:
        F = powerlaw.Fit(arr, discrete=True, verbose=False)
        R, p = F.distribution_compare("truncated_power_law", "exponential", normalized_ratio=True)
        return key, 1, int(R > 0 and p < 0.05)
    except Exception:
        return key, 1, 0

if __name__ == "__main__":
    verd = json.load(open(os.path.join(OUT, "bo_verdict.json")))
    V = {(r["pair"], r["design"], r["observable"], r["Nh"]): r for r in verd}

    # ---- build cell pools + xmins (from verdict) ----
    cells = []          # (obs, pair, design)
    data = {}           # key -> (arr, xmin)
    point = {}          # key -> discrete TPL point estimate (from verdict)
    for obs in ("argmax", "laminar"):
        for pair in PAIRS:
            for d in DESIGNS:
                arr = argmax_pool(d, pair, 10) if obs == "argmax" else laminar_pool(d, pair)
                rec = V[(pair, d, obs, 10)]
                key = f"{obs}|{pair}|{d}"
                data[key] = (arr, float(rec["xmin"]))
                point[key] = float(rec["alpha_TPL"])
                cells.append(key)
    print(f"cells={len(cells)}  B={B}  cores={min(14, os.cpu_count())}")

    # ---- Part 5: parallel bootstrap ----
    tasks = []
    for ci, key in enumerate(cells):
        for j in range(B):
            tasks.append((key, BASE_SEED + ci * 100003 + j))
    nproc = min(14, os.cpu_count())
    with mp.Pool(nproc, initializer=_init, initargs=(data,)) as pool:
        res = pool.map(_boot_one, tasks, chunksize=20)
    arrs = {key: [] for key in cells}
    for key, a in res:
        arrs[key].append(a)
    boot = {}
    for key in cells:
        a = np.asarray(arrs[key])
        obs, pair, d = key.split("|")
        boot[key] = dict(observable=obs, pair=pair, design=d, xmin=data[key][1],
                         point=point[key], B=int(len(a)), alphas=a.tolist(),
                         median=float(np.median(a)),
                         iqr=[float(np.percentile(a, 25)), float(np.percentile(a, 75))],
                         ci90=[float(np.percentile(a, 5)), float(np.percentile(a, 95))])
    json.dump(boot, open(os.path.join(OUT, "bootstrap_exponents.json"), "w"))
    print("PART5 done. summary:")
    for key in cells:
        b = boot[key]
        print(f"  {key:24} pt={b['point']:6.3f} med={b['median']:6.3f} "
              f"IQR=[{b['iqr'][0]:.3f},{b['iqr'][1]:.3f}] 90%=[{b['ci90'][0]:.3f},{b['ci90'][1]:.3f}]")

    # ---- Part 6: dynamics ----
    dyn = []
    for pair in PAIRS:
        for d in DESIGNS:
            mp_s = max_p_series(d, pair)
            lengths = laminar_pool(d, pair)
            dyn.append(dict(pair=pair, design=d, occupancy=float(np.mean(mp_s > THETA)),
                            n_laminar_events=int(len(lengths)),
                            mean_laminar_len=float(lengths.mean()) if len(lengths) else 0.0,
                            median_laminar_len=float(np.median(lengths)) if len(lengths) else 0.0,
                            mean_maxP=float(mp_s.mean()), median_maxP=float(np.median(mp_s))))
    with open(os.path.join(OUT, "bo_dynamics_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dyn[0].keys())); w.writeheader(); w.writerows(dyn)
    print("PART6 done.")
    for r in dyn:
        print(f"  {r['pair']:9} {r['design']:6} occ={r['occupancy']:6.3f} "
              f"n_lam={r['n_laminar_events']:7d} meanLen={r['mean_laminar_len']:7.1f} meanMaxP={r['mean_maxP']:.3f}")

    # ---- Part 4: per-run laminar pass-rate (parallel) ----
    prtasks = []
    for pair in PAIRS:
        for d in DESIGNS:
            for arr in laminar_perrun(d, pair):
                prtasks.append((f"{pair}|{d}", arr))
    with mp.Pool(nproc) as pool:
        prres = pool.map(_perrun_one, prtasks, chunksize=4)
    agg = {}
    for key, nv, npass in prres:
        a = agg.setdefault(key, [0, 0]); a[0] += nv; a[1] += npass
    pr = []
    for pair in PAIRS:
        for d in DESIGNS:
            nv, npass = agg.get(f"{pair}|{d}", [0, 0])
            pr.append(dict(pair=pair, design=d, observable="laminar",
                           runs_valid=nv, runs_pass=npass,
                           pass_rate=(npass / nv if nv else float("nan"))))
    with open(os.path.join(OUT, "per_run_passrate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pr[0].keys())); w.writeheader(); w.writerows(pr)
    print("PART4 done (per-run argmax deferred: pooled-only data).")
    for r in pr:
        print(f"  {r['pair']:9} {r['design']:6} laminar per-run {r['runs_pass']}/{r['runs_valid']} = {r['pass_rate']:.2f}")
    print("ALL DONE ->", OUT)
