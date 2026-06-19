#!/usr/bin/env python3
r"""Re-scan-x_min bootstrap (paper convention): Clauset x_min re-selected on each
resample, B=200, matching app/main-text "B=200 truncated-power-law refits".
Outputs bootstrap_exponents_rescan.json + prints per-cell IQR and the
cross-design-spread-vs-per-design-IQR comparison (tests manuscript L581-583).

Cell-by-cell parallel so progress is visible; ~1.6-4 s per auto-xmin fit.
"""
import json, os, warnings
import numpy as np
import powerlaw, figdata
import multiprocessing as mp
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bo_verdict")
DESIGNS = ["rs", "ra", "ss", "sa"]; PAIRS = ["bib-bib", "bo-bo"]
THETA = 0.4; LAM = "simulation_tie_mode_ablation/data/scheme_ablation/huge_laminar"
B = 200; BASE_SEED = 20533918

def argmax_pool(d, pair, nh=10):
    sub = f"reward_huge_{d}" if nh == 10 else f"reward_huge_v3_{d}_h{nh}"
    j = json.load(open(figdata.find(f"simulation/reward_huge/data/{sub}/durations_{pair}_m50_huge.json")))
    return np.asarray([v for k in ("T_argmax1", "T_argmax2") for v in j.get(k, []) if v >= 1], float)

def laminar_pool(d, pair, theta=THETA):
    tag = "eq3" if pair == "bib-bib" else "bo"
    z = np.load(figdata.find(f"{LAM}/pmax_{d}_{tag}.npz"))
    out = []
    for r in range(z["pmax1"].shape[0]):
        for key in ("pmax1", "pmax2"):
            il = z[key][r] > theta
            dd = np.diff(il.astype(np.int8)); b = np.concatenate([[0], np.flatnonzero(dd) + 1, [il.size]])
            out += np.diff(b)[il[b[:-1]]].tolist()
    return np.asarray(out, float)

_DATA = {}
def _init(data):
    global _DATA; _DATA = data

def _boot(args):
    key, seed = args
    arr = _DATA[key]
    rng = np.random.default_rng(seed)
    rs = arr[rng.integers(0, len(arr), len(arr))]
    F = powerlaw.Fit(rs, discrete=True, verbose=False)   # auto Clauset x_min each resample
    return float(F.truncated_power_law.alpha), float(F.xmin)

if __name__ == "__main__":
    data = {}; cells = []
    for obs in ("argmax", "laminar"):
        for pair in PAIRS:
            for d in DESIGNS:
                key = f"{obs}|{pair}|{d}"
                data[key] = argmax_pool(d, pair, 10) if obs == "argmax" else laminar_pool(d, pair)
                cells.append(key)
    nproc = min(14, os.cpu_count())
    print(f"re-scan bootstrap: cells={len(cells)} B={B} cores={nproc}", flush=True)
    boot = {}
    with mp.Pool(nproc, initializer=_init, initargs=(data,)) as pool:
        for ci, key in enumerate(cells):
            tasks = [(key, BASE_SEED + ci * 100003 + j) for j in range(B)]
            res = pool.map(_boot, tasks, chunksize=4)
            a = np.array([r[0] for r in res]); xm = np.array([r[1] for r in res])
            boot[key] = dict(alphas=a.tolist(), xmins=xm.tolist(),
                             median=float(np.median(a)),
                             iqr=[float(np.percentile(a, 25)), float(np.percentile(a, 75))],
                             iqr_width=float(np.percentile(a, 75) - np.percentile(a, 25)),
                             ci90=[float(np.percentile(a, 5)), float(np.percentile(a, 95))],
                             xmin_mode=float(np.bincount(xm.astype(int)).argmax()),
                             xmin_unique=int(len(np.unique(xm))))
            print(f"  [{ci+1:2}/16] {key:24} med={np.median(a):6.3f} "
                  f"IQRw={boot[key]['iqr_width']:.3f} 90%w={np.percentile(a,95)-np.percentile(a,5):.3f} "
                  f"xmin(uniq={boot[key]['xmin_unique']})", flush=True)
    json.dump(boot, open(os.path.join(OUT, "bootstrap_exponents_rescan.json"), "w"))

    # ---- manuscript L581-583 test: BIB cross-design spread vs per-design IQR ----
    print("\n=== L581-583 test (re-scan convention) ===", flush=True)
    for obs in ("argmax", "laminar"):
        for pair in PAIRS:
            meds = [boot[f"{obs}|{pair}|{d}"]["median"] for d in DESIGNS]
            iqrw = [boot[f"{obs}|{pair}|{d}"]["iqr_width"] for d in DESIGNS]
            spread = max(meds) - min(meds)
            print(f"  {obs:8} {pair:9} cross-design spread={spread:.3f}  "
                  f"median per-design IQRw={np.median(iqrw):.3f}  "
                  f"ratio={spread/np.median(iqrw):.1f}x", flush=True)
    print("DONE ->", os.path.join(OUT, "bootstrap_exponents_rescan.json"), flush=True)
