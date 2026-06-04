#!/usr/bin/env python3
"""
TASK 1 (close seam B): laminar-phase length across the Nh sweep, to test whether
the laminar observable shares the argmax finite-size exponent z (=1.45 +/- 0.06).

Canonical settings mirror run_eq3_laminar_chunked.py (the generator behind the
Nh=10 laminar data in main Fig 2 C/D): AgentReward('bib', H_LEN=50, H_NUM=Nh,
init/predict per design, tie='skip', defeat='random_other'); laminar phase =
consecutive steps with max_h P(h) > theta=0.4; pooled over BOTH agents (the same
pooling used for the argmax analysis). BIB-BIB, four designs, Nh in {3,6,10,15,20}.

Checkpointed + wall-clock budgeted so it can be re-invoked until complete.
  cache: laminar_nhsweep_cache.json  {f"{tag}|{nh}": {runs_done, lengths:[...]}}
Env: NR (target runs/combo, default 8), T (default 40000), BURN (default 8000),
     BUDGET (s/call, default 40), WORKERS (default 4).
Run repeatedly:  python run_laminar_nhsweep_prod.py
Then fit:        python run_laminar_nhsweep_prod.py --fit
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np

def _find_engine():
    env = os.environ.get("ENGINE_DIR")
    cands = []
    if env: cands.append(Path(env))
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        for sub in ("simulation_tie_mode_ablation/files",
                    "BIB_Levy_github/simulation_tie_mode_ablation/files"):
            cands.append(base / sub)
    for c in cands:
        if (c / "rpsgame_reward_tie.py").exists():
            return str(c)
    raise SystemExit("rpsgame_reward_tie.py not found; set ENGINE_DIR")

ENG = _find_engine()
sys.path.insert(0, ENG)
from rpsgame_reward_tie import AgentReward  # noqa: E402

THETA = 0.4
H_LEN = 50
_ALLNH = [3, 6, 10, 15, 20]
NH = [int(x) for x in os.environ.get("NHS", "3,6,10,15,20").split(",")]
_ALLDES = [("rs", "random", "sample"), ("ra", "random", "argmax"),
           ("ss", "structured", "sample"), ("sa", "structured", "argmax")]
_dsel = os.environ.get("DESIGNS")
DESIGNS = [d for d in _ALLDES if (not _dsel or d[0] in _dsel.split(","))]
CACHE = Path(__file__).resolve().parent / os.environ.get("CACHE_NAME", "laminar_nhsweep_cache.json")


def runlengths(mask):
    """run-lengths of consecutive True in a boolean array."""
    out = []
    cur = 0
    for v in mask:
        if v:
            cur += 1
        else:
            if cur:
                out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def run_one(args):
    nh, init, pred, seed, T, burn = args
    a1 = AgentReward("bib", H_LEN, nh, init, pred,
                     tie_mode="skip", defeat_mode="random_other", seed=seed)
    a2 = AgentReward("bib", H_LEN, nh, init, pred,
                     tie_mode="skip", defeat_mode="random_other", seed=seed + 100000)
    m1 = np.empty(T, dtype=bool); m2 = np.empty(T, dtype=bool)
    for t in range(T):
        h1 = a1.choice(); h2 = a2.choice()
        m1[t] = a1.bayes.h_prov.max() > THETA
        m2[t] = a2.bayes.h_prov.max() > THETA
        a1.update_from_outcome(h1, h2); a2.update_from_outcome(h2, h1)
    return runlengths(m1[burn:]) + runlengths(m2[burn:])


def fit():
    import powerlaw, warnings
    warnings.filterwarnings("ignore")
    data = json.load(open(CACHE))
    # pool designs per Nh
    per_nh = {nh: [] for nh in NH}
    per_design_nh = {}
    for k, v in data.items():
        tag, nh = k.split("|"); nh = int(nh)
        per_nh[nh] += v["lengths"]
        per_design_nh.setdefault(tag, {})[nh] = v["lengths"]

    def tpl(lengths):
        a = np.array([x for x in lengths if x >= 1])
        f = powerlaw.Fit(a, discrete=True, verbose=False)
        lam = float(f.truncated_power_law.Lambda)
        return dict(alpha=float(f.truncated_power_law.alpha), Lambda=lam,
                    cutoff=(1.0 / lam if lam > 0 else np.nan),
                    xmin=float(f.xmin), n=int(len(a)), mean=float(a.mean()))

    print("== pooled-over-designs laminar TPL per Nh ==")
    rows = []
    for nh in NH:
        r = tpl(per_nh[nh]); rows.append((nh, r))
        print(f"  Nh={nh:2d}: alpha_lam={r['alpha']:.3f}  cutoff(1/Lambda)={r['cutoff']:.1f}"
              f"  xmin={r['xmin']:.0f}  mean={r['mean']:.1f}  n={r['n']}")

    def zfit(nhs, cutoffs):
        x = np.log(np.array(nhs, float)); y = np.log(np.array(cutoffs, float))
        n = len(x); sx, sy = x.mean(), y.mean()
        sxx = ((x - sx) ** 2).sum(); sxy = ((x - sx) * (y - sy)).sum()
        slope = sxy / sxx; inter = sy - slope * sx
        yhat = inter + slope * x
        ss_res = ((y - yhat) ** 2).sum(); ss_tot = ((y - sy) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        se_slope = np.sqrt(ss_res / (n - 2) / sxx) if n > 2 else np.nan
        return -slope, se_slope, r2  # z = -slope (cutoff ~ Nh^-z)

    for label, nhs in [("core {3,6,10,15,20}", NH), ("excl-boundary {3,6,10,15}", [3,6,10,15])]:
        cz = [dict(rows)[nh]["cutoff"] for nh in nhs]
        z, se, r2 = zfit(nhs, cz)
        print(f"== z_lam [{label}]: z={z:.3f} +/- {se:.3f}  (R^2={r2:.3f})  "
              f"vs argmax z=1.45+/-0.06 ==")

    print("== per-design z_lam (core, {3,6,10,15,20}) ==")
    for tag, _, _ in DESIGNS:
        d = per_design_nh.get(tag, {})
        if all(nh in d for nh in NH):
            cz = [tpl(d[nh])["cutoff"] for nh in NH]
            z, se, r2 = zfit(NH, cz)
            print(f"  {tag}: z_lam={z:.3f} +/- {se:.3f}  (R^2={r2:.3f})")
    json.dump({"per_nh_fit": {str(nh): r for nh, r in rows}}, open(
        CACHE.with_name("laminar_nhsweep_fit.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()
    if args.fit:
        fit(); return
    from multiprocessing import Pool
    NR = int(os.environ.get("NR", "8"))
    T = int(os.environ.get("T", "40000"))
    BURN = int(os.environ.get("BURN", "8000"))
    BUDGET = float(os.environ.get("BUDGET", "40"))
    WORKERS = int(os.environ.get("WORKERS", "4"))
    data = json.load(open(CACHE)) if CACHE.exists() else {}
    t0 = time.time()
    with Pool(WORKERS) as pool:
        for nh in NH:
            for tag, init, pred in DESIGNS:
                if time.time() - t0 > BUDGET:
                    break
                key = f"{tag}|{nh}"
                rec = data.get(key, {"runs_done": 0, "lengths": []})
                need = NR - rec["runs_done"]
                if need <= 0:
                    continue
                # run `need` more runs in worker-sized batches, respecting budget
                seed0 = 7000 + 137 * nh + rec["runs_done"]
                jobs = [(nh, init, pred, seed0 + 3 * i, T, BURN) for i in range(need)]
                i = 0
                while i < len(jobs) and time.time() - t0 < BUDGET:
                    batch = jobs[i:i + WORKERS]
                    for lens in pool.imap_unordered(run_one, batch):
                        rec["lengths"] += lens; rec["runs_done"] += 1
                    i += WORKERS
                    data[key] = rec
                    json.dump(data, open(CACHE, "w"))
                print(f"{key}: runs_done={rec['runs_done']}/{NR} "
                      f"phases={len(rec['lengths'])} elapsed={time.time()-t0:.0f}s", flush=True)
    done = sum(1 for tag, _, _ in DESIGNS for nh in NH
               if data.get(f"{tag}|{nh}", {}).get("runs_done", 0) >= NR)
    print(f"complete combos: {done}/{len(DESIGNS)*len(NH)}")


if __name__ == "__main__":
    main()
