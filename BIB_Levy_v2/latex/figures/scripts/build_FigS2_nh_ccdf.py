"""Rebuild Fig 4 (file fig_nh_ccdf.pdf): argmax-persistence CCDFs
across the hypothesis count N_h, a 2x4 grid (rows = BIB-BIB / BO-BO;
columns = the four designs rs/ra/ss/sa), each panel overlaying
N_h in {3,6,10,15,20}.  A truncated-power-law reference is drawn on the
BIB-BIB panels (fitted at the canonical N_h=10).

Data-first, reproducible: data resolved via figdata (env $PAPERA_DATA ->
<repo>/data -> in-repo simulation tree).  CCDFs are cached (chunked) so the
script fits short time budgets; re-invoke until it prints "Saved:".

House style: bold lower-case (a)-(h) panel labels top-left; design names as
column headers and BIB-BIB/BO-BO as row labels are kept as grid structure.
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.constrained_layout.use"] = True
import powerlaw
import warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata

OUT_PDF = figdata.FIG_DIR / "fig_nh_ccdf.pdf"
OUT_PNG = figdata.FIG_DIR / "fig_nh_ccdf.png"
CACHE = figdata.REPO / "simulation" / "nhand" / "data" / "figS2_ccdf_cache.json"

DES = ["rs", "ra", "ss", "sa"]
DES_NAME = {"rs": "rs (random + sample)", "ra": "ra (random + argmax)",
            "ss": "ss (struct + sample)", "sa": "sa (struct + argmax)"}
RHF = {"rs": "reward_huge_rs", "ra": "reward_huge_ra",
       "ss": "reward_huge_ss", "sa": "reward_huge_sa"}
NHS = [3, 6, 10, 15, 20]
PAIRS = ["bib-bib", "bo-bo"]


def dur_relpath(design, pair, nh):
    if nh == 10:
        return f"simulation/reward_huge/data/{RHF[design]}/durations_{pair}_m50_huge.json"
    return (f"simulation/reward_huge/data/reward_huge_v3_{design}_h{nh}/"
            f"durations_{pair}_m50_huge.json")


def load_pool(design, pair, nh):
    j = json.load(open(figdata.find(dur_relpath(design, pair, nh))))
    out = []
    for k in ("T_argmax1", "T_argmax2"):
        out += [v for v in j.get(k, []) if v >= 1]
    return np.asarray(out, float)


def ccdf_ds(arr, npts=400):
    a = np.sort(arr)
    n = len(a)
    if n == 0:
        return [], []
    ux, idx = np.unique(a, return_index=True)
    p = (n - idx) / n
    if len(ux) > npts:
        sel = np.unique(np.round(np.logspace(0, np.log10(len(ux) - 1), npts))
                        .astype(int))
        ux, p = ux[sel], p[sel]
    return ux.tolist(), p.tolist()


def build_cache(budget=36.0):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    t0 = time.time(); done = 0
    for design in DES:
        for pair in PAIRS:
            for nh in NHS:
                key = f"{design}|{pair}|{nh}"
                if key in cache:
                    continue
                if time.time() - t0 > budget:
                    json.dump(cache, open(CACHE, "w")); return cache, done, False
                arr = load_pool(design, pair, nh)
                x, p = ccdf_ds(arr)
                entry = {"x": x, "p": p}
                if pair == "bib-bib" and nh == 10:   # TPL reference fit
                    a = arr[arr >= 1]
                    entry["alpha"] = float(powerlaw.Fit(
                        a, discrete=True, verbose=False).truncated_power_law.alpha)
                cache[key] = entry
                done += 1
    json.dump(cache, open(CACHE, "w"))
    return cache, done, True


def plot(cache):
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif",
                         "axes.linewidth": 0.8, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    cmap = plt.cm.viridis
    ncol = {nh: cmap(i / (len(NHS) - 1)) for i, nh in enumerate(NHS)}
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":6,"axes.linewidth":0.8,"lines.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,"xtick.major.size":3,"ytick.major.size":3,"xtick.minor.size":1.8,"ytick.minor.size":1.8,"axes.spines.top":True,"axes.spines.right":True,"pdf.fonttype":42,"ps.fonttype":42})  # PNAS-unified
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), sharex=True, sharey=True)
    tags = "abcdefgh"
    for r, pair in enumerate(PAIRS):
        for c, design in enumerate(DES):
            ax = axes[r, c]
            for nh in NHS:
                e = cache[f"{design}|{pair}|{nh}"]
                ax.loglog(e["x"], e["p"], color=ncol[nh], lw=1.8,
                          label=fr"$N_h={nh}$")
            if pair == "bib-bib":
                e10 = cache[f"{design}|bib-bib|10"]
                a = e10.get("alpha")
                if a:
                    x10, p10 = np.array(e10["x"]), np.array(e10["p"])
                    x0 = 10.0
                    y0 = np.exp(np.interp(np.log(x0), np.log(x10), np.log(p10)))
                    xr = np.logspace(np.log10(3), np.log10(2e3), 50)
                    yr = y0 * (xr / x0) ** (-(a - 1.0))
                    ax.loglog(xr, yr, "--", color="0.35", lw=1.3)
                    ax.text(0.60, 0.70, fr"TPL fit ($\alpha={a:.2f}$)",
                            transform=ax.transAxes, fontsize=6.5,
                            color="0.25", rotation=-22, ha="center")
            ax.set_ylim(1e-6, 1.6)
            ax.set_xlim(1, 1e5)
            ax.grid(False)
            # (panel letters removed: the caption labels panels by
            #  column=design and row=BIB/BO, not by letter)
            if r == 0:                                   # column headers
                ax.set_title(DES_NAME[design], fontsize=7.5)
            if c == 0:                                   # row labels
                lab = r"\BIB" if False else ("BIB-BIB" if pair == "bib-bib" else "BO-BO")
                ax.set_ylabel(f"{lab}\n" r"CCDF $P(T\geq t)$", fontsize=7.5)
            if r == 1:
                ax.set_xlabel(r"$T_{\mathrm{argmax}}$ (steps)", fontsize=7.5)
    handles = [plt.Line2D([0], [0], color=ncol[nh], lw=2.2) for nh in NHS]
    fig.legend(handles, [fr"$N_h={nh}$" for nh in NHS], loc="lower center",
               ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=160)
    print("Saved:", OUT_PDF)


def main():
    cache, done, complete = build_cache()
    total = len(DES) * len(PAIRS) * len(NHS)
    have = sum(1 for d in DES for p in PAIRS for n in NHS if f"{d}|{p}|{n}" in cache)
    print(f"cached this call: {done}; series {have}/{total}")
    if have == total:
        plot(cache)
    else:
        print("re-invoke to finish caching")
        sys.exit(2)


if __name__ == "__main__":
    main()
