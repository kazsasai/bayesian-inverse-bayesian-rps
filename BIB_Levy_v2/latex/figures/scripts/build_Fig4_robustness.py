"""Rebuild Fig 6 (file fig_robustness.pdf): robustness of the BIB
argmax-persistence universality, as a clean two-panel VECTOR PDF.

Design (data-first; see project memory feedback_visualize_over_fits):
  (a) Window-size m sweep -- pooled BIB-BIB T_argmax CCDFs for m=10,20,50,100;
      the alpha=1.43 reference is drawn ONLY over the power-law region and
      anchored on the m=50 curve so it visibly parallels the data.
  (b) Number-of-hands N sweep -- raw T_argmax CCDFs at N=3,5,7 for BIB (solid)
      and BO (dashed). The BIB curves keep a common slope (~alpha=1.43) across
      N; BO does not. No fitted BO exponent is reported (BO fits are
      version/x_min-sensitive); the data itself carries the message.

Two-stage + cached so it fits short time budgets: each call computes the
pooled CCDFs it can (cache JSON), and once all 10 series are cached it draws
the figure. Re-invoke until it prints "Saved:".

Panel labels: bold lower-case, top-left just above the axes (house style).
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import powerlaw
import warnings
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figdata
FIG_DIR = str(figdata.FIG_DIR)
OUT_PDF = os.path.join(FIG_DIR, "fig_robustness.pdf")
OUT_PNG = os.path.join(FIG_DIR, "fig_robustness.png")

# Intermediate CCDF cache: keep it in the in-repo tree (always writable),
# independent of where the source data resolves ($PAPERA_DATA / <repo>/data).
CACHE = str(figdata.REPO / "simulation" / "nhand" / "data" /
            "fig6_ccdf_cache.json")

DES = ["rs", "ra", "ss", "sa"]
RHF = {"rs": "reward_huge_rs", "ra": "reward_huge_ra",
       "ss": "reward_huge_ss", "sa": "reward_huge_sa"}


def rh(pair, m, d):
    # reward_huge durations, resolved via figdata ($PAPERA_DATA / <repo>/data
    # / in-repo simulation tree).
    return str(figdata.find(
        f"simulation/reward_huge/data/{RHF[d]}/"
        f"durations_{pair}_m{m}_huge.json"))


def nhd(N, pair, d):
    return str(figdata.find(
        f"simulation/nhand/data/nhand_grid_N{N}_{d}/"
        f"durations_N{N}_{pair}_m50_huge.json"))


def pool(paths, keys=("T_argmax1", "T_argmax2")):
    out = []
    for p in paths:
        j = json.load(open(p))
        for k in keys:
            out += [v for v in j.get(k, []) if v >= 1]
    return np.asarray(out, float)


def ccdf_ds(arr, npts=400):
    a = np.sort(arr)
    n = len(a)
    ux, idx = np.unique(a, return_index=True)
    p = (n - idx) / n
    if len(ux) > npts:                       # log-spaced downsample
        sel = np.unique(np.round(np.logspace(0, np.log10(len(ux) - 1), npts))
                        .astype(int))
        ux, p = ux[sel], p[sel]
    return ux.tolist(), p.tolist()


def tpl_alpha(arr):
    if arr.size < 30:
        return float("nan")
    return float(powerlaw.Fit(arr, discrete=True,
                              verbose=False).truncated_power_law.alpha)


SERIES = (
    [("a", f"m{m}", lambda m=m: [rh("bib-bib", m, d) for d in DES])
     for m in (10, 20, 50, 100)]
    + [("b", f"N{N}_{pair}",
        lambda N=N, pair=pair: ([rh(pair, 50, d) for d in DES] if N == 3
                                else [nhd(N, pair, d) for d in DES]))
       for N in (3, 5, 7) for pair in ("bib-bib", "bo-bo")])


def build_cache(budget=36.0):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    t0 = time.time()
    done = 0
    for panel, tag, paths_fn in SERIES:
        key = f"{panel}|{tag}"
        if key in cache:
            continue
        if time.time() - t0 > budget:
            break
        arr = pool(paths_fn())
        x, p = ccdf_ds(arr)
        entry = {"x": x, "p": p}
        if panel == "a" or tag.endswith("bib-bib"):
            entry["alpha"] = tpl_alpha(arr[(arr >= 5) & (arr <= 10000)])
        cache[key] = entry
        done += 1
    json.dump(cache, open(CACHE, "w"))
    return cache, done


def panel_label(ax, letter):
    ax.text(-0.17, 1.04, letter.upper(), transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


def plot(cache):
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif",
                         "axes.linewidth": 0.8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.bbox": "tight"})
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":6,"axes.linewidth":0.8,"lines.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","xtick.major.size":3,"ytick.major.size":3,"xtick.minor.size":1.8,"ytick.minor.size":1.8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})  # PNAS-unified
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.05))

    # ---- (a) m sweep ----
    cmap = plt.cm.viridis
    mlist = [10, 20, 50, 100]
    mcol = {m: cmap(0.12 + 0.72 * i / 3) for i, m in enumerate(mlist)}
    for m in mlist:
        e = cache[f"a|m{m}"]
        axA.loglog(e["x"], e["p"], color=mcol[m], lw=2.0,
                   label=fr"$m={m}$  ($\alpha\approx{e['alpha']:.2f}$)")
    # alpha=1.43 reference: power-law region only, anchored on m=50 curve
    e50 = cache["a|m50"]
    x50, p50 = np.array(e50["x"]), np.array(e50["p"])
    x0 = 10.0
    y0 = np.exp(np.interp(np.log(x0), np.log(x50), np.log(p50)))
    xr = np.logspace(np.log10(4), np.log10(400), 50)
    yr = y0 * (xr / x0) ** (-(1.43 - 1.0))
    axA.loglog(xr, yr, "--", color="0.25", lw=1.6,
               label=r"$\alpha=1.43$ reference")
    axA.set_xlabel(r"$T_{\mathrm{argmax}}$ (steps)")
    axA.set_ylabel(r"CCDF $\;P(T \geq t)$")
    axA.set_ylim(1e-6, 1.5)
    axA.grid(False)
    axA.legend(loc="lower left", frameon=False, fontsize=7.5)
    panel_label(axA, "a")

    # ---- (b) N sweep CCDFs, BIB solid / BO dashed ----
    Ncol = {3: "#1b9e77", 5: "#7570b3", 7: "#d95f02"}
    for N in (3, 5, 7):
        eb = cache[f"b|N{N}_bib-bib"]
        axB.loglog(eb["x"], eb["p"], color=Ncol[N], ls="-", lw=2.1,
                   label=fr"\BIB{{}} $N={N}$" if False else f"BIB  N={N}")
        eo = cache[f"b|N{N}_bo-bo"]
        axB.loglog(eo["x"], eo["p"], color=Ncol[N], ls="--", lw=1.6,
                   alpha=0.9, label=f"BO   N={N}")
    # alpha=1.43 slope reference anchored on BIB N=3
    e3 = cache["b|N3_bib-bib"]
    x3, p3 = np.array(e3["x"]), np.array(e3["p"])
    x0 = 10.0
    y0 = np.exp(np.interp(np.log(x0), np.log(x3), np.log(p3)))
    xr = np.logspace(np.log10(4), np.log10(400), 50)
    yr = y0 * (xr / x0) ** (-(1.43 - 1.0))
    axB.loglog(xr, yr, ":", color="0.2", lw=1.6,
               label=r"$\alpha=1.43$ slope")
    axB.set_xlabel(r"$T_{\mathrm{argmax}}$ (steps)")
    axB.set_ylabel(r"CCDF $\;P(T \geq t)$")
    axB.set_ylim(1e-6, 1.5)
    axB.grid(False)
    axB.legend(loc="lower left", frameon=False, fontsize=6.5, ncol=1)
    panel_label(axB, "b")

    fig.subplots_adjust(wspace=0.24)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print("Saved:", OUT_PDF)


def main():
    cache, done = build_cache()
    need = [f"{p}|{t}" for p, t, _ in SERIES]
    have = [k for k in need if k in cache]
    print(f"cached this call: {done};  series {len(have)}/{len(need)}")
    if len(have) == len(need):
        plot(cache)
    else:
        print("re-invoke to finish caching")
        sys.exit(2)


if __name__ == "__main__":
    main()
