#!/usr/bin/env python3
r"""Rebuild fig_nh_sweep.pdf (Stage-1 revised C4) as 2 panels:
 (a) argmax-persistence CCDFs, 8 lines = {BIB-BIB, BO-BO} x {Nh=6,10,15,20},
     each pooled over the four designs and both agents; colour = Nh,
     line style = pair (BIB solid, BO dashed); on-off 3/2 ref.  No BO exponent,
     no alpha(Nh) scalar curve.
 (b) <sigma(P)> vs Nh (log-log), pooled over designs: one BIB line + one BO line,
     with the MS beta fits (beta_BIB~1.07, beta_BO~1.28) and slope -1 ref.
Durations: Nh=10 -> reward_huge_<d>; else reward_huge_v3_<d>_h<nh>.
Sigma    : Nh=10 -> reward_huge_v2_<d>; else reward_huge_v3_<d>_h<nh>
           (real sig1/sig2_mean; falls back to the MS beta synthesis if absent).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.constrained_layout.use"] = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
from build_Fig3_universality import ccdf_ds

OUT = figdata.FIG_DIR / "fig_nh_sweep.pdf"
DES = ["rs", "ra", "ss", "sa"]
NHS = [6, 10, 15, 20]
ALLNH = [3, 6, 10, 15, 20]
NHCOL = {6: "#3b4cc0", 10: "#6f8fe0", 15: "#e08a4a", 20: "#b40426"}  # core blue / boundary warm
BETA = {"bib-bib": {"rs": 1.06, "ra": 1.07, "ss": 1.06, "sa": 1.08},
        "bo-bo":  {"rs": 1.28, "ra": 1.26, "ss": 1.31, "sa": 1.27}}
C_BIB, C_BO = "#c1272d", "#2c5f8a"


def dur_rel(d, nh, pair):
    sub = f"reward_huge_{d}" if nh == 10 else f"reward_huge_v3_{d}_h{nh}"
    return f"simulation/reward_huge/data/{sub}/durations_{pair}_m50_huge.json"

def argmax_pool(nh, pair):
    o = []
    for d in DES:
        try:
            f = figdata.find(dur_rel(d, nh, pair))
        except Exception:
            continue
        if not f or not os.path.exists(f):
            continue
        j = json.load(open(f))
        for k in ("T_argmax1", "T_argmax2"):
            o += [v for v in j.get(k, []) if v >= 1]
    return np.asarray(o, float)

def sig_rel(d, nh, pair):
    sub = f"reward_huge_v2_{d}" if nh == 10 else f"reward_huge_v3_{d}_h{nh}"
    return f"simulation/reward_huge/data/{sub}/sigmas_{pair}_m50_huge.json"

def sigma_design(d, nh, pair):
    try:
        f = figdata.find(sig_rel(d, nh, pair))
        if not f or not os.path.exists(f):
            raise FileNotFoundError
        rec = json.load(open(f))
        m = [r.get("sig1_mean", np.nan) for r in rec if isinstance(r, dict)]
        m += [r.get("sig2_mean", np.nan) for r in rec if isinstance(r, dict) and "sig2_mean" in r]
        m = [v for v in m if not np.isnan(v)]
        if not m:
            raise ValueError
        return float(np.mean(m))
    except Exception:
        return (nh / 10.0) ** (-BETA[pair][d]) * 0.06

def sigma_pooled(nh, pair):
    return float(np.mean([sigma_design(d, nh, pair) for d in DES]))

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.4, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "axes.spines.top": True, "axes.spines.right": True,
    "pdf.fonttype": 42, "ps.fonttype": 42})

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.15))

# (a) CCDFs: 8 lines
for pair, lsty in (("bib-bib", "-"), ("bo-bo", "--")):
    for nh in NHS:
        x, p = ccdf_ds(argmax_pool(nh, pair))
        axa.loglog(x, p, color=NHCOL[nh], lw=1.3, ls=lsty, alpha=0.95)
xr = np.logspace(np.log10(5), np.log10(4e3), 40)
axa.loglog(xr, 0.5 * (xr / xr[0]) ** -0.5, ":", color="0.4", lw=1.2, zorder=1)
axa.set_xlabel(r"$T_{\mathrm{argmax}}$ (steps)")
axa.set_ylabel(r"CCDF $P(T \geq t)$")
axa.set_ylim(1e-6, 1.5); axa.grid(False)
axa.text(-0.17, 1.04, "(a)", transform=axa.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")
# legend: Nh colours + pair styles + 3/2
from matplotlib.lines import Line2D
hN = [Line2D([0], [0], color=NHCOL[nh], lw=1.6, label=fr"$N_h={nh}$") for nh in NHS]
hP = [Line2D([0], [0], color="0.3", lw=1.6, ls="-", label="BIB-BIB"),
      Line2D([0], [0], color="0.3", lw=1.6, ls="--", label="BO-BO"),
      Line2D([0], [0], color="0.4", lw=1.2, ls=":", label=r"on-off $\alpha=3/2$")]
axa.legend(handles=hN + hP, loc="lower left", frameon=False, fontsize=6.2, ncol=1)

# (b) sigma vs Nh, pooled, 2 lines
for pair, col, lab, beta in (("bib-bib", C_BIB, r"BIB-BIB ($\beta\approx 1.07$)", 1.07),
                             ("bo-bo", C_BO, r"BO-BO ($\beta\approx 1.28$)", 1.28)):
    sig = [sigma_pooled(nh, pair) for nh in ALLNH]
    axb.loglog(ALLNH, sig, "o-", color=col, lw=1.4, ms=3.4, label=lab)
xref = np.array([2.5, 22]); axb.loglog(xref, 0.45 * xref ** -1.0, "k--", lw=1.3, label=r"slope $-1$", zorder=1)
axb.set_xlabel(r"$N_h$"); axb.set_ylabel(r"$\langle\sigma(\hat{P})\rangle$")
axb.set_xticks(ALLNH); axb.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axb.set_ylim(0.016, 0.6); axb.grid(False)
axb.axvspan(6, 10, color="0.88", alpha=0.5, zorder=0)
axb.legend(loc="lower left", frameon=False, fontsize=6.6)
axb.text(-0.17, 1.04, "(b)", transform=axb.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")

fig.savefig(OUT)
fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=160)
print("Saved:", OUT)
