#!/usr/bin/env python3
r"""Build fig_exponent_bootstrap.pdf -- bootstrap TPL-alpha distributions,
BIB-BIB vs BO-BO, per design, for argmax (a) and laminar (b) observables.

Reads bo_verdict/bootstrap_exponents.json (B=1000 alpha arrays) and
bo_verdict/bo_verdict.json (Akaike verdict, for FAIL annotations).  BO cells
whose Akaike winner is not a heavy tail (PL/TPL) are annotated with the winning
model; their TPL-alpha violin is drawn faded because the exponent is not a
well-defined power-law slope there.

Style matches the manuscript figures (APS: full box, inward ticks on all four
sides, lowercase panel labels, constrained layout, no bbox-tight).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bo_verdict")
boot = json.load(open(os.path.join(OUT, "bootstrap_exponents.json")))
verd = json.load(open(os.path.join(OUT, "bo_verdict.json")))
V = {(r["pair"], r["design"], r["observable"], r["Nh"]): r for r in verd}

DESIGNS = ["rs", "ra", "ss", "sa"]
C_BIB, C_BO = "#c1272d", "#2c5f8a"   # crimson / steel blue
WIN_ABBR = {"power_law": "PL", "truncated_power_law": "TPL", "exponential": "exp",
            "lognormal": "logn", "stretched_exponential": "str-exp"}
REF = {"argmax": 1.5, "laminar": 1.5}  # on-off 3/2 reference

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15))

def panel(ax, obs, letter):
    ax.axhline(1.5, ls=":", lw=1.0, color="0.45", zorder=1,
               label=r"on-off $\alpha=3/2$")
    for xi, d in enumerate(DESIGNS):
        for pair, col, off in (("bib-bib", C_BIB, -0.19), ("bo-bo", C_BO, +0.19)):
            key = f"{obs}|{pair}|{d}"
            a = np.asarray(boot[key]["alphas"])
            pos = xi + off
            rec = V[(pair, d, obs, 10)]
            fail = rec["winner"] not in ("power_law", "truncated_power_law") or not rec["PASS"]
            face_alpha = 0.30 if (pair == "bo-bo" and fail) else 0.65
            vp = ax.violinplot([a], positions=[pos], widths=0.32,
                               showextrema=False, showmedians=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_edgecolor(col)
                b.set_alpha(face_alpha); b.set_linewidth(0.8)
            # pooled point estimate
            ax.plot(pos, boot[key]["point"], "o", ms=3.0, color="black", zorder=5)
            # annotate BO failing cells with the winning model
            if pair == "bo-bo" and fail:
                ax.annotate(WIN_ABBR[rec["winner"]], (pos, np.percentile(a, 95)),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=6.0, color=col, rotation=0)
    ax.set_xticks(range(len(DESIGNS)))
    ax.set_xticklabels(DESIGNS)
    ax.set_xlim(-0.6, len(DESIGNS) - 0.4)
    ax.set_xlabel("Design")
    ax.set_ylabel(r"bootstrap TPL exponent $\alpha$")
    ax.text(-0.16, 1.02, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")
    ax.set_title({"argmax": "Argmax persistence", "laminar": "Laminar phase"}[obs],
                 fontsize=8)

panel(axes[0], "argmax", "a")
panel(axes[1], "laminar", "b")

# shared legend (BIB / BO / ref)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [Patch(facecolor=C_BIB, alpha=0.65, label="BIB-BIB"),
           Patch(facecolor=C_BO, alpha=0.5, label="BO-BO"),
           Line2D([0], [0], marker="o", color="black", lw=0, ms=3, label="pooled estimate"),
           Line2D([0], [0], ls=":", color="0.45", label=r"on-off $\alpha=3/2$")]
axes[0].legend(handles=handles, loc="upper left", frameon=False, fontsize=6.8)

OUTPDF = os.path.join(OUT, "fig_exponent_bootstrap.pdf")
fig.savefig(OUTPDF)
fig.savefig(OUTPDF.replace(".pdf", ".png"), dpi=200)
print("wrote", OUTPDF)
