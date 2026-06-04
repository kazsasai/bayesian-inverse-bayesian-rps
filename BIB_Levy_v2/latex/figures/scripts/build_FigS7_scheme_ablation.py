#!/usr/bin/env python3
"""
build_fig_scheme_ablation.py
Regenerates fig_scheme_ablation.pdf (paper Fig 7) as a clean VECTOR PDF.

Why this script exists
----------------------
The previous FigS7 PDF was a raster image whose two panels carried no panel
letters (title only). The LaTeX caption now uses "(a, left) / (b, right)", so
the figure must show bold lower-case "(a)" and "(b)" inside the panels. The
original assembly script was lost; this rebuild reads the authoritative fit
summary and reproduces the same two grouped bar charts.

Data source (authoritative, in repo):
    simulation_tie_mode_ablation/data/scheme_ablation/medium/scheme_summary.csv
    columns: design,pair,scheme,...,alpha,...

Panels
------
(a) BIB-BIB: alpha(T_argmax1) across the four designs x four observation schemes.
(b) BO-BO:   same, for BO-BO play.

Output: ../fig_scheme_ablation.pdf (and .png preview)
"""
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figdata
FIG_DIR = str(figdata.FIG_DIR)

# scheme_summary.csv via figdata ($PAPERA_DATA / <repo>/data / in-repo
# simulation_tie_mode_ablation tree), with the bundled copy as final fallback.
_rel = ("simulation_tie_mode_ablation/data/scheme_ablation/medium/"
        "scheme_summary.csv")
CSV_PATH = str(figdata.find(_rel)) if figdata.exists(_rel) \
    else str(figdata.bundled("scheme_summary.csv"))

OUT_PDF = os.path.join(FIG_DIR, "fig_scheme_ablation.pdf")
OUT_PNG = os.path.join(FIG_DIR, "fig_scheme_ablation.png")

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def panel_label(ax, letter):
    ax.text(-0.17, 1.04, letter.upper(), transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

DESIGNS = ["rs", "ra", "ss", "sa"]
SCHEMES = ["eq3", "caseA", "hybrid", "defeatGT"]
SCHEME_COLORS = {
    "eq3":      "#2c5f8a",   # blue
    "caseA":    "#c0392b",   # red
    "hybrid":   "#6e8b3d",   # olive/green
    "defeatGT": "#7d5ba6",   # purple
}

# alpha[pair][design][scheme]
alpha = {"bib-bib": {d: {} for d in DESIGNS},
         "bo-bo":   {d: {} for d in DESIGNS}}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        p, d, s = row["pair"], row["design"], row["scheme"]
        if p in alpha and d in alpha[p] and s in SCHEMES:
            alpha[p][d][s] = float(row["alpha"])

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"font.size":8,"axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":6,"axes.linewidth":0.8,"lines.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","xtick.major.size":3,"ytick.major.size":3,"xtick.minor.size":1.8,"ytick.minor.size":1.8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})  # PNAS-unified
fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.05), sharey=True)

def draw(ax, pair, title, letter):
    x = np.arange(len(DESIGNS))
    nb = len(SCHEMES)
    w = 0.78 / nb
    ax.axhspan(1.0, 3.0, color="#fff2a8", alpha=0.45, lw=0, zorder=0)  # Levy regime 1<alpha<3
    for k, s in enumerate(SCHEMES):
        vals = [alpha[pair][d].get(s, np.nan) for d in DESIGNS]
        ax.bar(x + (k - (nb - 1) / 2) * w, vals, w,
               color=SCHEME_COLORS[s], edgecolor="black", lw=0.5, label=s)
    ax.axhline(1.0, ls="--", color="#b8860b", lw=1.0, alpha=0.7)  # exp/Poisson ref
    ax.axhline(1.5, ls=":", color="0.5", lw=1.0, alpha=0.7)       # on-off 3/2 ref
    ax.set_xticks(x)
    ax.set_xticklabels(DESIGNS)
    ax.set_ylim(0, 3.1)
    ax.set_ylabel(r"$\alpha$ (T_argmax1, best fit)")
    ax.legend(ncol=2, fontsize=7, loc="upper left", frameon=False)
    ax.set_facecolor("white")
    panel_label(ax, letter)

draw(axL, "bib-bib", r"BIB-BIB: $\alpha$ across designs $\times$ schemes", "a")
draw(axR, "bo-bo",  r"BO-BO: $\alpha$ across designs $\times$ schemes", "b")

fig.subplots_adjust(wspace=0.10)
fig.tight_layout()
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=200)
print("loaded alpha values:")
for p in ("bib-bib", "bo-bo"):
    for d in DESIGNS:
        print(f"  {p} {d}: " +
              ", ".join(f"{s}={alpha[p][d].get(s, float('nan')):.3f}" for s in SCHEMES))
print("wrote:", OUT_PDF)
print("wrote:", OUT_PNG)
