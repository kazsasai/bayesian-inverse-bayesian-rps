"""
Publication-quality figure (PNAS SI style) for the Brockbank & Vul re-analysis.
Two panels:
  A. CCDF of human transition-run length: fixed/exploitable bots (v2) vs adaptive bots (v3)
  B. Exploitability (human win rate) vs behavioural persistence across 15 bot conditions
Outputs: PDF (vector) + PNG (300 dpi).

REPRODUCIBILITY
---------------
Raw data = Brockbank & Vul public repo  https://github.com/erik-brockbank/rps
  data/v2/*.json  -> 7 FIXED-strategy (exploitable) bots
  data/v3/*.json  -> 8 ADAPTIVE bots
The raw set is ~115 MB so it is NOT vendored in the paper repo. This script works two ways:

  (1) WITH raw data  -> set $RPS_DATA to a dir containing v2/ and v3/  (or place it at
      ./rps/data next to this script, e.g. via fetch_bv_data.sh). It recomputes everything
      AND writes a small cache (figS_behaviour_cache.json, ~tens of kB) next to this script.
  (2) WITHOUT raw data -> if the cache exists it is loaded and the figure is rebuilt from it.

So after running once with the data, the figure is fully regenerable from the cached
summary alone (no 115 MB dependency). Numbers: transition-run Spearman rho=+0.854 (p=1e-4).
"""
import os, json, glob, collections, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "figS_behaviour_cache.json")
# raw-data dir: $RPS_DATA, else ./rps/data next to script, else ./rps/data in CWD
_cand = [os.environ.get("RPS_DATA"), os.path.join(HERE, "rps", "data"), "rps/data"]
DATA_DIR = next((d for d in _cand if d and os.path.isdir(os.path.join(d, "v2"))), None)

# ---------- PNAS-style rcParams ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.linewidth": 0.8, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_FIX = "#D55E00"   # Okabe-Ito vermillion : fixed / exploitable (v2)
C_ADA = "#0072B2"   # Okabe-Ito blue       : adaptive (v3)
MOVE = {"rock": 0, "paper": 1, "scissors": 2}


def runs(symbols):
    out, cur, prev = [], 0, object()
    for s in symbols:
        if s == prev: cur += 1
        else:
            if cur: out.append(cur)
            cur, prev = 1, s
    if cur: out.append(cur)
    return out


def ccdf(a):
    a = np.asarray(a); x = np.unique(a)
    return x, np.array([np.mean(a >= xi) for xi in x])


def load(ver):
    recs = []
    for f in glob.glob(os.path.join(DATA_DIR, ver, "*.json")):
        if "freeResp" in f or "sliderData" in f: continue
        try: d = json.load(open(f))
        except Exception: continue
        strat, rounds = d.get("player2_bot_strategy"), d.get("rounds")
        if not strat or not rounds: continue
        mv, win = [], 0
        for r in rounds:
            m, o = r.get("player1_move"), r.get("player1_outcome")
            if m in MOVE and o in ("win", "loss", "tie"):
                mv.append(MOVE[m]); win += (o == "win")
        if len(mv) < 50: continue
        mv = np.array(mv)
        recs.append(dict(ver=ver, strat=strat, win_rate=win / len(mv),
                         tran=runs((np.diff(mv) % 3).tolist())))
    return recs


def build_from_raw():
    recs = load("v2") + load("v3")
    if not recs:
        raise SystemExit(f"No usable JSON under {DATA_DIR}/(v2|v3).")
    tran = {"v2": [v for r in recs if r["ver"] == "v2" for v in r["tran"]],
            "v3": [v for r in recs if r["ver"] == "v3" for v in r["tran"]]}
    cond = collections.defaultdict(lambda: dict(tran=[], wr=[], ver=None))
    for r in recs:
        c = cond[r["strat"]]; c["ver"] = r["ver"]
        c["tran"] += r["tran"]; c["wr"].append(r["win_rate"])
    rows = [dict(strat=k, ver=c["ver"], wr=float(np.mean(c["wr"])),
                 mean_tran=float(np.mean(c["tran"]))) for k, c in cond.items()]
    cx2, cy2 = ccdf(tran["v2"]); cx3, cy3 = ccdf(tran["v3"])
    cache = {"curveA": {"v2": [cx2.tolist(), cy2.tolist()],
                        "v3": [cx3.tolist(), cy3.tolist()]},
             "rows": rows, "source": "github.com/erik-brockbank/rps data/v2,v3"}
    with open(CACHE, "w") as fh:
        json.dump(cache, fh)
    print(f"[raw] {len(recs)} games from {DATA_DIR} -> wrote cache {os.path.basename(CACHE)}")
    return cache


def load_cache():
    with open(CACHE) as fh:
        return json.load(fh)


# ---------- acquire data (raw if available -> also refresh cache; else cache) ----------
if DATA_DIR is not None:
    C = build_from_raw()
elif os.path.exists(CACHE):
    C = load_cache(); print(f"[cache] rebuilt figure from {os.path.basename(CACHE)} (no raw data needed)")
else:
    raise SystemExit("No raw data ($RPS_DATA or ./rps/data) and no cache. "
                     "Run fetch_bv_data.sh first (clones github.com/erik-brockbank/rps).")

(cx2, cy2) = map(np.array, C["curveA"]["v2"])
(cx3, cy3) = map(np.array, C["curveA"]["v3"])
rows = C["rows"]
wr = np.array([r["wr"] for r in rows]); mt = np.array([r["mean_tran"] for r in rows])
rho_t, p_t = spearmanr(wr, mt)
print(f"transition-run gradient rho={rho_t:+.3f} p={p_t:.4f}")

# ---------- figure ----------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.05))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.93, wspace=0.32)

# Panel A
for (x, y), lab, col, mk in [((cx2, cy2), "fixed bots (exploitable)", C_FIX, "o"),
                             ((cx3, cy3), "adaptive bots", C_ADA, "s")]:
    axA.loglog(x, y, marker=mk, ms=3.0, color=col, mfc=col, mec=col, lw=1.0, label=lab)
axA.set_xlabel("Human transition-run length  $\\ell$")
axA.set_ylabel("CCDF   $P(L \\geq \\ell)$")
axA.set_ylim(2e-4, 1.3)
axA.legend(frameon=False, handlelength=1.4, loc="upper right",
           borderaxespad=0.3, labelspacing=0.3)
axA.tick_params(which="both")

# Panel B
for r in rows:
    col, mk = (C_FIX, "o") if r["ver"] == "v2" else (C_ADA, "s")
    axB.scatter(r["wr"], r["mean_tran"], color=col, marker=mk, s=30,
                edgecolor="black", linewidth=0.4, zorder=3)
b, a0 = np.polyfit(wr, mt, 1)
xs = np.linspace(wr.min(), wr.max(), 50)
axB.plot(xs, a0 + b * xs, ls=(0, (5, 3)), color="0.45", lw=1.0, zorder=2)
axB.axvline(1/3, color="0.3", ls=":", lw=0.8, zorder=1)
axB.text(1/3 + 0.006, axB.get_ylim()[0] + 0.04, "Nash 1/3",
         fontsize=6.5, color="0.3", rotation=0, va="bottom", ha="left")
axB.set_xlabel("Exploitability  (human win rate)")
axB.set_ylabel("Mean transition-run length")
axB.text(0.04, 0.93, f"Spearman $\\rho = {rho_t:+.2f}$",
         transform=axB.transAxes, fontsize=7.5)
axB.legend(handles=[Line2D([], [], marker="o", ls="", mfc=C_FIX, mec="black",
                           mew=0.4, ms=5, label="fixed (exploitable)"),
                    Line2D([], [], marker="s", ls="", mfc=C_ADA, mec="black",
                           mew=0.4, ms=5, label="adaptive")],
           frameon=False, loc="lower right", handlelength=1.0,
           borderaxespad=0.3, labelspacing=0.3)

# panel letters
for ax, L in [(axA, "A"), (axB, "B")]:
    ax.text(-0.17, 1.04, L, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"figS_bib_behaviour.{ext}"), bbox_inches="tight")
print("saved figS_bib_behaviour.pdf / .png")
