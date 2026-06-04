"""Shared data-path resolver for Paper A figure-build scripts.

Resolution order for a dataset-relative path:
  1. $PAPERA_DATA            (explicit override; e.g. extracted Zenodo archive)
  2. <repo>/data             (default Zenodo extraction target)
  3. <repo>                  (in-repo simulation/ tree; works before archiving)

Relative paths passed to ``find`` are given from the data root and include the
top-level simulation tree, e.g.
    "simulation/reward_huge/data/reward_huge_rs/durations_bib-bib_m50_huge.json"
so the same call resolves against the in-repo tree (now) and against the
Zenodo archive extracted to <repo>/data (after release).

Usage in a build script:
    import figdata
    p = figdata.find("simulation/reward_huge/data/.../durations_...json")
    fig.savefig(figdata.FIG_DIR / "MyFig.pdf")
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve()
# figures/scripts -> figures -> latex -> BIB_Levy_v2 -> BIB_Analyze (= data root,
# the tree that holds simulation/, nhand/, etc.)
REPO = _HERE.parents[4]
FIG_DIR = _HERE.parents[1]          # .../latex/figures (where PDFs are written)


def data_roots():
    """Candidate data roots, highest priority first."""
    roots = []
    env = os.environ.get("PAPERA_DATA")
    if env:
        roots.append(Path(env))
    roots.append(REPO / "data")     # Zenodo archive extracted here
    roots.append(REPO)              # in-repo simulation/* tree
    return roots


def find(relpath):
    """Return the first existing <root>/<relpath>; raise if none found."""
    tried = []
    for r in data_roots():
        p = r / relpath
        tried.append(str(p))
        try:
            if p.exists():
                return p
        except OSError:
            continue
    raise FileNotFoundError(
        "Data file not found. Set $PAPERA_DATA or extract the Zenodo "
        "archive to <repo>/data/. Looked for:\n  " + "\n  ".join(tried))


def exists(relpath):
    for r in data_roots():
        try:
            if (r / relpath).exists():
                return True
        except OSError:
            continue
    return False


BUNDLED = _HERE.parent / "data"          # figures/scripts/data (small inputs in-repo)


def bundled(name):
    """Path to a small input bundled in the repo (figures/scripts/data/)."""
    return BUNDLED / name
