#!/usr/bin/env python3
"""Build every paper figure from its reproduction script.

Resolves data via figdata (set $PAPERA_DATA, or extract the Zenodo archive to
<repo>/data/; otherwise the in-repo simulation/ tree is used).  Each figure has
exactly one build command below.  Run all, or pass figure numbers:

    python build_all.py            # all figures
    python build_all.py 3 6 8      # only Fig 3, 6, 8

Fig 1 is a hand-drawn schematic (no data) and is not built here.
"""
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# paper figure number -> (script, extra args).  Output files are fig_<name>.pdf.
FIGS = {
    "2":  ("regenerate_figures.py", ["2"]),               # fig_dynamics_demo
    "3":  ("build_Fig3_universality.py", []),             # fig_universality
    "4":  ("build_FigS3b_plateau.py", []),                # fig_plateau
    "5":  ("build_FigS2_nh_ccdf.py", []),                 # fig_nh_ccdf
    "6":  ("regenerate_figures.py", ["8"]),               # fig_nh_sweep
    "7":  ("build_Fig4_robustness.py", []),               # fig_robustness
    "8":  ("build_FigS7_scheme_ablation.py", []),         # fig_scheme_ablation
    "9":  ("build_Fig_bib_vs_bo.py", []),                 # fig_bib_vs_bo (reward + tournament)
    "10": ("build_Fig_scope_internal_behaviour.py", []),  # fig_scope
    "11": ("build_FigS6_smoothing_comparison.py", []),    # fig_smoothing
}


def _resolve(script):
    """Find a build script in this scripts/ dir or its parent figures/ dir.
    (regenerate_figures.py lives one level up in figures/, not in scripts/.)"""
    for base in (HERE, os.path.dirname(HERE)):
        p = os.path.join(base, script)
        if os.path.exists(p):
            return p
    return os.path.join(HERE, script)  # fall back; will error visibly if missing


def main():
    targets = sys.argv[1:] or list(FIGS)
    ok, fail = [], []
    for k in targets:
        if k not in FIGS:
            print(f"[skip] unknown figure {k}"); continue
        script, args = FIGS[k]
        print(f"\n=== Fig {k}: {script} {' '.join(args)} ===", flush=True)
        # exit code 2 = "cache incomplete, re-invoke" (chunked CCDF builders);
        # 0 = done; anything else = failure.
        for _ in range(40):
            r = subprocess.run([PY, _resolve(script), *args])
            if r.returncode != 2:
                break
        (ok if r.returncode == 0 else fail).append(k)
    print(f"\nDONE. built: {ok}" + (f"   FAILED: {fail}" if fail else ""))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
