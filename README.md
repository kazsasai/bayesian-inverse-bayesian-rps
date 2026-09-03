# Internal-state criticality in Bayesian–inverse-Bayesian inference

Reproduction code for the figures and quantitative results of

> K. Sasai and Y.-P. Gunji, *Internal-state criticality in
> Bayesian–inverse-Bayesian inference.*

This repository contains **only the code and small cached summaries needed to
reproduce the figures and numbers** of the paper. The large raw simulation
outputs (~16 GB) are archived separately on Zenodo; the manuscript, journal
LaTeX template, and other editorial material are **not** part of this
repository.

---

## Contents

| Path | What it is |
|---|---|
| `simulation/` | Simulation engines: BIB / Bayes-only inference on the *N*-hand RPS game and reward-based variants (see [`simulation/README.md`](simulation/README.md) for the engine map and the `rpsgame_reward.py` version table) |
| `simulation_tie_mode_ablation/` | Tie-mode and observation-rule (scheme) ablation runners + analysis |
| `prod_reruns/` | Locked full-scale production re-runs (cached summaries + runners): matching pennies, laminar finite-size lock, drift/renewal prongs |
| `BIB_Levy_v2/latex/figures/` | Figure-build scripts (`scripts/build_*.py`, `regenerate_figures.py`, `figdata.py`) and rendered figure outputs |
| `BIB_Levy_v2/experiments/` | `tie_vs_smoothing` experiment (engine + data + plots) |
| `pnas/figures/` | Main-text figure scripts and outputs (Fig 1 composite, Fig 4 matching pennies) |
| `pnas_si/figures/` | SI figure scripts and outputs (dynamics demo, drift identity, reduced-walk scaling, behavioural re-analysis) |
| `requirements.txt` | Python dependencies (pinned) |
| `FIGURE_INVENTORY.md`, `BUILD_NOTES.md` | Figure ↔ script ↔ data notes |

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, scipy, matplotlib, powerlaw
```
Verified with Python 3.13.

## Data

The data-driven figures read aggregated simulation outputs archived on Zenodo
(DOI `10.5281/zenodo.20533918`). Two archives are provided; the *figure-only*
one (~1.1 GB) is sufficient to rebuild every figure.

```bash
# extract anywhere and point PAPERA_DATA at it
tar xzf paperA_data_figure_only.tar.gz -C /path/to/papera_data
export PAPERA_DATA=/path/to/papera_data
```
`figdata.py` searches `$PAPERA_DATA`, then `<repo>/data/`, then the in-repo
`simulation/` tree, in that order.

## Reproducing the figures

```bash
# the 10 data-driven figures (universality, finite-size scaling, ablations, ...)
python BIB_Levy_v2/latex/figures/scripts/build_all.py

# main-text figures not driven by the Zenodo trees
python pnas/figures/build_Fig1_bc.py            # Fig 1  (inline demo simulation, seed=42)
python pnas/figures/build_FigMP.py              # Fig 4  (matching pennies; prod_reruns cache)

# SI figures that are fully self-contained (no external data)
python pnas_si/figures/make_item1_drift_identity.py
python pnas_si/figures/make_prong2_part1_reduced.py
python pnas_si/figures/build_FigSI_dynamics.py
```

### Behavioural re-analysis (SI)

The Brockbank & Vul human-vs-bot re-analysis uses their public dataset
(~115 MB), fetched on demand:

```bash
bash pnas_si/figures/fetch_bv_data.sh           # clones github.com/erik-brockbank/rps
python pnas_si/figures/make_figS_behaviour.py   # rebuilds figS_bib_behaviour + cache
python pnas_si/figures/verify_figS_behaviour_clauset.py   # TPL-vs-EXP, Clauset auto-x_min
```

### Figure → script map

| Figure | Output | Script |
|---|---|---|
| Fig 1 | `fig_soc_mechanism.pdf` | hand-drawn schematic (not script-generated) |
| Fig 2 | `fig_dynamics_demo.pdf` | `BIB_Levy_v2/latex/figures/regenerate_figures.py 2` |
| Fig 3 | `fig_universality.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_fig_universality.py` |
| Fig 4 | `fig_control_ab.pdf` | `pnas_si/figures/build_fig_control.py` |
| Fig 5 | `fig_drift_residual.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_drift_residual.py` |
| Fig 6 | `fig_nh_sweep.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_fig_nh_sweep.py` |
| Fig 7 | `fig_bib_vs_bo.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_Fig_bib_vs_bo.py` |
| Fig 8 | `fig_scope.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_Fig_scope_internal_behaviour.py` |
| Fig 9 | `figS_bib_behaviour.pdf` | `pnas_si/figures/make_figS_behaviour.py` |
| Fig 10 | `fig_smoothing.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_FigS6_smoothing_comparison.py` |
| Fig 11 | `prong2_part1_reduced.pdf` | `pnas_si/figures/make_prong2_part1_reduced.py` |
| Fig 12 | `fig_robustness.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_Fig4_robustness.py` |
| Fig 13 | `fig_scheme_ablation.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_FigS7_scheme_ablation.py` |

See `FIGURE_INVENTORY.md` for the full list including SI figures.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
The archived data on Zenodo are released under CC-BY-4.0.

## Citation

Please cite the paper and the Zenodo data record. The repository README will be
updated with the full citation on publication.
