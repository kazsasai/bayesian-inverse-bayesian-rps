# Figure inventory — `Internal-state criticality in Bayesian–inverse-Bayesian inference`

Figure ↔ `\label` ↔ output file ↔ build script, in order of appearance in the
current manuscript (`apssamp.tex`, Physical Review Research version). For the
data tier behind each figure, see
[`BIB_Levy_v2/latex/figures/scripts/zenodo_data_manifest.txt`](BIB_Levy_v2/latex/figures/scripts/zenodo_data_manifest.txt);
Zenodo DOI `10.5281/zenodo.20533918`.

## Main text

| # | `\label` | Output file | Build script | Content |
|---:|---|---|---|---|
| 1 | `fig:soc-schematic` | `fig_soc_mechanism.pdf` | `pnas/figures/build_Fig1_bc.py` | Schematic of the Bayesian and inverse-Bayesian steps and the resulting power-law persistence. (Also emits `fig1_composite.pdf`.) |
| 2 | `fig:dynamics` | `fig_dynamics_demo.pdf` | `BIB_Levy_v2/latex/figures/regenerate_figures.py` | Hypothesis-space dynamics, BIB-BIB vs BO-BO (posterior trajectory + argmax track + σ). |
| 3 | `fig:univ` | `fig_universality.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_Fig3_universality.py` | Design-independent collapse in BIB and its absence in BO (argmax and laminar CCDFs). |
| 4 | `fig:plateau` | `fig_plateau.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_FigS3b_plateau.py` | Plateau-length CCDFs pooled over the four designs. |
| 5 | `fig:learners` | `fig_control_ab.pdf` | `pnas_si/figures/build_fig_control.py` | Specificity of the critical class vs adaptive learners (WSLS, Q-learning, regret matching, SIR). |
| 6 | `fig:nh-ccdf` | `fig_nh_ccdf.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_FigS2_nh_ccdf.py` | Argmax-persistence CCDFs across N_h ∈ {3,6,10,15,20}. |
| 7 | `fig:nhswp` | `fig_nh_sweep.pdf` | `BIB_Levy_v2/latex/figures/regenerate_figures.py` | N_h sweep test of criticality (argmax/laminar exponents and σ vs N_h). |
| 8 | `fig:robust` | `fig_robustness.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_Fig4_robustness.py` | BIB robustness sweeps (window size m; α(N)). |
| 9 | `fig:mp` | `fig_mp_generality.pdf` | `pnas/figures/build_FigMP.py` | Game-independence of the BIB critical class via matching pennies. |
| 10 | `fig:scheme-ablation` | `fig_scheme_ablation.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_FigS7_scheme_ablation.py` | Observation-rule (scheme) ablation at medium scale. |
| 11 | `fig:bib-vs-bo` | `fig_bib_vs_bo.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_Fig_bib_vs_bo.py` | BIB vs BO head-to-head reward and cross-design tournament. |
| 12 | `fig:scope` | `fig_scope.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_Fig_scope_internal_behaviour.py` | Internal-state criticality vs behavioral expression. |
| 13 | `fig:human` | `figS_bib_behaviour.pdf` | `pnas_si/figures/make_figS_behaviour.py` | Predicted behavioral crossover in existing human play (Brockbank & Vul re-analysis). |

## Appendix

| `\label` | Output file | Build script | Content |
|---|---|---|---|
| `fig:smoothing` | `fig_smoothing.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_FigS6_smoothing_comparison.py` | σ(P(h)) dynamics, conditional vs always-on posterior smoothing. |
| `fig:drift-identity` | `item1_drift_identity.pdf` | `pnas_si/figures/make_item1_drift_identity.py` | Log-posterior drift as a difference of Kullback–Leibler divergences. |
| `fig:drift-scaling` | `prong2_part1_reduced.pdf` | `pnas_si/figures/make_prong2_part1_reduced.py` | Drift as the relevant variable, with d = 0 the critical line. |
| `fig:drift-residual` | `fig_drift_residual.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_drift_residual.py` | Extrapolation of the argmax exponent to 3/2 as finite-sample drift → 0. |

---

*This inventory was regenerated against the current manuscript. The earlier
PRE-version inventory (different figure numbering and `*_PROD.pdf` file names) is
obsolete.*
