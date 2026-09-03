# Figure inventory — `Internal-state criticality in Bayesian–inverse-Bayesian inference`

Figure ↔ `\label` ↔ output file ↔ build script, in order of appearance in the
current manuscript (`apssamp.tex`, Physical Review Research version). For the
data tier behind each figure, see
[`BIB_Levy_v2/latex/figures/scripts/zenodo_data_manifest.txt`](BIB_Levy_v2/latex/figures/scripts/zenodo_data_manifest.txt);
Zenodo DOI `10.5281/zenodo.20533918`.

## Main text

| # | `\label` | Output file | Build script | Content |
|---:|---|---|---|---|
| 1 | `fig:soc-schematic` | `fig_soc_mechanism.pdf` | hand-drawn schematic (not script-generated) | Schematic of the Bayesian and inverse-Bayesian steps and the resulting power-law persistence. `pnas/figures/build_Fig1_bc.py` *consumes* this file as panel A and emits `fig1_composite.pdf`, which the manuscript does not use. |
| 2 | `fig:dynamics` | `fig_dynamics_demo.pdf` | `BIB_Levy_v2/latex/figures/regenerate_figures.py 2` | Hypothesis-space dynamics, BIB-BIB vs BO-BO: hand sequence, posterior `P(h)` heatmap with argmax track, and top-3 `P(h)` trajectories. |
| 3 | `fig:univ` | `fig_universality.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_fig_universality.py` | Design-independent collapse of the BIB argmax-persistence exponent (α≈1.43) and its absence under Bayes-only updating. (Single-panel argmax CCDF; loaders from `build_Fig3_universality.py`.) |
| 4 | `fig:learners` | `fig_control_ab.pdf` | `pnas_si/figures/build_fig_control.py` | Specificity of the critical class vs adaptive learners (WSLS, Q-learning, regret matching, SIR). |
| 5 | `fig:drift-residual` | `fig_drift_residual.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_drift_residual.py` | Extrapolation of the argmax exponent to 3/2 as the finite-sample drift residual → 0 (RPS window sweep + matching pennies). |
| 6 | `fig:nhswp` | `fig_nh_sweep.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_fig_nh_sweep.py` | N_h sweep test of criticality: argmax-persistence CCDFs by N_h and posterior-sharpness σ vs N_h. |
| 7 | `fig:bib-vs-bo` | `fig_bib_vs_bo.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_Fig_bib_vs_bo.py` | BIB vs BO head-to-head reward and cross-design tournament. |
| 8 | `fig:scope` | `fig_scope.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_Fig_scope_internal_behaviour.py` | Internal-state criticality vs behavioral expression. |
| 9 | `fig:human` | `figS_bib_behaviour.pdf` | `pnas_si/figures/make_figS_behaviour.py` | Predicted behavioral crossover in existing human play (Brockbank & Vul re-analysis). |

## Appendix

| `\label` | Output file | Build script | Content |
|---|---|---|---|
| `fig:smoothing` | `fig_smoothing.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_FigS6_smoothing_comparison.py` | σ(P(h)) dynamics, conditional vs always-on posterior smoothing. |
| `fig:drift-scaling` | `prong2_part1_reduced.pdf` | `pnas_si/figures/make_prong2_part1_reduced.py` | Drift as the relevant variable, with d = 0 the critical line. |
| `fig:robust` | `fig_robustness.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_Fig4_robustness.py` | BIB robustness sweeps (window size m; α(N)). |
| `fig:scheme-ablation` | `fig_scheme_ablation.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_FigS7_scheme_ablation.py` | Observation-rule (scheme) ablation at medium scale. |

## Reference figures (kept in the repository, not shown in the manuscript)

These three heavy-tailed-CCDF figures were removed from the manuscript during the
§III ↔ §IV reorganization because each one repeats the same collapse already
carried by `fig:univ`; their numerical results are reported in the prose
instead. The rendered PDFs and their build scripts are retained here for
reference and reuse (they remain in use in the PNAS/PRE versions of the paper).

| Output file | Build script | Result now reported in the body text |
|---|---|---|
| `fig_laminar.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_fig_universality.py` (emitted alongside `fig_universality.pdf`) | §IV.B (`sec:soc-onoff`): laminar exponent α<sup>lam</sup> = 1.325 ± 0.006, cross-design range 0.014; BO-BO fails the heavy-tail criterion. |
| `fig_mp_generality.pdf` | `pnas/figures/build_FigMP.py` | §VI (`sec:robust-mp`): matching-pennies argmax α ≈ 1.47, laminar α ≈ 1.42, truncated power law preferred (R = 75–89); coincide with the RPS exponents. |
| `fig_plateau.pdf` | `BIB_Levy_v2/latex/figures/scripts/build_fig_plateau.py` | Appendix (`app:plateau`): ≈ 3.6 decades, gap 0.18, α ≈ 1.23; BO-BO does not pass. |

---

*This inventory reflects the current PRR manuscript after the §III/§IV
reorganization (single-observable §III, on-off analysis consolidated into §IV.B)
and the text-ification of the three lookalike CCDF figures above. The earlier
PRE-version inventory (different figure numbering and `*_PROD.pdf` file names) is
obsolete.*
