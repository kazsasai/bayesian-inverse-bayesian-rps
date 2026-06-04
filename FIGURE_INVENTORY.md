# Paper A: Figure / Table Inventory

**重要**: ファイル名と論文中の番号がずれている箇所があります。

---

## Main paper figures (main_PRE.tex)

| 論文番号 | ファイル名 | ラベル | 配置サブセクション | 内容 |
|---:|---|---|---|---|
| **Fig 1** | `Fig1_SOC_mechanism.pdf` | `fig:soc-schematic` | Introduction (§I) | **SOC schematic**: Bayesian update (driving, 左) + Inverse Bayesian replacement (relaxation, 右) → universal power-law persistence-time. 概念図、テキスト・矢印のみ |
| **Fig 2** | `Fig2_dynamics_demo_PROD.pdf` | `fig:dynamics` | Results 冒頭 (§III.1) | **Hypothesis-space dynamics demo**: BIB-BIB (上段) vs BO-BO (下段)、同じ seed・rs design・N_h=10 で 1 ラン分の posterior trajectory + argmax track + Sigma(P(h)) を 3列で表示 |
| **Fig 3** | `Fig3_universality_2panel_H_PROD.pdf` | `fig:univ` | BIB universality (§III.1) | **Universality contrast (2-panel H)**: (a) Argmax persistence T_argmax CCDF, (b) Laminar phase length L_lam CCDF。各panelに 8 CCDFs (4設計 × BIB/BO)。BIB 全部 collapse、BO 分散 |
| **Fig 4** | `Fig4_robustness_sweep_PROD.pdf` | `fig:robust` | Robustness sweeps (§III.3) | **Robustness 2-panel H**: (a, 左) window-size m sweep CCDF、(b, 右) α(N) for N∈{3,5,7} |
| **Fig 5** | `Fig6_reward_2panel_H_PROD.pdf` ⚠️ | `fig:reward` | Reward analysis (§III.4) | **Reward analysis 2-panel H**: (a, 左) BIB net advantage per step by design (棒+z-score)、(b, 右) cumulative reward post-burn-in。**rs (BIB-favoured z=+4.70) → sa (BO-favoured z=−4.44) の設計依存性** |
| **Fig 6** | `Fig8_Nh_sweep_SOC_test_PROD.pdf` ⚠️ | `fig:nhswp` | Additional scaling tests (§III.6) | **N_h sweep SOC test, 4-panel (2×2)**: (a) α(N_h) BIB、(b) α(N_h) BO、(c) σ(P(h)) vs N_h log-log BIB、(d) σ(P(h)) BO。Core regime N_h∈{6,10} shaded |

⚠️ = ファイル名と論文番号がずれている（古いファイル名のまま）

---

## Main paper tables

| 論文番号 | ラベル | 配置 | 内容 |
|---:|---|---|---|
| **Table 1** | `tab:sweep` | §II.4 Agent types | Six-axis experimental sweep table (parameters × ranges × where measured) |
| **Table 2** | `tab:alphaN` | §III.3 Robustness | Per-N statistics for BIB-BIB α_TPL at m=50 (N=3,5,7) |
| **Table 3** | `tab:reward` | §III.4 Reward analysis | BIB win rate against BO (per design, z-score, p-value) |
| **Table 4** | `tab:core` | §III.5 N_h-invariance | BIB-BIB α in core central regime (per design × N_h=6,10) |

---

## Supplement figures (supplement.tex)

| 論文番号 | ファイル名 | ラベル | 配置 | 内容 |
|---:|---|---|---|---|
| **Fig S1** | `FigS0a_duality.pdf` | `fig:s0a` | §S0 Concept figures | T_argmax と τ_replace のduality (concept) |
| **Fig S2** | `FigS0b_simplex.pdf` | `fig:s0b` | §S0 Concept figures | Boundary-driven simplex structure (N=3, 5) |
| **Fig S3** | `FigS0c_lineages.pdf` | `fig:s0c` | §S0 Concept figures | Two BIB lineages (argmin vs argmax replacement) |
| **Fig S4** | `FigS1_BO_tournament_summary.pdf` | `fig:s1` | §S1 BO tournament | 5×5 BO tournament Nash-baseline z-scores |
| **Fig S5** | `FigS6_smoothing_comparison.pdf` | `fig:s6` | §S6 Smoothing | σ(P(h)) dynamics under 4 posterior-smoothing rules |
| **Fig S6** | `FigS7_scheme_ablation.pdf` | `fig:scheme-ablation` | §S7 Observation-rule ablation | 4-scheme ablation bar chart (Task 3 で追加) |

---

## Supplement tables

| 論文番号 | ラベル | 配置 | 内容 |
|---:|---|---|---|
| **Table S1** | `tab:nash-check` | §S1 Sanity check | Nash-equilibrium sanity check (BO vs Nash random) |
| **Table S2** | `tab:bo-internal` | §S1 BO internal | BO-internal tournament ranking (ss > rs > sa > ra) |
| **Table S3** | `tab:alphaN-perdesign` | §S5 α(N) per design | Per-design α_TPL at N=3,5,7 |
| **Table S4** | `tab:smoothing-impl` | §S6 Smoothing | Four smoothing implementation rules |
| **Table S5** | `tab:s6` | §S6 Smoothing results | Smoothing-implementation comparison |

---

## まとめ

**Main paper**: Fig 1〜6 (6 figures), Table 1〜4 (4 tables)
**Supplement**: Fig S1〜S6 (6 figures), Table S1〜S5 (5 tables)

**ファイル名と論文番号のズレ**:
- Fig 5 (paper) = `Fig6_reward_*` (filename)
- Fig 6 (paper) = `Fig8_Nh_sweep_*` (filename)

理由: 過去に Fig 5 = window_dependence と Fig 6 = alpha_N を Fig 4 (robustness_sweep) に統合した際、後続のファイル名 (Fig6_reward, Fig8_Nh_sweep) はそのまま残した。次回 figure 再生成時にファイル名も整合させると良い。
