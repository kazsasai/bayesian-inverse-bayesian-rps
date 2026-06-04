# Build & repository notes (2026-05-21)

This is the **single source of truth** for Paper A. Earlier scattered copies
(`paper_A_BIB_Levy_v2_repo_v2/`, `BIB_Levy研究 (1)/`, loose `.zip`s, …) were
moved to `BIB_Analyze/_archive/` to remove the "which copy is canonical?"
ambiguity that was breaking builds.

## Canonical files

- Main text: `main_PRE_v2.tex`   (← compile this; `main_PRE.tex` is the older v1)
- Supplement: `supplement_v2.tex`
- Bibliography: `refs.bib`
- Figures: `figures/*.pdf`
- Figure build scripts: `figures/scripts/*.py`  (NEW — see below)

## How to compile

Figures live in `figures/`, a **child** of this `latex/` directory, and
`\graphicspath{{../figures/}{figures/}}` resolves both child- and sibling-style
layouts. Build the main text first (it produces `main_PRE_v2.aux`, which the
supplement reads via `\externaldocument{main_PRE_v2}`), then the supplement:

```bash
latexmk -pdf -pdflatex='pdflatex %O %S' main_PRE_v2.tex
latexmk -pdf -pdflatex='pdflatex %O %S' supplement_v2.tex
```

The explicit `-pdflatex='pdflatex %O %S'` override is required because the local
`~/.latexmkrc` rewrites pdflatex→lualatex, and lualatex + the new `array.sty`
trips the Table VIII/IX layout. Plain `pdflatex` is the supported engine.

## Figure regeneration (figures are reproducible now)

The three SI-origin figures used to be **raster** PDFs with panel letters baked
into the pixels, and their assembly scripts were lost — that is why they
"couldn't be fixed". Two have now been rebuilt as **vector** PDFs with editable
text and lower-case `(a)/(b)/(c)` panel labels:

| Fig | File | Script | Data source |
|---|---|---|---|
| Fig 9 | `FigS1_BO_tournament_summary.pdf` | `figures/scripts/build_FigS1_BO_tournament.py` | `scripts/data/bo_tournament_results.json` |
| Fig 7 | `FigS7_scheme_ablation.pdf` | `figures/scripts/build_FigS7_scheme_ablation.py` | `scripts/data/scheme_summary.csv` |
| Fig 10 | `FigS6_smoothing_comparison.pdf` | `figures/scripts/build_FigS6_smoothing_comparison.py` | `scripts/data/sigma_{conditional,always}_rs_bib-bib.json` |

Each script first looks for the live data under the sibling `simulation*/` trees
and falls back to the bundled copies in `figures/scripts/data/`, so they run
even from a fresh checkout. Regenerate with:

```bash
cd figures/scripts && python3 build_FigS1_BO_tournament.py && python3 build_FigS7_scheme_ablation.py
```

## Fig 10 note (rebuilt as 2-panel, 2026-05-21)

Fig 10 was originally four panels, but only two carry surviving per-step
`sig1_downsampled` trajectories in the workspace (conditional from
`simulation_r2`, always-on JM 0.10 from `simulation_r3_JM0.10`). Per
`simulation_r4_fallback/files/CHANGES.md` the two missing conditional variants
are statistically indistinguishable from the conditional case, so the figure was
rebuilt as the essential **conditional vs always-on** contrast (sig.bar 0.20 vs
0.015, ~13x collapse). The Appendix-A caption (`fig:smoothing`) and the
four-implementation tables (`tab:smoothing-impl`, `tab:s6`) were kept; the
caption now states a single conditional representative is shown. To restore the
full four panels, re-run the two missing variants with the `r4_fallback` engine
(rs x bib-bib, T=2000, 20 runs, recording `sig1_downsampled`).

## Still outstanding

- Submission placeholders (unchanged): GitHub URL, Zenodo/figshare DOI, KAKENHI
  number (26H01196 to confirm), Gunji ORCID/email, Shinohara 2021 quote check,
  reviewer emails, cover-letter reframing.
