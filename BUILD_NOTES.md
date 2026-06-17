# Build & reproduction notes

This repository is **code and small cached summaries only** — the manuscript and
its LaTeX template are kept elsewhere and are not part of this repository, so
there is nothing here to `pdflatex`. What this document covers is how the
**figures and numbers** regenerate.

## Where to start

- [`README.md`](README.md) — environment setup, data download (Zenodo
  `10.5281/zenodo.20533918`), and the figure-build commands.
- [`FIGURE_INVENTORY.md`](FIGURE_INVENTORY.md) — every figure ↔ `\label` ↔
  output file ↔ build script, in manuscript order.
- [`simulation/README.md`](simulation/README.md) — the simulation **engine** map
  (what each `rpsgame_*.py` is, and why several folders share the name
  `rpsgame_reward.py`).

## Data resolution

Every data-driven figure script resolves its inputs through
[`BIB_Levy_v2/latex/figures/scripts/figdata.py`](BIB_Levy_v2/latex/figures/scripts/figdata.py),
which searches in order:

1. `$PAPERA_DATA` (the extracted Zenodo archive),
2. `<repo>/data/`,
3. the in-repo `simulation*/` tree (bundled caches under `scripts/data/`).

So a fresh checkout that includes the `simulation/` tree rebuilds the
schematic- and cache-backed figures with no extra download; the large
duration/σ figures additionally need the Zenodo archive.

## Notes

- Plain `pdflatex` is the supported engine for the (out-of-repo) manuscript; if a
  local `~/.latexmkrc` rewrites `pdflatex`→`lualatex`, override it with
  `latexmk -pdf -pdflatex='pdflatex %O %S'`.
- The earlier PRE-version build notes (referring to `main_PRE_v2.tex`,
  `simulation_r2/`–`r4_fallback/`, and a since-renamed BO-tournament script) are
  obsolete and have been removed.
