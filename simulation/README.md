# `simulation/` — engine map and provenance

This directory holds the simulation **engines** (the inference rules) and their
run scripts. Large outputs are *not* stored here (`.gitignore` excludes every
`simulation/*/data/`); they are archived on Zenodo
(DOI `10.5281/zenodo.20533918`). For which **data tier** feeds which **figure**,
see [`../BIB_Levy_v2/latex/figures/scripts/zenodo_data_manifest.txt`](../BIB_Levy_v2/latex/figures/scripts/zenodo_data_manifest.txt).

## Why several files share the name `rpsgame_reward.py`

Each experiment folder is **self-contained**: it bundles the exact engine
snapshot it was run with, so that `cd <folder> && python rpsgame_reward.py grid …`
reproduces that folder's data with no path juggling, and the analysis scripts in
the folder can simply `from rpsgame_reward import …`. As a result the same file
name appears in several folders. They are **not all identical** — there are three
generations plus one exact copy:

| File | Lines | md5 (head) | What it is |
|---|---:|---|---|
| `reward_huge/rpsgame_reward.py` | 1019 | `fd4b6e` | **Production engine, gen-1.** Adds `likelihood_spread()` (posterior-spread σ → β exponent) and `streak_lengths()`; emits `durations_*` and `sigmas_*.json`. Ships the production sweep runners (`run_phase1/2_nh_sweep.sh`, `run_all_huge_v2/v3.sh`). |
| `analyze_sharpness_plateau/rpsgame_reward.py` | 1019 | `fd4b6e` | **Byte-identical copy of the gen-1 engine** (same md5), bundled so `run_sharpness_plateau.py` and `run_bo_tournament.py` can `import rpsgame_reward` standalone. *Not a different version.* |
| `reward_huge_v2/rpsgame_reward.py` | 935 | `bb74f3` | **Production engine, gen-2.** Same inference model; `likelihood_spread()` removed (σ not recorded). Produces the reward-tournament tier `reward_huge_v2_<design>` (`rewards_bib-bo_*`). |
| `_deprecated/reward/rpsgame_reward.py` | 661 | `646723` | **Earliest pilot** (faithful translation of Ibuka & Sasai 2024). No `likelihood_spread`/`streak_lengths`. **Superseded by `reward_huge`; not used for any paper figure or Zenodo dataset.** Kept for lineage under [`_deprecated/`](_deprecated/). |

Reproduce the table yourself:

```bash
find simulation -name rpsgame_reward.py | xargs md5   # macOS;  md5sum on Linux
```

## The distinct engines (different names = different models)

| Engine | Role |
|---|---|
| `rpsgame_levy.py` | Original **Gunji-type, observation-driven** BIB (1v1 discrete RPS / imitation) — the Lévy-walk verification the project started from. |
| `reward_huge/rpsgame_reward.py` | **Reward-based** BIB (observation = win/lose reward, not the opponent hand). The main reward-analysis engine — see the table above for the v1/v2 split. |
| `nhand/rpsgame_nhand.py` | **N-hand cyclic-dominance** generalisation (N = 3, 5, 7, 9, …) used for the N-hand RPS results. See [`nhand/README_nhand_huge.md`](nhand/README_nhand_huge.md). |
| `nonstationary/rpsgame_nonstationary.py` | Non-stationary (cycle-reversal) reward-based BIB variant. |
| `../simulation_tie_mode_ablation/files/rpsgame_reward_tie.py` | Tie-mode ablation variant; extends the reward engine and is self-labelled in its header. |

## Provenance note (σ / β data)

The posterior-spread series `sigmas_*.json` are emitted by the **gen-1** engine
(`reward_huge`, via `likelihood_spread()`); the **gen-2** engine records rewards
only. The N_h-sweep-with-σ tier (`reward_huge_v3_*` in the Zenodo manifest) was
produced by the gen-1-class engine. If you re-derive the β finite-size scaling,
use the gen-1 engine (or read the cached `sigmas_*.json`).
