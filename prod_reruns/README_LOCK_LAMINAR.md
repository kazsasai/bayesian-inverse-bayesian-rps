# Lock the laminar finite-size numbers at full production scale

Implements COWORK_FOLLOWUP_BRIEF (TASK A: z_lam; TASK B: alpha_lam Nh-invariance), by the
IDENTICAL pipeline used for the argmax z = 1.45 (rs design, truncated-power-law 1/Lambda,
Clauset-Shalizi-Newman x_min). Run on a machine that can do the full sweep.

## Requirements
- This repo checked out (engine `simulation_tie_mode_ablation/files/rpsgame_reward_tie.py` present).
- `pip install numpy powerlaw`

## Step 1 - generate the full-scale laminar Nh sweep (resumable, checkpointed)
4 designs (rs/ra/ss/sa) x Nh{3,6,10,15,20} x 20 runs, T=2e5, burn=1e5, theta=0.4, both agents pooled.
```
cd prod_reruns
CACHE_NAME=laminar_FULL_cache.json NHS=3,6,10,15,20 NR=20 T=200000 BURN=100000 \
    BUDGET=100000000 WORKERS=$(nproc) python run_laminar_nhsweep_prod.py
```
- One pass runs everything (BUDGET huge). It checkpoints after every batch to
  `laminar_FULL_cache.json`, so you can Ctrl-C and re-run the same line to resume.
- It prints `complete combos: 20/20` when done.
- Runtime ~ 400 runs x ~19 s / (cores). 4 cores ~ 30 min; 16 cores ~ 8 min.
- (Only need z_lam fast? add `DESIGNS=rs` -> 100 runs, ~8 min on 4 cores; but TASK B needs all 4.)

## Step 2 - analyse + lock
```
python analyze_laminar_lock.py            # uses laminar_FULL_cache.json
```
Prints, and writes `laminar_lock_results.json`:
- TASK A: z_lam (rs) +/- SE, R^2 over {3,6,10,15,20} and excl-boundary {3,6,10,15};
  plus the argmax-z consistency anchor computed the SAME way on the existing full-scale
  argmax durations (should reproduce ~1.45) and the published 1/Lambda series.
- TASK B: per-design alpha_lam(Nh); core-regime (Nh in {6,10}) pooled alpha_lam +/- SD (n=8);
  Nh-invariance verdict vs 1.34 +/- 0.05.
- The decision-rule branch for each task, and the auto-filled SI Sec 9 draft sentences.

## What to send back
`laminar_FULL_cache.json` + `laminar_lock_results.json` (or just paste the printed report).
I will then replace the provisional "z_lam ~ 2.2 to 2.3" in SI Sec 9 with the locked value
and add the alpha_lam-invariance sentence (+ optional Table S12 column), per the decision rules.

## Expected outcome (preview from the medium-scale run already done, 8 runs x T=4e4)
- z_lam(rs) ~ 2.30 +/- 0.10 (R^2=0.99); argmax anchor z ~ 1.51 (reproduces paper 1.45-1.49).
- core-regime pooled alpha_lam ~ 1.32 +/- 0.02 (n=8), Nh-invariant.
- => Decision A: z_lam clearly above argmax z -> keep conservative framing, lock value.
     Decision B: alpha_lam Nh-invariant -> add the dual-observable shared-tail sentence.
Full scale (T=2e5, 20 runs) should sharpen these; if z_lam instead comes out consistent with
1.45, or alpha_lam drifts, the script flags a STOP branch and I surface it before editing.

## Note on a leftover cache
A partial `laminar_FULL_cache.json` (only `rs|3`, made during a sandbox test) may be present.
For a clean local run either delete it first (`rm laminar_FULL_cache.json`) or just re-run
Step 1 as-is to resume (the engine is deterministic per seed, so resuming is equivalent).
