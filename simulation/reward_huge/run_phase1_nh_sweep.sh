#!/bin/bash
#
# Phase 1: N_h sweep at representative condition (design = rs)
#
# This runs N_h ∈ {3, 6, 15, 20} at huge scale with the rs design.
# N_h=10 is already in reward_huge_v2_rs/ from the previous run.
#
# Expected runtime: 4 designs × ~10-15 min/design = ~1 hour
# (assuming similar throughput to previous huge run)
#
# Output: ./data/reward_huge_v3_rs_h{3,6,15,20}/
#   - durations_*.json (24 conditions per N_h)
#   - rewards_*.json
#   - sigmas_*.json   ← NEW
#   - summary_huge.csv (with new sig1_mean, sig2_mean, h_num columns)

set -e
cd "$(dirname "$0")"  # adjust if needed

LOG=./data/reward_huge_v3_phase1.log
mkdir -p ./data
echo "Phase 1 N_h sweep starting at $(date)" | tee -a "$LOG"
echo "  Design: rs (random init + sample predict)" | tee -a "$LOG"
echo "  N_h values: 3, 6, 15, 20 (10 already done in v2)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for h in 3 6 15 20; do
  out="./data/reward_huge_v3_rs_h${h}"
  echo "" | tee -a "$LOG"
  echo "=== N_h=$h at $(date) ===" | tee -a "$LOG"
  python rpsgame_reward.py grid \
    --scale huge \
    --init random \
    --predict sample \
    --h_num "$h" \
    --output "$out" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "Phase 1 done at $(date)" | tee -a "$LOG"
echo "Next: tar czf data_v3_phase1.tar.gz ./data/reward_huge_v3_rs_h*/" \
  | tee -a "$LOG"
