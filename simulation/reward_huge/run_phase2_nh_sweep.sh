#!/bin/bash
#
# Phase 2: N_h sweep extended to all 4 designs (ra, ss, sa)
#
# Phase 1 (rs design only) confirmed BIB α invariance at N_h ∈ {6, 10, 15}.
# Phase 2 verifies that this invariance holds across all design combinations.
#
# Plan:
#   - Designs: ra, ss, sa (rs already done in Phase 1)
#   - N_h: {3, 6, 15, 20} (h=10 already done in v2 for all designs)
#   - Total: 3 × 4 = 12 huge runs × 24 conditions = 288 conditions
#
# Expected runtime: 12 × ~12 min = ~2.5 hours
# (more if a single design takes longer; can run overnight)
#
# Output: ./data/reward_huge_v3_{design}_h{N_h}/

set -e
cd "$(dirname "$0")"

LOG=./data/reward_huge_v3_phase2.log
mkdir -p ./data
echo "Phase 2 N_h sweep (extended) starting at $(date)" | tee -a "$LOG"
echo "  Designs: ra, ss, sa (rs is in Phase 1)" | tee -a "$LOG"
echo "  N_h: 3, 6, 15, 20 (h=10 is in v2)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Designs to sweep
declare -a DESIGNS=(
  "random argmax ra"
  "structured sample ss"
  "structured argmax sa"
)

for design in "${DESIGNS[@]}"; do
  read -r init pred tag <<< "$design"
  for h in 3 6 15 20; do
    out="./data/reward_huge_v3_${tag}_h${h}"
    echo "" | tee -a "$LOG"
    echo "=== ${tag} (init=${init}, predict=${pred}), N_h=${h} at $(date) ===" \
      | tee -a "$LOG"
    python rpsgame_reward.py grid \
      --scale huge \
      --init "$init" \
      --predict "$pred" \
      --h_num "$h" \
      --output "$out" 2>&1 | tee -a "$LOG"
  done
done

echo "" | tee -a "$LOG"
echo "Phase 2 done at $(date)" | tee -a "$LOG"
echo "Tar with: tar czf data_v3_phase2.tar.gz ./data/reward_huge_v3_{ra,ss,sa}_h*/" \
  | tee -a "$LOG"
