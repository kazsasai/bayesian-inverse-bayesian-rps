#!/bin/bash
set -e

#cd ~/.../BIB_Analyze/simulation/reward_huge   # 適宜パスを修正

LOG=./data/reward_huge_v2_run.log
echo "Starting huge v2 run at $(date)" | tee -a "$LOG"

for design in "random sample rs" "random argmax ra" \
              "structured sample ss" "structured argmax sa"; do
  read -r init pred tag <<< "$design"
  out="./data/reward_huge_v2_${tag}"
  echo "" | tee -a "$LOG"
  echo "=== ${tag} (init=${init}, predict=${pred}) at $(date) ===" \
    | tee -a "$LOG"
  python rpsgame_reward.py grid \
    --scale huge \
    --init "$init" \
    --predict "$pred" \
    --output "$out" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "All four designs done at $(date)" | tee -a "$LOG"

