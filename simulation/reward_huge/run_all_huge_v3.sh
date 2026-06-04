#!/bin/bash
set -e

#cd ~/.../BIB_Analyze/simulation/reward_huge   # 適宜パスを修正

# rsのみで N_h sweep × 5 値（合計 5 conditions × 24 windows-pairs = 120 conditions）
for h in 3 6 15 20; do  # h=10 は既に取得済み
  python rpsgame_reward.py grid \
    --scale huge \
    --init random \
    --predict sample \
    --h_num $h \
    --output ./data/reward_huge_v3_rs_h${h}
done
