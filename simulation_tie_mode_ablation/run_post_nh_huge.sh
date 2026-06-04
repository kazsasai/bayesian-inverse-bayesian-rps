#!/bin/bash
set -e
cd "$(dirname "$0")"
LOG=run_post_nh_huge.log
echo "=== waiting for huge nh sweep to finish ===" > $LOG
while pgrep -f "run_nh_sweep_hybrid.py" > /dev/null; do sleep 60; done
echo "[$(date)] sweep process exited" >> $LOG

CELLS=$(ls data/scheme_ablation/huge_nh_hybrid/durations_*.json 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date)] cell count: $CELLS/20" >> $LOG
if [ "$CELLS" -lt 20 ]; then
  echo "[$(date)] ABORT — only $CELLS/20 cells" >> $LOG
  exit 1
fi

cd files
echo "[$(date)] === analyzer (huge) ===" >> ../$LOG
python3 analyze_nh_sweep_hybrid.py --scale huge >> ../$LOG 2>&1
echo "[$(date)] === DONE ===" >> ../$LOG
