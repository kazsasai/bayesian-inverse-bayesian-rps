#!/bin/bash
set -e
cd "$(dirname "$0")"
LOG=run_post_nh.log
echo "=== waiting for nh sweep to finish ===" > $LOG
while pgrep -f "run_nh_sweep_hybrid.py" > /dev/null; do sleep 30; done
echo "[$(date)] sweep process exited" >> $LOG

CELLS=$(ls data/scheme_ablation/medium_nh_hybrid/durations_*.json 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date)] cell count: $CELLS/20" >> $LOG
if [ "$CELLS" -lt 20 ]; then
  echo "[$(date)] ABORT — only $CELLS/20 cells" >> $LOG
  exit 1
fi

cd files
echo "[$(date)] === analyzer ===" >> ../$LOG
python3 analyze_nh_sweep_hybrid.py >> ../$LOG 2>&1
echo "[$(date)] === DONE ===" >> ../$LOG
