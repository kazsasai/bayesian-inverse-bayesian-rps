#!/bin/bash
set -e
cd "$(dirname "$0")"
LOG=run_post.log
echo "=== waiting for sim to finish ===" > $LOG
while pgrep -f "run_scheme_ablation.py" > /dev/null; do sleep 30; done
echo "[$(date)] sim process exited" >> $LOG

# Verify cells
CELLS=$(ls data/scheme_ablation/huge/durations_*.json 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date)] cell count: $CELLS/12" >> $LOG
if [ "$CELLS" -lt 12 ]; then
  echo "[$(date)] ABORT — only $CELLS cells; check run_huge.log" >> $LOG
  exit 1
fi

cd files
echo "[$(date)] === fit phase ===" >> ../$LOG
# Run prefit until cache is full; each invocation does 4 fits
for i in $(seq 1 20); do
  OUT=$(python3 prefit_scheme.py --scale huge --n 4 2>&1)
  echo "[$(date)] prefit iter $i: $(echo "$OUT" | tail -3)" >> ../$LOG
  # Stop early if no more work
  if echo "$OUT" | grep -qiE "nothing|done|0 fits|no.*pending|complete"; then
    echo "[$(date)] fit cache full at iter $i" >> ../$LOG
    break
  fi
done

echo "[$(date)] === plot phase ===" >> ../$LOG
python3 make_scheme_plots.py --scale huge >> ../$LOG 2>&1
echo "[$(date)] === DONE ===" >> ../$LOG
ls -la ../data/scheme_ablation/huge/fig_*.png ../data/scheme_ablation/huge/scheme_summary.csv >> ../$LOG 2>&1
