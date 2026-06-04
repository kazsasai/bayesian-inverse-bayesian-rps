#!/bin/bash
#
# run_nhand_huge.sh — Run huge-scale N-hand RPS for Paper A §3.5 (R4)
#
# Purpose:
#   Establish whether BIB universality (α ≈ 1.43) extends from N=3 to
#   higher cyclic-dominance dimensions N=5, 7. This produces the data
#   needed for Figure 5 in the paper.
#
# What it runs:
#   - N=3 huge (rs design): for direct comparison with Phase 1/2 main data
#                            (skip if you already have data_v2_rs)
#   - N=5 huge (rs design): NEW
#   - N=7 huge (rs design): NEW
#   Default design is rs (random init + sample predict), the canonical
#   reference design used throughout the paper.
#
# Wall-clock estimate (Apple M1 / 13 parallel workers):
#   - Each huge condition: ~10–20 sec/process
#   - Per N value, full grid (4 designs × 6 pairs × 4 windows = 96 conditions):
#     ~30–60 minutes
#   - Single design (rs, 24 conditions): ~10–15 minutes per N value
#   - Total for N=5,7 at rs only: ~20–30 minutes
#
# Data size estimate (per N value, rs design only):
#   - durations_*.json + sigmas_*.json + summary CSV: ~150–250 MB
#
# Output structure:
#   ./data/nhand_grid_N5_rs/
#       summary_huge.csv
#       durations_*_m{10,20,50,100}_huge.json
#       sigmas_*_m{10,20,50,100}_huge.json
#       rewards_*_m{10,20,50,100}_huge.json
#   ./data/nhand_grid_N7_rs/
#       (same structure)
#
# Usage:
#   bash run_nhand_huge.sh           # runs N=5 then N=7 in rs design
#   bash run_nhand_huge.sh 5         # only N=5
#   bash run_nhand_huge.sh 7         # only N=7
#   WORKERS=8 bash run_nhand_huge.sh # override worker count
#

set -e

WORKERS="${WORKERS:-13}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NHAND_PY="$SCRIPT_DIR/rpsgame_nhand.py"

if [ ! -f "$NHAND_PY" ]; then
    echo "ERROR: rpsgame_nhand.py not found in $SCRIPT_DIR"
    echo "Place this script in the same directory as rpsgame_nhand.py"
    exit 1
fi

# Determine which N values to run
if [ -n "$1" ]; then
    N_VALUES=("$1")
else
    N_VALUES=(5 7)
fi

echo "=========================================="
echo "N-hand RPS huge data acquisition"
echo "=========================================="
echo "  N values: ${N_VALUES[*]}"
echo "  Design:    sa (structured init + argmax predict)"
echo "  Scale:     huge (T=200,000 × 20 runs)"
echo "  Workers:   $WORKERS"
echo "  Output:    ./data/nhand_grid_N{N}_rs/"
echo "=========================================="
echo ""

# Main runs
for N in "${N_VALUES[@]}"; do
    OUTDIR="./data/nhand_grid_N${N}_sa"
    echo ""
    echo "=========================================="
    echo "[huge N=$N] $OUTDIR"
    echo "=========================================="
    START=$(date +%s)

    python3 "$NHAND_PY" --workers "$WORKERS" grid \
        --N "$N" \
        --scale huge \
        --init structured \
        --predict argmax \
        --output "$OUTDIR"

    END=$(date +%s)
    ELAPSED=$((END - START))
    echo ""
    echo "[done N=$N] ${ELAPSED}s elapsed"
    if [ -f "$OUTDIR/summary_huge.csv" ]; then
        ROWS=$(wc -l < "$OUTDIR/summary_huge.csv")
        echo "[done N=$N] summary_huge.csv has $ROWS rows"
    fi
done

echo ""
echo "=========================================="
echo "All N-hand huge runs complete."
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Verify each ./data/nhand_grid_N{N}_rs/summary_huge.csv has 25 rows"
echo "     (1 header + 24 conditions)"
echo "  2. Compress for upload:"
echo ""
echo "     tar -czf nhand_huge.tar.gz \\"
echo "         data/nhand_grid_N5_rs/ \\"
echo "         data/nhand_grid_N7_rs/"
echo ""
echo "  3. If size > 200MB, split for chat upload:"
echo "     split -b 200m nhand_huge.tar.gz nhand_huge.tar.gz."
echo ""
