# N-hand Huge Data Acquisition — README

For Paper A §3.5 (R4): testing whether BIB universality (α ≈ 1.43)
extends from N=3 to higher cyclic-dominance dimensions N ∈ {5, 7}.

## Files

| File | Description |
|---|---|
| `rpsgame_nhand.py` | N-hand cyclic-dominance RPS simulator (standalone) |
| `run_nhand_huge.sh` | Driver script that runs both N=5 and N=7 huge |

The Python file is self-contained — no dependency on `rpsgame_reward.py`.

## Setup

- Python 3.9+
- Required packages: `numpy`, `pandas`, `powerlaw`, `matplotlib`
  - Install: `pip install numpy pandas powerlaw matplotlib`
- Place both files in the same directory.

## Quick verification (~1 minute)

Before running the full huge sweep, verify the rule and code work:

```bash
# Pilot (T=2,000 × 30 runs, fast)
python3 rpsgame_nhand.py pilot --N 3 --output ./data/test_N3
python3 rpsgame_nhand.py pilot --N 5 --output ./data/test_N5
python3 rpsgame_nhand.py pilot --N 7 --output ./data/test_N7
```

Expected output: `./data/test_N{3,5,7}/summary_pilot.csv` with row counts.
Each pilot takes ~10–30 seconds.

## Main run

```bash
# Run both N=5 and N=7 huge in rs design
bash run_nhand_huge.sh

# Or one at a time
bash run_nhand_huge.sh 5
bash run_nhand_huge.sh 7
```

### Estimates (Apple M1, 13 parallel workers)

| Step | Wall-clock | Output size |
|---|---|---|
| Pilot N=3 | ~10 s | ~1 MB |
| Pilot N=5 | ~20 s | ~1 MB |
| Pilot N=7 | ~30 s | ~1 MB |
| Huge N=5 (rs design, 24 conditions) | ~10–15 min | ~150 MB |
| Huge N=7 (rs design, 24 conditions) | ~15–20 min | ~200 MB |

Adjust workers if needed:
```bash
WORKERS=8 bash run_nhand_huge.sh
```

## Output structure

```
./data/
  nhand_pilot_N{3,5,7}/             # short verification runs
  nhand_grid_N5_rs/                 # huge data N=5 rs design
    summary_huge.csv                # 25 rows: 1 header + 24 conditions
    durations_<pair>_m<m>_huge.json # per-condition run-length data
    sigmas_<pair>_m<m>_huge.json    # σ(P(h)) time series (downsampled)
    rewards_<pair>_m<m>_huge.json   # win/loss/cumR per step (downsampled)
  nhand_grid_N7_rs/                 # huge data N=7 rs design
    (same structure)
```

## Compression for upload

Once both N=5 and N=7 huge runs complete:

```bash
# Compress
tar -czf nhand_huge.tar.gz \
    data/nhand_grid_N5_rs/ \
    data/nhand_grid_N7_rs/

# Check size
ls -lh nhand_huge.tar.gz

# If > 200MB, split for chat upload
split -b 200m nhand_huge.tar.gz nhand_huge.tar.gz.
ls -lh nhand_huge.tar.gz.*
```

## What the data tests (R4)

The Phase 1/2 data established BIB universality at N=3. The N-hand huge
data tests whether the same α ≈ 1.43 universality holds at N=5
("Rock, Paper, Scissors, Lizard, Spock" structure, k=2) and N=7
(k=3). If it does, this confirms that the SOC mechanism is not specific
to N=3 but generalizes across cyclic-dominance dimensions, supporting
the boundary-driven simplex framing in §4.3.

Pilot data already shows BIB exponents in the Lévy regime
(1 < α ≤ 3) for all three N values; the huge data is needed to
quantify α(N) and produce Figure 5 of the paper.

## Sanity checks after run

```bash
# Check each summary CSV has the expected 25 rows
wc -l data/nhand_grid_N5_rs/summary_huge.csv  # → expect 25
wc -l data/nhand_grid_N7_rs/summary_huge.csv  # → expect 25

# Quick check of α values for bib-bib at m=50 (should be ~1.43)
python3 -c "
import pandas as pd
for N in [5, 7]:
    df = pd.read_csv(f'data/nhand_grid_N{N}_rs/summary_huge.csv')
    sub = df[(df.window == 50) & (df.a1 == 'bib') & (df.a2 == 'bib')]
    print(f'N={N}: bib-bib α at m=50 = {sub.T_argmax1_alpha.values}')
"
```

## Troubleshooting

- **`ModuleNotFoundError: powerlaw`**: `pip install powerlaw`
- **Slow on small CPU**: reduce workers with `WORKERS=4`
- **Disk full**: each huge run produces ~150–200MB; clean up old data
  before running

## Next step after data acquisition

Upload the `nhand_huge.tar.gz` (or split parts) to the chat. I will:
1. Extract α(N) values across N ∈ {3, 5, 7}
2. Generate Figure 5 (α(N) plot)
3. Update Paper A §3.5 with final R4 wording
4. Confirm whether α ≈ 1.43 holds across N or shows a systematic shift
