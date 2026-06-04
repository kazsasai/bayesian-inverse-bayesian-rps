#!/usr/bin/env bash
# Fetch Brockbank & Vul raw RPS data for SI Fig S10 (figS_bib_behaviour, panel A).
# Source: https://github.com/erik-brockbank/rps  (data/v2 = 7 fixed/exploitable bots,
# data/v3 = 8 adaptive bots; 451 completed 300-round games). ~115 MB, so not vendored.
# After running this, `python make_figure_si.py` recomputes from raw and refreshes the cache.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HERE/_bv_rps_repo"
[ -d "$REPO/.git" ] || git clone --depth 1 https://github.com/erik-brockbank/rps.git "$REPO"
mkdir -p "$HERE/rps"
ln -sfn "$REPO/data" "$HERE/rps/data"
echo "OK: $HERE/rps/data -> $REPO/data  (v2/v3). Now run:  python make_figure_si.py"
