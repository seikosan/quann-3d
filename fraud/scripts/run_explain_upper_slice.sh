#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p fraud/logs fraud/results
LOG="fraud/logs/run_explain_upper_slice.log"
exec > >(tee -a "$LOG") 2>&1

printf '=== [%s] 1/1: full 4.7M test upper-slice explanation (50K attribution) ===\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')"

python3 -u fraud/src/explain_pilot.py \
  --data big-test \
  --n-explain 50000 \
  --top-tokens 50 \
  --cases 5 \
  --top-k 5 \
  --chunk 5000 \
  --out fraud/results/explain_upper_slice.json

printf '=== [%s] done -> fraud/results/explain_upper_slice.json ===\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')"
