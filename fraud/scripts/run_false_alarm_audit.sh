#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p fraud/logs fraud/results
LOG="fraud/logs/run_false_alarm_audit.log"
exec > >(tee -a "$LOG") 2>&1

printf '=== [%s] 1/1: high-prob normal false-alarm audit (full 4.7M test, operational threshold) ===\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')"

python3 -u fraud/src/audit_false_alarms.py \
  --data big-test \
  --tn-n 2000 \
  --slices 12 \
  --pairs 3 \
  --cases 8 \
  --top-k 5 \
  --chunk 5000 \
  --out fraud/results/false_alarm_audit.json

printf '=== [%s] done -> fraud/results/false_alarm_audit.json ===\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')"
