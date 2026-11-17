#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p fraud/logs fraud/results/deploy
LOG="fraud/logs/run_export_deploy.log"
exec > >(tee -a "$LOG") 2>&1

printf '=== [%s] 1/1: deploy package export (ONNX + quantizer + token decoder) ===\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')"

python3 -u fraud/src/export_deploy.py \
  --out-dir fraud/results/deploy \
  --n-cases 3

printf '=== [%s] done -> fraud/results/deploy/ ===\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')"
