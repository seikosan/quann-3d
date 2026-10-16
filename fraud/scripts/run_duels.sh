#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p fraud/logs
python3 -u fraud/src/train_duels.py --models xgboost,lightgbm,ftt 2>&1 \
  | tee -a fraud/logs/run_duels.log
