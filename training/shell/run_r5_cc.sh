#!/bin/bash
# R5 CC kuyruğu: ayrık 80/10/10 holdout (seed=42) ile stable → memory → global yeniden eğitim.
# Eski holdout'suz checkpoint'ler contextcloud/checkpoints/pre_r5_notholdout/ altında arşivlendi.
cd "$(dirname "$0")/../.."  # repo root
set -o pipefail

LOG=logs/run_r5_cc.log

echo "=== [$(date)] 1/4: pin code state ===" | tee -a "$LOG"
git rev-parse HEAD | tee -a "$LOG"
shasum -a 256 contextcloud/task_utils.py contextcloud/train_stable.py contextcloud/train_memory_context.py contextcloud/train_global_context.py | tee -a "$LOG"
df -h /System/Volumes/Data | tail -1 | tee -a "$LOG"

echo "=== [$(date)] 2/4: CC stable (ayrık holdout) ===" | tee -a "$LOG"
python3 -u contextcloud/train_stable.py 2>&1 | tee -a logs/run_r5_cc_stable.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] CC stable FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 3/4: CC memory (ayrık holdout) ===" | tee -a "$LOG"
python3 -u contextcloud/train_memory_context.py 2>&1 | tee -a logs/run_r5_cc_memory.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] CC memory FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 4/4: CC global (ayrık holdout) ===" | tee -a "$LOG"
python3 -u contextcloud/train_global_context.py 2>&1 | tee -a logs/run_r5_cc_global.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] CC global FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] R5 CC QUEUE COMPLETE ===" | tee -a "$LOG"
