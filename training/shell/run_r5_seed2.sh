#!/bin/bash
# R5 seed-2 tekrarı: 0.08'lik mekanizma farkının (memory-stablesz − ablation, tek seed)
# sağlamlığı. NOT: R5 orijinalleri tohumlanmamıştı (manual_seed yok); seed=43 koşuları
# "aynı seed'in tekrarı" değil, BAĞIMSIZ ikinci örneklerdir. Bölme seed=42 sabit (§4.1).
# Checkpoint adları *_s43 — orijinal kanıt checkpoint'leri üzerine yazılmaz.
cd "$(dirname "$0")/../.."  # repo root
set -o pipefail

LOG=logs/run_r5_seed2.log

echo "=== [$(date)] 1/4: pin code state ===" | tee -a "$LOG"
git rev-parse HEAD | tee -a "$LOG"
shasum -a 256 contextcloud/train_stable.py contextcloud/train_memory_context.py contextcloud/ablate_stable_on_memorytask.py contextcloud/task_utils.py | tee -a "$LOG"
df -h /System/Volumes/Data | tail -1 | tee -a "$LOG"

echo "=== [$(date)] 2/4: ablation (stable mek. / memory görevi) seed=43 ===" | tee -a "$LOG"
python3 -u contextcloud/ablate_stable_on_memorytask.py --seed 43 --ckpt-name ablate_stable_on_memorytask_s43_best.pt 2>&1 | tee -a logs/run_r5_seed2_ablate_s43.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] ablation s43 FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 3/4: memory stablesz (d128/d256, v10k) seed=43 ===" | tee -a "$LOG"
python3 -u contextcloud/train_memory_context.py --vocab-size 10000 --d-model 128 --d-sentence 256 --ckpt-name context_memory_stablesz_v10k_s43_best.pt --seed 43 2>&1 | tee -a logs/run_r5_seed2_memory_stablesz_s43.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] memory stablesz s43 FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 4/4: stable seed=43 ===" | tee -a "$LOG"
python3 -u contextcloud/train_stable.py --seed 43 --ckpt-name context_stable_s43_best.pt 2>&1 | tee -a logs/run_r5_seed2_cc_stable_s43.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] stable s43 FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] R5 SEED2 QUEUE COMPLETE ===" | tee -a "$LOG"
