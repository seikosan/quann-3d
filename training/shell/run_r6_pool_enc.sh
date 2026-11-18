#!/bin/bash
# R6 katkı deneyi: memory-stablesz modelinde ölçülmemiş iki katkının ayrımı
# (RESEARCH.md "Sıradaki Aksiyonlar"):
#   1) padding'li havuzlama  → --masked-pool     (train_memory_context.py forward)
#   2) paylaşımlı encoder_layer → --unshared-encoder (aynı dosya, encoder_stack)
# Kontrol: seed=43 memory-stablesz = TEST 3.6336 (logs/run_r5_seed2_memory_stablesz_s43.log:631)
# — yeniden koşulmaz, mevcut örnek kullanılır. Bayraklar kapalıyken davranış R5 ile birebir.
# Bölme seed=42 sabit (§4.1); ckpt adları varyant-spesifik, mevcut kanıtlara yazılmaz.
cd "$(dirname "$0")/../.."  # repo root
set -o pipefail

LOG=logs/run_r6_pool_enc.log
MEM_ARGS="--vocab-size 10000 --d-model 128 --d-sentence 256 --seed 43"

echo "=== [$(date)] 1/4: pin code state ===" | tee -a "$LOG"
git rev-parse HEAD | tee -a "$LOG"
shasum -a 256 contextcloud/train_memory_context.py contextcloud/task_utils.py | tee -a "$LOG"
df -h /System/Volumes/Data | tail -1 | tee -a "$LOG"

echo "=== [$(date)] 2/4: masked-pool only seed=43 ===" | tee -a "$LOG"
python3 -u contextcloud/train_memory_context.py $MEM_ARGS --masked-pool \
  --ckpt-name context_memory_stablesz_maskpool_s43_best.pt 2>&1 | tee -a logs/run_r6_maskpool_s43.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] maskpool s43 FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 3/4: unshared-encoder only seed=43 ===" | tee -a "$LOG"
python3 -u contextcloud/train_memory_context.py $MEM_ARGS --unshared-encoder \
  --ckpt-name context_memory_stablesz_unshared_s43_best.pt 2>&1 | tee -a logs/run_r6_unshared_s43.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] unshared s43 FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 4/4: masked-pool + unshared-encoder seed=43 ===" | tee -a "$LOG"
python3 -u contextcloud/train_memory_context.py $MEM_ARGS --masked-pool --unshared-encoder \
  --ckpt-name context_memory_stablesz_both_s43_best.pt 2>&1 | tee -a logs/run_r6_both_s43.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] both s43 FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] R6 POOL/ENC QUEUE COMPLETE ===" | tee -a "$LOG"
