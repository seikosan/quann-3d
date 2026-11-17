#!/bin/bash
# R4 resume kuyruğu: D64'ü epoch 15 best'inden devam ettir, sonra duel eval + contextcloud.
# Orijinal run_r4_full.sh disk dolusu nedeniyle 4/7'de öldü (logs/run_r4_full.log traceback).
cd "$(dirname "$0")/../.."  # repo root
set -o pipefail

LOG=logs/run_r4_resume.log
SAVE64=checkpoints/wiki_checkpoints_r4_d64_200k
CORPUS=data/turkish_wiki_1500k.txt

echo "=== [$(date)] 1/5: pin code state ===" | tee -a "$LOG"
git rev-parse HEAD | tee -a "$LOG"
shasum -a 256 training/train_wiki.py training/eval_r4_duel.py cloudlm/model.py cloudlm/tokenizer.py | tee -a "$LOG"
df -h /System/Volumes/Data | tail -1 | tee -a "$LOG"

echo "=== [$(date)] 2/5: resume D=64 from epoch 15 best ===" | tee -a "$LOG"
python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 200000 \
  --d-dim 64 --save-dir "$SAVE64/" --epochs 50 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --layers 4 \
  --resume "$SAVE64/wiki_cloud_best.pt" 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] D=64 resume FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 3/5: R4 duel evaluation ===" | tee -a "$LOG"
python3 -u training/eval_r4_duel.py 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] R4 duel eval FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 4/5: contextcloud stable (sızdırmasız görev) ===" | tee -a "$LOG"
mkdir -p contextcloud/checkpoints/pre_r4_leaky
for f in contextcloud/checkpoints/*.pt; do
  [ -e "$f" ] && mv -n "$f" contextcloud/checkpoints/pre_r4_leaky/
done
python3 -u contextcloud/train_stable.py 2>&1 | tee -a logs/run_r4_contextcloud_stable.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] CC stable FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 5/5: contextcloud memory + global ===" | tee -a "$LOG"
python3 -u contextcloud/train_memory_context.py 2>&1 | tee -a logs/run_r4_contextcloud_memory.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] CC memory FAILED ===" | tee -a "$LOG"
  exit 1
fi
python3 -u contextcloud/train_global_context.py 2>&1 | tee -a logs/run_r4_contextcloud_global.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] CC global FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] R4 RESUME QUEUE COMPLETE ===" | tee -a "$LOG"
