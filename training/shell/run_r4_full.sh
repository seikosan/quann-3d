#!/bin/bash
# R4 tam yeniden eğitim kuyruğu: güncel kodla D32/D64 duel + sızıntısız contextcloud.
# Protokol R1 ile birebir (eğitim satırları [0,200000); holdout [200000,202000)).
# Beklenen süre: duel ~41 saat, contextcloud ~30 saat (MPS tek, sıralı).
cd "$(dirname "$0")/../.."  # repo root
set -o pipefail

LOG=logs/run_r4_full.log
BPE=checkpoints/wiki_checkpoints_200k/wiki_bpe.json
CORPUS=data/turkish_wiki_1500k.txt
SAVE32=checkpoints/wiki_checkpoints_r4_d32_200k
SAVE64=checkpoints/wiki_checkpoints_r4_d64_200k

echo "=== [$(date)] 1/7: pin code state ===" | tee -a "$LOG"
git rev-parse HEAD | tee -a "$LOG"
echo "dirty_files=$(git status --porcelain | wc -l | tr -d ' ')" | tee -a "$LOG"
shasum -a 256 training/train_wiki.py training/compare_llm.py training/eval_r1_duel.py \
  training/eval_r4_duel.py cloudlm/model.py cloudlm/tokenizer.py \
  contextcloud/task_utils.py contextcloud/train_stable.py \
  contextcloud/train_memory_context.py contextcloud/train_global_context.py | tee -a "$LOG"

echo "=== [$(date)] 2/7: prepare R4 checkpoint dirs + shared BPE ===" | tee -a "$LOG"
mkdir -p "$SAVE32" "$SAVE64"
cp -n "$BPE" "$SAVE32/wiki_bpe.json"
cp -n "$BPE" "$SAVE64/wiki_bpe.json"
if ! cmp -s "$BPE" "$SAVE32/wiki_bpe.json" || ! cmp -s "$BPE" "$SAVE64/wiki_bpe.json"; then
  echo "BPE mismatch: $BPE vs R4 dirs" | tee -a "$LOG"
  exit 1
fi
echo "BPE ready for both R4 dirs" | tee -a "$LOG"

echo "=== [$(date)] 3/7: train D=32 (50 ep, rows [0,200000)) ===" | tee -a "$LOG"
python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 200000 \
  --d-dim 32 --save-dir "$SAVE32/" --epochs 50 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --layers 4 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] D=32 training FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 4/7: train D=64 (50 ep, rows [0,200000)) ===" | tee -a "$LOG"
python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 200000 \
  --d-dim 64 --save-dir "$SAVE64/" --epochs 50 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --layers 4 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] D=64 training FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 5/7: R4 duel evaluation ===" | tee -a "$LOG"
python3 -u training/eval_r4_duel.py 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] R4 duel eval FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 6/7: contextcloud stable (sızdırmasız görev) ===" | tee -a "$LOG"
mkdir -p contextcloud/checkpoints/pre_r4_leaky
for f in contextcloud/checkpoints/*.pt; do
  [ -e "$f" ] && mv -n "$f" contextcloud/checkpoints/pre_r4_leaky/
done
echo "eski sızıntılı contextcloud checkpoint'leri pre_r4_leaky/ altına taşındı" | tee -a "$LOG"
python3 -u contextcloud/train_stable.py 2>&1 | tee -a logs/run_r4_contextcloud_stable.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== [$(date)] CC stable FAILED ===" | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 7/7: contextcloud memory + global ===" | tee -a "$LOG"
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

echo "=== [$(date)] R4 QUEUE COMPLETE ===" | tee -a "$LOG"
