#!/bin/bash
# R1 clean capacity control: CloudLM D=32, first 200K wiki lines, 50 epochs.
cd "$(dirname "$0")/../.."  # repo root
set -o pipefail

LOG=logs/run_d32_200k.log
BPE=checkpoints/wiki_checkpoints_200k/wiki_bpe.json
SAVE=checkpoints/wiki_checkpoints_wiki200k_d32
CORPUS=data/turkish_wiki_1500k.txt

echo "=== [$(date)] 1/2: prepare D=32 checkpoint directory and shared BPE ===" | tee -a "$LOG"
mkdir -p "$SAVE"
cp -n "$BPE" "$SAVE/wiki_bpe.json"
echo "BPE copied: $BPE -> $SAVE/wiki_bpe.json" | tee -a "$LOG"
echo "=== [$(date)] 1/2 complete ===" | tee -a "$LOG"

echo "=== [$(date)] 2/2: train CloudLM D=32, 200K-wiki, 50 epochs ===" | tee -a "$LOG"
python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 200000 \
  --d-dim 32 --save-dir "$SAVE" --epochs 50 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --layers 4 2>&1 | tee -a "$LOG"
TRAIN_RC=${PIPESTATUS[0]}
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "=== [$(date)] D=32 training FAILED (exit=$TRAIN_RC) ===" | tee -a "$LOG"
  exit "$TRAIN_RC"
fi
echo "=== [$(date)] D=32 training complete ===" | tee -a "$LOG"
