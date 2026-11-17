#!/bin/bash
# CloudLM d=384, 4L 200K wiki (clean rerun) — NOT baseline causal transformer
cd "$(dirname "$0")/../.."
set -euo pipefail

CORPUS=data/turkish_wiki_1500k.txt
BPE=checkpoints/wiki_checkpoints_200k/wiki_bpe.json
DIR=checkpoints/baseline_200k_clean
LOG=logs/run_baseline_200k.log

mkdir -p "$DIR"
if [ ! -f "$DIR/wiki_bpe.json" ]; then cp -n "$BPE" "$DIR/wiki_bpe.json"; fi

echo "=== [$(date)] Baseline 200K (d=384, 4L, 50ep) ===" | tee -a "$LOG"
python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 200000 \
  --save-dir "$DIR" --epochs 50 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 384 --d-cloud 512 --d-dim 32 --layers 4 --pool max 2>&1 | tee -a "$LOG"

echo "=== [$(date)] Baseline 200K complete ===" | tee -a "$LOG"
