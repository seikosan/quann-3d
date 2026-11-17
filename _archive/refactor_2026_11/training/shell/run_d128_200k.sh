#!/bin/bash
# R1 D=128 capacity test (200K wiki, 10ep)
cd "$(dirname "$0")/../.."
set -euo pipefail

CORPUS=data/turkish_wiki_1500k.txt
BPE=checkpoints/wiki_checkpoints_200k/wiki_bpe.json
DIR=checkpoints/d128_200k
LOG=logs/run_d128_200k.log

mkdir -p "$DIR"
if [ ! -f "$DIR/wiki_bpe.json" ]; then cp -n "$BPE" "$DIR/wiki_bpe.json"; fi

echo "=== [$(date)] R1 D=128 (200K, 10ep) ===" | tee -a "$LOG"
python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 200000 \
  --save-dir "$DIR" --epochs 10 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --d-dim 128 --layers 4 --pool max 2>&1 | tee -a "$LOG"

echo "=== [$(date)] R1 D=128 complete ===" | tee -a "$LOG"
