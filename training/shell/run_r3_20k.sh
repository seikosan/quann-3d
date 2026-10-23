#!/bin/bash
# R3 matched 20K pre-test: max-pool control, then attention-weighted pooling.
# Both arms share the same BPE/corpus/config and run sequentially on single MPS.
cd "$(dirname "$0")/../.."  # repo kökü
set -euo pipefail

CORPUS=data/turkish_wiki_1500k.txt
BPE=checkpoints/wiki_checkpoints_200k/wiki_bpe.json
MAX_DIR=checkpoints/r3_20k/max
ATTN_DIR=checkpoints/r3_20k/attn
MAX_LOG=logs/run_r3_20k_max_v2.log
ATTN_LOG=logs/run_r3_20k_attn_v2.log

mkdir -p "$MAX_DIR" "$ATTN_DIR"
if [ ! -f "$MAX_DIR/wiki_bpe.json" ]; then cp -n "$BPE" "$MAX_DIR/wiki_bpe.json"; fi
if [ ! -f "$ATTN_DIR/wiki_bpe.json" ]; then cp -n "$BPE" "$ATTN_DIR/wiki_bpe.json"; fi

echo "=== [$(date)] 1/2: R3 matched max-pool control (20K, 10ep, D=32) ===" | tee -a "$MAX_LOG"
python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 20000 \
  --save-dir "$MAX_DIR" --epochs 10 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --d-dim 32 --layers 4 --pool max 2>&1 | tee -a "$MAX_LOG"

echo "=== [$(date)] 2/2: R3 attention-weighted pooling (20K, 10ep, D=32) ===" | tee -a "$ATTN_LOG"
python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 20000 \
  --save-dir "$ATTN_DIR" --epochs 10 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --d-dim 32 --layers 4 --pool attn 2>&1 | tee -a "$ATTN_LOG"

echo "=== [$(date)] R3 matched 20K queue complete ===" | tee -a "$MAX_LOG" "$ATTN_LOG"
