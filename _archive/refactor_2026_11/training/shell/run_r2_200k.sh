#!/bin/bash
# R2 cloud attention 200K test: cloud self-attention vs control
cd "$(dirname "$0")/../.."  # repo kökü
set -euo pipefail

CORPUS=data/turkish_wiki_1500k.txt
BPE=checkpoints/wiki_checkpoints_200k/wiki_bpe.json
CTRL_DIR=checkpoints/r2_200k/control
ATTN_DIR=checkpoints/r2_200k/cloud_attn
CTRL_LOG=logs/run_r2_200k_control.log
ATTN_LOG=logs/run_r2_200k_cloud_attn.log

mkdir -p "$CTRL_DIR" "$ATTN_DIR"
if [ ! -f "$CTRL_DIR/wiki_bpe.json" ]; then cp -n "$BPE" "$CTRL_DIR/wiki_bpe.json"; fi
if [ ! -f "$ATTN_DIR/wiki_bpe.json" ]; then cp -n "$BPE" "$ATTN_DIR/wiki_bpe.json"; fi

echo "=== [$(date)] 1/2: R2 control (no cloud attn, 200K, 10ep, D=32) ===" | tee -a "$CTRL_LOG"
python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 200000 \
  --save-dir "$CTRL_DIR" --epochs 10 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --d-dim 32 --layers 4 --pool max 2>&1 | tee -a "$CTRL_LOG"

echo "=== [$(date)] 2/2: R2 cloud self-attention (200K, 10ep, D=32) ===" | tee -a "$ATTN_LOG"
python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 200000 \
  --save-dir "$ATTN_DIR" --epochs 10 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --d-dim 32 --layers 4 --pool max --cloud-attn 2>&1 | tee -a "$ATTN_LOG"

echo "=== [$(date)] R2 200K queue complete ===" | tee -a "$CTRL_LOG" "$ATTN_LOG"
