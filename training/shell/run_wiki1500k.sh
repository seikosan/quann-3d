#!/bin/bash
# Quann-3D — B: CloudLM 32D, Türkçe Wikipedia alt küme (1.5M cümle)
cd "$(dirname "$0")/../.."  # repo kökü
LOG=logs/run_wiki1500k.log

echo "=== [$(date)] B: CloudLM 32D (1.5M wiki, 20ep) ===" | tee -a $LOG
python3 -u training/train_wiki.py --step all --corpus data/turkish_wiki_1500k.txt --max-texts 1500000 \
  --d-dim 32 --save-dir checkpoints/wiki_checkpoints_wiki1500k --epochs 20 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --layers 4 >> $LOG 2>&1

echo "=== [$(date)] B DONE ===" | tee -a $LOG
