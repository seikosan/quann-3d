#!/bin/bash
# Quann-3D — 200K pipeline: A (CloudLM 32D) → C (causal baseline) → karşılaştırma
# -u: unbuffered stdout, epoch satırları log'a anında düşer.
cd "$(dirname "$0")/../.."  # repo kökü
LOG=logs/run_200k.log

echo "=== [$(date)] A: CloudLM 32D (200K, 50ep) ===" | tee -a $LOG
python3 -u training/train_wiki.py --step train --corpus data/turkish_sentences.txt --max-texts 200000 \
  --d-dim 32 --save-dir checkpoints/wiki_checkpoints_200k --epochs 50 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --layers 4 >> $LOG 2>&1

echo "=== [$(date)] C: causal baseline (200K, 50ep) ===" | tee -a $LOG
python3 -u training/train_baseline.py --step train --arch causal --corpus data/turkish_sentences.txt --max-texts 200000 \
  --bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json --save-dir checkpoints/baseline_200k --epochs 50 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 384 --layers 4 >> $LOG 2>&1

echo "=== [$(date)] Karşılaştırma ===" | tee -a $LOG
python3 -u training/train_baseline.py --step compare --corpus data/turkish_sentences.txt --max-texts 200000 \
  --bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json \
  --cloud-ckpt checkpoints/wiki_checkpoints_200k/wiki_cloud_best.pt \
  --base-ckpt checkpoints/baseline_200k/baseline_best.pt \
  --eval-n 2000 >> $LOG 2>&1

echo "=== [$(date)] HEPSİ BİTTİ ===" | tee -a $LOG
