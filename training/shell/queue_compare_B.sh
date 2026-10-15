#!/bin/bash
# C (baseline) bittiğinde compare + B'yi başlatır
cd "$(dirname "$0")/../.."  # repo kökü
echo "[compare_B] C'nin 50. epoch'unu bekliyorum..."
while [ ! -f checkpoints/baseline_200k/baseline_epoch50.pt ]; do
  sleep 120
done
echo "[compare_B] C bitti! Karşılaştırma başlıyor: $(date)"
python3 training/train_baseline.py --step compare --corpus data/turkish_sentences.txt --max-texts 200000 \
  --bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json \
  --cloud-ckpt checkpoints/wiki_checkpoints_200k/wiki_cloud_best.pt \
  --base-ckpt checkpoints/baseline_200k/baseline_best.pt \
  --eval-n 2000 2>&1 | tee -a logs/run_compare.log
echo "[compare_B] Karşılaştırma bitti! B (1.5M wiki) başlıyor: $(date)"
python3 training/train_wiki.py --step all --corpus data/turkish_wiki_1500k.txt --max-texts 1500000 \
  --d-dim 32 --save-dir checkpoints/wiki_checkpoints_wiki1500k --epochs 10 --batch 256 \
  --lr 2e-4 --seq-len 10 --d-model 256 --d-cloud 512 --layers 4 2>&1 | tee -a logs/run_B.log
echo "[compare_B] B TAMAMLANDI! Tüm pipeline bitti!"
