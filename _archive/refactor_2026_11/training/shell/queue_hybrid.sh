#!/bin/bash
# Quann-3D — Hibrit QUANN-Cloud kuyruğu (MPS tek cihaz → sıralı)
# 1) Baseline resume 45→50   2) 200K karşılaştırma
# 3) Control CloudLM 20K     4) Hibrit QUANN 20K   5) 20K karşılaştırma
cd "$(dirname "$0")/../.."  # repo kökü
set -euo pipefail
LOG=quann_cloud/logs/queue_hybrid.log
mkdir -p quann_cloud/experiments/control_20k
cp -n checkpoints/wiki_checkpoints_200k/wiki_bpe.json quann_cloud/experiments/control_20k/wiki_bpe.json

echo "[$(date)] === KUYRUK BAŞLADI ===" | tee -a "$LOG"

echo "[$(date)] 1/5: Baseline resume (epoch 45→50)" | tee -a "$LOG"
python3 -u training/train_baseline.py --step train --arch causal --corpus data/turkish_sentences.txt \
  --max-texts 200000 --bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json --save-dir checkpoints/baseline_200k \
  --epochs 50 --batch 256 --lr 2e-4 --seq-len 10 --d-model 384 --layers 4 \
  --resume checkpoints/baseline_200k/baseline_45.pt 2>&1 | tee -a logs/run_C_resume2.log
echo "[$(date)] Baseline tamam" | tee -a "$LOG"

python3 -u training/train_baseline.py --step compare --corpus data/turkish_sentences.txt --max-texts 200000 \
  --bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json \
  --cloud-ckpt checkpoints/wiki_checkpoints_200k/wiki_cloud_best.pt \
  --base-ckpt checkpoints/baseline_200k/baseline_best.pt \
  --eval-n 2000 2>&1 | tee -a logs/run_compare.log
echo "[$(date)] 2/5: 200K karşılaştırma tamam" | tee -a "$LOG"

echo "[$(date)] 3/5: Control CloudLM 20K (D=16, 10ep)" | tee -a "$LOG"
python3 -u training/train_wiki.py --step train --corpus data/turkish_sentences.txt --max-texts 20000 \
  --d-dim 16 --save-dir quann_cloud/experiments/control_20k --epochs 10 --batch 64 \
  --lr 2e-4 --seq-len 10 --d-model 256 --d-cloud 512 --layers 4 \
  2>&1 | tee -a quann_cloud/logs/control_20k.log
echo "[$(date)] Control tamam" | tee -a "$LOG"

echo "[$(date)] 4/5: Hibrit QUANN-Cloud 20K (10ep)" | tee -a "$LOG"
python3 -u quann_cloud/experiments/tiny_corpus.py --max-texts 20000 --epochs 10 \
  --eval-texts 2000 --hybrid 2>&1 | tee -a quann_cloud/logs/hybrid_20k.log
echo "[$(date)] Hibrit tamam" | tee -a "$LOG"

echo "[$(date)] 5/5: 20K karşılaştırma" | tee -a "$LOG"
python3 -u quann_cloud/experiments/compare_20k.py 2>&1 | tee -a quann_cloud/logs/compare_20k.log
echo "[$(date)] === HEPSİ BİTTİ ===" | tee -a "$LOG"
