#!/bin/bash
# CloudLM 1.5M sonrasındaki test ve ortak held-out karşılaştırması.
cd "$(dirname "$0")/../.."  # repo kökü
set -u
LOG=logs/run_wiki1500k_posttrain.log

echo "=== [$(date)] 0/3: 1.5M train bitene kadar bekle (PID 56169) ===" | tee -a "$LOG"
while kill -0 56169 2>/dev/null; do
  sleep 300
done

echo "=== [$(date)] 1.5M train bitti ===" | tee -a "$LOG"
echo "=== [$(date)] 1/3: CloudLM test/üretim/geometri ===" | tee -a "$LOG"
python3 -u training/train_wiki.py --step test --corpus data/turkish_wiki_1500k.txt --max-texts 1500000 \
  --d-dim 32 --save-dir checkpoints/wiki_checkpoints_wiki1500k --d-model 256 --d-cloud 512 \
  --layers 4 --seq-len 10 --epochs 20 --batch 64 --lr 2e-4 2>&1 | tee -a "$LOG"

echo "=== [$(date)] 2/3: ortak held-out PPL karşılaştırması (2000 cümle) ===" | tee -a "$LOG"
python3 -u training/train_baseline.py --step compare --corpus data/turkish_sentences.txt --max-texts 200000 \
  --cloud-bpe checkpoints/wiki_checkpoints_wiki1500k/wiki_bpe.json \
  --base-bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json \
  --cloud-ckpt checkpoints/wiki_checkpoints_wiki1500k/wiki_cloud_best.pt \
  --base-ckpt checkpoints/baseline_200k/baseline_best.pt --eval-n 2000 2>&1 | tee -a "$LOG"

echo "=== [$(date)] 3/3: HEPSİ BİTTİ ===" | tee -a "$LOG"
