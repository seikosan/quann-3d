#!/bin/bash
cd "$(dirname "$0")/../.."  # repo kökü
echo "[C-waiter] A'nın 50. epoch checkpoint'ını bekliyorum..." | tee -a logs/queue_C.log
while [ ! -f checkpoints/wiki_checkpoints_200k/wiki_cloud_epoch50.pt ]; do
  sleep 120
done
echo "[C-waiter] A bitti! C başlıyor: $(date)" | tee -a logs/queue_C.log
python3 training/train_baseline.py --step train --arch causal --corpus data/turkish_sentences.txt --max-texts 200000 \
  --bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json --save-dir checkpoints/baseline_200k --epochs 50 --batch 256 \
  --lr 2e-4 --seq-len 10 --d-model 384 --layers 4 > logs/run_C.log 2>&1
echo "[C-waiter] C bitti! Karşılaştırma başlıyor..." | tee -a logs/queue_C.log
python3 training/train_baseline.py --step compare --corpus data/turkish_sentences.txt --max-texts 200000 \
  --bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json \
  --cloud-ckpt checkpoints/wiki_checkpoints_200k/wiki_cloud_best.pt \
  --base-ckpt checkpoints/baseline_200k/baseline_best.pt \
  --eval-n 2000 > logs/run_compare.log 2>&1
echo "[C-waiter] Karşılaştırma bitti! B bekleyicisini başlatıyorum: $(date)" | tee -a logs/queue_C.log
bash training/shell/queue_B.sh
