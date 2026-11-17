#!/bin/bash
cd "$(dirname "$0")/../.."  # repo kökü
set -euo pipefail
echo "[B-waiter] C + karşılaştırma bitene kadar bekleniyorum..." | tee -a logs/queue_B.log
while [ ! -f logs/run_compare.log ]; do
  sleep 60
done
echo "[B-waiter] C + karşılaştırma bitti! B başlıyor: $(date)" | tee -a logs/queue_B.log
python3 training/train_wiki.py --step all --corpus data/turkish_wiki_1500k.txt --max-texts 1500000 \
  --d-dim 32 --save-dir checkpoints/wiki_checkpoints_wiki1500k --epochs 20 --batch 256 \
  --lr 2e-4 --seq-len 10 --d-model 256 --d-cloud 512 --layers 4 > logs/run_B.log 2>&1
echo "[B-waiter] B TAMAMLANDI!" | tee -a logs/queue_B.log
