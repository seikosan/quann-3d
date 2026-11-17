#!/bin/bash
# Quann-3D — Final pipeline (15 Ekim):
#   1) C: causal baseline resume 45→50 (200K, aynı BPE)
#   2) Karşılaştırma: CloudLM (A best) vs baseline (C best), 2000 cümle + üretim
#   3) B: CloudLM 32D (patent versiyonu) 1.5M wiki, 20 epoch  [büyük veri seti]
cd "$(dirname "$0")/../.."  # repo kökü
LOG=logs/run_final_pipeline.log

echo "=== [$(date)] C: resume 45→50 (200K) ===" | tee -a $LOG
python3 -u training/train_baseline.py --step train --arch causal --corpus data/turkish_sentences.txt --max-texts 200000 \
  --bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json --save-dir checkpoints/baseline_200k --epochs 50 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 384 --layers 4 --resume checkpoints/baseline_200k/baseline_45.pt >> $LOG 2>&1
C_OK=$?
echo "=== [$(date)] C tamamlandı (exit=$C_OK). Karşılaştırma başlıyor ===" | tee -a $LOG

python3 -u training/train_baseline.py --step compare --corpus data/turkish_sentences.txt --max-texts 200000 \
  --bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json \
  --cloud-ckpt checkpoints/wiki_checkpoints_200k/wiki_cloud_best.pt \
  --base-ckpt checkpoints/baseline_200k/baseline_best.pt \
  --eval-n 2000 >> $LOG 2>&1
CMP_OK=$?
echo "=== [$(date)] Karşılaştırma tamamlandı (exit=$CMP_OK). B (1.5M wiki) başlıyor ===" | tee -a $LOG

python3 -u training/train_wiki.py --step all --corpus data/turkish_wiki_1500k.txt --max-texts 1500000 \
  --d-dim 32 --save-dir checkpoints/wiki_checkpoints_wiki1500k --epochs 20 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --layers 4 >> $LOG 2>&1
B_OK=$?

echo "=== [$(date)] HEPSİ BİTTİ (C=$C_OK, compare=$CMP_OK, B=$B_OK) ===" | tee -a $LOG
