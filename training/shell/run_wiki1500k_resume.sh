#!/bin/bash
# Quann-3D — B (1.5M wiki): resume — epoch 1'den (wiki_cloud_best.pt) devam
# Orijinal run 14:38'de epoch 1/20'da ölmüştü (15 Ekim). BPE yeniden eğitilmiyor.
cd "$(dirname "$0")/../.."  # repo kökü
LOG=logs/run_final_pipeline.log
SAVE=checkpoints/wiki_checkpoints_wiki1500k

echo "=== [$(date)] B (1.5M wiki): resume epoch 1→20 ===" | tee -a $LOG
python3 -u training/train_wiki.py --step train --corpus data/turkish_wiki_1500k.txt --max-texts 1500000 \
  --d-dim 32 --save-dir $SAVE --epochs 20 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --layers 4 \
  --resume $SAVE/wiki_cloud_best.pt >> $LOG 2>&1
B_OK=$?

echo "=== [$(date)] B tamamlandı (exit=$B_OK). Test/üretim başlıyor ===" | tee -a $LOG
python3 -u training/train_wiki.py --step test --corpus data/turkish_wiki_1500k.txt \
  --d-dim 32 --save-dir $SAVE --d-model 256 --d-cloud 512 --layers 4 >> $LOG 2>&1

echo "=== [$(date)] B resume HEPSİ BİTTİ (train=$B_OK) ===" | tee -a $LOG
