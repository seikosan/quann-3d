#!/bin/bash
# R2 Cloud Attention + Derinlik Artışı Ön-Test (20K)
# ===================================================
# Değişiklikler:
#   - Transformer katmanları: 4 → 8
#   - Cloud attention heads: 4 → 8
#   - Cloud attention: aktif
#
# Kullanım: bash training/shell/run_r2_cloud_attn_20k.sh

set -e

cd /Users/serkankose/Documents/quann-3d

echo "=== [$(date +%Y-%m-%d)] R2 Cloud Attention + Deep Model (20K ön-test) ==="
echo "Mimari:"
echo "  - Transformer layers: 4 → 8"
echo "  - Cloud attention heads: 4 → 8"
echo "  - Cloud self-attention: aktif"
echo ""

# 20K corpus (hızlı ön-test)
CORPUS="data/turkish_sentences.txt"
MAX_TEXTS=20000
SAVE_DIR="checkpoints/wiki_checkpoints_r2_cloud_attn_20k"

# R2 varyant: cloud attention + 8 katman
echo "=== [$(date +%Y-%m-%d)] 1/4: Tokenizer ==="
python3 -u training/train_wiki.py \
    --step tokenizer \
    --corpus "$CORPUS" \
    --max-texts "$MAX_TEXTS" \
    --bpe-vocab 20000 \
    --save-dir "$SAVE_DIR" \
    2>&1 | tee -a "logs/run_r2_cloud_attn_20k_tokenizer.log"

echo "=== [$(date +%Y-%m-%d)] 2/4: Eğitim (R2 Cloud Attn + 8 layers) ==="
python3 -u training/train_wiki.py \
    --step train \
    --corpus "$CORPUS" \
    --max-texts "$MAX_TEXTS" \
    --epochs 50 \
    --batch 64 \
    --lr 1e-4 \
    --seq-len 10 \
    --d-model 256 \
    --d-cloud 512 \
    --d-dim 16 \
    --layers 8 \
    --pool max \
    --cloud-attn \
    --cloud-attn-heads 8 \
    --save-dir "$SAVE_DIR" \
    2>&1 | tee -a "logs/run_r2_cloud_attn_20k.log"

echo "=== [$(date +%Y-%m-%d)] 3/4: Test ==="
python3 -u training/train_wiki.py \
    --step test \
    --corpus "$CORPUS" \
    --max-texts "$MAX_TEXTS" \
    --save-dir "$SAVE_DIR" \
    2>&1 | tee -a "logs/run_r2_cloud_attn_20k_test.log"

# Baseline kontrolü (max-pool, 4 layers, cloud-attn yok)
echo "=== [$(date +%Y-%m-%d)] 4/4: Baseline Kontrol (4 layers, no cloud-attn) ==="
python3 -u training/train_wiki.py \
    --step train \
    --corpus "$CORPUS" \
    --max-texts "$MAX_TEXTS" \
    --epochs 50 \
    --batch 64 \
    --lr 2e-4 \
    --seq-len 10 \
    --d-model 256 \
    --d-cloud 512 \
    --d-dim 16 \
    --layers 4 \
    --pool max \
    --save-dir "${SAVE_DIR}_baseline" \
    2>&1 | tee -a "logs/run_r2_cloud_attn_20k_baseline.log"

echo ""
echo "✅ R2 Cloud Attention + Deep Model ön-test TAMAM"
echo "Çıktılar:"
echo "  - R2: $SAVE_DIR"
echo "  - Baseline: ${SAVE_DIR}_baseline"
echo "Loglar: logs/run_r2_cloud_attn_20k*.log"
