#!/bin/bash
# R1 Kapasite Ön-Testi (quann_cloud/REVISION.md):
#   20K Türkçe cümle × 10 epoch — CloudLM D=32 (kontrol) / D=64 / D=128 + causal baseline
# Protokol: 200K karşılaştırmasıyla aynı (seed=0, eğitim kümesinden 2000 cümle örneği).
# Beklenti (REVISION.md): held-out 6.45 → D=64 ~4-5, D=128 ~3-4. Transformer 2.92'yi geçer mi?
cd "$(dirname "$0")/../.."  # repo kökü
set -euo pipefail
LOG=logs/run_rev1_20k.log
CORPUS=data/turkish_sentences.txt
BPE=checkpoints/wiki_checkpoints_200k/wiki_bpe.json
BASE=checkpoints/rev1_20k
mkdir -p "$BASE"

echo "=== [$(date)] R1 ön-test başlıyor (20K, 10ep, MPS) ===" | tee -a "$LOG"

# 1) Baseline (causal, 22.7M) — tüm karşılaştırmaların referansı
echo "=== [$(date)] 1/4: causal baseline (22.7M) ===" | tee -a "$LOG"
python3 -u training/train_baseline.py --step train --arch causal --corpus "$CORPUS" --max-texts 20000 \
  --bpe "$BPE" --save-dir "$BASE/base" --epochs 10 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 384 --layers 4 2>&1 | tee -a "$LOG"
BASE_OK=${PIPESTATUS[0]}

for D in 32 64 128; do
  DIR="$BASE/d$D"
  mkdir -p "$DIR"
  cp -n "$BPE" "$DIR/wiki_bpe.json"

  echo "=== [$(date)] CloudLM D=$D (20K, 10ep) ===" | tee -a "$LOG"
  python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 20000 \
    --d-dim "$D" --save-dir "$DIR" --epochs 10 --batch 64 --lr 2e-4 --seq-len 10 \
    --d-model 256 --d-cloud 512 --layers 4 2>&1 | tee -a "$LOG"

  if [ -f "$DIR/wiki_cloud_best.pt" ]; then
    echo "=== [$(date)] Karşılaştırma: CloudLM D=$D vs baseline (2000 cümle) ===" | tee -a "$LOG"
    python3 -u training/train_baseline.py --step compare --corpus "$CORPUS" --max-texts 20000 \
      --bpe "$BPE" --cloud-ckpt "$DIR/wiki_cloud_best.pt" \
      --base-ckpt "$BASE/base/baseline_best.pt" --eval-n 2000 2>&1 | tee -a "$LOG"
  else
    echo "=== [$(date)] CloudLM D=$D checkpoint YOK — karşılaştırma atlandı ===" | tee -a "$LOG"
  fi
done

echo "=== [$(date)] R1 ön-test HEPSİ BİTTİ (baseline=$BASE_OK) ===" | tee -a "$LOG"
echo "Sonraki adım (REVISION.md): vaadi 200K'a ölçekle — kararlaştırılacak." | tee -a "$LOG"
