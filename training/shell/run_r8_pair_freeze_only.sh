#!/bin/bash
# R8 varyant-kolu kuyruğu: yalnız pair-freeze kolu.
# Gerekçe: 18 Kas 23:05 koşusunda kontrol kolu sonucunu yazdı
# (logs/run_r8_quann_pair.log:30) fakat süreç sonucun ARDINDAN Killed: 9 aldı
# (sebep doğrulanmadı); betik bunu "R8 control FAILED" sayıp exit 1 etti ve
# varyant kolu hiç başlamadı. Kod da56961'den beri değişmedi (git diff boş);
# kontrol sonucu geçerli olduğundan yalnız varyant koşılır.
cd "$(dirname "$0")/../.."  # repo root
set -euo pipefail

LOG=logs/run_r8_pair_freeze_only.log
PAIR_RUNNER=training/quann_pair_experiment.py
mkdir -p logs

echo "=== [$(date)] 1/3: MPS control — reject concurrent train/eval ===" | tee -a "$LOG"
if pgrep -af '[t]rain_|[e]val_|[q]uann_pair' | tee -a "$LOG"; then
  echo "MPS busy: train_/eval_/quann_pair süreci bulundu; çıkıyor." | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 2/3: pin code state ===" | tee -a "$LOG"
git rev-parse HEAD | tee -a "$LOG"
shasum -a 256 "$PAIR_RUNNER" | tee -a "$LOG"

echo "=== [$(date)] 3/3: run pair-freeze variant only (control sonucu mevcut) ===" | tee -a "$LOG"
echo "R8_VARIANT_ONLY | control sonucu referansi: logs/run_r8_quann_pair.log:30 | argumanlar run_r8_quann_pair.sh:43 ile ayni" | tee -a "$LOG"
if python3 -u "$PAIR_RUNNER" --condition pair-freeze --corpus data/turkish_sentences.txt --train-limit 20000 --epochs 10 --seed 42 --split 80/10/10 --search-channel pair --freeze-head-during-search --finetune-head-after-search 2>&1 | tee -a "$LOG"; then
  echo "=== [$(date)] R8 PAIR-FREEZE ONLY COMPLETE ===" | tee -a "$LOG"
else
  echo "R8 pair/freeze variant FAILED" | tee -a "$LOG"
  exit 1
fi
