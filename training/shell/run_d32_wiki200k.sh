#!/bin/bash
# R1 matched capacity duel. Training uses rows [0,200000); held-out evaluation
# uses the disjoint, deterministic consecutive rows [200000,202000).
cd "$(dirname "$0")/../.."  # repo root
set -o pipefail

LOG=logs/run_d32_wiki200k.log
LEGACY_LOG=logs/run_d32_200k.log
BPE=checkpoints/wiki_checkpoints_200k/wiki_bpe.json
SAVE=checkpoints/wiki_checkpoints_wiki200k_d32
CORPUS=data/turkish_wiki_1500k.txt
EXISTING_RUN_MATCH='training/train_wiki.py --step train --corpus data/turkish_wiki_1500k.txt --max-texts 200000 --d-dim 32 --save-dir checkpoints/wiki_checkpoints_wiki200k_d32'

echo "=== [$(date)] 1/3: prepare D=32 checkpoint directory and shared BPE ===" | tee -a "$LOG"
mkdir -p "$SAVE"
cp -n "$BPE" "$SAVE/wiki_bpe.json"
if ! cmp -s "$BPE" "$SAVE/wiki_bpe.json"; then
  echo "BPE mismatch: $BPE vs $SAVE/wiki_bpe.json" | tee -a "$LOG"
  exit 1
fi
echo "BPE ready: $BPE -> $SAVE/wiki_bpe.json" | tee -a "$LOG"
echo "=== [$(date)] 1/3 complete ===" | tee -a "$LOG"

echo "=== [$(date)] 2/3: train/adopt CloudLM D=32; first 200K corpus lines ===" | tee -a "$LOG"
ACTIVE_PID=$(ps -axo pid=,command= | awk -v needle="$EXISTING_RUN_MATCH" '$0 ~ needle && $0 !~ /awk/ {print $1; exit}')
if [ -n "$ACTIVE_PID" ]; then
  echo "Adopting matching active D32 training PID=$ACTIVE_PID; waiting before evaluation." | tee -a "$LOG"
  while kill -0 "$ACTIVE_PID" 2>/dev/null; do sleep 60; done
  if ! grep -q "D=32 training complete" "$LEGACY_LOG" && ! grep -q "D32 training complete" "$LOG"; then
    echo "Existing D32 training ended without a completion marker; refusing evaluation." | tee -a "$LOG"
    exit 1
  fi
else
  python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 200000 \
    --d-dim 32 --save-dir "$SAVE/" --epochs 50 --batch 64 --lr 2e-4 --seq-len 10 \
    --d-model 256 --d-cloud 512 --layers 4 2>&1 | tee -a "$LOG"
  TRAIN_RC=${PIPESTATUS[0]}
  if [ "$TRAIN_RC" -ne 0 ]; then
    echo "=== [$(date)] D=32 training FAILED (exit=$TRAIN_RC) ===" | tee -a "$LOG"
    exit "$TRAIN_RC"
  fi
fi
echo "=== [$(date)] 2/3 complete ===" | tee -a "$LOG"

echo "=== [$(date)] 3/3: matched D32/D64 evaluation; fixed holdout rows [200000,202000), same BPE, seed=0, n=2000, seq-len=10 ===" | tee -a "$LOG"
python3 -u training/eval_r1_duel.py 2>&1 | tee -a "$LOG"
EVAL_RC=${PIPESTATUS[0]}
if [ "$EVAL_RC" -ne 0 ]; then
  echo "=== [$(date)] matched duel evaluation FAILED (exit=$EVAL_RC) ===" | tee -a "$LOG"
  exit "$EVAL_RC"
fi
echo "=== [$(date)] 3/3 complete ===" | tee -a "$LOG"
