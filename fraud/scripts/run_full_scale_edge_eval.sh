#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p fraud/logs fraud/results
LOG="fraud/logs/run_full_scale_edge_eval.log"
exec > >(tee -a "$LOG") 2>&1

printf '=== [%s] 1/4: train fresh full-scale QUANN hybrid ===\n' "$(date '+%Y-%m-%d %H:%M:%S')"
python3 -u fraud/src/train_quann.py \
  --configs hybrid \
  --epochs 1 \
  --n-train 1020000 \
  --n-val 100000 \
  --n-test 100000 \
  --train-fraud-frac 0.02 \
  --test-fraud-frac 0.0013 \
  --scale full \
  --out fraud/results/full_scale_hybrid.json

printf '=== [%s] 2/4: pin fresh checkpoint as quann_hybrid_full.pt ===\n' "$(date '+%Y-%m-%d %H:%M:%S')"
cp fraud/results/quann_full_hybrid.pt fraud/results/quann_hybrid_full.pt

printf '=== [%s] 3/4: edge_eval bundled-quantizer AUC on full natural test ===\n' "$(date '+%Y-%m-%d %H:%M:%S')"
python3 -u fraud/src/edge_eval.py \
  --with-auc \
  --n-test-auc 4700000 \
  --test-frac 0.0013 \
  --checkpoint fraud/results/quann_hybrid_full.pt
cp fraud/results/edge_eval.json fraud/results/edge_eval_full_scale.json

printf '=== [%s] 4/4: final full-scale experiment summary ===\n' "$(date '+%Y-%m-%d %H:%M:%S')"
python3 - <<'PY'
import json

with open("fraud/results/full_scale_hybrid.json") as f:
    train = json.load(f)
with open("fraud/results/edge_eval_full_scale.json") as f:
    edge = json.load(f)
test = train["configs"]["hybrid"]["test"]
auc = edge.get("int8_auc", {})
print("FULL_SCALE_SUMMARY")
print(f"train/val/test rows={train['split']['train']}/{train['split']['val']}/{train['split']['test']}")
print(f"hybrid test metrics={json.dumps(test, sort_keys=True)}")
print(f"natural-test edge AUC={json.dumps(auc, sort_keys=True)}")
print("checkpoint=fraud/results/quann_hybrid_full.pt (trained quantizer bundled)")
print("results=fraud/results/full_scale_hybrid.json, fraud/results/edge_eval_full_scale.json")
print("FULL_SCALE_SUMMARY_END")
PY
