#!/bin/bash
# CloudLM 1.5M wiki scale gate: resume, OOD reference, 200K control, matched on-wiki eval.
cd "$(dirname "$0")/../.."  # repo root
LOG=logs/run_wiki_scale_gate.log
SAVE15=checkpoints/wiki_checkpoints_wiki1500k
SAVE200=checkpoints/wiki_checkpoints_wiki200k
CORPUS=data/turkish_wiki_1500k.txt

echo "=== [$(date)] 1/5: 1.5M wiki resume epoch 17→20 ===" | tee -a "$LOG"
nohup bash training/shell/monitor_wiki1500k_resume.sh "$$" 600 >/dev/null 2>&1 &
MONITOR_PID=$!
echo "[$(date '+%F %T')] RSS/free-RAM monitor PID=$MONITOR_PID; log=logs/monitor_wiki1500k_resume.log" | tee -a "$LOG"
python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 1500000 \
  --d-dim 32 --save-dir "$SAVE15" --epochs 20 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --layers 4 \
  --resume "$SAVE15/wiki_cloud_best.pt" 2>&1 | tee -a "$LOG"
RESUME_RC=${PIPESTATUS[0]}
if [ "$RESUME_RC" -ne 0 ]; then
  echo "=== [$(date)] 1.5M resume FAILED (exit=$RESUME_RC); queue stopped ===" | tee -a "$LOG"
  exit "$RESUME_RC"
fi

echo "=== [$(date)] 2/5: 1.5M on-wiki vs 200K baseline OOD reference ===" | tee -a "$LOG"
echo "NOTE: baseline_200k eğitimi cümle dağıtımında; wiki sonucu OOD referansıdır, ölçek kararı değildir." | tee -a "$LOG"
python3 -u training/train_baseline.py --step compare --corpus "$CORPUS" --max-texts 1500000 \
  --cloud-bpe "$SAVE15/wiki_bpe.json" --base-bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json \
  --cloud-ckpt "$SAVE15/wiki_cloud_best.pt" --base-ckpt checkpoints/baseline_200k/baseline_best.pt \
  --eval-n 2000 2>&1 | tee -a "$LOG"
CMP_RC=${PIPESTATUS[0]}
if [ "$CMP_RC" -ne 0 ]; then
  echo "=== [$(date)] 1.5M OOD-reference compare FAILED (exit=$CMP_RC); queue stopped ===" | tee -a "$LOG"
  exit "$CMP_RC"
fi

echo "=== [$(date)] 3/5: prepare and train 200K-wiki control (50 epochs) ===" | tee -a "$LOG"
mkdir -p "$SAVE200"
cp "$SAVE15/wiki_bpe.json" "$SAVE200/wiki_bpe.json"
echo "Subset: --max-texts 200000 applies train_wiki.load_txt's first-200K-lines cap; no data/ file is created." | tee -a "$LOG"
python3 -u training/train_wiki.py --step train --corpus "$CORPUS" --max-texts 200000 \
  --d-dim 32 --save-dir "$SAVE200" --epochs 50 --batch 64 --lr 2e-4 --seq-len 10 \
  --d-model 256 --d-cloud 512 --layers 4 2>&1 | tee -a "$LOG"
CTRL_RC=${PIPESTATUS[0]}
if [ "$CTRL_RC" -ne 0 ]; then
  echo "=== [$(date)] 200K-wiki control FAILED (exit=$CTRL_RC); queue stopped ===" | tee -a "$LOG"
  exit "$CTRL_RC"
fi

echo "=== [$(date)] 4/5: matched on-wiki evaluation (same wiki corpus/BPE, n=2000) ===" | tee -a "$LOG"
python3 -u - 2>&1 <<'PY' | tee -a "$LOG"
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "training"))
import torch
from cloudlm import BPETokenizer, CloudLM
from training.train_baseline import evaluate_ppl, load_txt

corpus = "data/turkish_wiki_1500k.txt"
tok_path = "checkpoints/wiki_checkpoints_wiki1500k/wiki_bpe.json"
texts = load_txt(corpus, 1_500_000)
tok = BPETokenizer.load(tok_path)
device = "mps" if torch.backends.mps.is_available() else "cpu"
results = {}
for label, path in (
    ("cloud_1500k", "checkpoints/wiki_checkpoints_wiki1500k/wiki_cloud_best.pt"),
    ("cloud_200k_wiki", "checkpoints/wiki_checkpoints_wiki200k/wiki_cloud_best.pt"),
):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CloudLM(ckpt["vocab_size"], **ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    ppl = evaluate_ppl(model, tok, texts, device, 10, "cloud", n=2000, seed=0)
    results[label] = (ppl, ckpt["epoch"])
    print(f"ON_WIKI_EVAL | {label} | epoch={ckpt['epoch']} | held-out PPL={ppl:.2f} | n=2000 | BPE={tok_path}", flush=True)
    del model
    if device == "mps":
        torch.mps.empty_cache()

p15, e15 = results["cloud_1500k"]
p200, e200 = results["cloud_200k_wiki"]
exposure15 = 609.0 * 20 / 17  # prior corpus-token estimate was 609M at 17 completed epochs
exposure200 = 231.0           # prior estimate for 200K × 50 epochs
ratio = exposure15 / exposure200
if p15 < p200:
    answer = "20-epoch 1.5M model has lower same-distribution PPL than the 50-epoch 200K control."
elif p15 > p200:
    answer = "20-epoch 1.5M model has higher same-distribution PPL than the 50-epoch 200K control; scale alone did not win at these budgets."
else:
    answer = "Both models tie on the same-distribution evaluation to the reported precision."
print("=== SCALE-GATE FINAL SUMMARY ===", flush=True)
print(f"On-wiki PPL: 1.5M@{e15}ep={p15:.2f}; 200K-wiki@{e200}ep={p200:.2f} (same corpus/BPE, eval-n=2000).", flush=True)
print(f"Approx. token exposure: 1.5M×20 ≈ {exposure15:.0f}M vs 200K×50 ≈ {exposure200:.0f}M ({ratio:.1f}×; prior estimates 609M@17 and 231M).", flush=True)
print(f"Ölçek sorusuna yanıt: {answer}", flush=True)
PY
EVAL_RC=${PIPESTATUS[0]}
if [ "$EVAL_RC" -ne 0 ]; then
  echo "=== [$(date)] matched on-wiki evaluation FAILED (exit=$EVAL_RC) ===" | tee -a "$LOG"
  exit "$EVAL_RC"
fi

echo "=== [$(date)] 5/5: wiki scale gate queue complete ===" | tee -a "$LOG"
