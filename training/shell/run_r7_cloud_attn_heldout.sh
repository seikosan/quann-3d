#!/bin/bash
# R7: mevcut R4 ve R2 Cloud-attention checkpoint'lerini aynı, eğitim-dışı
# seeded held-out diliminde değerlendirir; model eğitimi yapmaz.
cd "$(dirname "$0")/../.."  # repo root
set -euo pipefail

LOG=logs/run_r7_cloud_attn_heldout.log
CORPUS=data/turkish_wiki_1500k.txt
BPE=checkpoints/wiki_checkpoints_200k/wiki_bpe.json

mkdir -p logs
echo "=== [$(date)] 1/3: MPS busy-process guard ===" | tee -a "$LOG"
if pgrep -af '[t]rain_|[e]val_' | tee -a "$LOG"; then
  echo "MPS/process guard: train/eval işi çalışıyor; R7 durduruldu, boşta tekrar başlatın." | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 2/3: pin code state ===" | tee -a "$LOG"
git rev-parse HEAD | tee -a "$LOG"
shasum -a 256 training/train_wiki.py training/compare_llm.py \
  training/eval_r1_duel.py training/eval_r4_duel.py \
  cloudlm/model.py cloudlm/tokenizer.py | tee -a "$LOG"

echo "=== [$(date)] 3/3: evaluate matched seeded holdout rows [202000,212000), sample=5000 seed=42 ===" | tee -a "$LOG"
if python3 -u - "$CORPUS" "$BPE" <<'PY' 2>&1 | tee -a "$LOG"
import os
import sys
import torch

from training.compare_llm import evaluate_ppl, get_device, load_bpe, load_cloud_ckpt

corpus_path, bpe_path = sys.argv[1:]
source_start, source_end = 202_000, 212_000
sample_size, seed, seq_len = 5_000, 42, 10
root = os.getcwd()

# R4 no-attention references: D32/D64. R2 attention variants: D32/D64/D128.
# R4 and R2 training schedules/layer counts differ; these are descriptive
# held-out comparisons, not an isolated causal estimate of attention's effect.
models = [
    ("R4_D32_control", "checkpoints/wiki_checkpoints_r4_d32_200k/wiki_cloud_best.pt"),
    ("R4_D64_control", "checkpoints/wiki_checkpoints_r4_d64_200k/wiki_cloud_best.pt"),
    ("R2_attn_D32", "checkpoints/wiki_checkpoints_r2_cloud_attn_200k_d32/wiki_cloud_best.pt"),
    ("R2_attn_D64", "checkpoints/wiki_checkpoints_r2_cloud_attn_200k_d64/wiki_cloud_best.pt"),
    ("R2_attn_D128", "checkpoints/wiki_checkpoints_r2_cloud_attn_200k_d128/wiki_cloud_best.pt"),
]

texts = []
source_rows = 0
with open(corpus_path, "r", encoding="utf-8") as source:
    for row, line in enumerate(source):
        if row >= source_end:
            break
        if row >= source_start:
            source_rows += 1
            text = line.strip()
            if text:
                texts.append(text)
if source_rows != source_end - source_start:
    raise RuntimeError(f"Expected {source_end-source_start} source rows; got {source_rows}")
if len(texts) < sample_size:
    raise RuntimeError(f"Expected at least {sample_size} nonempty texts in holdout; got {len(texts)}")
if not os.path.isfile(bpe_path):
    raise FileNotFoundError(f"Shared BPE not found: {bpe_path}")
for label, rel_path in models:
    if not os.path.isfile(os.path.join(root, rel_path)):
        raise FileNotFoundError(f"Checkpoint not found for {label}: {rel_path}")

device = get_device()
tokenizer = load_bpe(bpe_path)
print(f"R7_PROTOCOL | rows=[{source_start},{source_end}) | sample={sample_size} | seed={seed} | seq_len={seq_len} | device={device}", flush=True)
for label, rel_path in models:
    model, state = load_cloud_ckpt(os.path.join(root, rel_path))
    model.to(device)
    ppl, _, tokens = evaluate_ppl(model, tokenizer, texts, device, seq_len, "cloud", n=sample_size, seed=seed)
    print(f"R7_HELDOUT_RESULT | model={label} | epoch={state['epoch']} | heldout_PPL={ppl:.4f} | tokens={tokens} | rows=[{source_start},{source_end}) | sample={sample_size} | seed={seed}", flush=True)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
PY
then
  status=0
else
  status=$?
fi
if [ "$status" -ne 0 ]; then
  echo "=== [$(date)] R7 heldout evaluation FAILED (status=$status) ===" | tee -a "$LOG"
  exit "$status"
fi
echo "=== [$(date)] R7 CLOUD-ATTN HELDOUT COMPLETE ===" | tee -a "$LOG"
