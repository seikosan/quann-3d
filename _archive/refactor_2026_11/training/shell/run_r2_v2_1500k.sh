#!/bin/bash
# CloudLM R2 Training - V2 (1.5M corpus, D=64, Nokta=20, cloud_attn=True)

cd /Users/serkankose/Documents/quann-3d

echo "=== [$(date +%Y-%m-%d)] Starting CloudLM R2 V2 Training ==="
echo "Config: D=64, cloud_attn=True, 8 layers, 1.5M corpus"
echo "Checkpoint: checkpoints/wiki_checkpoints_r2_cloud_attn_1500k_d64_n20_v2/"
echo ""

python3 -u training/train_simple_r2_v2.py 2>&1 | tee -a logs/run_simple_r2_v2_100ep.log

echo ""
echo "=== Training complete ==="
