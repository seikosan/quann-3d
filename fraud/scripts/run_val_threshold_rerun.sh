#!/bin/bash
# Fraud threshold-based re-run script (FIXED protocol: val-based threshold)
# Run AFTER live CloudLM training completes
# This regenerates fraud JSONs with correct val-based threshold selection

cd "$(dirname "$0")/../.."
set -euo pipefail

LOG=logs/run_fraud_threshold_rerun.log
echo "=== [$(date)] Fraud threshold-based re-run başlıyor ===" | tee -a "$LOG"

# 1) QAT Gate (FIXED: threshold from val, not test)
echo "=== [$(date)] 1/4: QAT Gate re-run (val-based threshold) ===" | tee -a "$LOG"
python3 -u fraud/scripts/run_qat_gate_fixed.py --threshold 0.9969 2>&1 | tee -a "$LOG"
echo "=== [$(date)] QAT Gate tamam ===" | tee -a "$LOG"

# 2) Online Cloud Separation (FIXED: merchant cohort from train/val audit)
echo "=== [$(date)] 2/4: Online Cloud Separation re-run ===" | tee -a "$LOG"
python3 -u fraud/src/run_online_cloud_separation.py 2>&1 | tee -a "$LOG"
echo "=== [$(date)] Online Cloud Separation tamam ===" | tee -a "$LOG"

# 3) Strong Reader (FIXED: threshold from val)
echo "=== [$(date)] 3/4: Strong Reader re-run (val-based threshold) ===" | tee -a "$LOG"
python3 -u fraud/src/train_strong_reader.py 2>&1 | tee -a "$LOG"
echo "=== [$(date)] Strong Reader tamam ===" | tee -a "$LOG"

# 4) Explain Pilot (FIXED: threshold protocol documented)
echo "=== [$(date)] 4/4: Explain Pilot re-run ===" | tee -a "$LOG"
python3 -u fraud/src/explain_pilot.py 2>&1 | tee -a "$LOG"
echo "=== [$(date)] Explain Pilot tamam ===" | tee -a "$LOG"

echo "=== [$(date)] Fraud threshold-based re-run HEPSİ BİTTİ ===" | tee -a "$LOG"
echo "NOT: JSON'lar val-based threshold protokolüyle güncellendi." | tee -a "$LOG"
