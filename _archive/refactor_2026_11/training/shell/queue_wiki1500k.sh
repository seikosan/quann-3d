#!/bin/bash
# Quann-3D — Kuyruk bekleyici: run_200k (A→C→karşılaştırma) bitince B'yi başlat
cd "$(dirname "$0")/../.."  # repo kökü
echo "[waiter] run_200k bitene kadar bekleniyor..." | tee -a logs/queue_wiki1500k.log
while ! grep -q "HEPSİ BİTTİ" logs/run_200k.log 2>/dev/null; do
  sleep 120
done
echo "[waiter] run_200k bitti, B (wiki 1.5M) başlatılıyor: $(date)" | tee -a logs/queue_wiki1500k.log
bash training/shell/run_wiki1500k.sh
