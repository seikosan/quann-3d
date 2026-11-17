#!/bin/bash
# Quann-3D — 1.5M wiki run monitörü: süreç/RSS/swap/epoch/checkpoint, 10dk aralık
# append-only: logs/monitor_wiki1500k.log
cd "$(dirname "$0")/../.."  # repo kökü
LOG=logs/monitor_wiki1500k.log
SAVE=checkpoints/wiki_checkpoints_wiki1500k
PIPELOG=logs/run_final_pipeline.log
PID=${1:-56169}
INTERVAL=${2:-600}

echo "=== [$(date)] MONITOR başladı: PID=$PID aralık=${INTERVAL}s ===" | tee -a $LOG

while true; do
  if ! kill -0 "$PID" 2>/dev/null; then
    LAST_CKPT=$(ls -lt $SAVE/*.pt 2>/dev/null | head -1 | awk '{print $6" "$7" "$8" -> "$9}')
    echo "[$(date '+%F %T')] SÜREÇ YOK — son checkpoint: ${LAST_CKPT:-bulunamadı}" | tee -a $LOG
    echo "=== [$(date)] MONITOR: süreç düştü, izleme durduruldu ===" | tee -a $LOG
    break
  fi
  RSS=$(ps -o rss= -p "$PID" 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
  SWAP=$(sysctl -n vm.swapusage 2>/dev/null)
  EPOCH=$(grep -E "Epoch +[0-9]+/20" $PIPELOG 2>/dev/null | tail -1 | sed 's/^ *//')
  CKPT=$(ls -lt $SAVE/*.pt 2>/dev/null | head -1 | awk '{print $6" "$7" "$8" -> "$9}')
  echo "[$(date '+%F %T')] alive | RSS ${RSS}GB | ${SWAP} | ${EPOCH:-epoch satırı yok} | ckpt: ${CKPT:-yok}" | tee -a $LOG
  sleep $INTERVAL
done
