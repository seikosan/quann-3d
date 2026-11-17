#!/bin/bash
# Resume-only host RSS/free-RAM monitor; logs append-only at 10-minute intervals.
cd "$(dirname "$0")/../.."  # repo root
QUEUE_PID=${1:?queue PID required}
INTERVAL=${2:-600}
LOG=logs/monitor_wiki1500k_resume.log

echo "=== [$(date)] 1.5M resume monitor started: queue PID=$QUEUE_PID interval=${INTERVAL}s ===" | tee -a "$LOG"
TRAIN_PID=
while kill -0 "$QUEUE_PID" 2>/dev/null && [ -z "$TRAIN_PID" ]; do
  TRAIN_PID=$(ps -axo pid=,command= | awk '/train_wiki.py --step train --corpus data\/turkish_wiki_1500k.txt/ && !/awk/ {print $1; exit}')
  [ -n "$TRAIN_PID" ] || sleep 2
done

if [ -z "$TRAIN_PID" ]; then
  echo "[$(date '+%F %T')] 1.5M resume process was not observed; monitor stopped." | tee -a "$LOG"
  exit 0
fi

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  RSS=$(ps -o rss= -p "$TRAIN_PID" 2>/dev/null | awk '{printf "%.2f", $1/1048576}')
  FREE_GB=$(vm_stat | awk -F: '/Pages free/ {gsub(/[^0-9]/, "", $2); f=$2} /Pages speculative/ {gsub(/[^0-9]/, "", $2); s=$2} END {printf "%.1f", (f+s)*16384/1073741824}')
  SWAP=$(sysctl -n vm.swapusage 2>/dev/null)
  EPOCH=$(grep -E 'Epoch +[0-9]+/20' logs/run_wiki_scale_gate.log 2>/dev/null | tail -1 | sed 's/^ *//')
  echo "[$(date '+%F %T')] alive | train PID=$TRAIN_PID | RSS ${RSS:-unknown} GiB | free+speculative ${FREE_GB} GiB | ${SWAP} | ${EPOCH:-epoch line not emitted yet}" | tee -a "$LOG"
  sleep "$INTERVAL"
done
echo "[$(date '+%F %T')] 1.5M resume process stopped; monitor complete." | tee -a "$LOG"
