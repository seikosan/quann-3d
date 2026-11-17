#!/bin/bash
# SSD yedeği (harici disk): tam git geçmişi (bundle) + repo'nun tamamı (tar,
# gitignored artifact'lar dahil). 1 jenerasyon rotasyon: önceki tar -> .prev.
# AGENTS.md §3.4 konvansiyonu: repo kökü, append-only log, adım başlıkları.
#
# Kullanım:
#   bash scripts/backup_ssd.sh              # tek koş
#   bash scripts/backup_ssd.sh --install    # launchd ile her gün 03:00
#   bash scripts/backup_ssd.sh --uninstall  # launchd görevini kaldır
# Ortam: BACKUP_VOLUME (varsayılan /Volumes/x)
set -euo pipefail
cd "$(dirname "$0")/.."

VOLUME="${BACKUP_VOLUME:-/Volumes/x}"
DEST="$VOLUME/backups/quann-3d"
LOG="logs/backup_ssd.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

case "${1:-}" in
  --install)
    plist="$HOME/Library/LaunchAgents/com.quann3d.backup-ssd.plist"
    mkdir -p "$(dirname "$plist")"
    cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.quann3d.backup-ssd</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$PWD/scripts/backup_ssd.sh</string>
  </array>
  <key>WorkingDirectory</key><string>$PWD</string>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>3</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>$PWD/logs/backup_ssd_launchd.log</string>
  <key>StandardErrorPath</key><string>$PWD/logs/backup_ssd_launchd.log</string>
  <key>EnvironmentVariables</key>
  <dict><key>BACKUP_VOLUME</key><string>$VOLUME</string></dict>
</dict>
</plist>
EOF
    launchctl bootout "gui/$(id -u)" "$plist" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$plist"
    echo "=== [$(ts)] launchd görevi kuruldu: $plist (günlük 03:00) ==="
    exit 0
    ;;
  --uninstall)
    plist="$HOME/Library/LaunchAgents/com.quann3d.backup-ssd.plist"
    launchctl bootout "gui/$(id -u)" "$plist" 2>/dev/null || true
    rm -f "$plist"
    echo "=== [$(ts)] launchd görevi kaldırıldı ==="
    exit 0
    ;;
esac

BUNDLE="$DEST/quann-3d.bundle"
TAR="$DEST/quann-3d.tar"

echo "=== [$(ts)] 1/3: hedef kontrol ($VOLUME) ==="
if ! mount | grep -q " on $VOLUME "; then
  echo "HATA: $VOLUME mountlu değil — SSD'yi tak, koş tekrar dene. (log: $LOG)"
  exit 2
fi
mkdir -p "$DEST"
avail_kb=$(df -Pk "$DEST" | awk 'NR==2 {print $4}')
if [ "$avail_kb" -lt 15360000 ]; then
  echo "HATA: $VOLUME'da <15GB boş — yedek yapılamaz ($((avail_kb/1024)) MB boş)."
  exit 3
fi

echo "=== [$(ts)] 2/3: git bundle (tam geçmiş) ==="
git bundle create "$BUNDLE" --all
git bundle verify "$BUNDLE" | grep -q "complete history" || { echo "HATA: bundle doğrulaması başarısız"; exit 4; }

echo "=== [$(ts)] 3/3: tam repo tar (artifact'lar dahil) ==="
[ -f "$TAR" ] && mv -f "$TAR" "${TAR}.prev"
tar -cf "$TAR" \
  --exclude 'quann-3d/.git' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'node_modules' \
  --exclude '.opencode' \
  --exclude '.gstack' \
  --exclude '.DS_Store' \
  -C "$(dirname "$PWD")" quann-3d
tar -tf "$TAR" > /dev/null   # read-back bütünlük
head -5 "$TAR" > /dev/null 2>&1 || { echo "HATA: tar okunamadı"; exit 5; }

size=$(du -sh "$TAR" | awk '{print $1}')
heads=$(git log --oneline -1 | awk '{print $1}')
echo "=== [$(ts)] OK: bundle $(du -sh "$BUNDLE" | awk '{print $1}') + tar $size (HEAD $heads) -> $DEST ==="
cat "$DEST/README.md" 2>/dev/null | head -3 || echo "not: $DEST/README.md yok"
