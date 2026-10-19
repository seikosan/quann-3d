#!/bin/bash
# Doküman satır bütçesi (AGENTS.md §3.7): tüm *.md ≤200 satır.
# İstisna (formal teslimat): docs/patent/**, docs/paper_draft.md
# _archive/ hariç (bakımsız içerik).
cd "$(dirname "$0")/.." || exit 2
max=200
fail=0
while IFS= read -r f; do
  n=$(wc -l < "$f")
  [ "$n" -le "$max" ] && continue
  # Grandfather listesi (geçici — RESEARCH.md "Sıradaki Aksiyonlar"):
  # Ağ geri gelince archiver (2 patent → _archive/) + cleaner (4 dosya → ≤200)
  # çalıştırılıp bu listeden satırlar silinecek.
  case "$f" in
    ./fraud/README.md|\
    ./quann_cloud/PROJECT_SUMMARY.md|\
    ./quann_cloud/PATENT_V2.md|\
    ./quann_cloud/PATENT_QUANN_CLOUD.md|\
    ./quann_cloud/FINDINGS.md|\
    ./quann_cloud/DESIGN.md)
      echo "GRANDFATHERED (distil/arşiv kuyruğunda): $f ($n lines)"
      ;;
    *)
      echo "OVER $max: $f ($n lines)"
      fail=1
      ;;
  esac
done < <(find . -name "*.md" \
  -not -path "./.git/*" \
  -not -path "./.opencode/node_modules/*" \
  -not -path "./_archive/*" \
  -not -path "./docs/patent/*" \
  -not -path "./docs/paper_draft.md")
if [ "$fail" -eq 1 ]; then
  echo "FAIL: doküman satır bütçesi aşıldı (AGENTS.md §3.7)"
  exit 1
fi
echo "OK: yeni ihlal yok (grandfather listesi ayrı gösterilir)"
