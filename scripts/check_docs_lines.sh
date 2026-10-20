#!/bin/bash
# Doküman satır bütçesi (AGENTS.md §3.7): tüm *.md ≤200 satır.
# İstisna (formal teslimat): docs/patent/**, docs/paper_draft.md
# _archive/ hariç (bakımsız içerik).
cd "$(dirname "$0")/.." || exit 2
max=200
fail=0
while IFS= read -r f; do
  n=$(wc -l < "$f")
  if [ "$n" -gt "$max" ]; then
    echo "OVER $max: $f ($n lines)"
    fail=1
  fi
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
echo "OK: tüm *.md ≤ $max satır"
