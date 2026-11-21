#!/bin/bash
# Talep formu (PCT/RO/101) eksik alan kaniti: BASVURU_PAKETI.md kendi satirlarini
# yukari tasimasin diye o dosya grep'ten cikarilir (--exclude). Bos cikti = eksik.
cd "$(dirname "$0")/../../.." || exit 3
D=docs/patent/pct_v2
run() {
  printf '## %s\n' "$1"
  printf '$ %s\n' "$2"
  eval "$2"
  printf 'CIKIS=%s (1 = hic karsilasmadi = eksik)\n\n' "$?"
}
run "1) applicant / basvuru sahibi (BASVURU_PAKETI.md hariç)" \
    "grep -rniE 'applicant|başvuru sahibi' $D/*.md --exclude BASVURU_PAKETI.md"
run "2) mucit / inventor (BASVURU_PAKETI.md hariç)" \
    "grep -rniE 'mucit|inventor' $D/*.md --exclude BASVURU_PAKETI.md"
run "3) arama otoritesi / ISA seçildi (BASVURU_PAKETI.md hariç)" \
    "grep -rniE 'arama otoritesi|ISA seçildi' $D/*.md --exclude BASVURU_PAKETI.md"
run "4) priority / rüçhan (BASVURU_PAKETI.md hariç)" \
    "grep -rniE 'priority|rüçhan' $D/*.md --exclude BASVURU_PAKETI.md"
run "5) ayni terimlerin basilabilir metin nüshalarinda karsiligi" \
    "grep -rniE 'applicant|mucit|priority|rüçhan|arama otoritesi' $D/filing/"
