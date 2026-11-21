#!/usr/bin/env bash
# KAPI_OUT_2026-11-20.txt üretici — makine kapılarının ham komut + ham çıktısı.
# Çıkış kodları PIPESTATUS ile alınır (bir boru hattındaki `tail` çıkış kodu MASKELENMEZ).
# Kosum: bash uretim_kapi.py yerine bash uretim_kapi.sh   (bu dizinden)
set -u
cd "$(dirname "$0")/../../../.." || exit 3   # depo kökü
D=patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20
OUT=$D/KAPI_OUT_2026-11-20.txt
{
  echo "# KAPI_OUT_2026-11-20 — bu koşumun makine kapısı çıktıları (ham komut + ham çıktı)"
  echo
  echo "## 1) rapor sayı denetçisi (bu koşumun üç çıktısı)"
  echo "komut: python3 ~/.config/opencode/denetim/rapor_denetle.py <INCELEME_RAPORU|MOTOR_KAYITLARI|METRIKLER>_2026-11-20.md"
  python3 ~/.config/opencode/denetim/rapor_denetle.py \
    "$D/INCELEME_RAPORU_2026-11-20.md" "$D/MOTOR_KAYITLARI_2026-11-20.md" "$D/METRIKLER_2026-11-20.md"
  echo "RAPOR_DENETMEN_CIKIS=${PIPESTATUS[0]}"
  echo
  echo "## 2) küresel denetim hattı (hepsi) — yalnız ÖZET satırı + kayıt yolu"
  echo "komut: python3 ~/.config/opencode/denetim/denetim_kos.py hepsi"
  python3 ~/.config/opencode/denetim/denetim_kos.py hepsi > /tmp/hepsi_$$.txt 2>&1
  echo "KURESEL_CIKIS=$?"
  grep -E "^ÖZET|^kayıt" /tmp/hepsi_$$.txt
  echo
  echo "### Bu depoya/patent dosyalarına ait bulgu satırları (tam liste)"
  grep -E "^  ✗" /tmp/hepsi_$$.txt | grep -E "quann-3d|patent-cikti|pct_v2" || echo "(bu depoya ait BULGU satırı yok)"
  echo
  echo "### BULGU satırlarının depo dağılımı (tam liste)"
  grep -E "^  ✗" /tmp/hepsi_$$.txt | sed -E 's/.*← //' | sort | uniq -c | sort -rn
  rm -f /tmp/hepsi_$$.txt
} > "$OUT" 2>&1
echo "yazildi: $OUT"
