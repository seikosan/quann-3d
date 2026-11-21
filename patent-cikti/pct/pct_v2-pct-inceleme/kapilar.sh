#!/bin/bash
# pct_v2 paketini yeniden üretir ve DOKUZ kapidan gecirir; ALTI tanesi KARAR kapisi:
#  0) sekil hatti          -> SVG uretimi + A4 PDF derleme + harici sekil kapisi [KARAR]
#     a) generate_figures_pct.py  kendi ic kapilari: kutu-sigma, carisma, 0,21 cm
#     b) build_drawings.py        SVG -> gri PNG -> docx -> PDF; KABUL: 8 sayfa A4
#     c) dogrul_figurler.py       SVG x kunye x tarifname x ham olcu JSON x PDF (dis baglam)
#  1) make_filing.py       -> filing kopyalari + paragraf/istem safligi      [KARAR]
#  2) patentline lint      -> yerel kural hatti (oncul dayanak, bolge, sozluk) [bilgi]
#  3) run_pct_analizi.py   -> patent-creator motor bataryasi (PCT/EPO/US/paket) [bilgi]
#  4) kanit_artefaktlar.py + kanit_antecedent.py -> bulgularin siniflandirma kaniti [bilgi]
#  5) atif_haritasi.py     -> her sayinin gectigi ham satiri                  [KARAR]
#  6) rapor_denetle.py     -> UYGUNLUK_NOTU + OLCUM_KAYITLARI sayi-atif kapisi [KARAR]
#  7) dogrul_tablolar.py   -> tablo icindeki her kunye + motor degeri         [KARAR]
#  8) dogrul_olcum.py      -> OLCUM_KAYITLARI: her kunye + her sayi ham kaynakta [KARAR]
# 2/3/4 bilgi amaclidir: kalan motor/lint bulgularinin neden metinde duzeltilmedigi
# UYGUNLUK_NOTU.md paragraf 4'te (artefakt kanitlariyla) yazilidir.
# Ham ciktilar patent-cikti/pct/pct_v2-pct-inceleme/ altina yazilir.
set -u
cd /Users/serkankose/Documents/quann-3d || exit 3
OUT=patent-cikti/pct/pct_v2-pct-inceleme
mkdir -p "$OUT"

echo "=== 0/9 sekil hatti ==="
python3 docs/patent/pct_v2/generate_figures_pct.py > "$OUT/KANIT_kutu_sigma_yesil.txt" 2>&1
SEKIL_URET=$?
echo "sekil uretim cikis=$SEKIL_URET  $(tail -1 "$OUT/KANIT_kutu_sigma_yesil.txt")"
python3 docs/patent/pct_v2/build_drawings.py > "$OUT/build_drawings_konsol.txt" 2>&1
SEKIL_BUILD=$?
echo "sekil build cikis=$SEKIL_BUILD  $(grep -E '^KABUL|^RED' "$OUT/build_drawings_konsol.txt" | tail -1)"
python3 "$OUT/dogrul_figurler.py" > "$OUT/dogrul_figurler_konsol.txt" 2>&1
SEKIL=$?
echo "sekil kapisi cikis=$SEKIL  $(tail -1 "$OUT/dogrul_figurler_konsol.txt")"

echo "=== 1/9 filing kopyalari ==="
python3 docs/patent/pct_v2/make_filing.py > "$OUT/uretim.json" 2>&1
python3 - <<'PY'
import json
import sys
d = json.load(open("patent-cikti/pct/pct_v2-pct-inceleme/uretim.json", encoding="utf-8"))
print("uretim_ok:", d["ok"])
sorun = 0
for f in d["files"]:
    print(f"  {f['file']:38} bayt={f['bytes']:6} istem={f['claim_count']:2} "
          f"paragraf={f['paragraph_count']:2} surekli={f['paragraph_gapless']} sorun={f['problems']}")
    sorun += len(f["problems"])
if not d["ok"] or sorun or len(d["files"]) != 6:
    print(f"URETIM KIRMIZI: ok={d['ok']} sorun={sorun} dosya={len(d['files'])}")
    sys.exit(1)
PY
URETIM=$?

echo "=== 2/9 patentline lint ==="
printf '%s\n' \
  "DESCRIPTION_EN.md en" "CLAIMS_EN.md en" "DESCRIPTION_TR.md tr" "ISTEMLER_TR.md tr" \
  "filing/DESCRIPTION_FILING-COPY_EN.txt en" "filing/CLAIMS_FILING-COPY_EN.txt en" \
  "filing/TARIFNAME_FILING-COPY_TR.txt tr" "filing/ISTEMLER_FILING-COPY_TR.txt tr" |
while read -r f loc; do
  base=$(echo "$f" | tr '/.' '__')
  /Users/serkankose/Documents/patent-hatti/.venv/bin/python -m patentline lint \
      "docs/patent/pct_v2/$f" --locale "$loc" > "$OUT/lint_$base.txt" 2>&1
  printf '%-38s cikis=%s  %s\n' "$f" "$?" "$(head -1 "$OUT/lint_$base.txt")"
done

echo "=== 3/9 motor bataryasi ==="
/Users/serkankose/patent-creator-venv/bin/python "$OUT/run_pct_analizi.py" > "$OUT/konsola_cikti.txt" 2>&1
echo "batarya cikis=$?"

echo "=== 4/9 siniflandirma kaniti ==="
/Users/serkankose/patent-creator-venv/bin/python "$OUT/kanit_artefaktlar.py" > "$OUT/KANIT_artefaktlar_konsol.txt" 2>&1
echo "arte fakt kanit cikis=$?"
python3 "$OUT/kanit_antecedent.py" > "$OUT/KANIT_antecedent_konsol.txt" 2>&1
echo "antecedent kanit cikis=$?"
python3 - <<'PY'
import json
for row in json.load(open("patent-cikti/pct/pct_v2-pct-inceleme/KANIT_antecedent.json", encoding="utf-8")):
    print(f"{row['dosya']:44} istem={row['istem_sayisi']} bulgu={row['toplam_bulgu']:2} {row['ozet']}")
PY

echo "=== 5/9 sayi-atif haritasi ==="
python3 "$OUT/atif_haritasi.py" > "$OUT/atif_haritasi_konsol.txt" 2>&1
ATIF=$?
echo "atif haritasi cikis=$ATIF  $(tail -1 "$OUT/atif_haritasi_konsol.txt")"

echo "=== 6/9 rapor kapisi (sayi -> ham kaynak) ==="
python3 "$HOME/.config/opencode/denetim/rapor_denetle.py" \
    docs/patent/pct_v2/UYGUNLUK_NOTU.md docs/patent/pct_v2/OLCUM_KAYITLARI.md \
    docs/patent/pct_v2/BASVURU_PAKETI.md docs/patent/pct_v2/ON_SANAT.md \
    > "$OUT/rapor_kapisi.txt" 2>&1
KAPI=$?
echo "rapor kapisi cikis=$KAPI  $(grep '^TOPLAM' "$OUT/rapor_kapisi.txt")"

echo "=== 7/9 tablo ici kunye + motor deger denetimi ==="
python3 "$OUT/dogrul_tablolar.py" > "$OUT/dogrul_tablolar_konsol.txt" 2>&1
TABLO=$?
head -3 "$OUT/dogrul_tablolar_konsol.txt"

echo "=== 8/9 olcum defteri kapisi (OLCUM_KAYITLARI) ==="
python3 "$OUT/dogrul_olcum.py" > "$OUT/dogrul_olcum_konsol.txt" 2>&1
OLCUMKAP=$?
head -6 "$OUT/dogrul_olcum_konsol.txt"

echo "--- KAPI KARARLARI ---"
echo "0 sekil hatti   : $SEKIL_URET/$SEKIL_BUILD/$SEKIL  (0/0/0 = uretim + A4 PDF + harici kapi)"
echo "1 uretim        : $URETIM    (0 = 6 kopya, paragraf/istem/iz temiz)"
echo "5 atif haritasi : $ATIF      (0 = her girdi kendi ham satirinda)"
echo "6 rapor kapisi  : $KAPI      (0 temiz / 1 bulgu / 2 yalniz uyari)"
echo "7 tablo kapisi  : $TABLO     (0 = her kunye acildi, motor degerleri uslesmez)"
echo "8 olcum kapisi  : $OLCUMKAP  (0 = her kunye + her sayi ham kaynakta)"
if [ "$KAPI" -ge 3 ]; then echo "SONUC: KULLANIM HATASI"; exit 3; fi
KIRMIZI=0
[ "$SEKIL_URET" -ne 0 ] && KIRMIZI=1
[ "$SEKIL_BUILD" -ne 0 ] && KIRMIZI=1
[ "$SEKIL" -ne 0 ] && KIRMIZI=1
[ "$URETIM" -ne 0 ] && KIRMIZI=1
[ "$ATIF" -ne 0 ] && KIRMIZI=1
[ "$KAPI" -eq 1 ] && KIRMIZI=1
[ "$TABLO" -ne 0 ] && KIRMIZI=1
[ "$OLCUMKAP" -ne 0 ] && KIRMIZI=1
if [ "$KIRMIZI" -eq 1 ]; then
  echo "SONUC: KIRMIZI — en az bir karar kapisi takildi (ustteki konsol kayitlarina bak)"
  exit 1
fi
echo "SONUC: yeşil — altı karar kapısı temiz (şekil + filing + atıf + rapor + tablo + ölçüm); kalan motor/lint bulguları UYGUNLUK_NOTU paragraf 4'te"
