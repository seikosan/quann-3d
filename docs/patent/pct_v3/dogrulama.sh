#!/usr/bin/env bash
# pct_v3 yapısal özdenetim — KURULUM_PLAN.md §6 kapı 1.
# Ham çıktı: patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt (yönlendirerek koşun):
#   bash docs/patent/pct_v3/dogrulama.sh > patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt 2>&1
# Kural: her ölçüm komutuyla birlikte basılır; beklenen değerin sapması exit koduna girer.
set -u
cd "$(dirname "$0")/../../.." || exit 3   # depo kökü (docs/patent/pct_v3 -> kök)
F=docs/patent/pct_v3/filing
FAIL=0
chk() { # chk <beklenen> <gerçek> <ad>
  if [ "$1" = "$2" ]; then echo "OK   ${3}: $2"; else echo "KIRIK ${3}: beklenen=$1 gerçek=$2"; FAIL=1; fi
}

echo "== istem sayaçları =="
chk 15 "$(grep -c -E '^[0-9]+\.' $F/CLAIMS_FILING-COPY_EN.txt)" "EN istem"
chk 3 "$(grep -E '^[0-9]+\.' $F/CLAIMS_FILING-COPY_EN.txt | grep -vc 'of claim')" "EN bağımsız"
chk 15 "$(grep -c -E '^[0-9]+\.' $F/ISTEMLER_FILING-COPY_TR.txt)" "TR istem"
chk 12 "$(grep -c -E '^[0-9]+\. .*İstem [0-9]+' $F/ISTEMLER_FILING-COPY_TR.txt)" "TR bağımlı (İstem N'e göre)"
chk 0 "$(grep -ci -w claim $F/ISTEMLER_FILING-COPY_TR.txt)" "TR'de İngilizce claim"
chk 0 "$(grep -cE '\b[0-9]{3}\b' $F/CLAIMS_FILING-COPY_EN.txt)" "EN istemde 3 basamaklı işaret (K13.4)"
chk 0 "$(grep -cE '\b[0-9]{3}\b' $F/ISTEMLER_FILING-COPY_TR.txt)" "TR istemde 3 basamaklı işaret"

echo "== tarifname paragraf numaraları (kesintisiz, tekil, EN=TR) =="
python3 - "$F" <<'PY'
import re, sys
from pathlib import Path
f = Path(sys.argv[1]); out = []
for name in ("DESCRIPTION_FILING-COPY_EN.txt", "TARIFNAME_FILING-COPY_TR.txt"):
    # yalnız satırbaşı paragraf numaraları; gövde içi [00NN] çapraz atıflar sayılmaz
    nums = [int(m) for m in re.findall(r"(?m)^\[(\d{4})\]", (f / name).read_text(encoding="utf-8"))]
    ok_seq = nums == list(range(1, len(nums) + 1))
    dup = len(nums) != len(set(nums))
    print(f"{'OK ' if ok_seq and not dup else 'KIRIK'} {name}: {len(nums)} paragraf, "
          f"ilk={nums[0]:05d} son={nums[-1]:05d} kesintisiz={ok_seq} tekil={not dup}")
    out.append(len(nums))
print(("OK " if out[0] == out[1] else "KIRIK") + f" EN=TR paragraf sayısı {out}")
PY

echo "== referans işaretleri: tablonun tamamı EN tarifnamede =="
python3 - "$F" <<'PY'
import re, sys
from pathlib import Path
tab = Path("docs/patent/pct_v3/REFERANS_NUMARALARI.md").read_text(encoding="utf-8")
nums = sorted({int(m) for m in re.findall(r"^\| (\d{3}) \|", tab, re.M)})
en = (Path(sys.argv[1]) / "DESCRIPTION_FILING-COPY_EN.txt").read_text(encoding="utf-8")
oz = (Path(sys.argv[1]) / "ABSTRACT_FILING-COPY_EN.txt").read_text(encoding="utf-8")
eksik = [n for n in nums if f"({n})" not in en]
print(f"{'OK ' if not eksik else 'KIRIK'} tarifname kapsama: {len(nums) - len(eksik)}/{len(nums)} eksik={eksik}")
print(f"{'OK ' if re.search(r'\(\d{3}\)', oz) else 'KIRIK'} özetde parantezli işaret var (Rule 47(4)/K8.1)")
PY

echo "== özet kelime sayıları (PCT K8.1: ≤150) =="
for f in $F/ABSTRACT_FILING-COPY_EN.txt $F/OZET_FILING-COPY_TR.txt; do
  n=$(wc -w < $f | tr -d ' '); [ "$n" -le 150 ] && echo "OK   $f: $n kelime" || { echo "KIRIK $f: $n kelime"; FAIL=1; }
done

echo "== şekil PDF'i =="
PDF=$F/SEKILLER_FILING-COPY.pdf
if [ -f "$PDF" ]; then
  P=$(pdfinfo "$PDF" | awk '/^Pages:/{print $2}')
  chk 8 "$P" "PDF sayfa"
  NREF=$(python3 - <<'PY'
import re, glob
tab = open("docs/patent/pct_v3/REFERANS_NUMARALARI.md", encoding="utf-8").read()
nums = sorted({m for m in re.findall(r"^\| (\d{3}) \|", tab, re.M)})
blob = " ".join(" ".join(re.findall(r">([^<>]+)</text>",
               open(p, encoding="utf-8").read()))
               for p in sorted(glob.glob("docs/patent/pct_v3/fig/fig*.svg")))
eks = [n for n in nums if not re.search(rf"(?<![0-9A-Za-z]){n}(?![0-9])", blob)]
print(len(eks))
PY
)
  chk 0 "$NREF" "tabloda çizilmeyen numara"
else
  echo "KIRIK PDF yok"; FAIL=1
fi

echo "== eşikler/ölçümlü değer koruması =="
for pat in '0\.9451' '0\.9365' '0\.9367' '528\.9 KB' '138\.2 KB'; do
  a=$(grep -o -E "$pat" docs/patent/pct_v2/fig/fig[67]*.svg | wc -l | tr -d ' ')
  b=$(grep -o -E "$pat" docs/patent/pct_v3/fig/fig[67]*.svg | wc -l | tr -d ' ')
  chk "$a" "$b" "ölçüm $pat pct_v2=pct_v3"
done

echo "== pct_v2 dokunulmazlığı (sha1, DEGISIKLIK_KAYDI_SEKIL.md §4'teki referansla kıyaslayın) =="
shasum docs/patent/pct_v2/filing/* docs/patent/pct_v2/fig/fig_manifest.json

exit $FAIL
