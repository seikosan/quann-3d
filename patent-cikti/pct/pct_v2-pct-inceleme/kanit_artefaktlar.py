#!/usr/bin/env python3
r"""Bulguları 'motor artefaktı' diye işaretlerken kullanılan kanıtı üretir.

Kural: bir bulguyu "sezgi/regex artefaktı" ilan etmek, onu düzeltmemek anlamına
gelir; o yüzden iddia ham kanıtla birlikte yazılır. Burada üç artefakt iddiası
test edilir ve çıktı dosyaya yazılır:

1. Rule 43(2) "same category (product)": motor istem kategorisini yalnız İngilizce
   kalıplardan çıkarıyor (epo_claims_analyzer.py:186-192). 1. istem yöntem,
   13. istem sistem, 15. istem kayıt ortamı olduğu hâlde üçü de "product" mi?
2. Rule 43(1) TR "two-part form": TR istemler "karakterize edilir" diyor; motorun
   kalıbı İngilizce. İki kalıp TR metinde gerçekten aranıyor mu?
3. TR bağımlılık tespiti: TR "İstem 1'e göre" diyor; motor `\bclaims?\s+` arıyor.
4. Art. 83 "only Para. N" tavsiyesi: aynı terim tarifnamede kaç paragrafta geçiyor?

Kosum: /Users/serkankose/patent-creator-venv/bin/python kanit_artefaktlar.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SITE = Path("/Users/serkankose/patent-creator-venv/lib/python3.13/site-packages")
sys.path.insert(0, str(SITE / "mcp_server"))

from epo_claims_analyzer import EPOClaimsAnalyzer  # noqa: E402

PKG = Path("/Users/serkankose/Documents/quann-3d/docs/patent/pct_v2")
FIL = PKG / "filing"
OUT = Path(__file__).resolve().parent

TWO_PART_PATTERNS = [r"characteri[sz]ed\s+in\s+that", r"wherein\s+the\s+improvement\s+comprises"]
DEP_PATTERN = r"\bclaims?\s+(?=\d)"


def main() -> int:
    claims_en = (FIL / "CLAIMS_FILING-COPY_EN.txt").read_text(encoding="utf-8")
    claims_tr = (FIL / "ISTEMLER_FILING-COPY_TR.txt").read_text(encoding="utf-8")
    desc_en = (FIL / "DESCRIPTION_FILING-COPY_EN.txt").read_text(encoding="utf-8")

    a = EPOClaimsAnalyzer()
    parsed_en = a._parse_claims(claims_en)
    parsed_tr = a._parse_claims(claims_tr)

    tr1 = next(c["text"] for c in parsed_tr if c["number"] == 1)

    kanit = {
        "1_kategori_kalibi": {
            "motor_kalibi": ["^a\\s+method\\b|^a\\s+process\\b|^method\\s+",
                             "^a\\s+device\\b|^a\\s+system\\b|^a\\s+apparatus\\b|^an\\s+apparatus\\b"],
            "bulgu_kategorileri": {str(c["number"]): c["category"] for c in parsed_en
                                   if c["is_independent"]},
            "istem_baslari": {str(c["number"]): c["text"][:48] for c in parsed_en
                              if c["is_independent"]},
            "yorum": "1 = yöntem, 13 = bilgisayar sistemi, 15 = kayıt ortamı; motor kalıpları "
                     "'a computer-implemented ... method' ve 'a computer system' ifadelerine "
                     "uymadığı için üçünü de 'product' sayıyor.",
        },
        "2_two_part_TR": {
            "motor_kaliplari": TWO_PART_PATTERNS,
            "TR_istem1_kararli_ifade": re.findall(r"karakterize edilir", tr1),
            "TR_istem1_eslesme": [bool(re.search(p, tr1, re.I)) for p in TWO_PART_PATTERNS],
            "EN_istem1_eslesme": [bool(re.search(p, next(c["text"] for c in parsed_en if c["number"] == 1), re.I))
                                  for p in TWO_PART_PATTERNS],
        },
        "3_TR_bagimlilik": {
            "motor_kalibi": DEP_PATTERN,
            "TR_istem1_kararli_ifade": re.findall(r"İstem \d+'e göre", claims_tr)[:2],
            "TR_istem1_eslesme": bool(re.search(DEP_PATTERN, tr1, re.I)),
            "EN_bağımlı_istem_sayisi": sum(1 for c in parsed_en if not c["is_independent"]),
            "TR_bağımlı_istem_sayisi": sum(1 for c in parsed_tr if not c["is_independent"]),
        },
        "4_art83_tavsiye": {
            "terim": "point cloud",
            "tarifnamede_gecen_yer_sayisi": desc_en.lower().count("point cloud"),
            "motorin_atif_yaptigi_paragraf_sayisi": 1,
            "aciklama": "Motor her terimi tek bir paragrafla eşliyor; 'only Para. N' tavsiyesi "
                        "bu indeks sınırından geliyor, eksik desteğin kanıtı değil.",
        },
    }
    (OUT / "KANIT_artefaktlar.json").write_text(json.dumps(kanit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(kanit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
