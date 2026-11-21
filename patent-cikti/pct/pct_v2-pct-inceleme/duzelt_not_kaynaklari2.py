r"""UYGUNLUK_NOTU'nda kalan TEK kapı bulgusunu (kaynakta yok) kapatır + §3 atıflarını
sayının geçtiği satıra taşır. Beklenen tek eşleşmeyle uygulanır.

Koşum: python3 duzelt_not_kaynaklari2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

NOT = Path("/Users/serkankose/Documents/quann-3d/docs/patent/pct_v2/UYGUNLUK_NOTU.md")
PC = "patent-cikti/pct/pct_v2-pct-inceleme"
CE = "docs/patent/pct_v2/CLAIMS_EN.md"
DE = "docs/patent/pct_v2/DESCRIPTION_EN.md"

EDITS: list[tuple[str, str]] = [
    # Karar: iki kısaltılmış atıf tam yola çevrilir (kapı `:10` biçimini dosya adıyla tanımıyordu)
    (f"""(`{PC}/H_pkg_full.json:10` · `I_pkg_filing.json:10`) ve US istem
skoru (`{PC}/J_us_claims_en.json:4` ve `:10`) tam.""",
     f"""(`{PC}/H_pkg_full.json:10` ve `{PC}/I_pkg_filing.json:10`) ile US
istem skoru (`{PC}/J_us_claims_en.json:4` ve `{PC}/J_us_claims_en.json:10`) tam."""),
    # madde 1: sayıların geçtiği satıra atıf
    (f"""1. EN bağımsız istem 1/13/15 iki parçalı biçime getirildi ("characterized in that"); TR 1/13/15
   ön yükleri önceki tekniği anlatan gövde + "karakterize edilir" olarak yeniden yazıldı.""",
     f"""1. EN bağımsız istem 1/13/15 iki parçalı biçime getirildi ("characterized in that"; satırları
   `{CE}:10`, `{CE}:52`, `{CE}:66`); TR karşılıklarının ön yükleri önceki tekniği anlatan gövde +
   "karakterize edilir" olarak yeniden yazıldı."""),
    # madde 2-5: atıf, sayının geçtiği satıra alınır
    (f"""2. [0063] "0.64 on 25 critical records" → "0.64 on 25 records of the critical stage"
   (`{DE}:199`; TR aynası aynı satır numarasında) → H/I""",
     f"""2. [0063] ({DE}:199) "0.64 on 25 critical records" → "0.64 on 25 records of the critical
   stage" (TR aynası aynı satır numarasında) → H/I"""),
    (f"""3. [0065]'e istem on'un terimi eklendi: "a text consisting of a plurality of sentences"
   (`{DE}:207`; TR aynası) → F'nin CRITICAL'lerinden biri kalktı (§2 F).""",
     f"""3. [0065] ({DE}:207) istem on'un terimini aldı: "a text consisting of a plurality of
   sentences" (TR aynası) → F'nin CRITICAL'lerinden biri kalktı (§2 F)."""),
    (f"""4. [0050]'ye istem on dört'ün terimi eklendi: "a distribution store holding" (TR: "bir dağıtım
   deposu taşır", `{DE}:159`) → F'nin ikinci CRITICAL'i kalktı (§2 F).""",
     f"""4. [0050] ({DE}:159) istem on dört'ün terimini aldı: "a distribution store holding"
   (TR: "bir dağıtım deposu taşır") → F'nin ikinci CRITICAL'i kalktı (§2 F)."""),
    (f"""5. [0034] istem dört'ün terimleriyle genişletildi: "a result of the max-pooling / a result of
   the average-pooling" (`{DE}:105`; TR aynası) → F kritikleri""",
     f"""5. [0034] ({DE}:105) istem dört'ün terimleriyle genişletildi: "a result of the max-pooling /
   a result of the average-pooling" (TR aynası) → F kritikleri"""),
]


def main() -> int:
    text = NOT.read_text(encoding="utf-8")
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"beklenen 1 eşleşme, bulunan {n}: {old[:70]!r}")
        text = text.replace(old, new)
        print(f"OK: {old[:52]!r}")
    NOT.write_text(text, encoding="utf-8")
    assert f"`I_pkg_filing.json:10`" not in text, "kısaltılmış atıf duruyor"
    print(f"\nSAGLAMA OK: {len(text.splitlines())} satır")
    return 0


if __name__ == "__main__":
    sys.exit(main())
