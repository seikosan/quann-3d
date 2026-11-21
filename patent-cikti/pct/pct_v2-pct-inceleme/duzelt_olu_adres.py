r"""UYGUNLUK_NOTU'ndaki iki ölü adresi düzeltir (2026-11-19) — `dogrul_tablolar.py` bulgusu.

Bulgu 1: §1 Rule 6.2(a) satırı `CLAIMS_EN.md:10-67` diyordu; dosya 66 satır
        (`wc`-eşdeğeri betikte ölçüldü) → gövde 10–66.
Bulgu 2: §7 "EN istem 1 (iii)" satırı `CLAIMS_EN.md:13`'ü gösteriyordu; 13. satır boş
        (dosyada her istem satırının ardı boş). (iii) fıkrasının gerçek satırı 26.
Her iki düzeltme de hedef satırın içeriği OKUNARAK yapılır; betik, yazacağı satırda
beklenen metnin gerçekten geçtiğini doğrular (assertion'lı, tek eşleşmeli).

Kosum: python3 patent-cikti/pct/pct_v2-pct-inceleme/duzelt_olu_adres.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/Users/serkankose/Documents/quann-3d")
NOT = REPO / "docs/patent/pct_v2/UYGUNLUK_NOTU.md"
CL = REPO / "docs/patent/pct_v2/CLAIMS_EN.md"


def satir(dosya: Path, n: int) -> str:
    return dosya.read_text(encoding="utf-8").splitlines()[n - 1]


def main() -> int:
    # --- hedef satırların içeriğini ham dosyadan doğrula --------------------------
    body = CL.read_text(encoding="utf-8").splitlines()
    n_son = len(body)
    assert n_son == 66, f"CLAIMS_EN.md satır sayısı {n_son}, beklenen 66"
    assert body[9].lstrip().startswith("1. A computer-implemented"), body[9][:60]
    assert body[65].lstrip().startswith("15. A non-transitory"), body[65][:60]
    assert "(iii) region shift" in body[25] and "reference data group" in body[25], body[25][:80]

    text = NOT.read_text(encoding="utf-8")
    for old, new in [
        ("`docs/patent/pct_v2/CLAIMS_EN.md:10-67`", f"`docs/patent/pct_v2/CLAIMS_EN.md:10-{n_son}`"),
        ("`docs/patent/pct_v2/CLAIMS_EN.md:13`", "`docs/patent/pct_v2/CLAIMS_EN.md:26`"),
    ]:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"beklenen 1 eşleşme, bulunan {n}: {old!r}")
        text = text.replace(old, new)
        print(f"OK: {old}  ->  {new}")

    NOT.write_text(text, encoding="utf-8")
    assert ":10-67" not in text and "`CLAIMS_EN.md:13`" not in text
    print(f"-> {NOT.relative_to(REPO)} yazıldı ({len(text.splitlines())} satır)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
