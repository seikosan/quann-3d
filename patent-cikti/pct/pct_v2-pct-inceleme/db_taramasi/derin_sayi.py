#!/usr/bin/env python3
"""Derin incelemeye inilen kayit sayisinin ham uretecisi (KANIT_derin_kayit_sayisi.txt).

Sayi elle yazilmaz: patentlar/ dizinindeki ham sayfa dosyalari sayilir.
Kosum: python3 derin_sayi.py   (bu dizinden)
"""
from __future__ import annotations

from pathlib import Path

Dizin = Path(__file__).resolve().parent


def main() -> None:
    sayifalar = sorted((Dizin / "patentlar").glob("*.md"))
    adlar = [p.stem for p in sayifalar]
    # '## ' basliklari; bos basliklar (dilimleme artigi) sayilmaz
    ham_istem = sorted(
        t for t in (l[3:].strip() for l in
                    (Dizin / "DERIN_kayit.md").read_text(encoding="utf-8", errors="replace").splitlines()
                    if l.startswith("## ")) if t
    )
    satirlar = [
        "# KANIT — derin inceleme kayit sayisi (uretim: derin_sayi.py)",
        "# komut: sorted(patentlar/*.md) ve DERIN_kayit.md '## ' basliklari",
        "# tarih: 2026-11-19",
        f"ham_sayfa_dosyasi | {len(sayifalar)} | patentlar/*.md",
        f"derin_kayit_basligi | {len(ham_istem)} | DERIN_kayit.md '## ' (bos '## ' sayilmadi)",
        "yayin_nolari | " + " · ".join(adlar),
        "derin_kayit_nolari | " + " · ".join(ham_istem),
    ]
    out = Dizin / "KANIT_derin_kayit_sayisi.txt"
    out.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print("\n".join(satirlar[:5]))
    print("yazildi:", out)


if __name__ == "__main__":
    main()
