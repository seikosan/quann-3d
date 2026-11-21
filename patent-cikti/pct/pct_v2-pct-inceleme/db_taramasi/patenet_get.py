#!/usr/bin/env python3
"""Tek bir patent kaydinin tam metnini r.jina.ai uzerinden cekip diske yazar.

Kullanim: python3 patenet_get.py US12639556B2 [tr]
Cikti   : patentlar/<NO>.md (ham sayfa metni) + ekranda istem/ozet dilimi.
Sayfa buyuk oldugu icin ekrana yalniz secilen dilim basilir; ham hâli diskte kalir.
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "patentlar"
OUT.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"


def metin(no: str, dil: str = "en") -> str:
    hedef = f"https://patents.google.com/patent/{no}/{dil}"
    with urllib.request.urlopen("https://r.jina.ai/" + hedef, timeout=120) as r:  # noqa: S310
        return r.read().decode("utf-8", "replace")


def dilim(metin: str, baslik: str, n: int) -> str:
    i = metin.lower().find(baslik.lower())
    if i < 0:
        return f"(baslik bulunamadi: {baslik})"
    parca = metin[i:i + n]
    return re.sub(r"\n{3,}", "\n\n", parca)


if __name__ == "__main__":
    no = sys.argv[1]
    dil = sys.argv[2] if len(sys.argv) > 2 else "en"
    m = metin(no, dil)
    yol = OUT / f"{no}.md"
    yol.write_text(m, encoding="utf-8")
    print(f"{no}: {len(m)} karakter -> {yol}")
    print("\n=== OZET ===\n" + dilim(m, "Abstract", 900))
    print("\n=== 1. ISTEM ===\n" + dilim(m, "Claims (", 2200))
