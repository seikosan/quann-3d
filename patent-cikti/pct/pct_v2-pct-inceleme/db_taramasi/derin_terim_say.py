#!/usr/bin/env python3
"""Derin incelenen patent ham metinlerinde terim-varlik sayimi + bibliyografya.

Amac: ON_SANAT.md'de yazilacak "su kayitta X terimi geciyor / gecmiyor" cumleleri
sayi olarak satira baglansin. Sayim ham dosya uzerinde buyuk/kucuk harf duyarsiz
alt-dizi sayimidir; anlam cikarimi yapmaz.

Cikti: KANIT_derin_terim.txt
"""
from __future__ import annotations

import re
from pathlib import Path

BURADA = Path(__file__).resolve().parent
Dizin = BURADA / "patentlar"

# terimler: istem ogelerine karsilik gelen anahtar kelimeler
TERIMLER = [
    "language model", "language", "text", "next token", "point cloud",
    "permutation", "deep set", "slot", "set aggregation", "pooling",
    "explanation", "attention", "cluster", "surprisal", "exclu",
    "quantiz", "int8", "lookup table", "backpropagat",
]

KAYITLAR = [
    "US12639556B2", "US12510888B2", "CN117392424A", "US12190228B2",
    "US20250156898A1", "WO2026046480A1", "WO2026170161A1", "US20260017347A1",
    "US12039636B2",
]


def biblio(metin: str) -> dict:
    out = {}
    for etiket, kalip in [
        ("atfedilen", r"Assignee[:\s]+(.{0,60})"),
        ("onculluk", r"Priority date.*?(\d{4}-\d{2}-\d{2})"),
        ("yayim", r"Publication of ([A-Z]{2}\d{4,11}[A-Z]\d?)"),
    ]:
        s = re.search(kalip, metin, re.I | re.S)
        out[etiket] = re.sub(r"\s+", " ", s.group(1)).strip()[:60] if s else "-"
    return out


def main() -> int:
    satir = [
        "# KANIT — derin incelenen patentlarda terim sayimi (uretim: derin_terim_say.py)",
        "# sayim = ham dosyada kucuk/harf duyarsiz alt-dizi sayisi; anlam iddiasi degil",
        "# ham: db_taramasi/patentlar/<NO>.md",
    ]
    for no in KAYITLAR:
        yol = Dizin / f"{no}.md"
        if not yol.exists():
            satir.append(f"\n## {no}\n- HAM DOSYA YOK")
            continue
        m = yol.read_text(encoding="utf-8", errors="replace")
        b = biblio(m)
        sayim = {t: m.lower().count(t) for t in TERIMLER}
        satir.append(f"\n## {no}")
        satir.append(f"- karakter | {len(m)} | ham: patentlar/{no}.md")
        satir.append(f"- atfedilen | {b['atfedilen']} | onculluk | {b['onculluk']}")
        satir.append("- terim sayimi: " + " · ".join(f"{t}={n}" for t, n in sayim.items()))
        print(no, "karakter", len(m), "| language model", sayim["language model"],
              "| point cloud", sayim["point cloud"], "| next token", sayim["next token"])
    yol = BURADA / "KANIT_derin_terim.txt"
    yol.write_text("\n".join(satir) + "\n", encoding="utf-8")
    print("->", yol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
