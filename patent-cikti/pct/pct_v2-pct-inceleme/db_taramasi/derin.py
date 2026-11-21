#!/usr/bin/env python3
"""Secilen patent kayitlarinin tam metnini cekip 1. istem dilimini cikarir.

Girdi : YUKLE listesi (bu dosyada) — db_taramasi/index_ozet.txt'ten secildi,
        secim olcuti: istem 1/9/12/14 ozelliklerine en yakin gorunen yayim numaralari.
Cikti : patentlar/<NO>.md          — ham sayfa metni (jina uzerinden)
        DERIN_kayit.md             — her kayit icin ozet + 1. istem dilimi
Not   : sayfa metni ticari veritabani degil, Google Patents'in yayimladigi metindir.
"""
from __future__ import annotations

import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BURADA = Path(__file__).resolve().parent
PA = BURADA / "patentlar"
PA.mkdir(exist_ok=True)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

# (yayim_no, neden secildi — hangi istem ozelligine)
YUKLE = [
    ("US20250156898A1", "istem 1(a)+(e): ozellik nokta bulutu + aciklanabilirlik iddiasi"),
    ("US12039636B2", "istem 1(c): deep set tabanli semetrik agregasyon"),
    ("CN117392424A", "istem 1(b)+(c): geometri ozellikleri + deep sets siniflandirma"),
    ("WO2026046480A1", "istem 1(e)(i): girdi cikarma ile AI testi/aciklama"),
    ("US12190228B2", "istem 9/14: INT8 agirlik + lookup table + kenet cihaz"),
    ("US12510888B2", "istem 1(e)(ii-iii): bolge/kume tabanli aciklama + dagilim karsilastirma"),
    ("WO2026170161A1", "istem 1(a): sembolik cok boyutlu matris/gosterim"),
    ("US20260017347A1", "istem 1(e)(ii): latent uzay kume etiketleme"),
]


def cek(no: str) -> str:
    hedef = f"https://patents.google.com/patent/{no}/en"
    with urllib.request.urlopen("https://r.jina.ai/" + hedef, timeout=150) as r:  # noqa: S310
        return r.read().decode("utf-8", "replace")


def bul(metin: str, isaret: str, n: int) -> str:
    i = metin.find(isaret)
    if i < 0:
        return ""
    parca = metin[i:i + n]
    parca = re.sub(r"\*\*", "", parca)
    return re.sub(r"\n{3,}", "\n\n", parca).strip()


def main() -> int:
    satirlar = [f"# Derin inceleme — 1. istem metinleri (uretim: derin.py, {datetime.now(timezone.utc).date()})\n"]
    for i, (no, sebep) in enumerate(YUKLE):
        kayit = f"\n## {no}\n- neden: {sebep}\n"
        try:
            m = cek(no)
            (PA / f"{no}.md").write_text(m, encoding="utf-8")
            ozet = bul(m, "Abstract", 800)
            ozet = ozet[len("Abstract"):].strip() if ozet else ""
            istem = bul(m, "What is claimed is:", 2600)
            if not istem:
                istem = bul(m, "Claims (", 2600)
            kayit += f"- sayfa: {len(m)} karakter · ham: `patentlar/{no}.md`\n"
            kayit += f"- OZET: {ozet[:700]}\n"
            kayit += f"- 1. ISTEM: {istem[:2400]}\n"
            print(f"{no} OK {len(m)} karakter", flush=True)
        except Exception as e:  # noqa: BLE001
            kayit += f"- HATA: {type(e).__name__}: {e}\n"
            print(f"{no} HATA {type(e).__name__}", flush=True)
        satirlar.append(kayit)
        if i < len(YUKLE) - 1:
            time.sleep(25)
    (BURADA / "DERIN_kayit.md").write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print("->", BURADA / "DERIN_kayit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
