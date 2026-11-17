#!/usr/bin/env python3
"""Kan (durum) raporu — sayılar MODELİN HAFIZASINDAN değil, dosyadan okunur.

Neden var: ajan, günlüğü yeniden okumadan sayı uydurdu ("eski run 1000 cümleyle
eğitildi" → günlükte 1.500.000 yazıyor) ve saniyeyi saat sandı (25.900 s → "25,9 saat").
Bu betik her sayıyı ham kaynaktan basar ve yanına kaynağını yazar; rapor bu çıktıdan
kopyalanır, hatırlanmaz.

Kullanım:  python3 scripts/durum.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
GUNLUK = KOK / "logs/run_resume_r2_fixed.log"
VERI = KOK / "data/turkish_wiki_1500k.txt"
HEDEF_PPL = 13.48          # RESEARCH.md'deki hedef
SATIR = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s*\|\s*loss=([\d.]+)\s*\|\s*ppl=([\d.]+)\s*\|\s*(\d+)s"
)


def saat(saniye: float) -> str:
    """Saniyeyi TEK birimde değil, iki birimde ver — birim karışması hata kaynağı."""
    return f"{saniye:.0f} s = {saniye/3600:.2f} saat"


def main() -> int:
    if not GUNLUK.exists():
        print(f"YOK: {GUNLUK} (koşu günlüğü bulunamadı)", file=sys.stderr)
        return 1

    satirlar = GUNLUK.read_text(errors="replace").splitlines()
    kayitlar = []
    for i, s in enumerate(satirlar, 1):
        m = SATIR.search(s)
        if m:
            kayitlar.append((i, int(m[1]), int(m[2]), float(m[3]),
                             float(m[4]), int(m[5]), s.strip()))
    if not kayitlar:
        print(f"YOK: {GUNLUK} içinde epoch satırı yok", file=sys.stderr)
        return 1

    son = kayitlar[-1]
    en_iyi = min(kayitlar, key=lambda k: k[4])
    satir_no, epoch, toplam, loss, ppl, saniye, ham = son

    print("=== EĞİTİM DURUMU (kaynak: ham dosya) ===")
    print(f"kaynak        : logs/{GUNLUK.name}:{satir_no}")
    print(f"ham satır     : {ham}")
    print(f"epoch         : {epoch}/{toplam}  ({100*epoch/toplam:.0f}% tamamlandı)")
    print(f"loss / ppl    : {loss} / {ppl}")
    print(f"geçen süre    : {saat(saniye)}")
    kalan_ep = toplam - epoch
    if len(kayitlar) >= 2:
        hiz = (kayitlar[-1][5] - kayitlar[0][5]) / (kayitlar[-1][1] - kayitlar[0][1])
        print(f"epoch hızı    : {hiz:.0f} s/epoch ({len(kayitlar)} ölçülen epoch arası)")
        print(f"kalan tahmin  : {saat(kalan_ep * hiz)}  ({kalan_ep} epoch)")
    else:
        print(f"kalan tahmin  : hesaplanamadı (tek epoch kaydı)")

    print(f"\n=== EN İYİ KAYIT (aynı günlük) ===")
    print(f"logs/{GUNLUK.name}:{en_iyi[0]} → epoch {en_iyi[1]} · loss={en_iyi[3]} · ppl={en_iyi[4]}")

    print("\n=== HEDEF KIYASI ===")
    print(f"hedef ppl     : {HEDEF_PPL} (kaynak: RESEARCH.md)")
    if ppl <= HEDEF_PPL:
        print(f"hüküm         : HEDEFE ULAŞILDI (ppl {ppl} ≤ {HEDEF_PPL})")
    else:
        print(f"hüküm         : henüz ULAŞILMADI — ppl {ppl} ↔ hedef {HEDEF_PPL} "
              f"(fark {(ppl/HEDEF_PPL - 1)*100:+.1f}%)")

    print("\n=== VERİ (iddia değil, sayım) ===")
    if VERI.exists():
        with VERI.open(errors="replace") as f:
            n = sum(1 for _ in f)
        print(f"data/{VERI.name} : {n} satır (gerçek sayım, sabit metin değil)")
    else:
        print(f"YOK: {VERI}")

    print("\n=== KOŞU SÜRECİ ===")
    cikti = subprocess.run(["pgrep", "-fl", "train_resume_r2_fixed"], capture_output=True, text=True).stdout.strip()
    print(cikti or "YOK: eğitim süreci bulunamadı (koşu bitmiş ya da ölmüş olabilir)")

    print("\nKURAL: Bu çıktıda olmayan bir sayıyı rapora YAZMA. "
          "Farklı bir sayı gerekiyorsa onu da bir dosyadan/grep'ten türet ve kaynağını yaz.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
