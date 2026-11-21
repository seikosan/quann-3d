r"""OLCUM_KAYITLARI.md — `dogrul_olcum.py` bulgularının metin tarafındaki düzeltmeleri (2026-11-19).

Dört düzeltme, hepsi ham kaynak okunarak yapıldı:

1. **Aritmetik düzeltme (gerçek kusur):** satır 52 "(50/13)² = 14.7970" diyordu.
   (50/13)² = 14.792899… → 14.7929. Tarifnamede yuvarlanmış 14.8 yazılı
   (`docs/patent/pct_v2/DESCRIPTION_EN.md:123` ve `DESCRIPTION_TR.md:123`), yani tarifname
   doğru; hatalı olan defterin elle yazdığı fazladan hassasiyetti.
2. **Ham biçim künyesi (satır 153):** `logs/run_compare.log:13-14` satırlarında sayılar
   ayraçlı: "20,999,648" ve "22,724,768". Defter ayraçsız değeri ham gibi gösteriyordu;
   ikisi yan yana yazıldı.
3. **Künye yolu açıldı (D1 bloku, satır 204 sonrası):** `manifest.json`, `TARIFNAME.md`,
   `TEKNIK_ETKI.md`, `edge_eval_full_scale.json`, `complaints_scale_stability.json` çıplak
   dosya adlarıyla geçiyordu; tam yol'a çevrildi (kapı bu adresleri açamıyordu).
4. Satır 51'in oranı ham iki satırdan yeniden hesaplandığı için olduğu gibi bırakıldı.

Kosum: python3 patent-cikti/pct/pct_v2-pct-inceleme/duzelt_olcum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/Users/serkankose/Documents/quann-3d")
OLCUM = REPO / "docs/patent/pct_v2/OLCUM_KAYITLARI.md"


def main() -> int:
    # --- ham kaynak kontrolleri: duzeltmenin dayanagi -----------------------------
    cmp_log = (REPO / "logs/run_compare.log").read_text(encoding="utf-8").splitlines()
    assert "20,999,648" in cmp_log[12] and "22,724,768" in cmp_log[13], cmp_log[12:14]
    en = (REPO / "docs/patent/pct_v2/DESCRIPTION_EN.md").read_text(encoding="utf-8").splitlines()
    assert "(50/13)^2, is 14.8" in en[122], en[122][:120]
    assert round((50 / 13) ** 2, 4) == 14.7929, (50 / 13) ** 2

    text = OLCUM.read_text(encoding="utf-8")
    for old, new in [
        ("| nokta oranının karesi | 14.8 ((50/13)² = 14.7970) | hesap: yukarıdaki iki satır |",
         "| nokta oranının karesi | 14.8 — ham hesap (50/13)² = 14.7929; tarifname 14.8'e "
         "yuvarlıyor (`docs/patent/pct_v2/DESCRIPTION_EN.md:123`) | hesap: yukarıdaki iki satır |"),
        ("| karşılaştırılan parametre sayıları | 20999648 / 22724768 | logs/run_compare.log:13-14 |",
         "| karşılaştırılan parametre sayıları | 20999648 / 22724768 — ham log biçimi ayraçlı: "
         "\"20,999,648 params\" / \"22,724,768 params\" | logs/run_compare.log:13-14 |"),
    ]:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"beklenen 1 eşleşme, bulunan {n}: {old[:70]!r}")
        text = text.replace(old, new)
        print(f"OK: {old[:56]!r}")

    # --- D1 bloku: tam yol (yalniz satir 204'ten sonra) ---------------------------
    satirlar = text.splitlines(keepends=True)
    kesim = 203  # 0-tabanlı: 204. satırdan sonrası (D1 bloku)
    bulunan = 0
    for i in range(kesim, len(satirlar)):
        s = satirlar[i]
        for eski, yeni in [
            ("TARIFNAME.md:", "docs/patent/final_v3/TARIFNAME.md:"),
            ("TEKNIK_ETKI.md:", "docs/patent/final_v3/TEKNIK_ETKI.md:"),
            ("manifest.json:", "fraud/results/deploy/manifest.json:"),
            ("edge_eval_full_scale.json:", "fraud/results/edge_eval_full_scale.json:"),
            ("complaints_scale_stability.json:", "complaints/results/complaints_scale_stability.json:"),
        ]:
            # zaten tam yol olan seyi tekrar uzatma
            for on_ek in ("docs/patent/final_v3/", "fraud/results/deploy/", "fraud/results/",
                          "complaints/results/"):
                if on_ek + eski in s:
                    break
            else:
                if eski in s:
                    bulunan += s.count(eski)
                    s = s.replace(eski, yeni)
        satirlar[i] = s
    text = "".join(satirlar)
    print(f"OK: D1 blokunda {bulunan} künye tam yola çevrildi")

    OLCUM.write_text(text, encoding="utf-8")
    t = OLCUM.read_text(encoding="utf-8")
    assert "14.7970" not in t and "14.7929" in t
    assert "20,999,648" in t
    print(f"-> {OLCUM.relative_to(REPO)} yazıldı ({len(t.splitlines())} satır)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
