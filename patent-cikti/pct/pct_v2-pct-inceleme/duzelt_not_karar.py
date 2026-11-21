r"""UYGUNLUK_NOTU §8 Karar'ını ve §4'ün iki iç atfını düzeltir (2026-11-19).

Kapı hedefi: `rapor_denetle.py` "kaynakta yok" bulgusu = 0. Karar paragrafı artık
her ölçümü AYNI satırda kendi `dosya:satır` künyesiyle taşıyor; böylece okuyan kişi
cümlenin altındaki ham JSON satırını açabilir.

Koşum: python3 patent-cikti/pct/pct_v2-pct-inceleme/duzelt_not_karar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

NOT = Path("/Users/serkankose/Documents/quann-3d/docs/patent/pct_v2/UYGUNLUK_NOTU.md")
PC = "patent-cikti/pct/pct_v2-pct-inceleme"

KARAR = f"""Metin kapıları bu hâliyle temiz, her ölçüm kendi künyesiyle:

- PCT biçim denetimi EN'de bulgu yok: `{PC}/A_pct_en.json:86-87`.
- PCT biçim denetimi TR'de bulgu yok: `{PC}/B_pct_tr.json:86-87`.
- İstem–tarifname desteğinde karşılanmayan istem listesi boş (`"unsupported_claims": []`),
  kapsam tam: `{PC}/F_epo_spec_support_en.json:631-635`.
- Paket bütünlüğü tam not: `{PC}/H_pkg_full.json:10` ve `{PC}/I_pkg_filing.json:10`.
- US istem denetiminde bulgu yok, not tam: `{PC}/J_us_claims_en.json:4` ve
  `{PC}/J_us_claims_en.json:10`.

Kalan D/E/F/G bulguları ile yerel `antecedent_basis` hatalarının tamamı §4 ve §6'da
kanıtlanmış araç sınırı ya da tasarım kararıdır. Metin dışı katman (§5) hiç denenmedi;
**ön sanat araması yapılmadı**.
"""

KARAR_SONU = "**Başvuru kararı için vekil onayı gerekli.**\n"

EDITS: list[tuple[str, str]] = [
    # §4.5 künyesindeki üretici betiğin gerçek adı küçük harfli
    (f"`{PC}/KANIT_antecedent.py` (üretici)", f"`{PC}/kanit_antecedent.py` (üretici betik)"),
    # §3 madde imleri kaldırılıp tabloya çevrilmişti: "madde 2" pointer'ı tablo satırına döner
    ("§3 madde 2'nin yeniden yazımıyla bulgu gerçek düzeltmeye dönüştü",
     "§3'ün [0063] satırındaki yeniden yazımla bulgu gerçek düzeltmeye dönüştü"),
]


def main() -> int:
    text = NOT.read_text(encoding="utf-8")

    bas = "Metin kapıları bu hâliyle temiz"
    if bas not in text:
        raise SystemExit(f"Karar başlangıcı bulunamadı: {bas!r}")
    if text.count(KARAR_SONU) != 1:
        raise SystemExit("Karar son cümlesi tek değil")
    i = text.index(bas)
    j = text.index(KARAR_SONU)
    text = text[:i] + KARAR + text[j:]

    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"beklenen 1 eşleşme, bulunan {n}: {old[:70]!r}")
        text = text.replace(old, new)
        print(f"OK: {old[:56]!r}")

    NOT.write_text(text, encoding="utf-8")

    t = NOT.read_text(encoding="utf-8")
    for gerekli in ("unsupported_claims", "ön sanat araması yapılmadı",
                    KARAR_SONU.strip(), "kanit_antecedent.py"):
        assert gerekli in t, f"içerik kaybı: {gerekli!r}"
    for yasak in ("KANIT_antecedent.py", "§3 madde 2"):
        assert yasak not in t, f"eski metin duruyor: {yasak!r}"
    print(f"\nSAGLAMA OK: not {len(t.splitlines())} satır")
    return 0


if __name__ == "__main__":
    sys.exit(main())
