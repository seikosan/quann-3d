r"""UYGUNLUK_NOTU §1 kural matrisini ham ölçüm satırlarına bağlar (2026-11-19).

İki şey yapılır:
1. §1 satırlarındaki "motor şu alanı diyor" anlatımı gerçek `dosya:satır` künyesine çevrilir.
2. TR başlık kelime sayısı düzeltilir: §1 "TR 8 kelime" diyordu, motor bugün `word_count: 7`
   diyor (`B_pct_tr.json:12`) — başlık kısaltması sonrası §1 güncellenmemişti.
Ayrıca Rule 6.2(a) satırına gerçek tarama sonucu yazılır (TR'deki tek "şekil" eşleşmesi
"hesaplayacak şekilde" sözcüğüdür, şekil atfı değildir).

Kosum: python3 patent-cikti/pct/pct_v2-pct-inceleme/duzelt_not_bolum1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

NOT = Path("/Users/serkankose/Documents/quann-3d/docs/patent/pct_v2/UYGUNLUK_NOTU.md")
PC = "patent-cikti/pct/pct_v2-pct-inceleme"
D = "docs/patent/pct_v2"

EDITS: list[tuple[str, str]] = [
    # Rule 4.3 — TR başlığı 8 değil 7 kelime; künyeler satır numarasıyla
    ("""| Rule 4.3 | Başlık kısa, teknik alanla uylu | EN 6 kelime, TR 8 kelime; motor `compliant: true` | `A_pct_en.json` `results.title.word_count=6`; `B_pct_tr.json` `=8` |""",
     f"""| Rule 4.3 | Başlık kısa, teknik alanla uylu | EN altı, TR yedi kelime; motor iki dilde de `compliant: true` | `{PC}/A_pct_en.json:12` (`word_count: 6`) ve `:13` · `{PC}/B_pct_tr.json:12` (`word_count: 7`) ve `:13` |"""),
    # Rule 5.1(a)
    ("""| Rule 5.1(a) | Altı başlık sırayla | Altısı da var | `DESCRIPTION_EN.md:11,21,39,67,87,217`; `A_pct_en.json` `results.description.found_elements` 5 ✓ `missing_elements: []` |""",
     f"""| Rule 5.1(a) | Teknik alan, önceki teknik, buluşun özeti, şekillerin kısa tarifi, en iyi biçim (+ sanayiye uygulanabilirlik) | Altı başlık da sırayla; motorun denetlediği beş alanın eksik listesi boş | `{D}/DESCRIPTION_EN.md:11,21,39,67,87,217` · `{PC}/A_pct_en.json:17-24` (`found_elements` beş alan, `missing_elements: []`) · `{PC}/B_pct_tr.json:24` |"""),
    # Rule 6.1(b)
    ("""| Rule 6.1(b) | İstemler ardışık numaralı | 1–15 ardışık, 15 istem | `A_pct_en.json` `results.claims.claim_numbers=[1..15]`; `uretim.json` `claim_count=15` (EN+TR) |""",
     f"""| Rule 6.1(b) | İstemler ardışık numaralı | On beş istem, numaralar ardışık | `{PC}/A_pct_en.json:28-29` (`total_claims: 15`) · `{PC}/B_pct_tr.json:28-29` · `{PC}/uretim.json:40` ve `:82` (`claim_count: 15`) |"""),
    # Rule 6.2(a) — gerçek tarama sonucu
    ("""| Rule 6.2(a) | İstemde tarifname/şekil atfı yok | Hiçbir istemde "Şekil/FIG./paragraf" atfı yok | `CLAIMS_EN.md:10-66`, `ISTEMLER_TR.md` taraması; `lint_*.txt`'te atıf bulgusu yok |""",
     f"""| Rule 6.2(a) | İstem gövdesinde tarifname/şekil atfı yok | EN gövdesinde tarama boş; TR'de tek eşleşme "hesaplayacak **şekilde**" sözcüğü — şekil atfı değil | `{D}/CLAIMS_EN.md:10-66` · `{D}/ISTEMLER_TR.md:11-39` taraması; eşleşme `{D}/ISTEMLER_TR.md:37` |"""),
    # Rule 6.2(b)
    ("""| Rule 6.2(b) | Referans işaretleri, anlaşmayı hızlandırmıyorsa zorunlu değil | Bilinçli olarak hiç işaret yok (numara şeması uydurulmadı) | `CLAIMS_EN.md:7` künye satırı; `G_epo_formal_en.json` 4. bölüm |""",
     f"""| Rule 6.2(b) | Referans işaretleri, anlaşmayı hızlandırmıyorsa zorunlu değil | Bilinçli olarak hiç işaret yok (numara şeması uydurulmadı) | `{D}/CLAIMS_EN.md:7` tasarım notu · `{PC}/G_epo_formal_en.json:62` (kural bu eksikliği uyarıya çeviriyor, §4.4) |"""),
    # Rule 5.2(a)/11.13(m) — "4 dokümanda" değil: iki tarifnede
    ("""| Rule 5.2(a) / 11.13(m) | Paragraf numaralandırması | [0001]–[0070] kesintisiz, EN ve TR | `uretim.json` `paragraf=70 surekli=True` (4 dokümanda) |""",
     f"""| Rule 5.2(a) / 11.13(m) | Paragraf numaralandırması | Her iki tarifnede de yedi yüz paragraf, aralıksız | `{PC}/uretim.json:14-15` (EN: `paragraph_count: 70`, `paragraph_gapless: true`) · `:56-57` (TR aynı) |"""),
    # Rule 8.1(b)
    ("""| Rule 8.1(b) | Özet 50–150 kelime | EN 137 · TR 106 kelime | `A_pct_en.json` `results.abstract.word_count=137`; `B_pct_tr.json` `=106` |""",
     f"""| Rule 8.1(b) | Özet 50–150 kelime | EN yüz otuz yedi · TR yüz altı kelime, iki dilde `compliant: true` | `{PC}/A_pct_en.json:6-7` · `{PC}/B_pct_tr.json:6-7` |"""),
    # Rule 9.1 — yerel lint satır numarası
    ("""| Rule 9.1 (birlik) | Tek genel konsept | 3 bağımsız (yöntem/sistem/kayıt ortamı) + 12 bağımlı, aynı buluş kavramı | `ISTEMLER_TR.md` ve `CLAIMS_EN.md` yapısı; yerel lint `claim_unity` = **info** (uyarı değil) |""",
     f"""| Rule 9.1 (birlik) | Tek genel konsept | Üç bağımsız (yöntem/sistem/kayıt ortamı) + on iki bağımlı | yerel lint `claim_unity` = **info** (uyarı değil): `{PC}/lint_CLAIMS_EN_md.txt:26` · `{PC}/lint_ISTEMLER_TR_md.txt:11` |"""),
    # Rule 11.13(l)-(n)
    ("""| Rule 11.13(l)–(n) | Metin temizliği: işaret/paragraf/terim tutarlılığı | Motor `terminology` ✓ (metrik dışim birim yok) | `A_pct_en.json` / `B_pct_tr.json` `results.terminology.compliant=true` |""",
     f"""| Rule 11.13(l)–(n) | Metin temizliği: işaret/paragraf/terim tutarlılığı | Terim denetimi iki dilde de temiz: metrik dışı birim ve aykırı sıcaklık listeleri boş | `{PC}/A_pct_en.json:49-52` · `{PC}/B_pct_tr.json:49-52` |"""),
]


def main() -> int:
    text = NOT.read_text(encoding="utf-8")
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"beklenen 1 eşleşme, bulunan {n}: {old[:80]!r}")
        text = text.replace(old, new)
        print(f"OK: {old[:52]!r}")

    # §4 tablosu ile ## 5 basligi arasina: markdown icin bos satir
    old = "sezgisel kaldı. |\n## 5."
    if text.count(old) == 1:
        text = text.replace(old, "sezgisel kaldı. |\n\n## 5.")
        print("OK: ## 5 başlığından önce boş satır eklendi")

    NOT.write_text(text, encoding="utf-8")

    t = NOT.read_text(encoding="utf-8")
    assert "TR 8 kelime" not in t, "bayat TR başlık sayısı duruyor"
    assert t.count("word_count: 7") >= 1 and "hesaplayacak" in t and "paragraph_gapless" in t
    print(f"\nSAGLAMA OK: not {len(t.splitlines())} satır")
    return 0


if __name__ == "__main__":
    sys.exit(main())
