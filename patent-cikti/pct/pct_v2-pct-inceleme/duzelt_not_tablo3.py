r"""§3 listesini kanıt sütunlu tabloya çevirir (2026-11-19).

Neden tablo: `rapor_denetle.py` madde imlerini ("1.", "5.") sayı sayıyor ve satır
içi sarmalama yüzünden sayı ile atıf farklı satırlarda kalıyordu. Notun geri kalanı
zaten tablo (§1, §2, §4, §6, §7); içerik korunuyor, izlenebilirlik artıyor.

Koşum: python3 duzelt_not_tablo3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

NOT = Path("/Users/serkankose/Documents/quann-3d/docs/patent/pct_v2/UYGUNLUK_NOTU.md")
PC = "patent-cikti/pct/pct_v2-pct-inceleme"
CE = "docs/patent/pct_v2/CLAIMS_EN.md"
DE = "docs/patent/pct_v2/DESCRIPTION_EN.md"

UC = f"""| Düzeltme | Metindeki karşılığı | Kapıya etkisi |
|---|---|---|
| EN bağımsız istem 1/13/15 iki parçalı biçime getirildi ("characterized in that"); TR karşılıklarının ön yükü önceki tekniği anlatan gövde + "karakterize edilir" olarak yeniden yazıldı | `{CE}:10` · `{CE}:52` · `{CE}:66` | D'nin Rule 43(1) bulguları kalktı; bugün tek bulgu Rule 43(2) (kural §1, ölçüm §2) |
| [0063]'teki saha ifadesi istem terimiyle eşleştirildi: "0.64 on 25 critical records" → "0.64 on 25 records of the critical stage" (TR: "kritik kademedeki 25 kayıtta") | `{DE}:199` (TR aynası aynı satırda) | H/I `filing_copy_purity` CRITICAL'i kalktı; paket bütünlüğü tam not (§2 H/I) |
| [0065] istem on'un terimini aldı: "a text consisting of a plurality of sentences" | `{DE}:207` | F'nin CRITICAL'lerinden biri kalktı (§2 F) |
| [0050] istem on dört'ün terimini aldı: "a distribution store holding" (TR: "bir dağıtım deposu taşır") | `{DE}:159` | F'nin ikinci CRITICAL'i kalktı (§2 F) |
| [0034] istem dört'ün terimleriyle genişletildi: "a result of the max-pooling" / "a result of the average-pooling" | `{DE}:105` · `{PC}/F_epo_spec_support_en.json:631-633` | F kritikleri sıfırlandı, coverage tam (§2 F) |
| Bölüm başlığı "BEST MODE(S) FOR CARRYING OUT THE INVENTION" → "BEST MODE FOR CARRYING OUT THE INVENTION" | §1 Rule 5.1(a) satırı; G koşusu ölçümü §2'de | G'deki "Missing required description section" CRITICAL'i kalktı |
| Öncül dayanak ve anlatım netleştirmeleri (EN + TR aynası); [0028] ve [0068]'deki "only/must", "yalnız/zorunda" ifadeleri çıkarıldı (yerel `buzzword` uyarıları) | `{CE}:32-66` arası istem satırları | TR istem dosyalarında hata kalmadı (`{PC}/lint_ISTEMLER_TR_md.txt:1`, §2 son satır) |
"""


def main() -> int:
    lines = NOT.read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("1. EN bağımsız istem"))
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("## 4."))
    NOT.write_text("\n".join(lines[:i] + UC.rstrip("\n").splitlines() + [""] + lines[j:]) + "\n",
                   encoding="utf-8")
    t = NOT.read_text(encoding="utf-8")
    for anahtar in ("characterized in that", "records of the critical stage",
                    "a plurality of sentences", "a distribution store holding",
                    "average-pooling", "BEST MODE FOR CARRYING OUT", "buzzword", "## 4."):
        assert anahtar in t, f"içerik kaybı: {anahtar!r}"
    print(f"§3 satır {i+1}-{j} -> tablo; not {len(t.splitlines())} satır")
    return 0


if __name__ == "__main__":
    sys.exit(main())
