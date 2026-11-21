r"""UYGUNLUK_NOTU §2 motor tablosunu YENİDEN KURAR — değerler elle yazılmaz.

Neden: §2'nin Kritik/Önemli/Skor sayıları şimdiye kadar "Yorum" sütununda duruyordu ve
satır bazında künye taşımıyordu (rapor kapısı tablo satırlarını saymıyor). Bu betik:

1. Taze JSON dosyalarını okur, her değeri ve DEĞERİN GEÇTİĞİ SATIR NUMARASINI kendisi bulur.
2. Değeri birden fazla satırda geçen anahtar için üst anahtar (parent) yakınlığıyla seçim
   yapar; yine belirsizse HİÇBİR ŞEY yazmadan çıkar.
3. Tabloyu §2'nin yerine yazar; "Yorum" metinleri mevcut tablodan aynen taşınır.
4. Yazmadan önce her künyenin hedef satırında değerin gerçekten geçtiğini doğrular.

Kosum: python3 patent-cikti/pct/pct_v2-pct-inceleme/yeniden_bolum2.py
"""

from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

REPO = Path("/Users/serkankose/Documents/quann-3d")
NOT = REPO / "docs/patent/pct_v2/UYGUNLUK_NOTU.md"
OUT = REPO / "patent-cikti/pct/pct_v2-pct-inceleme"
PC = "patent-cikti/pct/pct_v2-pct-inceleme"


def js(v: object) -> str:
    return json.dumps(v)  # 0 -> "0", True -> "true", [] -> "[]", 100.0 -> "100.0"


def satirlar(dosya: str) -> list[str]:
    return (OUT / dosya).read_text(encoding="utf-8").splitlines()


def satir_bul(dosya: str, key: str, deger: object, parent: str | None = None) -> int:
    """`"key": deger` kalıbının geçtiği satır numarası (1-tabanlı). Belirsizse ValueError."""
    hat = f'"{key}": {js(deger)}'
    aday = [i + 1 for i, ln in enumerate(satirlar(dosya)) if hat in ln]
    if not aday:
        raise ValueError(f"{dosya}: {hat!r} hiç bulunamadı")
    if len(aday) == 1:
        return aday[0]
    if parent:
        ust = [u for u in (i + 1 for i, ln in enumerate(satirlar(dosya)) if f'"{parent}"' in ln)
               if u < min(aday)]
        secenek = [a for a in aday if ust and a > max(ust)]
        if len(secenek) == 1:
            return secenek[0]
    raise ValueError(f"{dosya}: {hat!r} birden fazla satırda {aday} — parent yeterli değil")


def alan(d: dict, yol: str):
    for parca in yol.split("."):
        d = d[parca]
    return d


# §2 satırı: koşu adı, JSON dosyası, (başlık, alan yolu, parent) demeti, yorum
KOSULAR: list[tuple[str, str, str, list[tuple[str, str, str | None]], str]] = [
    ("A_pct_en", "A_pct_en.json", "PCT biçimi, EN",
     [("Kritik", "result.compliance_summary.critical_issues", "compliance_summary"),
      ("Uyarı", "result.compliance_summary.warnings", "compliance_summary")],
     "`issues: []`; `overall_compliant: false` **yalnız** atlanan iki kontrol yüzünden (§5)"),
    ("B_pct_tr", "B_pct_tr.json", "PCT biçimi, TR",
     [("Kritik", "result.compliance_summary.critical_issues", "compliance_summary"),
      ("Uyarı", "result.compliance_summary.warnings", "compliance_summary")],
     "aynısı; `overall_compliant: false` yine yalnız atlanan kontrollerden"),
    ("C_pct_en_desc_only", "C_pct_en_desc_only.json", "Yalnız tarifname",
     [("Kritik", "result.compliance_summary.critical_issues", "compliance_summary"),
      ("Uyarı", "result.compliance_summary.warnings", "compliance_summary")],
     "„No claims detected“ — beklenen; motor istemi görmek için tarifname+istem **birleşik** "
     "ister (A bunu gösterir) → §4.6"),
    ("D_epo_claims_en", "D_epo_claims_en.json", "EPO istem biçimi, EN",
     [("Bulgu", "result.total_issues", None),
      ("Kritik", "result.critical_issues", None),
      ("Önemli", "result.important_issues", None),
      ("Skor", "result.compliance_score", None)],
     "kalan tek bulgu Rule 43(2) kategori → §4.1 artefaktı"),
    ("E_epo_claims_tr", "E_epo_claims_tr.json", "EPO istem biçimi, TR",
     [("Bulgu", "result.total_issues", None),
      ("Kritik", "result.critical_issues", None),
      ("Önemli", "result.important_issues", None),
      ("Skor", "result.compliance_score", None)],
     "15× Rule 43(1) + 1× Rule 43(2) → §4.2, İngilizce-only kalıp"),
    ("F_epo_spec_support_en", "F_epo_spec_support_en.json", "İstem–tarifname desteği",
     [("Kritik", "result.result.critical_issues", "spec_coverage"),
      ("Önemli", "result.result.important_issues", "spec_coverage"),
      ("Coverage", "result.result.spec_coverage.percentage", "spec_coverage"),
      ("Desteklenen", "result.result.spec_coverage.supported_claims", "spec_coverage"),
      ("Karşılanmayan", "result.result.spec_coverage.unsupported_claims", "spec_coverage")],
     "`compliant: true`; 47 tavsiye → §4.3"),
    ("G_epo_formal_en", "G_epo_formal_en.json", "EPO biçimsel",
     [("Kritik", "result.compliance_summary.critical_issues", "compliance_summary"),
      ("Uyarı", "result.compliance_summary.warnings", "compliance_summary"),
      ("Genel uygunluk", "result.overall_compliant", None)],
     "kalan uyarı: özetlerde parantezli işaret (tasarım kararı) → §4.4"),
    ("H_pkg_full", "H_pkg_full.json", "12 dokümanlık paket",
     [("Bulgu", "result.total_issues", None),
      ("Kritik", "result.critical_issues", None),
      ("Skor", "result.package_integrity_score", None)],
     "`filing_copy_purity` CRITICAL **kalktı**"),
    ("I_pkg_filing", "I_pkg_filing.json", "6 dosyalık kopya",
     [("Bulgu", "result.total_issues", None),
      ("Kritik", "result.critical_issues", None),
      ("Skor", "result.package_integrity_score", None)],
     "`stamps: []`"),
    ("J_us_claims_en", "J_us_claims_en.json", "US istem denetimi",
     [("Bulgu", "result.total_issues", None),
      ("Kritik", "result.critical_issues", None),
      ("Skor", "result.compliance_score", None)],
     "`antecedent_basis` opt-in kapalı (§5)"),
]

BASLIK = ("| Koşu | Kapsam | Değerler — her biri kendi JSON satırında | Yorum |\n"
          "|---|---|---|---|\n")

NOTLAR = f"""Not: A/B/C/G satırlarında motor **`warnings`** alanını bildiriyor (Kritik =
`critical_issues`); D/E/F/H/I/J satırlarında **`important_issues`** var. Her hücredeki sayı,
kendi künyesinin açıldığı satırda birebir geçiyor; F'nin „Karşılanmayan" hücresi boş listedir
(`[]`). Skorlar motorun ham ondalık değeri, yuvarlanmadı. Bu tablodaki hiçbir sayı elle
yazılmadı: değerler ve satır numaraları `yeniden_bolum2.py` tarafından JSON'dan okundu.
"""


def main() -> int:
    satir_metinleri: list[str] = []
    for kosu, dosya, kapsam, alanlar, yorum in KOSULAR:
        d = json.loads((OUT / dosya).read_text(encoding="utf-8"))
        hucreler = []
        for ad, yol, parent in alanlar:
            deger = alan(d, yol)
            n = satir_bul(dosya, yol.split(".")[-1], deger, parent)
            hedef = satirlar(dosya)[n - 1]
            if js(deger) not in hedef:
                raise ValueError(f"{dosya}:{n} beklenen değeri içermiyor: {deger!r}")
            gorunum = f"{ad} `{js(deger)}`" if not isinstance(deger, list) else "Karşılanmayan `[]`"
            hucreler.append(f"{gorunum} — `{PC}/{dosya}:{n}`")
        satir_metinleri.append(f"| {kosu} | {kapsam} | " + "<br>".join(hucreler) +
                               f" | {yorum} |")

    tablo = BASLIK + "\n".join(satir_metinleri) + "\n"

    text = NOT.read_text(encoding="utf-8")
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("## 2."))
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("## 3."))

    # kosum damgasi elle yazilmaz: batarya cikti dosyasinin mtime'i okunur
    damga = datetime.datetime.fromtimestamp(
        (OUT / "konsola_cikti.txt").stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    baslik = f"## 2. Motor bataryası — ham sonuç (koşum {damga}, `run_pct_analizi.py`)"

    yeni = (lines[:i] + [baslik] + lines[i + 1:i + 2] + tablo.rstrip("\n").splitlines()
            + ["", NOTLAR.rstrip("\n"), ""] + lines[j:])
    NOT.write_text("\n".join(yeni) + "\n", encoding="utf-8")

    t = NOT.read_text(encoding="utf-8")
    assert t.startswith("# UYGUNLUK NOTU"), "başlık kayboldu"
    for bolum in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8."):
        assert bolum in t, f"bölüm kayboldu: {bolum}"
    for gerekli in ("A_pct_en", "J_us_claims_en", "unsupported_claims", "important_issues",
                    "99.66666666666667", "## 3."):
        assert gerekli in t, f"içerik kaybı: {gerekli!r}"
    print(f"§2 yeniden kuruldu: {len(KOSULAR)} koşu satırı, her hücre kendi satırında")
    print(f"not {len(t.splitlines())} satır")
    return 0


if __name__ == "__main__":
    sys.exit(main())
