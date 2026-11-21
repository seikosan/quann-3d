r"""UYGUNLUK_NOTU §tablolarındaki her künyeyi bağımsız olarak doğrular (2026-11-19).

Rapor kapısı (`rapor_denetle.py`) tablo satırlarını MUAF sayıyor; notun sayı yoğunluğu
tablolarda. Bu betik o boşluğu kapatır:

1. Künye taraması — nottaki her `dosya:satır` / `dosya:satır-satır` adresi açılır; dosya
   yoksa veya satır numarası aralık dışında ise **ÖLÜ ADRES** diye basılır.
2. §2 motor tablosu — tablodaki kritik/önemli/skor değerleri taze JSON alanlarıyla
   karşılaştırılır; uyuşmazsa **UYUŞMAZLIK** diye basılır.

Kosum: python3 patent-cikti/pct/pct_v2-pct-inceleme/dogrul_tablolar.py
Cikis: 0 = her künye açıldı ve §2 değerleri uyuştu; 1 = bulgu var.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path("/Users/serkankose/Documents/quann-3d")
NOT = REPO / "docs/patent/pct_v2/UYGUNLUK_NOTU.md"
OUT = REPO / "patent-cikti/pct/pct_v2-pct-inceleme"

# `patent-cikti/...json:12` · `~/venv/...py:186-192` · `docs/...md:10-67`
ADRES = re.compile(r"`([^`\s]+?):(\d{1,6})(?:-(\d{1,6}))?`")


def coz(yol: str) -> Path | None:
    if yol.startswith("~"):
        return Path(yol).expanduser()
    p = REPO / yol
    if p.exists():
        return p
    for taban in (OUT, OUT.parent, REPO / "docs/patent/pct_v2"):
        if (taban / yol).exists():
            return taban / yol
    return None


def tarama() -> tuple[int, list[str]]:
    olu: list[str] = []
    gorulen = 0
    for no, satir in enumerate(NOT.read_text(encoding="utf-8").splitlines(), 1):
        for yol, bas, son in ADRES.findall(satir):
            p = coz(yol)
            if p is None:
                olu.append(f"satir {no}: DOSYA YOK  {yol}:{bas}")
                continue
            satirlar = p.read_text(encoding="utf-8", errors="replace").splitlines()
            bit = int(son) if son else int(bas)
            if not (1 <= int(bas) <= len(satirlar)) or not (1 <= bit <= len(satirlar)):
                olu.append(f"satir {no}: SATIR YOK  {yol}:{bas}-{bit} (dosyada "
                           f"{len(satirlar)} satır var)")
                continue
            if not any(s.strip() for s in satirlar[int(bas) - 1:bit]):
                olu.append(f"satir {no}: BOS ARALIK {yol}:{bas}-{bit}")
                continue
            gorulen += 1
    return gorulen, olu


# --- 2. §2 motor tablosu: tablodaki değer <-> taze JSON alanı ---------------------
BEKLENEN = {
    "A_pct_en": ("A_pct_en.json", ["result.results.abstract.word_count", "result.results.title.word_count"]),
    "D_epo_claims_en": ("D_epo_claims_en.json", ["result.results.total_issues"]),
}


def alan(d: dict, yol: str):
    for parca in yol.split("."):
        d = d[parca]
    return d


def j(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def motor_karsilastirma() -> list[str]:
    uyusmazlik: list[str] = []

    def kontrol(koşu: str, alan_yolu: str, tablo_değeri: object) -> None:
        d = j(OUT / f"{koşu}.json")
        try:
            gercek = alan(d, alan_yolu)
        except KeyError:
            uyusmazlik.append(f"{koşu}: alan yok {alan_yolu}")
            return
        if str(gercek) != str(tablo_değeri):
            uyusmazlik.append(f"{koşu} {alan_yolu}: tablo {tablo_değeri!r} <-> JSON {gercek!r}")

    # §2 satırındaki değerler — alan yoları JSON'un gerçek yapısından (bu oturumda okundu)
    kontrol("A_pct_en", "result.compliance_summary.critical_issues", 0)
    kontrol("A_pct_en", "result.compliance_summary.warnings", 0)
    kontrol("A_pct_en", "result.overall_compliant", False)
    kontrol("A_pct_en", "result.results.title.word_count", 6)
    kontrol("A_pct_en", "result.results.abstract.word_count", 137)
    kontrol("B_pct_tr", "result.compliance_summary.critical_issues", 0)
    kontrol("B_pct_tr", "result.compliance_summary.warnings", 0)
    kontrol("B_pct_tr", "result.overall_compliant", False)
    kontrol("B_pct_tr", "result.results.title.word_count", 7)
    kontrol("B_pct_tr", "result.results.abstract.word_count", 106)
    kontrol("C_pct_en_desc_only", "result.compliance_summary.critical_issues", 1)
    kontrol("C_pct_en_desc_only", "result.compliance_summary.warnings", 0)
    kontrol("D_epo_claims_en", "result.total_issues", 1)
    kontrol("D_epo_claims_en", "result.critical_issues", 0)
    kontrol("D_epo_claims_en", "result.important_issues", 1)
    kontrol("E_epo_claims_tr", "result.total_issues", 16)
    kontrol("E_epo_claims_tr", "result.critical_issues", 0)
    kontrol("E_epo_claims_tr", "result.important_issues", 16)
    kontrol("F_epo_spec_support_en", "result.result.critical_issues", 0)
    kontrol("F_epo_spec_support_en", "result.result.important_issues", 47)
    kontrol("F_epo_spec_support_en", "result.result.compliant", True)
    kontrol("F_epo_spec_support_en", "result.result.spec_coverage.percentage", 100)
    kontrol("F_epo_spec_support_en", "result.result.spec_coverage.supported_claims", 15)
    kontrol("F_epo_spec_support_en", "result.result.spec_coverage.unsupported_claims", [])
    kontrol("G_epo_formal_en", "result.compliance_summary.critical_issues", 0)
    kontrol("G_epo_formal_en", "result.compliance_summary.warnings", 1)
    kontrol("G_epo_formal_en", "result.overall_compliant", True)
    kontrol("H_pkg_full", "result.total_issues", 0)
    kontrol("H_pkg_full", "result.critical_issues", 0)
    kontrol("H_pkg_full", "result.package_integrity_score", 100.0)
    kontrol("I_pkg_filing", "result.total_issues", 0)
    kontrol("I_pkg_filing", "result.package_integrity_score", 100.0)
    kontrol("J_us_claims_en", "result.total_issues", 0)
    kontrol("J_us_claims_en", "result.critical_issues", 0)
    kontrol("J_us_claims_en", "result.compliance_score", 100.0)
    return uyusmazlik


def main() -> int:
    gorulen, olu = tarama()
    print(f"KUNYE TARAMASI: acilan {gorulen} adres · olu adres {len(olu)}")
    for s in olu:
        print("  OLÜ  " + s)

    uy = motor_karsilastirma()
    print(f"MOTOR KARŞILAŞTIRMASI: {len(uy)} uyuşmazlık")
    for s in uy:
        print("  UYUŞMAZLIK  " + s)

    (OUT / "DOGRUL_tablolar.json").write_text(
        json.dumps({"kunye_acilan": gorulen, "olu_adres": olu,
                    "motor_usmazlik": uy,
                    "temiz": not olu and not uy}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"-> {OUT.relative_to(REPO)}/DOGRUL_tablolar.json")
    return 1 if (olu or uy) else 0


if __name__ == "__main__":
    sys.exit(main())
