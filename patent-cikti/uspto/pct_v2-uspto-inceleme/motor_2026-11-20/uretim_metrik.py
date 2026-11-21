#!/usr/bin/env python3
"""METRIKLER_2026-11-20.md ureteci — ham arac JSON kayitlarindan sayi tablosu.

Sayilar elle yazilmaz: bu dizindeki ham kayit dosyalari okunur. Tekrar kosum:
  /Users/serkankose/patent-creator-venv/bin/python uretim_metrik.py   (bu dizinden)
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
FILING = Path("/Users/serkankose/Documents/quann-3d/docs/patent/pct_v2/filing")
GIRDI = [
    "ABSTRACT_FILING-COPY_EN.txt", "CLAIMS_FILING-COPY_EN.txt", "DESCRIPTION_FILING-COPY_EN.txt",
    "ISTEMLER_FILING-COPY_TR.txt", "OZET_FILING-COPY_TR.txt", "TARIFNAME_FILING-COPY_TR.txt",
]


def guncelle(dosya: str) -> dict:
    v = json.loads((D / dosya).read_text(encoding="utf-8"))
    v = v.get("result", v)
    return v.get("sonuc", v)


ROWS = [
    ("A_us_claims_en_default.json", "review_patent_claims", [("claim_count", "istem"), ("independent_claims", "bağımsız"), ("dependent_claims", "bağımlı"), ("compliance_score", "compliance_score"), ("total_issues", "toplam bulgu"), ("critical_issues", "kritik"), ("important_issues", "önemli"), ("minor_issues", "minör")]),
    ("B_us_claims_en_antecedentON.json", "ClaimsAnalyzer (opt-in antecedent ON)", [("claim_count", "istem"), ("independent_count", "bağımsız"), ("dependent_count", "bağımlı"), ("compliance_score", "compliance_score"), ("total_issues", "toplam bulgu"), ("critical_issues", "kritik"), ("important_issues", "önemli"), ("minor_issues", "minör")]),
    ("C_us_spec_support_en.json", "review_specification", [("specification_paragraphs", "tarifname paragrafı"), ("indexed_terms", "indekslenen terim"), ("total_issues", "toplam bulgu"), ("critical_issues", "kritik"), ("important_issues", "önemli"), ("written_description_issues", "yazılı-tanım"), ("enablement_issues", "enablement"), ("compliant", "compliant alanı")]),
    ("D_us_formal_abstract_filingcopy.json", "check_formalities", [("critical_issues", "kritik"), ("warnings", "uyarı"), ("info", "bilgi"), ("overall_compliant", "overall_compliant"), ("ready_to_file", "ready_to_file")]),
    ("E_epo_claims_en.json", "review_epo_claims", [("claim_count", "istem"), ("independent_claims", "bağımsız"), ("dependent_claims", "bağımlı"), ("total_issues", "toplam bulgu"), ("critical_issues", "kritik"), ("important_issues", "önemli")]),
    ("F_epo_spec_support_en.json", "review_epo_specification", [("specification_paragraphs", "tarifname paragrafı"), ("indexed_terms", "indekslenen terim"), ("total_issues", "toplam bulgu"), ("critical_issues", "kritik"), ("important_issues", "önemli"), ("compliant", "compliant alanı")]),
    ("H_epo_formal_en.json", "check_epo_formalities", [("critical_issues", "kritik"), ("warnings", "uyarı"), ("info", "bilgi"), ("overall_compliant", "overall_compliant"), ("ready_to_file", "ready_to_file")]),
    ("K_pct_formal_en.json", "check_pct_formalities", [("critical_issues", "kritik"), ("warnings", "uyarı"), ("info", "bilgi"), ("overall_compliant", "overall_compliant"), ("ready_to_file", "ready_to_file")]),
    ("ETR_epo_claims_tr.json", "review_epo_claims (TR nüsha)", [("claim_count", "istem"), ("independent_claims", "bağımsız"), ("dependent_claims", "bağımlı"), ("total_issues", "toplam bulgu"), ("critical_issues", "kritik"), ("important_issues", "önemli")]),
    ("FTR_epo_spec_support_tr.json", "review_epo_specification (TR nüsha)", [("specification_paragraphs", "tarifname paragrafı"), ("indexed_terms", "indekslenen terim"), ("total_issues", "toplam bulgu"), ("critical_issues", "kritik"), ("important_issues", "önemli"), ("compliant", "compliant alanı")]),
    ("L_package_filing.json", "check_package", [("files_scanned", "taranan dosya"), ("total_issues", "toplam bulgu"), ("critical_issues", "kritik"), ("package_integrity_score", "paket bütünlük skoru")]),
]
TUM = [r[0] for r in ROWS if r[0] != "L_package_filing.json"]


def main() -> None:
    sat = [
        "# METRIKLER_2026-11-20 — ham araç çıktısından üretilen sayı künyesi",
        "",
        "Bu tablo elle yazılmadı; `uretim_metrik.py` aynı dizindeki ham JSON kayıtlarını okuyarak üretti.",
        "Her satır: `kayıt dosyası | alan | değer`. Kayıt dosyaları bu dizindedir.",
        "",
        "| Kayıt | Alan | Değer |",
        "|---|---|---|",
    ]
    for f in GIRDI:
        b = (FILING / f).read_bytes()
        sat.append(f"| girdi `{f}` | bayt / sha256(16) | {len(b)} · {hashlib.sha256(b).hexdigest()[:16]} |")
    sat.append("")
    for dosya, arac, alanlar in ROWS:
        r = guncelle(dosya)
        for key, tr in alanlar:
            v = r.get(key, "-")
            if isinstance(v, float):
                v = f"{v:.2f}"
            sat.append(f"| `{dosya}` | {arac} · {tr} (`{key}`) | {v} |")
        sat.append("")
    for dosya in ("D_us_formal_abstract_filingcopy.json", "H_epo_formal_en.json", "K_pct_formal_en.json"):
        r = guncelle(dosya)
        for alt in ("abstract", "title", "drawings", "sections", "description", "claims", "terminology", "claims_fees"):
            if isinstance(r.get(alt), dict):
                sat.append(f"| `{dosya}` | {alt} alt alanı | {json.dumps(r[alt], ensure_ascii=False)} |")
        sat.append("")
    sat += ["## Atlanan kontroller (ham `checks_skipped`)", "",
            "| Kayıt | Atlanan | Gerekçe (araç çıktısı) |", "|---|---|---|"]
    for dosya in TUM:
        sk = guncelle(dosya).get("checks_skipped")
        if sk is None:
            sat.append(f"| `{dosya}` | (çıktıda `checks_skipped` alanı yok) | — |")
        elif not sk:
            sat.append(f"| `{dosya}` | (boş liste — hiçbir kontrol atlanmadı) | — |")
        else:
            for s in sk:
                sat.append(f"| `{dosya}` | `{s.get('check')}` | {str(s.get('reason'))[:160]} |")
    sat += ["", "## Bulgu tip dağılımları (ham `issues`)", "",
            "| Kayıt | Bulgu kaydı | Tip dağılımı | Güven dağılımı |", "|---|---|---|---|"]
    for dosya in TUM:
        iss = guncelle(dosya).get("issues", [])
        t = collections.Counter((i.get("type") or i.get("issue_type") or i.get("section") or "?", i.get("severity", "?")) for i in iss)
        conf = collections.Counter(i.get("confidence", "-") for i in iss)
        tipler = " · ".join(f"`{a}`/{b}: {n}" for (a, b), n in t.items()) or "—"
        gun = " · ".join(f"{a}: {n}" for a, n in conf.items()) or "—"
        sat.append(f"| `{dosya}` | {len(iss)} | {tipler} | {gun} |")
    out = D / "METRIKLER_2026-11-20.md"
    out.write_text("\n".join(sat) + "\n", encoding="utf-8")
    print("yazildi:", out, len(sat), "satir")


if __name__ == "__main__":
    main()
