#!/usr/bin/env python3
"""pct_v2 PCT paketi için biçim + içerik analizi (motor: Claude-Patent-Creator).

Neden MCP aracına elle metin yazmak yerine doğrudan motor: ~60 KB başvuru
metnini araç çağrısının argümanına elle kopyalamak tek hanelik rakam kayması
riski taşır. Burada metin DOSYADAN HAM BAYT olarak okunur. MCP sarmalayıcıları
(mcp_server/tools/*.py) aynı sınıfları çağırıp sonucu olduğu gibi geçirir.

Motorun bilinen sınırı (her raporda yazılır): bulgular sezgisel (regex/istatistik)
ve İNGİLİZCE kalıplara göredir; TR metinde İngilizce bölüm başlığı arar. Bu
yüzden "0 bulgu" temizlik kanıtı değildir, "bulgu yok" anlamına gelir.

Kosular:
  A) PCT biçim  : EN başvuru metni = tarifname + istemler (Kural 5/6 taraması)
  B) PCT biçim  : TR başvuru metni = tariname + istemler (İngilizce regex sınırı)
  C) PCT biçim  : yalnız EN tarifname (istem bölümü ayrı dosyada -> motor ne diyor?)
  D) EPO istem  : CLAIMS filing copy (EN)
  E) EPO istem  : ISTEMLER filing copy (TR)
  F) EPO tarifname-istem desteği : EN istemler + EN tarifname
  G) EPO biçim  : EN başvuru metni
  H) Paket bütünlüğü: docs/patent/pct_v2 (çalışma nüshaları + filing/)
  I) Paket bütünlüğü: docs/patent/pct_v2/filing (yalnız basılacak metinler)
  J) ABD istem denetimi (ClaimsAnalyzer): EN istemler — istem kalitesi taşıyıcı mı?
"""

from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SITE = Path("/Users/serkankose/patent-creator-venv/lib/python3.13/site-packages")
sys.path.insert(0, str(SITE / "mcp_server"))

from claims_analyzer import ClaimsAnalyzer  # noqa: E402
from epo_claims_analyzer import EPOClaimsAnalyzer  # noqa: E402
from epo_formalities_checker import EPOFormalitiesChecker  # noqa: E402
from epo_specification_analyzer import EPOSpecificationAnalyzer  # noqa: E402
from package_checker import PackageChecker  # noqa: E402
from pct_formalities_checker import PCTFormalitiesChecker  # noqa: E402

PKG = Path("/Users/serkankose/Documents/quann-3d/docs/patent/pct_v2")
FIL = PKG / "filing"
OUT = Path(__file__).resolve().parent


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def title_of(text: str) -> str:
    for ln in text.split("\n"):
        for label in ("Title of the invention: ", "Buluşun başlığı: "):
            if ln.startswith(label):
                return ln[len(label):].strip()
    raise ValueError("başlık satırı bulunamadı")


def run(fn, **kw) -> dict:
    try:
        return {"ok": True, "result": fn(**kw) if kw else fn()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}


def main() -> int:
    desc_en = read(FIL / "DESCRIPTION_FILING-COPY_EN.txt")
    claims_en = read(FIL / "CLAIMS_FILING-COPY_EN.txt")
    abst_en = read(FIL / "ABSTRACT_FILING-COPY_EN.txt").strip()
    desc_tr = read(FIL / "TARIFNAME_FILING-COPY_TR.txt")
    claims_tr = read(FIL / "ISTEMLER_FILING-COPY_TR.txt")
    abst_tr = read(FIL / "OZET_FILING-COPY_TR.txt").strip()

    app_en = desc_en.rstrip() + "\n\n" + claims_en
    app_tr = desc_tr.rstrip() + "\n\n" + claims_tr
    t_en, t_tr = title_of(desc_en), title_of(desc_tr)
    parsed_en = EPOClaimsAnalyzer()  # yalnizca referans; parse asagida

    with ThreadPoolExecutor(max_workers=6) as ex:
        f = {
            "A_pct_en": ex.submit(
                run, fn=PCTFormalitiesChecker().check_all_formalities,
                abstract=abst_en, title=t_en, specification=app_en, drawings_present=True),
            "B_pct_tr": ex.submit(
                run, fn=PCTFormalitiesChecker().check_all_formalities,
                abstract=abst_tr, title=t_tr, specification=app_tr, drawings_present=True),
            "C_pct_en_desc_only": ex.submit(
                run, fn=PCTFormalitiesChecker().check_all_formalities,
                abstract=abst_en, title=t_en, specification=desc_en, drawings_present=True),
            "D_epo_claims_en": ex.submit(run, fn=EPOClaimsAnalyzer().analyze_claims, claims_text=claims_en),
            "E_epo_claims_tr": ex.submit(run, fn=EPOClaimsAnalyzer().analyze_claims, claims_text=claims_tr),
            "G_epo_formal_en": ex.submit(
                run, fn=EPOFormalitiesChecker().check_all_formalities,
                abstract=abst_en, title=t_en, specification=app_en, drawings_present=True),
            "H_pkg_full": ex.submit(run, fn=PackageChecker().analyze, directory=str(PKG)),
            "I_pkg_filing": ex.submit(run, fn=PackageChecker().analyze, directory=str(FIL)),
            "J_us_claims_en": ex.submit(run, fn=ClaimsAnalyzer().analyze_claims, claims_text=claims_en),
        }
        out = {k: x.result() for k, x in f.items()}

    def spec_support() -> dict:
        parsed = ClaimsAnalyzer()._parse_claims(claims_en)
        return {"ok": True, "parsed_claim_count": len(parsed),
                "result": EPOSpecificationAnalyzer().analyze_specification_support(parsed, desc_en)}

    out["F_epo_spec_support_en"] = run(spec_support)

    out["_giris"] = {
        "title_en": t_en, "title_en_words": len(t_en.split()),
        "title_tr": t_tr, "title_tr_words": len(t_tr.split()),
        "abstract_en_words": len(abst_en.split()), "abstract_tr_words": len(abst_tr.split()),
        "app_en_bytes": len(app_en.encode()), "app_tr_bytes": len(app_tr.encode()),
        "desc_en_bytes": len(desc_en.encode()), "claims_en_bytes": len(claims_en.encode()),
        "drawings_present": True,
        "drawings_note": "8 SVG: docs/patent/final_v3/fig/fig1..fig8 (kaynak; pct_v2 paylasir)",
        "combined_text_note": (
            "Motor Kural 6 istem taramasını 'specification' argümanı üzerinde yapar "
            "(pct_formalities_checker.py:420-438: CLAIMS başlığı ya da numara kalıbı aranır); "
            "bu yüzden A/B kosularında tarifname+istemler tek metin verildi. C kosusu tek başına "
            "tarifname verildiğinde motorun ne dediğini gösterir."),
    }

    for k, v in out.items():
        (OUT / f"{k}.json").write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(out["_giris"], ensure_ascii=False, indent=2))
    for k, v in out.items():
        if k == "_giris":
            continue
        print(f"\n=== {k} ===")
        if not v.get("ok"):
            print(f"[HATA] {v.get('error')}")
            continue
        r = v["result"]
        for key in ("overall_compliant", "compliance_score", "package_integrity_score",
                    "total_issues", "critical_issues", "important_issues", "minor_issues",
                    "claim_count", "checks_run"):
            if key in r:
                print(f"{key}: {r[key]}")
        if "compliance_summary" in r:
            print("compliance_summary:", json.dumps(r["compliance_summary"], ensure_ascii=False)[:600])
        if "checks_skipped" in r:
            print("checks_skipped:", json.dumps(r["checks_skipped"], ensure_ascii=False)[:500])
        iss = r.get("issues") or []
        print("issues:", len(iss))
        for i in iss[:12]:
            print("  -", i.get("severity"), "|", i.get("legal_ref") or i.get("check") or i.get("rule"),
                  "|", (i.get("problem") or "")[:110])
    return 0


if __name__ == "__main__":
    sys.exit(main())
