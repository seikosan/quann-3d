#!/usr/bin/env python3
"""final_v3 patent paketi icin ABD (USPTO) icicerik + bicim analizi.

Motor: Claude-Patent-Creator (patent-creator MCP sunucusunun kullandigi ayni
siniflar). MCP sarmalayicisi `mcp_server/tools/analyzer_tools.py:71-121` sadece
su cagrilari yapip sonucu oldugu gibi geciriyor:
    ClaimsAnalyzer().analyze_claims(claims_text)
    SpecificationAnalyzer().analyze_specification_support(parsed_claims, spec)
    FormalitiesChecker().check_all_formalities(abstract, title, spec, drawings_present)
Bu betik ayni cagrilari kosar; tek fark: metin dosyadan HAM BAYT olarak okunur
(elle yeniden yazim yok => tek hanelik rakam kaymasi riski yok). MPEP atiflari
ayrica search_mpep araci ile alinir (o blok MCP tarafinda index aramasi yapar).

Kosular:
  A) claims  : varsayilan (antecedent_basis KAPALI -> checks_skipped'de gorunur)
  B) claims  : PATENT_ENABLE_ANTECEDENT_CHECK=1 (opt-in, yuksek yanlis-pozitif)
  C) spec    : 112(a) yazili tanimlama + canlandirma
  D) formal  : ozet varyant 1 = OZET.md paragraf 5 (anlati paragrafi)
  E) formal  : ozet varyant 2 = OZET.md govdesi (5 + 7 + 9, olcum paragrafi dahil)
"""

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SITE = Path("/Users/serkankose/patent-creator-venv/lib/python3.13/site-packages")
sys.path.insert(0, str(SITE / "mcp_server"))

from claims_analyzer import ClaimsAnalyzer  # noqa: E402
from formalities_checker import FormalitiesChecker  # noqa: E402
from specification_analyzer import SpecificationAnalyzer  # noqa: E402

PKG = Path("/Users/serkankose/Documents/quann-3d/docs/patent/final_v3")
OUT = Path(__file__).resolve().parent

TITLE_PREFIX = "**Buluşun başlığı:**"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def split_lines(t: str) -> dict[int, str]:
    return {i: ln for i, ln in enumerate(t.split("\n"), start=1)}


def main() -> int:
    claims_raw = read(PKG / "ISTEMLER.md")
    spec_raw = read(PKG / "TARIFNAME.md")
    abstract_raw = read(PKG / "OZET.md")

    # --- parca ayirma (dosya satir numaralari korunur) ---
    il = split_lines(claims_raw)
    al = split_lines(abstract_raw)
    title = None
    for ln in (al.get(3, ""), il.get(3, "")):
        if ln.startswith(TITLE_PREFIX):
            title = ln[len(TITLE_PREFIX):].strip()
            break
    abstract_short = al.get(5, "").strip()
    abstract_full = "\n\n".join(
        s for s in (al.get(5, "").strip(), al.get(7, "").strip(), al.get(9, "").strip()) if s
    )

    def task_claims(enable_antecedent: bool) -> dict:
        if enable_antecedent:
            os.environ["PATENT_ENABLE_ANTECEDENT_CHECK"] = "1"
        else:
            os.environ.pop("PATENT_ENABLE_ANTECEDENT_CHECK", None)
        try:
            res = ClaimsAnalyzer().analyze_claims(claims_raw)
            return {"ok": True, "antecedent_env": os.environ.get("PATENT_ENABLE_ANTECEDENT_CHECK"), "result": res}
        except Exception as e:  # pragma: no cover
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}

    def task_spec() -> dict:
        try:
            parsed = ClaimsAnalyzer()._parse_claims(claims_raw)
            res = SpecificationAnalyzer().analyze_specification_support(parsed, spec_raw)
            return {"ok": True, "parsed_claim_count": len(parsed), "result": res}
        except Exception as e:  # pragma: no cover
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}

    def task_formal(abstract_text: str) -> dict:
        try:
            res = FormalitiesChecker().check_all_formalities(
                abstract=abstract_text,
                title=title,
                specification=spec_raw,
                drawings_present=True,
            )
            return {"ok": True, "result": res}
        except Exception as e:  # pragma: no cover
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}

    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {
            "claims_A_default": ex.submit(task_claims, False),
            "claims_B_antecedent": ex.submit(task_claims, True),
            "spec_C_112a": ex.submit(task_spec),
            "formal_D_abstract_short": ex.submit(task_formal, abstract_short),
            "formal_E_abstract_full": ex.submit(task_formal, abstract_full),
        }
        out = {k: f.result() for k, f in futs.items()}

    out["_giris"] = {
        "title": title,
        "title_chars": len(title or ""),
        "claims_bytes": len(claims_raw.encode()),
        "spec_bytes": len(spec_raw.encode()),
        "abstract_short_words": len(abstract_short.split()),
        "abstract_full_words": len(abstract_full.split()),
        "abstract_short_source": "OZET.md:5",
        "abstract_full_source": "OZET.md:5+7+9",
        "drawings_present": True,
        "drawings_note": "docs/patent/final_v3/fig/*.svg = 8 dosya + sekiller_v3.pdf",
    }
    for k, v in out.items():
        (OUT / f"{k}.json").write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["_giris"], ensure_ascii=False, indent=2))
    for k, v in out.items():
        if k == "_giris":
            continue
        if not v.get("ok"):
            print(f"[HATA] {k}: {v.get('error')}")
            continue
        r = v["result"]
        keys = [
            "claim_count", "independent_claims", "dependent_claims", "compliance_score",
            "total_issues", "critical_issues", "important_issues", "minor_issues",
            "specification_paragraphs", "indexed_terms", "written_description_issues",
            "enablement_issues", "spec_coverage", "compliant",
        ]
        compact = {kk: r[kk] for kk in keys if kk in r}
        print(f"\n=== {k} ===")
        print(json.dumps(compact, ensure_ascii=False))
        print("checks_run:", r.get("checks_run"))
        print("checks_skipped:", json.dumps(r.get("checks_skipped"), ensure_ascii=False))
        print("issues:", len(r.get("issues") or []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
