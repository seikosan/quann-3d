#!/usr/bin/env python3
"""B varyanti: istem analizinin OPT-IN `antecedent_basis` kontrolu.

MCP araci (`review_patent_claims`) bu kontrolu varsayilan KAPALI kosar — motor
`checks_skipped` icinde "enable_with: PATENT_ENABLE_ANTECEDENT_CHECK=1" diyor.
Ortam degiskeni MCP sunucusu surecine yuklenemedigi icin ayni sinif burada,
ayni venv python'i ile ve GIRDILER DISK HAM OKUNARAK kosulur (elle yeniden yazim
yok). MCP sarmalayicisinin yaptigi cagri ile birebir ayni sinif/metot:
mcp_server/tools/analyzer_tools.py:81-82  ->  ClaimsAnalyzer().analyze_claims(claims_text)

Kosum:
  PATENT_ENABLE_ANTECEDENT_CHECK=1 \
  /Users/serkankose/patent-creator-venv/bin/python B_antecedent_kos.py
Cikti: B_us_claims_en_antecedentON.json (ham JSON, bu dizine)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mcp_server.claims_analyzer import ClaimsAnalyzer

GIRI = Path("/Users/serkankose/Documents/quann-3d/docs/patent/pct_v2/filing/CLAIMS_FILING-COPY_EN.txt")
OUT = Path(__file__).resolve().parent / "B_us_claims_en_antecedentON.json"


def main() -> None:
    claims_text = GIRI.read_text(encoding="utf-8")
    result = ClaimsAnalyzer().analyze_claims(claims_text)
    kayit = {
        "status": "fulfilled",
        "not": ("opt-in antecedent_basis; MCP araci bu varyanti kosamaz (bayrak MCP sunucusu "
                "surecinde). Sinif/metot MCP sarmalayicisi ile ayni; girdi disk HAM okundu."),
        "env": {"PATENT_ENABLE_ANTECEDENT_CHECK": os.environ.get("PATENT_ENABLE_ANTECEDENT_CHECK")},
        "girdi": str(GIRI),
        "sonuc": result,
    }
    OUT.write_text(json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("yazildi:", OUT)
    print("toplam:", result.get("total_issues"), "| kritik:", result.get("critical_issues"),
          "| onearli:", result.get("important_issues"), "| minör:", result.get("minor_issues"))
    print("kosulanlar:", json.dumps(result.get("checks_run"), ensure_ascii=False)[:400])
    print("atlananlar:", json.dumps(result.get("checks_skipped"), ensure_ascii=False)[:200])
    print("tiplere gore:", json.dumps(result.get("issues_by_type"), ensure_ascii=False)[:400])
    for it in result.get("issues", []):
        sev = it.get("severity") or "?"
        typ = it.get("issue_type") or it.get("type") or "?"
        num = it.get("claim_number") or it.get("claim") or "?"
        term = it.get("term") or it.get("element") or ""
        print(f"  - {sev:9s} | {typ:18s} | istem {num} | {term} | {str(it.get('problem'))[:110]}")


if __name__ == "__main__":
    main()
