#!/usr/bin/env python3
"""INCELEME_RAPORU_2026-11-20.md isaretlerini ham JSON kayitlariyla doldurur.

Anlati sabit; ozet/sayi/bulgu izleri elle yazilmaz, kayit dosyalarindan okunur.
Kosum: python3 rapor_doldur.py   (bu dizinden)
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

D = Path(__file__).resolve().parent
Rapor = D / "INCELEME_RAPORU_2026-11-20.md"


def kayit(dosya: str) -> dict:
    v = json.loads((D / dosya).read_text(encoding="utf-8"))
    v = v.get("result", v)
    return v.get("sonuc", v)


def ozet(dosya: str) -> str:
    return str(kayit(dosya).get("summary")).strip()


def refs(dosya: str) -> str:
    r = kayit(dosya).get("mpep_references") or kayit(dosya).get("epc_references") or []
    return " · ".join(f"`{x.get('section')}` (s.{x.get('page')})" for x in r) or "(çıktı yok)"


def olaylar(dosya: str) -> str:
    out = []
    for n, it in enumerate(kayit(dosya).get("issues", []), 1):
        out.append(f"  {n}. " + " · ".join(f"`{k}`: {v}" for k, v in it.items()))
    return "\n".join(out) or "  (bulgu yok)"


def iz_c(dosya: str) -> str:
    out = []
    for n, it in enumerate(kayit(dosya).get("issues", []), 1):
        prg = " · ".join(it.get("spec_refs") or []) or "-"
        out.append(f"{n}. istem **{it.get('claim')}** · `{it.get('element')}` · {prg} · {it.get('severity')}"
                   f" · `{it.get('type')}` · güven {it.get('confidence')} · araç atfı "
                   f"{it.get('mpep') or it.get('legal_ref') or '-'}")
    return "\n".join(out)


def iz_b() -> str:
    out = []
    for n, it in enumerate(kayit("B_us_claims_en_antecedentON.json").get("issues", []), 1):
        out.append(f"{n}. istem **{it.get('claim')}** · `{it.get('term')}` · konum `{it.get('location')}` · "
                   f"{it.get('severity')} · güven {it.get('confidence')}")
    return "\n".join(out)


def atlanan(dosya: str, kac: int = 0) -> str:
    sk = kayit(dosya).get("checks_skipped") or []
    if not sk:
        return "(atlanmış kontrol yok)"
    return str(sk[kac].get("reason")) if len(sk) > kac else str(sk[-1].get("reason"))


def b_dagilim() -> str:
    say = collections.Counter(i.get("claim") for i in kayit("B_us_claims_en_antecedentON.json").get("issues", []))
    return " · ".join(f"istem {k}: {v}" for k, v in sorted(say.items()))


m = Rapor.read_text(encoding="utf-8")
DOLGU = {
    "@@OZET_A@@": ozet("A_us_claims_en_default.json"),
    "@@RUN_A@@": " · ".join(f"`{c}`" for c in kayit("A_us_claims_en_default.json").get("checks_run", [])),
    "@@REFS_A@@": refs("A_us_claims_en_default.json"),
    "@@OZET_C@@": ozet("C_us_spec_support_en.json"),
    "@@REFS_C@@": refs("C_us_spec_support_en.json"),
    "@@İZ_C@@": iz_c("C_us_spec_support_en.json"),
    "@@OZET_D@@": ozet("D_us_formal_abstract_filingcopy.json"),
    "@@TAM_D@@": olaylar("D_us_formal_abstract_filingcopy.json"),
    "@@OZET_E@@": ozet("E_epo_claims_en.json"),
    "@@TAM_E@@": olaylar("E_epo_claims_en.json"),
    "@@OZET_F@@": ozet("F_epo_spec_support_en.json"),
    "@@OZET_H@@": ozet("H_epo_formal_en.json"),
    "@@TAM_H@@": olaylar("H_epo_formal_en.json"),
    "@@OZET_K@@": ozet("K_pct_formal_en.json"),
    "@@TAM_K@@": olaylar("K_pct_formal_en.json"),
    "@@OZET_L@@": ozet("L_package_filing.json"),
    "@@OZET_ETR@@": ozet("ETR_epo_claims_tr.json"),
    "@@OZET_FTR@@": ozet("FTR_epo_spec_support_tr.json"),
    "@@OZET_B@@": ozet("B_us_claims_en_antecedentON.json"),
    "@@B_DAGILIM@@": b_dagilim(),
    "@@İZ_B@@": iz_b(),
    "@@SK_A@@": atlanan("A_us_claims_en_default.json"),
    "@@SK_D@@": atlanan("D_us_formal_abstract_filingcopy.json"),
    "@@SK_H@@": atlanan("H_epo_formal_en.json"),
    "@@SK_K1@@": atlanan("K_pct_formal_en.json", 0),
    "@@SK_K2@@": atlanan("K_pct_formal_en.json", 1),
}
for isaret, deger in DOLGU.items():
    assert m.count(isaret) == 1, isaret
    m = m.replace(isaret, deger)
assert "@@" not in m, "doldurulmamış işaret kaldı"
Rapor.write_text(m, encoding="utf-8")
print("dolduruldu:", Rapor.name, len(m.splitlines()), "satır")
