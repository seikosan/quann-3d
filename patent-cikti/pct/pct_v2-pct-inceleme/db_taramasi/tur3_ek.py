#!/usr/bin/env python3
"""3. tur ek sorgular: isim cakismasi (V3) + bosluk sorgulari (istem 1 kombinasyonu).

Ayni kanal ve ayni disiplin: r.jina.ai -> patents.google.com/xhr/query, ham yanit
diske, sorgu + HTTP + toplam API alani index2.jsonl'e. Sayi uretilmez.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BURADA = Path(__file__).resolve().parent
sys.path.insert(0, str(BURADA))
from gp_tara import UA, cek, kayitlar, hedef_url, soy  # noqa: E402

SORGULAR = [
    ("x01", "ad_quann", 'ti:QUANN'),
    ("x02", "ad_cloudlm", 'ti:"cloud language model" OR ti:CloudLM'),
    ("x03", "ctx_enc_reader_cloud", '"context encoder" "point cloud" "reader" "next token"'),
    ("x04", "region_explain_combo", '"point cloud" explanation "clustering" "region" "language model"'),
    ("x05", "tabular_set_agg", '"tabular record" OR "tabular data" "set aggregation" "class assignment"'),
    ("x06", "no_separate_explainer", 'explanation "without a separate" model "point cloud" inference'),
    ("x07", "accept_revert_train", 'training "accept" "revert" candidate parameter change measured loss'),
]


def main() -> int:
    dizin = []
    for i, (qid, slug, sorgu) in enumerate(SORGULAR):
        url = hedef_url(sorgu)
        k = {"id": qid, "slug": slug, "sorgu": sorgu, "url": url,
             "kanal": "r.jina.ai -> patents.google.com/xhr/query", "durum": None,
             "toplam_sonuc": None, "hitler": [],
             "zaman": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            st, govde = cek(url)
            k["durum"] = st
            ham = soy(govde)
            (BURADA / f"{qid}_{slug}.json").write_text(ham, encoding="utf-8")
            v = json.loads(ham)
            k["toplam_sonuc"] = v.get("results", {}).get("total_num_results")
            k["hitler"] = kayitlar(v)[:5]
        except Exception as e:  # noqa: BLE001
            k["durum"] = "HATA"
            k["not"] = f"{type(e).__name__}: {e}"
        dizin.append(k)
        print(f"{qid} [{k['durum']}] toplam={k['toplam_sonuc']} :: {sorgu[:58]}", flush=True)
        if i < len(SORGULAR) - 1:
            time.sleep(18)
    with (BURADA / "index2.jsonl").open("w", encoding="utf-8") as f:
        for k in dizin:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    s = ["# 3. tur ek sorgular — isim cakismasi + bosluk (uretim: tur3_ek.py)\n"]
    for k in dizin:
        s.append(f"\n## {k['id']} — `{k['sorgu']}`")
        s.append(f"- HTTP `{k['durum']}` · toplam `{k['toplam_sonuc']}` · `{k['zaman']}`")
        s.append(f"- URL: `{k['url']}`")
        if k.get("not"):
            s.append(f"- not: {k['not']}")
        for h in k["hitler"]:
            s.append(f"  - **{h['yayim_no']}** {h['baslik'][:70]} · onculluk {h['onculluk']} · {h['atfedilen']}")
    (BURADA / "index2_ozet.txt").write_text("\n".join(s) + "\n", encoding="utf-8")
    print("->", BURADA / "index2.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
