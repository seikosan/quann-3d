#!/usr/bin/env python3
"""Patent-disi (NPL) on sanat taramasi — OpenAlex + arXiv, ham kayitli.

Neden bu kanal: patent veritabani API'lerinin dordunun de anahtari yok
(EPO OPS / USPTO ODP / SerpApi / BigQuery — bkz. UYGUNLUK_NOTU), Google Patents
dogrudan istemciye 503 donuyor. OpenAlex ve arXiv anahtarsiz acik API'lerdir ve
bu turda HTTP 200 dogrulandi. Cikti: her sorgunun ham yaniti + kisa dizin.

Alan adlari API'den oldugu gibi kopyalanir; bu betik sayi URETMEZ, yalniz
API'nin kendi alanlarini (meta.count / total_num_results vb.) kaydeder.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "npl"
OUT.mkdir(parents=True, exist_ok=True)
UA = "quann3d-priorart/1.0 (research prior-art sweep; mailto:serkankose@users.noreply.github.com)"
MAILTO = "serkankose@users.noreply.github.com"

OA = [
    ("n01", "point_cloud_lm", 'point cloud representation language model token'),
    ("n02", "deep_sets", 'deep sets permutation invariant neural network'),
    ("n03", "pointnet_set_abstraction", 'pointnet hierarchical feature aggregation point cloud'),
    ("n04", "latent_explainability", 'explainability hidden representation point cloud interpretation network'),
    ("n05", "occlusion_ablation_lm", 'occlusion ablation input removal language model explanation'),
    ("n06", "cluster_labeling", 'clustering latent representations automatic labelling interpretability'),
    ("n07", "drift_divergence", 'population drift latent distribution divergence explainability'),
    ("n08", "zeroth_order", 'zeroth-order derivative-free training neural network backpropagation alternative'),
    ("n09", "tabular_embedding", 'tabular data deep learning embedding row as set of vectors'),
    ("n10", "quantization_edge_nlp", 'INT8 post-training quantization on-device language model deployment'),
    ("n11", "hybrid_grad_free", 'hybrid gradient based and gradient free parameter update training'),
    ("n12", "sentence_window_ctx", 'sentence level embedding sliding window document representation aggregation'),
]
AX = [
    ("a01", 'all:"point cloud" AND all:"language model"', 'pclm'),
    ("a02", 'all:"set aggregation" AND all:"permutation invariant"', 'set_agg'),
    ("a03", 'all:"zeroth-order" AND all:"language model" AND all:"training"', 'zo_lm'),
    ("a04", 'all:"permutation invariant" AND all:"language model"', 'pi_lm'),
    ("a05", 'all:"point cloud" AND all:"explainability" AND all:"transformer"', 'pcl_xai'),
    ("a06", 'all:"tabular" AND all:"point cloud" AND all:"classification"', 'tab_cloud'),
    ("a07", 'all:"total variation" AND all:"cluster" AND all:"distribution shift"', 'tv_shift'),
]


def cek(url: str, zaman: int = 60) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=zaman) as r:  # noqa: S310
        return r.status, r.read().decode("utf-8", "replace")


def openalex(qid: str, slug: str, metin: str) -> dict:
    q = {"filter": f"title_and_abstract.search:{metin}", "per-page": "6",
         "select": "id,doi,title,display_name,publication_year,cited_by_count,type,authorships",
         "mailto": MAILTO, "sort": "cited_by_count:desc"}
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(q)
    kayit = {"id": qid, "kanal": "OpenAlex", "sorgu": metin, "url": url, "durum": None,
             "toplam": None, "hitler": [], "zaman": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        st, govde = cek(url)
        kayit["durum"] = st
        (OUT / f"{qid}_{slug}_ham.json").write_text(govde, encoding="utf-8")
        v = json.loads(govde)
        kayit["toplam"] = v.get("meta", {}).get("count")
        for w in v.get("results", [])[:6]:
            yz = [(a.get("author") or {}).get("display_name") for a in (w.get("authorships") or [])][:3]
            kayit["hitler"].append({"baslik": w.get("display_name"), "yil": w.get("publication_year"),
                                    "atif": w.get("cited_by_count"), "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                                    "tur": w.get("type"), "yazarlar": yz, "id": w.get("id")})
    except Exception as e:  # noqa: BLE001
        kayit["durum"] = "HATA"
        kayit["not"] = f"{type(e).__name__}: {e}"
    return kayit


def arxiv(qid: str, sorgu: str, slug: str) -> dict:
    url = ("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(sorgu)
           + "&start=0&max_results=5&sortBy=relevance")
    kayit = {"id": qid, "kanal": "arXiv", "sorgu": sorgu, "url": url, "durum": None,
             "toplam": None, "hitler": [], "zaman": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        st, govde = cek(url, 90)
        kayit["durum"] = st
        (OUT / f"{qid}_{slug}_ham.xml").write_text(govde, encoding="utf-8")
        kök = ET.fromstring(govde)
        ns = {"a": "http://www.w3.org/2005/Atom", "os": "http://a9.com/-/spec/opensearch/1.1/"}
        kayit["toplam"] = (kök.findtext("os:totalResults", namespaces=ns) or "").strip() or None
        for e in kök.findall("a:entry", ns):
            kayit["hitler"].append({
                "baslik": re.sub(r"\s+", " ", (e.findtext("a:title", "", ns) or "")).strip(),
                "yil": (e.findtext("a:published", "", ns) or "")[:10],
                "url": (e.findtext("a:id", "", ns) or "").strip(),
                "yazarlar": [x.findtext("a:name", "", ns) for x in e.findall("a:author", ns)][:3],
            })
    except Exception as e:  # noqa: BLE001
        kayit["durum"] = "HATA"
        kayit["not"] = f"{type(e).__name__}: {e}"
    return kayit


def main() -> int:
    dizin = []
    for i, (qid, slug, metin) in enumerate(OA):
        k = openalex(qid, slug, metin)
        dizin.append(k)
        print(f"{qid} [{k['durum']}] toplam={k['toplam']} hit={len(k['hitler'])} :: {metin[:52]}", flush=True)
        if i < len(OA) - 1:
            time.sleep(8)
    for i, (qid, sorgu, slug) in enumerate(AX):
        k = arxiv(qid, sorgu, slug)
        dizin.append(k)
        print(f"{qid} [{k['durum']}] toplam={k['toplam']} hit={len(k['hitler'])} :: {sorgu[:52]}", flush=True)
        if i < len(AX) - 1:
            time.sleep(4)

    with (OUT / "index.jsonl").open("w", encoding="utf-8") as f:
        for k in dizin:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    s = ["# NPL on sanat taramasi — OpenAlex + arXiv (uretim: npl_tara.py)\n"]
    for k in dizin:
        s.append(f"\n## {k['id']} [{k['kanal']}] — `{k['sorgu']}`")
        s.append(f"- HTTP `{k['durum']}` · toplam `{k['toplam']}` · zaman `{k['zaman']}`")
        if k.get("not"):
            s.append(f"- not: {k['not']}")
        for h in k["hitler"]:
            if k["kanal"] == "OpenAlex":
                s.append(f"  - {h['yil']} · atif {h['atif']} · {h['baslik'][:78]} · DOI {h['doi'] or '-'}")
            else:
                s.append(f"  - {h['yil']} · {h['baslik'][:78]} · {h['url']}")
    (OUT / "index_ozet.txt").write_text("\n".join(s) + "\n", encoding="utf-8")
    print("->", OUT / "index.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
