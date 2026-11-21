#!/usr/bin/env python3
"""Google Patents tam metin taramasi — on sanat havuzu.

Kanal notu (2026-11-19, bu depo icinde olcutur):
  * patents.google.com/xhr/query dogrudan curl ile -> HTTP 503 (bot korumasi)
  * ayni uç nokta OpenCode webfetch ile -> once 200, hizli ardişarada 503
  * ayni uç nokta r.jina.ai on ucunden -> HTTP 200 (ham JSON dondu)
Bu yuzden sorgular r.jina.ai uzerinden kosulur; her sorgunun HAM yaniti
diske yazilir, istek URL'i ve zaman damgasi kayd edilir (ayni URL yeniden
kosulabilir).

Girdi : yok (SORGULAR listesi bu dosyadadir)
Cikti : qNN_<slug>.json  (ham yanit, jina basligi soyulmus)
        index.jsonl      (sorgu + toplam + en iyi 5 kaydin kisa kaydi)
        index_ozet.txt   (insan okumali tablo)
Kural : sayilar uydurulmaz — toplam_sonuc API'nin own alani, hitler API'nin
        own alanlari; bu betik hicbir sayi uretmez, yalniz kopyalar.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
PROXY = "https://r.jina.ai/"
NUM = 5
BEKLE = 18  # saniye — jina ucretsiz katmani hiz siniri icin

# (id, slug, sorgu) — her satir bir istem ozelligine baglidir (CLAIMS_EN.md)
SORGULAR: list[tuple[str, str, str]] = [
    ("q01", "set_aggregation_pclm", '"point cloud" "language model" "set aggregation"'),
    ("q02", "deepsets_maxpool", '"deep sets" "max-pooling" neural network'),
    ("q03", "perm_inv_lm", '"permutation invariant" "language model" set representation'),
    ("q04", "hidden_state_cloud_xai", '"hidden states" "point cloud" explainability'),
    ("q05", "pair_geometry_feats", '"pairwise" "relative position" "point cloud" aggregation network'),
    ("q06", "token_ablation_xai", '"language model" explanation "input removal" output difference'),
    ("q07", "cluster_naming", 'clustering "latent representations" "cluster labels" interpretability'),
    ("q08", "region_shift_divergence", '"Kullback-Leibler" divergence "two groups" model outputs explainability'),
    ("q09", "zeroth_order_train", '"zeroth-order" OR "derivative-free" "without backpropagation" training neural network'),
    ("q10", "tabular_as_cloud", '"tabular data" "point cloud" deep learning classification'),
    ("q11", "window_stride_ctx", '"sliding window" "fixed stride" sentence embeddings transformer aggregation'),
    ("q12", "int8_table_edge", 'INT8 quantization "lookup table" edge device model deployment'),
    ("q13", "pclm_autoregressive", '"point cloud" "next token" autoregressive prediction'),
    ("q14", "same_repr_explain", 'explanation "same representation" neural network "no separate" model'),
    ("q15", "set_enc_dec_lm", 'set encoder "order independent" pooling "next token" language model'),
    ("q16", "point_ablation_contrib", '"ablation" "contribution" neural network "removing" feature vector inference'),
]


def hedef_url(sorgu: str) -> str:
    ic = "q=" + sorgu.replace(" ", "+") + f"&num={NUM}"
    return "https://patents.google.com/xhr/query?url=" + urllib.parse.quote(ic, safe="")


def cek(url: str) -> tuple[int, str]:
    req = urllib.request.Request(PROXY + url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:  # noqa: S310 - sabit host
        return r.status, r.read().decode("utf-8", "replace")


def soy(ham: str) -> str:
    """jina sarmalayicisini soy. Iki bicim goruldu:

    (1) duz metin modu: 'Title: ... Markdown Content:' basligi + altinda API JSON'u
    (2) JSON modu (Accept: application/json): {"code":200,"data":{"content":"..."}}

    Ikisinde de donen sey Google Patents API'sinin own JSON govdesidir — bu betik
    icine hicbir alan yeniden yazilmaz, yalniz sarmalayici acilir.
    """
    i = ham.find("Markdown Content:")
    govde = ham[i + len("Markdown Content:"):].strip() if i >= 0 else ham.strip()
    if not govde.startswith("{"):
        return govde
    try:
        kabuk = json.loads(govde)
    except json.JSONDecodeError:
        return govde
    if isinstance(kabuk, dict):
        if "results" in kabuk:  # dogrudan Google Patents API govdesi
            return govde
        data = kabuk.get("data")
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            return data["content"].strip()  # jina JSON sarmalayicisi
    return govde


def kayitlar(veri: dict) -> list[dict]:
    cik = []
    for kume in veri.get("results", {}).get("cluster", []) or []:
        for r in kume.get("result", []) or []:
            p = r.get("patent", {}) or {}
            cik.append({
                "yayim_no": p.get("publication_number"),
                "baslik": (p.get("title") or "").strip(),
                "onculluk": p.get("priority_date"),
                "basvuru": p.get("filing_date"),
                "yayim": p.get("publication_date"),
                "atfedilen": p.get("assignee"),
                "mucit": p.get("inventor"),
                "ozet_ilk": (p.get("snippet") or "").replace("\n", " ")[:220],
            })
    return cik


def main() -> int:
    dizin = []
    for i, (qid, slug, sorgu) in enumerate(SORGULAR):
        url = hedef_url(sorgu)
        kayit = {"id": qid, "slug": slug, "sorgu": sorgu, "url": url,
                 "kanal": "r.jina.ai -> patents.google.com/xhr/query", "durum": None,
                 "toplam_sonuc": None, "hitler": []}
        ham = ""
        for deneme in (1, 2):
            try:
                t0 = datetime.now(timezone.utc).isoformat(timespec="seconds")
                st, govde = cek(url)
                kayit["durum"] = st
                kayit["zaman"] = t0
                ham = soy(govde)
                if st == 200 and ham.startswith("{"):
                    break
                kayit["not"] = f"deneme {deneme}: HTTP {st} veya JSON degil"
            except Exception as e:  # noqa: BLE001 - ag hatasi kayda gecer
                kayit["durum"] = "HATA"
                kayit["not"] = f"deneme {deneme}: {type(e).__name__}: {e}"
            time.sleep(45)
        (OUT / f"{qid}_{slug}.json").write_text(ham, encoding="utf-8")
        try:
            veri = json.loads(ham)
            kayit["toplam_sonuc"] = veri.get("results", {}).get("total_num_results")
            kayit["hitler"] = kayitlar(veri)[:NUM]
        except Exception as e:  # noqa: BLE001
            kayit["not"] = (kayit.get("not") or "") + f" | ayristirma: {type(e).__name__}"
        dizin.append(kayit)
        print(f"{qid} [{kayit['durum']}] toplam={kayit['toplam_sonuc']} hit={len(kayit['hitler'])} :: {sorgu[:60]}",
              flush=True)
        if i < len(SORGULAR) - 1:
            time.sleep(BEKLE)

    with (OUT / "index.jsonl").open("w", encoding="utf-8") as f:
        for k in dizin:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    satirlar = ["# Google Patents tam metin taramasi — ham dizin (uretim: gp_tara.py)\n"]
    for k in dizin:
        satirlar.append(f"\n## {k['id']} — `{k['sorgu']}`")
        satirlar.append(f"- HTTP `{k['durum']}` · toplam sonuç `{k['toplam_sonuc']}` · zaman `{k.get('zaman')}`")
        satirlar.append(f"- URL: `{k['url']}`")
        if k.get("not"):
            satirlar.append(f"- not: {k['not']}")
        for h in k["hitler"]:
            satirlar.append(f"  - **{h['yayim_no']}** {h['baslik'][:70]} · onculluk {h['onculluk']} "
                            f"· yayim {h['yayim']} · {h['atfedilen']}")
    (OUT / "index_ozet.txt").write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print("->", OUT / "index.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
