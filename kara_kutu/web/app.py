"""FastAPI + tek sayfa arayüz (P1). Kayıt/kanıt disiplini CLI ile birebir aynıdır:
her koşum `runner.kosum`'dan geçer, `runs/<id>/` dört dosyasını üretir; sayfadaki
sayılar o koşunun ölçümleridir (elle yazılmış sayı yok)."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from kara_kutu.core.runner import RUNS_DIR, kosum

MAX_YUKLE_BAYT = 50 * 1024 * 1024  # 50 MiB — tek sayfalık eval arayüzü için tavan
KOSUM_ID_RE = re.compile(r"^\d{8}-\d{6}_[a-z0-9]+$")
INDIRILEBILIR = {"sonuc.json", "kanit.json", "ic_rapor.md", "dis_rapor.html"}
TABLO_SATIR = 50  # sayfada gösterilecek kayıt tavanı (tam liste sonuc.json'da)

_SAYFA = """<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kara Kutu — eval-only kanıt arayüzü</title><style>
body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:960px;padding:0 1rem;color:#1a1a2e}
h1{font-size:1.4rem} form{display:flex;flex-wrap:wrap;gap:.6rem;align-items:end;
background:#f4f4fa;padding:1rem;border-radius:8px}
label{display:flex;flex-direction:column;font-size:.8rem;gap:.2rem}
input{padding:.35rem;border:1px solid #bbb;border-radius:5px}
button{padding:.5rem 1rem;border:0;border-radius:5px;background:#16407c;color:#fff;cursor:pointer}
table{border-collapse:collapse;width:100%;margin-top:1rem;font-size:.85rem}
td,th{border:1px solid #d5d5e0;padding:.3rem .5rem;text-align:left}
.fraud{color:#b00020;font-weight:600}.kucuk{color:#666;font-size:.78rem}
#ozet b{font-variant-numeric:tabular-nums}</style></head><body>
<h1>Kara Kutu <span class="kucuk">— tahmin + gerekçe + kanıt (eval-only; eğitim yok)</span></h1>
<form id="f"><label>CSV dosyası<input name="dosya" type="file" accept=".csv" required></label>
<label>Eşik (boş = varsayılan)<input name="esik" type="number" step="0.0001" min="0" max="1"
placeholder="0.9969"></label><label>Limit<input name="limit" type="number" min="1"
placeholder="tümü"></label><label>Top-k<input name="top_k" type="number" min="0"
placeholder="5"></label><button type="submit">Koş</button></form>
<p id="durum" class="kucuk">Hazır. Koşumlar <code>kara_kutu/runs/</code> altına yazılır.</p>
<div id="ozet"></div><div id="tablo"></div>
<script>
const $=q=>document.querySelector(q);
function tabloHTML(v){let h='<table><tr><th>#</th><th>skor</th><th>karar</th>'+
 '<th>gerekçeler (Δlogit)</th></tr>';for(const r of v){h+=`<tr><td>${r.kimlik}</td>`+
 `<td>${r.skor.toFixed(6)}</td><td class="${r.karar==='FRAUD'?'fraud':''}">${r.karar}</td>`+
 `<td>${r.gerekceler.join('; ')||'—'}</td></tr>`}return h+'</table>'}
$('#f').addEventListener('submit',async e=>{e.preventDefault();
const fd=new FormData(e.target);$('#durum').textContent='Koşuyor… (ONNX CPU)';
const r=await fetch('/api/kosum',{method:'POST',body:fd});
const j=await r.json();if(!r.ok){$('#durum').textContent='Hata: '+(j.detail||r.status);return}
$('#durum').textContent=`Koşum ${j.ozet.kosum} bitti — ${j.verdictler.length}/${
 j.ozet.satir_sayisi} kayıt gösteriliyor (tam liste sonuc.json).`;
const m=Object.entries(j.ozet.metrikler||{}).map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');
$('#ozet').innerHTML=`<p><b>${j.ozet.satir_sayisi}</b> kayıt · <b>${j.ozet.fraud_sayisi
}</b> FRAUD · eşik <b>${j.ozet.esik.toFixed(4)}</b></p><table><tr><th>Metrik</th>
<th>Değer</th></tr>${m}</table><p>${Object.keys(j.ozet.raporlar).includes('md')
?`<a href="/kosumlar/${j.ozet.kosum}/ic_rapor.md">iç rapor ↓</a> · `:''}${
Object.keys(j.ozet.raporlar).includes('html')
?`<a href="/kosumlar/${j.ozet.kosum}/dis_rapor.html">dış rapor ↓</a> · `:''}
<a href="/kosumlar/${j.ozet.kosum}/kanit.json">kanıt ↓</a> ·
<a href="/kosumlar/${j.ozet.kosum}/sonuc.json">sonuç ↓</a></p>`;
$('#tablo').innerHTML=tabloHTML(j.verdictler);});
</script></body></html>"""


def _uygulama_adi(ad: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(ad).name) or "yukleme.csv"


def _sayi(ad: str, deger: str | None, tip: type) -> float | int | None:
    """Form boş alanları '' gönderir — None'a çevir; bozuk değeri 400 ile reddet."""
    if deger is None or str(deger).strip() == "":
        return None
    try:
        return tip(str(deger).strip())
    except ValueError as exc:
        raise HTTPException(400, f"{ad} sayı olmalı: {deger!r}") from exc


def _verdict_sayfasi(sonuc) -> list[dict]:
    sayfa = []
    for v in sonuc.verdictler[:TABLO_SATIR]:
        sayfa.append({
            "kimlik": v.kimlik, "skor": v.skor, "karar": v.karar,
            "gerekceler": [f"{g.etiket} ({g.delta_logit:+.2f})" for g in v.gerekceler[:5]],
        })
    return sayfa


def create_app(runs_dir: Path = RUNS_DIR) -> FastAPI:
    app = FastAPI(title="Kara Kutu", docs_url=None, redoc_url=None)
    app.state.runs_dir = Path(runs_dir)

    @app.get("/", response_class=HTMLResponse)
    def ana_sayfa() -> str:
        return _SAYFA

    @app.post("/api/kosum")
    async def api_kosum(dosya: UploadFile, adapter: str = Form("fraud"),
                        esik: str | None = Form(None), limit: str | None = Form(None),
                        top_k: str | None = Form(None)) -> JSONResponse:
        esik_v, limit_v, topk_v = _sayi("eşik", esik, float), \
            _sayi("limit", limit, int), _sayi("top-k", top_k, int)
        if limit_v is not None and not 1 <= limit_v <= 5000:
            raise HTTPException(400, "limit 1–5000 arasında olmalı")
        if not (dosya.filename or "").lower().endswith(".csv"):
            raise HTTPException(400, "yalnız .csv kabul edilir")
        yuk_dir = app.state.runs_dir / "uploads"
        yuk_dir.mkdir(parents=True, exist_ok=True)
        hedef = yuk_dir / (f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}_"
                           f"{_uygulama_adi(dosya.filename)}")
        toplam = 0
        with hedef.open("wb") as fh:  # akışla yaz — bellekte biriktirme yok
            while True:
                parc = await dosya.read(1024 * 1024)
                if not parc:
                    break
                toplam += len(parc)
                if toplam > MAX_YUKLE_BAYT:
                    fh.close()
                    hedef.unlink(missing_ok=True)
                    raise HTTPException(413, f"dosya {MAX_YUKLE_BAYT // 2**20} MiB üzerinde")
                fh.write(parc)
        try:
            cikti = kosum(adapter, hedef, limit=limit_v, esik=esik_v, top_k=topk_v,
                          runs_dir=app.state.runs_dir,
                          komut=f"web arayüzü yükleme: {dosya.filename}")
        except KeyError as exc:
            raise HTTPException(400, f"bilinmeyen adapter: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return JSONResponse({"ozet": cikti["ozet"], "verdictler": _verdict_sayfasi(cikti["sonuc"])})

    @app.get("/api/kosumlar")
    def api_kosumlar(request: Request) -> list[dict]:
        kok = app.state.runs_dir
        kayitlar = []
        for d in sorted((p for p in kok.iterdir() if p.is_dir() and KOSUM_ID_RE.match(p.name)),
                        reverse=True)[:20]:
            kayitlar.append({"kosum": d.name,
                             "raporlar": sorted(f.name for f in d.iterdir()
                                                if f.name in INDIRILEBILIR)})
        return kayitlar

    @app.get("/kosumlar/{kid}/{dosya}")
    def api_indir(kid: str, dosya: str) -> FileResponse:
        if not KOSUM_ID_RE.match(kid) or dosya not in INDIRILEBILIR:
            raise HTTPException(404, "bilinmeyen koşum/dosya")
        yol = app.state.runs_dir / kid / dosya
        if not yol.is_file():
            raise HTTPException(404, "dosya yok")
        return FileResponse(yol, filename=f"{kid}_{dosya}")

    return app


def calistir(host: str = "127.0.0.1", port: int = 8734,
             runs_dir: Path = RUNS_DIR) -> int:
    import uvicorn

    uvicorn.run(create_app(runs_dir), host=host, port=port, log_level="info")
    return 0
