"""Dış rapor (tek dosya HTML) — müşteri/vekil/yatırımcı tonu (plan §4).

Sunum güveni kuralı (plan §7.4): sayfa içindeki HER görünür sayı bir
`data-kaynak` taşır ("bu koşu" veya `dosya:satır`). Test, span'siz rakamı
olan metin düğümünü bulursa P0 kapısı kırılır. Bağımlılık yok: inline CSS +
inline SVG histogram, çevrimdışı açılır.

Dikey-geneli etiket sözleşmesi (kapanış kartı 2026-11-24): pozitif sınıf adı
ve skor başlığı runner tarafından adapter'dan mühürlenir — bu şablon hiçbir
dikeyin pozitif sınıfının adını bilmez (fraud/maintenance/complaints).
"""
from __future__ import annotations

import html
from collections import Counter

from kara_kutu.core.contract import KosumSonucu

_K = "bu koşunun ölçümü"


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def _alan(deger: object, kaynak: str) -> str:
    """Görünür içerik — rakam taşıyorsa data-kaynak span'i olmadan asla basılmaz."""
    return f'<span class="v" data-kaynak="{_esc(kaynak)}">{_esc(deger)}</span>'


def _statik(etiket: str) -> str:
    """Şablondan gelen sabit etiket (rakam taşıyabilir: örn. "sha256").

    Kaynağı uydurulmuş bir ölçüm değil, tam olarak bu şablon satırıdır.
    """
    return _alan(etiket, "kara_kutu/report/external.py (şablon etiketi)")


def _histogram(skorlar: list[float], genislik: int = 640, yukseklik: int = 170) -> str:
    """10 kutulu skor dağılımı; metinler yalnız SVG içindedir (grafik etiketi)."""
    kutular = Counter(min(int(s * 10), 9) for s in skorlar)
    tavan = max(kutular.values(), default=1) or 1
    parca = genislik / 10
    cubuk_alani = yukseklik - 34
    diciller: list[str] = []
    for b in range(10):
        n = kutular.get(b, 0)
        y = yukseklik - 20 - round(cubuk_alani * n / tavan)
        x = round(b * parca + 6)
        yuk = max(yukseklik - 20 - y, 1)
        etiket_x = round(b * parca + parca / 2)
        diciller.append(
            f'<rect x="{x}" y="{y}" width="{round(parca - 12)}" height="{yuk}" '
            f'fill="{"#c62828" if b >= 9 else "#546e7a"}" rx="2"/>'
            f'<text x="{etiket_x}" y="{yukseklik - 6}" text-anchor="middle" '
            f'class="ek">{b / 10:.1f}</text>'
            + (f'<text x="{etiket_x}" y="{y - 4}" text-anchor="middle" '
               f'class="ek">{n}</text>' if n else "")
        )
    return (f'<svg viewBox="0 0 {genislik} {yukseklik}" role="img" '
            f'aria-label="Skor dagilimi">{"".join(diciller)}</svg>')


def _binary_kartlar(sonuc: KosumSonucu, pozitif_n: int, negatif_n: int, oran: str) -> list[str]:
    """Dichotomy olmayan dikeylerde (çok sınıflı) basılmaz — hiçbir pozitif
    sınıf kelimesi uydurulmaz."""
    if not sonuc.pozitif_etiket:
        return []
    kartlar = [f'<div class="kart"><div class="buyuk">{_alan(pozitif_n, _K)}</div>'
               f'{_esc(sonuc.pozitif_etiket)} kararı</div>']
    if sonuc.negatif_etiket:
        kartlar.append(f'<div class="kart"><div class="buyuk">{_alan(negatif_n, _K)}</div>'
                       f'{_esc(sonuc.negatif_etiket)} kararı</div>')
    kartlar.append(f'<div class="kart"><div class="buyuk">{_alan(oran, _K)}</div>'
                   f'{_esc(sonuc.pozitif_etiket)} oranı</div>')
    return kartlar


def uret(sonuc: KosumSonucu, *, max_kayit: int = 20) -> str:
    k, m = sonuc.kanit, sonuc.meta
    skorlar = [v.skor for v in sonuc.verdictler]
    pozitif = sonuc.pozitif_etiket
    negatif = sonuc.negatif_etiket
    pozitif_n = sum(1 for v in sonuc.verdictler
                    if pozitif and v.karar == pozitif)
    negatif_n = len(skorlar) - pozitif_n
    oran = f"{pozitif_n / len(skorlar):.1%}" if skorlar and pozitif else "—"
    baslik = f"Kara Kutu Değerlendirme Raporu — {sonuc.adapter}"
    fraud_mu = sonuc.adapter == "fraud"
    govde = [
        "<!DOCTYPE html>",
        '<html lang="tr"><head><meta charset="utf-8">',
        f"<title>{_esc(baslik)}</title>",
        "<style>",
        "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;",
        "max-width:56rem;color:#212121;line-height:1.45;padding:0 1rem}",
        "h1{font-size:1.45rem}h2{font-size:1.1rem;margin-top:1.6rem;",
        "border-bottom:1px solid #e0e0e0}",
        "table{border-collapse:collapse;width:100%;font-size:.86rem}",
        "th,td{border:1px solid #cfd8dc;padding:.35rem .5rem;text-align:left}",
        "th{background:#eceff4}.kartlar{display:flex;gap:1rem;flex-wrap:wrap}",
        ".kart{border:1px solid #cfd8dc;border-radius:8px;padding:.7rem 1rem;",
        "min-width:10rem}.buyuk{font-size:1.5rem;font-weight:600}",
        ".f{color:#c62828}.n{color:#2e7d32}.ek{font-size:10px;fill:#607d8b}",
        ".kuye{font-size:.75rem;color:#546e7a;word-break:break-all;margin-top:2rem;",
        "border-top:1px solid #e0e0e0;padding-top:.6rem}",
        "</style></head><body>",
        f"<h1>{_esc(baslik)}</h1>",
    ]
    intro = (f"{_alan(sonuc.satir_sayisi, _K)} kayıt değerlendirildi; karar eşiği "
             f"{_alan(f'{sonuc.esik:.4f}', m.esik_kaynak)} üzeri "
             f"{_alan(pozitif_n, _K)} kayıt {_esc(pozitif)} olarak işaretlendi. "
             f"Dağılım aşağıdaki grafiktedir ({_alan(len(skorlar), _K)} skor).") \
        if pozitif else (f"{_alan(sonuc.satir_sayisi, _K)} kayıt değerlendirildi. "
                         f"Dağılım aşağıdaki grafiktedir ({_alan(len(skorlar), _K)} skor).")
    govde += [f"<p>{intro}</p>", "<h2>Skor dağılımı</h2>", _histogram(skorlar),
              '<div class="kartlar">']
    govde += _binary_kartlar(sonuc, pozitif_n, negatif_n, oran)
    govde += ["</div>", "<h2>Referans kartı</h2>", '<div class="kartlar">']
    for b in sonuc.baseline:
        govde.append(
            f'<div class="kart">{_esc(b.ad)}<br><span class="buyuk">'
            f'{_alan(f"{b.deger:.3f}", b.kaynak)}</span><br>{_esc(b.metrik)}</div>'
        )
    govde += ["</div>"]
    # Fraud'ın dış referansı (edge_eval tam doğal test ölçümleri) yalnız fraud'da
    if fraud_mu:
        govde.append(
            "<p>Modelimizin tam doğal test ölçümü: AUC-ROC "
            f"{_alan('0.9451 — FP32', 'fraud/logs/run_full_scale_edge_eval.log:52')}, "
            f"{_alan('0.9367 — INT8-channel', 'fraud/logs/run_full_scale_edge_eval.log:54')}. "
            "Bu rapor eğitimsiz (eval-only) bir koşumdur; model ağırlığı değiştirilmemiştir.</p>"
        )
    else:
        govde.append(
            "<p>Bu rapor eğitimsiz (eval-only) bir koşumdur; model ağırlığı "
            "değiştirilmemiştir.</p>"
        )
    if sonuc.metrikler:
        govde.append("<h2>Ölçülen metrikler</h2><ul>")
        for ad, deger in sonuc.metrikler.items():
            govde.append(f"<li>{_statik(ad)}: {_alan(deger, _K)}</li>")
        govde.append("</ul>")
    govde += [
        f"<h2>Kayıt bazlı kararlar (ilk {_alan(min(max_kayit, len(sonuc.verdictler)), _K)})</h2>",
        f"<table><tr><th>#</th><th>Kimlik</th><th>{_esc(sonuc.skor_etiket)}</th><th>Karar</th>"
        "<th>Neden (en güçlü etkenler)</th></tr>",
    ]
    for i, v in enumerate(sonuc.verdictler[:max_kayit], 1):
        gerekce = "; ".join(
            f"{g.etiket} ({g.delta_logit:+.2f} logit)" for g in v.gerekceler[:3]
        ) or "tek-etken etkisi ölçülemedi"
        sinif = "f" if pozitif and v.karar == pozitif else ("n" if negatif and v.karar == negatif else "")
        govde.append(
            f"<tr><td>{_alan(i, _K)}</td><td>{_alan(v.kimlik, _K)}</td>"
            f"<td>{_alan(f'{v.skor:.6f}', _K)}</td>"
            f'<td class="{sinif}">{_esc(v.karar)}</td><td>{_alan(gerekce, _K)}</td></tr>'
        )
    govde += [
        "</table>",
        '<div class="kuye">Bu ölçüm aşağıdaki koşumla üretildi — dışarıdan elle sayı '
        "girilmemiştir.<br>"
        f"Model: {_esc(m.ad)} · {_statik('paket sha256')}: {_alan(k.model_sha256, _K)}<br>"
        f"Veri kümesi {_statik('sha256')}: {_alan(k.dataset_sha256, _K)} · "
        f"git: {_alan(k.git_head, _K)}<br>"
        f"Koşum tarihi (UTC): {_alan(k.tarih, _K)} · üretici: "
        f"{_alan('kara_kutu P0', 'kara_kutu/__init__.py')}</div>",
        "</body></html>",
    ]
    return "\n".join(govde)
