"""Sözleşme + dış-rapor sayı-kanıtı testleri (AGENTS §13'ün kod kapısı)."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from kara_kutu.core.contract import (
    Baseline,
    KanitBlogu,
    KosumSonucu,
    ModelMeta,
    Verdict,
    kaynakli,
)
from kara_kutu.report.external import uret as uret_dis

REPO = Path(__file__).resolve().parents[2]


class _SayiDenetci(HTMLParser):
    """data-kaynak span'i ve SVG DIŞINDA kalan metin düğümünde rakam arar.

    <style>/<title>/<script> içerikleri sunum kodudur, görünür sayı değildir.
    """

    def __init__(self) -> None:
        super().__init__()
        self.derinlik = 0  # data-kaynak span'i veya svg içinde miyiz
        self.gizli = 0     # style/title/script içinde miyiz
        self.ihlaller: list[str] = []

    def handle_starttag(self, tag, attrs):
        if (tag == "span" and dict(attrs).get("data-kaynak") is not None) or tag == "svg":
            self.derinlik += 1
        elif tag in ("style", "title", "script"):
            self.gizli += 1

    def handle_endtag(self, tag):
        if tag in ("span", "svg") and self.derinlik:
            self.derinlik -= 1
        elif tag in ("style", "title", "script") and self.gizli:
            self.gizli -= 1

    def handle_data(self, data):
        if self.derinlik or self.gizli:
            return
        if re.search(r"\d", data):
            self.ihlaller.append(data.strip()[:80])


def _ornek_sonuc() -> KosumSonucu:
    from kara_kutu.report.provenance import sha256_dosya

    meta = ModelMeta(ad="test", esik=0.9969, esik_kaynak="fraud/pilot/infer.py:40",
                     olusturma="test")
    kanit = KanitBlogu(komut="test", git_head="x", tarih="2026-11-21T00:00:00Z",
                       model_sha256=sha256_dosya(REPO / "fraud/results/deploy/model.onnx"),
                       dataset_sha256="d", ek_atiflar={"python": "3", "platform": "t"})
    verdictler = [
        Verdict(kimlik=i, skor=0.5 + 0.4 * (i % 2),
                karar="FRAUD" if i % 2 else "NORMAL")
        for i in range(4)
    ]
    baseline = (Baseline("Random Forest", "AUC-ROC", 0.973, "docs/market_priorities.md:8"),)
    return KosumSonucu(adapter="fraud", esik=0.9969, satir_sayisi=4, verdictler=verdictler,
                       baseline=baseline, meta=meta, kanit=kanit,
                       metrikler={"AUC-ROC (girdi CSV etiketleri, bu koşunun ölçümü)": "0.9451"},
                       pozitif_etiket="FRAUD", negatif_etiket="NORMAL",
                       skor_etiket="P(fraud)")


def test_kaynakli_dogrulanmadi_basari():
    assert "DOĞRULANMADI" in kaynakli(0.5, None)
    assert "a.py:1" in kaynakli(0.5, "a.py:1")


ATIF_SATIR = re.compile(r"^`?([\w./@~-]+\.(?:py|md|json|log|csv|txt|sh|yaml|yml|db)):(\d+)`?$")
ONDALIK = re.compile(r"^\d+[.,]\d+$")


def _floatlar(metin: str) -> set[float]:
    # ".973" (başında sıfır olmayan ondalık) ve "0.973" biçimlerini birlikte toplar
    degerler: set[float] = set()
    for tok in re.findall(r"\.\d+|\d+(?:[.,]\d+)*", metin):
        tok = tok.replace(",", ".")
        try:
            degerler.add(float(tok))
        except ValueError:  # "1.2.3" gibi sürüm dizileri sayı değildir
            continue
    return degerler


def _tablo_atif_denetimi(md: str) -> int:
    """`|` satırlarında son hücresi `dosya:satır` olan her tablo satırını
    denetler: ondalık hücreler atfedilen kaynak satırında geçmeli.
    rapor_denetle'nin `|` muafiyeti (rapor_denetle.py:29) yüzünden iç rapor
    sayıları dış kapıdan kaçıyor — bu test depo tarafı ikinci kapıdır."""
    denetlenen = 0
    for satir in md.splitlines():
        if not satir.startswith("|"):
            continue
        hucreler = [h.strip() for h in satir.strip("|").split("|")]
        if len(hucreler) < 3:
            continue
        m = ATIF_SATIR.match(hucreler[-1])
        if not m:
            continue
        hedef_yol, hedef_satir = REPO / m.group(1), int(m.group(2))
        assert hedef_yol.is_file(), f"ölü adres: {m.group(1)}"
        satirlar = hedef_yol.read_text(errors="replace").splitlines()
        assert hedef_satir <= len(satirlar), f"satır taşması: {m.group(1)}:{hedef_satir}"
        kaynak_float = _floatlar(satirlar[hedef_satir - 1])
        for hucre in hucreler[:-1]:
            if not ONDALIK.match(hucre):
                continue
            assert float(hucre.replace(",", ".")) in kaynak_float or (
                float(hucre.replace(",", ".")) == 0.0 and not kaynak_float
            ), f"{hucre} değeri {m.group(1)}:{hedef_satir} satırında yok: {satirlar[hedef_satir - 1][:80]!r}"
            denetlenen += 1
    return denetlenen


def test_ic_rapor_tablo_atiflari_kaynagiyla_usuruluyor():
    from kara_kutu.report.internal import uret as uret_ic

    denetlenen = _tablo_atif_denetimi(uret_ic(_ornek_sonuc()))
    assert denetlenen >= 3, f"denetlenen atıflı sayı beklenenden az: {denetlenen}"


def test_ic_rapor_fraud_izleri_diger_dikeye_sizmiyor():
    """Geçici önlem (2026-11-22, KARA_KUTU_PLAN §9): P(fraud) başlığı ve Ekim
    fraud AUC referansları fraud dışı adaptör raporunda basılmaz."""
    from kara_kutu.core.contract import Gerekce, Verdict as V
    from kara_kutu.report.internal import uret as uret_ic

    ham = _ornek_sonuc()
    kopya = KosumSonucu(
        adapter="complaints", esik=0.0, satir_sayisi=ham.satir_sayisi,
        verdictler=[V(kimlik=v.kimlik, skor=v.skor, karar="Managing an account",
                      gerekceler=list(v.gerekceler) + [Gerekce("mcc", 0.4 if i % 2 else -0.4)])
                    for i, v in enumerate(ham.verdictler)],
        baseline=ham.baseline, meta=ham.meta, kanit=ham.kanit, metrikler={},
        pozitif_etiket=None, negatif_etiket=None, skor_etiket="P(issue)")
    md = uret_ic(kopya)
    assert "P(fraud)" not in md
    assert "0.9451" not in md and "0.9367" not in md  # Ekim fraud AUC'leri sızmez
    assert "Karar eşiği" in md and "uygulanmıyor" in md
    assert "fraud-destekliyor" not in md and "fraud-karşıtı" not in md
    # fraud koşumu aynı çıktıyı vermeye devam eder (gerileme koruması)
    fraud_md = uret_ic(ham)
    assert "P(fraud)" in fraud_md and "0.9451" in fraud_md


def test_kosmus_raporlarin_tablo_atiflari_dogru():
    """kara_kutu/runs/ altındaki gerçek koşum raporları da aynı denetimden
    geçer (klasör gitignore'dadır; taze klonada atlanır, yerelde kalkan)."""
    import pytest

    klasorler = sorted((REPO / "kara_kutu" / "runs").glob("*_fraud"))
    if not klasorler:
        pytest.skip("koşum dizini yok (gitignore'da)")
    for k in klasorler:
        md = (k / "ic_rapor.md").read_text()
        assert _tablo_atif_denetimi(md) >= 3, f"zayıf denetim: {k}"


def test_dis_rapor_span_siz_sayi_icermemez():
    denetci = _SayiDenetci()
    denetci.feed(uret_dis(_ornek_sonuc()))
    assert not denetci.ihlaller, f"span'siz rakam: {denetci.ihlaller[:5]}"
