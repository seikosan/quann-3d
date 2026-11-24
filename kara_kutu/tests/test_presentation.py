"""Dikey-geneli presentasyon sözleşmesi (KARA_KUTU_PLAN §10kapanış kartı).

runner, adapter'dan `pozitif_etiket`/`negatif_etiket`/`skor_etiket` stampeder;
rapor üreticileri (iç md + dış html) bunları kullanır — hiçbir pozitif sınıf
kelimesi şablona hardcode edilmez. Bu testler protokolü kilitler:
- maintenance (critical/ok): FRAUD/NORMAL sızmadı, P(critical) basıldı
- complaints (multi-class, dichotomy None): çift kart yok, P(issue) basıldı
- fraud (FRAUD/NORMAL): gerileme koruması
"""
from __future__ import annotations

from kara_kutu.core.contract import (
    Baseline, KanitBlogu, KosumSonucu, ModelMeta, Verdict,
)
from kara_kutu.report.internal import uret as uret_ic
from kara_kutu.report.external import uret as uret_dis


def _temel() -> KosumSonucu:
    meta = ModelMeta(ad="test", esik=0.23, esik_kaynak="test.txt:1", olusturma="test")
    kanit = KanitBlogu(komut="test", git_head="x", tarih="2026-11-24T00:00:00Z",
                       model_sha256="m", dataset_sha256="d",
                       ek_atiflar={"python": "3", "platform": "t"})
    verdictler = [
        Verdict(kimlik=0, skor=0.9, karar="critical", gerekceler=[]),
        Verdict(kimlik=1, skor=0.1, karar="ok", gerekceler=[]),
    ]
    baseline = (Baseline("test referansı", "AUC-ROC", 0.99, "test.log:1"),)
    return KosumSonucu(adapter="maintenance", esik=0.23, satir_sayisi=2,
                       verdictler=verdictler, baseline=baseline, meta=meta,
                       kanit=kanit, metrikler={}, pozitif_etiket="critical",
                       negatif_etiket="ok", skor_etiket="P(critical)")


def test_maintenance_dikey_icin_fraud_sizmiyor():
    sonuc = _temel()
    md = uret_ic(sonuc)
    html = uret_dis(sonuc)
    for metin in (md, html):
        assert "critical" in metin
        assert "P(critical)" in metin
        assert "FRAUD" not in metin and "NORMAL" not in metin
        assert "fraud-destekliyor" not in metin and "fraud-karşıtı" not in metin


def test_maintenance_dikey_kararlari_negatif_etiketi_disari_vermiyor():
    """binary çift kartını dichotomy'nin gerçekten var olduğuna bakarak basar."""
    sonuc = _temel()
    html = uret_dis(sonuc)
    assert "critical kararı" in html and "ok kararı" in html
    assert "critical oranı" in html


def test_complaints_dikey_dichotomy_yok():
    sonuc = _temel()
    sonuc = KosumSonucu(
        adapter="complaints", esik=0.0, satir_sayisi=2,
        verdictler=[
            Verdict(kimlik=0, skor=0.9, karar="Managing an account", gerekceler=[]),
            Verdict(kimlik=1, skor=0.1, karar="Billing", gerekceler=[]),
        ],
        baseline=sonuc.baseline, meta=sonuc.meta, kanit=sonuc.kanit,
        metrikler={}, pozitif_etiket=None, negatif_etiket=None,
        skor_etiket="P(issue)")
    md = uret_ic(sonuc)
    html = uret_dis(sonuc)
    assert "P(issue)" in md and "P(issue)" in html
    assert "FRAUD" not in md and "FRAUD" not in html
    assert "NORMAL" not in md and "NORMAL" not in html
    assert "critical" not in md and "critical" not in html
    assert "seçileni destekliyor" not in md  # nötr yön kelimesi leak olmadığı için
    # çok sınıflı: dichotomie kartları basılmadı
    assert "oranı" not in html


def test_fraud_regression_presentasyon():
    """Fraud koşumu aynı output'a devam eder (geri dönüş koruması)."""
    sonuc = _temel()
    sonuc = KosumSonucu(
        adapter="fraud", esik=0.9969, satir_sayisi=2,
        verdictler=[
            Verdict(kimlik=0, skor=0.9, karar="FRAUD", gerekceler=[]),
            Verdict(kimlik=1, skor=0.1, karar="NORMAL", gerekceler=[]),
        ],
        baseline=sonuc.baseline, meta=sonuc.meta, kanit=sonuc.kanit,
        metrikler={}, pozitif_etiket="FRAUD", negatif_etiket="NORMAL",
        skor_etiket="P(fraud)")
    md = uret_ic(sonuc)
    html = uret_dis(sonuc)
    assert "P(fraud)" in md and "P(fraud)" in html
    assert "FRAUD kararı" in html  # cards düşer


def test_yon_wording_pozitif_etikete_gore():
    from kara_kutu.core.contract import yonu_metni

    assert yonu_metni(0.5, "FRAUD") == "fraud-destekliyor"
    assert yonu_metni(-0.5, "FRAUD") == "fraud-karşıtı"
    assert yonu_metni(0.5, "critical") == "critical-destekliyor"
    assert yonu_metni(0.5, None) == "seçileni destekliyor"
    assert yonu_metni(-0.5, None) == "seçilenin karşıtı"
