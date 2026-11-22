"""Complaints adaptörünün sözleşme + gerekçe + atıf testleri (AGENTS §13 kapısı).

Edge paketinin parçaları (`complaints/results/edge/reader.onnx` + `reader.pt`) ve
numpy/onnxruntime/torch/tokenizers varsa model kullanan testler koşar; yoksa o
testler atlanır. Sözleşme/biçim/atıf testleri modelsiz de koşar.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from kara_kutu.adapters.complaints import ADAPTER, CHECKPOINT, EDGE_DIR, ETIKET_KOLON, METIN_KOLON
from kara_kutu.core.contract import Adapter, Baseline, Kayit, VeriSeti

REPO = Path(__file__).resolve().parents[2]

KISA_METINLER = [
    "The bank charged me an overdraft fee twice and I want a refund.",
    "My credit report shows a late payment that does not belong to me.",
    "The collection agency keeps calling about a debt I already paid off.",
]


def _hazir_mi() -> bool:
    """Model dosyaları ve bağımlılıklar tam mı (eval-only kapı: eksikte atla)."""
    eksik = [m for m in ("numpy", "onnxruntime", "torch", "tokenizers")
             if importlib.util.find_spec(m) is None]
    return not eksik and (EDGE_DIR / "reader.onnx").is_file() and CHECKPOINT.is_file()


model_gerekli = pytest.mark.skipif(not _hazir_mi(),
                                   reason="edge paketi (reader.onnx / reader.pt) veya "
                                          "numpy/onnxruntime/torch/tokenizers yok")


def _veri_seti(metinler: list[str], etiketler: list[str] | None = None) -> VeriSeti:
    """JSON satırlarından VeriSeti — runner'ın ham satır şeklini taklit eder."""
    satirlar = [{METIN_KOLON: metin} for metin in metinler]
    if etiketler:
        for satir, etiket in zip(satirlar, etiketler):
            satir[ETIKET_KOLON] = etiket
    kayitlar = [Kayit(i, satir) for i, satir in enumerate(json.loads(json.dumps(satirlar)))]
    kolonlar = sorted({kolon for kayit in kayitlar for kolon in kayit.ham})
    return VeriSeti(ad="sahte-json", kolonlar=kolonlar, kayitlar=kayitlar, kaynak="test")


def _referans():
    """Bağımsız referans paketi: (numpy, tokenizer, id2token, session, W, b).

    Adaptörün yükleme/koşum kodunu KULLANMAZ — sayıların doğrulanması ayrı bir
    yoldan gelmeli (test_parity.py'nin fraud'taki yaklaşımı).
    """
    import numpy as np
    import onnxruntime as ort
    import torch

    from cloudlm.tokenizer import BPETokenizer

    tokenizer = BPETokenizer.load(str(EDGE_DIR / "tokenizer.json"))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    w = checkpoint["model_state_dict"]["issue_head.weight"].numpy().astype("float32")
    b = checkpoint["model_state_dict"]["issue_head.bias"].numpy().astype("float32")
    session = ort.InferenceSession(str(EDGE_DIR / "reader.onnx"),
                                   providers=["CPUExecutionProvider"])
    id2token = {int(kimlik): str(parca) for parca, kimlik in tokenizer.vocab.items()}
    return np, tokenizer, id2token, session, w, b


def _olasilik(np, session, w, b, tokenizer, metin, maskelenen: int | None = None):
    """batch=1 sade koşu → (135 sınıflı softmax, token id listesi)."""
    ids = tokenizer.encode(metin)[:64] or [2]
    tk = np.zeros((1, 64), dtype="int64")
    tk[0, :len(ids)] = ids
    mask = tk != 0
    if maskelenen is not None:
        mask[0, maskelenen] = False
    z = session.run(["z"], {"tokens": tk, "mask": mask})[0]
    logits = (z @ w.T + b).astype("float64")
    p = np.exp(logits - logits.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True), ids


def _temiz(id2token: dict[int, str], parca_id: int) -> str:
    """token id → Ġ/Ċ işaretleri düşülmüş etiket (adaptörün temizleme kuralı)."""
    return id2token.get(int(parca_id), "").replace("Ġ", " ").replace("Ċ", " ").strip()


# ----------------------------------------------------------------- sözleşme
def test_protokol_model_istemeden_karsilanir() -> None:
    """Adapter Protocol + input_spec model yüklemeden çalışır (registry böyle keşfeder)."""
    assert isinstance(ADAPTER, Adapter)
    assert ADAPTER.name == "complaints"
    spec = ADAPTER.input_spec()
    assert spec.adapter == ADAPTER.name
    assert spec.kolonlar == (METIN_KOLON,)
    assert "complaints/src/datasets_cfpb.py:11" in spec.aciklama


@model_gerekli
def test_sozlesme_cagrilir_ve_skor_araligi() -> None:
    """predict/model_meta/baseline dönebilir; skor olasılık, eşik uydurma değil."""
    verdictler = ADAPTER.predict(_veri_seti(KISA_METINLER))
    assert len(verdictler) == len(KISA_METINLER)
    for verdict in verdictler:
        assert 0.0 < verdict.skor <= 1.0
        assert verdict.karar
    meta = ADAPTER.model_meta()
    assert meta.esik == 0.0 and "argmax" in meta.esik_kaynak
    assert meta.ek_bilgi["girdi_imzasi"].startswith("tokens int64 (B,64)")
    assert len(meta.ek_bilgi["paket_sha256_manifest"]) == 64


@model_gerekli
def test_kolon_eksikse_anlasilir_hata() -> None:
    """Zorunlu kolon yoksa sessiz boş skor yerine adını söyleyen ValueError."""
    ds = VeriSeti(ad="bozuk", kolonlar=["metin"], kayitlar=[Kayit(0, {"metin": "x"})])
    with pytest.raises(ValueError, match=METIN_KOLON):
        ADAPTER.predict(ds)


# --------------------------------------------------------------- skor doğruluğu
@model_gerekli
def test_skor_bagimsiz_referansla_ayni() -> None:
    """P(issue): adaptörün sabit-batch + head + softmax yolu == sade batch=1 referansı.

    Tolerans 1e-6: float32 matmul biriktirme farkı payı. Skor = en yüksek sınıfın
    softmax olasılığı (max-over-classes), argmax kararıyla aynı sınıf.
    """
    np, tokenizer, _, session, w, b = _referans()
    for metin in KISA_METINLER:
        referans = float(_olasilik(np, session, w, b, tokenizer, metin)[0].max())
        bulunan = ADAPTER.predict(_veri_seti([metin]))[0].skor
        assert abs(referans - bulunan) < 1e-6, f"{metin!r}: {referans} ≠ {bulunan}"


@model_gerekli
def test_batch_sayisi_skoru_degistirmez() -> None:
    """ONNX batch ekseni değişince çıktı sessiz bozuluyor — sabit-batch korumasının testi.

    Aynı metin tek başına ve 3'lü grubun içinde birebir aynı skoru ve gerekçe
    sırasını vermeli (grup koşumu == tek satır koşumu).
    """
    tek = ADAPTER.predict(_veri_seti([KISA_METINLER[2]]))[0]
    grup = ADAPTER.predict(_veri_seti(KISA_METINLER))[2]
    assert abs(grup.skor - tek.skor) < 1e-9
    assert [g.etiket for g in grup.gerekceler] == [g.etiket for g in tek.gerekceler]
    assert all(abs(a.delta_logit - b.delta_logit) < 1e-9
               for a, b in zip(grup.gerekceler, tek.gerekceler))


# -------------------------------------------------------------------- gerekçe
@model_gerekli
def test_gerekceler_girdi_metninin_gercek_parcasi() -> None:
    """Gerekçe etiketi uydurulamaz: girdi metninde (harf-kayıtsız) geçmeli + |delta| sıralı."""
    for metin in KISA_METINLER:
        (verdict,) = ADAPTER.predict(_veri_seti([metin]))
        assert verdict.gerekceler, f"hiç gerekçe çıkamadı: {metin!r}"
        for gerekce in verdict.gerekceler:
            assert gerekce.etiket.casefold() in metin.casefold()
            assert abs(gerekce.delta_logit) > 1e-6
        ilk = abs(verdict.gerekceler[0].delta_logit)
        assert all(abs(g.delta_logit) <= ilk + 1e-9 for g in verdict.gerekceler[1:])


@model_gerekli
def test_gerekce_yonu_maskelemeyle_dogrulanir() -> None:
    """'+logit' = destekliyor iddiası: token'ın pozisyonu maskelenince aynı fark çıkar.

    Bir etiket metinde birden çok token'da geçebilir; adaptör en büyük |delta|'yı
    bildirir → test de aynı aday kümesinin en büyüğüyle karşılaştırır.
    """
    metin = KISA_METINLER[0]
    (verdict,) = ADAPTER.predict(_veri_seti([metin]))
    np, tokenizer, id2token, session, w, b = _referans()
    tam, ids = _olasilik(np, session, w, b, tokenizer, metin)
    sinif = int(tam.argmax())
    assert verdict.gerekceler
    for gerekce in verdict.gerekceler:
        pozisyonlar = [i for i, id_ in enumerate(ids) if _temiz(id2token, id_) == gerekce.etiket]
        assert pozisyonlar, f"etiket token'a bağlı değil: {gerekce.etiket!r}"
        farklar = []
        for pozisyon in pozisyonlar:
            maskeli, _ = _olasilik(np, session, w, b, tokenizer, metin, maskelenen=pozisyon)
            assert maskeli[0, sinif] != tam[0, sinif], "maskeleme çıktıyı hiç değiştirmedi"
            farklar.append(float(np.log(tam[0, sinif]) - np.log(maskeli[0, sinif])))
        beklenen = max(farklar, key=abs)
        assert abs(beklenen - gerekce.delta_logit) < 1e-4, (gerekce, farklar)
        isaret = 1 if gerekce.delta_logit > 0 else -1
        assert isaret == (1 if tam[0, sinif] > maskeli[0, sinif] else -1), gerekce


# ------------------------------------------------------- edge paketinin bütünlüğü
@pytest.mark.skipif(importlib.util.find_spec("onnx") is None or not _hazir_mi(),
                    reason="onnx veya edge paketi yok")
def test_edge_grafigi_head_icermez_agirliklar_checkpointle_birebir() -> None:
    """Adaptörün head'i neden numpy'de uyguluyor: ONNX grafiğinde `issue_head` YOK.

    Kontrol: reader.pt'nin 29 tensöründen 27'si grafikte bit düzeyinde aynı
    (19'u `model.<anahtar>`, 8'i `onnx::MatMul_*` transpoze); karşılığı olmayanlar
    yalnız `issue_head.weight` + `issue_head.bias`. Bu, `complaints/src/export_edge.py:24` ZOnly
    sözleşmesinin mekanik kanıtı — head uygulanmazsa 135 sınıflı skor üretilemez.
    """
    import onnx
    import torch

    graph = onnx.load(str(EDGE_DIR / "reader.onnx"), load_external_data=True)
    initializer = {i.name: onnx.numpy_helper.to_array(i) for i in graph.graph.initializer}
    tensörler = torch.load(CHECKPOINT, map_location="cpu",
                           weights_only=True)["model_state_dict"]
    anonim = {ad: deger for ad, deger in initializer.items() if ad.startswith("onnx::")}
    eksik = []
    for anahtar, deger in tensörler.items():
        ham = deger.numpy().astype("float32")
        adaylar = [initializer[f"model.{anahtar}"]] if f"model.{anahtar}" in initializer else []
        adaylar += list(anonim.values())
        if any(float(bit_fark(a, ham)) == 0.0 for a in adaylar):
            continue
        eksik.append(anahtar)
    assert eksik == ["issue_head.weight", "issue_head.bias"], eksik


def bit_fark(ilg_degeri, hedef) -> float:
    """min(|a-hedef|, |aᵀ-hedef|) — ONNX Linear ağırlıkları transpoze saklanabiliyor.

    Şekli uyuşmayan yönelim yok sayılır (broadcast'ın sessiz yanlış sonucu yerine ∞).
    """
    import numpy as np

    adaylar = [ilg_degeri] + ([ilg_degeri.T] if ilg_degeri.ndim == 2 else [])
    degerler = [float(np.abs(a - hedef).max()) for a in adaylar if a.shape == hedef.shape]
    return min(degerler) if degerler else float("inf")


# ------------------------------------------------------------------- baseline
def test_baseline_kaynak_atiflari_gercek_satirlarda() -> None:
    """Her baseline: `dosya:satır` biçimi + sayının atfedilen satırda gerçekten olması.

    AGENTS §13.1'in mekanik hâli: uydurma sayı ya da ölü adres burada kırılır.
    Rakam karşılaştırması nokta/ayraç duyarsızıdır (dijital dizi kümesi).
    """
    baselines = ADAPTER.baseline(_veri_seti([KISA_METINLER[0]]))
    assert baselines and all(isinstance(b, Baseline) for b in baselines)
    for b in baselines:
        assert b.ad and b.metrik
        assert re.fullmatch(r"[\w./-]+\.json:\d+", b.kaynak), b.kaynak
        yol, satir_no = b.kaynak.rsplit(":", 1)
        hedef = REPO / yol
        assert hedef.is_file(), f"ölü adres: {b.kaynak}"
        satirlar = hedef.read_text(encoding="utf-8").splitlines()
        assert int(satir_no) <= len(satirlar), f"satır numarası taşkın: {b.kaynak}"
        satir = satirlar[int(satir_no) - 1]
        rakam = re.sub(r"\D", "", f"{b.deger}")
        assert rakam and rakam in re.sub(r"\D", "", satir), \
            f"sayı satırda yok: {b.deger} ≠ {b.kaynak} → {satir.strip()!r}"


# ------------------------------------------------------------------ metrikler
@model_gerekli
def test_ek_metrikler_etiket_kolonuyla_olculur() -> None:
    """`issue` kolonu varsa metrikler adlandırılır; kolon yoksa metrik de yok."""
    etiketler = [KISA_METINLER[0], "Olmayan bir etiket", "Incorrect information on your report"]
    ds = _veri_seti(KISA_METINLER, etiketler)
    verdictler = ADAPTER.predict(ds)
    metrikler = ADAPTER.ek_metrikler(ds, verdictler)
    assert any(ad.startswith("issue top-1 doğruluk") for ad in metrikler), metrikler
    assert all("ölçümü)" in ad for ad in metrikler), metrikler
    assert all(deger != "nan" for deger in metrikler.values()), metrikler
    assert ADAPTER.ek_metrikler(_veri_seti(KISA_METINLER), verdictler) == {}


# ------------------------------------------------------------------- registry
@model_gerekli
def test_registry_complaints_bulur() -> None:
    """`adapters/*.py` keşfi: `complaints` adı aynı sözleşmeyle bulunmalı.

    Registry her adaptörü izole bir modül adı altında yükler
    (`kara_kutu._adapter_<ad>` — kara_kutu/core/registry.py:33), bu yüzden
    singleton birebir aynı NESNE/sınıf değil; aynı ad ve aynı sözleşme doğrulanır.
    """
    from kara_kutu.core.registry import get_adapter, list_adapters

    assert "complaints" in list_adapters()
    bulunan = get_adapter("complaints")
    assert isinstance(bulunan, Adapter)
    assert type(bulunan).__name__ == type(ADAPTER).__name__ == "ComplaintsAdapter"
    assert bulunan.name == ADAPTER.name == "complaints"
    assert bulunan.input_spec() == ADAPTER.input_spec()
    assert bulunan.model_dosyasi() == ADAPTER.model_dosyasi()
    assert [b.kaynak for b in bulunan.baseline(_veri_seti([KISA_METINLER[0]]))] == \
        [b.kaynak for b in ADAPTER.baseline(_veri_seti([KISA_METINLER[0]]))]
