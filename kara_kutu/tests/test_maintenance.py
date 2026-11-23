"""Maintenance adaptörünün sözleşme + gerekçe + atıf testleri (AGENTS §13 kapısı).

Edge paketinin parçaları (`maintenance/results/edge/model.onnx` +
`maintenance/results/e0b_artifacts/e0b_preprocessing.json`) ve numpy/onnxruntime
varsa model kullanan testler koşar; yoksa o testler atlanır. Sözleşme/biçim/atıf
testleri modelsiz de koşar.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from kara_kutu.adapters.maintenance import (
    ADAPTER,
    ARTIFACT,
    EDGE_DIR,
    ESIK_VARSAYILAN,
    MaintenanceAdapter,
)
from kara_kutu.core.contract import Adapter, Baseline, Kayit, Verdict, VeriSeti

REPO = Path(__file__).resolve().parents[2]

KISA_PENCERE_SATIR = {
    "engine_id": "smoke-engine",
    "rul": 12,
    **{f"f{k}_{j}": 0.0 for k in range(10) for j in range(36)},
}


def _hazir_mi() -> bool:
    eksik = [m for m in ("numpy", "onnxruntime") if importlib.util.find_spec(m) is None]
    return not eksik and (EDGE_DIR / "model.onnx").is_file() and ARTIFACT.is_file()


model_gerekli = pytest.mark.skipif(not _hazir_mi(),
                                   reason="edge paketi (model.onnx / e0b_preprocessing.json) "
                                          "veya numpy/onnxruntime yok")


def _veri_seti(satirlar: list[dict]) -> VeriSeti:
    kayitlar = [Kayit(i, satir) for i, satir in enumerate(json.loads(json.dumps(satirlar)))]
    kolonlar = sorted({kolon for kayit in kayitlar for kolon in kayit.ham})
    return VeriSeti(ad="sahte-csv", kolonlar=kolonlar, kayitlar=kayitlar, kaynak="test")


def _pencere(satir_bir_benzeri: dict, rul: int | None = None) -> dict:
    satir = dict(KISA_PENCERE_SATIR)
    if rul is not None:
        satir["rul"] = rul
    return satir


# ----------------------------------------------------------------- sözleşme
def test_protokol_model_istemeden_karsilanir() -> None:
    """Sözleşme öğeleri model yüklemeden çağrılabilir (modelsiz kapı)."""
    assert ADAPTER.name == "maintenance"
    assert isinstance(ADAPTER, Adapter)
    spec = ADAPTER.input_spec()
    assert spec.adapter == "maintenance"
    assert "rul" in spec.aciklama
    assert "engine_id" in spec.aciklama
    assert (spec.kolonlar == ("engine_id", "rul"))
    assert ADAPTER.model_dosyasi().as_posix().endswith("maintenance/results/edge/model.onnx")
    assert ADAPTER.esik == ESIK_VARSAYILAN == pytest.approx(0.23788578033447244)
    assert "val.threshold" in ADAPTER.esik_kaynak


def test_ek_metrikler_etiket_kolonuyla_olculur_modelsiz() -> None:
    """Binary rul≤30 ölçümü verdict/liste eşleştirmesinden çıkar; model gerekmez."""
    adapter = MaintenanceAdapter()
    ds = _veri_seti([_pencere({}, 12), _pencere({}, 90)])
    verdictler = [
        Verdict(kimlik=0, skor=0.5, karar="critical", gerekceler=[]),
        Verdict(kimlik=1, skor=0.001, karar="ok", gerekceler=[]),
    ]
    metrik = adapter.ek_metrikler(ds, verdictler)
    assert metrik, "ölçülemedi: rul kolonu / verdict eşi yok"
    isabet = next(v for k, v in metrik.items() if "top-1 doğruluk" in k)
    assert isabet == "1.0000", metrik
    recall = next(v for k, v in metrik.items() if "recall" in k)
    assert recall == "1.0000", metrik
    assert any("AUC-ROC" in k for k in metrik), metrik


def test_ek_metrikler_rul_yoksa_cokluku_dogrular() -> None:
    """rul kolonu tamamen olmayan CSV'de ölçüm düşer — ham success yoktur."""
    adapter = MaintenanceAdapter()
    kayitlar = [Kayit(i, {"engine_id": "e", "f0_0": 0.0}) for i in range(2)]
    ds = VeriSeti(ad="sahte-csv", kolonlar=["engine_id", "f0_0"], kayitlar=kayitlar, kaynak="test")
    verdictler = [Verdict(kimlik=0, skor=0.5, karar="ok", gerekceler=[]),
                  Verdict(kimlik=1, skor=0.5, karar="ok", gerekceler=[])]
    assert adapter.ek_metrikler(ds, verdictler) == {}


def test_kolon_eksikse_anlasilir_hata() -> None:
    """Kısa pencere → ValueError (kolon adı ve küme yüzeyine çıkıyor)."""
    eksik = {"engine_id": "e", "rul": 12, "f0_0": 0.0}
    try:
        import numpy as np

        MaintenanceAdapter._pencere_matrisi(eksik, np, 10, 36)
    except ValueError as exc:
        assert "f0_1" in str(exc)
    else:
        raise AssertionError("kolon eksikliğinde ValueError fırlatılmadı")


def test_kolon_sayi_degilse_anlasilir_hata() -> None:
    """Sayısal olmayan kolon → ValueError (hangi kolon, hangi ham değer)."""
    ham = dict(_pencere({}, 12))
    ham["f3_7"] = "not-a-number"
    try:
        import numpy as np

        MaintenanceAdapter._pencere_matrisi(ham, np, 10, 36)
    except ValueError as exc:
        assert "f3_7" in str(exc) and "not-a-number" in str(exc)
    else:
        raise AssertionError("sayı-dışı kolonda ValueError fırlatılmadı")


# ---------------------------------------------------------------- baseline
def test_baseline_kaynak_atiflari_gercek_satirlarda() -> None:
    """Her baseline: `dosya:satır` biçimi + sayının atfedilen satırda gerçekten olması."""
    baselines = ADAPTER.baseline(_veri_seti([_pencere({}, 12)]))
    assert baselines and all(isinstance(b, Baseline) for b in baselines)
    for b in baselines:
        assert re.fullmatch(r"maintenance/results/e0b_cmapss_rerun_2026-11-23\.json:\d+", b.kaynak), b.kaynak
        yol, satir_no = b.kaynak.rsplit(":", 1)
        hedef = REPO / yol
        assert hedef.is_file(), f"ölü adres: {b.kaynak}"
        satirlar = hedef.read_text(encoding="utf-8").splitlines()
        satir = satirlar[int(satir_no) - 1]
        assert str(b.deger) in satir, f"sayı satırda yok: {b.deger} ≠ {b.kaynak} → {satir.strip()!r}"


# ------------------------------------------------------------------- model
@model_gerekli
def test_predict_ve_skor_araligi() -> None:
    """predict: 2 kayıt, skor [0,1], karar etiketli, gerekçe mevcut."""
    adapter = MaintenanceAdapter()
    ds = _veri_seti([_pencere({}, 12), _pencere({}, 90)])
    verdictler = adapter.predict(ds)
    assert len(verdictler) == 2
    for v in verdictler:
        assert isinstance(v, Verdict)
        assert 0.0 <= v.skor <= 1.0, f"skor aralık dışı: {v.skor}"
        assert v.karar in ("critical", "ok")
        assert len(v.gerekceler) == 5, "ilk 5 |Δ| pencere adımı gerekçe"


@model_gerekli
def test_meta_ve_ek_metrikler_model_ile_calisir() -> None:
    """model_meta + ek_metrikler: meta anahtarları + rul etiketli ölçüm dolu."""
    adapter = MaintenanceAdapter()
    meta = adapter.model_meta()
    assert meta.ad
    assert meta.ek_bilgi["paket_sha256_manifest"]
    import hashlib

    with ARTIFACT.open("rb") as f:
        assert meta.ek_bilgi["e0b_preprocessing sha256"] == hashlib.sha256(f.read()).hexdigest()
    ds = _veri_seti([_pencere({}, 12), _pencere({}, 90)])
    verdictler = adapter.predict(ds)
    metrik = adapter.ek_metrikler(ds, verdictler)
    assert any("top-1 doğruluk" in k for k in metrik), metrik
    assert any("AUC-ROC" in k for k in metrik), metrik


def test_registry_maintenance_bulur() -> None:
    """Registry keşfi: maintenance adapter'ı bulunur ve sözleşme karşılar."""
    from kara_kutu.core.registry import get_adapter

    adapter = get_adapter("maintenance")
    assert adapter.name == "maintenance"
    assert callable(adapter.predict)
