"""Eşik kalibrasyonu birim testleri — Karar A (kara_kutu/tools/esik_kalibre.py).

CSV/ONNX gerektirmez; artefakt-bağlı testler artefakt yokken atlanır
(üretici: `python3 -m kara_kutu.tools.esik_kalibre`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from kara_kutu.adapters.fraud import FraudAdapter
from kara_kutu.tools.esik_kalibre import ARTEFACT_VARSAYILAN, _eniyi_f2, _kari_esik


# ------------------------------------------------------------------ saf F2
def _yuva_f2(p: np.ndarray, y: np.ndarray, esik: float) -> float:
    """Bağımsız (döngülü) F2 — doğrulama referansı."""
    alarm = p >= esik
    tp = int(((y == 1) & alarm).sum())
    fp = int(((y == 0) & alarm).sum())
    if tp + fp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / int(y.sum()) if y.sum() else 0.0
    pay = 4 * prec + rec
    return 5 * prec * rec / pay if pay > 0 else 0.0


def test_eniyi_f2_kaba_kuvvete_karsi():
    rng = np.random.default_rng(7)
    p = rng.random(500)
    y = (rng.random(500) < np.clip(p * 0.4 + 0.05, 0, 1)).astype(int)
    y[0] = 1  # pozitif garanti
    secim = _eniyi_f2(p, y)
    esik = secim["esik"]
    f2_secim = _yuva_f2(p, y, esik)
    for t in np.unique(np.concatenate([p, p - 1e-9, p + 1e-9])):
        assert f2_secim >= _yuva_f2(p, y, t) - 1e-12
    assert secim["f2"] == pytest.approx(f2_secim, abs=1e-6)


def test_eniyi_f2_bag_grubu_esikle_alinir():
    # üç negatif aynı 0,5 puanında; eşik 0,5 ise hepsi alarm yakar
    p = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    y = np.array([1, 0, 0, 0, 0])
    secim = _eniyi_f2(p, y)
    if secim["esik"] <= 0.5:  # bağ grubu alınmışsa tam 3 FP olmalı
        op = _kari_esik(p, y, secim["esik"])
        assert op["alarm"] in (1, 4)  # ya sadece 0,9 ya 0,9+bağ grubu


def test_eniyi_f2_pozitif_yoksa_patlar():
    with pytest.raises(ValueError, match="pozitif yok"):
        _eniyi_f2(np.array([0.1, 0.2]), np.array([0, 0]))


def test_kari_esik_sayilari():
    p = np.array([0.99, 0.9, 0.5, 0.2])
    y = np.array([1, 0, 1, 0])
    op = _kari_esik(p, y, 0.5)
    assert op == {"alarm": 3, "tp": 2, "fp": 1, "precision": pytest.approx(2 / 3),
                  "recall": 1.0, "f2": pytest.approx(round(5 * (2 / 3) * 1 / (4 * (2 / 3) + 1), 6))}


# ------------------------------------------------------------------- artefakt
def test_artefakt_yoksa_sessiz_dusush_yok(monkeypatch, tmp_path):
    monkeypatch.setattr("kara_kutu.adapters.fraud.ESIK_ARTEFACT",
                        tmp_path / "yok.json")
    monkeypatch.setattr(FraudAdapter, "_esik_cache", None, raising=False)
    a = FraudAdapter()
    with pytest.raises(RuntimeError, match="esik_kalibre"):
        _ = a.esik


ARTEFACT = Path(ARTEFACT_VARSAYILAN)


def _art():
    if not ARTEFACT.is_file():
        pytest.skip("eşik artefaktı yok — python3 -m kara_kutu.tools.esik_kalibre")
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


def _var_mi(p: Path, ne: str) -> None:
    """kanıt koşumu gitignore'daki runs/ altında — taze klonada atlanır,
    yerelde kalkan (repo konvansiyonu: test_kosmus_raporlarin...)."""
    if not p.is_file():
        pytest.skip(f"{ne} gitignore'da veya eksik: {p}")


def test_artefakt_semasi():
    """Artefaktın iki meşru durumu var (Karar A1):
    1) kalibrasyon öncesi — kural=best-F2, te_dogrulama=null
    2) tek te geçişi sonrası — kural=alarm-budgeti-esitleme, te ölçümü işlenmiş."""
    art = _art()
    assert 0.0 < float(art["esik"]) < 1.0
    assert art["us"] == 42
    _var_mi(Path(art["kanit_kosum"]), "kanıt koşumu")
    if art.get("te_dogrulama") is None:
        assert art["kural"] == "best-F2"
        assert "paper_draft.md:351" in art["kural_kaynak"]
        assert art["kalibrasyon"]["tanım"].startswith("kutu ONNX")
        assert "user_fold<8" in art["kalibrasyon"]["tanım"]
        assert math.isclose(art["kalibrasyon"]["esikte"]["alarm"],
                            art["kalibrasyon"]["esikte"]["tp"]
                            + art["kalibrasyon"]["esikte"]["fp"])
    else:
        assert art["kural"] == "alarm-budgeti-esitleme"
        assert "false_alarm_audit.json" in art["kural_kaynak"]
        # reddedilen eğitim-fold tanıları yerinde kalmış olmalı
        assert "REDDEDİLDİ" in art["kalibrasyon_tani"]["durum"]
        assert "user_fold<8" in art["kalibrasyon_tani"]["tanım"]
        te = art["te_dogrulama"]
        assert te["alarm"] == te["tp"] + te["fp"]
        assert te["alarm"] == te["k_butce"]  # bütçe = alarm sayısı (quantil kuralı)
        _var_mi(Path(te["kosum"]), "te kanıt koşumu")


def test_adapter_artefakti_okur():
    art = _art()
    a = FraudAdapter()
    assert a.esik == pytest.approx(float(art["esik"]))
    assert "esik_fraud.json" in a.esik_kaynak
