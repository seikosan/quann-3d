"""Kara Kutu CLI uçtan uca testi — plan §5.4 komutunun mekanik karşısı.

Küçük alt-küme CSV'si üzerinde `run` koşar; dört çıktı dosyasını ve kanıt
alanlarını doğrular. Ağ/eğitim yok: ONNX CPU + quantizer (~saniyeler).
"""
from __future__ import annotations

import json
from pathlib import Path

from kara_kutu.__main__ import main

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "fraud" / "data" / "User0_credit_card_transactions.csv"


def test_cli_run_uretimi(tmp_path):
    import pytest

    pytest.importorskip("onnxruntime")
    cikis = tmp_path / "runs"
    rc = main(["run", "--adapter", "fraud", "--veri", str(CSV),
               "--rapor", "html,md", "--limit", "40",
               "--runs-dir", str(cikis)])
    assert rc == 0
    (kosum,) = list(cikis.iterdir())
    adlar = {p.name for p in kosum.iterdir()}
    assert {"sonuc.json", "kanit.json", "ic_rapor.md", "dis_rapor.html"} <= adlar

    kanit = json.loads((kosum / "kanit.json").read_text())
    assert len(kanit["model_sha256"]) == 64
    assert len(kanit["dataset_sha256"]) == 64
    assert kanit["git_head"]
    sonuc = json.loads((kosum / "sonuc.json").read_text())
    assert sonuc["adapter"] == "fraud"
    assert sonuc["satir_sayisi"] == 40
    assert len(sonuc["verdictler"]) == 40
    # AUC ancak iki sınıf varsa tanımlıdır (ilk 40 satırda fraud yok → beklenmez)
    import pandas as pd

    iki_sinif = (pd.read_csv(CSV, nrows=40)["Is Fraud?"].str.strip().str.lower()
                 .nunique() > 1)
    auc_anahtar = "AUC-ROC (girdi CSV etiketleri, bu koşunun ölçümü)"
    assert (auc_anahtar in sonuc["metrikler"]) == iki_sinif
    assert sonuc["metrikler"]["model.onnx sha256 ↔ manifest"] == "EVET ✓"
    assert (kosum / "ic_rapor.md").read_text().startswith("# Kara Kutu İç Rapor")
    assert "data-kaynak" in (kosum / "dis_rapor.html").read_text()


def test_cli_bilinmeyen_adapter(tmp_path, capsys):
    rc = main(["run", "--adapter", "yok-boyle", "--veri", str(CSV),
               "--runs-dir", str(tmp_path)])
    assert rc == 2
    assert "adapter bulunamadı" in capsys.readouterr().err
