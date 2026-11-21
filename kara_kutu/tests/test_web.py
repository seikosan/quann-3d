"""Kara Kutu web duman testi — P1 kapısının (browser turu) depo içi karşısı:
CSV yükle → özet+tablo → rapor indir. Gerçek browser turu ayrıca koşılır;
bu test HTTP sözleşmesini ve eval-only yazma hedefini (`runs/` altı) kilitler.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kara_kutu.web.app import create_app

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "fraud" / "data" / "User0_credit_card_transactions.csv"


@pytest.fixture()
def istemci(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("onnxruntime")
    return TestClient(create_app(tmp_path))


def test_ana_sayfa(istemci):
    r = istemci.get("/")
    assert r.status_code == 200
    assert "Kara Kutu" in r.text and "eval-only" in r.text


def test_kosum_veyu_rapor_indirme(istemci, tmp_path):
    with CSV.open("rb") as fh:  # tarayıcının boş sayıları gibi: esik/top_k eksik
        r = istemci.post("/api/kosum", files={"dosya": ("ornek.csv", fh, "text/csv")},
                         data={"limit": "40", "esik": "", "top_k": ""})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ozet"]["satir_sayisi"] == 40
    assert j["ozet"]["adapter"] == "fraud"
    assert len(j["verdictler"]) <= 50
    kid = j["ozet"]["kosum"]

    dosyalar = {p.name for p in (tmp_path / kid).iterdir()}
    assert {"sonuc.json", "kanit.json", "ic_rapor.md", "dis_rapor.html"} <= dosyalar

    html = istemci.get(f"/kosumlar/{kid}/dis_rapor.html")
    assert html.status_code == 200 and "data-kaynak" in html.text
    md = istemci.get(f"/kosumlar/{kid}/ic_rapor.md")
    assert md.status_code == 200 and md.text.startswith("# Kara Kutu İç Rapor")

    liste = istemci.get("/api/kosumlar").json()
    assert liste and liste[0]["kosum"] == kid


def test_kosum_esik_gecerli(istemci, tmp_path):
    with CSV.open("rb") as fh:
        r = istemci.post("/api/kosum", files={"dosya": ("ornek.csv", fh, "text/csv")},
                         data={"limit": "12", "esik": "0.5"})
    assert r.status_code == 200, r.text
    kid = r.json()["ozet"]["kosum"]
    assert r.json()["ozet"]["esik"] == pytest.approx(0.5)
    # override'ın kaynağı dosya-atfı DEĞİL, "kullanıcı override" notu olmalı
    import json
    sonuc = json.loads((tmp_path / kid / "sonuc.json").read_text())
    kanit = json.loads((tmp_path / kid / "kanit.json").read_text())
    assert sonuc["esik"] == pytest.approx(0.5)
    assert sonuc["meta"]["esik_kaynak"] == "kullanıcı override (CLI --esik / web formu)"
    assert kanit["ek_atiflar"]["esik_kaynak"] == "kullanıcı override (CLI --esik / web formu)"
    assert "infer.py" not in json.dumps(kanit)  # sahte dosya-atfı sızmamalı


def test_csv_disi_red(istemci):
    r = istemci.post("/api/kosum", files={"dosya": ("notlar.txt", b"merhaba", "text/plain")})
    assert r.status_code == 400


def test_bozuk_esik_red(istemci):
    with CSV.open("rb") as fh:
        r = istemci.post("/api/kosum", files={"dosya": ("ornek.csv", fh, "text/csv")},
                         data={"esik": "yogurt"})
    assert r.status_code == 400


def test_yol_ziplama_red(istemci):
    for kid in ("..", "20260101-000000_fraud/../../etc", "otis"):
        r = istemci.get(f"/kosumlar/{kid}/kanit.json")
        assert r.status_code in (400, 404, 422)
