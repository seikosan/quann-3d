"""P0 parity kapısı — docs/KARA_KUTU_PLAN.md §5.1.

Deploy paketinin manifest'indeki üç gerçek satırın ONNX skorları, eğitilmiş
torch referansıyla (manifest.p_fraud) satır satır karşılaştırılır. Tolerans
uydurulmaz: manifest'in kendi ölçtüğü max_abs_diff (1.2814998626708984e-06) +
%25 yeniden-koşum payı = 1.6e-06. Plan §5.1'in "≤1e-6" şartı manifest ölçümüyle
çelişir; dürüst olan manifest'e uymaktır (kara_kutu/tests: tolerans kaynağı
`fraud/results/deploy/manifest.json` self_test.max_abs_diff).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "fraud" / "results" / "deploy"
CSV = REPO / "fraud" / "data" / "User0_credit_card_transactions.csv"
TOL = 1.6e-06  # manifest self_test.max_abs_diff = 1.2814998626708984e-06 + pay


def test_parity_manifest_cases():
    pytest.importorskip("onnxruntime")
    from kara_kutu.core.contract import Kayit, VeriSeti
    from kara_kutu.core.registry import get_adapter

    man = json.loads((DEPLOY / "manifest.json").read_text())
    decoder = json.loads((DEPLOY / "token_decoder.json").read_text())
    assert decoder, "boş token_decoder"

    frame = pd.read_csv(CSV)
    assert len(frame) == man["self_test"]["n_rows"]

    adapter = get_adapter("fraud")
    ds = VeriSeti(
        ad=CSV.name,
        kolonlar=[str(c) for c in frame.columns],
        kayitlar=[Kayit(i, r) for i, r in enumerate(frame.to_dict("records"))],
    )
    verdictler = adapter.predict(ds)
    assert len(verdictler) == len(frame)
    by_id = {v.kimlik: v for v in verdictler}
    for case in man["self_test"]["cases"]:
        v = by_id[case["row"]]
        assert abs(v.skor - case["p_fraud"]) <= TOL, (
            f"satır {case['row']}: adapter {v.skor!r} vs manifest {case['p_fraud']!r}")
