from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from loader_cmapss import FEATURE_DIM, _trajectory, load_cmapss


def test_raw_rows_convert_to_36d_and_derive_reverse_cycle_rul():
    raw = np.asarray([
        [1, 1, 0.1, 1.0, 2.0], [1, 2, 0.2, 1.5, 2.5],
        [1, 3, 0.3, 2.0, 3.0], [2, 1, 0.0, 4.0, 5.0],
        [2, 2, 0.1, 4.5, 5.5],
    ])
    parsed = _trajectory(raw)
    assert parsed["features"].shape == (5, FEATURE_DIM)
    assert parsed["rul"].tolist() == [2, 1, 0, 1, 0]
    assert np.allclose(parsed["features"][0, 3:], 0)
    assert np.allclose(parsed["features"][1, 3:5], [0.1, 0.5])


def test_official_fd001_is_complete_when_local_data_is_available():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "cmapss"
    if not data_dir.exists() or not list(data_dir.rglob("train_FD001.txt")):
        return
    data = load_cmapss(data_dir)
    assert data["train"]["raw_rows"] == 20631
    assert data["test"]["raw_rows"] == 13096
    assert data["train"]["raw_columns"] == 26
    assert data["train"]["features"].shape == (20631, FEATURE_DIM)
    assert data["test"]["engines"] == len(np.loadtxt(data_dir / "CMaps" / "RUL_FD001.txt", ndmin=1))
