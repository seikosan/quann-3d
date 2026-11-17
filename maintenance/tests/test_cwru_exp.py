import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "maintenance" / "src"))

from loader_cwru_exp import CATALOG, _mat_record, catalog_summary
from run_cwru_exp import recording_disjoint_split


def _record(name, fault_type, load=0):
    label = int(fault_type != "normal")
    x = np.full((4, 32), label + load / 10, dtype=np.float32)
    return {"features": x, "segments": len(x), "recording_id": name,
            "fault_type": fault_type, "load_hp": load}


def test_catalog_covers_two_positions_and_lists_unavailable_severity():
    positions = {spec.position for spec in CATALOG}
    assert positions == {"DE", "FE"}
    assert {spec.fault_type for spec in CATALOG} == {"normal", "ball", "inner", "outer"}
    assert {spec.severity_in for spec in CATALOG if spec.fault_type == "ball"} == {0.007, 0.014, 0.021, 0.028}
    assert "ball 0.035 inch" in catalog_summary()["not_available_in_official_12k_tables"]


def test_mat_feature_extraction_is_32d_and_tracks_rpm(tmp_path):
    spec = next(spec for spec in CATALOG if spec.file_id == 97)
    path = tmp_path / spec.filename
    t = np.arange(4096, dtype=np.float64) / 12_000
    savemat(path, {"X105_DE_time": np.sin(2 * np.pi * 120 * t),
                   "X105_FE_time": np.sin(2 * np.pi * 80 * t), "X105RPM": np.array([[1797]])})
    record = _mat_record(path, spec, max_segments=2)
    assert record["features"].shape == (2, 32)
    assert record["rpm"] == 1797.0
    assert record["rpm_source"] == "mat_rpm"
    assert record["channels_available"] == ["DE", "FE"]


def test_both_tasks_hold_out_whole_source_recordings():
    records = []
    for class_name in ("normal", "ball", "inner", "outer"):
        for index in range(4):
            records.append(_record(f"{class_name}-{index}.mat", class_name, index % 4))
    binary = recording_disjoint_split(records, "binary")
    fault_type = recording_disjoint_split(records, "fault_type")
    for split in (binary, fault_type):
        train = set(split["split"]["train_recording_ids"])
        test = set(split["split"]["test_recording_ids"])
        assert train.isdisjoint(test)
        assert len(split["train_y"]) > 0 and len(split["test_y"]) > 0
    assert fault_type["class_names"] == ("ball", "inner", "outer")
    assert set(fault_type["train_y"]) == {0, 1, 2}
    assert set(fault_type["test_y"]) == {0, 1, 2}
