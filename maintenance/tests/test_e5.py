import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "maintenance" / "src"))
sys.path.insert(0, str(ROOT))

from e5_type_design import (GRID, Design, _cloud_points, _pair_descriptors,
                            _record_windows, _split_records)


def _record(name, fault_type, value):
    features = np.full((6, 32), value, dtype=np.float32)
    features[:, 30:] = 0.0
    return {"recording_id": name, "fault_type": fault_type, "features": features,
            "segments": len(features)}


def test_grid_covers_matched_angle_bucket_and_window_contrasts():
    names = {design.name for design in GRID}
    assert {"mixed_raw_B8_K1", "mixed_angle_B8_K1"} <= names
    assert {"mixed_angle_B4_K1", "mixed_angle_B8_K1"} <= names
    assert {"mixed_angle_B8_K1", "mixed_angle_B8_K3"} <= names
    assert len(GRID) == 6


def test_split_is_recording_disjoint_and_pair_cloud_has_expected_geometry():
    records = [_record(f"{fault}-{i}.mat", fault, i + class_id * 10)
               for class_id, fault in enumerate(("ball", "inner", "outer"))
               for i in range(4)]
    split = _split_records(records)
    train_ids = {r["recording_id"] for r in split["train"]}
    test_ids = {r["recording_id"] for r in split["test"]}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) == 9 and len(test_ids) == 3

    design = Design("test", "mixed", "angle_pair", 8, 3)
    mean, scale = np.zeros(30, dtype=np.float32), np.ones(30, dtype=np.float32)
    descriptors = {r["recording_id"]: _pair_descriptors(r["features"], mean, scale, design)
                   for r in split["train"]}
    windows = _record_windows(split["train"], descriptors, 3,
                              {"ball": 0, "inner": 1, "outer": 2})
    assert len(windows) == 18
    assert windows[0]["descriptors"].shape == (3, 8, 3)
    points = _cloud_points(windows[0])
    assert points.shape == (24, 32)
    assert np.all(points[:, :8].sum(axis=1) == 1.0)
