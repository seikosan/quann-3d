from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import config
from e1_real import _distances, _feature_profile, _reason, _stage_distance_summary
from loader_cwru import FEATURE_DIM as CWRU_FEATURE_DIM, load_cwru


def test_stage_distance_summary_reports_own_other_ratio_and_retrieval_accuracy():
    labels = np.asarray([0, 0, 1, 1])
    distances = np.asarray([
        [0.2, 1.0, 1.5, 1.7], [0.4, 0.8, 1.2, 1.6],
        [0.9, 0.3, 0.8, 1.2], [0.7, 0.5, 0.9, 1.0],
    ])

    summary = _stage_distance_summary(labels, distances)

    assert summary[config.STAGES[0]]["nearest_prototype_accuracy"] == 1.0
    assert np.isclose(summary[config.STAGES[0]]["median_own_distance"], 0.3)
    assert np.isclose(summary[config.STAGES[0]]["median_nearest_other_distance"], 0.9)
    assert np.isclose(summary[config.STAGES[0]]["median_own_to_nearest_other_ratio"], 0.35)
    assert summary[config.STAGES[2]]["n"] == 0
    assert summary[config.STAGES[2]]["nearest_prototype_accuracy"] is None


def test_reason_is_based_on_window_sensor_statistics_and_names_distance():
    train_clouds = np.zeros((4, 10, 36), dtype=np.float32)
    train_clouds[:, :, 3:24] = 1.0
    train_clouds[0, :, 3:24] = 0.5
    labels = np.asarray([0, 0, 0, 0])
    profile = _feature_profile(train_clouds, labels)
    window = np.zeros((10, 36), dtype=np.float32)
    window[:, 3] = np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 3.0])
    window[:, 4:24] = 1.0

    reason = _reason(window, 0, profile, 0.21)

    assert "Sensör 1 RMS" in reason
    assert "crest" in reason
    assert "mesafe 0.210" in reason


def test_distances_are_euclidean_in_encoder_space():
    points = np.asarray([[0.0, 0.0], [3.0, 4.0]])
    prototypes = np.asarray([[0.0, 0.0], [0.0, 4.0]])

    assert np.allclose(_distances(points, prototypes), [[0.0, 4.0], [5.0, 3.0]])


def test_local_cwru_pair_loads_as_finite_32d_normal_and_fault_points():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "cwru"
    if not (data_dir / "97.mat").is_file() or not (data_dir / "118.mat").is_file():
        return

    records = load_cwru(data_dir)

    assert records["normal"]["features"].shape == (119, CWRU_FEATURE_DIM)
    assert records["ball_fault_007"]["features"].shape == (59, CWRU_FEATURE_DIM)
    assert records["normal"]["channels"] == ["DE", "FE"]
    assert all(np.isfinite(record["features"]).all() for record in records.values())
