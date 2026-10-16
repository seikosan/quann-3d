from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from e2p_design import GRID, SEED, TrainQuantizer, _add_pair_angles, _rank_anonymous_features, _split


def test_e2p_grid_covers_point_budget_pair_angle_bucket_and_window_variants():
    assert len(GRID) == 6
    assert {design.window for design in GRID} == {5, 10, 20}
    assert {design.buckets for design in GRID} == {4, 8}
    assert {design.point_family for design in GRID} == {"anonymous_pair", "dual_channel"}
    assert any(not design.explicit_pair_angles for design in GRID)
    assert any(design.explicit_pair_angles for design in GRID)


def test_quantizer_fits_edges_on_train_points_and_bounds_bucket_coordinates():
    train = np.asarray([[[0.0, 2.0], [1.0, 4.0]], [[2.0, 6.0], [3.0, 8.0]]], dtype=np.float32)
    quantizer = TrainQuantizer(4).fit(train)
    transformed = quantizer.transform(np.asarray([[[-100.0, 1000.0], [3.0, 8.0]]], dtype=np.float32))

    assert quantizer.actual_bucket_counts == [4, 4]
    assert np.all(transformed >= -1.0)
    assert np.all(transformed <= 1.0)
    assert np.array_equal(quantizer.edges[0], np.asarray([0.75, 1.5, 2.25], dtype=np.float32))


def test_pair_angles_append_time_anonymous_and_in_out_channels():
    points = np.asarray([[[1.0, 1.0, 0.0, 1.0]]], dtype=np.float32)
    augmented = _add_pair_angles(points)

    assert augmented.shape == (1, 1, 6)
    assert np.isclose(augmented[0, 0, 4], 0.25)
    assert np.isclose(augmented[0, 0, 5], 0.0)


def test_anonymous_feature_selection_is_deterministic_and_training_scoped():
    rng = np.random.default_rng(SEED)
    raw = rng.normal(size=(100, 6))
    labels = np.asarray([0] * 80 + [1] * 20)
    raw[80:, 4] += 2.5
    first = _rank_anonymous_features(raw, labels, np.arange(100), count=2)
    second = _rank_anonymous_features(raw, labels, np.arange(100), count=2)

    assert first == second
    assert 4 in first


def test_e2p_split_is_stratified_and_repeatable():
    labels = np.asarray([0] * 1800 + [1] * 200)
    split_a = _split(labels)
    split_b = _split(labels)

    assert all(np.array_equal(a, b) for a, b in zip(split_a, split_b))
    assert [int(labels[ids].sum()) for ids in split_a] == [120, 40, 40]
