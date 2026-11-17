from __future__ import annotations

import os
import sys

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

import config
from generator import generate_turbofan, make_windows, stage_labels
from train_maintenance import MaintenanceCloud, _scale_by_train


def test_synthetic_turbofan_has_36_features_and_wear_trend():
    records = generate_turbofan(n_engines=6, max_cycles=100, seed=117)
    assert records[0]["features"].shape == (36,)
    assert {row["engine_id"] for row in records} == set(range(6))
    engine = [row for row in records if row["engine_id"] == 0]
    early = np.stack([row["features"][3:] for row in engine[:15]]).mean()
    late = np.stack([row["features"][3:] for row in engine[-15:]]).mean()
    assert abs(late - early) > 0.05
    clouds, rul, _ = make_windows(records)
    assert clouds.shape[1:] == (config.WINDOW_SIZE, config.FEATURE_COUNT)
    assert np.all(np.diff(rul[:40]) <= 0)
    assert set(stage_labels(np.asarray([10, 45, 75, 120])).tolist()) == {0, 1, 2, 3}


def test_window_to_cluster_token_cloud_and_quann_smoke():
    records = generate_turbofan(n_engines=8, max_cycles=85, seed=91)
    clouds, rul, _ = make_windows(records)
    split = max(10, int(len(clouds) * 0.7))
    scaled, _, _ = _scale_by_train(clouds[:split], clouds[split:split + 2], clouds[split + 2:split + 12])
    train, heldout, _ = scaled
    cluster = MiniBatchKMeans(n_clusters=6, random_state=13, n_init=2, batch_size=64)
    cluster.fit(train.reshape(-1, 36))
    tokens = cluster.predict(train[:12].reshape(-1, 36)).reshape(12, 10)
    target = int(rul[0] <= 30)
    model = MaintenanceCloud(int(tokens.max()) + 1, n_classes=2)
    tt = torch.as_tensor(tokens[:1], dtype=torch.long)
    model.quann_update(tt, target)
    with torch.no_grad():
        logits = model(model.points_for(tt))
    assert logits.shape == (1, 2)
    assert model.points_for(tt).shape == (1, 10, 36)
    assert len(stage_labels(rul)) == len(clouds)
