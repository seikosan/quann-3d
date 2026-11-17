"""Synthetic 36-feature turbofan degradation fallback for C-MAPSS FD001 E0."""
from __future__ import annotations

import numpy as np

import config


def generate_turbofan(n_engines: int = 48, max_cycles: int = 140, seed: int = config.SEED):
    """Generate engine trajectories with 3 operating settings + 33 sensors.

    Sensors combine operating-condition offsets, engine-specific baselines,
    monotonic nonlinear wear trends and measurement noise. Returns chronological
    records with engine_id, cycle, RUL, and 36 features.
    """
    rng = np.random.default_rng(seed)
    operating_centers = np.asarray([[0.10, 0.04, -0.05], [0.45, -0.10, 0.18], [0.82, 0.13, 0.06]])
    sensor_offsets = rng.normal(0, 0.8, size=(3, 33))
    wear_direction = rng.choice([-1.0, 1.0], size=33)
    wear_scale = rng.uniform(0.15, 1.25, size=33)
    engines = []
    for engine in range(n_engines):
        lifetime = int(rng.integers(max(80, int(max_cycles * 0.72)), max_cycles + 1))
        base = rng.normal(0.0, 0.35, 33)
        for cycle in range(1, lifetime + 1):
            progress = cycle / lifetime
            op = (cycle // 17 + engine) % 3
            setting = operating_centers[op] + rng.normal(0, [0.025, 0.015, 0.02])
            trend = wear_direction * wear_scale * (progress ** 1.7)
            sensor = base + sensor_offsets[op] + trend + rng.normal(0, 0.06 + 0.04 * progress, 33)
            engines.append({
                "engine_id": engine,
                "cycle": cycle,
                "rul": lifetime - cycle,
                "features": np.concatenate([setting, sensor]).astype(np.float32),
            })
    return engines


def make_windows(records, window_size: int = config.WINDOW_SIZE, step: int = 2):
    """Turn each engine trajectory into stride-2 rolling 10×36 clouds."""
    by_engine = {}
    for row in records:
        by_engine.setdefault(row["engine_id"], []).append(row)
    clouds, rul, groups = [], [], []
    for engine_id, rows in by_engine.items():
        rows.sort(key=lambda row: row["cycle"])
        values = np.stack([row["features"] for row in rows])
        for end in range(window_size - 1, len(rows), step):
            clouds.append(values[end - window_size + 1:end + 1])
            rul.append(int(rows[end]["rul"]))
            groups.append(int(engine_id))
    return np.asarray(clouds, dtype=np.float32), np.asarray(rul, dtype=np.int64), np.asarray(groups, dtype=np.int64)


def stage_labels(rul):
    """Return four RUL bands, with class 0 as the <=30-cycle critical stage."""
    return np.digitize(np.asarray(rul), bins=[30, 60, 90]).astype(np.int64)
