"""Load raw NASA C-MAPSS FD001 files into fixed-width sensor features."""
from __future__ import annotations

from pathlib import Path

import numpy as np


FEATURE_DIM = 36


def _locate(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found below {root}")
    return matches[0]


def _read_matrix(path: Path) -> np.ndarray:
    """Read whitespace-delimited C-MAPSS text or a numeric CSV as float64."""
    delimiter = "," if path.suffix.lower() == ".csv" else None
    try:
        values = np.loadtxt(path, delimiter=delimiter, dtype=np.float64, ndmin=2)
    except ValueError:
        values = np.loadtxt(path, delimiter=delimiter, dtype=np.float64, ndmin=2, skiprows=1)
    if values.ndim != 2 or values.shape[1] < 5 or not np.isfinite(values).all():
        raise ValueError(f"invalid C-MAPSS matrix {path}: shape={values.shape}")
    return values


def _features(measurements: np.ndarray, engine_ids: np.ndarray) -> np.ndarray:
    """Keep measurements and append per-engine baseline deltas to reach 36D."""
    width = measurements.shape[1]
    if width >= FEATURE_DIM:
        return measurements[:, :FEATURE_DIM].astype(np.float32)
    extra = FEATURE_DIM - width
    deltas = np.zeros((len(measurements), extra), dtype=np.float64)
    for engine in np.unique(engine_ids):
        rows = np.flatnonzero(engine_ids == engine)
        count = min(extra, width)
        deltas[rows, :count] = measurements[rows, :count] - measurements[rows[0], :count]
    return np.concatenate([measurements, deltas], axis=1).astype(np.float32)


def _trajectory(raw: np.ndarray, endpoint_rul: np.ndarray | None = None) -> dict[str, np.ndarray]:
    engine_ids = raw[:, 0].astype(np.int64)
    cycles = raw[:, 1].astype(np.int64)
    measurements = raw[:, 2:]
    features = _features(measurements, engine_ids)
    rul = np.empty(len(raw), dtype=np.int64)
    engines = np.unique(engine_ids)
    if endpoint_rul is not None and len(endpoint_rul) != len(engines):
        raise ValueError(f"RUL labels={len(endpoint_rul)} do not match engines={len(engines)}")
    for index, engine in enumerate(engines):
        rows = np.flatnonzero(engine_ids == engine)
        terminal_cycle = int(cycles[rows].max())
        offset = int(endpoint_rul[index]) if endpoint_rul is not None else 0
        rul[rows] = offset + terminal_cycle - cycles[rows]
    return {"features": features, "engine_ids": engine_ids, "cycles": cycles, "rul": rul,
            "raw_rows": len(raw), "raw_columns": raw.shape[1], "engines": len(engines)}


def load_cmapss(data_dir: str | Path, subset: str = "FD001") -> dict[str, dict[str, np.ndarray]]:
    """Load train/test trajectories; official test endpoint labels are applied if present."""
    root = Path(data_dir).expanduser().resolve()
    train_path = _locate(root, f"train_{subset}.txt") if list(root.rglob(f"train_{subset}.txt")) else _locate(root, f"train_{subset}.csv")
    test_path = _locate(root, f"test_{subset}.txt") if list(root.rglob(f"test_{subset}.txt")) else _locate(root, f"test_{subset}.csv")
    train = _trajectory(_read_matrix(train_path))
    test_raw = _read_matrix(test_path)
    label_paths = sorted(root.rglob(f"RUL_{subset}.txt")) + sorted(root.rglob(f"RUL_{subset}.csv"))
    endpoint_rul = None
    if label_paths:
        label_path = label_paths[0]
        delimiter = "," if label_path.suffix.lower() == ".csv" else None
        endpoint_rul = np.loadtxt(label_path, delimiter=delimiter, dtype=np.float64, ndmin=1).reshape(-1).astype(np.int64)
    test = _trajectory(test_raw, endpoint_rul)
    return {"train": train, "test": test, "sources": {"train": str(train_path), "test": str(test_path),
            "rul": str(label_paths[0]) if label_paths else None}}
