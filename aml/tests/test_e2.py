from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from e2_elliptic import POINT_DIM, WINDOW, _metrics, _split


def test_account_proxy_experiment_contract_dimensions_and_stratified_split():
    labels = np.asarray([0] * 90 + [1] * 10)
    clouds = np.zeros((len(labels), WINDOW, POINT_DIM), dtype=np.float32)
    train, val, test = _split(clouds, labels)
    for split in (train, val, test):
        assert len(np.unique(labels[split])) == 2
    assert clouds.shape[1:] == (10, 36)


def test_e2_metrics_include_requested_fraud_recall_and_thresholds():
    y_val = np.asarray([0] * 90 + [1] * 10)
    p_val = np.column_stack([1 - np.linspace(0, 1, 100), np.linspace(0, 1, 100)])
    y_test = np.asarray([0, 0, 1, 1])
    p_test = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    result = _metrics(y_val, p_val, y_test, p_test)
    assert {"auc_roc", "auc_pr", "f2", "lift", "fraud_recall_at_1pct_fpr"} <= result["test"].keys()
    assert 0 <= result["test"]["fraud_recall_at_1pct_fpr"] <= 1


def test_official_elliptic_csvs_are_complete_when_local_data_is_available():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "elliptic"
    if not data_dir.exists() or not list(data_dir.rglob("elliptic_txs_classes.csv")):
        return
    import pandas as pd
    root = next(data_dir.rglob("elliptic_txs_classes.csv")).parent
    labels = pd.read_csv(root / "elliptic_txs_classes.csv")
    edges = pd.read_csv(root / "elliptic_txs_edgelist.csv")
    assert len(labels) == 203769
    assert len(edges) == 234355
    assert int((labels["class"].astype(str) == "1").sum()) == 4545
