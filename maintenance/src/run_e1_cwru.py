"""Tiny CPU comparison on a normal / ball-fault pair from CWRU."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from loader_cwru import FEATURE_DIM, SAMPLE_RATE, SEGMENT_SAMPLES, load_cwru
from train_maintenance import MaintenanceCloud


SEED = 20261016
CLASS_NAMES = ("normal", "ball_fault_007")


def _log(path: Path, message: str) -> None:
    line = message if message.startswith("=== ") else f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _split_recordings(records: dict[str, dict]) -> tuple[np.ndarray, ...]:
    train_x, train_y, test_x, test_y = [], [], [], []
    for _, record in records.items():
        features = record["features"]
        labels = record["labels"]
        cut = max(1, int(len(features) * 0.70))
        if cut >= len(features):
            raise ValueError(f"not enough non-overlapping segments to hold out from {record['source']}")
        train_x.append(features[:cut])
        train_y.append(labels[:cut])
        test_x.append(features[cut:])
        test_y.append(labels[cut:])
    return tuple(np.concatenate(values, axis=0) for values in (train_x, train_y, test_x, test_y))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix_normal_fault": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def _fit_cloud(regime: str, train_x: np.ndarray, train_tokens: np.ndarray,
               train_y: np.ndarray, test_x: np.ndarray, test_tokens: np.ndarray,
               vocab_size: int) -> np.ndarray:
    torch.manual_seed(SEED)
    model = MaintenanceCloud(vocab_size=vocab_size, n_classes=2, D=FEATURE_DIM)
    optimizer = torch.optim.AdamW(
        list(model.cloud_encoder.parameters()) + list(model.class_head.parameters()),
        lr=0.002, weight_decay=1e-4,
    )
    rng = np.random.default_rng(SEED)
    selected = []
    for class_id in np.unique(train_y):
        candidates = np.flatnonzero(train_y == class_id)
        selected.extend(rng.choice(candidates, min(len(candidates), 60), replace=False).tolist())
    rng.shuffle(selected)
    torch.set_num_threads(1)
    for index in selected:
        token_batch = torch.as_tensor(train_tokens[index:index + 1], dtype=torch.long)
        target = int(train_y[index])
        model.train()
        if regime == "hybrid":
            points = model.points_for(token_batch).detach()
        else:
            points = torch.as_tensor(train_x[index:index + 1, None, :], dtype=torch.float32)
        logits = model(points)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if regime == "hybrid":
            model.eval()
            model.quann_update(token_batch, target)

    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_tokens), 64):
            token_batch = torch.as_tensor(test_tokens[start:start + 64], dtype=torch.long)
            if regime == "hybrid":
                points = model.points_for(token_batch)
            else:
                points = torch.as_tensor(test_x[start:start + 64, None, :], dtype=torch.float32)
            predictions.extend(model(points).argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(predictions, dtype=np.int64)


def _plot(clouds: np.ndarray, labels: np.ndarray, train_ids: np.ndarray, path: Path) -> float:
    pca = PCA(n_components=2, random_state=SEED)
    projected = pca.fit_transform(clouds)
    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    colors = ("#3978a8", "#c83e4d")
    for class_id, name in enumerate(CLASS_NAMES):
        points = projected[labels == class_id]
        train_points = projected[train_ids[labels[train_ids] == class_id]]
        ax.scatter(points[:, 0], points[:, 1], s=22, color=colors[class_id], alpha=0.55,
                   label=f"{name} segments")
        center = train_points.mean(axis=0)
        ax.scatter(*center, marker="*", s=250, color=colors[class_id], edgecolor="#202020",
                   linewidth=0.8, label=f"{name} train centroid")
    variance = float(100 * pca.explained_variance_ratio_.sum())
    ax.set_title(f"CWRU DE/FE vibration-feature points · PCA-2D ({variance:.1f}% variance)")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return variance


def run_e1_cwru(data_dir: str | Path | None = None, out_path: str | Path | None = None,
                figure_path: str | Path | None = None, log_path: str | Path | None = None) -> dict:
    project = ROOT / "maintenance"
    data_dir = Path(data_dir or project / "data" / "cwru")
    out_path = Path(out_path or project / "results" / "e1_cwru.json")
    figure_path = Path(figure_path or project / "figures" / "e1_cwru_clouds.png")
    log_path = Path(log_path or project / "logs" / "run_e1_smoke.log")
    for directory in (out_path.parent, figure_path.parent, log_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    _log(log_path, f"=== [{datetime.now().date()}] CWRU 1/4: load 12 kHz normal / 0.007-inch ball-fault recordings ===")
    _log(log_path, "CWRU source checks: official Case pages/files reachable; GitHub data mirror AiChiXiaoXiongBingGan/CWRU-dataset available; Filters-whynot/CWRU-dataset and rigas/CWRU_Bearing_NumPy returned HTTP 404.")
    records = load_cwru(data_dir)
    train_x, train_y, test_x, test_y = _split_recordings(records)
    train_ids = np.arange(len(train_x))
    all_x = np.concatenate([train_x, test_x])
    all_y = np.concatenate([train_y, test_y])
    _log(log_path, f"segments normal={int(np.sum(all_y == 0))}, ball_fault_007={int(np.sum(all_y == 1))}; train/test={len(train_x)}/{len(test_x)}; channels=DE,FE")

    _log(log_path, "=== CWRU 2/4: train-only scaling and feature-token clustering ===")
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_train = ((train_x - mean) / scale).astype(np.float32)
    x_test = ((test_x - mean) / scale).astype(np.float32)
    x_all = ((all_x - mean) / scale).astype(np.float32)
    clusterer = MiniBatchKMeans(n_clusters=12, random_state=SEED, n_init=3, batch_size=128)
    clusterer.fit(x_train)
    train_tokens = clusterer.predict(x_train).reshape(-1, 1)
    test_tokens = clusterer.predict(x_test).reshape(-1, 1)

    _log(log_path, "=== CWRU 3/4: fit compact GBT, QUANN-hybrid, and backprop readers ===")
    gbt = HistGradientBoostingClassifier(max_iter=70, max_leaf_nodes=7, learning_rate=0.08,
                                         l2_regularization=0.1, random_state=SEED)
    gbt.fit(x_train, train_y)
    results = {"gbt": _metrics(test_y, gbt.predict(x_test))}
    torch_models = {}
    for regime in ("hybrid", "backprop"):
        predictions = _fit_cloud(regime, x_train, train_tokens, train_y,
                                 x_test, test_tokens, int(clusterer.n_clusters))
        results[regime] = _metrics(test_y, predictions)
        torch_models[regime] = "one-pass, class-capped CPU fit"

    _log(log_path, "=== CWRU 4/4: save 32D point PCA and comparison results ===")
    pca_variance = _plot(x_all, all_y, train_ids, figure_path)
    class_counts = {name: int(np.sum(all_y == class_id)) for class_id, name in enumerate(CLASS_NAMES)}
    result = {
        "experiment": "tiny CWRU normal vs ball fault comparison",
        "data": "CWRU 12 kHz, motor load 0 HP, bearing ball defect 0.007 inch",
        "sources": {
            "official_dataset_page": "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data",
            "normal_file": "https://engineering.case.edu/sites/default/files/97.mat",
            "fault_file": "https://engineering.case.edu/sites/default/files/118.mat",
            "working_github_data_mirror": "https://github.com/AiChiXiaoXiongBingGan/CWRU-dataset",
            "mirror_checks": [
                {"url": "https://github.com/Filters-whynot/CWRU-dataset", "status": "404"},
                {"url": "https://github.com/rigas/CWRU_Bearing_NumPy", "status": "404"},
            ],
        },
        "recordings": {key: {"file": Path(record["source"]).name,
                              "source_samples": record["source_samples"],
                              "segments": record["segments"], "rpm": record["rpm"],
                              "load_hp": record["load_hp"], "channels_used": record["channels"]}
                       for key, record in records.items()},
        "features": {"dimension": FEATURE_DIM, "sample_rate_hz": SAMPLE_RATE,
                     "segment_samples": SEGMENT_SAMPLES,
                     "layout": "DE and FE each: RMS, crest, 13 shaft/bearing-frequency band energies; plus normalized RPM and load"},
        "split": {"method": "first 70% of non-overlapping segments per recording train, final 30% held out",
                  "train_segments": int(len(train_y)), "test_segments": int(len(test_y)),
                  "class_counts_total": class_counts},
        "models": results,
        "protocol": {"device": "cpu", "seed": SEED, "torch_fit": torch_models,
                     "point_cloud": "one 32D feature point per raw 2048-sample segment"},
        "figure": str(figure_path),
        "pca_explained_variance_percent": pca_variance,
        "limitations": [
            "This is a tiny two-recording smoke comparison, not a benchmark: load is fixed at 0 HP and fault type/severity is one 0.007-inch ball defect.",
            "Held-out segments are contiguous later portions from the same two recordings; this is not recording-disjoint validation.",
            "DE and FE are the common channels; BA is omitted because it is absent from the normal file.",
            "Metrics on the small held-out set are high-variance and not tuned estimates.",
        ],
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _log(log_path, f"CWRU saved {out_path}; counts={class_counts}; metrics={results}")
    return result


if __name__ == "__main__":
    run_e1_cwru()
