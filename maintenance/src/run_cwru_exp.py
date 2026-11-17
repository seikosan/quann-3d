"""Recording-disjoint expanded CWRU benchmark, CPU-only."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, fbeta_score,
                             recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from loader_cwru_exp import (FEATURE_DIM, SAMPLE_RATE, SEGMENT_SAMPLES,
                             catalog_summary, download_recordings,
                             load_expanded_records)
from train_maintenance import MaintenanceCloud


SEED = 20261016
CLASSES = {"binary": ("normal", "fault"), "fault_type": ("ball", "inner", "outer")}


def _log(path: Path, message: str) -> None:
    line = message if message.startswith("=== ") else f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def recording_disjoint_split(records: list[dict], task: str, seed: int = SEED,
                             train_fraction: float = 0.75) -> dict:
    """Split whole .mat files per target class; no segment from a file crosses the split."""
    eligible = [r for r in records if task != "fault_type" or r["fault_type"] != "normal"]
    class_names = CLASSES[task]
    label_by_record = lambda record: (int(record["fault_type"] != "normal") if task == "binary"
                                      else class_names.index(record["fault_type"]))
    train_records, test_records = [], []
    rng = np.random.default_rng(seed)
    for class_id, class_name in enumerate(class_names):
        local = [r for r in eligible if label_by_record(r) == class_id]
        local = sorted(local, key=lambda r: r["recording_id"])
        if len(local) < 2:
            raise ValueError(f"task={task} class={class_name} needs >=2 recordings, found {len(local)}")
        order = rng.permutation(len(local))
        cut = min(len(local) - 1, max(1, int(round(len(local) * train_fraction))))
        train_records.extend(local[index] for index in order[:cut])
        test_records.extend(local[index] for index in order[cut:])
    train_records.sort(key=lambda r: r["recording_id"])
    test_records.sort(key=lambda r: r["recording_id"])

    def unpack(rows):
        x = np.concatenate([r["features"] for r in rows], axis=0).astype(np.float32)
        y = np.concatenate([np.full(r["segments"], label_by_record(r), dtype=np.int64) for r in rows])
        return x, y

    train_x, train_y = unpack(train_records)
    test_x, test_y = unpack(test_records)
    train_ids = {r["recording_id"] for r in train_records}
    test_ids = {r["recording_id"] for r in test_records}
    if train_ids & test_ids:
        raise AssertionError("recording leakage in split")
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y,
            "train_records": train_records, "test_records": test_records,
            "class_names": class_names,
            "split": {"method": "recording-disjoint stratified 75/25 by target class",
                      "train_recordings": len(train_records), "test_recordings": len(test_records),
                      "train_recording_ids": sorted(train_ids), "test_recording_ids": sorted(test_ids),
                      "train_segments": len(train_y), "test_segments": len(test_y)}}


def _f2_binary(y: np.ndarray, pred: np.ndarray) -> float:
    return float(fbeta_score(y, pred, beta=2, zero_division=0))


def _metrics(y: np.ndarray, probabilities: np.ndarray, class_names: tuple[str, ...]) -> dict:
    prediction = probabilities.argmax(axis=1)
    recall = recall_score(y, prediction, labels=np.arange(len(class_names)), average=None, zero_division=0)
    if len(class_names) == 2:
        positive = probabilities[:, 1]
        return {"n": int(len(y)), "accuracy": float(accuracy_score(y, prediction)),
                "auc_roc": float(roc_auc_score(y, positive)),
                "auc_pr": float(average_precision_score(y, positive)),
                "f2": _f2_binary(y, prediction),
                "class_recall": {name: float(recall[i]) for i, name in enumerate(class_names)}}
    one_hot = np.eye(len(class_names), dtype=np.int64)[y]
    class_f2 = [fbeta_score(one_hot[:, i], prediction == i, beta=2, zero_division=0)
                for i in range(len(class_names))]
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, prediction)),
            "auc_roc_macro_ovr": float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro")),
            "auc_pr_macro": float(average_precision_score(one_hot, probabilities, average="macro")),
            "f2_macro": float(np.mean(class_f2)),
            "class_recall": {name: float(recall[i]) for i, name in enumerate(class_names)}}


def _scale(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-7] = 1.0
    return ((train_x - mean) / scale).astype(np.float32), ((other_x - mean) / scale).astype(np.float32), mean, scale


def _fit_torch(regime: str, train_x: np.ndarray, train_tokens: np.ndarray, train_y: np.ndarray,
               test_x: np.ndarray, test_tokens: np.ndarray, n_classes: int,
               task_tag: str) -> tuple[np.ndarray, dict]:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = MaintenanceCloud(vocab_size=int(train_tokens.max()) + 1, n_classes=n_classes, D=FEATURE_DIM)
    reader_parameters = list(model.cloud_encoder.parameters()) + list(model.class_head.parameters())
    optimizer = torch.optim.AdamW(reader_parameters, lr=0.002, weight_decay=1e-4)
    rng = np.random.default_rng(SEED + (0 if regime == "hybrid" else 1))
    selected = []
    for class_id in range(n_classes):
        ids = np.flatnonzero(train_y == class_id)
        selected.extend(rng.choice(ids, min(len(ids), 70), replace=False).tolist())
    rng.shuffle(selected)
    for index in selected:
        token = torch.as_tensor(train_tokens[index:index + 1], dtype=torch.long)
        target = torch.tensor([int(train_y[index])], dtype=torch.long)
        model.train()
        if regime == "hybrid":
            points = model.points_for(token).detach()
        else:
            points = torch.as_tensor(train_x[index:index + 1, None, :], dtype=torch.float32)
        loss = torch.nn.functional.cross_entropy(model(points), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if regime == "hybrid":
            model.eval()
            model.quann_update(token, int(train_y[index]))

    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(test_x), 128):
            tokens = torch.as_tensor(test_tokens[start:start + 128], dtype=torch.long)
            if regime == "hybrid":
                points = model.points_for(tokens)
            else:
                points = torch.as_tensor(test_x[start:start + 128, None, :], dtype=torch.float32)
            outputs.append(torch.softmax(model(points), dim=1).cpu().numpy())
    info = model.convergence_info() if regime == "hybrid" else {"points": 0, "converged": 0, "mean_width": None}
    info.update({"fit_examples": len(selected), "regime": regime, "task": task_tag})
    return np.concatenate(outputs), info


def _run_task(records: list[dict], task: str) -> dict:
    data = recording_disjoint_split(records, task)
    x_train, x_test, mean, scale = _scale(data["train_x"], data["test_x"])
    y_train, y_test = data["train_y"], data["test_y"]
    n_classes = len(data["class_names"])
    cluster_count = min(12, max(2, len(x_train) // 20))
    clusterer = MiniBatchKMeans(n_clusters=cluster_count, random_state=SEED, n_init=3,
                                batch_size=min(256, max(32, len(x_train))))
    clusterer.fit(x_train)
    train_tokens = clusterer.predict(x_train).reshape(-1, 1)
    test_tokens = clusterer.predict(x_test).reshape(-1, 1)
    gbt = HistGradientBoostingClassifier(max_iter=70, max_leaf_nodes=9, learning_rate=0.08,
                                         l2_regularization=0.1, random_state=SEED)
    gbt.fit(x_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    gbt_prob = gbt.predict_proba(x_test)
    models = {"gbt": {**_metrics(y_test, gbt_prob, data["class_names"]),
                       "points": 0, "points_converged": 0, "mean_width": None}}
    for regime in ("hybrid", "backprop"):
        probabilities, convergence = _fit_torch(regime, x_train, train_tokens, y_train,
                                                x_test, test_tokens, n_classes, task)
        models[regime] = {**_metrics(y_test, probabilities, data["class_names"]),
                          "points": convergence["points"],
                          "points_converged": convergence["converged"],
                          "mean_width": convergence["mean_width"],
                          "fit_examples": convergence["fit_examples"]}
    return {"class_names": list(data["class_names"]), "split": data["split"], "models": models,
            "tokenizer": {"type": "train-only MiniBatchKMeans", "clusters": cluster_count},
            "scaling": {"fit_on_train_only": True, "mean": mean.tolist(), "scale": scale.tolist()}}


def _plot_clouds(records: list[dict], path: Path) -> float:
    fault_records = [r for r in records if r["fault_type"] in ("ball", "inner", "outer")]
    x = np.concatenate([r["features"] for r in fault_records])
    y = np.concatenate([np.full(r["segments"], r["fault_type"]) for r in fault_records])
    pca = PCA(n_components=3, random_state=SEED)
    points = pca.fit_transform(x)
    fig = plt.figure(figsize=(10, 7.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    palette = {"ball": "#d45356", "inner": "#3978a8", "outer": "#32936f"}
    for fault in ("ball", "inner", "outer"):
        selected = y == fault
        ax.scatter(points[selected, 0], points[selected, 1], points[selected, 2],
                   s=17, alpha=.48, color=palette[fault], label=f"{fault} raceway")
    variance = float(pca.explained_variance_ratio_.sum() * 100)
    ax.set_title(f"CWRU vibration-feature clouds by fault type · PCA-3D ({variance:.1f}% variance)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return variance


def run_cwru_exp(data_dir: str | Path | None = None, out_path: str | Path | None = None,
                 figure_path: str | Path | None = None, log_path: str | Path | None = None,
                 do_download: bool = True) -> dict:
    project = ROOT / "maintenance"
    data_dir = Path(data_dir or project / "data" / "cwru" / "expanded")
    out_path = Path(out_path or project / "results" / "cwru_expansion.json")
    figure_path = Path(figure_path or project / "figures" / "e1_cwru_exp_clouds.png")
    log_path = Path(log_path or project / "logs" / "run_cwru_exp.log")
    for directory in (out_path.parent, figure_path.parent, log_path.parent, data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
    _log(log_path, f"=== [{datetime.now().date()}] 1/5: official CWRU 12 kHz DE/FE catalog and bounded retrieval ===")
    download = (download_recordings(data_dir, lambda message: _log(log_path, message))
                if do_download else {"skipped_by_option": True})
    _log(log_path, "=== CWRU 2/5: extract up to 24 distributed non-overlapping 2048-sample segments per recording ===")
    records = load_expanded_records(data_dir, lambda message: _log(log_path, message))
    _log(log_path, f"readable recordings={len(records)}; positions={sorted({r['position'] for r in records})}; classes={sorted({r['fault_type'] for r in records})}")
    _log(log_path, "=== CWRU 3/5: recording-disjoint binary and fault-type splits ===")
    binary = _run_task(records, "binary")
    fault_type = _run_task(records, "fault_type")
    _log(log_path, "=== CWRU 4/5: render three-dimensional raw-feature clouds by fault type ===")
    pca_variance = _plot_clouds(records, figure_path)
    loaded_counts = {}
    for r in records:
        key = f"{r['position']}/{r['fault_type']}/{r['severity_in']:.3f}in"
        loaded_counts[key] = loaded_counts.get(key, 0) + 1
    result = {
        "experiment": "CWRU expanded recording-disjoint bearing fault comparison",
        "source": "Official Case Western Reserve University 12 kHz drive-end and fan-end datasets; GitHub mirror fallback for DE files only",
        "catalog": catalog_summary(),
        "download": download,
        "recordings_loaded": len(records),
        "recordings_loaded_by_position_type_size": loaded_counts,
        "recording_metadata": [{k: r[k] for k in ("recording_id", "position", "fault_type", "severity_in", "fault_location", "load_hp", "rpm", "rpm_source", "channels_available", "segments", "source_samples")} for r in records],
        "features": {"dimension": FEATURE_DIM, "sample_rate_hz": SAMPLE_RATE,
                     "segment_samples": SEGMENT_SAMPLES, "max_segments_per_recording": 24,
                     "layout": "DE 15 features (RMS, crest, 13 bearing-frequency bands), FE 15 same; RPM and load context; absent channel slots are zero"},
        "tasks": {"binary_normal_vs_fault": binary, "fault_type_ball_inner_outer": fault_type},
        "protocol": {"device": "cpu", "seed": SEED, "split_group": "source .mat recording",
                     "points_converged": "hybrid QUANN point count and convergence; GBT/backprop have no QUANN points"},
        "figure": str(figure_path), "figure_pca_explained_variance_percent": pca_variance,
        "limitations": [
            "The public 12 kHz tables provide 0.007/0.014/0.021 inch faults plus 0.028 inch ball at DE; no official 0.035 inch CWRU recording was listed, so it was not fabricated.",
            "Fault location is CWRU drive-end versus fan-end dataset provenance; available vibration channels vary by .mat file, and the channel availability is recorded per recording.",
            "At most 24 evenly distributed segments per source file are used; segments are not independent experimental units, while the train/test assignment is disjoint by full source recording.",
            "The 12 kHz official catalog has only one healthy recording per load; binary test estimates therefore have few independent normal recordings.",
        ],
    }
    _log(log_path, "=== CWRU 5/5: save result JSON ===")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _log(log_path, f"saved {out_path}; readable={len(records)}; binary={binary['models']}; fault_type={fault_type['models']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    run_cwru_exp(do_download=not args.skip_download)
