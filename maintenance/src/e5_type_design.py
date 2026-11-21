"""E5 CWRU fault-type point/feature design grid; CPU-only and recording-disjoint."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
from sklearn.metrics import (accuracy_score, average_precision_score, fbeta_score,
                             recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from loader_cwru_exp import FEATURE_DIM, load_expanded_records
from run_cwru_exp import SEED, recording_disjoint_split
from train_maintenance import MaintenanceCloud


CLASSES = ("ball", "inner", "outer")
# Feature indices follow loader_cwru_exp: per channel [RMS, crest, 13 bearing bands].
# In the band block BSF (ball) is local 7, BPFO (outer) local 4, BPFI (inner) local 9.
PAIR_SETS = {
    "rms_crest": (("rms_crest", (0, 1)),),
    "bearing_energy": (("ball_inner", (9, 11)), ("inner_outer", (11, 6)),
                       ("ball_outer", (9, 6))),
    "mixed": (("rms_crest", (0, 1)), ("ball_inner", (9, 11)),
              ("inner_outer", (11, 6)), ("ball_outer", (9, 6))),
}


@dataclass(frozen=True)
class Design:
    name: str
    pair_set: str
    representation: str
    buckets: int
    window_segments: int


GRID = (
    Design("rms_crest_B8_K1", "rms_crest", "raw_pair", 8, 1),
    Design("bearing_energy_B8_K1", "bearing_energy", "angle_pair", 8, 1),
    Design("mixed_raw_B8_K1", "mixed", "raw_pair", 8, 1),
    Design("mixed_angle_B8_K1", "mixed", "angle_pair", 8, 1),
    Design("mixed_angle_B4_K1", "mixed", "angle_pair", 4, 1),
    Design("mixed_angle_B8_K3", "mixed", "angle_pair", 8, 3),
)


def _log(path: Path, message: str) -> None:
    line = message if message.startswith("=== ") else f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _split_records(records: list[dict], seed: int = SEED) -> dict:
    """Use exactly the fault_type recording stratification and seed of cwru_exp."""
    split = recording_disjoint_split(records, "fault_type", seed=seed)
    train_ids = set(split["split"]["train_recording_ids"])
    test_ids = set(split["split"]["test_recording_ids"])
    by_id = {record["recording_id"]: record for record in records
             if record["fault_type"] in CLASSES}
    return {"train": [by_id[name] for name in sorted(train_ids)],
            "test": [by_id[name] for name in sorted(test_ids)],
            "split": split["split"]}


def _feature_scaler(train_records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([r["features"][:, :30] for r in train_records], axis=0)
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1e-7] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _pair_descriptors(features: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                      design: Design) -> np.ndarray:
    """Return per-segment (channel × pair, descriptor) points from the 32D layout."""
    normalized = (features[:, :30] - mean) / scale
    specs = PAIR_SETS[design.pair_set]
    descriptors = []
    for channel in range(2):
        offset = channel * 15
        for _, (left, right) in specs:
            x, y = normalized[:, offset + left], normalized[:, offset + right]
            if design.representation == "angle_pair":
                theta = np.arctan2(y, x)
                radius = np.log1p(np.hypot(x, y))
                pair = np.stack((np.cos(theta), np.sin(theta), radius), axis=-1)
            else:
                pair = np.stack((x, y), axis=-1)
            descriptors.append(pair.astype(np.float32))
    return np.stack(descriptors, axis=1)


def _descriptor_scaler(train_descriptors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(train_descriptors, axis=0)
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1e-7] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _record_windows(records: list[dict], descriptors: dict[str, np.ndarray], k: int,
                    label_index: dict[str, int]) -> list[dict]:
    windows = []
    for record in records:
        rows = descriptors[record["recording_id"]]
        stop = len(rows) - (len(rows) % k)
        for start in range(0, stop, k):
            windows.append({"recording_id": record["recording_id"],
                            "y": label_index[record["fault_type"]],
                            "descriptors": rows[start:start + k]})
    return windows


def _flatten_window(window: dict) -> np.ndarray:
    points = window["descriptors"]
    summary = np.concatenate((points.mean(axis=0), points.std(axis=0)), axis=-1)
    return summary.reshape(-1).astype(np.float32)


def _cloud_points(window: dict) -> np.ndarray:
    """Embed each measured feature-pair descriptor in a role-tagged 32D point."""
    values = window["descriptors"].reshape(-1, window["descriptors"].shape[1],
                                           window["descriptors"].shape[2])
    n_roles, d_desc = values.shape[1:]
    if n_roles > 8 or d_desc > 3:
        raise ValueError(f"unsupported point design shape roles={n_roles}, descriptor={d_desc}")
    points = np.zeros((len(values) * n_roles, FEATURE_DIM), dtype=np.float32)
    for time_index in range(len(values)):
        for role in range(n_roles):
            row = time_index * n_roles + role
            points[row, role] = 1.0
            points[row, 8:8 + d_desc] = values[time_index, role]
            points[row, 12:12 + d_desc] = np.sign(values[time_index, role]) * np.sqrt(
                np.abs(values[time_index, role]) + 1e-8)
    return points


def _fit_tokenizers(train_windows: list[dict], n_roles: int, buckets: int,
                    seed: int) -> tuple[list[dict], int]:
    tokenizers, vocab_size = [], 0
    for role in range(n_roles):
        values = np.concatenate([window["descriptors"][:, role, :] for window in train_windows], axis=0)
        unique_count = len(np.unique(np.round(values, 6), axis=0))
        n_clusters = min(buckets, len(values), max(1, unique_count))
        if n_clusters == 1:
            tokenizers.append({"model": None, "active": np.asarray([0]), "offset": vocab_size})
            vocab_size += 1
            continue
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed + role, n_init=3,
                                batch_size=min(256, max(32, len(values))))
        model.fit(values)
        active = np.unique(model.predict(values))
        tokenizers.append({"model": model, "active": active, "offset": vocab_size})
        vocab_size += len(active)
    return tokenizers, vocab_size


def _window_tokens(window: dict, tokenizers: list[dict]) -> np.ndarray:
    values = window["descriptors"]
    token_rows = []
    for role, tokenizer in enumerate(tokenizers):
        model, active = tokenizer["model"], tokenizer["active"]
        role_values = values[:, role, :]
        if model is None:
            mapped = np.zeros(len(role_values), dtype=np.int64)
        else:
            raw = model.predict(role_values)
            # Empty MiniBatchKMeans buckets are discarded; map to the nearest active centroid.
            centers = model.cluster_centers_[active]
            distances = ((role_values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
            mapped = distances.argmin(axis=1).astype(np.int64)
        token_rows.append(tokenizer["offset"] + mapped)
    return np.stack(token_rows, axis=1).reshape(-1).astype(np.int64)


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    pred = probabilities.argmax(axis=1)
    one_hot = np.eye(len(CLASSES), dtype=np.int64)[y]
    f2 = [fbeta_score(one_hot[:, i], pred == i, beta=2, zero_division=0)
          for i in range(len(CLASSES))]
    recall = recall_score(y, pred, labels=np.arange(len(CLASSES)), average=None, zero_division=0)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "auc_roc_macro_ovr": float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro")),
            "auc_pr_macro": float(average_precision_score(one_hot, probabilities, average="macro")),
            "f2_macro": float(np.mean(f2)),
            "class_recall": {name: float(recall[i]) for i, name in enumerate(CLASSES)}}


def _fit_torch(regime: str, train_windows: list[dict], test_windows: list[dict],
               tokenizers: list[dict], vocab_size: int, seed: int) -> tuple[np.ndarray, dict, MaintenanceCloud]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = MaintenanceCloud(vocab_size=max(1, vocab_size), n_classes=len(CLASSES), D=FEATURE_DIM)
    parameters = list(model.cloud_encoder.parameters()) + list(model.class_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=0.002, weight_decay=1e-4)
    labels = np.asarray([window["y"] for window in train_windows], dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(len(CLASSES)):
        ids = np.flatnonzero(labels == class_id)
        selected.extend(rng.choice(ids, min(len(ids), 70), replace=False).tolist())
    rng.shuffle(selected)

    def tokens_for(window: dict) -> torch.Tensor:
        return torch.as_tensor(_window_tokens(window, tokenizers)[None, :], dtype=torch.long)

    for index in selected:
        window = train_windows[index]
        token = tokens_for(window)
        points = (model.points_for(token).detach() if regime == "hybrid" else
                  torch.as_tensor(_cloud_points(window)[None, :, :], dtype=torch.float32))
        model.train()
        loss = torch.nn.functional.cross_entropy(model(points), torch.tensor([window["y"]]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if regime == "hybrid":
            model.eval()
            model.quann_update(token, int(window["y"]))

    outputs = []
    model.eval()
    with torch.no_grad():
        for window in test_windows:
            token = tokens_for(window)
            points = (model.points_for(token) if regime == "hybrid" else
                      torch.as_tensor(_cloud_points(window)[None, :, :], dtype=torch.float32))
            outputs.append(torch.softmax(model(points), dim=1).cpu().numpy()[0])
    info = model.convergence_info() if regime == "hybrid" else {
        "points": 0, "converged": 0, "mean_width": None}
    info.update({"fit_examples": len(selected), "regime": regime})
    return np.asarray(outputs), info, model


def _fit_design(split: dict, design: Design, seed: int) -> tuple[dict, dict]:
    train_records, test_records = split["train"], split["test"]
    label_index = {name: index for index, name in enumerate(CLASSES)}
    feature_mean, feature_scale = _feature_scaler(train_records)
    raw_desc = {r["recording_id"]: _pair_descriptors(r["features"], feature_mean,
                                                    feature_scale, design)
                for r in train_records + test_records}
    descriptor_mean, descriptor_scale = _descriptor_scaler(
        [raw_desc[r["recording_id"]] for r in train_records])
    descriptors = {key: ((value - descriptor_mean) / descriptor_scale).astype(np.float32)
                   for key, value in raw_desc.items()}
    train_windows = _record_windows(train_records, descriptors, design.window_segments, label_index)
    test_windows = _record_windows(test_records, descriptors, design.window_segments, label_index)
    x_train = np.stack([_flatten_window(window) for window in train_windows])
    x_test = np.stack([_flatten_window(window) for window in test_windows])
    y_train = np.asarray([window["y"] for window in train_windows], dtype=np.int64)
    y_test = np.asarray([window["y"] for window in test_windows], dtype=np.int64)

    gbt = HistGradientBoostingClassifier(max_iter=70, max_leaf_nodes=9, learning_rate=0.08,
                                         l2_regularization=0.1, random_state=seed)
    gbt.fit(x_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    grid_models = {"gbt": _metrics(y_test, gbt.predict_proba(x_test))}

    n_roles = train_windows[0]["descriptors"].shape[1]
    tokenizers, vocab_size = _fit_tokenizers(train_windows, n_roles, design.buckets, seed)
    models = {}
    hybrid_model = None
    for regime in ("hybrid", "backprop"):
        probabilities, info, fitted_model = _fit_torch(regime, train_windows, test_windows,
                                                       tokenizers, vocab_size, seed)
        grid_models[regime] = {**_metrics(y_test, probabilities),
                               "points": info["points"], "points_converged": info["converged"],
                               "mean_width": info["mean_width"],
                               "fit_examples": info["fit_examples"]}
        if regime == "hybrid":
            hybrid_model = fitted_model
        models[regime] = fitted_model

    details = {"test_y": y_test, "test_windows": test_windows, "train_windows": train_windows,
               "tokenizers": tokenizers, "models": models, "gbt": gbt,
               "test_features": x_test, "split": split["split"]}
    result = {"config": {"name": design.name, "feature_pair_set": design.pair_set,
                         "representation": design.representation,
                         "pairs_per_channel": len(PAIR_SETS[design.pair_set]),
                         "points_per_segment": n_roles, "buckets_per_pair_channel": design.buckets,
                         "window_segments": design.window_segments,
                         "window_policy": "non-overlapping consecutive extracted segments within each recording; incomplete tail dropped"},
              "n_train_windows": len(train_windows), "n_test_windows": len(test_windows),
              "split": split["split"], "models": grid_models,
              "tokenizer": {"fit": "train-only MiniBatchKMeans independently per channel-feature pair",
                            "requested_buckets": design.buckets,
                            "vocabulary_tokens": vocab_size}}
    return result, details


def _figure(details: dict, design: Design, path: Path) -> dict:
    model = details["models"]["hybrid"]
    windows = details["test_windows"]
    y = details["test_y"]
    sample_points = []
    point_labels = []
    means = []
    for window in windows:
        token = torch.as_tensor(_window_tokens(window, details["tokenizers"])[None, :], dtype=torch.long)
        with torch.no_grad():
            points = model.points_for(token).cpu().numpy()[0]
        sample_points.append(points)
        point_labels.extend([window["y"]] * len(points))
        means.append(points.mean(axis=0))
    flat = np.concatenate(sample_points, axis=0)
    point_labels = np.asarray(point_labels, dtype=np.int64)
    pca = PCA(n_components=3, random_state=SEED)
    projected = pca.fit_transform(flat)
    sample_pca = pca.transform(np.stack(means))[:, :2]

    fig = plt.figure(figsize=(13, 6.5), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    colors = ("#d45356", "#3978a8", "#32936f")
    for class_id, name in enumerate(CLASSES):
        mask = point_labels == class_id
        ax3d.scatter(projected[mask, 0], projected[mask, 1], projected[mask, 2],
                     s=10, alpha=.34, color=colors[class_id], label=f"{name} cloud points")
    ax3d.set_title("Hybrid 32D token-point clouds · PCA-3D")
    ax3d.set_xlabel("PC1"); ax3d.set_ylabel("PC2"); ax3d.set_zlabel("PC3")
    ax3d.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(1, 2, 2)
    centroids = np.stack([sample_pca[y == class_id].mean(axis=0)
                          for class_id in range(len(CLASSES))])
    lower = sample_pca.min(axis=0) - 0.5
    upper = sample_pca.max(axis=0) + 0.5
    gx, gy = np.meshgrid(np.linspace(lower[0], upper[0], 180),
                         np.linspace(lower[1], upper[1], 180))
    grid = np.stack((gx, gy), axis=-1)
    labels = ((grid[..., None, :] - centroids[None, None, :, :]) ** 2).sum(axis=-1).argmin(axis=-1)
    ax.contourf(gx, gy, labels, levels=np.arange(3 + 1) - 0.5,
                colors=colors, alpha=.16)
    for class_id, name in enumerate(CLASSES):
        mask = y == class_id
        ax.scatter(sample_pca[mask, 0], sample_pca[mask, 1], s=18, alpha=.58,
                   color=colors[class_id], label=name)
        ax.scatter(*centroids[class_id], marker="*", s=210, color=colors[class_id],
                   edgecolor="#222", linewidth=.7)
    ax.set_title("2D cloud-mean map · nearest-class-centroid regions")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(frameon=False)
    fig.suptitle(f"E5 best design: {design.name} · test recordings only")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return {"pca_explained_variance_percent": (pca.explained_variance_ratio_ * 100).tolist(),
            "region_map": "nearest class centroid in 2D PCA of per-window hybrid cloud means",
            "plotted_test_windows": int(len(windows)),
            "hybrid_cloud_points_per_window": int(len(sample_points[0]))}


def run_e5(data_dir: str | Path | None = None, out_path: str | Path | None = None,
           figure_path: str | Path | None = None, log_path: str | Path | None = None) -> dict:
    project = ROOT / "maintenance"
    data_dir = Path(data_dir or project / "data" / "cwru" / "expanded")
    out_path = Path(out_path or project / "results" / "e5_type_design.json")
    figure_path = Path(figure_path or project / "figures" / "e5_type_design.png")
    log_path = Path(log_path or project / "logs" / "run_e5_smoke.log")
    for directory in (out_path.parent, figure_path.parent, log_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
    started = time.monotonic()
    _log(log_path, f"=== [{datetime.now().date()}] 1/4: load local expanded CWRU records and lock common recording split ===")
    records = load_expanded_records(data_dir, lambda message: _log(log_path, message))
    split = _split_records(records)
    train_ids, test_ids = set(split["split"]["train_recording_ids"]), set(split["split"]["test_recording_ids"])
    if train_ids & test_ids:
        raise AssertionError("recording leakage in E5 split")
    _log(log_path, f"fault recordings={len(train_ids) + len(test_ids)}; train={len(train_ids)} test={len(test_ids)}; seed={SEED}")

    _log(log_path, "=== E5 2/4: run six feature-pair / angle / bucket / window designs across GBT, hybrid, backprop ===")
    grid, detail_by_name = [], {}
    for number, design in enumerate(GRID, 1):
        _log(log_path, f"=== E5 design {number}/{len(GRID)}: {design.name} ===")
        result, details = _fit_design(split, design, SEED)
        grid.append(result)
        detail_by_name[design.name] = details
        _log(log_path, f"{design.name}: " + json.dumps(result["models"], ensure_ascii=False))

    best = max(grid, key=lambda row: (row["models"]["hybrid"]["auc_roc_macro_ovr"],
                                      row["models"]["hybrid"]["class_recall"]["inner"],
                                      row["models"]["hybrid"]["auc_pr_macro"]))
    best_design = next(design for design in GRID if design.name == best["config"]["name"])
    _log(log_path, "=== E5 3/4: render best hybrid class clouds and geometric region map ===")
    figure = _figure(detail_by_name[best_design.name], best_design, figure_path)

    baseline_path = project / "results" / "cwru_expansion.json"
    baseline_json = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = baseline_json["tasks"]["fault_type_ball_inner_outer"]["models"]
    best_models = best["models"]
    hybrid_gap = best_models["gbt"]["auc_roc_macro_ovr"] - best_models["hybrid"]["auc_roc_macro_ovr"]
    inner_recall = best_models["hybrid"]["class_recall"]["inner"]
    grid_gaps = [{"configuration": row["config"]["name"],
                  "hybrid_gbt_roc_gap": row["models"]["gbt"]["auc_roc_macro_ovr"] - row["models"]["hybrid"]["auc_roc_macro_ovr"],
                  "inner_recall": row["models"]["hybrid"]["class_recall"]["inner"]}
                 for row in grid]
    smallest_gap = min(grid_gaps, key=lambda row: row["hybrid_gbt_roc_gap"])
    highest_inner = max(grid_gaps, key=lambda row: row["inner_recall"])
    comparison = {
        "reference": "maintenance/results/cwru_expansion.json; same 20261016 seed and 75/25 recording-disjoint split",
        "cwru_exp_fault_type": {name: {key: baseline[name][key] for key in
                                       ("auc_roc_macro_ovr", "auc_pr_macro", "f2_macro", "class_recall")}
                                for name in ("gbt", "hybrid", "backprop")},
        "e5_best": {name: {key: best_models[name][key] for key in
                           ("auc_roc_macro_ovr", "auc_pr_macro", "f2_macro", "class_recall")}
                    for name in ("gbt", "hybrid", "backprop")},
        "best_hybrid_delta_vs_cwru_exp": {
            "auc_roc_macro_ovr": best_models["hybrid"]["auc_roc_macro_ovr"] - baseline["hybrid"]["auc_roc_macro_ovr"],
            "auc_pr_macro": best_models["hybrid"]["auc_pr_macro"] - baseline["hybrid"]["auc_pr_macro"],
            "inner_recall": inner_recall - baseline["hybrid"]["class_recall"]["inner"]},
        "cwru_exp_hybrid_gbt_roc_gap": baseline["gbt"]["auc_roc_macro_ovr"] - baseline["hybrid"]["auc_roc_macro_ovr"],
        "best_configuration_hybrid_minus_gbt_roc_gap": -hybrid_gap,
        "criteria": {"inner_recall_at_least_0_5": inner_recall >= 0.5,
                     "hybrid_gbt_roc_gap_at_most_0_10": hybrid_gap <= 0.10,
                     "both_met": inner_recall >= 0.5 and hybrid_gap <= 0.10},
        "grid_goal_scan": {"best_inner_recall": highest_inner,
                           "smallest_hybrid_gbt_roc_gap": smallest_gap,
                           "any_configuration_meets_inner_recall": highest_inner["inner_recall"] >= 0.5,
                           "any_configuration_meets_roc_gap": smallest_gap["hybrid_gbt_roc_gap"] <= 0.10,
                           "any_configuration_meets_both": any(
                               row["inner_recall"] >= 0.5 and row["hybrid_gbt_roc_gap"] <= 0.10
                               for row in grid_gaps)}}

    angle_effect = (next(row for row in grid if row["config"]["name"] == "mixed_angle_B8_K1")["models"]["hybrid"]["auc_roc_macro_ovr"] -
                    next(row for row in grid if row["config"]["name"] == "mixed_raw_B8_K1")["models"]["hybrid"]["auc_roc_macro_ovr"])
    bucket_effect = (next(row for row in grid if row["config"]["name"] == "mixed_angle_B8_K1")["models"]["hybrid"]["auc_roc_macro_ovr"] -
                     next(row for row in grid if row["config"]["name"] == "mixed_angle_B4_K1")["models"]["hybrid"]["auc_roc_macro_ovr"])
    window_effect = (next(row for row in grid if row["config"]["name"] == "mixed_angle_B8_K3")["models"]["hybrid"]["auc_roc_macro_ovr"] -
                     next(row for row in grid if row["config"]["name"] == "mixed_angle_B8_K1")["models"]["hybrid"]["auc_roc_macro_ovr"])
    result = {"experiment": "E5 CWRU three-class fault-type point/feature design grid",
              "classes": list(CLASSES), "grid": grid,
              "best_configuration": {"name": best_design.name, "models": best_models,
                                     "selection": "maximum hybrid macro-OvR ROC; ties by inner recall then PR",
                                     "geometry_figure": str(figure_path), "figure_details": figure},
              "design_effects": {"hybrid_roc_angle_vs_raw_mixed_B8_K1": angle_effect,
                                 "hybrid_roc_B8_minus_B4_mixed_angle_K1": bucket_effect,
                                 "hybrid_roc_K3_minus_K1_mixed_angle_B8": window_effect,
                                 "interpretation": "paired grid contrasts isolate angle encoding, bucket count, and temporal window length; feature-pair budget contrasts compare RMS/crest, bearing-frequency pairs, and mixed."},
              "e5_vs_cwru_exp": comparison,
              "protocol": {"device": "cpu", "seed": SEED,
                           "split": "recording-disjoint stratified 75/25 fault-type split, identical recording IDs to run_cwru_exp.py",
                           "channels": "DE and FE kept as separate point roles; missing FE feature slots remain loader zero-values",
                           "input_mapping": "per-channel feature-pair descriptor -> train-only pair tokenizer -> one learned 32D QUANN point per pair/channel/bucket -> multi-point cloud; backprop uses measured descriptor points in the same role-tagged 32D geometry",
                           "split_recording_ids": {"train": sorted(train_ids), "test": sorted(test_ids)},
                           "elapsed_seconds": round(time.monotonic() - started, 2)},
              "limitations": ["The 80 CWRU files are the source recordings, but fault-type evaluation excludes the four healthy recordings as in cwru_exp.",
                              "K3 groups consecutive extracted rows; loader rows are distributed through each source recording rather than a continuous overlapping stream, and incomplete tails are dropped.",
                              "Feature-pair role tags use a deterministic fixed 32D embedding; hybrid QUANN centers are learned only by the existing zero-order point update, not by gradient descent.",
                              "A single held-out recording split and one seed provide a design probe, not a robust estimate across split replicates."]}

    _log(log_path, "=== E5 4/4: save result JSON and comparison to CWRU expansion ===")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _log(log_path, f"saved {out_path}; best={best_design.name}; inner recall={inner_recall:.4f}; hybrid-Gbt ROC gap={hybrid_gap:.4f}; elapsed={result['protocol']['elapsed_seconds']}s")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    run_e5(data_dir=args.data_dir)
