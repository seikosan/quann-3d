"""E1 explainability and operating-condition drift demo on real C-MAPSS FD001."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

import config
from loader_cmapss import FEATURE_DIM, load_cmapss
from run_e0b_cmapss import SEED, WINDOW, _fit_cloud, _labels, _scale, _windows


STAGE_COLORS = ("#c83e4d", "#e68a32", "#3978a8", "#32936f")
OPERATION_DIMS = np.asarray([0, 1, 2], dtype=int)


def _log(path: Path, message: str) -> None:
    line = message if message.startswith("=== ") else f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _encode(model, tokens: np.ndarray, batch_size: int = 128) -> np.ndarray:
    encoded = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            token_batch = torch.as_tensor(tokens[start:start + batch_size], dtype=torch.long)
            points = model.points_for(token_batch)
            mask = torch.ones(token_batch.shape, dtype=torch.bool)
            encoded.append(model.cloud_encoder(points, mask).cpu().numpy())
    return np.concatenate(encoded, axis=0)


def _distances(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    return np.linalg.norm(embeddings[:, None, :] - prototypes[None, :, :], axis=2)


def _stage_distance_summary(labels: np.ndarray, distances: np.ndarray) -> dict:
    summary = {}
    for stage_id, name in enumerate(config.STAGES):
        selected = labels == stage_id
        if not np.any(selected):
            summary[name] = {"n": 0, "nearest_prototype_accuracy": None,
                             "median_own_distance": None, "median_nearest_other_distance": None,
                             "median_own_to_nearest_other_ratio": None,
                             "mean_own_to_nearest_other_ratio": None}
            continue
        stage_distances = distances[selected]
        own = stage_distances[:, stage_id]
        other = np.delete(stage_distances, stage_id, axis=1).min(axis=1)
        ratio = own / np.maximum(other, 1e-12)
        summary[name] = {
            "n": int(selected.sum()),
            "nearest_prototype_accuracy": float(np.mean(stage_distances.argmin(axis=1) == stage_id)),
            "median_own_distance": float(np.median(own)),
            "median_nearest_other_distance": float(np.median(other)),
            "median_own_to_nearest_other_ratio": float(np.median(ratio)),
            "mean_own_to_nearest_other_ratio": float(np.mean(ratio)),
        }
    return summary


def _feature_profile(clouds: np.ndarray, labels: np.ndarray) -> dict:
    """RMS and crest-factor reference profiles for the 21 measured sensors."""
    profile = {}
    sensor_clouds = clouds[:, :, 3:24]
    rms = np.sqrt(np.mean(np.square(sensor_clouds), axis=1))
    peak = np.max(np.abs(sensor_clouds), axis=1)
    crest = peak / np.maximum(rms, 1e-8)
    active_sensors = np.std(sensor_clouds.reshape(-1, 21), axis=0) > 1e-6
    for stage_id, name in enumerate(config.STAGES):
        selected = labels == stage_id
        if np.any(selected):
            profile[name] = {"rms": np.median(rms[selected], axis=0),
                             "crest": np.median(crest[selected], axis=0),
                             "active": active_sensors}
        else:
            profile[name] = {"rms": np.ones(21), "crest": np.ones(21), "active": active_sensors}
    return profile


def _reason(cloud: np.ndarray, predicted_stage: int, profile: dict,
            prototype_distance: float) -> str:
    values = cloud[:, 3:24]
    rms = np.sqrt(np.mean(np.square(values), axis=0))
    crest = np.max(np.abs(values), axis=0) / np.maximum(rms, 1e-8)
    reference = profile[config.STAGES[predicted_stage]]
    usable = reference["active"] & (reference["rms"] > 1e-4) & (reference["crest"] > 1e-4)
    if not np.any(usable):
        return (f"Bu pencerede karşılaştırılabilir değişken sensör istatistiği yok; "
                f"{config.STAGES[predicted_stage]} prototipine mesafe {prototype_distance:.3f}.")
    rms_ratio = rms[usable] / reference["rms"][usable]
    crest_ratio = crest[usable] / reference["crest"][usable]
    sensor_ids = np.flatnonzero(usable)
    # Select one sensor for each statistic, favoring the strongest relative deviation.
    rms_sensor_local = int(np.argmax(np.abs(np.log(np.maximum(rms_ratio, 1e-8)))))
    crest_sensor_local = int(np.argmax(np.abs(np.log(np.maximum(crest_ratio, 1e-8)))))
    rms_sensor = int(sensor_ids[rms_sensor_local])
    crest_sensor = int(sensor_ids[crest_sensor_local])
    rms_change = (rms_ratio[rms_sensor_local] - 1.0) * 100.0
    rms_word = "artmış" if rms_change >= 0 else "azalmış"
    crest_word = "artmış" if crest_ratio[crest_sensor_local] >= 1.0 else "azalmış"
    return (
        f"Sensör {rms_sensor + 1} RMS %{abs(rms_change):.0f} {rms_word}, "
        f"sensör {crest_sensor + 1} crest {crest_ratio[crest_sensor_local]:.2f}× "
        f"({crest_word}); {config.STAGES[predicted_stage]} prototipine mesafe "
        f"{prototype_distance:.3f}. İstatistikler bu pencerenin ölçeklenmiş sensör değerlerinden türetilmiştir."
    )


def _plot_region_map(embeddings: np.ndarray, labels: np.ndarray, train_ids: np.ndarray,
                     prototypes: np.ndarray, path: Path) -> float:
    pca = PCA(n_components=2, random_state=SEED)
    projected = pca.fit_transform(embeddings)
    projected_prototypes = pca.transform(prototypes)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    markers = ("o", "s", "^", "D")
    for stage_id, name in enumerate(config.STAGES):
        class_train = projected[train_ids[labels[train_ids] == stage_id]]
        class_all = projected[labels == stage_id]
        if len(class_train) >= 5:
            bandwidth = float(np.clip(np.median(np.std(class_train, axis=0)) * len(class_train) ** (-1 / 6), 0.06, 1.2))
            low = np.quantile(class_train, 0.01, axis=0)
            high = np.quantile(class_train, 0.99, axis=0)
            pad = max(3.0 * bandwidth, 0.08)
            gx, gy = np.meshgrid(np.linspace(low[0] - pad, high[0] + pad, 140),
                                 np.linspace(low[1] - pad, high[1] + pad, 140))
            grid = np.column_stack([gx.ravel(), gy.ravel()])
            density = np.exp(KernelDensity(kernel="gaussian", bandwidth=bandwidth)
                             .fit(class_train).score_samples(grid)).reshape(gx.shape)
            peak = float(density.max())
            if peak > 0:
                ax.contourf(gx, gy, density, levels=peak * np.asarray([0.08, 0.22, 0.48, 1.01]),
                            colors=[STAGE_COLORS[stage_id]], alpha=0.055, zorder=1)
                ax.contour(gx, gy, density, levels=peak * np.asarray([0.14, 0.36, 0.68]),
                           colors=[STAGE_COLORS[stage_id]], linewidths=[0.8, 1.1, 1.5], alpha=0.85, zorder=2)
        ax.scatter(class_all[:, 0], class_all[:, 1], s=18, marker=markers[stage_id],
                   facecolors="none", edgecolors=STAGE_COLORS[stage_id], alpha=0.62,
                   linewidths=0.7, label=name, zorder=4)
        px, py = projected_prototypes[stage_id]
        ax.scatter(px, py, marker="*", s=270, color=STAGE_COLORS[stage_id],
                   edgecolor="#202020", linewidth=0.7, zorder=6)
        ax.annotate(f"{name} prototype", (px, py), xytext=(8, 8),
                    textcoords="offset points", fontsize=8, weight="bold")
    variance = 100.0 * float(pca.explained_variance_ratio_.sum())
    handles, labels_legend = ax.get_legend_handles_labels()
    handles.extend([Line2D([0], [0], marker="*", color="#202020", linestyle="none",
                           markersize=12, label="train centroid"),
                    Patch(facecolor="#888888", alpha=0.13, edgecolor="#666666", label="train KDE region")])
    labels_legend.extend(["train centroid", "train KDE region"])
    ax.legend(handles, labels_legend, loc="best", framealpha=0.95, fontsize=8)
    ax.set_title(f"NASA C-MAPSS FD001 · CloudEncoderND regions\n{len(embeddings):,} windows · PCA-2D ({variance:.1f}% variance); KDE is descriptive")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(True, color="#d9dee5", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return variance


def _plot_drift(baseline: np.ndarray, unshifted: np.ndarray, shifted: np.ndarray,
                threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    bins = np.linspace(0, max(float(np.max(np.concatenate([baseline, unshifted, shifted]))), threshold) * 1.08, 32)
    ax.hist(baseline, bins=bins, density=True, alpha=0.30, color="#3978a8", label="train healthy baseline")
    ax.hist(unshifted, bins=bins, density=True, alpha=0.52, color="#32936f", label="held-out healthy, unchanged")
    ax.hist(shifted, bins=bins, density=True, alpha=0.48, color="#c83e4d", label="held-out healthy, settings shifted")
    ax.axvline(threshold, color="#202020", linestyle="--", linewidth=1.5,
               label=f"alarm threshold (baseline q95={threshold:.3f})")
    ax.set_title("Operating-condition drift · distance to healthy prototype")
    ax.set_xlabel("CloudEncoderND Euclidean distance")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def run_e1(data_dir: str | Path | None = None, out_path: str | Path | None = None,
           region_path: str | Path | None = None, drift_path: str | Path | None = None,
           log_path: str | Path | None = None) -> dict:
    """Run a deterministic CPU E1 demo; output paths are injectable for tests."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    project = ROOT / "maintenance"
    data_dir = Path(data_dir or project / "data" / "cmapss")
    out_path = Path(out_path or project / "results" / "e1_cmapss.json")
    region_path = Path(region_path or project / "figures" / "e1_region_map.png")
    drift_path = Path(drift_path or project / "figures" / "e1_drift.png")
    log_path = Path(log_path or project / "logs" / "run_e1_smoke.log")
    for directory in (out_path.parent, region_path.parent, drift_path.parent, log_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    _log(log_path, f"=== [{datetime.now().date()}] 1/6: load real FD001 and reproduce E0b engine split ===")
    loaded = load_cmapss(data_dir)
    train_data, test_data = loaded["train"], loaded["test"]
    engines = np.unique(train_data["engine_ids"])
    rng = np.random.default_rng(SEED)
    rng.shuffle(engines)
    cut = max(1, int(len(engines) * 0.8))
    train_engines = set(map(int, engines[:cut]))
    val_engines = set(map(int, engines[cut:]))
    train_cloud, train_rul, _ = _windows(train_data, train_engines)
    val_cloud, val_rul, _ = _windows(train_data, val_engines)
    test_cloud, test_rul, _ = _windows(test_data)
    train_y, val_y, test_y = (_labels(rul, "stage") for rul in (train_rul, val_rul, test_rul))
    train_scaled, val_scaled, test_scaled = _scale(train_cloud, val_cloud, test_cloud)
    train_mu = train_cloud.reshape(-1, FEATURE_DIM).mean(axis=0)
    train_sd = train_cloud.reshape(-1, FEATURE_DIM).std(axis=0)
    train_sd[train_sd < 1e-6] = 1.0
    _log(log_path, f"FD001 windows train/val/test={len(train_cloud)}/{len(val_cloud)}/{len(test_cloud)}; features={FEATURE_DIM}")

    _log(log_path, "=== 2/6: fit train-only quantizer and E0b hybrid CloudEncoderND ===")
    clusterer = MiniBatchKMeans(n_clusters=24, random_state=SEED, n_init=2, batch_size=512)
    clusterer.fit(train_scaled.reshape(-1, FEATURE_DIM))
    train_tokens = clusterer.predict(train_scaled.reshape(-1, FEATURE_DIM)).reshape(-1, WINDOW)
    val_tokens = clusterer.predict(val_scaled.reshape(-1, FEATURE_DIM)).reshape(-1, WINDOW)
    test_tokens = clusterer.predict(test_scaled.reshape(-1, FEATURE_DIM)).reshape(-1, WINDOW)
    _, model = _fit_cloud("hybrid", train_scaled, train_tokens, train_y,
                          val_scaled, val_tokens, val_y, test_scaled, test_tokens,
                          test_y, 4, "stage", log_path)

    _log(log_path, "=== 3/6: encode windows and build four training-centroid prototypes ===")
    all_scaled = np.concatenate([train_scaled, val_scaled, test_scaled], axis=0)
    all_tokens = np.concatenate([train_tokens, val_tokens, test_tokens], axis=0)
    all_y = np.concatenate([train_y, val_y, test_y], axis=0)
    train_ids = np.arange(len(train_scaled))
    test_ids = np.arange(len(train_scaled) + len(val_scaled), len(all_scaled))
    embeddings = _encode(model, all_tokens)
    prototypes = np.stack([embeddings[train_ids][train_y == stage_id].mean(axis=0)
                           for stage_id in range(4)])
    test_distances = _distances(embeddings[test_ids], prototypes)
    distance_summary = _stage_distance_summary(test_y, test_distances)
    nearest_accuracy = float(np.mean(test_distances.argmin(axis=1) == test_y))
    _log(log_path, f"held-out nearest-prototype accuracy={nearest_accuracy:.4f}; embedding_dim={embeddings.shape[1]}")

    _log(log_path, "=== 4/6: generate held-out prototype-distance explanations ===")
    profile = _feature_profile(train_scaled, train_y)
    predicted = test_distances.argmin(axis=1)
    example_ids = []
    for stage_id in range(4):
        candidates = np.flatnonzero(test_y == stage_id)
        count = min(3, len(candidates))
        if count:
            example_ids.extend(rng.choice(candidates, count, replace=False).tolist())
    explanations = []
    for local_id in example_ids:
        stage_prediction = int(predicted[local_id])
        distance_row = test_distances[local_id]
        explanations.append({
            "test_window_id": int(local_id),
            "true_stage": config.STAGES[int(test_y[local_id])],
            "nearest_prototype_stage": config.STAGES[stage_prediction],
            "nearest_prototype_distance": float(distance_row[stage_prediction]),
            "second_prototype_stage": config.STAGES[int(np.argsort(distance_row)[1])],
            "second_prototype_distance": float(np.sort(distance_row)[1]),
            "nearest_prototype_correct": bool(stage_prediction == int(test_y[local_id])),
            "reason": _reason(test_scaled[local_id], stage_prediction, profile,
                              float(distance_row[stage_prediction])),
        })

    _log(log_path, "=== 5/6: measure healthy-baseline alarm under controlled setting shift ===")
    healthy_id = 3
    healthy_train_ids = train_ids[train_y == healthy_id]
    healthy_test_local = np.flatnonzero(test_y == healthy_id)
    healthy_proto = prototypes[healthy_id]
    baseline_distances = np.linalg.norm(embeddings[healthy_train_ids] - healthy_proto, axis=1)
    unshifted_embeddings = embeddings[test_ids[healthy_test_local]]
    unshifted_distances = np.linalg.norm(unshifted_embeddings - healthy_proto, axis=1)
    shifted_cloud = test_scaled[healthy_test_local].copy()
    # Settings 1-3 are raw columns 0-2; dimensions 33-35 are engineered baseline deltas.
    operation_shift = np.asarray([6.0, 6.0, -6.0], dtype=np.float32)
    shifted_cloud[:, :, OPERATION_DIMS] += operation_shift[None, None, :]
    shifted_tokens = clusterer.predict(shifted_cloud.reshape(-1, FEATURE_DIM)).reshape(-1, WINDOW)
    shifted_embeddings = _encode(model, shifted_tokens)
    shifted_distances = np.linalg.norm(shifted_embeddings - healthy_proto, axis=1)
    changed_token_fraction = float(np.mean(shifted_tokens != test_tokens[healthy_test_local]))
    threshold = float(np.quantile(baseline_distances, 0.95))
    base_alarm = float(np.mean(unshifted_distances > threshold)) if len(unshifted_distances) else 0.0
    shifted_alarm = float(np.mean(shifted_distances > threshold)) if len(shifted_distances) else 0.0
    _plot_drift(baseline_distances, unshifted_distances, shifted_distances, threshold, drift_path)
    _log(log_path, f"healthy q95={threshold:.4f}; held-out alarm unchanged={base_alarm:.3f}, shifted={shifted_alarm:.3f}; n={len(shifted_distances)}")

    _log(log_path, "=== 6/6: save PCA/KDE region map and machine-readable E1 result ===")
    pca_variance = _plot_region_map(embeddings, all_y, train_ids, prototypes, region_path)
    result = {
        "experiment": "maintenance E1 real C-MAPSS FD001 explainability and drift",
        "data": "cmapss_fd001",
        "sources": loaded["sources"],
        "split": {"train_windows": len(train_scaled), "validation_windows": len(val_scaled),
                  "official_test_windows": len(test_scaled), "train_grouping": "engine-disjoint, E0b 80/20 split"},
        "encoder": {"pipeline": "E0b train-only MiniBatchKMeans tokens + hybrid MaintenanceCloud",
                    "embedding": "CloudEncoderND output", "embedding_dim": int(embeddings.shape[1]),
                    "prototype": "mean training embedding by RUL stage", "distance": "Euclidean"},
        "heldout_test": {"n": int(len(test_y)), "nearest_prototype_accuracy": nearest_accuracy,
                         "stage_distance_summary": distance_summary, "explanation_examples": explanations},
        "drift_demo": {
            "reference_population": "training windows with RUL > 90 cycles",
            "distance": "Euclidean distance to healthy_gt_90 training centroid",
            "alarm_threshold": {"quantile": 0.95, "distance": threshold},
            "heldout_healthy_n": int(len(healthy_test_local)),
            "unchanged_alarm_rate": base_alarm,
            "shifted_alarm_rate": shifted_alarm,
            "shifted_to_unchanged_alarm_rate_ratio": shifted_alarm / max(base_alarm, 1e-12),
            "alarm_counts": {"unchanged": int(np.sum(unshifted_distances > threshold)),
                             "shifted": int(np.sum(shifted_distances > threshold))},
            "changed_token_fraction": changed_token_fraction,
            "applied_standardized_shift": {"dimensions": OPERATION_DIMS.tolist(), "values": operation_shift.tolist(),
                                           "meaning": "controlled operating-setting shift on raw setting columns 1-3"},
            "feature_mapping_note": "The loader keeps the three operating settings at dimensions 0-2; dimensions 33-35 are engineered baseline deltas, not fan/load settings.",
        },
        "region_map": {"path": str(region_path), "pca_explained_variance_percent": pca_variance,
                       "kde_population": "training windows by stage; contours are descriptive, not calibrated boundaries"},
        "drift_figure": str(drift_path),
        "limitations": [
            "Nearest-centroid results are retrieval metrics, not classifier accuracy or calibrated confidence.",
            "Feature-based reason sentences compare normalized sensor RMS/crest statistics with the training reference for the nearest-prototype stage; they are review aids, not causal diagnoses.",
            "FD001 is a single operating-condition subset; the standardized setting shift is a controlled stress test, not an observed field drift event.",
            "KDE/PCA regions are descriptive projections and do not define decision boundaries.",
        ],
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _log(log_path, f"saved {out_path}; region={region_path}; drift={drift_path}")
    return result


if __name__ == "__main__":
    run_e1()
