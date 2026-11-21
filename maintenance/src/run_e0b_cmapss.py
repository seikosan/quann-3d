"""Deterministic CPU-only E0b evaluation on real NASA C-MAPSS FD001."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

import config
from loader_cmapss import FEATURE_DIM, load_cmapss
from train_maintenance import MaintenanceCloud, _binary_metrics, _stage_metrics


SEED = 20261016
WINDOW = 10
STRIDE = 12
KAGGLE_SLUG = "behrad3d/nasa-cmaps"


def _log(path: Path, message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _windows(data: dict, engine_ids: set[int] | None = None):
    clouds, rul, groups = [], [], []
    for engine in np.unique(data["engine_ids"]):
        if engine_ids is not None and int(engine) not in engine_ids:
            continue
        rows = np.flatnonzero(data["engine_ids"] == engine)
        rows = rows[np.argsort(data["cycles"][rows])]
        for end in range(WINDOW - 1, len(rows), STRIDE):
            selected = rows[end - WINDOW + 1:end + 1]
            clouds.append(data["features"][selected])
            rul.append(data["rul"][rows[end]])
            groups.append(int(engine))
    if not clouds:
        raise ValueError("no complete C-MAPSS windows")
    return np.asarray(clouds, np.float32), np.asarray(rul, np.int64), np.asarray(groups, np.int64)


def _scale(train, *others):
    mean = train.reshape(-1, FEATURE_DIM).mean(axis=0)
    scale = train.reshape(-1, FEATURE_DIM).std(axis=0)
    scale[scale < 1e-6] = 1.0
    return tuple(((x - mean) / scale).astype(np.float32) for x in (train, *others))


def _labels(rul: np.ndarray, task: str) -> np.ndarray:
    return (rul <= 30).astype(np.int64) if task == "binary" else np.digitize(rul, [30, 60, 90]).astype(np.int64)


def _score(task, y_val, p_val, y_test, p_test):
    if task == "binary":
        val_metrics = _binary_metrics(y_val, p_val)
        test_metrics = _binary_metrics(y_test, p_test, val_metrics["threshold"])
        prediction = p_test[:, 1] >= val_metrics["threshold"]
        test_metrics["class_recall"] = {
            "rul_le_30": float(np.mean(prediction[y_test == 1])) if np.any(y_test == 1) else None,
            "rul_gt_30": float(np.mean(~prediction[y_test == 0])) if np.any(y_test == 0) else None,
        }
        return val_metrics, test_metrics
    val_metrics = _stage_metrics(y_val, p_val)
    test_metrics = _stage_metrics(y_test, p_test, val_metrics["class_thresholds_at_1pct_fpr"], val_metrics["critical_threshold"])
    return val_metrics, test_metrics


def _standard(task, x_train, y_train, x_val, y_val, x_test, y_test, log_path):
    models = {
        "lr": LogisticRegression(max_iter=600, class_weight="balanced", random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=60, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=SEED),
        "gbt": HistGradientBoostingClassifier(max_iter=70, max_leaf_nodes=15, learning_rate=0.08, random_state=SEED),
        "mlp": MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=60, early_stopping=True, random_state=SEED),
    }
    results = {}
    for name, model in models.items():
        started = time.time()
        model.fit(x_train, y_train)
        val_prob, test_prob = model.predict_proba(x_val), model.predict_proba(x_test)
        vm, tm = _score(task, y_val, val_prob, y_test, test_prob)
        tm["fit_sec"] = round(time.time() - started, 2)
        results[name] = {"val": vm, "test": tm}
        _log(log_path, f"standard/{task}/{name}: roc={tm.get('auc_roc', tm.get('auc_roc_macro')):.4f}")
    return results


def _fit_cloud(regime, train_clouds, train_tokens, y_train, val_clouds, val_tokens, y_val,
               test_clouds, test_tokens, y_test, n_classes, task, log_path):
    torch.manual_seed(SEED)
    model = MaintenanceCloud(vocab_size=int(train_tokens.max()) + 1, n_classes=n_classes, D=FEATURE_DIM)
    optimizer = None
    if regime == "hybrid":
        optimizer = torch.optim.AdamW(list(model.cloud_encoder.parameters()) + list(model.class_head.parameters()), lr=0.002, weight_decay=1e-4)
    elif regime == "backprop":
        optimizer = torch.optim.AdamW(list(model.cloud_encoder.parameters()) + list(model.class_head.parameters()), lr=0.002, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    selected = []
    for label in np.unique(y_train):
        candidates = np.flatnonzero(y_train == label)
        selected.extend(rng.choice(candidates, min(len(candidates), 36), replace=False).tolist())
    rng.shuffle(selected)
    started = time.time()
    torch.set_num_threads(1)
    for index in selected:
        tokens = torch.as_tensor(train_tokens[index:index + 1], dtype=torch.long)
        target = int(y_train[index])
        if optimizer is not None:
            model.train()
            points = (model.points_for(tokens).detach() if regime == "hybrid" else
                      torch.as_tensor(train_clouds[index:index + 1], dtype=torch.float32))
            loss = torch.nn.functional.cross_entropy(model(points), torch.tensor([target]))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if regime in {"pure", "hybrid"}:
            model.eval(); model.quann_update(tokens, target)

    def predict(clouds, tokens):
        output = []; model.eval()
        with torch.no_grad():
            for start in range(0, len(tokens), 64):
                tt = torch.as_tensor(tokens[start:start + 64], dtype=torch.long)
                points = model.points_for(tt) if regime != "backprop" else torch.as_tensor(clouds[start:start + 64], dtype=torch.float32)
                output.append(torch.softmax(model(points), dim=-1).cpu().numpy())
        return np.concatenate(output)

    val_prob, test_prob = predict(val_clouds, val_tokens), predict(test_clouds, test_tokens)
    vm, tm = _score(task, y_val, val_prob, y_test, test_prob)
    tm.update({"train_sec": round(time.time() - started, 2), **model.convergence_info()})
    _log(log_path, f"cloud/{task}/{regime}: roc={tm.get('auc_roc', tm.get('auc_roc_macro')):.4f}, points={tm['points']}, converged={tm['converged']}")
    return {"val": vm, "test": tm}, model


def _plot(model, test_tokens, test_y, path: Path):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(SEED)
    selected = []
    for label in range(4):
        rows = np.flatnonzero(test_y == label)
        if len(rows):
            selected.extend(rng.choice(rows, min(18, len(rows)), replace=False).tolist())
    tokens = torch.as_tensor(test_tokens[selected], dtype=torch.long)
    with torch.no_grad():
        points = model.points_for(tokens).cpu().numpy()
    projected = PCA(n_components=3, random_state=SEED).fit_transform(points.reshape(-1, FEATURE_DIM)).reshape(len(selected), WINDOW, 3)
    fig = plt.figure(figsize=(11, 8)); colors = ["#d62728", "#ff7f0e", "#3182bd", "#31a354"]
    for label, name in enumerate(config.STAGES):
        ax = fig.add_subplot(2, 2, label + 1, projection="3d")
        for idx, original in enumerate(selected):
            if test_y[original] == label:
                cloud = projected[idx]
                ax.plot(cloud[:, 0], cloud[:, 1], cloud[:, 2], color=colors[label], alpha=0.4, linewidth=0.8)
                ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color=colors[label], s=8)
        ax.set_title(name); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    fig.suptitle("NASA C-MAPSS FD001 stage clouds — 10 points × 36D")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def run(data_dir: str | Path | None = None, out_path: str | Path | None = None,
        figure_path: str | Path | None = None, log_path: str | Path | None = None):
    np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
    project = ROOT / "maintenance"
    data_dir = Path(data_dir or project / "data" / "cmapss")
    out_path = Path(out_path or project / "results" / "e0b_cmapss.json")
    figure_path = Path(figure_path or project / "figures" / "e0b_stage_clouds.png")
    log_path = Path(log_path or project / "logs" / "run_e0b_cmapss.log")
    for directory in (out_path.parent, figure_path.parent, log_path.parent): directory.mkdir(parents=True, exist_ok=True)
    _log(log_path, f"=== {datetime.now().date()} E0b C-MAPSS FD001; CPU; seed={SEED} ===")
    loaded = load_cmapss(data_dir)
    train_data, official_test = loaded["train"], loaded["test"]
    engines = np.unique(train_data["engine_ids"])
    rng = np.random.default_rng(SEED); rng.shuffle(engines)
    cut = max(1, int(len(engines) * 0.8))
    train_engines, val_engines = set(map(int, engines[:cut])), set(map(int, engines[cut:]))
    train_cloud, train_rul, train_groups = _windows(train_data, train_engines)
    val_cloud, val_rul, val_groups = _windows(train_data, val_engines)
    test_cloud, test_rul, test_groups = _windows(official_test)
    if sum(map(len, (train_cloud, val_cloud, test_cloud))) > 3000:
        raise ValueError("smoke window cap exceeded")
    train_cloud, val_cloud, test_cloud = _scale(train_cloud, val_cloud, test_cloud)
    flat_train = train_cloud.reshape(len(train_cloud), -1)
    flat_val, flat_test = val_cloud.reshape(len(val_cloud), -1), test_cloud.reshape(len(test_cloud), -1)
    points_train = train_cloud.reshape(-1, FEATURE_DIM)
    cluster = MiniBatchKMeans(n_clusters=24, random_state=SEED, n_init=2, batch_size=512)
    cluster.fit(points_train)
    train_tokens = cluster.predict(points_train).reshape(-1, WINDOW)
    val_tokens = cluster.predict(val_cloud.reshape(-1, FEATURE_DIM)).reshape(-1, WINDOW)
    test_tokens = cluster.predict(test_cloud.reshape(-1, FEATURE_DIM)).reshape(-1, WINDOW)
    results = {"data": "cmapss_fd001", "kaggle_slug": KAGGLE_SLUG,
               "data_validation": {"train_raw_rows": train_data["raw_rows"], "test_raw_rows": official_test["raw_rows"],
                                   "raw_columns": train_data["raw_columns"], "feature_dim": FEATURE_DIM,
                                   "train_engines": train_data["engines"], "test_engines": official_test["engines"],
                                   "train_windows": len(train_cloud), "validation_windows": len(val_cloud), "test_windows": len(test_cloud),
                                   "sources": loaded["sources"]},
               "protocol": {"device": "cpu", "seed": SEED, "window": WINDOW, "stride": STRIDE,
                            "split": "engine-group holdout from official train; official FD001 test set for final test"},
               "tasks": {}}
    best_stage_model = None
    for task in ("binary", "stage"):
        y_train, y_val, y_test = (_labels(r, task) for r in (train_rul, val_rul, test_rul))
        n_classes = 2 if task == "binary" else 4
        task_result = {"standard": _standard(task, flat_train, y_train, flat_val, y_val, flat_test, y_test, log_path), "cloud": {}}
        for regime in ("pure", "hybrid", "backprop"):
            fitted, model = _fit_cloud(regime, train_cloud, train_tokens, y_train, val_cloud, val_tokens, y_val,
                                       test_cloud, test_tokens, y_test, n_classes, task, log_path)
            task_result["cloud"][regime] = fitted
            if task == "stage" and regime == "hybrid": best_stage_model = model
        results["tasks"][task] = task_result
    _plot(best_stage_model, test_tokens, _labels(test_rul, "stage"), figure_path)
    results["outputs"] = {"figure": str(figure_path), "log": str(log_path)}
    out_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _log(log_path, f"saved {out_path}; windows={len(train_cloud)+len(val_cloud)+len(test_cloud)}")
    return results


if __name__ == "__main__":
    run()
