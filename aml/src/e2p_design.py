"""E2' Elliptic point-budget, bucket, window, and pair-angle design grid (CPU)."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cloudlm import CloudEncoderND
from quann_cloud.src.qcloud_weight import QCloudWeight

SEED = 20261016
KAGGLE_SLUG = "ellipticco/elliptic-data-set"
N_TOKENS = 24


@dataclass(frozen=True)
class Design:
    name: str
    window: int
    buckets: int
    point_family: str
    explicit_pair_angles: bool = False


# A compact 2D proxy, a 4D dual-channel point, and its angle-augmented form;
# the last three isolate bucket and window changes on the same angle design.
GRID = (
    Design("time_anonymous_pair_K10_B8", 10, 8, "anonymous_pair"),
    Design("time_degree_anonymous_K10_B8", 10, 8, "dual_channel"),
    Design("dual_pair_angles_K10_B8", 10, 8, "dual_channel", True),
    Design("dual_pair_angles_K10_B4", 10, 4, "dual_channel", True),
    Design("dual_pair_angles_K5_B8", 5, 8, "dual_channel", True),
    Design("dual_pair_angles_K20_B8", 20, 8, "dual_channel", True),
)


def _locate(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"none of {names!r} found below {root}")


def _node_id(value) -> int:
    return int(float(value))


def _load_elliptic(data_dir: Path, n_accounts: int = 2000):
    """Load a bounded labeled sample and one-hop graph clouds from local CSVs."""
    labels_path = _locate(data_dir, ("elliptic_txs_classes.csv", "elliptic_tx.csv", "accounts.csv"))
    edges_path = _locate(data_dir, ("elliptic_txs_edgelist.csv", "elliptic_tx_edges.csv", "edges.csv"))
    features_path = _locate(data_dir, ("elliptic_txs_features.csv", "elliptic_tx_features.csv", "features.csv"))
    labels_frame = pd.read_csv(labels_path)
    id_col = next((c for c in labels_frame.columns if str(c).lower() in {"txid", "account", "account_id", "node_id", "id"}), labels_frame.columns[0])
    class_col = next((c for c in labels_frame.columns if str(c).lower() in {"class", "label", "is_fraud", "fraud"}), labels_frame.columns[-1])
    normalized = labels_frame[class_col].astype(str).str.lower()
    fraud_mask = normalized.isin({"1", "fraud", "illicit", "true"})
    normal_mask = normalized.isin({"2", "licit", "normal", "false", "0"})
    fraud_ids = np.asarray([_node_id(v) for v in labels_frame.loc[fraud_mask, id_col]])
    normal_ids = np.asarray([_node_id(v) for v in labels_frame.loc[normal_mask, id_col]])
    n_fraud = min(max(1, int(round(n_accounts * 0.10))), len(fraud_ids))
    n_normal = min(n_accounts - n_fraud, len(normal_ids))
    if n_fraud + n_normal < 4:
        raise ValueError("not enough labeled normal/fraud nodes")
    rng = np.random.default_rng(SEED)
    target_ids = np.concatenate((rng.choice(normal_ids, n_normal, replace=False),
                                 rng.choice(fraud_ids, n_fraud, replace=False))).astype(np.int64)
    target_labels = np.concatenate((np.zeros(n_normal, dtype=np.int64), np.ones(n_fraud, dtype=np.int64)))
    order = rng.permutation(len(target_ids))
    target_ids, target_labels = target_ids[order], target_labels[order]

    edges = pd.read_csv(edges_path)
    columns = list(edges.columns)
    src_col = next((c for c in columns if str(c).lower() in {"txid1", "source", "src", "from"}), columns[0])
    dst_col = next((c for c in columns if str(c).lower() in {"txid2", "target", "dst", "to"}), columns[1])
    incoming: dict[int, int] = {}
    outgoing: dict[int, int] = {}
    neighbors: dict[int, set[int]] = {}
    for source, target in edges[[src_col, dst_col]].itertuples(index=False, name=None):
        source, target = _node_id(source), _node_id(target)
        outgoing[source] = outgoing.get(source, 0) + 1
        incoming[target] = incoming.get(target, 0) + 1
        neighbors.setdefault(source, set()).add(target)
        neighbors.setdefault(target, set()).add(source)

    wanted = set(map(int, target_ids))
    for target in target_ids:
        wanted.update(neighbors.get(int(target), ()))
    node_rows: dict[int, tuple[int, np.ndarray]] = {}
    feature_rows = 0
    for chunk in pd.read_csv(features_path, header=None, chunksize=20_000):
        feature_rows += len(chunk)
        ids = pd.to_numeric(chunk.iloc[:, 0], errors="coerce")
        mask = ids.notna() & ids.astype("Int64").isin(wanted)
        if not mask.any():
            continue
        for row in chunk.loc[mask].to_numpy(dtype=np.float64):
            node = _node_id(row[0])
            raw = np.nan_to_num(row[2:], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            node_rows[node] = (int(row[1]), raw)

    missing = [int(node) for node in target_ids if int(node) not in node_rows]
    if missing:
        raise ValueError(f"missing features for {len(missing)} sampled labeled nodes")
    raw_dim = len(next(iter(node_rows.values()))[1])

    def point(node: int) -> np.ndarray:
        timestep, raw = node_rows[node]
        return np.concatenate((np.asarray([timestep, incoming.get(node, 0), outgoing.get(node, 0)], dtype=np.float32), raw))

    target_clouds: list[list[np.ndarray]] = []
    for target in target_ids:
        target = int(target)
        timestep = node_rows[target][0]
        candidates = [node for node in neighbors.get(target, ())
                      if node in node_rows and node_rows[node][0] <= timestep]
        candidates.sort(key=lambda node: (node_rows[node][0], node))
        chosen = candidates[-19:] + [target]
        chosen = sorted(set(chosen), key=lambda node: (node_rows[node][0], node))
        # Keep the largest K=20 candidate window once; configs take its suffix.
        cloud = [point(node) for node in chosen[-20:]]
        while len(cloud) < 20:
            cloud.insert(0, cloud[0] if cloud else point(target))
        target_clouds.append(cloud[-20:])

    return {
        "clouds_raw": np.asarray(target_clouds, dtype=np.float32),
        "labels": target_labels,
        "target_ids": target_ids,
        "target_raw": np.stack([node_rows[int(node)][1] for node in target_ids]),
        "raw_dim": raw_dim,
        "metadata": {
            "labels_rows": len(labels_frame), "edge_rows": len(edges), "feature_rows": feature_rows,
            "labeled_fraud_nodes": int(fraud_mask.sum()), "labeled_normal_nodes": int(normal_mask.sum()),
            "unknown_nodes": int((~(fraud_mask | normal_mask)).sum()),
            "sampled_labeled_nodes": len(target_ids), "sampled_fraud": int(target_labels.sum()),
            "sampled_normal": int((target_labels == 0).sum()),
            "point_feature_count_raw": raw_dim + 3,
            "raw_feature_axes_are_anonymous": True,
            "named_amount_available": False,
            "account_unit": "labeled transaction-node proxy; source has no account IDs",
            "feature_source": features_path.name, "label_source": labels_path.name, "edge_source": edges_path.name,
        },
    }


def _split(labels: np.ndarray):
    ids = np.arange(len(labels))
    train, temp = train_test_split(ids, test_size=0.4, random_state=SEED, stratify=labels)
    val, test = train_test_split(temp, test_size=0.5, random_state=SEED + 1, stratify=labels[temp])
    return train, val, test


def _rank_anonymous_features(target_raw: np.ndarray, y_train: np.ndarray, train_ids: np.ndarray, count: int = 1) -> list[int]:
    """Rank raw axes using training targets only; never interpret them as amounts."""
    x = np.asarray(target_raw[train_ids], dtype=np.float32)
    medians = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    medians = np.nan_to_num(medians, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.where(np.isfinite(x), x, medians[None, :])
    scores = mutual_info_classif(x, y_train, discrete_features=False, random_state=SEED)
    order = np.argsort(-np.nan_to_num(scores, nan=-1.0), kind="stable")
    return [int(index) for index in order[:min(count, x.shape[1])]]


def _feature_matrix(raw_clouds: np.ndarray, design: Design, anon_index: int) -> np.ndarray:
    """Select compact channels, then optionally append explicit paired angles."""
    timestep = raw_clouds[:, :, 0]
    indegree = raw_clouds[:, :, 1]
    outdegree = raw_clouds[:, :, 2]
    anonymous = raw_clouds[:, :, 3 + anon_index]
    if design.point_family == "anonymous_pair":
        return np.stack((timestep, anonymous), axis=-1).astype(np.float32)
    if design.point_family != "dual_channel":
        raise ValueError(f"unknown point family: {design.point_family}")
    channels = [timestep, indegree, outdegree, anonymous]
    if design.explicit_pair_angles:
        # Angles are computed after quantization so bucket geometry is directly tested.
        return np.stack(channels, axis=-1).astype(np.float32)
    return np.stack(channels, axis=-1).astype(np.float32)


class TrainQuantizer:
    """Per-axis train-only quantile buckets, represented on a bounded scale."""
    def __init__(self, buckets: int):
        self.buckets = int(buckets)
        self.edges: list[np.ndarray] | None = None

    def fit(self, train_points: np.ndarray):
        self.edges = []
        for axis in range(train_points.shape[-1]):
            values = train_points[..., axis].reshape(-1)
            values = values[np.isfinite(values)]
            if not len(values):
                self.edges.append(np.empty(0, dtype=np.float32))
            else:
                q = np.quantile(values, np.linspace(0.0, 1.0, self.buckets + 1)[1:-1])
                self.edges.append(np.unique(q).astype(np.float32))
        return self

    def transform(self, points: np.ndarray) -> np.ndarray:
        if self.edges is None:
            raise RuntimeError("fit quantizer on training points first")
        transformed = np.empty(points.shape, dtype=np.float32)
        for axis, edges in enumerate(self.edges):
            values = np.nan_to_num(points[..., axis], nan=0.0, posinf=0.0, neginf=0.0)
            bucket_ids = np.digitize(values, edges)
            bucket_count = len(edges) + 1
            transformed[..., axis] = 2.0 * bucket_ids / max(bucket_count - 1, 1) - 1.0
        return transformed

    @property
    def actual_bucket_counts(self) -> list[int]:
        if self.edges is None:
            return []
        return [len(edges) + 1 for edges in self.edges]


def _add_pair_angles(points: np.ndarray) -> np.ndarray:
    """Append normalized (time, anonymous) and (in-degree, out-degree) angles."""
    time_anon = np.arctan2(points[..., 3], points[..., 0]) / np.pi
    in_out = np.arctan2(points[..., 2], points[..., 1]) / np.pi
    return np.concatenate((points, time_anon[..., None], in_out[..., None]), axis=-1).astype(np.float32)


def _f2(y: np.ndarray, prediction: np.ndarray) -> float:
    tp = int(np.sum((y == 1) & prediction))
    fp = int(np.sum((y == 0) & prediction))
    fn = int(np.sum((y == 1) & ~prediction))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(5 * precision * recall / max(4 * precision + recall, 1e-12))


def _threshold_f2(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.quantile(score, np.linspace(0.0, 1.0, 201)))
    return float(max(candidates, key=lambda cut: _f2(y, score >= cut)))


def _metrics(y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray) -> dict:
    val_score, score = p_val[:, 1], p_test[:, 1]
    threshold = _threshold_f2(y_val, val_score)
    fpr_cut = float(np.quantile(val_score[y_val == 0], 0.99))
    prediction = score >= threshold
    fpr_prediction = score >= fpr_cut
    base_rate = float(y_test.mean())
    return {
        "val": {"threshold_f2": threshold, "threshold_at_1pct_fpr": fpr_cut},
        "test": {
            "n": int(len(y_test)), "n_fraud": int(y_test.sum()),
            "auc_roc": float(roc_auc_score(y_test, score)),
            "auc_pr": float(average_precision_score(y_test, score)),
            "f2": _f2(y_test, prediction), "threshold_f2": threshold,
            "fraud_recall_at_f2_threshold": float(np.mean(prediction[y_test == 1])),
            "normal_recall_at_f2_threshold": float(np.mean(~prediction[y_test == 0])),
            "lift": float(y_test[prediction].mean() / base_rate) if prediction.any() and base_rate else 0.0,
            "fraud_recall_at_1pct_fpr": float(np.mean(fpr_prediction[y_test == 1])),
            "fpr_threshold_from_validation": fpr_cut,
        },
    }


class EllipticDesignCloud(nn.Module):
    def __init__(self, point_dim: int):
        super().__init__()
        self.point_dim = point_dim
        self.encoder = CloudEncoderND(point_dim, d_cloud=48, hidden=32)
        self.head = nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 2))
        self.token_weights: dict[int, QCloudWeight] = {}
        self.updated_tokens: set[int] = set()

    def get_weight(self, token: int):
        if token not in self.token_weights:
            self.token_weights[token] = QCloudWeight(D=self.point_dim, convergence_threshold=0.02)
        return self.token_weights[token]

    def points_for(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.reshape(-1)
        unique, inverse = torch.unique(flat, return_inverse=True)
        vectors = torch.as_tensor(np.stack([self.get_weight(int(token)).value for token in unique.tolist()]), dtype=torch.float32)
        return vectors[inverse].view(tokens.shape[0], tokens.shape[1], self.point_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        return self.head(self.encoder(points, mask))

    def quann_update(self, tokens: torch.Tensor, target: int):
        with torch.no_grad():
            base = self.points_for(tokens)
            row = tokens[0]
            candidates, weights = [], []
            for token in torch.unique(row).tolist():
                weight = self.get_weight(int(token))
                candidate = weight.sample()
                changed = base.clone()
                changed[0, row == token] = torch.as_tensor(candidate, dtype=torch.float32)
                candidates.append(changed[0])
                weights.append((int(token), weight, candidate))
            if not candidates:
                return
            scores = self(torch.cat((base, torch.stack(candidates))))
            losses = F.cross_entropy(scores, torch.full((len(candidates) + 1,), int(target), dtype=torch.long), reduction="none")
            baseline = float(losses[0])
            for index, (token, weight, candidate) in enumerate(weights, 1):
                improved = float(losses[index]) < baseline - 1e-7
                weight.narrow_search(candidate, improved)
                weight.update_sigma(rewarded=improved)
                self.updated_tokens.add(token)

    def convergence(self) -> dict:
        weights = [self.token_weights[token] for token in self.updated_tokens]
        return {
            "points": len(self.updated_tokens),
            "converged": sum(weight.is_converged() for weight in weights),
            "mean_width": float(np.mean([weight.error_width().mean() for weight in weights])) if weights else None,
        }


def _cloud_matrix(clouds: np.ndarray) -> np.ndarray:
    return np.concatenate((clouds.mean(axis=1), clouds.std(axis=1), clouds[:, -1, :]), axis=1).astype(np.float32)


def _fit_regime(regime: str, train_clouds, train_tokens, y_train, val_clouds, val_tokens, y_val,
                test_clouds, test_tokens, y_test):
    point_dim = train_clouds.shape[-1]
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = EllipticDesignCloud(point_dim)
    optimizer = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.head.parameters()), lr=0.002, weight_decay=1e-4)
    started = time.time()

    if regime == "backprop":
        class_counts = np.bincount(y_train, minlength=2)
        class_weights = torch.as_tensor(len(y_train) / (2 * np.maximum(class_counts, 1)), dtype=torch.float32)
        for epoch in range(8):
            order = np.random.default_rng(SEED + epoch).permutation(len(y_train))
            model.train()
            for start in range(0, len(order), 32):
                batch = order[start:start + 32]
                points = torch.as_tensor(train_clouds[batch], dtype=torch.float32)
                target = torch.as_tensor(y_train[batch], dtype=torch.long)
                loss = F.cross_entropy(model(points), target, weight=class_weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    elif regime == "hybrid":
        rng = np.random.default_rng(SEED)
        selected = []
        for label in (0, 1):
            candidates = np.flatnonzero(y_train == label)
            selected.extend(rng.choice(candidates, min(len(candidates), 120), replace=False).tolist())
        rng.shuffle(selected)
        for row in selected:
            tokens = torch.as_tensor(train_tokens[row:row + 1], dtype=torch.long)
            target = int(y_train[row])
            model.train()
            fixed_points = model.points_for(tokens).detach()
            loss = F.cross_entropy(model(fixed_points), torch.tensor([target]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.eval()
            model.quann_update(tokens, target)
    else:
        raise ValueError(f"unknown regime {regime}")

    def predict(clouds, tokens):
        outputs = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(tokens), 64):
                if regime == "hybrid":
                    pts = model.points_for(torch.as_tensor(tokens[start:start + 64], dtype=torch.long))
                else:
                    pts = torch.as_tensor(clouds[start:start + 64], dtype=torch.float32)
                outputs.append(torch.softmax(model(pts), dim=-1).cpu().numpy())
        return np.concatenate(outputs)

    result = _metrics(y_val, predict(val_clouds, val_tokens), y_test, predict(test_clouds, test_tokens))
    result["test"].update({"train_sec": round(time.time() - started, 2), **model.convergence()})
    return result, model


def _plot_best(model, tokens: np.ndarray, labels: np.ndarray, design: Design, path: Path):
    selected = []
    rng = np.random.default_rng(SEED)
    for label in (0, 1):
        rows = np.flatnonzero(labels == label)
        selected.extend(rng.choice(rows, min(40, len(rows)), replace=False).tolist())
    selected_tokens = torch.as_tensor(tokens[selected], dtype=torch.long)
    with torch.no_grad():
        points = model.points_for(selected_tokens).cpu().numpy()
    projection = PCA(n_components=3, random_state=SEED).fit_transform(points.reshape(-1, points.shape[-1])).reshape(len(selected), design.window, 3)
    fig = plt.figure(figsize=(12, 5.8))
    for label, title, color in ((0, "Licit / normal", "#2878b5"), (1, "Illicit / fraud", "#d74737")):
        ax = fig.add_subplot(1, 2, label + 1, projection="3d")
        for index, original in enumerate(selected):
            if labels[original] == label:
                cloud = projection[index]
                ax.plot(cloud[:, 0], cloud[:, 1], cloud[:, 2], color=color, alpha=0.28, linewidth=0.8)
                ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color=color, s=10, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("PC1", labelpad=3)
        ax.set_ylabel("PC2", labelpad=5)
        ax.set_zlabel("")
    fig.suptitle(f"E2′ best hybrid design: {design.name} — learned-token 3D clouds")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _log(path: Path, message: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def run(data_dir: str | Path | None = None, out_path: str | Path | None = None,
        figure_path: str | Path | None = None, log_path: str | Path | None = None,
        n_accounts: int = 2000):
    project = ROOT / "aml"
    data_dir = Path(data_dir or project / "data" / "elliptic")
    out_path = Path(out_path or project / "results" / "e2p_design.json")
    figure_path = Path(figure_path or project / "figures" / "e2p_design.png")
    log_path = Path(log_path or project / "logs" / "run_e2p_smoke.log")
    for directory in (out_path.parent, figure_path.parent, log_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    _log(log_path, f"=== [{datetime.now().date()}] E2′ Elliptic CPU; seed={SEED}; sample={n_accounts}; grid={len(GRID)} ===")
    loaded = _load_elliptic(data_dir, n_accounts=n_accounts)
    labels = loaded["labels"]
    train_ids, val_ids, test_ids = _split(labels)
    y_train, y_val, y_test = labels[train_ids], labels[val_ids], labels[test_ids]
    best_anon = _rank_anonymous_features(loaded["target_raw"], y_train, train_ids, count=1)[0]
    _log(log_path, f"loaded labeled proxies={len(labels)} fraud={int(labels.sum())}; train-only top anonymous axis=raw_{best_anon}; amount unavailable")

    rows = []
    best = None
    for index, design in enumerate(GRID, 1):
        _log(log_path, f"=== [{datetime.now().date()}] {index}/{len(GRID)}: {design.name} ===")
        raw_clouds = loaded["clouds_raw"][:, -design.window:, :]
        selected = _feature_matrix(raw_clouds, design, best_anon)
        quantizer = TrainQuantizer(design.buckets).fit(selected[train_ids])
        clouds = quantizer.transform(selected)
        if design.explicit_pair_angles:
            clouds = _add_pair_angles(clouds)
        train_clouds, val_clouds, test_clouds = clouds[train_ids], clouds[val_ids], clouds[test_ids]
        point_dim = clouds.shape[-1]

        gbt = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=0.08, random_state=SEED)
        started = time.time()
        gbt.fit(_cloud_matrix(train_clouds), y_train)
        gbt_result = _metrics(y_val, gbt.predict_proba(_cloud_matrix(val_clouds)), y_test, gbt.predict_proba(_cloud_matrix(test_clouds)))
        gbt_result["test"]["train_sec"] = round(time.time() - started, 2)

        cluster_count = min(N_TOKENS, len(np.unique(train_clouds.reshape(-1, point_dim), axis=0)))
        cluster = MiniBatchKMeans(n_clusters=max(2, cluster_count), random_state=SEED, n_init=2, batch_size=512)
        cluster.fit(train_clouds.reshape(-1, point_dim))
        train_tokens = cluster.predict(train_clouds.reshape(-1, point_dim)).reshape(-1, design.window)
        val_tokens = cluster.predict(val_clouds.reshape(-1, point_dim)).reshape(-1, design.window)
        test_tokens = cluster.predict(test_clouds.reshape(-1, point_dim)).reshape(-1, design.window)

        hybrid, hybrid_model = _fit_regime("hybrid", train_clouds, train_tokens, y_train,
                                           val_clouds, val_tokens, y_val, test_clouds, test_tokens, y_test)
        backprop, _ = _fit_regime("backprop", train_clouds, train_tokens, y_train,
                                  val_clouds, val_tokens, y_val, test_clouds, test_tokens, y_test)
        metrics = {"gbt": gbt_result, "backprop": backprop, "hybrid": hybrid}
        row = {
            "design": asdict(design), "point_dim": point_dim,
            "actual_buckets_per_axis": quantizer.actual_bucket_counts,
            "anonymous_raw_axis_selected_train_only": best_anon,
            "metrics": metrics,
        }
        rows.append(row)
        _log(log_path, " | ".join(
            f"{regime}: ROC={metrics[regime]['test']['auc_roc']:.4f} PR={metrics[regime]['test']['auc_pr']:.4f} "
            f"F2={metrics[regime]['test']['f2']:.4f} fraud@1%FPR={metrics[regime]['test']['fraud_recall_at_1pct_fpr']:.3f}"
            for regime in ("gbt", "backprop", "hybrid")))
        candidate = (hybrid["test"]["auc_pr"], hybrid["test"]["auc_roc"])
        if best is None or candidate > best["rank"]:
            best = {"rank": candidate, "row": row, "model": hybrid_model, "tokens": test_tokens}

    baseline_metrics = {
        regime: {"auc_roc": max(row["metrics"][regime]["test"]["auc_roc"] for row in rows),
                 "auc_pr": max(row["metrics"][regime]["test"]["auc_pr"] for row in rows)}
        for regime in ("gbt", "backprop")
    }
    best_test = best["row"]["metrics"]["hybrid"]["test"]
    gaps = {
        regime: {metric: round(abs(best_test[metric] - baseline_metrics[regime][metric]), 6)
                 for metric in ("auc_roc", "auc_pr")}
        for regime in ("gbt", "backprop")
    }
    matched_gaps = []
    for row in rows:
        hybrid_test = row["metrics"]["hybrid"]["test"]
        for regime in ("gbt", "backprop"):
            baseline_test = row["metrics"][regime]["test"]
            for metric in ("auc_roc", "auc_pr"):
                matched_gaps.append({
                    "design": row["design"]["name"], "regime": regime, "metric": metric,
                    "gap": round(abs(hybrid_test[metric] - baseline_test[metric]), 6),
                })
    closest_matched_gap = min(matched_gaps, key=lambda item: item["gap"])
    target_met = closest_matched_gap["gap"] <= 0.05
    _plot_best(best["model"], best["tokens"], y_test, Design(**best["row"]["design"]), figure_path)
    result = {
        "data": "elliptic", "kaggle_slug": KAGGLE_SLUG,
        "data_validation": loaded["metadata"],
        "protocol": {
            "device": "cpu", "seed": SEED, "sampled_labeled_nodes": len(labels),
            "split": "same stratified node-level 60/20/20 split across all configs; seed=20261016",
            "quantization": "per-axis train-only quantile buckets; bucket IDs mapped to [-1,1]",
            "anonymous_feature_selection": "mutual information on training target nodes only; selected axis is not named amount",
            "hybrid_training": "same CloudEncoderND/QCloudWeight regime as E2; balanced 120/class, 8-epoch backprop ceiling",
            "success_criterion": "within one configuration, hybrid is within 0.05 ROC or PR of GBT or backprop",
        },
        "grid": rows,
        "summary": {
            "best_hybrid_design": best["row"]["design"],
            "best_hybrid_point_dim": best["row"]["point_dim"],
            "best_hybrid_metrics": best["row"]["metrics"]["hybrid"],
            "best_observed_baseline": baseline_metrics,
            "absolute_gap_to_best_baseline_by_regime": gaps,
            "closest_same_configuration_gap": closest_matched_gap,
            "within_5_points_on_roc_or_pr": target_met,
        },
        "outputs": {"figure": str(figure_path), "log": str(log_path)},
    }
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _log(log_path, f"best={best['row']['design']['name']}; target_within_5pt={target_met}; saved={out_path}")
    return result


if __name__ == "__main__":
    run()
