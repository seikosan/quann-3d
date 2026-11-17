"""CPU E2 Elliptic transaction-node/account-proxy cloud experiment."""
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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from cloudlm import CloudEncoderND
from quann_cloud.src.qcloud_weight import QCloudWeight


SEED = 20261016
WINDOW = 10
POINT_DIM = 36
KAGGLE_SLUG = "ellipticco/elliptic-data-set"
CHAIN_FEATURES = ("observed_hop_count", "hop_completion_ratio", "counterparty_shared_ratio",
                  "max_counterparty_degree", "counterparty_transition_rate", "dwell_time_cv")
UNAVAILABLE_CHAIN_FEATURES = ("amount_retention_median", "amount_retention_p10")


def _locate(root: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        matches = sorted(root.rglob(candidate))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"none of {candidates!r} found below {root}")


def _log(path: Path, message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _node_id(value) -> int:
    return int(float(value))


def _load_sample(root: Path, n_accounts: int = 2000):
    """Read the canonical Elliptic node/edge/features mirror and build K-node clouds.

    Elliptic has transaction nodes rather than account IDs. A labeled transaction is
    therefore the account proxy; its preceding one-hop transaction neighborhood is
    the K-transaction cloud. The raw feature set is anonymized and has no named amount.
    """
    labels_path = _locate(root, ("elliptic_txs_classes.csv", "elliptic_tx.csv", "accounts.csv"))
    edges_path = _locate(root, ("elliptic_txs_edgelist.csv", "elliptic_tx_edges.csv", "edges.csv"))
    feature_path = _locate(root, ("elliptic_txs_features.csv", "elliptic_tx_features.csv", "features.csv"))
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
    if n_normal + n_fraud < 4:
        raise ValueError("not enough labeled normal/fraud nodes")
    rng = np.random.default_rng(SEED)
    sampled_fraud = rng.choice(fraud_ids, n_fraud, replace=False)
    sampled_normal = rng.choice(normal_ids, n_normal, replace=False)
    target_ids = np.concatenate([sampled_normal, sampled_fraud]).astype(np.int64)
    labels = np.concatenate([np.zeros(n_normal, dtype=np.int64), np.ones(n_fraud, dtype=np.int64)])
    order = rng.permutation(len(target_ids)); target_ids, labels = target_ids[order], labels[order]

    edges = pd.read_csv(edges_path)
    edge_cols = list(edges.columns)
    src_col = next((c for c in edge_cols if str(c).lower() in {"txid1", "source", "src", "from"}), edge_cols[0])
    dst_col = next((c for c in edge_cols if str(c).lower() in {"txid2", "target", "dst", "to"}), edge_cols[1])
    incoming: dict[int, int] = {}; outgoing: dict[int, int] = {}; neighbors: dict[int, set[int]] = {}
    for source, target in edges[[src_col, dst_col]].itertuples(index=False, name=None):
        source, target = _node_id(source), _node_id(target)
        outgoing[source] = outgoing.get(source, 0) + 1
        incoming[target] = incoming.get(target, 0) + 1
        neighbors.setdefault(source, set()).add(target)
        neighbors.setdefault(target, set()).add(source)
    wanted = set(map(int, target_ids))
    for account in target_ids:
        wanted.update(neighbors.get(int(account), ()))

    node_rows: dict[int, tuple[int, np.ndarray]] = {}
    feature_rows = 0
    # The 689MB canonical feature CSV is streamed; only target nodes and their one-hop
    # neighbors are retained in memory for the bounded account smoke.
    for chunk in pd.read_csv(feature_path, header=None, chunksize=20_000):
        feature_rows += len(chunk)
        ids = pd.to_numeric(chunk.iloc[:, 0], errors="coerce")
        mask = ids.notna() & ids.astype("Int64").isin(wanted)
        if not mask.any():
            continue
        chosen = chunk.loc[mask].to_numpy(dtype=np.float64)
        for row in chosen:
            node = _node_id(row[0])
            node_rows[node] = (int(row[1]), np.nan_to_num(row[2:31], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    missing = [int(node) for node in target_ids if int(node) not in node_rows]
    if missing:
        raise ValueError(f"missing features for {len(missing)} sampled labeled nodes")

    def point(node: int) -> np.ndarray:
        timestep, raw = node_rows[node]
        adjacent = neighbors.get(node, set())
        adjacent_degrees = [len(neighbors.get(other, ())) for other in adjacent]
        ordered_times = sorted(node_rows[other][0] for other in adjacent if other in node_rows)
        hop_count = len(adjacent)
        completion = min(1.0, hop_count / max(WINDOW - 1, 1))
        max_degree = max(adjacent_degrees, default=0)
        transition = (np.mean(np.diff(ordered_times) != 0) if len(ordered_times) > 1 else 0.0)
        dwell_cv = (float(np.std(np.diff(ordered_times)) / max(np.mean(np.diff(ordered_times)), 1e-9))
                    if len(ordered_times) > 2 and np.mean(np.diff(ordered_times)) > 0 else 0.0)
        chain = np.asarray([timestep, hop_count, completion, 1.0 - completion,
                            max_degree, transition, dwell_cv], dtype=np.float32)
        return np.concatenate([raw, chain]).astype(np.float32)

    target_to_cloud = {}
    for account in target_ids:
        account = int(account)
        timestep = node_rows[account][0]
        candidates = [n for n in neighbors.get(account, set()) if n in node_rows and node_rows[n][0] <= timestep]
        candidates.sort(key=lambda n: (node_rows[n][0], n))
        chosen = candidates[-(WINDOW - 1):] + [account]
        chosen = sorted(set(chosen), key=lambda n: (node_rows[n][0], n))
        cloud = [point(n) for n in chosen[-WINDOW:]]
        while len(cloud) < WINDOW:
            cloud.insert(0, cloud[0] if cloud else point(account))
        target_to_cloud[account] = np.stack(cloud[-WINDOW:])
    clouds = np.stack([target_to_cloud[int(node)] for node in target_ids]).astype(np.float32)
    class_counts = {"fraud": int(fraud_mask.sum()), "normal": int(normal_mask.sum()),
                    "unknown": int((~(fraud_mask | normal_mask)).sum())}
    return {"clouds": clouds, "labels": labels, "target_ids": target_ids,
            "metadata": {"labels_rows": len(labels_frame), "edge_rows": len(edges),
                         "feature_rows": feature_rows, "labeled_fraud_nodes": class_counts["fraud"],
                         "labeled_normal_nodes": class_counts["normal"], "unknown_nodes": class_counts["unknown"],
                         "sampled_account_proxies": len(target_ids), "sampled_fraud": int(labels.sum()),
                         "sampled_normal": int((labels == 0).sum()), "K": WINDOW,
                         "feature_source": str(feature_path), "label_source": str(labels_path), "edge_source": str(edges_path),
                         "raw_transaction_amount_available": False,
                         "chain_family_features": list(CHAIN_FEATURES),
                         "unavailable_chain_features": list(UNAVAILABLE_CHAIN_FEATURES),
                         "chain_feature_note": "hop/degree plus coarse transition/dwell proxies from neighbor timesteps; amount retention is unavailable without named amounts and account ledger sequences"}}


def _split(clouds, labels):
    ids = np.arange(len(labels))
    train, temp = train_test_split(ids, test_size=0.4, random_state=SEED, stratify=labels)
    val, test = train_test_split(temp, test_size=0.5, random_state=SEED + 1, stratify=labels[temp])
    return train, val, test


def _scale(train, val, test):
    mu = train.reshape(-1, POINT_DIM).mean(axis=0)
    sd = train.reshape(-1, POINT_DIM).std(axis=0); sd[sd < 1e-6] = 1.0
    return tuple(((part - mu) / sd).astype(np.float32) for part in (train, val, test))


def _matrix(clouds):
    flat = clouds.reshape(len(clouds), WINDOW, POINT_DIM)
    return np.concatenate([flat.mean(axis=1), flat.std(axis=1), flat[:, -1, :]], axis=1).astype(np.float32)


def _f2(y, prediction):
    tp = np.sum((y == 1) & prediction); fp = np.sum((y == 0) & prediction); fn = np.sum((y == 1) & ~prediction)
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return float(5 * precision * recall / max(4 * precision + recall, 1e-12))


def _threshold(y, score):
    candidates = np.unique(np.quantile(score, np.linspace(0, 1, 201)))
    return float(max(candidates, key=lambda cut: _f2(y, score >= cut)))


def _metrics(y_val, p_val, y_test, p_test):
    val_score, score = p_val[:, 1], p_test[:, 1]
    decision = _threshold(y_val, val_score)
    fpr_cut = float(np.quantile(val_score[y_val == 0], 0.99))
    prediction = score >= decision
    fpr_prediction = score >= fpr_cut
    base_rate = float(y_test.mean())
    return {"val": {"threshold_f2": decision, "threshold_at_1pct_fpr": fpr_cut},
            "test": {"n": int(len(y_test)), "n_fraud": int(y_test.sum()),
                     "auc_roc": float(roc_auc_score(y_test, score)),
                     "auc_pr": float(average_precision_score(y_test, score)),
                     "f2": _f2(y_test, prediction), "threshold_f2": decision,
                     "class_recall_at_f2_threshold": {
                         "fraud": float(np.mean(prediction[y_test == 1])) if np.any(y_test == 1) else None,
                         "normal": float(np.mean(~prediction[y_test == 0])) if np.any(y_test == 0) else None,
                     },
                     "lift": float(y_test[prediction].mean() / base_rate) if prediction.any() and base_rate else 0.0,
                     "fraud_recall_at_1pct_fpr": float(np.mean(fpr_prediction[y_test == 1])) if np.any(y_test == 1) else None,
                     "class_recall_at_1pct_fpr": {
                         "fraud": float(np.mean(fpr_prediction[y_test == 1])) if np.any(y_test == 1) else None,
                         "normal": float(np.mean(~fpr_prediction[y_test == 0])) if np.any(y_test == 0) else None,
                     },
                     "fpr_threshold_from_validation": fpr_cut}}


class EllipticCloud(nn.Module):
    def __init__(self, n_tokens: int):
        super().__init__()
        self.encoder = CloudEncoderND(POINT_DIM, d_cloud=48, hidden=32)
        self.head = nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 2))
        self.token_weights: dict[int, QCloudWeight] = {}
        self.n_tokens = n_tokens

    def get_weight(self, token: int):
        if token not in self.token_weights:
            self.token_weights[token] = QCloudWeight(D=POINT_DIM, convergence_threshold=0.02)
        return self.token_weights[token]

    def points_for(self, tokens):
        flat = tokens.reshape(-1)
        unique, inverse = torch.unique(flat, return_inverse=True)
        vectors = torch.as_tensor(np.stack([self.get_weight(int(t)).value for t in unique.tolist()]), dtype=torch.float32)
        return vectors[inverse].view(tokens.shape[0], tokens.shape[1], POINT_DIM)

    def forward(self, points):
        mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        return self.head(self.encoder(points, mask))

    def quann_update(self, tokens, target):
        with torch.no_grad():
            base = self.points_for(tokens)
            row = tokens[0]
            candidates, weights = [], []
            for token in torch.unique(row).tolist():
                weight = self.get_weight(int(token)); candidate = weight.sample()
                changed = base.clone()
                changed[0, row == token] = torch.as_tensor(candidate, dtype=torch.float32)
                candidates.append(changed[0]); weights.append((weight, candidate))
            if not candidates:
                return
            scores = self(torch.cat([base, torch.stack(candidates)]))
            losses = F.cross_entropy(scores, torch.full((len(candidates) + 1,), int(target), dtype=torch.long), reduction="none")
            baseline = float(losses[0])
            for index, (weight, candidate) in enumerate(weights, 1):
                improved = float(losses[index]) < baseline - 1e-7
                weight.narrow_search(candidate, improved); weight.update_sigma(rewarded=improved)

    def convergence(self):
        return {"points": len(self.token_weights), "converged": sum(w.is_converged() for w in self.token_weights.values()),
                "mean_width": float(np.mean([w.error_width().mean() for w in self.token_weights.values()])) if self.token_weights else None}


def _cloud_fit(regime, train_clouds, train_tokens, y_train, val_clouds, val_tokens, y_val,
               test_clouds, test_tokens, y_test):
    torch.manual_seed(SEED); torch.set_num_threads(1)
    model = EllipticCloud(24)
    optimizer = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.head.parameters()), lr=0.002, weight_decay=1e-4)
    started = time.time()
    if regime == "backprop":
        class_counts = np.bincount(y_train, minlength=2)
        weights = torch.as_tensor(len(y_train) / (2 * np.maximum(class_counts, 1)), dtype=torch.float32)
        for epoch in range(8):
            order = np.random.default_rng(SEED + epoch).permutation(len(y_train))
            model.train()
            for start in range(0, len(order), 32):
                batch = order[start:start + 32]
                points = torch.as_tensor(train_clouds[batch], dtype=torch.float32)
                target = torch.as_tensor(y_train[batch], dtype=torch.long)
                loss = F.cross_entropy(model(points), target, weight=weights)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    else:
        rng = np.random.default_rng(SEED); selected = []
        for label in (0, 1):
            candidates = np.flatnonzero(y_train == label)
            selected.extend(rng.choice(candidates, min(len(candidates), 120), replace=False).tolist())
        rng.shuffle(selected)
        for row in selected:
            tokens = torch.as_tensor(train_tokens[row:row + 1], dtype=torch.long)
            target = int(y_train[row]); model.train()
            fixed_points = model.points_for(tokens).detach()
            loss = F.cross_entropy(model(fixed_points), torch.tensor([target]))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            model.eval(); model.quann_update(tokens, target)

    def predict(clouds, tokens):
        outputs = []; model.eval()
        with torch.no_grad():
            for start in range(0, len(tokens), 64):
                if regime == "hybrid":
                    pts = model.points_for(torch.as_tensor(tokens[start:start + 64], dtype=torch.long))
                else:
                    pts = torch.as_tensor(clouds[start:start + 64], dtype=torch.float32)
                outputs.append(torch.softmax(model(pts), dim=-1).cpu().numpy())
        return np.concatenate(outputs)

    p_val, p_test = predict(val_clouds, val_tokens), predict(test_clouds, test_tokens)
    result = _metrics(y_val, p_val, y_test, p_test)
    result["test"].update({"train_sec": round(time.time() - started, 2), **model.convergence()})
    return result, model


def _plot(model, test_tokens, y_test, path: Path):
    from sklearn.decomposition import PCA
    selected = []
    rng = np.random.default_rng(SEED)
    for label in (0, 1):
        rows = np.flatnonzero(y_test == label)
        selected.extend(rng.choice(rows, min(40, len(rows)), replace=False).tolist())
    tokens = torch.as_tensor(test_tokens[selected], dtype=torch.long)
    with torch.no_grad():
        points = model.points_for(tokens).cpu().numpy()
    projection = PCA(n_components=3, random_state=SEED).fit_transform(points.reshape(-1, POINT_DIM)).reshape(len(selected), WINDOW, 3)
    fig = plt.figure(figsize=(11, 7))
    for label, name, color in ((0, "licit / normal", "#3182bd"), (1, "illicit / fraud", "#d62728")):
        ax = fig.add_subplot(1, 2, label + 1, projection="3d")
        for index, original in enumerate(selected):
            if y_test[original] == label:
                cloud = projection[index]
                ax.plot(cloud[:, 0], cloud[:, 1], cloud[:, 2], color=color, alpha=0.35, linewidth=0.8)
                ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color=color, s=9)
        ax.set_title(name); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    fig.suptitle("Elliptic labeled transaction-node proxy clouds — K=10, 36D")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run(data_dir: str | Path | None = None, out_path: str | Path | None = None,
        figure_path: str | Path | None = None, log_path: str | Path | None = None,
        n_accounts: int = 2000):
    np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
    project = ROOT / "aml"
    data_dir = Path(data_dir or project / "data" / "elliptic")
    out_path = Path(out_path or project / "results" / "e2_elliptic.json")
    figure_path = Path(figure_path or project / "figures" / "e2_elliptic_clouds.png")
    log_path = Path(log_path or project / "logs" / "run_e2_elliptic.log")
    for directory in (out_path.parent, figure_path.parent, log_path.parent): directory.mkdir(parents=True, exist_ok=True)
    _log(log_path, f"=== {datetime.now().date()} E2 Elliptic; CPU; seed={SEED}; requested_sample={n_accounts} ===")
    loaded = _load_sample(data_dir, n_accounts=n_accounts)
    clouds, labels = loaded["clouds"], loaded["labels"]
    train_ids, val_ids, test_ids = _split(clouds, labels)
    train_clouds, val_clouds, test_clouds = _scale(clouds[train_ids], clouds[val_ids], clouds[test_ids])
    y_train, y_val, y_test = labels[train_ids], labels[val_ids], labels[test_ids]
    x_train, x_val, x_test = (_matrix(part) for part in (train_clouds, val_clouds, test_clouds))
    gbt = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=0.08, random_state=SEED)
    started = time.time(); gbt.fit(x_train, y_train)
    gbt_result = _metrics(y_val, gbt.predict_proba(x_val), y_test, gbt.predict_proba(x_test))
    gbt_result["test"]["fit_sec"] = round(time.time() - started, 2)
    _log(log_path, f"GBT: ROC={gbt_result['test']['auc_roc']:.4f}, PR={gbt_result['test']['auc_pr']:.4f}, F2={gbt_result['test']['f2']:.4f}")

    cluster = MiniBatchKMeans(n_clusters=24, random_state=SEED, n_init=2, batch_size=512)
    cluster.fit(train_clouds.reshape(-1, POINT_DIM))
    train_tokens = cluster.predict(train_clouds.reshape(-1, POINT_DIM)).reshape(-1, WINDOW)
    val_tokens = cluster.predict(val_clouds.reshape(-1, POINT_DIM)).reshape(-1, WINDOW)
    test_tokens = cluster.predict(test_clouds.reshape(-1, POINT_DIM)).reshape(-1, WINDOW)
    hybrid, hybrid_model = _cloud_fit("hybrid", train_clouds, train_tokens, y_train, val_clouds, val_tokens, y_val,
                                      test_clouds, test_tokens, y_test)
    backprop, _ = _cloud_fit("backprop", train_clouds, train_tokens, y_train, val_clouds, val_tokens, y_val,
                             test_clouds, test_tokens, y_test)
    for name, metrics in (("hybrid", hybrid), ("backprop", backprop)):
        _log(log_path, f"{name}: ROC={metrics['test']['auc_roc']:.4f}, PR={metrics['test']['auc_pr']:.4f}, F2={metrics['test']['f2']:.4f}, points={metrics['test']['points']}, converged={metrics['test']['converged']}")
    _plot(hybrid_model, test_tokens, y_test, figure_path)
    result = {"data": "elliptic", "kaggle_slug": KAGGLE_SLUG, "data_validation": loaded["metadata"],
              "protocol": {"device": "cpu", "seed": SEED, "sampled_labeled_nodes": len(labels),
                           "account_unit": "labeled transaction-node proxy; original mirror has no account identifiers",
                           "split": "stratified node-level 60/20/20; one-hop graph features are transductive"},
              "metrics": {"gbt": gbt_result, "hybrid": hybrid, "backprop": backprop},
              "outputs": {"figure": str(figure_path), "log": str(log_path)}}
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _log(log_path, f"saved {out_path}; labeled_sample={len(labels)}; test={len(test_ids)}")
    return result


if __name__ == "__main__":
    run()
