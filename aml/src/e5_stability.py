#!/usr/bin/env python3
"""E5: audit-batch pilot for AML naming and SHAP retraining stability."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parent
ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import config
from e2_elliptic import POINT_DIM, WINDOW, EllipticCloud, _load_sample, _matrix, _scale, _split


SEED = config.SEED
DEVICE = "cpu"
LABEL_NAMES = ("licit", "illicit")
FEATURE_STATS = ("mean", "std", "last")
CHAIN_FEATURES = (
    "timestep", "observed_hop_count", "hop_completion_ratio", "1-hop_completion",
    "max_counterparty_degree", "counterparty_transition_rate", "dwell_time_cv",
)


def _log(path: Path, message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _feature_name(index: int) -> str:
    statistic, feature_index = divmod(int(index), POINT_DIM)
    feature = (
        f"anonymous_axis_{feature_index + 1:02d}"
        if feature_index < 29
        else CHAIN_FEATURES[feature_index - 29]
    )
    return f"{FEATURE_STATS[statistic]} · {feature}"


def _reason(cloud: np.ndarray) -> str:
    """Summarize the named one-hop proxy fields; raw Elliptic axes are anonymous."""
    last = np.asarray(cloud[-1], dtype=float)
    return (
        "1-hop transaction-neighborhood proxy: "
        f"observed_hop_count={last[30]:.0f}, "
        f"hop_completion_ratio={last[31]:.3f}, "
        f"max_counterparty_degree={last[33]:.0f}, "
        f"counterparty_transition_rate={last[34]:.3f}, "
        f"dwell_time_cv={last[35]:.3f}; raw feature axes are anonymized."
    )


def _encode(model: EllipticCloud, tokens: np.ndarray, batch_size: int = 64):
    """Encode E2 quantized clouds in its trained CloudEncoderND representation."""
    embeddings, logits = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = torch.as_tensor(tokens[start:start + batch_size], dtype=torch.long, device=DEVICE)
            points = model.points_for(batch)
            mask = torch.ones(batch.shape, dtype=torch.bool, device=DEVICE)
            encoded = model.encoder(points, mask)
            embeddings.append(encoded.cpu().numpy())
            logits.append(model.head(encoded).cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(logits)


def _fit_hybrid(seed: int, train_tokens: np.ndarray, y_train: np.ndarray):
    """Retrain E2's hybrid reader and QCloudWeight points at one explicit seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    model = EllipticCloud(24).to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.head.parameters()),
        lr=0.002,
        weight_decay=1e-4,
    )

    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        candidates = np.flatnonzero(y_train == label)
        selected.extend(rng.choice(candidates, min(len(candidates), 120), replace=False).tolist())
    rng.shuffle(selected)

    for row in selected:
        tokens = torch.as_tensor(train_tokens[row:row + 1], dtype=torch.long, device=DEVICE)
        target = int(y_train[row])
        model.train()
        fixed_points = model.points_for(tokens).detach()
        loss = torch.nn.functional.cross_entropy(model(fixed_points), torch.tensor([target], device=DEVICE))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        model.quann_update(tokens, target)
    return model, len(selected)


def _positive_class_shap(values: object, n_rows: int, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    array = np.asarray(values)
    if array.ndim == 3:
        if array.shape[-1] == 2:
            array = array[:, :, 1]
        elif array.shape[0] == 2:
            array = array[1]
        else:
            raise ValueError(f"Unexpected SHAP output shape {array.shape}")
    if array.shape != (n_rows, n_features):
        raise ValueError(f"SHAP shape mismatch: got {array.shape}, expected {(n_rows, n_features)}")
    return array.astype(np.float64, copy=False)


def _align_pca(reference_train: np.ndarray, train_xy: np.ndarray, test_xy: np.ndarray) -> np.ndarray:
    """Rotate a train-fit PCA plane to the reference plane using shared train rows."""
    left = train_xy - train_xy.mean(axis=0, keepdims=True)
    right = reference_train - reference_train.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(left.T @ right, full_matrices=False)
    rotation = u @ vt
    return (test_xy - train_xy.mean(axis=0, keepdims=True)) @ rotation + reference_train.mean(axis=0)


def _shap_feature_rows(shap_values: np.ndarray, sample_positions: np.ndarray,
                       target_ids: np.ndarray) -> list[dict]:
    rows = []
    for row, position in enumerate(sample_positions):
        top = np.argsort(-np.abs(shap_values[row]), kind="stable")[:3]
        rows.append({
            "node_id": int(target_ids[position]),
            "top3": [{"index": int(index), "name": _feature_name(index)} for index in top],
        })
    return rows


def run(seeds: int = 5, smoke: bool = False, shap_n: int = 500,
        n_accounts: int = 2000, data_dir: str | Path | None = None,
        out_path: str | Path | None = None, log_path: str | Path | None = None):
    if seeds < 1:
        raise ValueError("--seeds must be at least 1")
    if shap_n < 1:
        raise ValueError("--shap-n must be at least 1")
    if n_accounts < 4:
        raise ValueError("--n-accounts must be at least 4")

    started = time.monotonic()
    effective_accounts = min(n_accounts, 200) if smoke else n_accounts
    seeds_used = list(range(SEED, SEED + seeds))
    data_dir = Path(data_dir or PROJECT / "data" / "elliptic")
    out_path = Path(out_path or Path(config.RESULTS_DIR) / "e5_stability.json")
    log_path = Path(log_path or Path(config.LOGS_DIR) / "run_e5_stability.log")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_default_device(DEVICE)

    _log(log_path, f"=== [{datetime.now().date()}] 1/4: fixed E2 sample/split; CPU; seeds={seeds_used} ===")
    loaded = _load_sample(data_dir, n_accounts=effective_accounts)
    clouds, labels, target_ids = loaded["clouds"], loaded["labels"], loaded["target_ids"]
    train_ids, val_ids, test_ids = _split(clouds, labels)
    train_clouds, val_clouds, test_clouds = _scale(
        clouds[train_ids], clouds[val_ids], clouds[test_ids],
    )
    y_train = labels[train_ids]
    cluster = MiniBatchKMeans(n_clusters=24, random_state=SEED, n_init=2, batch_size=512)
    cluster.fit(train_clouds.reshape(-1, POINT_DIM))
    train_tokens = cluster.predict(train_clouds.reshape(-1, POINT_DIM)).reshape(-1, WINDOW)
    test_tokens = cluster.predict(test_clouds.reshape(-1, POINT_DIM)).reshape(-1, WINDOW)
    x_train, x_test = _matrix(train_clouds), _matrix(test_clouds)

    shap_count = min(shap_n, len(test_ids))
    shap_positions = np.sort(
        np.random.default_rng(SEED + 97).choice(len(test_ids), shap_count, replace=False)
    )
    shap_x = x_test[shap_positions]

    _log(log_path, f"=== [{datetime.now().date()}] 2/4: hybrid reader retraining and train-only prototypes ===")
    name_rows, pca_rows, audit_reference = [], [], None
    reference_train_xy = None
    hybrid_fit_seconds = []
    for run_index, seed in enumerate(seeds_used):
        fit_started = time.monotonic()
        model, fit_rows = _fit_hybrid(seed, train_tokens, y_train)
        train_embeddings, _ = _encode(model, train_tokens)
        test_embeddings, test_logits = _encode(model, test_tokens)
        prototypes = np.stack([
            train_embeddings[y_train == label].mean(axis=0)
            for label in (0, 1)
        ])
        distances = np.linalg.norm(test_embeddings[:, None, :] - prototypes[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        runner_up = 1 - nearest
        predicted = test_logits.argmax(axis=1)

        pca = PCA(n_components=2, random_state=seed)
        train_xy = pca.fit_transform(train_embeddings)
        test_xy = pca.transform(test_embeddings)
        if run_index == 0:
            reference_train_xy = train_xy.copy()
            aligned_test_xy = test_xy
        else:
            aligned_test_xy = _align_pca(reference_train_xy, train_xy, test_xy)

        name_rows.append([LABEL_NAMES[index] for index in nearest])
        pca_rows.append(aligned_test_xy)
        hybrid_fit_seconds.append(time.monotonic() - fit_started)
        if run_index == 0:
            audit_reference = {
                "nearest": nearest,
                "runner_up": runner_up,
                "distances": distances,
                "predicted": predicted,
                "pca2": aligned_test_xy,
                "fit_rows": fit_rows,
                "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
            }
        _log(log_path, f"hybrid seed={seed}: trained_rows={fit_rows}; fit+encode={hybrid_fit_seconds[-1]:.2f}s")

    names = np.asarray(name_rows, dtype=object)
    name_reference = names[0]
    name_stability = float(np.mean(np.all(names == name_reference[None, :], axis=0)))
    coordinates = np.stack(pca_rows)
    if len(seeds_used) > 1:
        mean_pca2_disp = float(np.linalg.norm(coordinates[1:] - coordinates[0], axis=2).mean())
    else:
        mean_pca2_disp = 0.0

    _log(log_path, f"=== [{datetime.now().date()}] 3/4: GBM seed fits and TreeExplainer on fixed subsample ===")
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("E5 requires SHAP (tested protocol version 0.54.1)") from exc
    if shap.__version__ != "0.54.1":
        _log(log_path, f"SHAP version note: installed={shap.__version__}; requested protocol=0.54.1")

    shap_values_by_seed, top3_by_seed, shap_fit_seconds = [], {}, []
    for seed in seeds_used:
        fit_started = time.monotonic()
        gbm = GradientBoostingClassifier(
            n_estimators=100, max_leaf_nodes=15, learning_rate=0.08, random_state=seed,
            subsample=0.8,
        )
        gbm.fit(x_train, y_train)
        explainer = shap.TreeExplainer(gbm)
        shap_values = _positive_class_shap(
            explainer.shap_values(shap_x), len(shap_positions), x_train.shape[1],
        )
        shap_values_by_seed.append(shap_values)
        top3_by_seed[str(seed)] = _shap_feature_rows(shap_values, shap_positions, target_ids[test_ids])
        shap_fit_seconds.append(time.monotonic() - fit_started)
        _log(log_path, f"gbm+TreeExplainer seed={seed}: nodes={len(shap_positions)}; elapsed={shap_fit_seconds[-1]:.2f}s")

    top_indices = np.stack([
        np.argsort(-np.abs(values), axis=1, kind="stable")[:, :3]
        for values in shap_values_by_seed
    ])
    top1_stability = float(np.mean(np.all(top_indices[:, :, 0] == top_indices[0:1, :, 0], axis=0)))
    pair_jaccards, pair_spearman = [], []
    for first in range(len(seeds_used)):
        for second in range(first + 1, len(seeds_used)):
            for row in range(len(shap_positions)):
                left = set(map(int, top_indices[first, row]))
                right = set(map(int, top_indices[second, row]))
                pair_jaccards.append(len(left & right) / len(left | right))
                correlation = spearmanr(shap_values_by_seed[first][row], shap_values_by_seed[second][row]).statistic
                if np.isfinite(correlation):
                    pair_spearman.append(float(correlation))
    if len(seeds_used) == 1:
        jaccard_mean = 1.0
    else:
        jaccard_mean = float(np.mean(pair_jaccards)) if pair_jaccards else None
    spearman_mean = float(np.mean(pair_spearman)) if pair_spearman else None

    _log(log_path, f"=== [{datetime.now().date()}] 4/4: reference-seed audit batch and JSON ===")
    reference = audit_reference
    audit_batch = []
    for position, split_position in enumerate(test_ids):
        nearest_id = int(reference["nearest"][position])
        runner_id = int(reference["runner_up"][position])
        nearest_dist = float(reference["distances"][position, nearest_id])
        runner_dist = float(reference["distances"][position, runner_id])
        ratio = nearest_dist / runner_dist if runner_dist > 1e-12 else None
        audit_batch.append({
            "node_id": int(target_ids[split_position]),
            "true_label": LABEL_NAMES[int(labels[split_position])],
            "predicted": LABEL_NAMES[int(reference["predicted"][position])],
            "nearest_name": LABEL_NAMES[nearest_id],
            "nearest_dist": nearest_dist,
            "runner_up_name": LABEL_NAMES[runner_id],
            "separation_ratio": ratio,
            "pca2": [float(value) for value in reference["pca2"][position]],
            "rationale": _reason(clouds[split_position]),
        })

    elapsed = time.monotonic() - started
    result = {
        "experiment": "E5 retraining stability and audit batch",
        "protocol": {
            "device": DEVICE,
            "seed": SEED,
            "S": seeds,
            "seeds": seeds_used,
            "test_n": int(len(test_ids)),
            "shap_subsample": int(len(shap_positions)),
            "shap_n_requested": int(shap_n),
            "sampled_labeled_nodes": int(len(labels)),
            "window_K": WINDOW,
            "split": "same e2 60/20/20 stratified split (random_state=SEED, SEED+1)",
            "embedding": "hybrid CloudEncoderND output",
            "prototype_fit": "per-seed train-only class centroid; Euclidean distance",
            "pca": "per-seed PCA-2 fit on train embeddings; orthogonally aligned on shared train rows to reference seed",
            "hybrid_reader": "E2 EllipticCloud(24), AdamW reader + QCloudWeight narrowing; balanced up to 120/class",
            "gbm": "GradientBoostingClassifier(n_estimators=100,max_leaf_nodes=15,learning_rate=0.08,subsample=0.8,random_state=seed); classic GBM (not Hist) with subsample<1 so retraining genuinely varies (non-degenerate SHAP stability contrast; E2's Hist-GBM is deterministic)",
            "shap": f"TreeExplainer; installed version {shap.__version__}; fixed test subsample seed={SEED + 97}",
            "smoke": bool(smoke),
        },
        "geometry": {
            "name_stability": name_stability,
            "name_stability_definition": "fraction of test nodes whose nearest name matches the reference-seed name across all retrains",
            "mean_pca2_disp": mean_pca2_disp,
            "mean_pca2_disp_definition": "mean Euclidean test-coordinate displacement from reference seed after train-anchor PCA-plane alignment",
            "reference_seed": int(seeds_used[0]),
            "reference_pca_train_variance_ratio": reference["pca_variance_ratio"],
            "mean_hybrid_fit_encode_sec": float(np.mean(hybrid_fit_seconds)),
        },
        "shap_contrast": {
            "shap_top1_stability": top1_stability,
            "shap_top3_jaccard_mean": jaccard_mean,
            "shap_spearman_mean": spearman_mean,
            "spearman_valid_node_seed_pairs": len(pair_spearman),
            "seed_top3_features": top3_by_seed,
            "mean_gbm_treeexplainer_sec": float(np.mean(shap_fit_seconds)),
        },
        "audit_batch": audit_batch,
        "runtime_sec": round(elapsed, 3),
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _log(log_path, f"saved {out_path}; test_n={len(test_ids)}; name_stability={name_stability:.4f}; runtime={elapsed:.2f}s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="number of consecutive retraining seeds")
    parser.add_argument("--smoke", action="store_true", help="cap the sampled data at 200 labeled nodes")
    parser.add_argument("--shap-n", type=int, default=500, help="fixed deterministic test SHAP subsample size")
    parser.add_argument("--n-accounts", type=int, default=2000, help="number of labeled transaction-node proxies")
    args = parser.parse_args()
    run(seeds=args.seeds, smoke=args.smoke, shap_n=args.shap_n, n_accounts=args.n_accounts)


if __name__ == "__main__":
    main()
