"""CPU E0 for turbofan health: standard baselines and QUANN point clouds."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from generator import generate_turbofan, make_windows, stage_labels
from cloudlm import CloudEncoderND
from quann_cloud.src.qcloud_weight import QCloudWeight


def _log(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(os.path.join(config.LOGS_DIR, "run_e0_smoke.log"), "a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _scale_by_train(train, val, test):
    mu = train.reshape(-1, train.shape[-1]).mean(axis=0)
    sd = train.reshape(-1, train.shape[-1]).std(axis=0)
    sd[sd < 1e-6] = 1.0
    return tuple(((x - mu) / sd).astype(np.float32) for x in (train, val, test)), mu, sd


class MaintenanceCloud(nn.Module):
    def __init__(self, vocab_size, n_classes, D=36):
        super().__init__()
        self.D = D
        self.cloud_encoder = CloudEncoderND(D, d_cloud=48, hidden=32)
        self.class_head = nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.Linear(32, n_classes))
        self.token_weights = {}

    def get_weight(self, token):
        if token not in self.token_weights:
            self.token_weights[token] = QCloudWeight(D=self.D, convergence_threshold=0.02)
        return self.token_weights[token]

    def points_for(self, tokens):
        ids = tokens.reshape(-1)
        unique, inverse = torch.unique(ids, return_inverse=True)
        values = torch.as_tensor(np.stack([self.get_weight(int(t)).value for t in unique.tolist()]), dtype=torch.float32)
        return values[inverse].view(tokens.shape[0], tokens.shape[1], self.D)

    def forward(self, points):
        mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        return self.class_head(self.cloud_encoder(points, mask))

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with backward compatibility for legacy checkpoints."""
        cloud_prefix = "cloud_encoder."
        if cloud_prefix + "attn_proj.weight" not in state_dict:
            actual_d_cloud = self.cloud_encoder.point_mlp[-1].out_features
            state_dict[cloud_prefix + "attn_proj.weight"] = torch.zeros(1, actual_d_cloud)
            state_dict[cloud_prefix + "attn_proj.bias"] = torch.zeros(1)
        super().load_state_dict(state_dict, strict=strict)

    def quann_update(self, tokens, target):
        with torch.no_grad():
            base = self.points_for(tokens)
            token_row = tokens[0]
            candidates, weights = [], []
            for token in torch.unique(token_row).tolist():
                weight = self.get_weight(int(token))
                candidate = weight.sample()
                candidate_tensor = torch.as_tensor(candidate, dtype=torch.float32)
                cloud = base.clone()
                cloud[0, token_row == token] = candidate_tensor
                candidates.append(cloud[0]); weights.append((weight, candidate))
            clouds = torch.cat([base, torch.stack(candidates)], dim=0)
            logits = self.forward(clouds)
            y = torch.full((len(candidates) + 1,), int(target), dtype=torch.long)
            losses = F.cross_entropy(logits, y, reduction="none")
            base_loss = float(losses[0])
            for i, (weight, candidate) in enumerate(weights, 1):
                reward = float(losses[i]) < base_loss - 1e-7
                weight.narrow_search(candidate, reward)
                weight.update_sigma(rewarded=reward)

    def convergence_info(self):
        widths = [float(w.error_width().mean()) for w in self.token_weights.values()]
        return {"points": len(self.token_weights), "converged": sum(w.is_converged() for w in self.token_weights.values()),
                "mean_width": float(np.mean(widths)) if widths else None}


def _f2(y, pred):
    tp = np.sum((y == 1) & pred); fp = np.sum((y == 0) & pred); fn = np.sum((y == 1) & ~pred)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return float(5 * p * r / max(4 * p + r, 1e-12))


def _best_threshold(y, score):
    cands = np.unique(np.quantile(score, np.linspace(0, 1, 301)))
    return float(max(cands, key=lambda t: _f2(y, score >= t)))


def _binary_metrics(y, probs, threshold=None):
    score = probs[:, 1]
    threshold = _best_threshold(y, score) if threshold is None else threshold
    pred = score >= threshold
    base = float(y.mean())
    return {"n": len(y), "n_positive": int(y.sum()), "base_rate": base, "threshold": float(threshold),
            "auc_roc": float(roc_auc_score(y, score)), "auc_pr": float(average_precision_score(y, score)),
            "f2": _f2(y, pred), "recall": float(np.mean(pred[y == 1])) if np.any(y == 1) else 0.0,
            "lift": float(y[pred].mean() / base) if pred.any() and base else 0.0}


def _stage_metrics(y, probs, class_thresholds=None, critical_threshold=None):
    recalls, thresholds = {}, {}
    aucs, aps = [], []
    for c, name in enumerate(config.STAGES):
        positive = y == c; negative = ~positive
        threshold = (float(class_thresholds[name]) if class_thresholds is not None else
                     (float(np.quantile(probs[negative, c], 0.99)) if negative.any() else 1.0))
        thresholds[name] = threshold
        recalls[name] = float(np.mean(probs[positive, c] >= threshold)) if positive.any() else None
        binary = positive.astype(int)
        if len(np.unique(binary)) > 1:
            aucs.append(float(roc_auc_score(binary, probs[:, c])))
            aps.append(float(average_precision_score(binary, probs[:, c])))
    critical_y = (y == 0).astype(int)
    critical_score = probs[:, 0]
    critical_thr = _best_threshold(critical_y, critical_score) if critical_threshold is None else float(critical_threshold)
    return {"n": len(y), "auc_roc_macro": float(np.mean(aucs)) if aucs else None,
            "auc_pr_macro": float(np.mean(aps)) if aps else None,
            "top1_accuracy": float(accuracy_score(y, probs.argmax(axis=1))),
            "top4_accuracy": float(np.mean([int(y[i] in np.argsort(probs[i])[-4:]) for i in range(len(y))])),
            "class_recall_at_1pct_fpr": recalls,
            "class_thresholds_at_1pct_fpr": thresholds,
            "critical_f2": _f2(critical_y, critical_score >= critical_thr),
            "critical_threshold": critical_thr}


def _standard(task, xtr, ytr, xva, yva, xte, yte):
    classes = np.unique(ytr)
    models = {
        "lr": LogisticRegression(max_iter=900, class_weight="balanced", random_state=config.SEED),
        "rf": RandomForestClassifier(n_estimators=70, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=config.SEED),
        "gbt": HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, learning_rate=0.08, random_state=config.SEED),
        "mlp": MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=70, early_stopping=True, random_state=config.SEED),
    }
    results = {}
    for name, model in models.items():
        started = time.time(); model.fit(xtr, ytr)
        pva, pte = model.predict_proba(xva), model.predict_proba(xte)
        if task == "binary":
            val_metrics = _binary_metrics(yva, pva)
            test_metrics = _binary_metrics(yte, pte, val_metrics["threshold"])
        else:
            val_metrics = _stage_metrics(yva, pva)
            test_metrics = _stage_metrics(yte, pte, val_metrics["class_thresholds_at_1pct_fpr"], val_metrics["critical_threshold"])
        test_metrics["fit_sec"] = round(time.time() - started, 2)
        results[name] = {"test": test_metrics, "val": val_metrics}
        _log(f"standard/{task}/{name} test={test_metrics.get('auc_roc', test_metrics.get('auc_roc_macro'))} top1={test_metrics.get('top1_accuracy', '-')}")
    return results


def _fit_cloud(regime, task, train_clouds, train_tokens, ytr, val_clouds, val_tokens, yva, test_clouds, test_tokens, yte, n_classes):
    torch.manual_seed(config.SEED)
    model = MaintenanceCloud(int(train_tokens.max()) + 1, n_classes=n_classes).to(config.DEVICE)
    if regime == "hybrid":
        optimizer = torch.optim.AdamW(list(model.cloud_encoder.parameters()) + list(model.class_head.parameters()), lr=0.002, weight_decay=1e-4)
    elif regime == "backprop":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    else:
        optimizer = None
    rng = np.random.default_rng(config.SEED)
    selected = []
    cap = 110 if task == "stage" else 220
    for label in np.unique(ytr):
        ids = np.flatnonzero(ytr == label)
        selected.extend(rng.choice(ids, min(len(ids), cap), replace=False).tolist())
    rng.shuffle(selected)
    started = time.time()
    for row in selected:
        tokens = torch.as_tensor(train_tokens[row:row + 1], dtype=torch.long)
        target = int(ytr[row])
        if optimizer is not None:
            model.train()
            points = model.points_for(tokens).detach() if regime == "hybrid" else torch.as_tensor(train_clouds[row:row + 1], dtype=torch.float32)
            logits = model(points)
            loss = F.cross_entropy(logits, torch.tensor([target]))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if regime in {"pure", "hybrid"}:
            model.eval(); model.quann_update(tokens, target)
    train_sec = time.time() - started

    def predict(clouds, tokens):
        output = []; model.eval()
        with torch.no_grad():
            for start in range(0, len(tokens), 128):
                tt = torch.as_tensor(tokens[start:start + 128], dtype=torch.long)
                points = (model.points_for(tt) if regime != "backprop" else
                          torch.as_tensor(clouds[start:start + 128], dtype=torch.float32))
                output.append(torch.softmax(model(points), dim=-1).cpu().numpy())
        return np.concatenate(output)

    pva = predict(val_clouds, val_tokens); pte = predict(test_clouds, test_tokens)
    if task == "binary":
        vm = _binary_metrics(yva, pva); tm = _binary_metrics(yte, pte, vm["threshold"])
    else:
        vm = _stage_metrics(yva, pva)
        tm = _stage_metrics(yte, pte, vm["class_thresholds_at_1pct_fpr"], vm["critical_threshold"])
    tm.update({"regime": regime, "train_sec": round(train_sec, 2), **model.convergence_info()})
    path = os.path.join(config.RESULTS_DIR, f"quann_{task}_{regime}.pt")
    torch.save({"model_state": model.state_dict(), "token_weights": model.token_weights,
                "config": {"D": 36, "n_classes": n_classes}}, path)
    _log(f"quann/{task}/{regime} test={tm.get('auc_roc', tm.get('auc_roc_macro'))} F2={tm.get('f2', tm.get('critical_f2'))} converged={tm['converged']}/{tm['points']} ({train_sec:.1f}s)")
    return {"test": tm, "val": vm}, model


def _plot_stages(model, tokens, stage_y):
    rng = np.random.default_rng(config.SEED)
    selected = []
    for stage in range(4):
        ids = np.flatnonzero(stage_y == stage)
        if len(ids): selected.extend(rng.choice(ids, min(30, len(ids)), replace=False).tolist())
    tt = torch.as_tensor(tokens[selected], dtype=torch.long)
    with torch.no_grad():
        points = model.points_for(tt).cpu().numpy()
    flat = points.reshape(-1, 36)
    projected = PCA(n_components=3, random_state=config.SEED).fit_transform(flat).reshape(len(selected), 10, 3)
    fig = plt.figure(figsize=(12, 9)); colors = ["#d62728", "#ff7f0e", "#3182bd", "#31a354"]
    for stage, name in enumerate(config.STAGES):
        ax = fig.add_subplot(2, 2, stage + 1, projection="3d")
        local = [j for j, original in enumerate(selected) if stage_y[original] == stage]
        for j in local[:20]:
            cloud = projected[j]
            ax.plot(cloud[:, 0], cloud[:, 1], cloud[:, 2], color=colors[stage], alpha=0.35, linewidth=0.8)
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=8, color=colors[stage], alpha=0.65)
        ax.set_title(name); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    fig.suptitle("Turbofan wear-stage clouds (10 steps × 36D, PCA projection)")
    fig.tight_layout(); path = os.path.join(config.FIGURES_DIR, "e0_stage_clouds.png")
    fig.savefig(path, dpi=130); plt.close(fig); _log(f"saved figure {path}")


def run_e0(n_engines=48, max_cycles=140, out_path=None):
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True); os.makedirs(config.FIGURES_DIR, exist_ok=True); os.makedirs(config.LOGS_DIR, exist_ok=True)
    _log("=== [2026-10-16] 1/5: C-MAPSS retrieval unavailable; synthetic fallback ===")
    records = generate_turbofan(n_engines=n_engines, max_cycles=max_cycles)
    clouds, rul, engine_ids = make_windows(records)
    labels_stage = stage_labels(rul); labels_binary = (rul <= 30).astype(np.int64)
    engines = np.unique(engine_ids); rng = np.random.default_rng(config.SEED); rng.shuffle(engines)
    n_train = max(2, int(len(engines) * 0.70)); n_val = max(1, int(len(engines) * 0.15))
    train_engines = set(engines[:n_train]); val_engines = set(engines[n_train:n_train + n_val]); test_engines = set(engines[n_train + n_val:])
    tr = np.flatnonzero(np.isin(engine_ids, list(train_engines))); va = np.flatnonzero(np.isin(engine_ids, list(val_engines))); te = np.flatnonzero(np.isin(engine_ids, list(test_engines)))
    scaled, mu, sd = _scale_by_train(clouds[tr], clouds[va], clouds[te])
    xtr, xva, xte = scaled
    y = {"binary": labels_binary, "stage": labels_stage}
    _log(f"synthetic fallback rows={len(records):,}, windows={len(clouds):,}, engines={len(engines)}, splits={len(tr)}/{len(va)}/{len(te)}, D=36 W=10 device=cpu")

    _log("=== [2026-10-16] 2/5: fit train-only clustering tokens and standard baselines ===")
    clusterer = MiniBatchKMeans(n_clusters=24, random_state=config.SEED, n_init=3, batch_size=512)
    clusterer.fit(xtr.reshape(-1, 36))
    tokens = clusterer.predict(xtr.reshape(-1, 36)).reshape(len(tr), 10)
    val_tokens = clusterer.predict(xva.reshape(-1, 36)).reshape(len(va), 10)
    test_tokens = clusterer.predict(xte.reshape(-1, 36)).reshape(len(te), 10)
    # Baselines see the same 10×36 window flattened to a train-scaled vector.
    flat_tr, flat_va, flat_te = xtr.reshape(len(tr), -1), xva.reshape(len(va), -1), xte.reshape(len(te), -1)
    task_results, hybrid_models = {}, {}
    for task, n_classes in (("binary", 2), ("stage", 4)):
        ytr, yva, yte = y[task][tr], y[task][va], y[task][te]
        _log(f"=== [2026-10-16] 3/5: standard LR/RF/GBT/MLP, task={task} ===")
        standard = _standard(task, flat_tr, ytr, flat_va, yva, flat_te, yte)
        _log(f"=== [2026-10-16] 4/5: QUANN pure/hybrid/backprop, task={task} ===")
        configs, models = {}, {}
        for regime in ("pure", "hybrid", "backprop"):
            configs[regime], models[regime] = _fit_cloud(regime, task, xtr, tokens, ytr, xva, val_tokens, yva, xte, test_tokens, yte, n_classes)
        task_results[task] = {"standard": standard, "configs": configs}
        hybrid_models[task] = models["hybrid"]

    _log("=== [2026-10-16] 5/5: write results and four-stage 3D figure ===")
    _plot_stages(hybrid_models["stage"], test_tokens, y["stage"][te])
    out_path = out_path or os.path.join(config.RESULTS_DIR, "e0_smoke.json")
    result = {
        "data": "synthetic_fallback",
        "data_source": {"requested": "C-MAPSS FD001", "attempted": ["NASA PCoE repository redirect (download timed out)", "GitHub raw mirror 1 (404)", "GitHub raw mirror 2 (404)"], "n_engines": len(engines), "trajectory_rows": len(records), "windows": len(clouds)},
        "tasks": task_results,
        "split": {"train_windows": len(tr), "val_windows": len(va), "test_windows": len(te), "grouping": "engine-disjoint"},
        "model": {"D": 36, "window_size": 10, "cluster_tokens": int(clusterer.n_clusters), "epochs": 1, "device": "cpu"},
        "limitations": ["C-MAPSS FD001 could not be retrieved; all metrics use the synthetic fallback.", "Top-4 accuracy over exactly four stages is trivially 1.0.", "E0 uses one pass and capped QUANN updates; these are smoke metrics, not tuned estimates."],
    }
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    _log(f"saved result {out_path}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-engines", type=int, default=48)
    parser.add_argument("--max-cycles", type=int, default=140)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_e0(args.n_engines, args.max_cycles, args.out)
