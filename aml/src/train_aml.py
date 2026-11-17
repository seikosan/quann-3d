"""CPU E0: synthetic AML typologies, standard and QUANN cloud regimes."""
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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from generator import generate_transactions
from quantize import FeatureQuantizer
from cloudlm import CloudEncoderND, WordToND
from quann_cloud.src.qcloud_weight import QCloudWeight


def _log(message: str):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with open(os.path.join(config.LOGS_DIR, "run_e0_smoke.log"), "a", encoding="utf-8") as stream:
        stream.write(line + "\n")


class AMLCloud(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int = 4, D: int = 8):
        super().__init__()
        self.D = D
        self.embedding = nn.Embedding(vocab_size, 24)
        self.to_nd = WordToND(24, D, hidden=24)
        self.cloud_encoder = CloudEncoderND(D, d_cloud=48, hidden=32)
        self.class_head = nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.Linear(32, n_classes))
        self.token_weights: dict[int, QCloudWeight] = {}

    def get_weight(self, token_id: int):
        if token_id not in self.token_weights:
            self.token_weights[token_id] = QCloudWeight(D=self.D, convergence_threshold=0.02)
        return self.token_weights[token_id]

    def points_for(self, tokens):
        ids = tokens.reshape(-1)
        unique, inverse = torch.unique(ids, return_inverse=True)
        vectors = torch.as_tensor(
            np.stack([self.get_weight(int(t)).value for t in unique.tolist()]),
            dtype=torch.float32, device=tokens.device)
        return vectors[inverse].view(tokens.shape[0], tokens.shape[1], self.D)

    def forward(self, tokens, points_override=None):
        if points_override is None:
            points = self.to_nd(self.embedding(tokens) * np.sqrt(24.0))
        else:
            points = points_override
        mask = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device)
        return self.class_head(self.cloud_encoder(points, mask)), points

    def quann_update(self, tokens, target, hybrid=False):
        with torch.no_grad():
            base = self.points_for(tokens)
            ids = tokens[0]
            unique = torch.unique(ids)
            candidates, weights = [], []
            for token in unique.tolist():
                weight = self.get_weight(int(token))
                candidate = torch.as_tensor(weight.sample(), dtype=torch.float32)
                cloud = base.clone()
                cloud[0, ids == token] = candidate
                candidates.append(cloud[0])
                weights.append((weight, candidate.numpy()))
            if not candidates:
                return
            logits, _ = self.forward(tokens.expand(len(candidates) + 1, -1),
                                     torch.cat([base, torch.stack(candidates)], dim=0))
            targets = torch.full((len(candidates) + 1,), int(target), dtype=torch.long)
            losses = F.cross_entropy(logits, targets, reduction="none")
            base_loss = float(losses[0])
            for idx, (weight, candidate) in enumerate(weights, start=1):
                improved = float(losses[idx]) < base_loss - 1e-7
                weight.narrow_search(candidate, improved)
                weight.update_sigma(rewarded=improved)

    def convergence_info(self):
        widths = [float(w.error_width().mean()) for w in self.token_weights.values()]
        return {
            "points": len(self.token_weights),
            "converged": sum(w.is_converged() for w in self.token_weights.values()),
            "mean_width": float(np.mean(widths)) if widths else None,
        }


def _feature_matrix(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    numeric = [c for c in train.columns if c not in {"window_id", "typology", "counterparty_id", "channel"}]
    categories = ["channel"]
    xtr = pd.get_dummies(train[numeric + categories], columns=categories, dtype=float)
    xva = pd.get_dummies(val[numeric + categories], columns=categories, dtype=float).reindex(columns=xtr.columns, fill_value=0)
    xte = pd.get_dummies(test[numeric + categories], columns=categories, dtype=float).reindex(columns=xtr.columns, fill_value=0)
    return xtr.to_numpy(dtype=np.float32), xva.to_numpy(dtype=np.float32), xte.to_numpy(dtype=np.float32)


def _metrics(y, probs, threshold=None, class_thresholds=None):
    labels = np.arange(len(config.CLASSES))
    abnormal = (y != 0).astype(int)
    score = 1.0 - probs[:, 0]
    if threshold is None:
        candidates = np.unique(np.quantile(score, np.linspace(0, 1, 301)))
        threshold = max(candidates, key=lambda t: _f2(abnormal, score >= t))
    pred = score >= threshold
    f2 = _f2(abnormal, pred)
    recalls, thresholds = {}, {}
    for c, name in enumerate(config.CLASSES):
        positive = y == c
        negatives = ~positive
        cut = (float(class_thresholds[name]) if class_thresholds is not None else
               (float(np.quantile(probs[negatives, c], 0.99)) if negatives.any() else 1.0))
        thresholds[name] = cut
        recalls[name] = float(np.mean(probs[positive, c] >= cut)) if positive.any() else None
    aucs, aps = [], []
    for c in labels:
        binary = (y == c).astype(int)
        aucs.append(float(roc_auc_score(binary, probs[:, c])))
        aps.append(float(average_precision_score(binary, probs[:, c])))
    base = float(abnormal.mean())
    predicted = score >= threshold
    lift = float(abnormal[predicted].mean() / base) if predicted.any() and base else 0.0
    return {
        "n": int(len(y)), "n_abnormal": int(abnormal.sum()), "base_rate": base,
        "threshold": float(threshold), "auc_roc": float(np.mean(aucs)),
        "auc_pr": float(np.mean(aps)), "f2": float(f2), "lift": lift,
        "class_recall_at_1pct_fpr": recalls,
        "class_thresholds_at_1pct_fpr": thresholds,
        "accuracy": float(accuracy_score(y, probs.argmax(axis=1))),
    }


def _f2(y, pred):
    tp = np.sum((y == 1) & pred); fp = np.sum((y == 0) & pred); fn = np.sum((y == 1) & ~pred)
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return float(5 * precision * recall / max(4 * precision + recall, 1e-12))


def _standard_models(xtr, ytr, xva, yva, xte, yte):
    models = {
        "lr": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=config.SEED),
        "rf": RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=config.SEED),
        "gbt": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=0.08, random_state=config.SEED),
    }
    results = {}
    for name, model in models.items():
        started = time.time()
        model.fit(xtr, ytr)
        pva, pte = model.predict_proba(xva), model.predict_proba(xte)
        vm = _metrics(yva, pva)
        tm = _metrics(yte, pte, vm["threshold"], vm["class_thresholds_at_1pct_fpr"])
        tm["fit_sec"] = round(time.time() - started, 2)
        results[name] = {"test": tm, "val": vm}
        _log(f"standard/{name} test roc={tm['auc_roc']:.4f} pr={tm['auc_pr']:.4f} F2={tm['f2']:.4f}")
    return results


def _fit_cloud(regime, tokens, y, val_tokens, yval, test_tokens, ytest, n_classes, vocab_size):
    torch.manual_seed(config.SEED)
    model = AMLCloud(vocab_size=vocab_size, n_classes=n_classes).to(config.DEVICE)
    optimizer = None
    if regime == "hybrid":
        optimizer = torch.optim.AdamW(list(model.cloud_encoder.parameters()) + list(model.class_head.parameters()), lr=0.003, weight_decay=1e-4)
    elif regime == "backprop":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

    # Keep the derivative-free smoke bounded while all held-out rows remain used.
    rng = np.random.default_rng(config.SEED)
    train_ids = np.arange(len(y))
    if len(train_ids) > 480:
        selected = []
        for c in np.unique(y):
            candidates = train_ids[y == c]
            selected.extend(rng.choice(candidates, min(len(candidates), 120), replace=False).tolist())
        train_ids = np.asarray(selected, dtype=int)
        rng.shuffle(train_ids)
    model.train()
    started = time.time()
    for row in train_ids:
        t = torch.as_tensor(tokens[row:row + 1], dtype=torch.long)
        target = int(y[row])
        if regime in {"hybrid", "backprop"}:
            model.train()
            logits, _ = model(t, points_override=model.points_for(t).detach() if regime == "hybrid" else None)
            loss = F.cross_entropy(logits, torch.tensor([target]))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if regime in {"pure", "hybrid"}:
            model.eval()
            model.quann_update(t, target, hybrid=regime == "hybrid")
    train_sec = time.time() - started

    def predict(matrix):
        model.eval(); output = []
        with torch.no_grad():
            for start in range(0, len(matrix), 256):
                t = torch.as_tensor(matrix[start:start + 256], dtype=torch.long)
                pts = model.points_for(t) if regime != "backprop" else None
                logits, _ = model(t, points_override=pts)
                output.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(output)

    val_prob, test_prob = predict(val_tokens), predict(test_tokens)
    val_metrics = _metrics(yval, val_prob)
    test_metrics = _metrics(ytest, test_prob, val_metrics["threshold"], val_metrics["class_thresholds_at_1pct_fpr"])
    test_metrics.update({"regime": regime, "train_sec": round(train_sec, 2), **model.convergence_info()})
    cp_path = os.path.join(config.RESULTS_DIR, f"quann_{regime}.pt")
    torch.save({"model_state": model.state_dict(), "token_weights": model.token_weights,
                "config": {"D": model.D, "vocab_size": model.embedding.num_embeddings}}, cp_path)
    _log(f"quann/{regime} test roc={test_metrics['auc_roc']:.4f} pr={test_metrics['auc_pr']:.4f} F2={test_metrics['f2']:.4f} points={test_metrics['points']} converged={test_metrics['converged']} ({train_sec:.1f}s)")
    return {"test": test_metrics, "val": val_metrics}, model


def _plot_clouds(model, tokens, y):
    rng = np.random.default_rng(config.SEED)
    selected = []
    for c in range(len(config.CLASSES)):
        ids = np.flatnonzero(y == c)
        selected.extend(rng.choice(ids, min(40, len(ids)), replace=False).tolist())
    t = torch.as_tensor(tokens[selected], dtype=torch.long)
    with torch.no_grad():
        points = model.points_for(t).cpu().numpy()
    flat = points.reshape(-1, points.shape[-1])
    projected = PCA(n_components=3, random_state=config.SEED).fit_transform(flat).reshape(len(selected), tokens.shape[1], 3)
    fig = plt.figure(figsize=(12, 9))
    colors = ["#6c757d", "#e6550d", "#3182bd", "#31a354"]
    for c, name in enumerate(config.CLASSES):
        ax = fig.add_subplot(2, 2, c + 1, projection="3d")
        local = [j for j, original in enumerate(selected) if y[original] == c]
        for j in local[:24]:
            cloud = projected[j]
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=10, alpha=0.65, c=colors[c])
        ax.set_title(f"{name} account-window clouds")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    fig.suptitle("AML typology point clouds (PCA projection of QUANN feature points)")
    fig.tight_layout()
    path = os.path.join(config.FIGURES_DIR, "e0_typology_clouds.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    _log(f"saved figure {path}")


def run_e0(n_transactions=20_000, out_path=None):
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True); os.makedirs(config.FIGURES_DIR, exist_ok=True); os.makedirs(config.LOGS_DIR, exist_ok=True)
    _log("=== [2026-10-16] 1/5: deterministic synthetic transactions/windows ===")
    tx, windows = generate_transactions(n_transactions=n_transactions)
    labels = np.asarray([config.CLASSES.index(x) for x in windows.typology])
    all_ids = np.arange(len(windows))
    train_ids, temp_ids = train_test_split(all_ids, test_size=0.4, random_state=config.SEED, stratify=labels)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=config.SEED + 1, stratify=labels[temp_ids])
    train, val, test = (windows.iloc[idx].reset_index(drop=True) for idx in (train_ids, val_ids, test_ids))
    ytr, yva, yte = (labels[idx] for idx in (train_ids, val_ids, test_ids))
    _log(f"generated {len(tx):,} transactions => {len(windows):,} account windows; split={len(train)}/{len(val)}/{len(test)}; device=cpu")

    _log("=== [2026-10-16] 2/5: fit standard baselines and operating thresholds ===")
    xtr, xva, xte = _feature_matrix(train, val, test)
    standard = _standard_models(xtr, ytr, xva, yva, xte, yte)

    _log("=== [2026-10-16] 3/5: fit feature quantizer on train split ===")
    quantizer = FeatureQuantizer().fit(train)
    ttr, tva, tte = quantizer.transform(train), quantizer.transform(val), quantizer.transform(test)
    _log(f"quantized K={ttr.shape[1]} features, vocab={quantizer.vocab_size}, D=8")

    _log("=== [2026-10-16] 4/5: train QUANN pure/hybrid/backprop on CPU ===")
    configs, models = {}, {}
    for regime in ("pure", "hybrid", "backprop"):
        configs[regime], models[regime] = _fit_cloud(regime, ttr, ytr, tva, yva, tte, yte, 4, quantizer.vocab_size)

    _log("=== [2026-10-16] 5/5: write JSON and 3D typology figure ===")
    _plot_clouds(models["hybrid"], tte, yte)
    out_path = out_path or os.path.join(config.RESULTS_DIR, "e0_smoke.json")
    result = {
        "data": "synthetic_aml_typologies",
        "generator": {"transactions": int(len(tx)), "windows": int(len(windows)), "window_size": 10, "classes": list(config.CLASSES), "seed": config.SEED},
        "standard": standard,
        "configs": configs,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "model": {"D": 8, "d_cloud": 48, "epochs": 1, "K": int(ttr.shape[1]), "vocab": int(quantizer.vocab_size), "device": "cpu"},
        "limitations": ["Sentetik tipolojiler; gerçek bankacılık verisi değildir.", "Enriched dengeli sınıflar nedeniyle lift/base-rate gerçek operasyonu temsil etmez."],
    }
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    _log(f"saved result {out_path}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-transactions", type=int, default=20_000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_e0(args.n_transactions, args.out)
