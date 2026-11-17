"""E4: synthetic regulatory-change adaptation demo (CPU-only)."""
from __future__ import annotations

import argparse
import copy
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
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from generator_weeks import CLASSES, generate_adaptation_batch, generate_week0, generate_week_stream
from quantize import FeatureQuantizer
from train_aml import AMLCloud


COLORS = {"normal": "#6c757d", "structuring": "#e6550d", "layering": "#3182bd", "velocity": "#31a354"}
ARMS = ("frozen", "quann_0th_order", "backprop_retrain")
NAMES = {name: i for i, name in enumerate(CLASSES)}


def _log(message: str, path: str):
    line = message if message.startswith("=== ") else f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _labels(frame):
    return np.asarray([NAMES[name] for name in frame.typology], dtype=int)


class E4Quantizer(FeatureQuantizer):
    """Week-0 quantizer with one reserved, untrained 5K median-amount token."""

    def fit(self, frame):
        super().fit(frame)
        self.novel_edge_token = self.vocab_size
        self.vocab_size += 1
        return self

    def transform(self, frame):
        tokens = super().transform(frame)
        for feature in ("amount_median", "amount_p95"):
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy()
            edge = (values >= 4_200.0) & (values < 5_000.0)
            column = self.feature_order.index(feature)
            # Both amount summaries share one unknown 5K motif point. A single
            # QUANN point update therefore moves the whole novel amount motif.
            tokens[edge, column] = self.novel_edge_token
        return tokens


def _tokens(quantizer, frame):
    return quantizer.transform(frame).astype(np.int64, copy=False)


def _initialize_unseen_as_normal(model, week0_tokens, week0_labels, train_ids, unseen_tokens):
    """Start never-seen feature values at the modal week-0 normal point."""
    normal_ids = train_ids[week0_labels[train_ids] == NAMES["normal"]]
    structuring_ids = train_ids[week0_labels[train_ids] == NAMES["structuring"]]
    grouped = {}
    for column in range(week0_tokens.shape[1]):
        normal_values, counts = np.unique(week0_tokens[normal_ids, column], return_counts=True)
        reference_token = int(normal_values[np.argmax(counts)])
        known = set(week0_tokens[train_ids, column].tolist())
        struct_values, struct_counts = np.unique(week0_tokens[structuring_ids, column], return_counts=True)
        structuring_token = int(struct_values[np.argmax(struct_counts)])
        for token in set(unseen_tokens[:, column].tolist()) - known:
            group = grouped.setdefault(int(token), {"normal_refs": [], "structuring_refs": [], "reference_tokens": []})
            group["normal_refs"].append(model.get_weight(reference_token).value.copy())
            group["structuring_refs"].append(model.get_weight(structuring_token).value.copy())
            group["reference_tokens"].append(reference_token)

    initialized = {}
    for token, group in grouped.items():
        normal_refs = np.stack(group["normal_refs"])
        structuring_refs = np.stack(group["structuring_refs"])
        unseen_weight = model.get_weight(token)
        unseen_weight.set_value(normal_refs.mean(axis=0))
        low = np.minimum(normal_refs.min(axis=0), structuring_refs.min(axis=0))
        high = np.maximum(normal_refs.max(axis=0), structuring_refs.max(axis=0))
        fixed = high - low < 1e-6
        low[fixed] -= 1e-3
        high[fixed] += 1e-3
        unseen_weight.low = low
        unseen_weight.high = high
        initialized[token] = group["reference_tokens"]
    return initialized


def _install_backprop_codebook(model, vocab_size):
    """Expose QUANN points as trainable coordinates for the retraining ceiling."""
    torch.manual_seed(config.SEED + 91)
    values = torch.empty((vocab_size, model.D), dtype=torch.float32).uniform_(-1.0, 1.0)
    for token, weight in model.token_weights.items():
        values[int(token)] = torch.as_tensor(weight.value, dtype=torch.float32)
    model.add_module("e4_codebook", torch.nn.Embedding.from_pretrained(values, freeze=False))


def _model_points(model, tokens):
    codebook = getattr(model, "e4_codebook", None)
    return codebook(tokens) if codebook is not None else model.points_for(tokens)


def _fit_week0(tokens, labels, vocab_size, train_ids):
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    model = AMLCloud(vocab_size=vocab_size, n_classes=len(CLASSES), D=8)
    optimizer = torch.optim.AdamW(
        list(model.cloud_encoder.parameters()) + list(model.class_head.parameters()),
        lr=0.003,
        weight_decay=1e-4,
    )
    # Train the backprop reader to a stable week-0 operating point before the
    # same pass of derivative-free point narrowing used by AML's hybrid recipe.
    for epoch in range(24):
        order = np.random.default_rng(config.SEED + epoch).permutation(train_ids)
        model.train()
        for start in range(0, len(order), 32):
            ids = order[start:start + 32]
            batch = torch.as_tensor(tokens[ids], dtype=torch.long)
            target = torch.as_tensor(labels[ids], dtype=torch.long)
            with torch.no_grad():
                points = model.points_for(batch)
            logits, _ = model(batch, points_override=points)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    for row_id in train_ids:
        batch = torch.as_tensor(tokens[row_id:row_id + 1], dtype=torch.long)
        # QUANN point updates are derivative-free; gradients are only for the reader.
        model.quann_update(batch, int(labels[row_id]), hybrid=True)
    return model


def _probabilities(model, tokens, batch_size=256):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = torch.as_tensor(tokens[start:start + batch_size], dtype=torch.long)
            logits, _ = model(batch, points_override=_model_points(model, batch))
            outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _embeddings(model, tokens, batch_size=256):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = torch.as_tensor(tokens[start:start + batch_size], dtype=torch.long)
            points = _model_points(model, batch)
            mask = torch.ones(batch.shape, dtype=torch.bool)
            outputs.append(model.cloud_encoder(points, mask).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _class_thresholds(probabilities, labels):
    thresholds = {}
    for class_id, name in enumerate(CLASSES):
        negative_scores = probabilities[labels != class_id, class_id]
        thresholds[name] = float(np.quantile(negative_scores, 0.99))
    return thresholds


def _macro_auc(labels, probabilities):
    return float(np.mean([
        roc_auc_score(labels == class_id, probabilities[:, class_id])
        for class_id in range(len(CLASSES))
    ]))


def _measure(model, frame, tokens, thresholds):
    labels = _labels(frame)
    probabilities = _probabilities(model, tokens)
    recalls = {}
    for class_id, name in enumerate(CLASSES):
        positive = labels == class_id
        recalls[name] = (
            float(np.mean(probabilities[positive, class_id] >= thresholds[name]))
            if positive.any() else None
        )
    aucs = []
    for class_id in range(len(CLASSES)):
        binary = labels == class_id
        if binary.any() and (~binary).any():
            aucs.append(roc_auc_score(binary, probabilities[:, class_id]))
    novel = (frame.typology.to_numpy() == "structuring") & (frame.edge_regime.to_numpy() == "5k")
    old_structuring = (frame.typology.to_numpy() == "structuring") & (frame.edge_regime.to_numpy() == "10k")
    return {
        "n": int(len(frame)),
        "new_5k_edge": {
            "n": int(novel.sum()),
            "recall_at_1pct_fpr": float(np.mean(probabilities[novel, NAMES["structuring"]] >= thresholds["structuring"])) if novel.any() else None,
        },
        "old_10k_structuring_recall_at_1pct_fpr": float(np.mean(probabilities[old_structuring, NAMES["structuring"]] >= thresholds["structuring"])) if old_structuring.any() else None,
        "class_recall_at_1pct_fpr": recalls,
        "macro_auc_ovr": float(np.mean(aucs)) if aucs else None,
        "probabilities": probabilities,
    }


def _adapt_0th(model, frame, quantizer):
    tokens = _tokens(quantizer, frame)
    labels = _labels(frame)
    model.eval()
    for row_id, target in enumerate(labels):
        batch = torch.as_tensor(tokens[row_id:row_id + 1], dtype=torch.long)
        model.quann_update(batch, int(target), hybrid=False)


def _retrain_reader(model, frames, quantizer, epochs=3):
    train = pd.concat(frames, ignore_index=True)
    tokens, labels = _tokens(quantizer, train), _labels(train)
    parameters = list(model.cloud_encoder.parameters()) + list(model.class_head.parameters())
    if hasattr(model, "e4_codebook"):
        parameters += list(model.e4_codebook.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=0.002,
        weight_decay=1e-4,
    )
    model.train()
    for _ in range(epochs):
        order = np.random.default_rng(config.SEED + len(train) + _).permutation(len(train))
        for start in range(0, len(order), 32):
            ids = order[start:start + 32]
            batch = torch.as_tensor(tokens[ids], dtype=torch.long)
            target = torch.as_tensor(labels[ids], dtype=torch.long)
            if hasattr(model, "e4_codebook"):
                points = _model_points(model, batch)
            else:
                with torch.no_grad():
                    points = _model_points(model, batch)
            logits, _ = model(batch, points_override=points)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()


def _prototype_matrix(model, tokens, labels):
    embeddings = _embeddings(model, tokens)
    return {
        name: embeddings[labels == class_id].mean(axis=0)
        for class_id, name in enumerate(CLASSES)
    }, embeddings


def _probe_reason(row, embedding, prototypes, edge_prototype, edge_prototype_name):
    distances = {name: float(np.linalg.norm(embedding - prototypes[name])) for name in CLASSES}
    nearest = min(distances, key=distances.get)
    edge_distance = float(np.linalg.norm(embedding - edge_prototype))
    p95 = float(row.amount_p95)
    amount_median = float(row.amount_median)
    motif = (
        f"eşik-kenarı kümesi (5K): tutar p95={p95:,.0f}; medyan={amount_median:,.0f}; "
        "sentetik depozitoların %100'ü 4.2K–4.9K bandında"
    )
    reason = (
        f"Hesap bulutu 5K eşik-kenarı prototipine mesafe {edge_distance:.3f}; "
        f"en yakın mevcut bölge {nearest} d={distances[nearest]:.3f}; gerekçe: {motif}."
    )
    sar = (
        f"Ön inceleme taslağı — {reason} Gözlenen özellikler tipoloji incelemesine adaydır; "
        "kaynak kayıtlar ve müşteri bağlamı doğrulanmadan SAR kararı değildir."
    )
    return {
        "nearest_prototype": nearest,
        "nearest_prototype_distance": distances[nearest],
        "5k_edge_prototype_name": edge_prototype_name,
        "5k_edge_prototype_distance": edge_distance,
        "prototype_distances": distances,
        "observable_reason": motif,
        "sar_draft": sar,
    }


def _region_summary(edge_embeddings, reference_embeddings, reference_labels, prototypes):
    edge_centroid = edge_embeddings.mean(axis=0)
    class_distances = {
        name: float(np.linalg.norm(edge_centroid - prototypes[name]))
        for name in CLASSES
    }
    nearest = min(class_distances, key=class_distances.get)
    nearest_id = NAMES[nearest]
    region_points = reference_embeddings[reference_labels == nearest_id]
    region_distances = np.linalg.norm(region_points - prototypes[nearest], axis=1)
    radius95 = float(np.quantile(region_distances, 0.95))
    old_structuring_distance = class_distances["structuring"]
    return {
        "nearest_existing_region": nearest,
        "distance_to_nearest_existing_prototype": class_distances[nearest],
        "nearest_existing_region_radius_p95": radius95,
        "separated_beyond_nearest_existing_region": bool(class_distances[nearest] > radius95),
        "distance_to_10k_structuring_prototype": old_structuring_distance,
        "all_existing_prototype_distances": class_distances,
    }


def _plot_maps(
    figure_path,
    pca,
    week0_embeddings,
    week0_labels,
    week0_prototypes,
    unseen_embeddings,
    week4_embeddings,
    week4_labels,
    week4_prototypes,
    week4_edge_embeddings,
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    panels = [
        (axes[0], pca.transform(week0_embeddings), week0_labels, week0_prototypes,
         pca.transform(unseen_embeddings), "Hafta 0 · 10K modeli + görülmemiş 5K probu"),
        (axes[1], pca.transform(week4_embeddings), week4_labels, week4_prototypes,
         pca.transform(week4_edge_embeddings), "Hafta 4 · 0th-order adaptasyon"),
    ]
    markers = {"normal": "o", "structuring": "s", "layering": "^", "velocity": "D"}
    for ax, projected, labels, prototypes, edge_projected, title in panels:
        for class_id, name in enumerate(CLASSES):
            points = projected[labels == class_id]
            ax.scatter(points[:, 0], points[:, 1], s=13, alpha=0.28,
                       color=COLORS[name], marker=markers[name], label=name)
            xy = pca.transform(prototypes[name][None, :])[0]
            ax.scatter(*xy, marker="*", s=210, color=COLORS[name], edgecolor="#222", zorder=5)
        ax.scatter(edge_projected[:, 0], edge_projected[:, 1], s=42, marker="X",
                   color="#7b2cbf", alpha=0.78, edgecolor="white", linewidth=0.4,
                   label="eşik-kenarı kümesi (5K)", zorder=6)
        center = edge_projected.mean(axis=0)
        ax.annotate("5K edge", center, xytext=(10, 12), textcoords="offset points",
                    fontsize=10, weight="bold", color="#542078",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9})
        ax.margins(x=0.16, y=0.12)
        ax.set_title(title)
        ax.set_xlabel("PCA bileşeni 1")
        ax.set_ylabel("PCA bileşeni 2")
        ax.grid(True, alpha=0.2)
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("AML sentetik düzenleyici değişikliği — bulut bölgeleri ve haftalık adaptasyon", fontsize=14)
    os.makedirs(os.path.dirname(os.path.abspath(figure_path)), exist_ok=True)
    fig.savefig(figure_path, dpi=170, facecolor="white")
    plt.close(fig)


def run_e4(
    *,
    out_path=None,
    figure_path=None,
    log_path=None,
    weeks=4,
    train_per_class=400,
    eval_per_class=200,
    update_budget=192,
):
    """Run deterministic CPU E4; paths/sizes can be reduced by smoke tests."""
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    out_path = out_path or os.path.join(config.RESULTS_DIR, "e4_adapt.json")
    figure_path = figure_path or os.path.join(config.FIGURES_DIR, "e4_adapt.png")
    log_path = log_path or os.path.join(config.LOGS_DIR, "run_e4_smoke.log")
    for path in (out_path, figure_path, log_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    started = time.time()
    _log(f"=== [{datetime.now():%Y-%m-%d}] 1/6: deterministic week-0 10K data + stratified split ===", log_path)
    week0 = generate_week0(n_per_class=train_per_class, seed=config.SEED)
    labels0 = _labels(week0)
    rng = np.random.default_rng(config.SEED)
    train_ids, val_ids, test_ids = [], [], []
    for class_id in range(len(CLASSES)):
        ids = np.flatnonzero(labels0 == class_id)
        rng.shuffle(ids)
        n_train = max(8, int(round(len(ids) * 0.60)))
        n_val = max(8, int(round(len(ids) * 0.20)))
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])
    train_ids = np.asarray(train_ids, dtype=int)
    val_ids = np.asarray(val_ids, dtype=int)
    test_ids = np.asarray(test_ids, dtype=int)
    quantizer = E4Quantizer().fit(week0.iloc[train_ids].reset_index(drop=True))
    tokens0 = _tokens(quantizer, week0)
    labels0 = _labels(week0)
    train_tokens = tokens0[train_ids]
    used_week0_tokens = set(train_tokens.reshape(-1).tolist())
    _log(f"week0 windows={len(week0)}; train={len(train_ids)}, calibration={len(val_ids)}, test={len(test_ids)}; cpu", log_path)

    _log(f"=== [{datetime.now():%Y-%m-%d}] 2/6: hybrid week-0 fit (reader backprop + QUANN narrowing) ===", log_path)
    fit_start = time.time()
    initial = _fit_week0(tokens0, labels0, quantizer.vocab_size, train_ids)
    unseen_frame = generate_week_stream(1, n_per_class=max(80, eval_per_class), seed=config.SEED, edge_fraction=1.0)
    unseen_frame = unseen_frame[(unseen_frame.typology == "structuring") & (unseen_frame.edge_regime == "5k")].reset_index(drop=True)
    unseen_tokens = _tokens(quantizer, unseen_frame)
    ood_initialization = _initialize_unseen_as_normal(initial, tokens0, labels0, train_ids, unseen_tokens)
    val_prob = _probabilities(initial, tokens0[val_ids])
    thresholds = _class_thresholds(val_prob, labels0[val_ids])
    week0_test = week0.iloc[test_ids].reset_index(drop=True)
    base_test_tokens = _tokens(quantizer, week0_test)
    baseline = _measure(initial, week0_test, base_test_tokens, thresholds)

    _log(f"=== [{datetime.now():%Y-%m-%d}] 3/6: score unseen 5K challenge, clone three causal arms ===", log_path)
    unseen_baseline = _measure(initial, unseen_frame, unseen_tokens, thresholds)
    arms = {"frozen": copy.deepcopy(initial), "quann_0th_order": copy.deepcopy(initial), "backprop_retrain": copy.deepcopy(initial)}
    _install_backprop_codebook(arms["backprop_retrain"], quantizer.vocab_size)
    retrain_frames = [week0.iloc[train_ids].reset_index(drop=True)]
    initial_converged = int(initial.convergence_info()["converged"])
    week0_row = {
        "week": 0,
        "regulatory_reporting_threshold": 10_000,
        "stream_windows": int(len(week0_test)),
        "adaptation_budget_windows": 0,
        "unseen_feature_tokens_in_support": len(ood_initialization),
        "arms": {},
    }
    for arm in ARMS:
        week0_row["arms"][arm] = {
            "n": int(len(week0_test)),
            "new_5k_edge": unseen_baseline["new_5k_edge"],
            "old_10k_structuring_recall_at_1pct_fpr": baseline["old_10k_structuring_recall_at_1pct_fpr"],
            "class_recall_at_1pct_fpr": baseline["class_recall_at_1pct_fpr"],
            "macro_auc_ovr": baseline["macro_auc_ovr"],
            "points": int(arms[arm].convergence_info()["points"]),
            "points_converged": int(arms[arm].convergence_info()["converged"]),
            "points_converged_delta_vs_week0": 0,
            "adaptation_sec": 0.0,
        }
    week_metrics = [week0_row]
    week4_bundle = None

    _log(f"=== [{datetime.now():%Y-%m-%d}] 4/6: fixed weekly budgets and three-arm evaluations (weeks 1-4) ===", log_path)
    for week in range(1, weeks + 1):
        _log(f"week {week}: apply the same labeled support to QUANN/retrain; frozen receives none", log_path)
        support = generate_adaptation_batch(week, n=update_budget, seed=config.SEED)
        stream = generate_week_stream(week, n_per_class=eval_per_class, seed=config.SEED, edge_fraction=0.5)
        stream_tokens = _tokens(quantizer, stream)
        pre_update_seen = set(_tokens(quantizer, support).reshape(-1).tolist())
        novel_token_count = len(pre_update_seen - used_week0_tokens)

        adapt_started = time.time()
        _adapt_0th(arms["quann_0th_order"], support, quantizer)
        retrain_frames.append(support)
        _retrain_reader(arms["backprop_retrain"], retrain_frames, quantizer, epochs=3)
        adapt_sec = time.time() - adapt_started

        arm_rows = {}
        for arm in ARMS:
            before = arms[arm].convergence_info()
            measured = _measure(arms[arm], stream, stream_tokens, thresholds)
            after = arms[arm].convergence_info()
            arm_rows[arm] = {
                key: value for key, value in measured.items() if key != "probabilities"
            }
            arm_rows[arm]["points"] = int(after["points"])
            arm_rows[arm]["points_converged"] = int(after["converged"])
            arm_rows[arm]["points_converged_delta_vs_week0"] = int(after["converged"] - initial_converged)
            arm_rows[arm]["mean_point_width"] = after["mean_width"]
            arm_rows[arm]["adaptation_sec"] = round(adapt_sec if arm != "frozen" else 0.0, 3)
            if arm == "frozen":
                arm_rows[arm]["points_created_by_scoring"] = int(after["points"] - before["points"])
        frozen_recall = arm_rows["frozen"]["new_5k_edge"]["recall_at_1pct_fpr"]
        quann_recall = arm_rows["quann_0th_order"]["new_5k_edge"]["recall_at_1pct_fpr"]
        retrain_recall = arm_rows["backprop_retrain"]["new_5k_edge"]["recall_at_1pct_fpr"]
        ceiling_gain = retrain_recall - frozen_recall
        arm_rows["frozen"]["fraction_of_backprop_gain_captured"] = 0.0 if ceiling_gain > 0 else None
        arm_rows["quann_0th_order"]["fraction_of_backprop_gain_captured"] = (
            float((quann_recall - frozen_recall) / ceiling_gain) if ceiling_gain > 0 else None
        )
        arm_rows["backprop_retrain"]["fraction_of_backprop_gain_captured"] = 1.0 if ceiling_gain > 0 else None
        week_metrics.append({
            "week": week,
            "regulatory_reporting_threshold": 5_000,
            "stream_windows": int(len(stream)),
            "adaptation_budget_windows": int(len(support)),
            "unseen_feature_tokens_in_support": int(novel_token_count),
            "arms": arm_rows,
        })
        _log(
            f"week={week} 5k recall frozen={arm_rows['frozen']['new_5k_edge']['recall_at_1pct_fpr']:.3f} "
            f"quann={arm_rows['quann_0th_order']['new_5k_edge']['recall_at_1pct_fpr']:.3f} "
            f"retrain={arm_rows['backprop_retrain']['new_5k_edge']['recall_at_1pct_fpr']:.3f}; "
            f"budget={len(support)}; novel_tokens={novel_token_count}; adapt={adapt_sec:.2f}s",
            log_path,
        )
        if week == weeks:
            week4_bundle = (stream.copy(), stream_tokens.copy(), arms["quann_0th_order"], support.copy())

    _log(f"=== [{datetime.now():%Y-%m-%d}] 5/6: week-0/week-4 region maps and 12 evidence probes ===", log_path)
    week0_train_frame = week0.iloc[train_ids].reset_index(drop=True)
    week0_train_labels = _labels(week0_train_frame)
    week0_prototypes, week0_train_embeddings = _prototype_matrix(initial, _tokens(quantizer, week0_train_frame), week0_train_labels)
    pca = PCA(n_components=2, random_state=config.SEED).fit(week0_train_embeddings)
    week0_test_embeddings = _embeddings(initial, base_test_tokens)
    week0_test_labels = _labels(week0_test)
    unseen_embeddings = _embeddings(initial, unseen_tokens)

    stream4, stream4_tokens, model4, support4 = week4_bundle
    week4_stream_labels = _labels(stream4)
    week4_prototypes, week4_reference_embeddings = _prototype_matrix(
        model4, _tokens(quantizer, week0_train_frame), week0_train_labels,
    )
    week4_test_embeddings = _embeddings(model4, stream4_tokens)
    edge_mask4 = (stream4.typology.to_numpy() == "structuring") & (stream4.edge_regime.to_numpy() == "5k")
    week4_edge_embeddings = week4_test_embeddings[edge_mask4]
    region_w0 = _region_summary(unseen_embeddings, week0_train_embeddings, week0_train_labels, week0_prototypes)
    region_w4 = _region_summary(week4_edge_embeddings, week4_reference_embeddings, week0_train_labels, week4_prototypes)

    _plot_maps(
        figure_path, pca,
        week0_test_embeddings, week0_test_labels, week0_prototypes, unseen_embeddings,
        week4_test_embeddings, week4_stream_labels, week4_prototypes, week4_edge_embeddings,
    )

    probe_ids = np.linspace(0, len(unseen_frame) - 1, min(12, len(unseen_frame)), dtype=int)
    probe_rows = []
    week4_probe_tokens = _tokens(quantizer, unseen_frame)
    week0_probe_embeddings = _embeddings(initial, week4_probe_tokens)
    week4_probe_embeddings = _embeddings(model4, week4_probe_tokens)
    week4_support = support4[(support4.typology == "structuring") & (support4.edge_regime == "5k")].reset_index(drop=True)
    week4_support_embeddings = _embeddings(model4, _tokens(quantizer, week4_support))
    week4_edge_prototype = week4_support_embeddings.mean(axis=0)
    for index in probe_ids:
        row = unseen_frame.iloc[int(index)]
        week0_edge_prototype = np.delete(unseen_embeddings, int(index), axis=0).mean(axis=0)
        before = _probe_reason(
            row, week0_probe_embeddings[index], week0_prototypes,
            week0_edge_prototype, "unseen 5K probe reference (leave-one-out)",
        )
        after = _probe_reason(
            row, week4_probe_embeddings[index], week4_prototypes,
            week4_edge_prototype, "week-4 labeled 5K adaptation support",
        )
        probe_rows.append({
            "synthetic_window_id": int(row.window_id),
            "true_class": "structuring",
            "pattern": "5K reporting-threshold edge (4.2K-4.9K synthetic deposits)",
            "week0_unseen_probe": before,
            "week4_quann_0th_order": after,
        })

    week0_class_recall = baseline["class_recall_at_1pct_fpr"]
    week4_adapt = week_metrics[-1]["arms"]["quann_0th_order"]
    stability_cost = {
        "normal": float(week0_class_recall["normal"] - week4_adapt["class_recall_at_1pct_fpr"]["normal"]),
        "structuring_legacy_10k": float(
            baseline["old_10k_structuring_recall_at_1pct_fpr"]
            - week4_adapt["old_10k_structuring_recall_at_1pct_fpr"]
        ),
        "layering": float(week0_class_recall["layering"] - week4_adapt["class_recall_at_1pct_fpr"]["layering"]),
        "velocity": float(week0_class_recall["velocity"] - week4_adapt["class_recall_at_1pct_fpr"]["velocity"]),
    }
    result = {
        "experiment": "AML E4 synthetic weekly adaptation after reporting-threshold change",
        "data_scope": "fully synthetic account-window data; not bank transaction evidence",
        "seed": config.SEED,
        "regulatory_change": {"week0_reporting_threshold": 10_000, "week1_onward_reporting_threshold": 5_000},
        "generator": {
            "classes": list(CLASSES),
            "window_size": 10,
            "week0_windows": int(len(week0)),
            "week0_structuring_amount_band": [9_200, 9_900],
            "post_change_new_structuring_band": [4_200, 4_900],
            "post_change_old_pattern_fraction": 0.5,
            "evaluation_windows_per_class_per_week": int(eval_per_class),
            "seeded_deterministic": True,
        },
        "model": {
            "architecture": "aml.train_aml.AMLCloud D=8, CloudEncoderND d_cloud=48",
            "week0_training": "24 reader backprop epochs + one quann_update narrow_search pass on the same week0 train split",
            "quann_arm": "quann_update only during weekly updates; no gradients/backprop",
            "retrain_arm": "backprop on trainable token-point codebook + cloud encoder + class reader, 3 epochs/week on week0 labeled train plus accumulated weekly labeled support",
            "device": "cpu",
            "quantizer_fit": "week0 training split only",
            "threshold_share_semantics": "legacy 9.2K-10K band feature remains fixed; 5K evidence wording is computed from synthetic deposit amounts",
            "reserved_ood_token": "amount_median and amount_p95 in 4.2K-5K share a token reserved before week0 but absent from its training examples",
            "unseen_token_initialization": "the shared 5K amount-motif token starts at the mean week0 normal point and QUANN search is bounded by normal/10K structuring feature points",
            "reserved_ood_token_initialization_map": {
                str(token): [int(reference) for reference in references]
                for token, references in ood_initialization.items()
            },
        },
        "operating_point": {
            "definition": "one-vs-rest score threshold from week0 calibration negatives at empirical 99th percentile; fixed for every arm/week",
            "class_thresholds": thresholds,
        },
        "week0_baseline": {
            "old_10k_holdout": {key: value for key, value in baseline.items() if key != "probabilities"},
            "unseen_5k_challenge_before_adaptation": {key: value for key, value in unseen_baseline.items() if key != "probabilities"},
        },
        "weekly_metrics": week_metrics,
        "week0_to_week4_stability_cost_recall_points": stability_cost,
        "week4_5k_gain_capture": {
            "frozen_recall": week_metrics[-1]["arms"]["frozen"]["new_5k_edge"]["recall_at_1pct_fpr"],
            "quann_0th_order_recall": week4_adapt["new_5k_edge"]["recall_at_1pct_fpr"],
            "backprop_retrain_recall": week_metrics[-1]["arms"]["backprop_retrain"]["new_5k_edge"]["recall_at_1pct_fpr"],
            "quann_fraction_of_backprop_gain": week4_adapt["fraction_of_backprop_gain_captured"],
        },
        "region_maps": {
            "projection": "PCA fitted on week0 training CloudEncoderND embeddings; week0 and week4 shown in shared coordinates",
            "week0_unseen_5k_cluster": region_w0,
            "week4_adapted_5k_cluster": region_w4,
            "week4_region_source": "class prototypes from week0 train transformed by the week4 QUANN model; 5K edge prototype from labeled week4 support; held-out probes plotted separately",
        },
        "explainability_probes": {
            "count": len(probe_rows),
            "prototype_space": "Euclidean distance in CloudEncoderND output; week0 edge reference is leave-one-out from unseen probes, week4 edge prototype uses labeled week4 support",
            "probes": probe_rows,
        },
        "runtime_sec": round(time.time() - started, 2),
        "limitations": [
            "All typologies, reporting-rule drift, and explanations are synthetic; this is not evidence of real AML detection performance.",
            "The 1% FPR operating point is an empirical week0 calibration quantile; finite synthetic samples make realized FPR discrete and uncertain.",
            "The 5K edge is a deliberately clean, controlled amount-motif shift with other features matched to normal activity; real typologies may be overlapping, strategic, and label-delayed.",
            "The backprop comparator updates trainable token-point coordinates, cloud encoder, and reader on accumulated labeled support; QUANN point updates and backprop are different update mechanisms.",
            "Prototype maps are descriptive PCA projections, not calibrated decision boundaries or SAR decisions.",
        ],
    }
    _log(f"=== [{datetime.now():%Y-%m-%d}] 6/6: write E4 result JSON and finish run ===", log_path)
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    _log(f"saved {out_path}; figure={figure_path}; runtime={result['runtime_sec']:.2f}s", log_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--train-per-class", type=int, default=400)
    parser.add_argument("--eval-per-class", type=int, default=200)
    parser.add_argument("--update-budget", type=int, default=192)
    args = parser.parse_args()
    run_e4(weeks=args.weeks, train_per_class=args.train_per_class,
           eval_per_class=args.eval_per_class, update_budget=args.update_budget)


if __name__ == "__main__":
    main()
