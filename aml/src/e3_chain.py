"""E3: leakage-safe chain features for AML layering detection (CPU only)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from generator import generate_transactions
from quantize import FeatureQuantizer
from train_aml import AMLCloud, _metrics


BASE_SPECS = {
    name: spec for name, spec in FeatureQuantizer.SPECS.items()
    if name != "counterparty_id"
}
CHAIN_SPECS = {
    "observed_hop_count": ("quantile", 6),
    "hop_completion_ratio": ("quantile", 6),
    "counterparty_shared_ratio": ("quantile", 6),
    "max_counterparty_degree": ("quantile", 6),
    "counterparty_transition_rate": ("quantile", 6),
    "dwell_time_cv": ("quantile", 8),
    "amount_retention_median": ("quantile", 8),
    "amount_retention_p10": ("quantile", 8),
}
COLORS = {
    "normal": "#6c757d", "structuring": "#e6550d",
    "layering": "#3182bd", "velocity": "#31a354",
}
MARKERS = {"normal": "o", "structuring": "s", "layering": "^", "velocity": "D"}
LINE_STYLES = {"normal": ":", "structuring": "-", "layering": "--", "velocity": "-."}


class E3Quantizer(FeatureQuantizer):
    """E0 quantization with raw IDs removed and optional chain summaries added."""

    def __init__(self, include_chain: bool):
        self.SPECS = {**BASE_SPECS, **(CHAIN_SPECS if include_chain else {})}
        super().__init__()


def _log(message: str, log_path: str):
    line = message if message.startswith("=== ") else (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    )
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def add_chain_features(transactions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sequential, account-window chain proxies; never return raw IDs."""
    groups = {int(key): group for key, group in transactions.groupby("window_id", sort=False)}
    rows = []
    for window_id in windows.window_id.astype(int):
        group = groups[int(window_id)]
        counterparties = group.counterparty_id.astype(str).to_numpy()
        amounts = group.amount.to_numpy(dtype=float)
        gaps = group.inter_tx_interval_min.to_numpy(dtype=float)[1:]
        unique_count = len(set(counterparties))
        degree_counts = pd.Series(counterparties).value_counts().to_numpy()
        adjacent_amounts = np.minimum(amounts[:-1], amounts[1:]) / np.maximum(amounts[:-1], amounts[1:]).clip(1e-12)
        rows.append({
            "window_id": int(window_id),
            "observed_hop_count": float(unique_count),
            "hop_completion_ratio": float(unique_count / max(len(counterparties), 1)),
            "counterparty_shared_ratio": float(1.0 - unique_count / max(len(counterparties), 1)),
            # Direction is absent in the synthetic event schema; degree is undirected.
            "max_counterparty_degree": float(degree_counts.max(initial=0)),
            "counterparty_transition_rate": float(np.mean(counterparties[1:] != counterparties[:-1])) if len(counterparties) > 1 else 0.0,
            "dwell_time_cv": float(np.std(gaps) / max(float(np.mean(gaps)), 1e-12)) if len(gaps) else 0.0,
            # Ratio of adjacent transfer amounts is a conservation proxy, not a ledger balance.
            "amount_retention_median": float(np.median(adjacent_amounts)) if len(adjacent_amounts) else 1.0,
            "amount_retention_p10": float(np.quantile(adjacent_amounts, 0.10)) if len(adjacent_amounts) else 1.0,
        })
    engineered = pd.DataFrame(rows)
    result = windows.merge(engineered, on="window_id", how="left", validate="one_to_one", sort=False)
    return result


def _matrix(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    categories = [column for column in columns if column == "channel"]
    numeric = [column for column in columns if column not in categories]
    xtr = pd.get_dummies(train[numeric + categories], columns=categories, dtype=float)
    xva = pd.get_dummies(val[numeric + categories], columns=categories, dtype=float).reindex(columns=xtr.columns, fill_value=0)
    xte = pd.get_dummies(test[numeric + categories], columns=categories, dtype=float).reindex(columns=xtr.columns, fill_value=0)
    return tuple(frame.to_numpy(dtype=np.float32) for frame in (xtr, xva, xte))


def _score(yval, ytest, pval, ptest):
    val_metrics = _metrics(yval, pval)
    test_metrics = _metrics(
        ytest, ptest, val_metrics["threshold"], val_metrics["class_thresholds_at_1pct_fpr"],
    )
    test_metrics["accuracy"] = float(accuracy_score(ytest, ptest.argmax(axis=1)))
    return {"test": test_metrics, "val": val_metrics}


def _fit_gbt(xtr, ytr, xva, yva, xte, yte):
    started = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=15, learning_rate=0.08, random_state=config.SEED,
    )
    model.fit(xtr, ytr)
    result = _score(yva, yte, model.predict_proba(xva), model.predict_proba(xte))
    result["test"]["fit_sec"] = round(time.time() - started, 2)
    return result


def _fit_cloud(regime: str, tokens, ytr, val_tokens, yval, test_tokens, ytest, vocab_size: int):
    torch.manual_seed(config.SEED)
    model = AMLCloud(vocab_size=vocab_size, n_classes=len(config.CLASSES), D=8)
    optimizer = torch.optim.AdamW(
        model.parameters() if regime == "backprop" else list(model.cloud_encoder.parameters()) + list(model.class_head.parameters()),
        lr=0.003, weight_decay=1e-4,
    )
    rng = np.random.default_rng(config.SEED)
    train_ids = np.arange(len(ytr))
    if len(train_ids) > 480:
        selected = []
        for class_id in np.unique(ytr):
            class_rows = train_ids[ytr == class_id]
            selected.extend(rng.choice(class_rows, min(len(class_rows), 120), replace=False).tolist())
        train_ids = np.asarray(selected, dtype=int)
        rng.shuffle(train_ids)

    model.train()
    started = time.time()
    for row in train_ids:
        token_batch = torch.as_tensor(tokens[row:row + 1], dtype=torch.long)
        if regime == "hybrid":
            points = model.points_for(token_batch).detach()
            logits, _ = model(token_batch, points_override=points)
        else:
            logits, _ = model(token_batch)
        loss = F.cross_entropy(logits, torch.tensor([int(ytr[row])]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if regime == "hybrid":
            model.eval()
            model.quann_update(token_batch, int(ytr[row]), hybrid=True)
            model.train()
    fit_sec = time.time() - started

    def predict(matrix):
        model.eval()
        batches = []
        with torch.no_grad():
            for start in range(0, len(matrix), 256):
                tokens_batch = torch.as_tensor(matrix[start:start + 256], dtype=torch.long)
                points = model.points_for(tokens_batch) if regime == "hybrid" else None
                logits, _ = model(tokens_batch, points_override=points)
                batches.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(batches, axis=0)

    result = _score(yval, ytest, predict(val_tokens), predict(test_tokens))
    result["test"].update({"regime": regime, "train_sec": round(fit_sec, 2), **model.convergence_info()})
    return result, model


def _embeddings(model: AMLCloud, tokens: np.ndarray):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), 128):
            batch = torch.as_tensor(tokens[start:start + 128], dtype=torch.long)
            points = model.points_for(batch)
            outputs.append(model.cloud_encoder(points, torch.ones(batch.shape, dtype=torch.bool)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _plot_region_map(embeddings, labels, prototypes, figure_path):
    pca = PCA(n_components=2, random_state=config.SEED)
    projected = pca.fit_transform(embeddings)
    prototype_points = pca.transform(np.stack([prototypes[name] for name in config.CLASSES]))
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    for class_id, name in enumerate(config.CLASSES):
        points = projected[labels == class_id]
        bandwidth = max(0.015, min(0.3, float(np.median(np.std(points, axis=0)) * len(points) ** (-1 / 6))))
        lower, upper = np.quantile(points, [0.01, 0.99], axis=0)
        padding = max(4 * bandwidth, 0.04)
        gx, gy = np.meshgrid(
            np.linspace(lower[0] - padding, upper[0] + padding, 160),
            np.linspace(lower[1] - padding, upper[1] + padding, 160),
        )
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        density = np.exp(KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(points).score_samples(grid)).reshape(gx.shape)
        peak = float(density.max())
        ax.contourf(gx, gy, density, levels=peak * np.asarray([0.08, 0.2, 0.4, 1.01]), colors=[COLORS[name]], alpha=0.055)
        ax.contour(gx, gy, density, levels=peak * np.asarray([0.12, 0.3, 0.65]), colors=[COLORS[name]],
                   linewidths=[0.8, 1.1, 1.6], linestyles=LINE_STYLES[name], alpha=0.9)
        ax.scatter(points[:, 0], points[:, 1], s=42, marker=MARKERS[name], facecolors="none",
                   edgecolors=COLORS[name], alpha=0.78, linewidths=0.9, zorder=4)
    for class_id, name in enumerate(config.CLASSES):
        px, py = prototype_points[class_id]
        ax.scatter(px, py, marker="*", s=280, color=COLORS[name], edgecolor="#202020", linewidth=0.8, zorder=6)
        label_offsets = {"normal": (12, 14), "structuring": (18, -22), "layering": (10, 10), "velocity": (18, 0)}
        ax.annotate(f"{name} prototype", (px, py), xytext=label_offsets[name], textcoords="offset points",
                    fontsize=9, weight="bold", bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.84, "edgecolor": "none"})
    handles = [Line2D([0], [0], marker=MARKERS[n], color="none", markerfacecolor="none",
                      markeredgecolor=COLORS[n], markersize=8, label=n) for n in config.CLASSES]
    handles += [Line2D([0], [0], marker="*", color="#202020", linestyle="none", markersize=12, label="train prototype"),
                Patch(facecolor="#999999", alpha=0.14, edgecolor="#666666", label="KDE density region")]
    variance = 100 * float(pca.explained_variance_ratio_.sum())
    ax.set_title(f"AML account-cloud regions — E3 hybrid, base + chain features\n{len(embeddings):,} synthetic windows · PCA-2D ({variance:.1f}% variance) · KDE regions, not decision boundaries", fontsize=14, pad=14)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(True, color="#d9dee5", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=handles, loc="best", frameon=True, framealpha=0.95, ncol=2)
    os.makedirs(os.path.dirname(os.path.abspath(figure_path)), exist_ok=True)
    fig.savefig(figure_path, dpi=180, facecolor="white")
    plt.close(fig)
    return projected, prototype_points, pca.explained_variance_ratio_


def _prototype_summary(embeddings, labels, train_ids, test_ids):
    prototypes = {
        name: embeddings[train_ids][labels[train_ids] == class_id].mean(axis=0)
        for class_id, name in enumerate(config.CLASSES)
    }
    matrix = np.stack([prototypes[name] for name in config.CLASSES])
    test_distances = np.linalg.norm(embeddings[test_ids, None, :] - matrix[None, :, :], axis=2)
    summary = {}
    for class_id, name in enumerate(config.CLASSES):
        rows = np.flatnonzero(labels[test_ids] == class_id)
        own = test_distances[rows, class_id]
        other = np.delete(test_distances[rows], class_id, axis=1).min(axis=1)
        summary[name] = {
            "n_test": int(len(rows)),
            "median_own_distance": float(np.median(own)),
            "median_nearest_other_distance": float(np.median(other)),
            "own_to_other_ratio": float(np.median(own) / max(float(np.median(other)), 1e-12)),
            "nearest_prototype_accuracy": float(np.mean(test_distances[rows].argmin(axis=1) == class_id)),
        }
    return prototypes, summary


def run_e3(n_transactions=20_000, out_path=None, figure_path=None, log_path=None):
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    out_path = out_path or os.path.join(config.RESULTS_DIR, "e3_chain.json")
    figure_path = figure_path or os.path.join(config.FIGURES_DIR, "e3_region_map.png")
    log_path = log_path or os.path.join(config.LOGS_DIR, "run_e3_smoke.log")
    for path in (out_path, figure_path, log_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    _log(f"=== [{datetime.now().strftime('%Y-%m-%d')}] 1/5: regenerate deterministic E0 transactions/windows and group split ===", log_path)
    transactions, original = generate_transactions(n_transactions=n_transactions, seed=config.SEED)
    windows = add_chain_features(transactions, original)
    labels = np.asarray([config.CLASSES.index(name) for name in windows.typology], dtype=int)
    all_ids = np.arange(len(windows))
    train_ids, temp_ids = train_test_split(all_ids, test_size=0.4, random_state=config.SEED, stratify=labels)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=config.SEED + 1, stratify=labels[temp_ids])
    ytr, yva, yte = labels[train_ids], labels[val_ids], labels[test_ids]
    split = {"train": int(len(train_ids)), "val": int(len(val_ids)), "test": int(len(test_ids))}
    _log(f"generated {len(transactions):,} transactions => {len(windows):,} windows; split={split['train']}/{split['val']}/{split['test']}; CPU", log_path)

    _log(f"=== [{datetime.now().strftime('%Y-%m-%d')}] 2/5: leakage-safe base vs base+chain features; fit GBT ===", log_path)
    variants = {"base": list(BASE_SPECS), "base+chain": list(BASE_SPECS) + list(CHAIN_SPECS)}
    results = {"gbt": {}}
    matrices = {}
    quantizers = {}
    tokens = {}
    for variant, features in variants.items():
        train, val, test = (windows.iloc[ids].reset_index(drop=True) for ids in (train_ids, val_ids, test_ids))
        xtr, xva, xte = _matrix(train, val, test, features)
        results["gbt"][variant] = _fit_gbt(xtr, ytr, xva, yva, xte, yte)
        quantizer = E3Quantizer(include_chain=variant == "base+chain").fit(train)
        quantizers[variant] = quantizer
        tokens[variant] = tuple(quantizer.transform(frame) for frame in (train, val, test))
        matrices[variant] = (xtr, xva, xte)
        _log(f"GBT/{variant} test roc={results['gbt'][variant]['test']['auc_roc']:.4f} pr={results['gbt'][variant]['test']['auc_pr']:.4f} F2={results['gbt'][variant]['test']['f2']:.4f} layering-recall={results['gbt'][variant]['test']['class_recall_at_1pct_fpr']['layering']:.3f}", log_path)

    _log(f"=== [{datetime.now().strftime('%Y-%m-%d')}] 3/5: train hybrid and backprop cloud regimes on CPU ===", log_path)
    cloud_models = {}
    for regime in ("hybrid", "backprop"):
        results[regime] = {}
        for variant in variants:
            ttr, tva, tte = tokens[variant]
            model_result, model = _fit_cloud(regime, ttr, ytr, tva, yva, tte, yte, quantizers[variant].vocab_size)
            results[regime][variant] = model_result
            if regime == "hybrid":
                cloud_models[variant] = (model, ttr, tva, tte)
            test_result = model_result["test"]
            _log(f"{regime}/{variant} test roc={test_result['auc_roc']:.4f} pr={test_result['auc_pr']:.4f} F2={test_result['f2']:.4f} layering-recall={test_result['class_recall_at_1pct_fpr']['layering']:.3f} ({test_result['train_sec']:.1f}s)", log_path)

    _log(f"=== [{datetime.now().strftime('%Y-%m-%d')}] 4/5: E1-style train prototypes and PCA-2D chain region map ===", log_path)
    model, _, _, _ = cloud_models["base+chain"]
    all_tokens = quantizers["base+chain"].transform(windows)
    embeddings = _embeddings(model, all_tokens)
    prototypes, prototype_summary = _prototype_summary(embeddings, labels, train_ids, test_ids)
    base_model, _, _, _ = cloud_models["base"]
    base_embeddings = _embeddings(base_model, quantizers["base"].transform(windows))
    _, base_prototype_summary = _prototype_summary(base_embeddings, labels, train_ids, test_ids)
    projected, projected_prototypes, variance = _plot_region_map(embeddings, labels, prototypes, figure_path)
    layering_id = config.CLASSES.index("layering")
    normal_id = config.CLASSES.index("normal")
    pca_centroid_gap = float(np.linalg.norm(projected_prototypes[layering_id] - projected_prototypes[normal_id]))
    pca_scale = float(max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]), 1e-12))
    layering_region_separation = pca_centroid_gap / pca_scale
    _log(f"saved E3 region map {figure_path}; layering/normal PCA centroid gap={pca_centroid_gap:.4f} ({layering_region_separation:.3%} of map span)", log_path)

    _log(f"=== [{datetime.now().strftime('%Y-%m-%d')}] 5/5: compute ablation deltas and write E3 JSON ===", log_path)
    comparisons = {}
    for regime in ("gbt", "hybrid", "backprop"):
        base = results[regime]["base"]["test"]
        chain = results[regime]["base+chain"]["test"]
        comparisons[regime] = {
            "layering_recall_delta": float(chain["class_recall_at_1pct_fpr"]["layering"] - base["class_recall_at_1pct_fpr"]["layering"]),
            "auc_roc_delta": float(chain["auc_roc"] - base["auc_roc"]),
            "auc_pr_delta": float(chain["auc_pr"] - base["auc_pr"]),
            "f2_delta": float(chain["f2"] - base["f2"]),
            "class_recall_deltas": {name: float(chain["class_recall_at_1pct_fpr"][name] - base["class_recall_at_1pct_fpr"][name]) for name in config.CLASSES},
        }
    result = {
        "experiment": "AML E3 chain-aware features and dual-channel cloud design",
        "data": "synthetic_aml_typologies",
        "generator": {"transactions": int(len(transactions)), "windows": int(len(windows)), "window_size": 10, "classes": list(config.CLASSES), "seed": config.SEED},
        "split": {**split, "method": "same account-window stratified split as E0/E1", "train_seed": config.SEED, "validation_test_seed": config.SEED + 1},
        "feature_sets": {
            "base": {"features": list(BASE_SPECS), "raw_counterparty_id": False},
            "base+chain": {"features": list(BASE_SPECS) + list(CHAIN_SPECS), "raw_counterparty_id": False},
            "chain_definitions": {
                "observed_hop_count": "distinct counterparties within this account window",
                "hop_completion_ratio": "distinct counterparties / transfers",
                "counterparty_shared_ratio": "repeat counterparty events / transfers",
                "max_counterparty_degree": "maximum undirected event degree; direction unavailable",
                "counterparty_transition_rate": "fraction of adjacent transfers changing counterparty",
                "dwell_time_cv": "coefficient of variation of within-window inter-transfer delays",
                "amount_retention_median": "median adjacent-transfer min(amount)/max(amount), conservation proxy",
                "amount_retention_p10": "10th percentile adjacent-transfer amount ratio, conservation proxy",
            },
            "dual_channel": "each engineered summary becomes a point token; CloudEncoderND pairwise diff/distance/cosine-angle features encode interactions between base and chain points",
        },
        "regimes": results,
        "base_vs_base_plus_chain": comparisons,
        "prototypes": {
            "definition": "train-only class centroid in base+chain hybrid CloudEncoderND output space",
            "distance_metric": "Euclidean",
            "class_distance_separation": prototype_summary,
            "base_ablation_class_distance_separation": base_prototype_summary,
            "layering_own_to_other_ratio_delta_chain_minus_base": float(
                prototype_summary["layering"]["own_to_other_ratio"]
                - base_prototype_summary["layering"]["own_to_other_ratio"]
            ),
            "layering_vs_normal_pca_centroid_gap": pca_centroid_gap,
            "layering_vs_normal_gap_fraction_of_map_span": layering_region_separation,
            "pca_explained_variance": [float(value) for value in variance],
        },
        "figure": os.path.abspath(figure_path),
        "comparison_notes": [
            "E3 base and base+chain both achieve layering recall 1.00 in all three regimes; the added chain family contributes 0.00 layering-recall delta on this seed.",
            "The E0 hybrid reference recall 0.04 is not a controlled direct comparison: E3 removes the raw counterparty_id token present in E0, while retaining E0's other features.",
            "E3 base retains generator-provided chain_hops, which is label-conditioned in this synthetic generator; perfect GBT/base recall is therefore not evidence that the engineered chain family caused the separation.",
        ],
        "limitations": [
            "Synthetic, balanced windows only; these results do not establish real AML performance.",
            "The generator has no transaction direction or explicit cross-account transfer edges; fan-in/out and in/out speed ratios cannot be recovered.",
            "Amount retention is a sequential ratio proxy, not true ledger conservation; inferred edges may not reflect real layering chains.",
            "E0's chain_hops field is retained in base for comparability and is generated with a label-conditioned rule; it is synthetic-label leakage risk.",
            "Raw counterparty IDs are excluded from both E3 variants; E0's CloudQuantizer included them, so E3 base is ID-safe rather than byte-identical to E0 quantization.",
            "PCA/KDE regions are descriptive, not calibrated classifier boundaries; prototypes are centroids, not operational thresholds.",
        ],
    }
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    _log(f"saved result {out_path}; base+chain hybrid layering recall={results['hybrid']['base+chain']['test']['class_recall_at_1pct_fpr']['layering']:.3f}", log_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-transactions", type=int, default=20_000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--figure", default=None)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()
    run_e3(args.n_transactions, args.out, args.figure, args.log)
