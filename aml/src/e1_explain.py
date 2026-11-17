"""E1: prototype-distance explanations and a 2D AML typology region map."""
from __future__ import annotations

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
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from generator import generate_transactions
from quantize import FeatureQuantizer
from train_aml import AMLCloud


PROBE_COUNTS = {
    "structuring": 10,
    "layering": 5,
    "velocity": 10,
    "normal": 15,
}
COLORS = {
    "normal": "#6c757d",
    "structuring": "#e6550d",
    "layering": "#3182bd",
    "velocity": "#31a354",
}


def _log(message: str, log_path: str | None):
    if log_path is None:
        return
    line = message if message.startswith("=== ") else (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    )
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _encode(model: AMLCloud, tokens: np.ndarray, batch_size: int = 128):
    """Return E0 CloudEncoderND outputs and E0 classifier logits."""
    embeddings, logits_out = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = torch.as_tensor(tokens[start:start + batch_size], dtype=torch.long)
            points = model.points_for(batch)
            mask = torch.ones(batch.shape, dtype=torch.bool)
            encoded = model.cloud_encoder(points, mask)
            embeddings.append(encoded.cpu().numpy())
            logits_out.append(model.class_head(encoded).cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(logits_out)


def _load_e0_model(checkpoint_path: str, vocab_size: int):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_config = checkpoint.get("config", {})
    if int(saved_config.get("vocab_size", vocab_size)) != vocab_size:
        raise ValueError(
            "E0 checkpoint vocabulary does not match the deterministic E0 quantizer; "
            "refusing to silently encode tokens with a mismatched model."
        )
    model = AMLCloud(vocab_size=vocab_size, n_classes=len(config.CLASSES), D=8)
    model.load_state_dict(checkpoint["model_state"])
    model.token_weights = checkpoint.get("token_weights", {})
    return model


def _fit_fallback(tokens: np.ndarray, labels: np.ndarray, vocab_size: int):
    """Small CPU hybrid fit matching E0's one-pass training recipe."""
    torch.manual_seed(config.SEED)
    model = AMLCloud(vocab_size=vocab_size, n_classes=len(config.CLASSES), D=8)
    optimizer = torch.optim.AdamW(
        list(model.cloud_encoder.parameters()) + list(model.class_head.parameters()),
        lr=0.003,
        weight_decay=1e-4,
    )
    rng = np.random.default_rng(config.SEED)
    train_ids = np.arange(len(labels))
    if len(train_ids) > 480:
        selected = []
        for class_id in np.unique(labels):
            candidates = train_ids[labels == class_id]
            selected.extend(rng.choice(candidates, min(len(candidates), 120), replace=False).tolist())
        train_ids = np.asarray(selected, dtype=int)
        rng.shuffle(train_ids)
    model.train()
    for row in train_ids:
        token_batch = torch.as_tensor(tokens[row:row + 1], dtype=torch.long)
        model.train()
        points = model.points_for(token_batch)
        logits, _ = model(token_batch, points_override=points.detach())
        loss = F.cross_entropy(logits, torch.tensor([int(labels[row])]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        model.quann_update(token_batch, int(labels[row]), hybrid=True)
    return model


def _reason(row) -> tuple[str, str]:
    """Produce an evidence-first motif and a short human-review SAR draft."""
    p95 = float(row.amount_p95)
    threshold_ratio = 100.0 * p95 / 10_000.0
    threshold_share = 100.0 * float(row.threshold_share)
    interval = float(row.inter_tx_interval_min)
    rate = float(row.rate_per_hour)
    count_60m = int(row.window_count)
    hops = int(row.chain_hops)
    counterparties = int(row.counterparty_count)
    channel = str(row.channel)

    if p95 >= 9_200 or threshold_share >= 30.0:
        motif = (
            f"eşik-kenarı kümesi: tutar p95={p95:,.0f} (10K eşiğinin %{threshold_ratio:.0f}'i); "
            f"işlemlerin %{threshold_share:.0f}'i 9.2K–10K bandında, "
            f"60 dk işlem sayısı={count_60m}"
        )
    elif hops >= 4 or (channel == "wire" and counterparties >= 7 and interval < 45.0):
        motif = (
            f"çok-hop zincir işareti: {hops} hop / {counterparties} karşı taraf, "
            f"medyan aralık={interval:.1f} dk, kanal={channel}"
        )
    elif interval < 1.0 or (rate >= 60.0 and channel == "digital"):
        motif = (
            f"yüksek-hız kümesi: medyan aralık={interval:.2f} dk, "
            f"saatlik hız={rate:.1f} işlem/saat, kanal={channel}"
        )
    else:
        motif = (
            f"taban aktivite örüntüsü: p95={p95:,.0f}, medyan aralık={interval:.1f} dk, "
            f"{counterparties} karşı taraf"
        )

    sar_draft = (
        f"Ön inceleme taslağı — {motif}. Gözlenen özellikler tipoloji incelemesine "
        "adaydır; kaynak işlem kayıtları ve müşteri bağlamı doğrulanmadan SAR kararı değildir."
    )
    return motif, sar_draft


def _plot_region_map(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_ids: np.ndarray,
    prototypes: dict[str, np.ndarray],
    figure_path: str,
):
    pca = PCA(n_components=2, random_state=config.SEED)
    projected = pca.fit_transform(embeddings)
    projected_prototypes = pca.transform(np.stack([prototypes[name] for name in config.CLASSES]))

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    markers = {"normal": "o", "structuring": "s", "layering": "^", "velocity": "D"}
    line_styles = {"normal": ":", "structuring": "-", "layering": "--", "velocity": "-."}
    for class_id, name in enumerate(config.CLASSES):
        points = projected[labels == class_id]
        bandwidth = max(0.015, min(0.3, float(np.median(np.std(points, axis=0)) * len(points) ** (-1 / 6))))
        # Class-local grids retain contours for tight/overlapping classes even
        # when another class makes the global PCA range much larger.
        lower = np.quantile(points, 0.01, axis=0)
        upper = np.quantile(points, 0.99, axis=0)
        padding = max(4.0 * bandwidth, 0.04)
        x_grid = np.linspace(lower[0] - padding, upper[0] + padding, 160)
        y_grid = np.linspace(lower[1] - padding, upper[1] + padding, 160)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        density = np.exp(KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(points).score_samples(grid))
        density = density.reshape(grid_x.shape)
        peak = float(density.max())
        fill_levels = peak * np.asarray([0.08, 0.2, 0.4, 1.01])
        line_levels = peak * np.asarray([0.12, 0.3, 0.65])
        ax.contourf(grid_x, grid_y, density, levels=fill_levels, colors=[COLORS[name]], alpha=0.055, zorder=1)
        ax.contour(grid_x, grid_y, density, levels=line_levels, colors=[COLORS[name]],
                   linewidths=[0.8, 1.1, 1.6], linestyles=line_styles[name], alpha=0.9, zorder=2)
        ax.scatter(
            points[:, 0], points[:, 1], s=42, marker=markers[name], facecolors="none",
            edgecolors=COLORS[name], alpha=0.78, linewidths=0.9, zorder=4,
        )

    label_offsets = {"normal": (-112, -32), "structuring": (9, 10), "layering": (9, 10), "velocity": (9, 10)}
    for class_id, name in enumerate(config.CLASSES):
        px, py = projected_prototypes[class_id]
        ax.scatter(px, py, marker="*", s=280, color=COLORS[name], edgecolor="#202020", linewidth=0.8, zorder=6)
        ax.annotate(
            f"{name} prototype", (px, py), xytext=label_offsets[name], textcoords="offset points",
            fontsize=9, weight="bold", color="#202020",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            zorder=7,
        )

    projection_scale = max(float(np.ptp(projected[:, 0])), float(np.ptp(projected[:, 1])))
    overlap_cutoff = max(0.2, 0.025 * projection_scale)
    for first_id in range(len(config.CLASSES)):
        for second_id in range(first_id + 1, len(config.CLASSES)):
            gap = float(np.linalg.norm(projected_prototypes[first_id] - projected_prototypes[second_id]))
            if gap <= overlap_cutoff:
                midpoint = (projected_prototypes[first_id] + projected_prototypes[second_id]) / 2
                ax.annotate(
                    f"{config.CLASSES[first_id]} / {config.CLASSES[second_id]} prototypes overlap in PCA-2D",
                    xy=midpoint, xytext=(0, -48), textcoords="offset points", ha="center",
                    fontsize=9, weight="bold", color="#202020",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#8b929a"},
                    arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.0},
                    zorder=8,
                )

    class_handles = [
        Line2D([0], [0], marker=markers[name], color="none", markerfacecolor="none",
               markeredgecolor=COLORS[name], markersize=8, label=name)
        for name in config.CLASSES
    ]
    class_handles.append(Line2D([0], [0], marker="*", color="#202020", linestyle="none", markersize=12, label="train prototype"))
    class_handles.append(Patch(facecolor="#999999", alpha=0.14, edgecolor="#666666", label="KDE density region"))
    variance = 100.0 * float(pca.explained_variance_ratio_.sum())
    ax.set_title(
        f"AML account-cloud regions — E0 hybrid encoder\n"
        f"{len(embeddings):,} synthetic windows · PCA-2D ({variance:.1f}% variance) · "
        "regions are KDE contours, not calibrated decision boundaries",
        fontsize=14, pad=14,
    )
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(True, color="#d9dee5", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=class_handles, loc="best", frameon=True, framealpha=0.95, ncol=2)
    os.makedirs(os.path.dirname(os.path.abspath(figure_path)), exist_ok=True)
    fig.savefig(figure_path, dpi=180, facecolor="white")
    plt.close(fig)


def run_e1(
    n_transactions: int = 20_000,
    out_path: str | None = None,
    figure_path: str | None = None,
    checkpoint_path: str | None = None,
    log_path: str | None = None,
):
    """Run the deterministic CPU E1 demo; paths are injectable for smoke tests."""
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    out_path = out_path or os.path.join(config.RESULTS_DIR, "e1_explain.json")
    figure_path = figure_path or os.path.join(config.FIGURES_DIR, "e1_region_map.png")
    checkpoint_path = checkpoint_path or os.path.join(config.RESULTS_DIR, "quann_hybrid.pt")
    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(figure_path)), exist_ok=True)

    _log(f"=== [{datetime.now().strftime('%Y-%m-%d')}] 1/5: regenerate deterministic E0 windows and split ===", log_path)
    _, windows = generate_transactions(n_transactions=n_transactions, seed=config.SEED)
    labels = np.asarray([config.CLASSES.index(name) for name in windows.typology], dtype=int)
    all_ids = np.arange(len(windows))
    train_ids, temp_ids = train_test_split(
        all_ids, test_size=0.4, random_state=config.SEED, stratify=labels,
    )
    _, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=config.SEED + 1, stratify=labels[temp_ids],
    )
    quantizer = FeatureQuantizer().fit(windows.iloc[train_ids].reset_index(drop=True))
    tokens = quantizer.transform(windows)
    _log(f"generated {len(windows):,} account windows; train={len(train_ids)}, test={len(test_ids)}; device=cpu", log_path)

    _log(f"=== [{datetime.now().strftime('%Y-%m-%d')}] 2/5: load E0 hybrid checkpoint or fit bounded CPU fallback ===", log_path)
    source = "e0_hybrid_checkpoint"
    if os.path.isfile(checkpoint_path):
        model = _load_e0_model(checkpoint_path, quantizer.vocab_size)
        _log(f"loaded E0 hybrid checkpoint {checkpoint_path}", log_path)
    else:
        source = "e1_cpu_fallback_fit"
        started = time.time()
        model = _fit_fallback(tokens[train_ids], labels[train_ids], quantizer.vocab_size)
        _log(f"E0 checkpoint unavailable; fitted same one-pass hybrid recipe on CPU in {time.time() - started:.2f}s", log_path)

    _log(f"=== [{datetime.now().strftime('%Y-%m-%d')}] 3/5: encode clouds, build train-only class centroids ===", log_path)
    all_embeddings, all_logits = _encode(model, tokens)
    prototypes = {
        name: all_embeddings[train_ids][labels[train_ids] == class_id].mean(axis=0)
        for class_id, name in enumerate(config.CLASSES)
    }
    _log(f"encoded {len(all_embeddings):,} clouds in CloudEncoderND output space ({all_embeddings.shape[1]}D)", log_path)

    _log(f"=== [{datetime.now().strftime('%Y-%m-%d')}] 4/5: select held-out probes, distances and evidence-first reasons ===", log_path)
    rng = np.random.default_rng(config.SEED + 11)
    selected_ids = []
    requested_counts = dict(PROBE_COUNTS)
    actual_counts = {}
    for name, requested in PROBE_COUNTS.items():
        class_id = config.CLASSES.index(name)
        candidates = test_ids[labels[test_ids] == class_id]
        count = min(requested, len(candidates))
        chosen = rng.choice(candidates, count, replace=False) if count else np.asarray([], dtype=int)
        selected_ids.extend(map(int, chosen))
        actual_counts[name] = int(count)

    prototype_matrix = np.stack([prototypes[name] for name in config.CLASSES])
    distance_matrix = np.linalg.norm(
        all_embeddings[np.asarray(selected_ids), None, :] - prototype_matrix[None, :, :], axis=2,
    )
    probe_rows = []
    for position, window_id in enumerate(selected_ids):
        distances = distance_matrix[position]
        ordered = np.argsort(distances, kind="stable")
        nearest_id, second_id = map(int, ordered[:2])
        true_id = int(labels[window_id])
        predicted_id = int(np.argmax(all_logits[window_id]))
        motif, _ = _reason(windows.iloc[window_id])
        sar_draft = (
            f"Geometrik yakınlık: {config.CLASSES[nearest_id]} prototipi d={distances[nearest_id]:.3f}; "
            f"ikincisi {config.CLASSES[second_id]} d={distances[second_id]:.3f}. "
            f"Ön inceleme taslağı — {motif}. Gözlenen özellikler tipoloji incelemesine adaydır; "
            "kaynak işlem kayıtları ve müşteri bağlamı doğrulanmadan SAR kararı değildir."
        )
        probe_rows.append({
            "account_id": f"HESAP#{window_id + 1:04d}",
            "window_id": int(window_id),
            "true_class": config.CLASSES[true_id],
            "predicted_class": config.CLASSES[predicted_id],
            "classifier_correct": bool(predicted_id == true_id),
            "nearest_prototype": config.CLASSES[nearest_id],
            "nearest_prototype_distance": float(distances[nearest_id]),
            "second_prototype": config.CLASSES[second_id],
            "second_prototype_distance": float(distances[second_id]),
            "nearest_prototype_correct": bool(nearest_id == true_id),
            "observable_reason": motif,
            "sar_draft": sar_draft,
        })

    summary_by_class = {}
    for class_id, name in enumerate(config.CLASSES):
        class_probe_positions = [i for i, row in enumerate(probe_rows) if row["true_class"] == name]
        own_distances, nearest_other_distances, all_other_distances = [], [], []
        for pos in class_probe_positions:
            distances = distance_matrix[pos]
            others = np.delete(distances, class_id)
            own_distances.append(float(distances[class_id]))
            nearest_other_distances.append(float(np.min(others)))
            all_other_distances.extend(map(float, others))
        summary_by_class[name] = {
            "n_probes": len(class_probe_positions),
            "median_own_prototype_distance": float(np.median(own_distances)) if own_distances else None,
            "median_nearest_other_prototype_distance": float(np.median(nearest_other_distances)) if nearest_other_distances else None,
            "median_all_other_prototype_distances": float(np.median(all_other_distances)) if all_other_distances else None,
            "classifier_accuracy": float(np.mean([probe_rows[i]["classifier_correct"] for i in class_probe_positions])) if class_probe_positions else None,
            "nearest_prototype_accuracy": float(np.mean([probe_rows[i]["nearest_prototype_correct"] for i in class_probe_positions])) if class_probe_positions else None,
        }
    classifier_accuracy = float(np.mean([row["classifier_correct"] for row in probe_rows])) if probe_rows else 0.0
    nearest_accuracy = float(np.mean([row["nearest_prototype_correct"] for row in probe_rows])) if probe_rows else 0.0
    heldout_distances = np.linalg.norm(
        all_embeddings[test_ids, None, :] - prototype_matrix[None, :, :], axis=2,
    )
    heldout_classifier_accuracy = float(np.mean(np.argmax(all_logits[test_ids], axis=1) == labels[test_ids]))
    heldout_prototype_accuracy = float(np.mean(np.argmin(heldout_distances, axis=1) == labels[test_ids]))

    _log(f"=== [{datetime.now().strftime('%Y-%m-%d')}] 5/5: save 2D PCA region map and E1 JSON ===", log_path)
    _plot_region_map(all_embeddings, labels, train_ids, prototypes, figure_path)
    result = {
        "experiment": "AML E1 prototype-distance explainability demo",
        "data": "synthetic_aml_typologies",
        "encoder_source": source,
        "prototype_definition": "class centroid of E0 CloudEncoderND outputs on the training split",
        "distance_metric": "Euclidean distance in CloudEncoderND output space",
        "generator": {
            "windows": int(len(windows)), "window_size": 10,
            "classes": list(config.CLASSES), "seed": config.SEED,
        },
        "probe_sampling": {
            "split": "held-out test split", "requested_per_class": requested_counts,
            "actual_per_class": actual_counts,
        },
        "summary": {
            "probe_classifier_argmax_accuracy": classifier_accuracy,
            "probe_nearest_prototype_accuracy": nearest_accuracy,
            "heldout_test_classifier_argmax_accuracy": heldout_classifier_accuracy,
            "heldout_test_nearest_prototype_accuracy": heldout_prototype_accuracy,
            "class_distance_separation": summary_by_class,
        },
        "probes": probe_rows,
        "figure": os.path.abspath(figure_path),
        "limitations": [
            "E0 is synthetic and class-balanced; results are not evidence of real-world AML performance.",
            "Layering's E0 hybrid recall at 1% FPR remains 0.04; E1 explanations name observed chain features but do not close this detection gap.",
            "PCA compresses the encoder space to two dimensions; KDE contours are descriptive density regions, not calibrated classifier boundaries.",
            "Normal and layering prototype projections nearly coincide at this global map scale; their distance summaries show the weak separation.",
            "Reason text is a fixed feature-based template; it is an analyst-review draft, not an automated SAR determination.",
        ],
    }
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    _log(f"saved region map {figure_path}", log_path)
    _log(f"saved result {out_path}; classifier accuracy={classifier_accuracy:.3f}, nearest-prototype accuracy={nearest_accuracy:.3f}", log_path)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-transactions", type=int, default=20_000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--figure", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--log", default=os.path.join(config.LOGS_DIR, "run_e1_smoke.log"))
    args = parser.parse_args()
    run_e1(args.n_transactions, args.out, args.figure, args.checkpoint, args.log)
