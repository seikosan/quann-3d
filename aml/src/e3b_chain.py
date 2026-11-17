"""E3': controlled chain ablation and hard, imbalanced AML synthetic run."""
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
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from e3_chain import BASE_SPECS, CHAIN_SPECS, E3Quantizer, add_chain_features, _matrix, _prototype_summary
from e1_explain import _reason
from generator import generate_transactions
from generator_hard import CLASS_SHARES, PROPORTIONAL_NOISE_LOG_SIGMA, generate_hard_transactions
from train_aml import AMLCloud, _metrics


VARIANTS = {"base": list(BASE_SPECS), "base+chain": list(BASE_SPECS) + list(CHAIN_SPECS)}
REGIMES = ("gbt", "hybrid", "backprop")
PROBE_COUNTS = {"normal": 12, "structuring": 6, "layering": 6, "velocity": 6}
COLORS = {"normal": "#6c757d", "structuring": "#e6550d", "layering": "#3182bd", "velocity": "#31a354"}
MARKERS = {"normal": "o", "structuring": "s", "layering": "^", "velocity": "D"}


def _log(message: str, log_path: str):
    line = message if message.startswith("=== ") else f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _split(labels: np.ndarray):
    """Use the exact E0/E3 account-window stratified split seeds."""
    ids = np.arange(len(labels))
    train_ids, temp_ids = train_test_split(
        ids, test_size=0.4, random_state=config.SEED, stratify=labels,
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=config.SEED + 1, stratify=labels[temp_ids],
    )
    return train_ids, val_ids, test_ids


def _fit_gbt(xtr, ytr, xva, yva, xte, yte):
    model = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=15, learning_rate=0.08, random_state=config.SEED,
    )
    started = time.time()
    model.fit(xtr, ytr)
    pval, ptest = model.predict_proba(xva), model.predict_proba(xte)
    val = _metrics(yva, pval)
    test = _metrics(yte, ptest, val["threshold"], val["class_thresholds_at_1pct_fpr"])
    test.update({"regime": "gbt", "fit_sec": round(time.time() - started, 2), "points": 0, "converged": 0})
    return {"test": test, "val": val}, model


def _fit_cloud(regime, tokens, ytr, val_tokens, yval, test_tokens, ytest, vocab_size):
    torch.manual_seed(config.SEED)
    model = AMLCloud(vocab_size=vocab_size, n_classes=len(config.CLASSES), D=8)
    trainable = model.parameters() if regime == "backprop" else (
        list(model.cloud_encoder.parameters()) + list(model.class_head.parameters())
    )
    optimizer = torch.optim.AdamW(trainable, lr=0.003, weight_decay=1e-4)
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
        outputs = []
        with torch.no_grad():
            for start in range(0, len(matrix), 256):
                batch = torch.as_tensor(matrix[start:start + 256], dtype=torch.long)
                points = model.points_for(batch) if regime == "hybrid" else None
                logits, _ = model(batch, points_override=points)
                outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    pval, ptest = predict(val_tokens), predict(test_tokens)
    val = _metrics(yval, pval)
    test = _metrics(ytest, ptest, val["threshold"], val["class_thresholds_at_1pct_fpr"])
    train_counts = {config.CLASSES[int(class_id)]: int(np.sum(ytr[train_ids] == class_id)) for class_id in np.unique(ytr)}
    test.update({"regime": regime, "train_sec": round(fit_sec, 2), "train_rows_used": int(len(train_ids)),
                 "train_class_counts_after_cap": train_counts, **model.convergence_info()})
    return {"test": test, "val": val}, model, ptest


def _cloud_embeddings(model: AMLCloud, tokens: np.ndarray):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(tokens), 128):
            batch = torch.as_tensor(tokens[start:start + 128], dtype=torch.long)
            points = model.points_for(batch)
            mask = torch.ones(batch.shape, dtype=torch.bool)
            output.append(model.cloud_encoder(points, mask).cpu().numpy())
    return np.concatenate(output, axis=0)


def _probe_report(windows, labels, test_ids, test_prob, embeddings, prototypes):
    prototype_matrix = np.stack([prototypes[name] for name in config.CLASSES])
    distances = np.linalg.norm(embeddings[test_ids, None, :] - prototype_matrix[None, :, :], axis=2)
    rng = np.random.default_rng(config.SEED + 11)
    probes = []
    for name, requested in PROBE_COUNTS.items():
        class_id = config.CLASSES.index(name)
        candidates = np.flatnonzero(labels[test_ids] == class_id)
        selected = rng.choice(candidates, min(requested, len(candidates)), replace=False) if len(candidates) else []
        for local_id in selected:
            global_id = int(test_ids[int(local_id)])
            dists = distances[int(local_id)]
            order = np.argsort(dists, kind="stable")
            nearest, second = map(int, order[:2])
            motif, _ = _reason(windows.iloc[global_id])
            sar = (
                f"Geometrik yakınlık: {config.CLASSES[nearest]} prototipi d={dists[nearest]:.3f}; "
                f"ikincisi {config.CLASSES[second]} d={dists[second]:.3f}. Ön inceleme taslağı — {motif}. "
                "Gözlenen özellikler tipoloji incelemesine adaydır; kaynak işlem kayıtları ve müşteri "
                "bağlamı doğrulanmadan SAR kararı değildir."
            )
            probes.append({
                "window_id": global_id,
                "true_class": name,
                "predicted_class": config.CLASSES[int(np.argmax(test_prob[int(local_id)]))],
                "nearest_prototype": config.CLASSES[nearest],
                "nearest_prototype_distance": float(dists[nearest]),
                "second_prototype": config.CLASSES[second],
                "second_prototype_distance": float(dists[second]),
                "nearest_prototype_correct": bool(nearest == class_id),
                "observable_reason": motif,
                "sar_draft": sar,
            })
    return probes


def _plot_hard_regions(embeddings, labels, prototypes, path):
    pca = PCA(n_components=2, random_state=config.SEED)
    projected = pca.fit_transform(embeddings)
    prototype_xy = pca.transform(np.stack([prototypes[name] for name in config.CLASSES]))
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    for class_id, name in enumerate(config.CLASSES):
        points = projected[labels == class_id]
        ax.scatter(points[:, 0], points[:, 1], s=28, marker=MARKERS[name], color=COLORS[name],
                   alpha=0.45, linewidths=0.4, label=name)
        bandwidth = max(0.035, min(0.45, float(np.median(np.std(points, axis=0)) * len(points) ** (-1 / 6))))
        low, high = np.quantile(points, [0.01, 0.99], axis=0)
        pad = max(3.0 * bandwidth, 0.05)
        gx, gy = np.meshgrid(np.linspace(low[0] - pad, high[0] + pad, 120),
                             np.linspace(low[1] - pad, high[1] + pad, 120))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        density = np.exp(KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(points).score_samples(grid)).reshape(gx.shape)
        ax.contour(gx, gy, density, levels=np.linspace(density.max() * 0.15, density.max() * 0.8, 3),
                   colors=[COLORS[name]], linewidths=1.1, alpha=0.8)
    for class_id, name in enumerate(config.CLASSES):
        x, y = prototype_xy[class_id]
        ax.scatter(x, y, marker="*", s=240, color=COLORS[name], edgecolor="#202020", zorder=5)
        ax.annotate(f"{name} prototype", (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9)
    variance = 100.0 * float(pca.explained_variance_ratio_.sum())
    ax.set_title(f"AML hard synthetic — hybrid base + chain | PCA-2D ({variance:.1f}% variance)\nKDE contours are descriptive density regions, not classifier boundaries")
    ax.set_xlabel("PCA component 1"); ax.set_ylabel("PCA component 2")
    ax.grid(True, alpha=0.2); ax.legend(frameon=True, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _dataset_frames(n_transactions):
    easy_transactions, easy_windows = generate_transactions(n_transactions=n_transactions, seed=config.SEED)
    easy_windows = add_chain_features(easy_transactions, easy_windows)
    hard_transactions, hard_windows = generate_hard_transactions(n_transactions=n_transactions, seed=config.SEED)
    hard_windows = add_chain_features(hard_transactions, hard_windows)
    return {
        "easy-v1-idless": (easy_transactions, easy_windows),
        "hard-v2": (hard_transactions, hard_windows),
    }


def run_e3b(n_transactions=20_000, out_path=None, figure_path=None, log_path=None):
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    out_path = out_path or os.path.join(config.RESULTS_DIR, "e3b_chain.json")
    figure_path = figure_path or os.path.join(config.FIGURES_DIR, "e3b_region_map.png")
    log_path = log_path or os.path.join(config.LOGS_DIR, "run_e3b_smoke.log")
    for path in (out_path, figure_path, log_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    _log(f"=== [{datetime.now():%Y-%m-%d}] 1/5: deterministic easy-v1 and imbalanced hard-v2 windows + shared account split ===", log_path)
    datasets = _dataset_frames(n_transactions)
    result = {
        "experiment": "AML E3' controlled chain ablation and hard synthetic stress test",
        "seed": config.SEED,
        "classes": list(config.CLASSES),
        "feature_variants": {name: {"features": features, "raw_counterparty_id": False} for name, features in VARIANTS.items()},
        "chain_features": list(CHAIN_SPECS),
        "datasets": {},
        "figure": os.path.abspath(figure_path),
        "limitations": [
            "Both corpora are synthetic and establish neither real-world AML performance nor calibrated deployment thresholds.",
            "Easy-v1 retains the E0/E3 chain_hops field in the shared base schema; its generator assigns it conditionally by label, so it is a synthetic-label shortcut common to both ablation arms.",
            "Hard-v2 chain_hops is observed distinct-counterparty count; chain proxies remain within-window summaries and do not represent verified cross-account directed transfers.",
            "The imbalanced hard test split has few positive examples per typology; recall at 1% FPR has substantial sampling uncertainty.",
            "Hybrid/backprop retain the E3 cap of at most 120 training rows per class; hard-v2 neural fits therefore downsample normal windows, unlike GBT, and report the actual fit counts.",
            "SAR text is a fixed feature-based analyst-review draft, not an automated SAR determination.",
        ],
    }
    selected_hard = None
    _log(f"=== [{datetime.now():%Y-%m-%d}] 2/5: fit GBT, hybrid and backprop for both feature sets and corpora ===", log_path)
    for dataset_name, (transactions, windows) in datasets.items():
        labels = np.asarray([config.CLASSES.index(name) for name in windows.typology], dtype=int)
        train_ids, val_ids, test_ids = _split(labels)
        split = {"train": int(len(train_ids)), "val": int(len(val_ids)), "test": int(len(test_ids)),
                 "method": "stratified account-window split; E0/E3 seeds", "group_key": "window_id", "train_seed": config.SEED,
                 "validation_test_seed": config.SEED + 1}
        _log(f"{dataset_name}: {len(transactions):,} events / {len(windows):,} windows; split={split['train']}/{split['val']}/{split['test']}; class mix={windows.typology.value_counts().to_dict()}", log_path)
        dataset_result = {
            "generator": {"transactions": int(len(transactions)), "windows": int(len(windows)), "window_size": 10,
                          "seed": config.SEED, "class_counts": {name: int((labels == i).sum()) for i, name in enumerate(config.CLASSES)}},
            "split": split,
            "regimes": {regime: {} for regime in REGIMES},
            "base_vs_base_plus_chain": {},
        }
        if dataset_name == "hard-v2":
            dataset_result["generator"].update({
                "class_shares": CLASS_SHARES,
                "proportional_noise_log_sigma": PROPORTIONAL_NOISE_LOG_SIGMA,
                "noise_description": "multiplicative lognormal perturbation on amounts and gaps; log-space sigma 0.20 before clipping",
                "overlap_description": "class signals sampled as mixtures with overlapping amount, timing, channel and observed-hop distributions",
            })

        variant_inputs = {}
        for variant_name, features in VARIANTS.items():
            train, val, test = (windows.iloc[ids].reset_index(drop=True) for ids in (train_ids, val_ids, test_ids))
            xtr, xva, xte = _matrix(train, val, test, features)
            gbt_result, _ = _fit_gbt(xtr, labels[train_ids], xva, labels[val_ids], xte, labels[test_ids])
            dataset_result["regimes"]["gbt"][variant_name] = gbt_result
            quantizer = E3Quantizer(include_chain=variant_name == "base+chain").fit(train)
            ttr, tva, tte = (quantizer.transform(frame) for frame in (train, val, test))
            all_tokens = quantizer.transform(windows.reset_index(drop=True))
            variant_inputs[variant_name] = (ttr, tva, tte, all_tokens, quantizer.vocab_size)
            gbt_metrics = gbt_result["test"]
            _log(f"{dataset_name}/gbt/{variant_name}: roc={gbt_metrics['auc_roc']:.3f} pr={gbt_metrics['auc_pr']:.3f} F2={gbt_metrics['f2']:.3f} recall={gbt_metrics['class_recall_at_1pct_fpr']}", log_path)

        hybrid_artifacts = {}
        for regime in ("hybrid", "backprop"):
            for variant_name in VARIANTS:
                ttr, tva, tte, all_tokens, vocab_size = variant_inputs[variant_name]
                cloud_result, model, test_prob = _fit_cloud(
                    regime, ttr, labels[train_ids], tva, labels[val_ids], tte, labels[test_ids], vocab_size,
                )
                dataset_result["regimes"][regime][variant_name] = cloud_result
                embeddings = _cloud_embeddings(model, all_tokens)
                prototypes, distance_summary = _prototype_summary(embeddings, labels, train_ids, test_ids)
                cloud_result["test"]["prototype_distances"] = distance_summary
                if regime == "hybrid":
                    hybrid_artifacts[variant_name] = {
                        "model": model, "tokens": all_tokens, "embeddings": embeddings,
                        "prototypes": prototypes, "test_prob": test_prob,
                    }
                metrics = cloud_result["test"]
                _log(f"{dataset_name}/{regime}/{variant_name}: roc={metrics['auc_roc']:.3f} pr={metrics['auc_pr']:.3f} F2={metrics['f2']:.3f} recall={metrics['class_recall_at_1pct_fpr']}", log_path)

        for regime in REGIMES:
            base = dataset_result["regimes"][regime]["base"]["test"]
            chain = dataset_result["regimes"][regime]["base+chain"]["test"]
            dataset_result["base_vs_base_plus_chain"][regime] = {
                "class_recall_delta": {name: float(chain["class_recall_at_1pct_fpr"][name] - base["class_recall_at_1pct_fpr"][name]) for name in config.CLASSES},
                "auc_roc_delta": float(chain["auc_roc"] - base["auc_roc"]),
                "auc_pr_delta": float(chain["auc_pr"] - base["auc_pr"]),
                "f2_delta": float(chain["f2"] - base["f2"]),
            }
        if dataset_name == "hard-v2":
            selected_hard = (windows, labels, test_ids, hybrid_artifacts["base+chain"])
        result["datasets"][dataset_name] = dataset_result

    windows, labels, test_ids, artifacts = selected_hard
    _log(f"=== [{datetime.now():%Y-%m-%d}] 3/5: aggregate controlled deltas and E1-style prototype distances ===", log_path)
    _log(f"=== [{datetime.now():%Y-%m-%d}] 4/5: E1-style hard-v2 prototypes, 30-target SAR probes and region map ===", log_path)
    probes = _probe_report(windows, labels, test_ids, artifacts["test_prob"], artifacts["embeddings"], artifacts["prototypes"])
    actual_probe_counts = {name: sum(probe["true_class"] == name for probe in probes) for name in config.CLASSES}
    result["datasets"]["hard-v2"]["prototype_probes"] = {
        "probe_count": len(probes),
        "requested_per_class": PROBE_COUNTS,
        "actual_per_class": actual_probe_counts,
        "sanity": {
            "all_have_evidence_reason": all(bool(row["observable_reason"]) for row in probes),
            "all_include_prototype_distances": all(row["nearest_prototype_distance"] >= 0 and row["second_prototype_distance"] >= 0 for row in probes),
            "all_mark_as_non_determination": all("SAR kararı değildir" in row["sar_draft"] for row in probes),
        },
        "probes": probes,
    }
    _plot_hard_regions(artifacts["embeddings"], labels, artifacts["prototypes"], figure_path)
    _log(f"saved hard-v2 region map {figure_path}; SAR probes={len(probes)}; counts={actual_probe_counts}", log_path)

    _log(f"=== [{datetime.now():%Y-%m-%d}] 5/5: summarize controlled deltas and write E3' JSON ===", log_path)
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    _log(f"saved result {out_path}", log_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-transactions", type=int, default=20_000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--figure", default=None)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()
    run_e3b(args.n_transactions, args.out, args.figure, args.log)
