"""Figures for the fraud sub-project.

Produces:
  figures/standard_models.png   — bar chart of the 4 standard baselines
  figures/quann_regimes.png     — bar chart of the 3 QUANN regimes
  figures/quann_pr.png          — precision-recall curves, 3 regimes (rebuilt test)
  figures/quann_cloud_3d.png    — 3D point-cloud scatter (fraud vs normal)

The PR + cloud figures rebuild the canonical test population in chunks and load
the saved QUANN checkpoints, so no retraining is needed. PR uses a stratified
1M-row sample; the cloud stage uses 10K rows.

Run:  python3 fraud/src/make_figures.py
"""
from __future__ import annotations

import json
import gc
import os
import resource
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import train_quann
import quantize
from qcloud_fraud import QCloudFraudClassifier


CHUNK_SIZE = 1_000_000
FIG_N_TRAIN = 1_020_000
FIG_N_TEST = 4_700_000
FIG_TRAIN_FRAC = 0.5
FIG_TEST_FRAC = 0.05
PR_MAX_ROWS = 1_000_000
CLOUD_MAX_ROWS = 10_000
RAW_FEATURE_COLUMNS = [
    "Year", "Month", "Day", "Time", config.AMOUNT_COL, "Use Chip",
    "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
    "Errors?",
]


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def bar_figure(models: dict, title: str, out: str):
    names = list(models.keys())
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.6), 4.5))
    ax.bar(x - 1.5 * w, [models[n]["auc_pr"] for n in names], w, label="AUC-PR", color="#e6550d")
    ax.bar(x - 0.5 * w, [models[n]["auc_roc"] for n in names], w, label="AUC-ROC", color="#3182bd")
    ax.bar(x + 0.5 * w, [models[n]["fbeta"] for n in names], w, label="F2 (best-thr)", color="#31a354")
    ax.bar(x + 1.5 * w, [models[n]["recall"] for n in names], w, label="Recall (best-thr)", color="#756bb1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved -> {out}")


def _sample_rank_order(n_fraud, n_normal, n_total, fraud_frac, seed):
    """Return selected class-relative row ranks in the order used by build_data."""
    fraud_population = n_fraud
    n_fraud_target = int(round(n_total * fraud_frac))
    n_fraud_selected = min(n_fraud_target, n_fraud)
    n_normal_selected = min(n_total - n_fraud_selected, n_normal)
    rng = np.random.default_rng(seed)
    fraud_ranks = rng.choice(fraud_population, size=n_fraud_selected, replace=False)
    normal_ranks = rng.choice(n_normal, size=n_normal_selected, replace=False)
    # Encode the class in the rank so the original final shuffle is reproducible.
    selected = np.concatenate([fraud_ranks, normal_ranks + fraud_population])
    rng.shuffle(selected)
    return selected, n_fraud_selected, n_normal_selected


def _make_figure_data():
    """Stream the CSV and retain only the canonical train sample + plot samples.

    The split/user folds and row-rank sampling mirror train_quann.build_data and
    sample_from_users, without materializing the full raw dataframe or full test
    token matrix.
    """
    print(f"  scanning user/label counts in {CHUNK_SIZE:,}-row chunks ...", flush=True)
    user_parts = []
    user_counts = {}
    for chunk in pd.read_csv(
        config.BIG_CSV,
        usecols=[config.GROUP_COL, config.LABEL_COL],
        chunksize=CHUNK_SIZE,
    ):
        users = chunk[config.GROUP_COL].to_numpy()
        labels = (chunk[config.LABEL_COL].astype(str).str.strip().str.lower() == "yes").to_numpy()
        unique, inverse = np.unique(users, return_inverse=True)
        fraud_counts = np.bincount(inverse, weights=labels.astype(np.int8), minlength=len(unique))
        total_counts = np.bincount(inverse, minlength=len(unique))
        user_parts.append(unique)
        for user, fraud_n, total_n in zip(unique, fraud_counts, total_counts):
            counts = user_counts.setdefault(int(user), [0, 0])
            counts[1] += int(fraud_n)
            counts[0] += int(total_n - fraud_n)
        del chunk, users, labels, unique, inverse, fraud_counts, total_counts

    uniq = np.unique(np.concatenate(user_parts))
    del user_parts
    fold = train_quann.user_folds(len(uniq), n_splits=10)
    tr_users = uniq[fold < 7]
    te_users = uniq[fold >= 8]
    n_train_fraud = sum(user_counts[int(u)][1] for u in tr_users)
    n_train_normal = sum(user_counts[int(u)][0] for u in tr_users)
    n_test_fraud = sum(user_counts[int(u)][1] for u in te_users)
    n_test_normal = sum(user_counts[int(u)][0] for u in te_users)

    train_order, n_train_fraud_sel, n_train_normal_sel = _sample_rank_order(
        n_train_fraud, n_train_normal, FIG_N_TRAIN, FIG_TRAIN_FRAC, config.SEED)
    test_order, n_test_fraud_sel, n_test_normal_sel = _sample_rank_order(
        n_test_fraud, n_test_normal, FIG_N_TEST, FIG_TEST_FRAC, config.SEED + 2)
    # Sample the PR population stratified from the exact canonical test sample.
    pr_rng = np.random.default_rng(config.SEED)
    n_pr_fraud = min(n_test_fraud_sel, PR_MAX_ROWS)
    n_pr_normal = min(n_test_normal_sel, PR_MAX_ROWS - n_pr_fraud)
    test_fraud_ranks = test_order[test_order < n_test_fraud]
    test_normal_ranks = test_order[test_order >= n_test_fraud] - n_test_fraud
    pr_fraud_ranks = pr_rng.choice(test_fraud_ranks, size=n_pr_fraud, replace=False)
    pr_normal_ranks = pr_rng.choice(test_normal_ranks, size=n_pr_normal, replace=False)
    pr_fraud_sorted = np.sort(pr_fraud_ranks)
    pr_normal_sorted = np.sort(pr_normal_ranks)

    # The quantizer fit sample must match build_data's shuffled selected train rows.
    train_sorted = np.sort(train_order)
    train_order_for_sorted = np.argsort(train_order)
    train_fraud_sorted = train_sorted[train_sorted < n_train_fraud]
    train_normal_sorted = train_sorted[train_sorted >= n_train_fraud] - n_train_fraud
    train_fraud_order = train_order_for_sorted[train_sorted < n_train_fraud]
    train_normal_order = train_order_for_sorted[train_sorted >= n_train_fraud]

    train_frames, train_orders, pr_frames, pr_labels = [], [], [], []
    train_f_rank = train_n_rank = test_f_rank = test_n_rank = 0
    print("  extracting selected training rows + stratified PR rows ...", flush=True)
    usecols = [config.GROUP_COL, config.LABEL_COL] + RAW_FEATURE_COLUMNS

    def retain(chunk, candidate_mask, rank_start, selected_ranks, rank_order, sink, y):
        candidate_idx = np.flatnonzero(candidate_mask)
        ranks = rank_start + np.arange(len(candidate_idx), dtype=np.int64)
        at = np.searchsorted(selected_ranks, ranks)
        hit = at < len(selected_ranks)
        if hit.any():
            matched = np.zeros_like(hit)
            matched[hit] = selected_ranks[at[hit]] == ranks[hit]
            hit = matched
        if hit.any():
            selected_idx = candidate_idx[hit]
            selected = chunk.iloc[selected_idx][RAW_FEATURE_COLUMNS].copy()
            if sink == "train":
                train_frames.append(selected)
                train_orders.append(rank_order[at[hit]])
            else:
                pr_frames.append(selected)
                pr_labels.append(y[selected_idx])
        return rank_start + len(candidate_idx)

    for chunk in pd.read_csv(config.BIG_CSV, usecols=usecols, chunksize=CHUNK_SIZE):
        users = chunk[config.GROUP_COL].to_numpy()
        y = (chunk[config.LABEL_COL].astype(str).str.strip().str.lower() == "yes").to_numpy(dtype=np.int8)
        is_train = np.isin(users, tr_users)
        is_test = np.isin(users, te_users)
        is_fraud = y == 1
        is_normal = y == 0
        train_f_rank = retain(chunk, is_train & is_fraud, train_f_rank,
                              train_fraud_sorted, train_fraud_order, "train", y)
        train_n_rank = retain(chunk, is_train & is_normal, train_n_rank,
                              train_normal_sorted, train_normal_order, "train", y)
        test_f_rank = retain(chunk, is_test & is_fraud, test_f_rank,
                             pr_fraud_sorted, None, "pr", y)
        test_n_rank = retain(chunk, is_test & is_normal, test_n_rank,
                             pr_normal_sorted, None, "pr", y)
        del chunk, users, y, is_train, is_test

    if (train_f_rank != n_train_fraud or train_n_rank != n_train_normal
            or test_f_rank != n_test_fraud or test_n_rank != n_test_normal):
        raise RuntimeError("streamed row counts changed between CSV passes")

    # Restore sample_from_users' shuffled order for exact train-only category ties.
    df_train = pd.concat(train_frames, ignore_index=True)
    train_row_order = np.concatenate(train_orders)
    df_train = df_train.iloc[np.argsort(train_row_order)].reset_index(drop=True)
    del train_frames, train_orders, train_row_order
    df_pr = pd.concat(pr_frames, ignore_index=True)
    y_pr = np.concatenate(pr_labels)
    del pr_frames, pr_labels
    if len(df_pr) != n_pr_fraud + n_pr_normal or int(y_pr.sum()) != n_pr_fraud:
        raise RuntimeError("streamed PR sample does not match requested stratified test counts")

    fq = quantize.FeatureQuantizer(specs=quantize.LEGACY_QUANT_SPECS).fit(df_train)
    del df_train, user_counts, uniq, tr_users, te_users
    gc.collect()
    cloud_rng = np.random.default_rng(config.SEED + 1)
    n_cloud_fraud = min(300, int(y_pr.sum()))
    n_cloud_normal = min(CLOUD_MAX_ROWS - n_cloud_fraud, int((y_pr == 0).sum()))
    cloud_fraud_idx = cloud_rng.choice(np.flatnonzero(y_pr == 1), size=n_cloud_fraud, replace=False)
    cloud_normal_idx = cloud_rng.choice(np.flatnonzero(y_pr == 0), size=n_cloud_normal, replace=False)
    cloud_idx = np.concatenate([cloud_fraud_idx, cloud_normal_idx])
    df_cloud = df_pr.iloc[cloud_idx][RAW_FEATURE_COLUMNS].reset_index(drop=True)
    y_cloud = y_pr[cloud_idx]
    tok_cloud = fq.transform(df_cloud)
    del df_cloud
    gc.collect()
    tok_pr = fq.transform(df_pr)
    del df_pr, fq
    gc.collect()
    print(f"  canonical test: {FIG_N_TEST:,} rows ({n_test_fraud_sel:,} fraud); "
          f"PR sample: {len(y_pr):,} rows ({int(y_pr.sum()):,} fraud); "
          f"cloud sample: {len(y_cloud):,} rows ({int(y_cloud.sum()):,} fraud)", flush=True)
    return tok_pr, y_pr, tok_cloud, y_cloud


def quann_cloud_figures(tok_pr, y_pr, tok_cloud, y_cloud, outdir):
    """PR curves on a stratified 1M test sample + 3D scatter on 10K rows."""
    device = "cpu"

    import torch
    t_pr = torch.tensor(tok_pr, dtype=torch.long)
    t_cloud = torch.tensor(tok_cloud, dtype=torch.long)
    cloud_mask = (t_cloud != 0) & (t_cloud != 1)

    regimes = ["pure", "hybrid", "backprop"]
    probs = {}
    points_map = {}
    for cfg in regimes:
        cp = os.path.join(config.RESULTS_DIR, f"quann_{cfg}.pt")
        if not os.path.exists(cp):
            print(f"  (no checkpoint for {cfg} — skip)")
            continue
        m = QCloudFraudClassifier.load_quann(cp, torch.device(device))
        p = m.predict_proba(t_pr, use_quann_points=(cfg != "backprop")).cpu().numpy()
        probs[cfg] = p
        # the points each regime actually reads
        if cfg == "backprop":
            with torch.no_grad():
                _, pts = m(t_cloud, cloud_mask)
        else:
            pts = m.points_for(t_cloud, cloud_mask)
        points_map[cfg] = pts.cpu().numpy()  # (N, K, D)
        del m, p, pts
        gc.collect()

    # ---- PR curves
    if probs:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5),
                                     gridspec_kw={"width_ratios": [1.25, 1]})
        from sklearn.metrics import average_precision_score
        aggregate = {cfg: float(average_precision_score(y_pr, p))
                     for cfg, p in probs.items()}
        for cfg, p in probs.items():
            prec, rec, _ = precision_recall_curve(y_pr, p)
            ax.plot(rec, prec, label=f"{cfg} (AUC-PR {aggregate[cfg]:.3f})")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title("QUANN point-cloud — PR curves (1M stratified test sample)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)

        # Compare aggregate AUC-PR from existing result files; baseline score
        # artifacts are not required and no extra inference is performed.
        matched = _load(os.path.join(config.RESULTS_DIR, "duels_quann.json"))
        if matched:
            for key, label in (("xgboost", "XGB"), ("lightgbm", "LGBM"),
                               ("ftt", "FT-Transformer")):
                if key in matched.get("models", {}):
                    aggregate[label] = matched["models"][key]["auc_pr"]
        else:
            print("duels_quann.json not found — PR comparison omits matched baselines")
        strong = _load(os.path.join(config.RESULTS_DIR, "strong_reader.json"))
        if strong:
            for size, values in strong.get("results", {}).items():
                aggregate[f"reader {size}"] = values["auc_pr"]
        else:
            print("strong_reader.json not found — PR comparison omits reader aggregates")
        names = list(aggregate)
        bx.bar(np.arange(len(names)), [aggregate[name] for name in names], color="#3182bd")
        bx.set_xticks(np.arange(len(names)))
        bx.set_xticklabels(names, rotation=35, ha="right")
        bx.set_ylabel("AUC-PR")
        bx.set_title("AUC-PR aggregates\nQUANN sample; JSON full-test")
        bx.set_ylim(0, min(1.0, max(0.1, max(aggregate.values()) * 1.15)))
        bx.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = os.path.join(outdir, "quann_pr.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"saved -> {out}")

    # ---- 3D point-cloud scatter (subsample for clarity)
    if points_map:
        rng = np.random.default_rng(config.SEED)
        n_show = 250
        fr = np.flatnonzero(y_cloud == 1)
        no = np.flatnonzero(y_cloud == 0)
        sel = np.concatenate([rng.choice(fr, min(120, len(fr)), replace=False),
                              rng.choice(no, n_show - min(120, len(fr)), replace=False)])
        pts = points_map.get("hybrid", next(iter(points_map.values())))[sel]  # (n, K, D)
        flat = pts.reshape(-1, pts.shape[-1])
        pca = PCA(n_components=3, random_state=config.SEED)
        emb = pca.fit_transform(flat).reshape(len(sel), pts.shape[1], 3)
        lab = y_cloud[sel]

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection="3d")
        cfr = "#d62728"
        cno = "#7f7f7f"
        for i in range(len(sel)):
            c = cfr if lab[i] == 1 else cno
            ax.scatter(emb[i, :, 0], emb[i, :, 1], emb[i, :, 2],
                       c=c, s=14, alpha=0.85,
                       label="fraud" if i == 0 and lab[i] == 1 else
                             ("normal" if i == 0 else None))
        # one marker for each class to seed the legend
        ax.scatter([], [], [], c=cfr, s=14, label="fraud point")
        ax.scatter([], [], [], c=cno, s=14, label="normal point")
        ax.set_title("Point-cloud geometry (QUANN points, PCA->3D; one point/feature)\n"
                     "fraud clouds vs normal clouds")
        ax.set_xlabel("pc1"); ax.set_ylabel("pc2"); ax.set_zlabel("pc3")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(outdir, "quann_cloud_3d.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"saved -> {out}")

    del t_pr, t_cloud, cloud_mask, tok_pr, tok_cloud
    gc.collect()


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    std = _load(os.path.join(config.RESULTS_DIR, "standard_full.json"))
    if std:
        # standard: models[name] IS the test-metrics dict
        standard_models = dict(std["models"])
        duels = _load(os.path.join(config.RESULTS_DIR, "duels.json"))
        if duels:
            for key, label in (("xgboost", "XGB (duels)"),
                               ("lightgbm", "LGBM (duels)"),
                               ("ftt", "FT-Transformer (duels)")):
                if key in duels.get("models", {}):
                    standard_models[label] = duels["models"][key]
        else:
            print("duels.json not found — standard figure uses standard baselines only")
        bar_figure(standard_models,
                   "Standard fraud baselines (full IBM data, held-out test)",
                   os.path.join(config.FIGURES_DIR, "standard_models.png"))
    else:
        print("standard_full.json not found — skip")

    qu = _load(os.path.join(config.RESULTS_DIR, "quann_full.json"))
    if qu:
        # quann: configs[name] = {"test": metrics, "val": metrics}
        regimes = {k: v["test"] for k, v in qu["configs"].items()}
        strong = _load(os.path.join(config.RESULTS_DIR, "strong_reader.json"))
        if strong:
            for size, values in strong.get("results", {}).items():
                regimes[f"reader {size}"] = values
        bar_figure(regimes,
                   "QUANN point-cloud — 3 regimes (full data, 4.7M test)",
                   os.path.join(config.FIGURES_DIR, "quann_regimes.png"))
    else:
        print("quann_full.json not found — skip")

    # PR + 3D cloud (needs the full dataset + checkpoints)
    if os.path.exists(os.path.join(config.RESULTS_DIR, "quann_hybrid.pt")):
        print("Rebuilding full test set for PR + 3D cloud figures ...", flush=True)
        tok_pr, y_pr, tok_cloud, y_cloud = _make_figure_data()
        quann_cloud_figures(tok_pr, y_pr, tok_cloud, y_cloud, config.FIGURES_DIR)
        rss_scale = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2
        print(f"  process peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_scale:.1f} GB",
              flush=True)
    else:
        print("QUANN checkpoints missing — skip PR+3D")


if __name__ == "__main__":
    main()
