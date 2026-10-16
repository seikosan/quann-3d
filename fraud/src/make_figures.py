"""Figures for the fraud sub-project.

Produces:
  figures/standard_models.png   — bar chart of the 4 standard baselines
  figures/quann_regimes.png     — bar chart of the 3 QUANN regimes
  figures/quann_pr.png          — precision-recall curves, 3 regimes (rebuilt test)
  figures/quann_cloud_3d.png    — 3D point-cloud scatter (fraud vs normal)

The PR + cloud figures REBUILD the exact held-out test set (deterministic seed)
and load the saved QUANN checkpoints, so no retraining is needed.

Run:  python3 fraud/src/make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import datasets
import train_quann
from qcloud_fraud import QCloudFraudClassifier


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


def quann_cloud_figures(df, outdir, n_train=1020000, n_val=100000, n_test=4700000,
                        test_frac=0.05):
    """PR curves + 3D point-cloud scatter for the 3 QUANN regimes (full run)."""
    # Rebuild the exact held-out test set the run used (same args/seed).
    data = train_quann.build_data(
        df, n_train=n_train, n_val=n_val, n_test=n_test,
        train_frac=0.5, test_frac=test_frac)
    tok_te = data["tok_te"]
    yte = data["y"][data["te_sel"]]
    fq = data["fq"]
    device = "cpu"

    import torch
    t = torch.tensor(tok_te, dtype=torch.long)
    mask = (t != 0) & (t != 1)

    regimes = ["pure", "hybrid", "backprop"]
    probs = {}
    points_map = {}
    for cfg in regimes:
        cp = os.path.join(config.RESULTS_DIR, f"quann_{cfg}.pt")
        if not os.path.exists(cp):
            print(f"  (no checkpoint for {cfg} — skip)")
            continue
        m = QCloudFraudClassifier.load_quann(cp, torch.device(device))
        p = m.predict_proba(t, use_quann_points=(cfg != "backprop")).cpu().numpy()
        probs[cfg] = p
        # the points each regime actually reads
        if cfg == "backprop":
            with torch.no_grad():
                _, pts = m(t, mask)
        else:
            pts = m.points_for(t, mask)
        points_map[cfg] = pts.cpu().numpy()  # (N, K, D)

    # ---- PR curves
    if probs:
        fig, ax = plt.subplots(figsize=(6, 5))
        for cfg, p in probs.items():
            prec, rec, _ = precision_recall_curve(yte, p)
            from sklearn.metrics import average_precision_score
            ax.plot(rec, prec, label=f"{cfg} (AUC-PR {average_precision_score(yte, p):.3f})")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title("QUANN point-cloud — PR curve (held-out test)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(outdir, "quann_pr.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"saved -> {out}")

    # ---- 3D point-cloud scatter (subsample for clarity)
    if points_map:
        rng = np.random.default_rng(config.SEED)
        n_show = 250
        fr = np.flatnonzero(yte == 1)
        no = np.flatnonzero(yte == 0)
        sel = np.concatenate([rng.choice(fr, min(120, len(fr)), replace=False),
                              rng.choice(no, n_show - min(120, len(fr)), replace=False)])
        pts = points_map.get("hybrid", next(iter(points_map.values())))[sel]  # (n, K, D)
        flat = pts.reshape(-1, pts.shape[-1])
        pca = PCA(n_components=3, random_state=config.SEED)
        emb = pca.fit_transform(flat).reshape(len(sel), pts.shape[1], 3)
        lab = yte[sel]

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


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    std = _load(os.path.join(config.RESULTS_DIR, "standard_full.json"))
    if std:
        # standard: models[name] IS the test-metrics dict
        bar_figure(std["models"],
                   "Standard fraud baselines (full IBM data, held-out test)",
                   os.path.join(config.FIGURES_DIR, "standard_models.png"))
    else:
        print("standard_full.json not found — skip")

    qu = _load(os.path.join(config.RESULTS_DIR, "quann_full.json"))
    if qu:
        # quann: configs[name] = {"test": metrics, "val": metrics}
        bar_figure({k: v["test"] for k, v in qu["configs"].items()},
                   "QUANN point-cloud — 3 regimes (full data, 4.7M test)",
                   os.path.join(config.FIGURES_DIR, "quann_regimes.png"))
    else:
        print("quann_full.json not found — skip")

    # PR + 3D cloud (needs the full dataset + checkpoints)
    if os.path.exists(os.path.join(config.RESULTS_DIR, "quann_hybrid.pt")):
        print("Rebuilding full test set for PR + 3D cloud figures ...")
        df = datasets.load_big()
        quann_cloud_figures(df, config.FIGURES_DIR)
    else:
        print("QUANN checkpoints missing — skip PR+3D")


if __name__ == "__main__":
    main()
