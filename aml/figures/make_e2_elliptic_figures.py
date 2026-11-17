#!/usr/bin/env python3
"""Render E2 Elliptic's real-data region map and optional GBT/SHAP comparison."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.neighbors import KernelDensity

HERE = Path(__file__).resolve().parent
AML = HERE.parent
ROOT = AML.parent
RESULT = AML / "results" / "e2_explain.json"
PALETTE = {"licit": "#687581", "illicit": "#D65A31"}
INK, MUTED, GRID = "#202A33", "#64717D", "#E5E9ED"

CAPTION = (
    "Gerçek darknet verisi (Elliptic, 203K tx, ~%4.4 illicit). Held-out nearest-prototype %75 "
    "(illicit'te %45) — sentetik AML'den (%87.8) zayıf; yüzey gerçek veride mütevazı. "
    "Tespitte GBT farkı ölçüldü (AUC .838 vs .943). Claim: denetlenebilir geometrik yüzey — ham tespit üstünlüğü değil."
)
CAVEAT = (
    "İşlem-node proxy (hesap kimliği yok) · anonim özellikler · 1-hop graf özellikleri transductive. "
    "K10/B8 fraud@1%FPR: hybrid %2.5 · GBT %62.5. PCA/KDE betimleyicidir; prototip uzaklığı özgün uzaydadır."
)


def _load_json():
    with RESULT.open(encoding="utf-8") as stream:
        return json.load(stream)


def _draw_map(ax, result, hero, *, compact=False, reason_box=False):
    """Plot prepared train-PCA test coordinates, class KDEs and train prototypes."""
    region = result["region_map_data"]
    test = region["test_points"]
    coords = np.asarray([item["pca2"] for item in test], dtype=float)
    labels = np.asarray([item["true_label"] for item in test])
    prototypes = {p["label"]: np.asarray(p["pca2"], dtype=float) for p in region["prototype_points"]}
    hero_xy = np.asarray(hero["pca2"], dtype=float)
    classes = ("licit", "illicit")

    # Shared grid gives comparable KDE context while contours are class-local.
    lo = np.quantile(coords, 0.01, axis=0)
    hi = np.quantile(coords, 0.99, axis=0)
    span = np.maximum(hi - lo, 0.8)
    padding = np.maximum(span * 0.18, 0.3)
    gx, gy = np.meshgrid(np.linspace(lo[0] - padding[0], hi[0] + padding[0], 180),
                         np.linspace(lo[1] - padding[1], hi[1] + padding[1], 180))
    grid = np.c_[gx.ravel(), gy.ravel()]
    markers = {"licit": "o", "illicit": "s"}
    styles = {"licit": ":", "illicit": "-"}

    for name in classes:
        points = coords[labels == name]
        if len(points) >= 3:
            scale = float(np.median(np.std(points, axis=0)))
            bandwidth = float(np.clip(scale * len(points) ** (-1 / 6), 0.07, 0.7))
            density = np.exp(KernelDensity(kernel="gaussian", bandwidth=bandwidth)
                             .fit(points).score_samples(grid)).reshape(gx.shape)
            peak = float(density.max())
            ax.contourf(gx, gy, density, levels=peak * np.array([0.10, 0.25, 0.50, 1.01]),
                        colors=[PALETTE[name]], alpha=0.055, zorder=1)
            ax.contour(gx, gy, density, levels=peak * np.array([0.14, 0.34, 0.68]),
                       colors=[PALETTE[name]], linewidths=[0.75, 1.0, 1.35],
                       linestyles=styles[name], alpha=0.88, zorder=2)
        ax.scatter(points[:, 0], points[:, 1], s=17 if not compact else 11,
                   marker=markers[name], facecolors="none", edgecolors=PALETTE[name],
                   alpha=0.40, linewidths=0.65, zorder=3)

    for name in classes:
        px, py = prototypes[name]
        ax.scatter(px, py, marker="*", s=210 if not compact else 135,
                   color=PALETTE[name], edgecolor="white", linewidth=0.8, zorder=7)
        offset = (-36, -20) if name == "licit" else (9, 9)
        ax.annotate(f"{name.upper()} prototipi", (px, py), xytext=offset,
                    textcoords="offset points", fontsize=8.5 if not compact else 7.1,
                    weight="bold", color=INK,
                    bbox={"boxstyle": "round,pad=0.16", "facecolor": "white",
                          "alpha": 0.9, "edgecolor": "none"}, zorder=8)

    nearest = hero["nearest_prototype"]
    pxy = prototypes[nearest]
    ax.plot([hero_xy[0], pxy[0]], [hero_xy[1], pxy[1]], color="#737D86",
            alpha=0.3, linewidth=1.4, zorder=5)
    ax.annotate("", xy=pxy, xytext=hero_xy,
                arrowprops={"arrowstyle": "->", "color": PALETTE[nearest],
                            "linewidth": 1.7, "shrinkA": 7, "shrinkB": 8}, zorder=6)
    ax.scatter(*hero_xy, marker="*", s=275 if not compact else 185,
               color="#111820", edgecolor="white", linewidth=1.1, zorder=9)
    ax.annotate(f"TX {hero['transaction_node_id']}", hero_xy, xytext=(10, -18),
                textcoords="offset points", fontsize=8.2 if not compact else 6.8,
                weight="bold", color=INK,
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                      "alpha": 0.95, "edgecolor": "#CBD2D8"}, zorder=10)

    if reason_box:
        short_reason = (
            "1-hop işlem-komşuluğu proxy\n"
            "observed_hop_count = 1 · hop_completion_ratio = 0.111\n"
            "max_counterparty_degree = 3 · transition = 0.000\n"
            "dwell_time_cv = 0.000\n"
            "Anonim özellikler; ham tutar / hesap kimliği yok."
        )
        ax.text(0.025, 0.97, short_reason, transform=ax.transAxes, va="top", ha="left",
                fontsize=8.1, color=INK,
                bbox={"boxstyle": "round,pad=0.45", "facecolor": "white",
                      "alpha": 0.94, "edgecolor": "#D9DEE3"}, zorder=12)

    ax.set_xlabel("PCA 1", fontsize=9 if not compact else 7.2, color=MUTED, labelpad=3)
    ax.set_ylabel("PCA 2", fontsize=9 if not compact else 7.2, color=MUTED, labelpad=3)
    ax.tick_params(labelsize=7.5 if not compact else 6.2, colors=MUTED, length=2)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C9D0D6")
    return prototypes


def make_region_map(result, hero, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titleweight": "bold", "figure.facecolor": "white",
                         "axes.facecolor": "white"})
    fig = plt.figure(figsize=(15, 10.3), facecolor="white")
    fig.text(0.055, 0.965, "ELLIPTIC · GERÇEK DARKNET VERİSİ · GEOMETRİK BÖLGE HARİTASI",
             fontsize=16, weight="bold", color=INK, va="top")
    fig.text(0.055, 0.925,
             "Held-out transaction-node proxy clouds · train-fit PCA-2D · class-conditional KDE",
             fontsize=10.0, color=MUTED, va="top")
    ax = fig.add_axes([0.085, 0.275, 0.84, 0.59])
    _draw_map(ax, result, hero, reason_box=True)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=PALETTE["licit"], markersize=6, label="licit test"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none", markeredgecolor=PALETTE["illicit"], markersize=6, label="illicit test"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=PALETTE["licit"], markeredgecolor=PALETTE["licit"], markersize=9, label="train prototipi"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111820", markeredgecolor="#111820", markersize=9, label="örnek olay"),
    ], loc="lower right", fontsize=7.5, ncol=2, frameon=True, framealpha=0.94,
       edgecolor="#D9DEE3", borderpad=0.35, handletextpad=0.3, columnspacing=0.8)
    fig.text(0.085, 0.238, "ILLICIT", fontsize=18, weight="bold", color=PALETTE["illicit"], va="top")
    fig.text(0.235, 0.242,
             f"TX {hero['transaction_node_id']} · gerçek sınıf: {hero['true_label']} · nearest: {hero['nearest_prototype'].upper()} · d={hero['nearest_prototype_distance']:.3f}",
             fontsize=9.1, weight="bold", color=INK, va="top")
    fig.text(0.085, 0.203, CAPTION, fontsize=8.4, color=INK, va="top", wrap=True)
    fig.text(0.085, 0.142, CAVEAT, fontsize=7.9, color=MUTED, va="top", wrap=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _feature_name(index):
    statistic, feature_index = divmod(index, 36)
    stats = ("ortalama", "std", "son değer")
    if feature_index < 29:
        feature = f"anonim eksen {feature_index + 1:02d}"
    else:
        names = ("timestep", "observed_hop_count", "hop_completion_ratio",
                 "1−hop_completion", "max_counterparty_degree",
                 "counterparty_transition_rate", "dwell_time_cv")
        feature = names[feature_index - 29]
    return f"{stats[statistic]} · {feature}"


def _positive_class_shap(shap_values, n_features):
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            if values.shape[-1] == 2:
                values = values[0, :, 1]
            elif values.shape[0] == 2:
                values = values[1, 0, :]
            else:
                raise ValueError(f"Unexpected SHAP output shape {values.shape}")
        elif values.ndim == 2:
            values = values[0]
    values = values.reshape(-1)
    if len(values) != n_features:
        raise ValueError(f"SHAP feature mismatch: {len(values)} != {n_features}")
    return values


def make_naming_comparison(result, hero, output):
    """Re-fit the exact E2 GBT on CPU and explain the selected held-out transaction."""
    start = time.monotonic()
    sys.path.insert(0, str(AML / "src"))
    from e2_elliptic import SEED, _load_sample, _matrix, _scale, _split
    from sklearn.ensemble import HistGradientBoostingClassifier
    import torch

    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    loaded = _load_sample(AML / "data" / "elliptic", n_accounts=2000)
    clouds, labels, target_ids = loaded["clouds"], loaded["labels"], loaded["target_ids"]
    train_ids, val_ids, test_ids = _split(clouds, labels)
    train_clouds, val_clouds, test_clouds = _scale(clouds[train_ids], clouds[val_ids], clouds[test_ids])
    x_train = _matrix(train_clouds)
    x_test = _matrix(test_clouds)
    id_positions = np.flatnonzero(target_ids == int(hero["transaction_node_id"]))
    if len(id_positions) != 1:
        raise ValueError(f"Transaction {hero['transaction_node_id']} not found exactly once in E2 sample")
    pos = int(id_positions[0])
    test_pos = np.flatnonzero(test_ids == pos)
    if len(test_pos) != 1:
        raise ValueError(f"Transaction {hero['transaction_node_id']} is not in the held-out test split")
    if int(labels[pos]) != 1:
        raise ValueError("GBT example is not labeled illicit in reproduced sample")
    xhero = x_test[int(test_pos[0]):int(test_pos[0]) + 1]
    if x_train.shape[1] != 108:
        raise ValueError(f"Unexpected E2 GBT feature matrix dimension: {x_train.shape[1]}")

    gbt = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                         learning_rate=0.08, random_state=SEED)
    gbt.fit(x_train, labels[train_ids])
    probability = float(gbt.predict_proba(xhero)[0, 1])
    import shap
    shap_values = shap.TreeExplainer(gbt).shap_values(xhero)
    risk = _positive_class_shap(shap_values, x_train.shape[1])
    top = np.argsort(np.abs(risk))[::-1][:3]
    if time.monotonic() - start > 60:
        raise TimeoutError("sample load + GBT/SHAP exceeded 60 seconds; skipped naming comparison")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titleweight": "bold", "figure.facecolor": "white",
                         "axes.facecolor": "white"})
    fig = plt.figure(figsize=(16, 8.8), facecolor="white")
    fig.text(0.045, 0.96, "AYNI GERÇEK OLAY · İKİ AÇIKLANABİLİRLİK ÇIKTISI",
             fontsize=16, weight="bold", color=INK, va="top")
    fig.text(0.045, 0.918,
             f"Elliptic TX {hero['transaction_node_id']} · gerçek sınıf: ILLICIT · nearest-prototype: ILLICIT (doğru adlandırma)",
             fontsize=10, color=MUTED, va="top")
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.21, 0.85], transform=fig.transFigure,
                              color="#D9DEE3", linewidth=0.8))
    fig.text(0.055, 0.855, "GBT + SHAP · STANDART", fontsize=12, weight="bold", color=INK, va="top")
    fig.text(0.055, 0.817, "SKOR + feature — AD YOK", fontsize=9.5, color=MUTED, va="top")
    fig.text(0.535, 0.855, "ELLIPTIC HİBRİT · BÖLGE HARİTASI", fontsize=12, weight="bold", color=INK, va="top")
    fig.text(0.535, 0.817, "AD + geometri", fontsize=9.5, color=MUTED, va="top")
    fig.text(0.065, 0.755, "P(ILLICIT)", fontsize=8.4, weight="bold", color=MUTED, va="top")
    fig.text(0.065, 0.704, f"{probability * 100:.1f}%", fontsize=31, weight="bold",
             color=PALETTE["illicit"], va="top")
    fig.text(0.31, 0.731, "E2 test split · CPU · seed 20261016\nGBT: max_iter=100, max_leaf_nodes=15",
             fontsize=8.5, color=MUTED, va="top", linespacing=1.45)
    fig.text(0.065, 0.627, "Top 3 SHAP · signed contribution toward ILLICIT", fontsize=8.8,
             weight="bold", color=INK, va="top")
    ax_bar = fig.add_axes([0.19, 0.36, 0.28, 0.235])
    order = top[::-1]
    vals = risk[order]
    ypos = np.arange(len(vals))
    bar_colors = ["#D65A31" if v >= 0 else "#3478A8" for v in vals]
    ax_bar.barh(ypos, vals, color=bar_colors, height=0.57, edgecolor="none")
    ax_bar.axvline(0, color="#66717B", linewidth=0.8)
    ax_bar.set_yticks(ypos, [_feature_name(int(i)) for i in order], fontsize=7.6, color=INK)
    ax_bar.set_xlabel("SHAP signed value · illicit class", fontsize=7.6, color=MUTED, labelpad=3)
    ax_bar.tick_params(axis="x", labelsize=7.2, colors=MUTED, length=2)
    ax_bar.tick_params(axis="y", length=0, pad=6)
    ax_bar.grid(axis="x", color=GRID, linewidth=0.6)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top", "right", "left"]].set_visible(False)
    ax_bar.spines["bottom"].set_color("#C9D0D6")
    span = max(float(np.ptp(vals)), float(np.max(np.abs(vals))) * 0.25, 1e-4)
    ax_bar.set_xlim(min(0.0, float(vals.min()) - span * 0.35), max(0.0, float(vals.max()) + span * 0.35))
    for row, value in enumerate(vals):
        ax_bar.text(value + (span * 0.04 if value >= 0 else -span * 0.04), row,
                    f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right",
                    fontsize=7.6, color=INK)
    fig.text(0.065, 0.298, "Pozitif SHAP = model marjını illicit yönüne iter; anonim ham eksenler isimlendirilmez.",
             fontsize=7.6, color=MUTED, va="top")

    ax_map = fig.add_axes([0.555, 0.365, 0.405, 0.385])
    _draw_map(ax_map, result, hero, compact=True)
    ax_map.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=PALETTE["licit"], markersize=5, label="licit"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none", markeredgecolor=PALETTE["illicit"], markersize=5, label="illicit"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=PALETTE["licit"], markeredgecolor=PALETTE["licit"], markersize=7, label="prototip"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111820", markeredgecolor="#111820", markersize=7, label="olay")
    ], loc="lower right", fontsize=6.2, ncol=2, frameon=True, framealpha=0.94,
       edgecolor="#D9DEE3", borderpad=0.3, handletextpad=0.25, columnspacing=0.6)
    fig.text(0.555, 0.324, "ILLICIT", fontsize=16, weight="bold", color=PALETTE["illicit"], va="top")
    fig.text(0.69, 0.327, f"nearest d={hero['nearest_prototype_distance']:.3f} · 1-hop proxy",
             fontsize=7.7, weight="bold", color=INK, va="top")
    fig.text(0.555, 0.291,
             "observed_hop_count=1 · hop_completion_ratio=0.111 · max_counterparty_degree=3",
             fontsize=7.15, color=INK, va="top")
    fig.text(0.555, 0.267,
             "transition=0.000 · dwell_time_cv=0.000 · anonim eksenler; ham tutar / hesap kimliği yok",
             fontsize=7.15, color=MUTED, va="top")
    fig.add_artist(plt.Rectangle((0.045, 0.185), 0.91, 0.043, transform=fig.transFigure,
                                 facecolor="#F1F4F6", edgecolor="#DDE2E6", linewidth=0.7, zorder=-1))
    fig.text(0.5, 0.206, "GBT → skor + feature-seviyesinde neden  |  Geometrik yüzey → sınıf adı + prototip uzaklığı + proxy gerekçesi",
             fontsize=9.0, weight="bold", color=INK, ha="center", va="center")
    fig.text(0.045, 0.148, "Bu figür tespit üstünlüğü claimi değildir; Elliptic transaction-node/account-proxy verisinde adlandırılabilir geometrik yüzeyi gösterir.",
             fontsize=8.0, color=INK, va="top")
    fig.text(0.045, 0.105, "Kısıt: anonimleştirilmiş özellikler · hesap kimliği / ham tutar yok · 1-hop graf özellikleri transductive · PCA/KDE betimleyicidir.",
             fontsize=7.7, color=MUTED, va="top")
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"GBT_PROBABILITY={probability:.6f}")
    print("TOP3_SHAP=" + json.dumps([(_feature_name(int(i)), float(risk[i])) for i in top], ensure_ascii=False))
    print(f"GBT_SHAP_SECONDS={time.monotonic() - start:.2f}")


def main():
    result = _load_json()
    hero = next(p for p in result["summary"]["probes"] if p["transaction_node_id"] == "294455185")
    if not (hero["true_label"] == hero["nearest_prototype"] == "illicit"):
        raise AssertionError("Selected probe is not correctly named illicit")
    figures = Path(__file__).resolve().parent
    region_path = figures / "e2_elliptic_region_map.png"
    make_region_map(result, hero, region_path)
    print(f"REGION_MAP={region_path}")
    print(f"PROBE=TX {hero['transaction_node_id']} true={hero['true_label']} nearest={hero['nearest_prototype']} d={hero['nearest_prototype_distance']:.6f}")
    try:
        naming_path = figures / "e2_elliptic_naming_comparison.png"
        make_naming_comparison(result, hero, naming_path)
        print(f"NAMING_COMPARISON={naming_path}")
    except Exception as exc:
        print(f"NAMING_COMPARISON_SKIPPED={type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
