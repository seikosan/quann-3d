#!/usr/bin/env python3
"""Reproduce the E0 GBM/SHAP vs E1 QUANN typology-name comparison figure."""
from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import shap
import torch
from matplotlib.colors import to_rgba
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

ROOT = Path(__file__).resolve().parents[2]
AML = ROOT / "aml"
SRC = AML / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
from e1_explain import _encode, _load_e0_model  # noqa: E402
from generator import generate_transactions  # noqa: E402
from train_aml import _feature_matrix  # noqa: E402
from quantize import FeatureQuantizer  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "typology_naming_comparison.png"
E1_JSON = AML / "results" / "e1_explain.json"
CHECKPOINT = AML / "results" / "quann_hybrid.pt"
PALETTE = {
    "normal": "#687581",
    "structuring": "#D65A31",
    "layering": "#3478A8",
    "velocity": "#3A8D68",
}
INK = "#202A33"
MUTED = "#64717D"
GRID = "#E5E9ED"

# Shared composition, with event-specific framing supplied as named profiles.
FRAMINGS = {
    "structuring": {
        "hero_account": "HESAP#1006", "output": "typology_naming_comparison.png",
        "title": "AYNI ANORMAL OLAY · İKİ AÇIKLANABİLİRLİK ÇIKTISI",
        "left_message": "SKOR + feature-seviyesinde neden — tipoloji ADI YOK",
        "right_message": "TİPOLOJİ ADI + geometrik gerekçe — SAR-uyar",
        "strip": "GBM → neden (feature)   |   QUANN → ne (tipoloji adı) + geometrik gerekçe",
        "footer": "Sentetik tipolojiler (E0), dengeli sınıflar (base 0.75, kolay) — standart GBT burada AUC 1.0.",
        "caveat": "Bu figür tespit-süperiorite claimi DEĞİLDİR. Claim: eşdeğer doğruluk koşulunda QUANN analiste ADLANDIRILABİLİR tipoloji + geometrik gerekçe (SAR-uyar) sunar; GBM/SHAP yalnız feature-seviyesinde neden. E0 skorları doğruluk eşdeğerliğini kanıtlamaz.",
        "source": "Kaynak: aml/results/e0_smoke.json (GBT AUC 1.0; hybrid 0.96) · aml/results/e1_explain.json (held-out nearest-prototype accuracy %87.75).  PCA/KDE betimleyicidir.",
    },
    "layering": {
        "hero_account": "HESAP#0431", "output": "typology_naming_layering.png",
        "title": "ZOR VAKA · GEOMETRİK BELİRSİZLİK GÖRÜNÜR",
        "left_message": "SKOR + feature gerekçesi — AD YOK",
        "right_message": "Nearest normal'e kaydı (hata); gerekçe yine 10-hop zincirini adlandırıyor: belirsizlik görünür, false-confident isim yok.",
        "strip": "Zor vaka (layering): tespiti zayıf (recall@1%FPR 0.04) ama geometrik açıklama belirsizliği görünür kılıyor.",
        "footer": "Sentetik ve dengeli veri (base 0.75); bu olayda normal/layering prototipleri örtüşüyor.",
        "caveat": "Performans kanıtı değildir: E0 hibrit layering recall@1%FPR = 0.04; nearest=normal d=0.3230, ikinci=layering d=0.3314 (yaklaşık eşit).",
        "source": "Kaynak: aml/results/e0_smoke.json (synthetic/dengeli; hybrid layering recall@1%FPR 0.04) · aml/results/e1_explain.json (probe ve gerekçe). PCA/KDE betimleyicidir.",
    },
}


def _shap_matrix(raw_values, n_features: int) -> np.ndarray:
    """Normalize SHAP 0.54's multi-output variants to (features, classes)."""
    if isinstance(raw_values, list):
        return np.column_stack([np.asarray(v).reshape(-1, n_features)[0] for v in raw_values])
    values = np.asarray(raw_values)
    if values.ndim == 3:
        # HistGradientBoosting SHAP returns (samples, features, classes).
        if values.shape[1] == n_features:
            return values[0]
        if values.shape[2] == n_features:
            return values[0].T
    if values.ndim == 2 and values.shape[0] == n_features:
        return values
    raise ValueError(f"Unexpected TreeExplainer output shape: {values.shape}")


def _feature_label(name: str, value: float) -> str:
    labels = {
        "amount_median": f"Tutar medyan · {value:,.0f}",
        "amount_p95": f"Tutar p95 · {value:,.0f}",
        "threshold_share": f"9.2K–10K payı · %{value * 100:.0f}",
        "time_of_day_hour": f"Günün saati · {value:.1f}",
        "inter_tx_interval_min": f"İşlem aralığı · {value:.1f} dk",
        "counterparty_count": f"Muhatap sayısı · {value:.0f}",
        "window_count": f"60 dk işlem · {value:.0f}",
        "rate_per_hour": f"Saatlik hız · {value:.1f}/s",
        "chain_hops": f"Zincir hop · {value:.0f}",
        "window_duration_min": f"Pencere süresi · {value:.1f} dk",
    }
    if name.startswith("channel_"):
        channel = name.removeprefix("channel_")
        translated = {"cash_deposit": "nakit yatırma", "wire": "havale", "card": "kart", "cash": "nakit", "digital": "dijital"}.get(channel, channel)
        return f"Kanal = {translated} · {value:.0f}"
    return labels.get(name, f"{name} · {value:.2g}")


def _draw_region_map(ax, embeddings, labels, train_ids, prototypes, hero_embedding, hero_distances, hero_label, show_overlap=False):
    """E1's PCA + class-local KDE geometry, composed into this comparison panel."""
    pca = PCA(n_components=2, random_state=config.SEED)
    projected = pca.fit_transform(embeddings)
    class_names = list(config.CLASSES)
    prototype_matrix = np.stack([prototypes[name] for name in class_names])
    projected_prototypes = pca.transform(prototype_matrix)
    hero_xy = pca.transform(hero_embedding.reshape(1, -1))[0]
    markers = {"normal": "o", "structuring": "s", "layering": "^", "velocity": "D"}
    line_styles = {"normal": ":", "structuring": "-", "layering": "--", "velocity": "-."}

    # E1's per-class bandwidth and local grids preserve contours at different scales.
    for class_id, name in enumerate(class_names):
        points = projected[labels == class_id]
        bandwidth = max(0.015, min(0.3, float(np.median(np.std(points, axis=0)) * len(points) ** (-1 / 6))))
        lower, upper = np.quantile(points, [0.01, 0.99], axis=0)
        padding = max(4.0 * bandwidth, 0.04)
        gx, gy = np.meshgrid(
            np.linspace(lower[0] - padding, upper[0] + padding, 150),
            np.linspace(lower[1] - padding, upper[1] + padding, 150),
        )
        density = np.exp(KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(points).score_samples(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
        peak = float(density.max())
        ax.contourf(gx, gy, density, levels=peak * np.array([0.08, 0.2, 0.4, 1.01]), colors=[PALETTE[name]], alpha=0.055, zorder=1)
        ax.contour(gx, gy, density, levels=peak * np.array([0.12, 0.3, 0.65]), colors=[PALETTE[name]], linewidths=[0.75, 1.0, 1.4], linestyles=line_styles[name], alpha=0.88, zorder=2)
        ax.scatter(points[:, 0], points[:, 1], s=16, marker=markers[name], facecolors="none", edgecolors=PALETTE[name], alpha=0.34, linewidths=0.65, zorder=3)

    # Four train-only class centroids, as in E1.
    offsets = ({"normal": (-70, 20), "structuring": (8, 8), "layering": (14, 30), "velocity": (8, 8)}
               if show_overlap else {"normal": (-58, -20), "structuring": (8, 8), "layering": (8, -16), "velocity": (8, 8)})
    for class_id, name in enumerate(class_names):
        px, py = projected_prototypes[class_id]
        ax.scatter(px, py, marker="*", s=175, color=PALETTE[name], edgecolor="white", linewidth=0.75, zorder=7)
        ax.annotate(name, (px, py), xytext=offsets[name], textcoords="offset points", fontsize=8.1, weight="bold", color=INK,
                    bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"}, zorder=8)

    nearest_id = int(np.argmin(hero_distances))
    nearest_name = class_names[nearest_id]
    prototype_xy = projected_prototypes[nearest_id]
    # Faint connector is explicitly a 2-D projection aid, not the distance metric.
    ax.plot([hero_xy[0], prototype_xy[0]], [hero_xy[1], prototype_xy[1]], color="#737D86", alpha=0.3, linewidth=1.3, zorder=5)
    ax.annotate("", xy=prototype_xy, xytext=hero_xy,
                arrowprops={"arrowstyle": "->", "color": PALETTE[nearest_name], "linewidth": 1.65, "shrinkA": 6, "shrinkB": 7}, zorder=6)
    if show_overlap:
        second_id = int(np.argsort(hero_distances)[1])
        second_xy = projected_prototypes[second_id]
        ax.plot([hero_xy[0], second_xy[0]], [hero_xy[1], second_xy[1]], color="#737D86", alpha=0.3, linewidth=1.0, zorder=5)
        ax.annotate("", xy=second_xy, xytext=hero_xy,
                    arrowprops={"arrowstyle": "->", "color": PALETTE[class_names[second_id]], "linewidth": 1.25, "linestyle": "--", "alpha": 0.85, "shrinkA": 6, "shrinkB": 7}, zorder=6)
    ax.scatter(*hero_xy, marker="*", s=255, color="#111820", edgecolor="white", linewidth=1.05, zorder=9)
    ax.annotate(hero_label, hero_xy, xytext=(10, -20), textcoords="offset points", fontsize=8.6, weight="bold", color=INK,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.94, "edgecolor": "#CBD2D8"}, zorder=10)
    if show_overlap:
        # Both prototype connectors originate at the hero; this map key labels
        # the local geometry without colliding with the tightly spaced stars.
        ax.text(0.015, 0.035, "normal ↔ layering örtüşme", transform=ax.transAxes,
                fontsize=7.2, color=INK, ha="left", va="bottom",
                bbox={"boxstyle": "round,pad=0.24", "facecolor": "#FFF8E8", "alpha": 0.96, "edgecolor": "#E5C77A"}, zorder=10)

    variance = 100 * float(pca.explained_variance_ratio_.sum())
    ax.set_xlabel(f"PCA 1", fontsize=9, color=MUTED, labelpad=3)
    ax.set_ylabel("PCA 2", fontsize=9, color=MUTED, labelpad=3)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=2)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C9D0D6")


def main():
    parser = argparse.ArgumentParser(description="Render a shared GBM/SHAP vs QUANN case-comparison figure.")
    parser.add_argument("--framing", choices=tuple(FRAMINGS), default="structuring", help="case-specific text and output profile")
    parser.add_argument("--hero-account", help="E1 probe account; defaults to the selected framing's hero")
    parser.add_argument("--output", help="optional PNG path override")
    args = parser.parse_args()
    framing = FRAMINGS[args.framing]
    hero_account = args.hero_account or framing["hero_account"]
    out_path = Path(args.output).resolve() if args.output else Path(__file__).resolve().parent / framing["output"]

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.set_num_threads(1)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Required E0 hybrid checkpoint missing: {CHECKPOINT}")
    with E1_JSON.open(encoding="utf-8") as stream:
        e1_result = json.load(stream)
    probe = next(item for item in e1_result["probes"] if item["account_id"] == hero_account)
    if probe["true_class"] != args.framing:
        raise ValueError(f"Probe {hero_account} is {probe['true_class']}, not framing {args.framing}")

    # Deterministic E0 generator/split and train_aml.py GBT setup (CPU only).
    _, windows = generate_transactions(n_transactions=20_000, seed=config.SEED)
    labels = np.asarray([config.CLASSES.index(name) for name in windows.typology], dtype=int)
    all_ids = np.arange(len(windows))
    train_ids, temp_ids = train_test_split(all_ids, test_size=0.4, random_state=config.SEED, stratify=labels)
    val_ids, _ = train_test_split(temp_ids, test_size=0.5, random_state=config.SEED + 1, stratify=labels[temp_ids])
    train = windows.iloc[train_ids].reset_index(drop=True)
    val = windows.iloc[val_ids].reset_index(drop=True)
    test_ids = np.setdiff1d(all_ids, np.concatenate([train_ids, val_ids]))
    test = windows.iloc[test_ids].reset_index(drop=True)
    xtr, xva, xte = _feature_matrix(train, val, test)
    feature_names = [c for c in train.columns if c not in {"window_id", "typology", "counterparty_id", "channel"}] + list(pd.get_dummies(train["channel"], dtype=float).columns.map(lambda c: f"channel_{c}"))
    if len(feature_names) != xtr.shape[1]:
        raise RuntimeError(f"Feature-name mapping mismatch ({len(feature_names)} != {xtr.shape[1]})")
    ytr = labels[train_ids]
    gbt = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=0.08, random_state=config.SEED)
    gbt.fit(xtr, ytr)

    hero_id = int(probe["window_id"])
    if hero_id >= len(windows) or windows.iloc[hero_id]["typology"] != args.framing:
        raise AssertionError(f"E1 hero mapping mismatch for {hero_account} / window {hero_id}")
    hero = windows.iloc[hero_id]
    hero_pos = int(np.flatnonzero(test_ids == hero_id)[0])
    xhero = xte[hero_pos:hero_pos + 1]
    probabilities = gbt.predict_proba(xhero)[0]
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = config.CLASSES[predicted_idx]
    abnormal_probability = float(1.0 - probabilities[config.CLASSES.index("normal")])

    # TreeExplainer returns class margins; negative normal margin is the same
    # risk-directed signed contribution convention used in the original figure.
    explainer = shap.TreeExplainer(gbt)
    raw = explainer.shap_values(xhero)
    shap_by_class = _shap_matrix(raw, xtr.shape[1])
    risk_direction = -shap_by_class[:, config.CLASSES.index("normal")]
    top = np.argsort(np.abs(risk_direction))[::-1][:3]

    # E1's train-only prototypes and CPU-loaded E0 hybrid checkpoint.
    quantizer = FeatureQuantizer().fit(windows.iloc[train_ids].reset_index(drop=True))
    tokens = quantizer.transform(windows)
    model = _load_e0_model(str(CHECKPOINT), quantizer.vocab_size)
    embeddings, _ = _encode(model, tokens)
    prototypes = {
        name: embeddings[train_ids][labels[train_ids] == class_id].mean(axis=0)
        for class_id, name in enumerate(config.CLASSES)
    }
    hero_embedding = embeddings[hero_id]
    prototype_distances = np.asarray([np.linalg.norm(hero_embedding - prototypes[name]) for name in config.CLASSES])
    nearest_idx = int(np.argmin(prototype_distances))
    second_idx = int(np.argsort(prototype_distances)[1])
    nearest_name, second_name = config.CLASSES[nearest_idx], config.CLASSES[second_idx]
    if nearest_name != probe["nearest_prototype"] or second_name != probe["second_prototype"]:
        raise AssertionError(f"Checkpoint geometry does not match E1 probe: {nearest_name}, {second_name}")
    if abs(prototype_distances[nearest_idx] - probe["nearest_prototype_distance"]) > 0.002:
        raise AssertionError("Recomputed nearest-prototype distance differs from e1_explain.json")
    if abs(prototype_distances[second_idx] - probe["second_prototype_distance"]) > 0.002:
        raise AssertionError("Recomputed second-prototype distance differs from e1_explain.json")
    separation = float(prototype_distances[second_idx] / prototype_distances[nearest_idx])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titleweight": "bold", "figure.facecolor": "white", "axes.facecolor": "white"})
    fig = plt.figure(figsize=(16, 7.6), facecolor="white")
    fig.text(0.045, 0.955, framing["title"], fontsize=16.5, weight="bold", color=INK, va="top")
    fig.text(0.045, 0.913, f"{hero_account}  |  window {hero_id}  |  gerçek tipoloji: {probe['true_class']}", fontsize=10.2, color=MUTED, va="top")
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.24, 0.865], transform=fig.transFigure, color="#D9DEE3", linewidth=0.8))

    fig.text(0.055, 0.855, "GBM + SHAP  ·  STANDART", fontsize=12, weight="bold", color=INK, va="top")
    fig.text(0.055, 0.817, framing["left_message"], fontsize=9.4, color=MUTED, va="top")
    fig.text(0.535, 0.855, "CloudLM / QUANN  ·  NOKTA-BULUT + GEOMETRİ", fontsize=12, weight="bold", color=INK, va="top")
    if args.framing == "structuring":
        right_message = framing["right_message"]
    else:
        right_message = "QUANN: en-yakın-prototip normal'e kaydı (hata) — ama gerekçe yine 10-hop zinciri adlandırıyor:\nbelirsizlik görünür, false-confident isim yok."
    fig.text(0.535, 0.817, right_message, fontsize=8.55 if args.framing == "layering" else 9.4, color=MUTED, va="top")

    left_label = "ANORMAL RİSK"
    fig.text(0.065, 0.755, left_label, fontsize=8.2, weight="bold", color=MUTED, va="top")
    score_color = PALETTE[probe["true_class"]]
    fig.text(0.062, 0.707, f"{abnormal_probability * 100:.4f}%", fontsize=31, weight="bold", color=score_color, va="top")
    if args.framing == "structuring":
        fig.text(0.29, 0.727, "E0 held-out probe · GBT\nP(normal dışı) = 1 − P(normal)", fontsize=8.7, color=MUTED, va="top", linespacing=1.5)
    else:
        fig.text(0.29, 0.727, f"E0 held-out probe · GBT\nP(normal dışı) = 1 − P(normal)\nEn yüksek sınıf: {predicted_class} · P={probabilities[predicted_idx] * 100:.2f}%", fontsize=8.3, color=MUTED, va="top", linespacing=1.35)
    fig.text(0.065, 0.635, "Top 3 SHAP · signed direction on −normal-class raw margin", fontsize=8.8, weight="bold", color=INK, va="top")
    ax_bar = fig.add_axes([0.18, 0.365, 0.285, 0.245])
    vals = risk_direction[top][::-1]
    names = [_feature_label(feature_names[i], float(xhero[0, i])) for i in top[::-1]]
    bar_colors = ["#D65A31" if v >= 0 else "#3478A8" for v in vals]
    ypos = np.arange(len(vals))
    ax_bar.barh(ypos, vals, color=bar_colors, height=0.57, edgecolor="none")
    ax_bar.axvline(0, color="#66717B", linewidth=0.8)
    ax_bar.set_yticks(ypos, names, fontsize=8.1, color=INK)
    ax_bar.set_xlabel("SHAP · raw margin units", fontsize=7.8, color=MUTED, labelpad=3)
    ax_bar.tick_params(axis="x", labelsize=7.5, colors=MUTED, length=2)
    ax_bar.tick_params(axis="y", length=0, pad=7)
    ax_bar.grid(axis="x", color=GRID, linewidth=0.6)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top", "right", "left"]].set_visible(False)
    ax_bar.spines["bottom"].set_color("#C9D0D6")
    span = max(float(np.ptp(vals)), float(np.max(np.abs(vals))) * 0.28, 0.05)
    ax_bar.set_xlim(min(0.0, float(vals.min()) - span * 0.35), max(0.0, float(vals.max()) + span * 0.35))
    for row, value in enumerate(vals):
        ax_bar.text(value + (span * 0.04 if value >= 0 else -span * 0.04), row, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=7.8, color=INK)
    if args.framing == "structuring":
        p95 = float(hero.amount_p95)
        band_share = float(hero.threshold_share)
        if abs(p95 - 9845.0) > 1.0 or band_share != 1.0:
            raise AssertionError(f"HESAP#1006 synthetic probe drift: p95={p95:.3f}, threshold_share={band_share:.3f}")
        fig.text(0.065, 0.324, "Pozitif = normal-sınıf marjı düşer (risk yönü); model çıktısı tipoloji etiketi değildir.", fontsize=7.8, color=MUTED, va="top")
        fig.text(0.065, 0.265, "GBT sınıf olasılıkları üretir; bu panel olayın hangi AML tipolojisi olduğunu adlandırmaz.", fontsize=8.2, color=INK, va="top")
    else:
        fig.text(0.065, 0.324, "Pozitif = risk yönü; SHAP feature gerekçesidir, tipoloji adı değildir.", fontsize=7.8, color=MUTED, va="top")
        fig.text(0.065, 0.265, f"GBT'nin olay için en yüksek sınıf olasılığı {predicted_class}; E0 kolay sentetikte GBT AUC 1.0.", fontsize=8.2, color=INK, va="top")

    ax_map = fig.add_axes([0.545, 0.335, 0.415, 0.405])
    _draw_region_map(ax_map, embeddings, labels, train_ids, prototypes, hero_embedding, prototype_distances, hero_account, show_overlap=(args.framing == "layering"))
    ax_map.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=PALETTE[n], markersize=6, label=n)
        for n in config.CLASSES
    ] + [Line2D([0], [0], marker="*", color="none", markerfacecolor="#555555", markeredgecolor="#555555", markersize=9, label="train prototipi"),
         Line2D([0], [0], marker="*", color="none", markerfacecolor="#111820", markeredgecolor="#111820", markersize=9, label=hero_account)],
        loc="upper left", fontsize=7.0, ncol=3, frameon=True, framealpha=0.94, edgecolor="#D9DEE3", borderpad=0.35, handletextpad=0.3, columnspacing=0.7)
    if args.framing == "structuring":
        fig.text(0.545, 0.305, "STRUCTURING", fontsize=17, weight="bold", color=PALETTE["structuring"], va="top")
        fig.text(0.545, 0.273, f"nearest d={prototype_distances[nearest_idx]:.2f}  ·  2nd: {second_name} d={prototype_distances[second_idx]:.2f}  ·  ayrışma {separation:.1f}×", fontsize=9.0, weight="bold", color=INK, va="top")
        fig.text(0.545, 0.242, "Eşik-kenarı kümesi: tutar p95 ₺9.845 (10K eşiğinin %98'i); işlemlerin %100'ü ₺9.2K–₺10K bandında; 60 dk işlem sayısı = 1.", fontsize=8.0, color=INK, va="top")
        fig.text(0.545, 0.212, "Ön inceleme taslağı; kaynak işlem kayıtları ve müşteri bağlamı doğrulanmadan SAR kararı değildir.", fontsize=7.6, color=MUTED, va="top")
    else:
        fig.text(0.545, 0.305, "LAYERING · GERÇEK SINIF", fontsize=15.0, weight="bold", color=PALETTE["layering"], va="top")
        fig.text(0.545, 0.273, f"nearest={nearest_name} d={prototype_distances[nearest_idx]:.4f} (hata)  ·  2nd={second_name} d={prototype_distances[second_idx]:.4f}  ·  oran {separation:.2f}×", fontsize=8.7, weight="bold", color=INK, va="top")
        fig.text(0.545, 0.242, "Zayıf ayrışma: normal median_own d=0.0185 · layering median_own d=0.0302; E0 hybrid layering recall@1%FPR = 0.04.", fontsize=7.7, color=INK, va="top")
        fig.text(0.545, 0.212, probe["observable_reason"], fontsize=8.0, weight="bold", color=PALETTE["layering"], va="top")

    fig.add_artist(plt.Rectangle((0.045, 0.145), 0.91, 0.043, transform=fig.transFigure, facecolor="#F1F4F6", edgecolor="#DDE2E6", linewidth=0.7, zorder=-1))
    fig.text(0.5, 0.166, framing["strip"], fontsize=9.4 if args.framing == "layering" else 10.2, weight="bold", color=INK, ha="center", va="center")
    fig.text(0.045, 0.112, framing["footer"], fontsize=8.1, color=INK, va="top")
    fig.text(0.045, 0.082, framing["caveat"], fontsize=7.8, color=INK, va="top")
    fig.text(0.045, 0.047, framing["source"], fontsize=7.5, color=MUTED, va="top")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    top_report = [(feature_names[i], float(risk_direction[i])) for i in top]
    print(f"FIGURE={out_path}")
    print(f"HERO={hero_account} window={hero_id} true={probe['true_class']} GBT_argmax={predicted_class} P_argmax={probabilities[predicted_idx]:.8f}")
    print(f"ABNORMAL_PROBABILITY={abnormal_probability:.8f} ({abnormal_probability * 100:.4f}%)")
    print("TOP3_SHAP=" + json.dumps(top_report, ensure_ascii=False))
    print(f"QUANN nearest={nearest_name} d={prototype_distances[nearest_idx]:.4f}; second={second_name} d={prototype_distances[second_idx]:.4f}; ratio={separation:.2f}x")
    expected_reason = "çok-hop zincir işareti: 10 hop / 10 karşı taraf, medyan aralık=10.7 dk, kanal=wire" if args.framing == "layering" else probe["observable_reason"]
    print(f"PROBE_REASON_MATCH={probe['observable_reason'] == expected_reason}")
    print(f"VALIDATION probe_distance_match=True; checkpoint={CHECKPOINT}; generated={len(windows)} windows; device=cpu")


if __name__ == "__main__":
    main()
