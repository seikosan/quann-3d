#!/usr/bin/env python3
"""Patent v3 figure set — black-and-white SVGs for TÜRKPATENT filing.

Kural (BLUS_NOTU §4): ölçümlü figürler gerçek JSON/log değerlerinden çizilir;
şematik figürler künyesinde "şematiktir" taşır. Elle yazılmış sayı YOKTUR.
Gerçek veri kaynakları komut içinde dosya adı + alan adıyla belirtilir.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parent / "fig"
FIG.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "svg.fonttype": "none",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }
)
INK = "black"


def _box(ax, x, y, w, h, text, fs=10, style="round,pad=0.02", lw=1.2, fc="white"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                       edgecolor=INK, facecolor=fc)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1, dashed=False):
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                        linewidth=1.1, color=INK,
                        linestyle=(0, (4, 3)) if dashed else "solid")
    )


def _schematic_canvas(ax, title, note="Şematik çizimdir; ölçekte değildir."):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    ax.text(0.01, 0.015, note, transform=ax.transAxes, fontsize=8, style="italic")


def fig1_sistem_mimarisi() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    _schematic_canvas(ax, "Şekil 1 — Sistem mimarisi: tek buluttan görev çıktısı ve açıklama")
    _box(ax, 0.2, 3.2, 1.5, 1.0, "Girdi\n(metin satırı /\ntabular satır)", 9)
    _box(ax, 2.1, 3.2, 1.5, 1.0, "Bağlam\nkodlayıcı", 10)
    _box(ax, 4.0, 3.2, 1.6, 1.0, "D-boyutlu\nprojeksiyon", 10)
    _box(ax, 6.0, 3.4, 1.5, 0.7, "Nokta\nbulutu P", 10, fc="0.92")
    _box(ax, 5.7, 1.9, 2.1, 0.9, "Çift-geometri +\nsimetrik agregasyon", 9)
    _box(ax, 8.1, 2.0, 1.7, 0.7, "Okuyucu\n→ görev çıktısı", 9)
    _box(ax, 4.0, 0.35, 5.8, 0.85,
         "Açıklanabilirlik üçlüsü: (a) nokta çıkarma etkisi · (b) bölge isimlendirme · (c) bölge kayması", 9)
    _arrow(ax, 1.7, 3.7, 2.1, 3.7)
    _arrow(ax, 3.6, 3.7, 4.0, 3.7)
    _arrow(ax, 5.6, 3.7, 6.0, 3.7)
    _arrow(ax, 6.6, 3.4, 6.6, 2.8)
    _arrow(ax, 7.8, 2.35, 8.1, 2.35)
    _arrow(ax, 6.0, 3.4, 4.7, 1.32, dashed=True)
    _arrow(ax, 6.9, 1.9, 6.5, 1.32, dashed=True)
    fig.savefig(FIG / "fig1_sistem_mimarisi.svg", bbox_inches="tight")
    plt.close(fig)


def fig2_cift_geometri() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    _schematic_canvas(ax, "Şekil 2 — Nokta çifti ve çift-geometri özelliğinin bileşenleri")
    pts = [(1.2, 4.3, "p$_i$"), (3.4, 4.7, "p$_j$"), (1.9, 3.3, ""), (3.0, 3.5, ""), (2.4, 4.0, "")]
    for x, y, lab in pts:
        ax.plot(x, y, "o", ms=6, color=INK)
        if lab:
            ax.text(x - 0.1, y + 0.25, lab, fontsize=11)
    _arrow(ax, 1.35, 4.35, 3.25, 4.65)
    ax.text(2.3, 4.75, "Δp", fontsize=10)
    ax.plot([1.2, 3.4], [4.3, 4.7], ":", color=INK, lw=1)
    ax.text(2.0, 4.22, "‖Δp‖", fontsize=9)
    comp = [("fark\nΔp", "D"), ("mesafe\n‖Δp‖", "1"), ("kosinüs\nθ", "1"),
            ("norm\n‖p$_i$‖", "D"), ("norm\n‖p$_j$‖", "D")]
    x = 4.8
    for name, dim in comp:
        _box(ax, x, 4.05, 0.92, 0.72, f"{name}\n({dim})", 8)
        x += 1.0
    _box(ax, 4.8, 2.95, 4.92, 0.72,
         "e$_{ij}$ = [Δp, ‖Δp‖, cosθ, ‖p$_i$‖, ‖p$_j$‖]\n$\\in$ R$^{3D+2}$", 9)
    ax.text(4.82, 2.83, "Genişlik ölçülen uygulamanın kod tanımıdır (Tarifname §5.3).",
            fontsize=7, style="italic", ha="left", va="top")
    _arrow(ax, 7.2, 4.05, 7.2, 3.67)
    _box(ax, 4.8, 1.55, 4.92, 0.6, "çift MLP → çift özellikleri → ortalama havuzlama", 9)
    _arrow(ax, 7.2, 2.95, 7.2, 2.15)
    fig.savefig(FIG / "fig2_cift_geometri.svg", bbox_inches="tight")
    plt.close(fig)


def fig3_simetrik_agregasyon() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    _schematic_canvas(ax, "Şekil 3 — Simetrik agregasyon ve O(N) küresel-istatistik varyantı")
    _box(ax, 0.3, 4.2, 1.7, 0.8, "nokta özellikleri\nh_i = f(p$_i$)", 9)
    _box(ax, 0.3, 2.6, 1.7, 0.8, "çift özellikleri\ne$_{ij}$", 9)
    _box(ax, 2.6, 4.2, 1.7, 0.8, "maksimum\nhavuzlama", 10)
    _box(ax, 2.6, 2.6, 1.7, 0.8, "ortalama\nhavuzlama", 10)
    _box(ax, 5.0, 3.4, 1.5, 0.8, "birleştirme\n(katenim)", 10)
    _box(ax, 7.2, 3.4, 1.6, 0.8, "küme\ngösterimi z", 10, fc="0.92")
    _arrow(ax, 2.0, 4.6, 2.6, 4.6)
    _arrow(ax, 2.0, 3.0, 2.6, 3.0)
    _arrow(ax, 4.3, 4.5, 5.0, 4.0)
    _arrow(ax, 4.3, 3.0, 5.0, 3.6)
    _arrow(ax, 6.5, 3.8, 7.2, 3.8)
    _box(ax, 2.6, 0.7, 4.3, 0.9,
         "varyant: Transformer ve tam çift hesabı olmadan O(N) küresel istatistikler\n(merkezî değer ·varyans ·maksimum ·minimum)", 8.5)
    _arrow(ax, 2.0, 2.6, 2.6, 1.35, dashed=True)
    _arrow(ax, 6.9, 1.15, 7.6, 3.4, dashed=True)
    fig.savefig(FIG / "fig3_simetrik_agregasyon.svg", bbox_inches="tight")
    plt.close(fig)


def fig4_nokta_cikarma() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    _schematic_canvas(ax, "Şekil 4 — Nokta çıkarma etkisi (K+1 ileri geçiş)")
    _box(ax, 0.3, 3.6, 2.0, 0.9, "nokta bulutu P\n{p$_1$ … p$_K$}", 9, fc="0.92")
    _box(ax, 3.0, 3.7, 1.6, 0.7, "okuyucu g(P)", 10)
    _box(ax, 5.3, 3.7, 1.6, 0.7, "skor s$_{tam}$", 10)
    _box(ax, 0.3, 1.9, 2.0, 0.9, "P \\ {p$_k$}\n(bir nokta çıkarılır)", 9)
    _box(ax, 3.0, 2.0, 1.6, 0.7, "okuyucu g(P′)", 10)
    _box(ax, 5.3, 2.0, 1.6, 0.7, "skor s$_{-k}$", 10)
    _box(ax, 7.4, 1.9, 2.3, 0.9, "açıklama: etki\ne$_k$ = s$_{tam}$ − s$_{-k}$", 9, fc="0.92")
    _arrow(ax, 2.3, 4.05, 3.0, 4.05)
    _arrow(ax, 4.6, 4.05, 5.3, 4.05)
    _arrow(ax, 2.3, 2.35, 3.0, 2.35)
    _arrow(ax, 4.6, 2.35, 5.3, 2.35)
    _arrow(ax, 6.9, 3.9, 7.9, 2.8)
    _arrow(ax, 6.9, 2.35, 7.4, 2.35)
    ax.text(0.3, 1.0,
            "K nokta için K+1 geçiş; yüksek |e$_k$| → kararın dayandığı noktalar (karşı-olgusal etki).",
            fontsize=8.5, style="italic")
    fig.savefig(FIG / "fig4_nokta_cikarma.svg", bbox_inches="tight")
    plt.close(fig)


def fig5_bolge_isimlendirme() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    _schematic_canvas(ax, "Şekil 5 — Bölge/prototip isimlendirme akışı")
    _box(ax, 0.3, 2.7, 1.7, 0.9, "eğitim bulutları\n(özellik → P)", 9)
    _box(ax, 2.5, 2.7, 1.8, 0.9, "kümeleme\n(KMeans)", 10)
    _box(ax, 4.8, 2.7, 1.9, 0.9, "bölge istatistikleri\n(prototip + pay)", 9)
    _box(ax, 7.2, 2.7, 2.5, 0.9, "insan-okunur bölge\netiketi / neden cümlesi", 9, fc="0.92")
    _arrow(ax, 2.0, 3.15, 2.5, 3.15)
    _arrow(ax, 4.3, 3.15, 4.8, 3.15)
    _arrow(ax, 6.7, 3.15, 7.2, 3.15)
    import random

    random.seed(7)
    groups = [(1.4, 1.4), (4.2, 1.1), (7.0, 1.5)]
    labels = ["Bölge A", "Bölge B", "Bölge C"]
    for gx, gy in groups:
        xs = [gx + random.uniform(-0.8, 0.8) for _ in range(18)]
        ys = [gy + random.uniform(-0.55, 0.55) for _ in range(18)]
        ax.plot(xs, ys, "o", ms=3, color=INK, alpha=0.65)
        ax.plot(gx, gy, "x", ms=9, mew=1.6, color=INK)
    for (gx, gy), lab in zip(groups, labels):
        ax.text(gx, gy - 1.0, lab, fontsize=8.5, ha="center")
    ax.text(8.9, 1.4, "nokta dağılımı\ntemsilîdir", fontsize=7.5, ha="center", style="italic")
    fig.savefig(FIG / "fig5_bolge_isimlendirme.svg", bbox_inches="tight")
    plt.close(fig)


def fig6_bolge_kaymasi() -> None:
    src = REPO / "complaints/results/complaints_scale_stability.json"
    deltas = json.loads(src.read_text())["share_delta_drift_minus_train"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.6))
    regions = sorted(int(k) for k in deltas)
    vals = [deltas[str(r)] for r in regions]
    ax1.bar([str(r) for r in regions], vals, color="white", edgecolor=INK, hatch="//")
    ax1.axhline(0, color=INK, lw=0.8)
    lo = min(min(vals) - 0.06, -0.14)
    ax1.set_ylim(lo, max(vals) + 0.05)
    ax1.set_xlabel("bölge indeksi")
    ax1.set_ylabel("pay farkı (drift − eğitim)")
    ax1.set_title("(a) Bölge nokta-payı kayması", fontsize=10)
    for r, v in zip(regions, vals):
        ax1.text(str(r), v + (0.012 if v >= 0 else -0.028), f"{v:+.3f}",
                 ha="center", fontsize=7)
    msrc = REPO / "maintenance/results/e1_cmapss.json"
    demo = json.loads(msrc.read_text())["drift_demo"]
    rates = [demo["unchanged_alarm_rate"], demo["shifted_alarm_rate"]]
    ax2.bar(["değişmeyen\nayar", "kaydırılmış\nayar"], rates, color="white",
            edgecolor=INK, hatch="\\")
    ax2.set_title("(b) Ayar kayması → alarm artışı", fontsize=10)
    ax2.text(1.03, 0.5, "alarm oranı", transform=ax2.transAxes, rotation=90,
             ha="left", va="center", fontsize=11)
    ax2.set_ylim(0, max(rates) * 1.25)
    for i, v in enumerate(rates):
        ax2.text(i, v + max(rates) * 0.03, f"{v:.4f}", ha="center", fontsize=8)
    ax2.text(0.03, 0.93, f"orantı {demo['shifted_to_unchanged_alarm_rate_ratio']}×",
             transform=ax2.transAxes, ha="left", va="top", fontsize=8.5, style="italic")
    fig.subplots_adjust(bottom=0.24)
    fig.text(0.01, 0.015,
             "Veri: complaints/results/complaints_scale_stability.json (share_delta_drift_minus_train) · "
             "maintenance/results/e1_cmapss.json (drift_demo)",
             fontsize=6.5, style="italic")
    fig.savefig(FIG / "fig6_bolge_kaymasi.svg", bbox_inches="tight")
    plt.close(fig)


def fig7_int8_paket() -> None:
    src = REPO / "fraud/results/edge_eval_full_scale.json"
    data = json.loads(src.read_text())
    auc = data["int8_auc"]
    fp = data["footprint"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.6))
    names = ["FP32", "INT8\ntensor", "INT8\nkanal"]
    vals = [auc["auc_fp32"], auc["auc_int8_tensor"], auc["auc_int8_channel"]]
    bars = ax1.bar(names, vals, color="white", edgecolor=INK, hatch=["", "//", "\\\\"] )
    ax1.set_ylim(0.90, 0.966)
    ax1.set_title("(a) Kantizasyon → AUC (n={:,})".format(auc["n_test"]).replace(",", "."),
                  fontsize=10)
    ax1.text(-0.21, 0.5, "AUC-ROC", transform=ax1.transAxes, rotation=90,
             ha="left", va="center", fontsize=11)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.4f}",
                 ha="center", fontsize=8.5)
    ax1.tick_params(axis="x", labelsize=9)
    kb = [fp["deploy_fp32_kb"], fp["deploy_int8_kb"]]
    bars2 = ax2.bar(["FP32 paket", "INT8 paket"], kb, color="white",
                    edgecolor=INK, hatch=["", "//"])
    ax2.set_ylim(0, 620)
    ax2.set_title("(b) Uç paket ayak izi", fontsize=10)
    ax2.text(1.02, 0.5, "dağıtım boyutu (KB)", transform=ax2.transAxes, rotation=90,
             ha="left", va="center", fontsize=11)
    for b, v in zip(bars2, kb):
        ax2.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.1f} KB", ha="center", fontsize=8.5)
    fig.subplots_adjust(bottom=0.24)
    fig.text(0.01, 0.015,
             "Veri: fraud/results/edge_eval_full_scale.json (int8_auc, footprint)",
             fontsize=6.5, style="italic")
    fig.savefig(FIG / "fig7_int8_paket.svg", bbox_inches="tight")
    plt.close(fig)


def fig8_window_bulut() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    _schematic_canvas(ax, "Şekil 8 — Çoklu-cümle bağlam: cümle-bulut ve pencere-bulut hiyerarşisi")
    for i, x in enumerate([0.4, 2.9, 5.4]):
        _box(ax, x, 4.3, 1.9, 0.7, f"cümle {i + 1}", 9)
        _box(ax, x + 0.25, 3.1, 1.4, 0.65, "cümle-bulutu", 8)
        _arrow(ax, x + 0.95, 4.3, x + 0.95, 3.75)
    for x in (1.3, 3.8):
        _arrow(ax, x, 3.1, x + 1.6, 2.15, dashed=True)
    _arrow(ax, 6.35, 3.1, 6.35, 2.15, dashed=True)
    _box(ax, 2.9, 1.45, 4.2, 0.65, "komşu pencere-bulut (pencere boyunca geometri)", 9)
    _box(ax, 3.5, 0.45, 3.0, 0.65, "ContextCloudLM → görev çıktısı", 9, fc="0.92")
    _arrow(ax, 5.0, 1.45, 5.0, 1.1)
    ax.text(7.35, 1.6, "pencere = W cümle", fontsize=8, style="italic")
    fig.savefig(FIG / "fig8_window_bulut.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig1_sistem_mimarisi()
    fig2_cift_geometri()
    fig3_simetrik_agregasyon()
    fig4_nokta_cikarma()
    fig5_bolge_isimlendirme()
    fig6_bolge_kaymasi()
    fig7_int8_paket()
    fig8_window_bulut()
    for p in sorted(FIG.glob("fig*.svg")):
        print(p.relative_to(REPO), p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
