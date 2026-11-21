#!/usr/bin/env python3
"""pct_v2 PCT drawing set — black-and-white SVGs, English labels, "FIG. n" sheets.

Neden ayri betik: PCT basvurusu Ingilizce asil ile yapilir; final_v3 cizimleri Turkce
basliklidir (TURKPATENT TR nusyasi). Bu betik ayni 8 bicimin Ingilizce, sayfa-etiketli
("FIG. n") karsiligidir.

Terim kurali: cizimdeki HER sozcuk DESCRIPTION_EN.md tarifnamesinin kendi terimidir
(paragraf numaralari [0029], [0033], [0034], [0038], [0040]-[0041], [0037]). Tarifnamede
gecmeyen terim yazilamaz; referans isareti yoktur (UYGUNLUK_NOTU satir 30, 106).

Olcu kurali (DESCRIPTION_EN.md satir 87 [0028]): FIG. 1-5 ve FIG. 8 sematiktir, olcu
tassimaz; olcuyu yalniz FIG. 6 ve FIG. 7 tasir ve gercek JSON degerlerinden cizilir.
Yariizci uretim: olculu sekillerde elle yazilmis sayi YOKTUR; sematik sekillerde sayi
yoktur. Kaynak dogrulama: dogrul_figurler.py.

Kullanim:  python3 docs/patent/pct_v2/generate_figures_pct.py
Cikti:     docs/patent/pct_v2/fig/fig1..fig8.svg  + stdout'ta dosya listesi
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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


SAFE_W_CM, SAFE_H_CM = 16.0, 24.7   # build_drawings.py: A4 eksi 2,5 cm kenar boslugu
MIN_TEXT_CM = 0.21                   # PCT Kural 11.13: yazi yuksekligi asgari 0,21 cm
LINE_SPACING = 1.2                   # matplotlib varsayilan satir araligi
BOX_PAD_PX = 7.0                     # kutu icinde her yanda birakilan bosluk


def _pt_to_cm(fig) -> float:
    """SVG'deki 1 yazi-puaninin basili A4'teki karsiligi (cm), muhafazakar.

    build_drawings.py cizimi SAFE_W_CM genislige ortuyor; tight-bbox SVG'yi
    kisalttigi icin gercek oran biraz daha buyuktur — bu tahmin kurali sertlestirir.
    """
    return SAFE_W_CM / (fig.get_size_inches()[0] * 72.0)


def _min_pt(fig) -> float:
    """MIN_TEXT_CM'nin bu tuval icin nominal yazi boyutu karsiligi."""
    return round(MIN_TEXT_CM / _pt_to_cm(fig), 2)


def _measure(ax, s: str, fs: float, renderer) -> float:
    """Bir yazi parcasinin ekran-genisligi (px); gorunmeden olc, sonra sil."""
    t = ax.text(0, 0, s, fontsize=fs, alpha=0.0)
    px = t.get_window_extent(renderer=renderer).width
    t.remove()
    return px


def _wrap(ax, text: str, fs: float, max_px: float, renderer) -> list[str]:
    """Kutu genisligine acgoylu sarma; mevcut satir sonlari zorunlu kirilimdir."""
    lines: list[str] = []
    for hard in text.split("\n"):
        words = hard.split()
        if not words:
            lines.append("")
            continue
        cur = words[0]
        for word in words[1:]:
            trial = f"{cur} {word}"
            if _measure(ax, trial, fs, renderer) <= max_px:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def _box(ax, x, y, w, h, text, fs=9.0, lw=1.2, fc="white"):
    """Kutuyu verili metne gore cizer: yazi kutuya sigmiyorsa sarilir, gerekirse
    PCT asgari boyutuna kadar kuculur; yine sigmiyorsa _FITS'e kiril yazilir."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    ox, oy = ax.transData.transform((0.0, 0.0))
    bw = ax.transData.transform((w, 0.0))[0] - ox
    bh = ax.transData.transform((0.0, h))[1] - oy
    floor = _min_pt(fig)
    sizes = sorted({max(fs, floor), floor}, reverse=True)
    chosen = None
    for size in sizes:
        lines = _wrap(ax, text, size, bw - 2 * BOX_PAD_PX, renderer)
        widest = max((_measure(ax, ln, size, renderer) for ln in lines), default=0.0)
        tall = len(lines) * size * fig.dpi / 72.0 * LINE_SPACING
        if widest <= bw - 2 * BOX_PAD_PX and tall <= bh - 4.0:
            chosen = (size, lines)
            break
    if chosen is None:
        size = floor
        lines = _wrap(ax, text, size, bw - 2 * BOX_PAD_PX, renderer)
        _FITS.append(f"{_CUR['sheet']}: kutuya sigmiyor '{text}' "
                     f"({bw:.0f}x{bh:.0f}px, {len(lines)} satir @ {size}pt)")
        chosen = (size, lines)
    size, lines = chosen
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=lw,
                       edgecolor=INK, facecolor=fc)
    )
    t = ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
                fontsize=size, linespacing=LINE_SPACING)
    t._pct_box = (w, h)


def _arrow(ax, x0, y0, x1, y1, dashed=False):
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                        linewidth=1.1, color=INK,
                        linestyle=(0, (4, 3)) if dashed else "solid")
    )


def _route(ax, points, dashed=False, head=True):
    """Kutularin etrafindan dolanan kirik linha; son parca ok basi tasir."""
    for (x0, y0), (x1, y1) in zip(points, points[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=INK, lw=1.1,
                linestyle=(0, (4, 3)) if dashed else "solid")
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>" if head else "-",
                        mutation_scale=12, linewidth=1.1, color=INK,
                        linestyle=(0, (4, 3)) if dashed else "solid")
    )


SHEETS: list[dict] = []
_TEXTS: list[dict] = []
_FITS: list[str] = []
_TIGHT: list[tuple] = []
_CUR: dict = {"sheet": "", "svg": "", "measured": False, "sources": []}


def _start(sheet: str, svg: str, measured: bool = False) -> None:
    """Bir bicimin metin/fit kaydini sifirlar: manifest bu kayitlardan uretilir."""
    _CUR.update(sheet=sheet, svg=svg, measured=measured, sources=[], source="")
    _TEXTS.clear()


def _kind(t) -> str:
    if hasattr(t, "_pct_box"):
        return "box"
    if t.get_fontweight() == "bold":
        return "figure-designation"
    if t.get_style() == "italic":
        return "note"
    return "label"


def _entry(t) -> dict:
    e = {"text": " ".join(t.get_text().split()), "kind": _kind(t),
         "pt": round(float(t.get_fontsize()), 2)}
    if hasattr(t, "_pct_value"):
        e["kind"] = "measured-value"
        e["raw_value"] = t._pct_value
        e["source"] = getattr(t, "_pct_src", _CUR["source"])
        if hasattr(t, "_pct_formula"):
            e["formula"] = t._pct_formula
    return e


def _check_fits(fig) -> None:
    """Kutu yazisinin kutuya sigip sigmadigini piksel olcuyle denetler (tasarsal tahmin degil)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for t in ax.texts:
            dim = getattr(t, "_pct_box", None)
            if not dim:
                continue
            ox, oy = ax.transData.transform((0.0, 0.0))
            bw = abs(ax.transData.transform((dim[0], 0.0))[0] - ox)
            bh = abs(ax.transData.transform((0.0, dim[1]))[1] - oy)
            ext = t.get_window_extent(renderer=renderer)
            _TIGHT.append((ext.width / bw, ext.height / bh, _CUR["sheet"],
                           " ".join(t.get_text().split())))
            if ext.width > bw - 6.0 or ext.height > bh - 4.0:
                _FITS.append(f"{_CUR['sheet']}: '{t.get_text()}' yazi {ext.width:.0f}x"
                             f"{ext.height:.0f}px > kutu {bw:.0f}x{bh:.0f}px")


COLLIDE_AREA = 24.0   # px^2 — kirpik cizgi (1,2 px) temasini degil, gercek carismayi yakalar


def _overlap(a, b) -> float:
    """Iki ekran-dikdortgeninin kesisim alani (px^2)."""
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _texts_of(fig, renderer) -> list[tuple[str, object]]:
    """Sekildeki gorunur butun yazi dikdortgenleri (etiketler dahil)."""
    out: list[tuple[str, object]] = []
    for ax in fig.axes:
        items = [t for t in ax.texts] + list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        items += [ax.xaxis.label, ax.yaxis.label, ax.title]
        for t in items:
            if t.get_text().strip() and t.get_visible():
                out.append((" ".join(t.get_text().split()), t.get_window_extent(renderer)))
    for t in fig.texts:
        if t.get_text().strip() and t.get_visible():
            out.append((" ".join(t.get_text().split()), t.get_window_extent(renderer)))
    return out


def _inked_bodies(fig, renderer) -> list[tuple[str, object]]:
    """Ici baskili alanlar: taramali (hatch) veya gri dolu kutu/cubuklar."""
    out: list[tuple[str, object]] = []
    for ax in fig.axes:
        for p in ax.patches:
            if not isinstance(p, (Rectangle, FancyBboxPatch)):
                continue      # oklar/cizgiler cokgen kutusuyle yanlis alarm uretir
            try:
                ext = p.get_window_extent(renderer=renderer)
            except Exception:
                continue
            if ext.width <= 3 or ext.height <= 3:
                continue
            fc = p.get_facecolor()
            filled = fc is not None and len(fc) == 4 and fc[3] > 0 and fc[:3] != (1.0, 1.0, 1.0)
            if not (p.get_hatch() or filled):
                continue
            out.append((str(p.get_hatch() or fc[:3]), ext))
    return out


def _check_collisions(fig) -> None:
    """Yazi-yazi ve yazi-ici-baskili-alan carismasini piksel olcuyle denetler."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for t in ax.texts:
            if hasattr(t, "_pct_box"):
                continue
            t_ext = t.get_window_extent(renderer=renderer)
            for name, ext in _inked_bodies(fig, renderer):
                if _overlap(t_ext, ext) > COLLIDE_AREA:
                    _FITS.append(f"{_CUR['sheet']}: yazi tarali/gri alana giriyor "
                                 f"'{' '.join(t.get_text().split())[:34]}'")
    boxes = _texts_of(fig, renderer)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlap(boxes[i][1], boxes[j][1]) > COLLIDE_AREA:
                _FITS.append(f"{_CUR['sheet']}: yazi carismasi '{boxes[i][0][:26]}' x "
                             f"'{boxes[j][0][:26]}'")


def _svg_geom(path: Path) -> tuple[float, float, float]:
    """SVG'nin kendi px-boyutundan basili A4 genisligi ve 1 puanin cm karsiligi."""
    head = path.read_text(encoding="utf-8")[:600]
    w = float(re.search(r'width="([\d.]+)pt"', head).group(1))
    h = float(re.search(r'height="([\d.]+)pt"', head).group(1))
    w_cm = min(SAFE_W_CM, SAFE_H_CM * w / h)
    return w_cm, w, h


def _finish(fig) -> None:
    _check_fits(fig)
    _check_collisions(fig)
    for ax in fig.axes:
        for t in ax.texts:
            _TEXTS.append(_entry(t))
        # axis("off") sayfalarda tikler cizilmez: manifeste sadece cizilenler girer
        if ax.axison:
            for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                if lbl.get_text().strip() and lbl.get_visible():
                    _TEXTS.append({"text": lbl.get_text().strip(), "kind": "tick",
                                   "pt": round(float(lbl.get_fontsize()), 2)})
        for lbl in (ax.xaxis.label, ax.yaxis.label):
            if lbl.get_text().strip() and lbl.get_visible():
                _TEXTS.append({"text": " ".join(lbl.get_text().split()),
                               "kind": "axis-label",
                               "pt": round(float(lbl.get_fontsize()), 2)})
        if ax.get_title().strip():
            _TEXTS.append({"text": ax.get_title().strip(), "kind": "panel-title",
                           "pt": round(float(ax.title.get_fontsize()), 2)})
    for t in fig.texts:
        _TEXTS.append(_entry(t))
    fig.savefig(FIG / _CUR["svg"], bbox_inches="tight")
    plt.close(fig)
    w_cm, w_pt, h_pt = _svg_geom(FIG / _CUR["svg"])
    pt_cm = w_cm / w_pt
    smallest = min(_TEXTS, key=lambda e: e["pt"])
    for e in _TEXTS:
        e["cm"] = round(e["pt"] * pt_cm, 4)
        if e["cm"] < MIN_TEXT_CM - 1e-6:
            _FITS.append(f"{_CUR['sheet']}: yazı {e['pt']}pt = {e['cm']}cm < "
                         f"{MIN_TEXT_CM}cm (PCT 11.13) — '{e['text'][:36]}'")
    SHEETS.append({"sheet": _CUR["sheet"], "svg": _CUR["svg"], "measured": _CUR["measured"],
                   "sources": list(_CUR["sources"]),
                   "boxes": sum(1 for e in _TEXTS if e["kind"] == "box"),
                   "svg_pt": [w_pt, h_pt], "placed_w_cm": round(w_cm, 3),
                   "min_text_cm": round(smallest["pt"] * pt_cm, 4),
                   "min_text": smallest["text"][:40], "texts": list(_TEXTS)})


def _schematic_canvas(ax, fig_label, ymax=6.0):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    ax.text(0.5, 0.985, fig_label, transform=ax.transAxes, ha="center",
            va="top", fontsize=13, fontweight="bold")
    ax.text(0.99, 0.015, "Schematic; no measurement.", transform=ax.transAxes, ha="right",
            fontsize=8, style="italic")


def fig1_system_architecture() -> None:
    """Zincir DESCRIPTION_EN.md [0029] sirasiyla: task input -> ... -> output interface."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _start("FIG. 1", "fig1_system_architecture.svg")
    _schematic_canvas(ax, "FIG. 1")
    _box(ax, 0.15, 4.45, 1.6, 1.0, "task input\n(token sequence /\ntabular record)", 8)
    _box(ax, 2.15, 4.55, 1.7, 0.8, "context\nencoder", 10)
    _box(ax, 4.25, 4.55, 1.9, 0.8, "point projection\n(D-dimensional)", 9)
    _box(ax, 6.55, 4.55, 1.9, 0.8, "point cloud P\n(N points)", 10, fc="0.92")
    _arrow(ax, 1.75, 4.95, 2.15, 4.95)
    _arrow(ax, 3.85, 4.95, 4.25, 4.95)
    _arrow(ax, 6.15, 4.95, 6.55, 4.95)

    _box(ax, 6.8, 3.05, 1.9, 0.8, "pair-geometry encoder\n(points and pairs)", 8)
    _box(ax, 3.9, 2.95, 2.1, 1.0, "aggregation module\n(one set representation)", 7.5)
    _box(ax, 1.95, 3.05, 1.6, 0.8, "reader", 10)
    _arrow(ax, 7.75, 4.55, 7.75, 3.85)
    _arrow(ax, 6.8, 3.45, 6.0, 3.45)
    _arrow(ax, 3.9, 3.45, 3.55, 3.45)

    _box(ax, 1.95, 1.75, 1.6, 0.7, "task output", 10)
    _box(ax, 4.05, 1.6, 2.75, 1.0,
         "interpretation engine\n(point-removal effect ·\nregion naming · region shift)", 8)
    _box(ax, 7.45, 1.6, 2.4, 1.0, "output interface\n(task output and explanation\ndata presented jointly)", 8)
    _arrow(ax, 2.75, 3.05, 2.75, 2.45)
    _route(ax, [(6.6, 4.55), (6.6, 2.62)], dashed=True)
    _arrow(ax, 6.8, 2.1, 7.45, 2.1)
    _route(ax, [(2.75, 1.75), (2.75, 1.1), (8.65, 1.1), (8.65, 1.6)])
    ax.text(0.15, 0.9, "the interpretation engine is not a separate model; it re-runs the "
            "forward pass of the reader\nthat produces the task output, with the points of "
            "that same cloud modified (dashed path)",
            fontsize=7.5, style="italic", ha="left", va="top", linespacing=1.4)
    _finish(fig)


def fig2_pair_geometry() -> None:
    """Bes bilesen ve 3D + 2 genislik: DESCRIPTION_EN.md [0021], [0033]."""
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    _start("FIG. 2", "fig2_pair_geometry.svg")
    _schematic_canvas(ax, "FIG. 2", ymax=6.6)
    ax.plot(1.2, 5.4, "o", ms=8, color=INK)
    ax.plot(3.6, 5.85, "o", ms=8, color=INK)
    for x, y in [(4.4, 5.2), (5.2, 5.6), (6.1, 5.15), (6.9, 5.7)]:
        ax.plot(x, y, "o", ms=6, color=INK, alpha=0.5)
    ax.plot([1.2, 3.6], [5.4, 5.85], ":", color=INK, lw=1)
    _arrow(ax, 1.35, 5.43, 3.45, 5.82)
    ax.text(0.25, 5.45, "first point", fontsize=8, ha="left", va="center")
    ax.text(3.78, 5.9, "second point", fontsize=8, ha="left", va="center")
    ax.text(1.75, 5.78, "difference vector", fontsize=8, ha="left", va="bottom")
    ax.text(1.75, 5.22, "pair distance", fontsize=8, ha="left", va="top")

    comp = ["difference vector", "pair distance\n(Euclidean norm)", "cosine similarity",
            "norm vector of\na first point", "norm vector of\na second point"]
    for i, name in enumerate(comp):
        _box(ax, 0.25 + i * 1.96, 3.55, 1.76, 1.1, name, 8)
    _arrow(ax, 5.13, 3.55, 5.13, 3.15)

    _box(ax, 0.25, 2.3, 9.72, 0.85,
         "concatenation of five components\n"
         "difference vector · pair distance · cosine similarity · "
         "norm vector of a first point · norm vector of a second point", 8)
    _arrow(ax, 3.0, 2.3, 3.0, 1.95)
    _arrow(ax, 5.75, 1.65, 7.85, 1.65)
    _box(ax, 0.25, 1.35, 5.5, 0.6, "pair-feature width  3D + 2", 9)
    _box(ax, 7.85, 1.2, 2.12, 1.1, "average-pooling\nover the point pairs", 8)
    ax.text(0.25, 0.95, "3D + 2 is the pair input width defined in the implementation.",
            fontsize=8, style="italic", ha="left", va="top")
    _finish(fig)


def fig3_symmetric_aggregation() -> None:
    """Iki yol + concatenation layer ve kesikli varyant: DESCRIPTION_EN.md [0034], [0038]."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _start("FIG. 3", "fig3_symmetric_aggregation.svg")
    _schematic_canvas(ax, "FIG. 3")
    _box(ax, 0.3, 4.2, 1.85, 0.8, "point features", 9)
    _box(ax, 0.3, 2.75, 1.85, 0.8, "pair-geometry\nfeatures", 9)
    _box(ax, 2.65, 4.2, 1.75, 0.8, "max-pooling\nover the points", 9)
    _box(ax, 2.65, 2.75, 1.75, 0.8, "average-pooling\nover the point pairs", 9)
    _box(ax, 4.95, 3.5, 1.6, 0.8, "concatenation\nlayer", 9)
    _box(ax, 7.15, 3.5, 1.85, 0.8, "one set\nrepresentation", 9, fc="0.92")
    _arrow(ax, 2.15, 4.6, 2.65, 4.6)
    _arrow(ax, 2.15, 3.15, 2.65, 3.15)
    _arrow(ax, 4.4, 4.55, 4.95, 4.0)
    _arrow(ax, 4.4, 3.15, 4.95, 3.7)
    _arrow(ax, 6.55, 3.9, 7.15, 3.9)
    _box(ax, 0.3, 0.75, 6.1, 1.15,
         "variant replacing the pair computation:\nglobal set statistics — maximum · minimum · mean ·\nstandard deviation pooling, cloud centre, centre deviation", 8)
    _box(ax, 6.7, 0.95, 2.3, 0.75, "cost proportional to\nthe number of points", 8)
    _arrow(ax, 2.15, 2.75, 2.6, 1.9, dashed=True)
    _arrow(ax, 6.4, 1.33, 6.7, 1.33, dashed=True)
    _arrow(ax, 7.85, 1.7, 8.05, 3.5, dashed=True)
    ax.text(0.3, 0.6, "both poolings are operations over a set: the set representation is "
            "independent of the numbering order of the points", fontsize=7.5,
            style="italic", ha="left", va="top")
    _finish(fig)


def fig4_point_removal() -> None:
    """Maskalama ve karisim tanimi: DESCRIPTION_EN.md [0023], [0040], [0041] (N nokta, N+1 gecis)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    _start("FIG. 4", "fig4_point_removal.svg")
    _schematic_canvas(ax, "FIG. 4")
    _box(ax, 0.3, 3.6, 2.1, 0.9, "point cloud P\n(N points)", 9, fc="0.92")
    _box(ax, 3.05, 3.7, 1.85, 0.7, "reader, unmasked", 9)
    _box(ax, 5.55, 3.7, 1.75, 0.7, "base output", 9)
    _box(ax, 0.3, 1.85, 2.1, 1.1, "copy with point k set to\nzero value, mask closed", 8)
    _box(ax, 3.05, 2.05, 1.85, 0.7, "reader, masked", 9)
    _box(ax, 5.55, 2.05, 1.75, 0.7, "masked output", 9)
    _box(ax, 7.75, 1.85, 2.05, 1.1, "contribution\n(base output minus\nmasked output)", 8,
         fc="0.92")
    _arrow(ax, 2.4, 4.05, 3.05, 4.05)
    _arrow(ax, 4.9, 4.05, 5.55, 4.05)
    _arrow(ax, 2.4, 2.4, 3.05, 2.4)
    _arrow(ax, 4.9, 2.4, 5.55, 2.4)
    _arrow(ax, 6.43, 3.7, 8.3, 2.85)
    _arrow(ax, 7.3, 2.4, 7.75, 2.4)
    ax.text(0.3, 1.15, "N+1 forward passes for N points; the contribution is the base logit "
            "minus the logit with the point concerned masked", fontsize=8,
            style="italic", ha="left", va="top")
    ax.text(0.3, 0.72, "masked cloud variants are supplied jointly as one batch input",
            fontsize=8, style="italic", ha="left", va="top")
    _finish(fig)


def fig5_region_naming() -> None:
    """Merkez-nokta tabanili kumeleme -> insan-okunur etiket: DESCRIPTION_EN.md [0024], [0045]."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _start("FIG. 5", "fig5_region_naming.svg")
    _schematic_canvas(ax, "FIG. 5")
    _box(ax, 0.3, 2.7, 1.95, 0.9, "point cloud states\n(training window)", 8)
    _box(ax, 2.75, 2.7, 1.9, 0.9, "centroid-based\nclustering", 9)
    _box(ax, 5.15, 2.7, 2.0, 0.9, "region statistics\n(centre · point share)", 8)
    _box(ax, 7.65, 2.7, 2.15, 0.9, "human-readable\nregion label", 8.5, fc="0.92")
    _arrow(ax, 2.25, 3.15, 2.75, 3.15)
    _arrow(ax, 4.65, 3.15, 5.15, 3.15)
    _arrow(ax, 7.15, 3.15, 7.65, 3.15)

    random.seed(7)
    for gx, gy in [(1.5, 1.6), (4.3, 1.35), (7.1, 1.65)]:
        xs = [gx + random.uniform(-0.8, 0.8) for _ in range(18)]
        ys = [gy + random.uniform(-0.45, 0.45) for _ in range(18)]
        ax.plot(xs, ys, "o", ms=3, color=INK, alpha=0.65)
        ax.plot(gx, gy, "x", ms=9, mew=1.6, color=INK)
        ax.text(gx, gy - 0.85, "centre", fontsize=7.5, ha="center")
    ax.text(9.05, 1.5, "geometric regions\nof the point cloud", fontsize=7.5, ha="center",
            style="italic")
    ax.text(0.3, 0.42, "two geometrically distinct regions may\ncarry the same "
            "human-readable label", fontsize=7.5, style="italic", ha="left", va="top",
            linespacing=1.4)
    _finish(fig)


def fig6_region_shift() -> None:
    """Olcumlu sekil (DESCRIPTION_EN.md [0025]): pay kaymasi + ayar kaymasi -> alarm artisi."""
    src = REPO / "complaints/results/complaints_scale_stability.json"
    src_b = REPO / "maintenance/results/e1_cmapss.json"
    deltas = json.loads(src.read_text())["share_delta_drift_minus_train"]
    _start("FIG. 6", "fig6_region_shift.svg", measured=True)
    _CUR["sources"] = [str(src.relative_to(REPO)), str(src_b.relative_to(REPO))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    regions = sorted(int(k) for k in deltas)
    vals = [deltas[str(r)] for r in regions]
    ax1.bar([str(r) for r in regions], vals, color="white", edgecolor=INK, hatch="//")
    ax1.axhline(0, color=INK, lw=0.8)
    ax1.set_ylim(min(min(vals) - 0.06, -0.14), max(vals) + 0.05)
    ax1.set_xlabel("region")
    ax1.set_ylabel("region point-share difference\n(drifted window − training window)", fontsize=9)
    ax1.set_title("(a) Region point-share shift", fontsize=10)
    _CUR["source"] = str(src.relative_to(REPO))
    for r, v in zip(regions, vals):
        label = f"{v:+.3f}".replace("-", "\u2212")
        t = ax1.text(str(r), v + (0.012 if v >= 0 else -0.03), label, ha="center", fontsize=7)
        t._pct_value = v
        t._pct_src = _CUR["source"]

    demo = json.loads(src_b.read_text())["drift_demo"]
    rates = [demo["unchanged_alarm_rate"], demo["shifted_alarm_rate"]]
    ax2.bar(["before setting\nshift", "after setting\nshift"], rates, color="white",
            edgecolor=INK, hatch="\\")
    ax2.set_title("(b) Setting shift → alarm rate", fontsize=10)
    ax2.set_ylabel("alarm rate", fontsize=9)
    ax2.yaxis.set_label_position("right")
    ax2.set_ylim(0, max(rates) * 1.28)
    _CUR["source"] = str(src_b.relative_to(REPO))
    for i, v in enumerate(rates):
        t = ax2.text(i, v + max(rates) * 0.03, f"{v:.4f}", ha="center", fontsize=8)
        t._pct_value = v
        t._pct_src = _CUR["source"]
    t = ax2.text(0.03, 0.95, f"ratio {demo['shifted_to_unchanged_alarm_rate_ratio']}×",
                 transform=ax2.transAxes, ha="left", va="top", fontsize=8.5, style="italic")
    t._pct_value = demo["shifted_to_unchanged_alarm_rate_ratio"]
    t._pct_src = _CUR["source"]
    t._pct_formula = "shifted_alarm_rate / unchanged_alarm_rate"
    ax1.text(0.5, -0.42, "FIG. 6", transform=ax1.transAxes, ha="center", fontsize=12,
             fontweight="bold")
    fig.text(0.01, 0.015, "measured data", fontsize=7.5, style="italic")
    fig.subplots_adjust(bottom=0.3, wspace=0.5)
    _finish(fig)


def fig7_int8_package() -> None:
    """Olcumlu sekil (DESCRIPTION_EN.md [0026]): kantizasyon -> AUC ve paketlenmis ayak izi."""
    data = json.loads((REPO / "fraud/results/edge_eval_full_scale.json").read_text())
    auc, fp = data["int8_auc"], data["footprint"]
    _start("FIG. 7", "fig7_int8_package.svg", measured=True)
    _CUR["sources"] = ["fraud/results/edge_eval_full_scale.json"]
    _CUR["source"] = "fraud/results/edge_eval_full_scale.json"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    names = ["32-bit\nfloating point", "INT8\nper-tensor", "INT8\nper-channel"]
    vals = [auc["auc_fp32"], auc["auc_int8_tensor"], auc["auc_int8_channel"]]
    bars = ax1.bar(names, vals, color="white", edgecolor=INK, hatch=["", "//", "\\\\"])
    ax1.set_ylim(0.90, 0.973)
    ax1.set_ylabel("area under the receiver\noperating characteristic curve", fontsize=8)
    ax1.set_title("(a) Quantization effect", fontsize=10)
    for b, v in zip(bars, vals):
        t = ax1.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.4f}", ha="center",
                     fontsize=8.5)
        t._pct_value = v
        t._pct_src = _CUR["source"]
    ax1.tick_params(axis="x", labelsize=8.5)
    t = ax1.text(0.03, 0.95, f"{auc['n_test'] / 1_000_000:.1f} million test rows",
                 transform=ax1.transAxes, ha="left", va="top", fontsize=8, style="italic")
    t._pct_value = auc["n_test"]
    t._pct_src = _CUR["source"]
    t._pct_formula = "n_test / 1000000, one decimal"

    kb = [fp["deploy_fp32_kb"], fp["deploy_int8_kb"]]
    bars2 = ax2.bar(["32-bit floating point\npackage", "INT8 package"], kb, color="white",
                    edgecolor=INK, hatch=["", "//"])
    ax2.set_ylim(0, 620)
    ax2.set_ylabel("delivered package footprint (KB)", fontsize=9)
    ax2.yaxis.set_label_position("right")
    ax2.set_title("(b) Delivered package footprint", fontsize=10)
    for b, v in zip(bars2, kb):
        t = ax2.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.1f} KB", ha="center",
                     fontsize=8.5)
        t._pct_value = v
        t._pct_src = _CUR["source"]
    ax2.tick_params(axis="x", labelsize=8.5)
    ax2.text(0.5, -0.42, "FIG. 7", transform=ax2.transAxes, ha="center", fontsize=12,
             fontweight="bold")
    fig.text(0.01, 0.015, "measured data", fontsize=7.5, style="italic")
    fig.subplots_adjust(bottom=0.3, wspace=0.5)
    _finish(fig)


def fig8_window_cloud() -> None:
    """Uc katli zincir: DESCRIPTION_EN.md [0037] — sentence cloud state / window / window-geometry encoder."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _start("FIG. 8", "fig8_window_cloud.svg")
    _schematic_canvas(ax, "FIG. 8")
    ax.text(0.4, 5.3, "the task input is a text of several sentences", fontsize=8,
            style="italic", ha="left", va="bottom")
    for i, x in enumerate([0.4, 2.9, 5.4]):
        _box(ax, x, 4.55, 1.9, 0.65, f"sentence {i + 1}", 9)
        _arrow(ax, x + 0.95, 4.55, x + 0.95, 4.0)
    _box(ax, 0.4, 3.35, 6.9, 0.65, "sentence cloud states", 9, fc="0.92")
    ax.text(7.5, 3.68, "first tier", fontsize=8, style="italic", ha="left", va="center")
    _arrow(ax, 3.85, 3.35, 3.85, 2.85)
    _box(ax, 0.4, 2.2, 6.9, 0.65, "neighbouring window cloud states\n(slid by a fixed window size and stride)", 8.5)
    ax.text(7.5, 2.52, "second tier", fontsize=8, style="italic", ha="left", va="center")
    _arrow(ax, 3.85, 2.2, 3.85, 1.7)
    _box(ax, 0.4, 1.05, 3.3, 0.65, "window-geometry encoder", 9)
    _arrow(ax, 3.7, 1.38, 4.1, 1.38)
    _box(ax, 4.1, 1.05, 3.2, 0.65, "multi-sentence set representation", 8)
    _arrow(ax, 7.3, 1.38, 7.7, 1.38)
    _box(ax, 7.7, 1.05, 1.9, 0.65, "reader →\ntask output", 9, fc="0.92")
    ax.text(0.4, 0.75, "third tier; the window size and the stride are defined in the "
            "implementation", fontsize=7.5, style="italic", ha="left", va="top")
    _finish(fig)


def main() -> None:
    fig1_system_architecture()
    fig2_pair_geometry()
    fig3_symmetric_aggregation()
    fig4_point_removal()
    fig5_region_naming()
    fig6_region_shift()
    fig7_int8_package()
    fig8_window_cloud()
    manifest = {
        "uretic": "docs/patent/pct_v2/generate_figures_pct.py",
        "sayi": len(SHEETS),
        "bicimler": SHEETS,
    }
    (FIG / "fig_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in sorted(FIG.glob("fig*.svg")):
        print(p.relative_to(REPO), p.stat().st_size, "bytes")
    print("manifest:", (FIG / "fig_manifest.json").relative_to(REPO),
          len(SHEETS), "sekil",
          sum(1 for s in SHEETS if s["measured"]), "olcumlu")
    if _TIGHT:
        _TIGHT.sort(reverse=True)
        print("KUTU DOLULUK (yazi/kutu genisligi) en dolu 5:")
        for w, h, sheet, txt in _TIGHT[:5]:
            print(f"   {sheet}  {w:.0%} genislik  {h:.0%} yukseklik  '{txt[:38]}'")
    if _FITS:
        print("KIRMIZI — kutuya sigmayan yazi:")
        for f in _FITS:
            print("  ", f)
        raise SystemExit(1)
    print("KUTU-SIGMA: 0 tasma")


if __name__ == "__main__":
    main()
