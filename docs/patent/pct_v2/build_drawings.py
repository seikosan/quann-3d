#!/usr/bin/env python3
"""pct_v2 SVG cizimlerini PCT fiziksel kopyasina derler: SEKILLER_FILING-COPY.pdf.

Neden betik: cizimler A4 kagit, kenar bosluklari icinde ve siyah-beyaz olmali
(UYGUNLUK_NOTU.md satir 35: fiziksel bicim motorun kapsami disinda, "physical
requirements cannot be verified from text"; satir 106: cizim dosyasi denetimi
yapilmadi). Bu betik o fiziksel kopyayi uretir VE ucurumunde kendisi dogrular.

Hat: SVG --rsvg-convert--> 300 dpi gri PNG --> python-docx A4 sayfa (2,5 cm bosluk)
    --> soffice --headless --> PDF. (Vektor <image> saralayici denendi: librsvg harici
    SVG referansini cizmedi, PDF bos sayfaustu; kullanimdisi.)

Kullanim:  python3 docs/patent/pct_v2/build_drawings.py
Cikti:     docs/patent/pct_v2/filing/SEKILLER_FILING-COPY.pdf  + stdout dogrulama

Kabul olcutu (betik kendisi kontrol eder, aksi halde cikis kodu 1):
  1) pdfinfo Pages == 8
  2) pdfinfo Page size == A4 (595 x 842 pt)
  3) HER sayfada murekkep var (koyu piksel orani >= 0,05%) — bos sayfa kabul degil
  4) metin katmani yok (cizimler yazili metin tasimaz, Kural 11.13)
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE / "fig"
DST = HERE / "filing"

A4_W_CM, A4_H_CM = 21.0, 29.7
MARGIN_CM = 2.5
SAFE_W_CM = A4_W_CM - 2 * MARGIN_CM   # 16,0 cm
SAFE_H_CM = A4_H_CM - 2 * MARGIN_CM   # 24,7 cm
RENDER_PX_W = 2400                    # ~300 dpi basili genislik
INK_MIN = 0.0005                      # 0,05% koyu piksel esigi


def render_gray(svg: Path, out_png: Path, renderer: str) -> Image.Image:
    subprocess.run([renderer, "-w", str(RENDER_PX_W), "-o", str(out_png), str(svg)],
                   check=True)
    img = Image.open(out_png).convert("RGBA")
    page = Image.new("RGBA", img.size, (255, 255, 255, 255))
    page.alpha_composite(img)
    return page.convert("L")


def fit_cm(img: Image.Image) -> tuple[float, float]:
    w, h = img.size
    scale = min(SAFE_W_CM / w, SAFE_H_CM / h)
    return w * scale, h * scale


def build_docx(images: list[tuple[str, Image.Image]], tmp: Path) -> Path:
    doc = Document()
    for section in doc.sections:
        section.page_width = Cm(A4_W_CM)
        section.page_height = Cm(A4_H_CM)
        section.top_margin = section.bottom_margin = Cm(MARGIN_CM)
        section.left_margin = section.right_margin = Cm(MARGIN_CM)

    for i, (name, img) in enumerate(images):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        path = tmp / f"{name}.png"
        img.save(path, optimize=True)
        w_cm, h_cm = fit_cm(img)
        run = para.add_run()
        run.add_picture(str(path), width=Cm(round(w_cm, 3)), height=Cm(round(h_cm, 3)))
        if i < len(images) - 1:
            doc.add_page_break()
    out = tmp / "SEKILLER_FILING-COPY.docx"
    doc.save(out)
    return out


def to_pdf(docx_path: Path, tmp: Path) -> Path:
    soffice = shutil.which("soffice")
    if soffice is None:
        raise SystemExit("soffice yok: brew install --cask libreoffice")
    profile = (tmp / "lo-profile").as_uri()
    subprocess.run([soffice, "--headless", f"-env:UserInstallation={profile}",
                    "--convert-to", "pdf", "--outdir", str(tmp), str(docx_path)],
                   check=True, capture_output=True, text=True, timeout=300)
    pdf = tmp / (docx_path.stem + ".pdf")
    if not pdf.is_file():
        raise SystemExit(f"PDF olusmadi: {pdf}")
    return pdf


def pdf_info(pdf: Path) -> dict[str, str]:
    info = shutil.which("pdfinfo")
    if info is None:
        raise SystemExit("pdfinfo yok: brew install poppler")
    out = subprocess.run([info, str(pdf)], capture_output=True, text=True).stdout
    found: dict[str, str] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        if key in ("Pages", "Page size", "Page # size"):
            found[key] = val.strip()
    return found


def ink_ratios(pdf: Path, tmp: Path) -> list[float]:
    """Her sayfanin koyu piksel oranini olcer — bos sayfa bu kapidan gecmez."""
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise SystemExit("pdftoppm yok: brew install poppler")
    stem = tmp / "ink"
    subprocess.run([pdftoppm, "-r", "80", "-gray", str(pdf), str(stem)], check=True)
    ratios: list[float] = []
    for page_png in sorted(stem.parent.glob(stem.name + "-*.pgm")):
        img = Image.open(page_png).convert("L")
        data = img.getdata()
        dark = sum(1 for v in data if v < 200)
        ratios.append(dark / len(data))
    return ratios


def main() -> int:
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        raise SystemExit("rsvg-convert yok: brew install librsvg")

    svg_files = sorted(SRC.glob("fig[1-8]_*.svg"),
                       key=lambda p: int(re.search(r"fig(\d+)_", p.name).group(1)))
    nums = [int(re.search(r"fig(\d+)_", p.name).group(1)) for p in svg_files]
    if nums != list(range(1, 9)):
        raise SystemExit(f"beklenen fig1..fig8, bulunan: {nums}")

    DST.mkdir(exist_ok=True)
    out = DST / "SEKILLER_FILING-COPY.pdf"
    with tempfile.TemporaryDirectory(prefix="quann-pct-draw-") as tmpname:
        tmp = Path(tmpname)
        images: list[tuple[str, Image.Image]] = []
        for svg in svg_files:
            png = tmp / f"{svg.stem}.png"
            img = render_gray(svg, png, renderer)
            images.append((svg.stem, img))
            w_cm, h_cm = fit_cm(img)
            print(f"  {svg.name}: {img.size[0]}x{img.size[1]} px -> {w_cm:.1f} x {h_cm:.1f} cm")

        docx_path = build_docx(images, tmp)
        produced = to_pdf(docx_path, tmp)
        out.write_bytes(produced.read_bytes())
        print(f"yazildi: {out.relative_to(HERE.parent.parent)}  ({out.stat().st_size} B)")

    problems: list[str] = []
    info = pdf_info(out)
    print("pdfinfo:", info)
    pages = int(info.get("Pages", "0"))
    if pages != 8:
        problems.append(f"sayfa sayisi {pages} != 8")
    size = info.get("Page size", "")
    if "A4" not in size and not re.match(r"59[45]\.[0-9]+ x 84[12]\.[0-9]+ pts", size):
        problems.append(f"sayfa boyutu A4 degil: {size}")

    with tempfile.TemporaryDirectory(prefix="quann-pct-ink-") as ink_dir:
        ratios = ink_ratios(out, Path(ink_dir))
    for i, r in enumerate(ratios, 1):
        print(f"  sayfa {i} murekkep: {100 * r:.2f}%")
        if r < INK_MIN:
            problems.append(f"sayfa {i} bos (murekkep {100 * r:.3f}%)")

    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        text = subprocess.run([pdftotext, str(out), "-"], capture_output=True,
                              text=True).stdout.strip()
        print(f"  metin katmani: {len(text)} karakter{'  -> BEKLENEN 0' if len(text) else ''}")
        if text:
            problems.append(f"cizim PDF'inde metin katmani var: {text[:60]!r}")

    if problems:
        print("KABUL EDILMEDI:", "; ".join(problems))
        return 1
    print("KABUL: 8 sayfa A4, her sayfada murekkep, metin katmani yok.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
