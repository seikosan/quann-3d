#!/usr/bin/env python3
"""Nihai teslim dosyalari: kapak + birlesik PDF'ler (basvuru ve inceleme).

Girdi: build_v3.py'nin urettigi bes PDF (ozet/tarifname/istemler/teknik_etki/
sekiller). Cikti:
  - basvuru_v3.pdf  : kapaksiz, TURKPATENT sirasi (OZET, TARIFNAME, ISTEMLER, SEKILLER)
  - inceleme_v3.pdf : kapak + yukaridaki siraya ek TEKNIK_ETKI (inceleme/vakil paketi)
Sayfa sayilari pdfinfo'dan okunur; kagida elle sayi yazilmaz (BLUS_NOTU §4.2).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v3 import FIGURE_CAPTIONS, configure_document  # noqa: E402

PACKAGE = Path(__file__).resolve().parent
FONT = "Times New Roman"
# TÜRKPATENT sirasi: ozet, tarifname, istemler, sekiller (+ incelemede teknik etki)
FILING_ORDER = ("ozet_v3.pdf", "tarifname_v3.pdf", "istemler_v3.pdf", "sekiller_v3.pdf")
REVIEW_EXTRA = ("teknik_etki_v3.pdf",)


def page_count(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True,
                         text=True).stdout
    match = re.search(r"^Pages:\s+(\d+)", out, re.MULTILINE)
    if match is None:
        raise AssertionError(f"pdfinfo Pages satiri bulunamadi: {pdf.name}")
    return int(match.group(1))


def build_cover(sources: list[tuple[str, int]]) -> Path:
    doc = configure_document("Quann-3D patent v3 — teslim kapsami")
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("NOKTA-BULUTU GEOMETRI CEKIRDEGI VE GOMULU ACIKLANABILIRLIK\n"
                        "TÜRKPATENT başvuru paketi — nihai v3 teslimi")
    run.bold = True
    run.font.name = FONT
    run.font.size = Pt(15)
    run.font.color.rgb = RGBColor(0, 0, 0)
    doc.add_paragraph()

    doc.add_heading("Paket kapsamı", level=2)
    for r in doc.paragraphs[-1].runs:
        r.font.color.rgb = RGBColor(0, 0, 0)
        r.font.name = FONT
        r.font.size = Pt(13)
        r.bold = True
        r.italic = False
    table = doc.add_table(rows=len(sources) + 1, cols=3)
    table.style = "Table Grid"
    header = ("Belge", "Tanım", "Sayfa")
    definitions = {
        "ozet_v3.pdf": "Özet (≤300 kelime + teknik etki)",
        "tarifname_v3.pdf": "Tarifname (§1–§13, ölçümlü örnekler, §12 dürüstlük kayıtları)",
        "istemler_v3.pdf": "İstemler (15 istem; bağımsız: 1 yöntem, 13 sistem, 15 CRM)",
        "sekiller_v3.pdf": "Şekiller 1–8 (Şekil 6 ve 7 ölçümlüdür; diğerleri şematiktir)",
        "teknik_etki_v3.pdf": "Teknik etki (yalnız inceleme nüshasında; etki eksenleri + sınırlar)",
    }
    for col, text in enumerate(header):
        cell = table.cell(0, col)
        cell.text = text
        for paragraph in cell.paragraphs:
            for r in paragraph.runs:
                r.bold = True
                r.font.name = FONT
                r.font.size = Pt(10)
    for row, (name, pages) in enumerate(sources, start=1):
        for col, text in enumerate((name, definitions[name], str(pages))):
            cell = table.cell(row, col)
            cell.text = text
            for paragraph in cell.paragraphs:
                for r in paragraph.runs:
                    r.font.name = FONT
                    r.font.size = Pt(10)
    total = sum(pages for _, pages in sources)
    footer_row = doc.add_paragraph(f"Toplam {total} sayfa.")
    footer_row.paragraph_format.space_before = Pt(6)

    doc.add_heading("Şekiller", level=2)
    for r in doc.paragraphs[-1].runs:
        r.font.color.rgb = RGBColor(0, 0, 0)
        r.font.name = FONT
        r.font.size = Pt(13)
        r.bold = True
        r.italic = False
    for index, caption in enumerate(FIGURE_CAPTIONS, start=1):
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.add_run(f"Şekil {index} — {caption}")
        for run in paragraph.runs:
            run.font.name = FONT
            run.font.size = Pt(10)

    note = doc.add_paragraph(
        "Ölçümlü şekiller (6 ve 7) yalnız gerçek ölçüm dosyalarından çizilir; "
        "kaynakları şekil künyelerinde ve TEKNIK_ETKI.md'de dosya:satır olarak "
        "kayıtlıdır. Metinlerde doğruluk üstünlüğü iddiası yoktur; sınırlar "
        "TARIFNAME §6, §8 ve §12'dedir.")
    note.paragraph_format.space_before = Pt(10)
    for run in note.runs:
        run.font.name = FONT
        run.font.size = Pt(10)
        run.italic = True

    output = PACKAGE / "kapak_v3.docx"
    doc.save(output)
    return output


def pdfs_of(docx_files: list[Path]) -> list[Path]:
    soffice = shutil.which("soffice")
    if soffice is None:
        raise RuntimeError("LibreOffice (soffice) gerekli")
    with tempfile.TemporaryDirectory(prefix="quann-v3-deliv-") as profile:
        subprocess.run(
            [soffice, "--headless", f"-env:UserInstallation={Path(profile).as_uri()}",
             "--convert-to", "pdf", "--outdir", str(PACKAGE),
             *[str(path) for path in docx_files]],
            check=True, capture_output=True, text=True)
    outputs = [PACKAGE / f"{path.stem}.pdf" for path in docx_files]
    for pdf in outputs:
        if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
            raise AssertionError(f"kapak PDF üretilemedi: {pdf.name}")
    return outputs


def merge(parts: list[Path], output: Path) -> int:
    output.unlink(missing_ok=True)
    subprocess.run(["pdfunite", *[str(p) for p in parts], str(output)],
                   check=True, capture_output=True, text=True)
    pages = page_count(output)
    expected = sum(page_count(p) for p in parts)
    if pages != expected:
        raise AssertionError(f"{output.name}: {pages} sayfa, beklenen {expected}")
    return pages


def main() -> None:
    filing_parts = [PACKAGE / name for name in FILING_ORDER]
    for part in filing_parts + [PACKAGE / n for n in REVIEW_EXTRA]:
        if not part.is_file():
            raise RuntimeError(f"eksik girdi: {part.name} — önce build_v3.py koşun")

    cover_docx = build_cover(
        [(name, page_count(PACKAGE / name)) for name in FILING_ORDER + REVIEW_EXTRA])
    cover_pdf = pdfs_of([cover_docx])[0]

    filing_pages = merge(filing_parts, PACKAGE / "basvuru_v3.pdf")
    review_pages = merge([cover_pdf] + filing_parts
                         + [PACKAGE / n for n in REVIEW_EXTRA],
                         PACKAGE / "inceleme_v3.pdf")
    print("teslim dosyalari hazir:")
    print(f"- basvuru_v3.pdf: {filing_pages} sayfa "
          f"({' + '.join(FILING_ORDER)})")
    print(f"- inceleme_v3.pdf: {review_pages} sayfa (kapak + yukariler + "
          f"{' + '.join(REVIEW_EXTRA)})")


if __name__ == "__main__":
    main()
