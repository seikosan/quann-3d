#!/usr/bin/env python3
"""Build the TÜRKPATENT filing packet as editable DOCX and print-ready PDF."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image, ImageOps


REPO = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
FIGURE_DIR = REPO / "docs/patent/dil/fig"
FONT = "Times New Roman"
FIGURE_CAPTIONS = (
    "Sistem mimarisi ve modül akışı",
    "Girdi metninin nokta bulutu olarak temsili",
    "Token işleme sırasında nokta bulutunun oluşumu",
    "Farklı boyutlardaki bölge gösterimleri",
    "Kelime gösterimlerinin bölge dağılımı",
    "Çift-geometri özellik vektörünün bileşenleri",
    "Eğitim sırasında izlenen örnek eğri",
    "Morfolojik grupların geometrik dağılımı",
)


def strip_markdown(text: str) -> str:
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*|__(.*?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"(?<!\*)\*(.*?)\*(?!\*)|(?<!_)_(.*?)_(?!_)",
                  lambda m: m.group(1) or m.group(2), text)
    return text.replace("`", "").strip()


def configure_document(title: str) -> Document:
    doc = Document()
    doc.core_properties.title = title
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer.add_run("Sayfa ")
        run.font.name = FONT
        run.font.size = Pt(9)
        field_run = footer.add_run()
        field_begin = OxmlElement("w:fldChar")
        field_begin.set(qn("w:fldCharType"), "begin")
        instruction = OxmlElement("w:instrText")
        instruction.set(qn("xml:space"), "preserve")
        instruction.text = " PAGE "
        field_end = OxmlElement("w:fldChar")
        field_end.set(qn("w:fldCharType"), "end")
        field_run._r.extend((field_begin, instruction, field_end))
    return doc


def render_markdown(doc: Document, source: Path) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line == "---" or line.startswith(">"):
            if line.startswith(">"):
                paragraph = doc.add_paragraph(strip_markdown(line.lstrip("> ")))
                paragraph.paragraph_format.left_indent = Cm(0.5)
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            level = min(len(heading.group(1)), 3)
            paragraph = doc.add_heading(strip_markdown(heading.group(2)), level=level)
            paragraph.paragraph_format.keep_with_next = True
            for run in paragraph.runs:
                run.font.name = FONT
                run.font.color.rgb = RGBColor(0, 0, 0)
            continue
        if line.startswith("|"):
            rows: list[list[str]] = []
            while line.startswith("|"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
                    rows.append(cells)
                if index >= len(lines):
                    break
                line = lines[index].strip()
                index += 1
            if rows:
                columns = max(len(row) for row in rows)
                table = doc.add_table(rows=len(rows), cols=columns)
                table.style = "Table Grid"
                for row_index, row in enumerate(rows):
                    for column_index in range(columns):
                        cell = table.cell(row_index, column_index)
                        cell.text = strip_markdown(row[column_index]) if column_index < len(row) else ""
                        for paragraph in cell.paragraphs:
                            paragraph.paragraph_format.space_after = Pt(2)
                            for run in paragraph.runs:
                                run.font.name = FONT
                                run.font.size = Pt(10)
                                run.bold = row_index == 0
                doc.add_paragraph()
            continue
        bullet = re.match(r"^[-*]\s+(.*)$", line)
        if bullet:
            paragraph = doc.add_paragraph(style="List Bullet")
            paragraph.add_run(strip_markdown(bullet.group(1)))
        else:
            paragraph = doc.add_paragraph(strip_markdown(line))
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            paragraph.paragraph_format.space_after = Pt(6)
            paragraph.paragraph_format.line_spacing = 1.15
        for run in paragraph.runs:
            run.font.name = FONT


def build_text_docx(source_name: str, output_name: str, title: str) -> Path:
    output = PACKAGE / output_name
    doc = configure_document(title)
    render_markdown(doc, PACKAGE / source_name)
    doc.save(output)
    return output


def render_figure(source: Path, output: Path) -> None:
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        raise RuntimeError("rsvg-convert is required to render patent SVG drawings")
    subprocess.run([renderer, "-o", str(output), str(source)], check=True,
                   capture_output=True, text=True)
    with Image.open(output) as image:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.alpha_composite(rgba)
        grayscale = ImageOps.grayscale(white.convert("RGB"))
        monochrome = grayscale.point(lambda value: 255 if value >= 210 else 0, mode="1")
        monochrome.save(output, format="PNG")
        colors = monochrome.convert("L").getcolors(maxcolors=256)
        if colors is None or {value for _, value in colors} - {0, 255}:
            raise AssertionError(f"Drawing is not strictly black and white: {source.name}")


def build_figures_docx(temp_dir: Path) -> Path:
    doc = configure_document("Patent şekilleri")
    sources = sorted(FIGURE_DIR.glob("fig[1-8]_*.svg"),
                     key=lambda path: int(re.search(r"fig(\d+)_", path.name).group(1)))
    if len(sources) != len(FIGURE_CAPTIONS):
        raise RuntimeError(f"Expected {len(FIGURE_CAPTIONS)} drawings, found {len(sources)}")
    for index, (source, caption) in enumerate(zip(sources, FIGURE_CAPTIONS), start=1):
        image_path = temp_dir / f"figure_{index}.png"
        render_figure(source, image_path)
        if index > 1:
            doc.add_page_break()
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(8)
        run = paragraph.add_run(f"ŞEKİL {index}")
        run.bold = True
        run.font.name = FONT
        run.font.size = Pt(12)
        picture = doc.add_paragraph()
        picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
        picture.add_run().add_picture(str(image_path), width=Cm(15.5), height=Cm(19.4))
        label = doc.add_paragraph(f"Şekil {index} — {caption}")
        label.alignment = WD_ALIGN_PARAGRAPH.CENTER
        label.paragraph_format.space_before = Pt(8)
    output = PACKAGE / "sekiller.docx"
    doc.save(output)
    return output


def convert_pdfs(docx_files: list[Path]) -> None:
    soffice = shutil.which("soffice")
    if soffice is None:
        raise RuntimeError("LibreOffice (soffice) is required to create PDFs")
    for docx_path in docx_files:
        (PACKAGE / f"{docx_path.stem}.pdf").unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="quann-patent-lo-") as profile:
        profile_uri = Path(profile).as_uri()
        subprocess.run(
            [soffice, "--headless", f"-env:UserInstallation={profile_uri}",
             "--convert-to", "pdf", "--outdir", str(PACKAGE),
             *[str(path) for path in docx_files]],
            check=True, capture_output=True, text=True,
        )
    for docx_path in docx_files:
        pdf_path = PACKAGE / f"{docx_path.stem}.pdf"
        if not pdf_path.is_file() or pdf_path.stat().st_size < 1000:
            raise AssertionError(f"PDF export missing or unexpectedly small: {pdf_path}")
        if not pdf_path.read_bytes().startswith(b"%PDF-"):
            raise AssertionError(f"Invalid PDF signature: {pdf_path}")


def main() -> None:
    claims = (PACKAGE / "istemler_final.md").read_text(encoding="utf-8")
    if re.search(r"\(\s*\d+\s*\)", claims):
        raise AssertionError("Claims contain a figure reference numeral")
    docx_files = [
        build_text_docx("tarifname_final.md", "tarifname_final.docx", "Tarifname"),
        build_text_docx("istemler_final.md", "istemler_final.docx", "İstemler"),
        build_text_docx("basvuru_formu.md", "basvuru_formu.docx", "Başvuru formu"),
        build_text_docx("ozet.md", "ozet.docx", "Özet"),
        build_text_docx("teknik_etki.md", "teknik_etki.docx", "Teknik etki"),
    ]
    with tempfile.TemporaryDirectory(prefix="quann-patent-figures-") as temp:
        docx_files.append(build_figures_docx(Path(temp)))
    convert_pdfs(docx_files)
    print("Paket hazır:")
    for path in sorted(PACKAGE.glob("*.pdf")) + sorted(PACKAGE.glob("*.docx")):
        print(f"- {path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
