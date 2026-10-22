#!/usr/bin/env python3
"""Build filing-ready .docx for the patent package.

Reads the markdown source, reorders into the standard filing sequence
(Form -> ToC -> Title -> Abstract -> Description -> Claims -> Drawings ->
internal appendix), applies TÜRKPATENT/PCT document styles (A4, Times New
Roman, 1.15 spacing), and embeds the rendered figure PNGs (each on its own
page).

Usage:  python3 docs/patent/build_docx.py
"""
import os, re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RENDER = os.path.join(REPO, "docs/patent/_render")
FONT = "Times New Roman"

# ---------------------------------------------------------------------------
# inline formatting
# ---------------------------------------------------------------------------
INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)")

def add_runs(par, text, base_bold=False, base_italic=False, size=None, color=None):
    for tok in INLINE.split(text):
        if not tok:
            continue
        bold, italic, code = base_bold, base_italic, False
        t = tok
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            bold, t = True, tok[2:-2]
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            code, t = True, tok[1:-1]
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            italic, t = True, tok[1:-1]
        r = par.add_run(t)
        r.font.name = "Courier New" if code else FONT
        r.font.size = Pt(size if size else 11)
        r.bold = bold
        r.italic = italic
        if code:
            r.font.color.rgb = RGBColor(0x10, 0x10, 0x10)
        if color:
            r.font.color.rgb = color
    return par

def style_par(par, justify=True, space_after=6, line=1.15, align=None):
    pf = par.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    pf.line_spacing = line
    if align is not None:
        pf.alignment = align
    elif justify:
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

# ---------------------------------------------------------------------------
# markdown block parsing
# ---------------------------------------------------------------------------
def parse_blocks(md):
    lines = md.split("\n")
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        # code fence
        if s.startswith("```"):
            j = i + 1
            buf = []
            while j < n and not lines[j].strip().startswith("```"):
                buf.append(lines[j]); j += 1
            blocks.append(("code", "\n".join(buf)))
            i = j + 1
            continue
        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            blocks.append(("h", (len(m.group(1)), m.group(2).strip())))
            i += 1
            continue
        # hr
        if re.match(r"^-{3,}$", s):
            blocks.append(("hr", None)); i += 1; continue
        # table
        if s.startswith("|"):
            buf = []
            while i < n and lines[i].strip().startswith("|"):
                buf.append(lines[i].strip()); i += 1
            blocks.append(("table", buf))
            continue
        # list (bullet or numbered) - gather consecutive
        if re.match(r"^(-|\*|\d+\.)\s+", s):
            buf = []
            while i < n and re.match(r"^(-|\*|\d+\.)\s+", lines[i].strip()):
                st = lines[i].strip()
                mk = re.match(r"^(-|\*|\d+\.)\s+", st).group(1)
                buf.append((mk, re.sub(r"^(-|\*|\d+\.)\s+", "", st)))
                i += 1
            blocks.append(("list", buf))
            continue
        # blockquote
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^>\s?", "", lines[i].strip())); i += 1
            blocks.append(("quote", " ".join(buf)))
            continue
        # paragraph (gather until blank / special block)
        buf = [s]
        i += 1
        while i < n and lines[i].strip() and not re.match(
                r"^(#|```|\||>|---\s*$|(-|\*|\d+\.)\s)", lines[i].strip()):
            buf.append(lines[i].strip()); i += 1
        blocks.append(("p", " ".join(buf)))
    return blocks

def table_rows(buf):
    rows = []
    for ln in buf:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if all(re.match(r"^:?-{2,}:?$", c) for c in cells if c):
            continue
        rows.append(cells)
    return rows

# ---------------------------------------------------------------------------
# docx section / style setup
# ---------------------------------------------------------------------------
def setup_doc():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(11)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    for sec in doc.sections:
        sec.page_height = Cm(29.7); sec.page_width = Cm(21.0)
        sec.top_margin = Cm(2.2); sec.bottom_margin = Cm(2.2)
        sec.left_margin = Cm(2.5); sec.right_margin = Cm(2.5)
    # page number in footer
    for sec in doc.sections:
        fp = sec.footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_runs(fp, "Sayfa ", size=9)
        _field(fp, "PAGE")
    return doc

def _field(par, instr):
    r = par.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    r._r.append(f1); r._r.append(it); r._r.append(f2)
    r.font.size = Pt(9); r.font.name = FONT

def page_break(doc):
    doc.add_page_break()

def render_block(doc, kind, val, in_appendix=False):
    if kind == "h":
        level, text = val
        text = re.sub(r"\*\*", "", text)
        doc_level = max(1, min(level - 1, 3))
        h = doc.add_heading("", level=doc_level)
        r = h.add_run(text); r.font.name = FONT
        r.font.color.rgb = RGBColor(0, 0, 0)
        r.font.size = Pt({1: 20, 2: 15, 3: 12.5, 4: 11.5}.get(level, 11))
        r.bold = True
        h.paragraph_format.space_before = Pt(14 if level >= 2 else 4)
        h.paragraph_format.space_after = Pt(8)
        return
    if kind == "hr":
        return
    if kind == "code":
        for ln in val.split("\n"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            rr = p.add_run(ln if ln.strip() else " ")
            rr.font.name = "Courier New"; rr.font.size = Pt(11)
        return
    if kind == "table":
        rows = table_rows(val)
        if not rows:
            return
        t = doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        for ri, row in enumerate(rows):
            for ci, cell in enumerate(row):
                c = t.cell(ri, ci)
                c.text = ""
                p = c.paragraphs[0]
                add_runs(p, re.sub(r"\*\*", "", cell), size=10,
                         base_bold=(ri == 0))
                p.paragraph_format.space_after = Pt(2)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return
    if kind == "list":
        for item in val:
            mk, txt = item
            numbered = bool(re.match(r"^\d+\.$", mk))
            p = doc.add_paragraph(style="List Number" if numbered else "List Bullet")
            add_runs(p, txt)
            style_par(p, justify=False, space_after=3)
        return
    if kind == "quote":
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing = 1.15
        add_runs(p, val, base_italic=True, size=10.5)
        return
    # paragraph
    p = doc.add_paragraph()
    add_runs(p, val)
    style_par(p)
    return


# ---------------------------------------------------------------------------
# section splitting (keyed by canonical name) + figure embedding
# ---------------------------------------------------------------------------
from PIL import Image
from docx.shared import Cm

def split_sections(blocks):
    """Return (cover, {canonical_key: [blocks]})."""
    cover = []
    secs = {}
    cur = None
    for kind, val in blocks:
        if kind == "h" and val[0] == 2:
            cur = _canon(val[1])
            secs[cur] = [("h", val)]  # keep the section's own header
        else:
            (secs[cur] if cur is not None else cover).append((kind, val))
    return cover, secs

def _norm(s):
    # unify Turkish letter variants (i/I, ç, ş, ğ, ö, ü) for robust matching
    t = s.upper()
    for a, b in [("\u0130", "I"), ("\u0131", "I"), ("\u00c7", "C"),
                 ("\u015e", "S"), ("\u011e", "G"), ("\u00d6", "O"),
                 ("\u00dc", "U"), ("\u0130", "I")]:
        t = t.replace(a, b)
    return t

def _canon(title):
    t = _norm(title)
    if "FORM" in t or "REQUEST" in t: return "form"
    if "ICINDEK" in t or "CONTENT" in t: return "toc"
    if t.startswith("1.") or "BASLIK" in t or "TITLE" in t: return "title"
    if "ACIKLAMA" in t or "DESCRIPTION" in t: return "description"
    if "IDDI" in t or "CLAIM" in t: return "claims"
    if "OZET" in t or "ABSTRACT" in t: return "abstract"
    if "CIZIM" in t or "DRAWING" in t or "FIGURE" in t: return "drawings"
    if "ATIF" in t or "BIBLIOGRAPH" in t: return "bib"
    if "VEKI" in t or "AGENT" in t: return "agent"
    return t[:12]

def fit_size(path, max_w_cm=15.0, max_h_cm=20.0):
    im = Image.open(path)
    w, h = im.size
    scale = min(max_w_cm / (w / 28.35), max_h_cm / (h / 28.35))  # px->inch->cm
    # 28.35 cm per inch; treat px ~ 96/inch for a target
    w_in = w / 96.0; h_in = h / 96.0
    s = min(max_w_cm / (w_in * 2.54), max_h_cm / (h_in * 2.54))
    return Cm(w_in * 2.54 * s), Cm(h_in * 2.54 * s)

def add_figure(doc, png, caption, page_first=False):
    if page_first:
        page_break(doc)
    # figure page: centered
    pc = doc.add_paragraph(); pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pc.paragraph_format.space_before = Pt(6)
    w, h = fit_size(png)
    pc.add_run().add_picture(png, width=w, height=h)
    cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(8)
    r = cap.add_run(caption); r.bold = True; r.font.size = Pt(11); r.font.name = FONT

def add_form_header(doc, kind, val):
    """Render the form's intro + table + confidentiality quote (first section)."""
    for k, v in (kind, val):
        render_block(doc, k, v)


# ---------------------------------------------------------------------------
# TR dil document
# ---------------------------------------------------------------------------
TR_FIGS = [
    ("sekil_1.png", "Şekil 1 — Sistem mimarisi (100–150)"),
    ("sekil_2.png", "Şekil 2 — “kedi bahçede uyudu” 3D nokta bulutu (200, 210)"),
    ("sekil_3.png", "Şekil 3 — Bulut evrimi 1→4 nokta (300)"),
    ("sekil_4.png", "Şekil 4 — 16D ve 32D bölge haritaları (400, 410)"),
    ("sekil_5.png", "Şekil 5 — Kelime bölge coğrafyası (500)"),
    ("sekil_6.png", "Şekil 6 — Çift geometrisi vektörü bileşenleri (600–630)"),
    ("sekil_7.png", "Şekil 7 — Eğitim eğrisi PPL (700)"),
    ("sekil_8.png", "Şekil 8 — Morfolojik kümeleşme (800, 810)"),
]

def _render_cover(doc, cover_blocks):
    for kind, val in cover_blocks:
        if kind == "h":
            h = doc.add_heading("", level=0)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = h.add_run(val[1]); r.font.name = FONT; r.font.size = Pt(20)
            r.bold = True; r.font.color.rgb = RGBColor(0, 0, 0)
            h.paragraph_format.space_before = Pt(120)
        else:
            render_block(doc, kind, val)

def build_tr():
    md = open(os.path.join(REPO, "docs/patent/TURKPATENT_basvuru_v2.3.md")).read()
    blocks = parse_blocks(md)
    cover, secs = split_sections(blocks)
    doc = setup_doc()
    _render_cover(doc, cover)
    order = ["form", "toc", "title", "abstract", "description", "claims",
             "drawings", "bib", "agent"]
    for i, key in enumerate(order):
        if key not in secs:
            continue
        if i > 0:
            page_break(doc)
        for kind, val in secs[key]:
            if key == "drawings" and kind == "table":
                # replace the source-file table with real figures
                for png, cap in TR_FIGS:
                    add_figure(doc, os.path.join(RENDER, png), cap, page_first=True)
                continue
            render_block(doc, kind, val)
    out = os.path.join(REPO, "docs/patent/TURKPATENT_basvuru_v2.3.docx")
    doc.save(out)
    print("TR dil  ->", out)
    return out


# ---------------------------------------------------------------------------
# PCT EN document
# ---------------------------------------------------------------------------
PCT_FIGS = [
    ("Figure 1 — Universal architecture and module flow (100–290)", ["core_fig1.png"]),
    ("Figure 2 — Dual-geometry and symmetric fusion", ["core_fig2.png"]),
    ("Figure 3 — Score, ablation, prototype and drift from the same cloud", ["core_fig3.png"]),
    ("Figure 4 — AML typology naming examples", ["pct_fig4a.png", "pct_fig4b.png"]),
    ("Figure 5 — Fraud identity-bucket cloud", ["pct_fig5.png"]),
    ("Figure 6 — Maintenance cloud and operating-condition drift", ["pct_fig6a.png", "pct_fig6b.png"]),
    ("Figure 7 — Self-contained deployment and parity/manifest audit", ["pct_fig7.png"]),
]

def _render_pct_figures(doc):
    for cap, pngs in PCT_FIGS:
        page_break(doc)
        for k, png in enumerate(pngs):
            pc = doc.add_paragraph(); pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pc.paragraph_format.space_before = Pt(4)
            w, h = fit_size(os.path.join(RENDER, png))
            pc.add_run().add_picture(os.path.join(RENDER, png), width=w, height=h)
        capP = doc.add_paragraph(); capP.alignment = WD_ALIGN_PARAGRAPH.CENTER
        capP.paragraph_format.space_before = Pt(8)
        r = capP.add_run(cap); r.bold = True; r.font.size = Pt(11); r.font.name = FONT

def build_pct():
    md = open(os.path.join(REPO, "docs/patent/geometry_dual_use_core_v1_EN.md")).read()
    blocks = parse_blocks(md)
    cover, secs = split_sections(blocks)
    doc = setup_doc()
    _render_cover(doc, cover)
    order = ["form", "toc", "title", "abstract", "description", "claims",
             "drawings", "bib", "agent"]
    for i, key in enumerate(order):
        if key not in secs:
            continue
        if i > 0:
            page_break(doc)
        # section header (we re-add a clean one; the source ## is inside secs? no, stripped)
        for kind, val in secs[key]:
            if key == "drawings" and kind == "table":
                _render_pct_figures(doc)
                continue
            render_block(doc, kind, val)
    out = os.path.join(REPO, "docs/patent/geometry_dual_use_core_v1_EN.docx")
    doc.save(out)
    print("PCT EN  ->", out)
    return out


def main():
    build_tr()
    build_pct()

if __name__ == "__main__":
    main()
