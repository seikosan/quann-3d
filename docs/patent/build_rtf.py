#!/usr/bin/env python3
"""Render a Markdown document to RTF (rich text) for attorney delivery.

Block parsing lives here (it keeps hard-wrapped list continuations); inline
markup reuses ``build_docx.INLINE`` so the two builders share one grammar.
The HTML intermediate is written to the ignored ``docs/patent/_render/``
folder and converted with macOS ``textutil``.

Usage:  python3 docs/patent/build_rtf.py [dosya.md ...]
        (default: docs/patent/VEKIL_SORU_CEVAP.md)
"""
import html
import os
import re
import subprocess
import sys
from urllib.parse import quote

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RENDER = os.path.join(REPO, "docs/patent", "_render")
DEFAULT_SOURCE = os.path.join(REPO, "docs/patent", "VEKIL_SORU_CEVAP.md")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_docx import INLINE, table_rows  # noqa: E402  (shared inline grammar)

LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
LIST_ITEM = re.compile(r"^(-|\*|\d+\.)\s+")
BLOCK_START = re.compile(r"^(#|```|\||---\s*$|>\s?|(-|\*|\d+\.)\s)")
HEADING_PT = {1: 20, 2: 15, 3: 12.5, 4: 11.5, 5: 11, 6: 11}
BODY_PT = 11
TABLE_PT = 10
# textutil reads CSS point sizes as CSS pixels (measured: 12pt -> \fs32 = 16pt),
# so every CSS length is written as target * 0.75 to land on the target point size.
SCALE = 0.75


def css_pt(target: float) -> str:
    """Return a CSS length that textutil renders as ``target`` points."""
    return f"{target * SCALE:g}pt"


CSS = f"""body {{ font-family: 'Times New Roman', serif;
       font-size: {css_pt(BODY_PT)}; }}
p {{ text-align: justify; margin: 0 0 {css_pt(6)} 0; }}
li {{ margin: 0 0 {css_pt(3)} 0; }}
code {{ font-family: 'Courier New', monospace; }}
pre {{ font-family: 'Courier New', monospace; }}
blockquote {{ margin: 0 0 {css_pt(8)} {css_pt(18)}; font-style: italic; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 0.5pt solid #000; padding: {css_pt(2)} {css_pt(4)};
         font-size: {css_pt(TABLE_PT)}; }}
"""


def _styled_text(text: str) -> str:
    """Escape plain text and turn ``**bold**`` / ``*em*`` / ``code`` into HTML."""
    out: list[str] = []
    for token in INLINE.split(html.escape(text, quote=False)):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) > 4:
            out.append(f"<strong>{token[2:-2]}</strong>")
        elif token.startswith("`") and token.endswith("`") and len(token) > 2:
            out.append(f"<code>{token[1:-1]}</code>")
        elif token.startswith("*") and token.endswith("*") and len(token) > 2:
            out.append(f"<em>{token[1:-1]}</em>")
        else:
            out.append(token)
    return "".join(out)


def _href(url: str, base: str) -> str:
    """Resolve a relative markdown link against the source file's directory."""
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) or url.startswith("#"):
        return url
    return "file://" + quote(os.path.normpath(os.path.join(base, url)))


def inline_html(text: str, base: str = "") -> str:
    """Render inline markdown (links first, then emphasis/code) as HTML."""
    parts: list[str] = []
    cursor = 0
    for match in LINK.finditer(text):
        parts.append(_styled_text(text[cursor:match.start()]))
        href = html.escape(_href(match.group(2), base), quote=True)
        parts.append(f'<a href="{href}">{_styled_text(match.group(1))}</a>')
        cursor = match.end()
    parts.append(_styled_text(text[cursor:]))
    return "".join(parts)


def parse_blocks(md: str) -> list[tuple[str, object]]:
    """Split markdown into (kind, value) blocks, folding wrapped list items."""
    lines = md.split("\n")
    blocks: list[tuple[str, object]] = []
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            j = i + 1
            buf: list[str] = []
            while j < n and not lines[j].strip().startswith("```"):
                buf.append(lines[j])
                j += 1
            blocks.append(("code", "\n".join(buf)))
            i = j + 1
            continue
        heading = HEADING.match(stripped)
        if heading:
            blocks.append(("h", (len(heading.group(1)), heading.group(2).strip())))
            i += 1
            continue
        if re.match(r"^-{3,}$", stripped):
            blocks.append(("hr", None))
            i += 1
            continue
        if stripped.startswith("|"):
            buf = []
            while i < n and lines[i].strip().startswith("|"):
                buf.append(lines[i].strip())
                i += 1
            blocks.append(("table", buf))
            continue
        if LIST_ITEM.match(stripped):
            items: list[list[str]] = []
            while i < n:
                current = lines[i].strip()
                if LIST_ITEM.match(current):
                    marker = LIST_ITEM.match(current).group(1)
                    items.append([marker, LIST_ITEM.sub("", current, count=1)])
                    i += 1
                    continue
                wrapped = (current and items and lines[i][:1] in (" ", "\t")
                           and not BLOCK_START.match(current))
                if wrapped:
                    items[-1][1] += " " + current
                    i += 1
                    continue
                break
            blocks.append(("list", [(m, t) for m, t in items]))
            continue
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^>\s?", "", lines[i].strip()))
                i += 1
            blocks.append(("quote", " ".join(buf)))
            continue
        buf = [stripped]
        i += 1
        while i < n and lines[i].strip() and not BLOCK_START.match(lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        blocks.append(("p", " ".join(buf)))
    return blocks


def block_html(kind: str, value: object, base: str = "") -> str:
    """Render one parsed block to an HTML fragment."""
    if kind == "h":
        level, text = value  # type: ignore[misc]
        clean = re.sub(r"\*\*", "", str(text))
        size = HEADING_PT.get(level, 11)
        return (f"<h{level}><span style=\"font-size:{css_pt(size)}\">"
                f"{inline_html(clean, base)}</span></h{level}>")
    if kind == "hr":
        return "<hr>"
    if kind == "code":
        return f"<pre>{html.escape(str(value))}</pre>"
    if kind == "quote":
        return f"<blockquote>{inline_html(str(value), base)}</blockquote>"
    if kind == "table":
        rows = table_rows(value)  # type: ignore[arg-type]
        if not rows:
            return ""
        head, *body = rows
        thead = "".join(f"<th>{inline_html(re.sub(r'\\*\\*', '', cell), base)}</th>"
                        for cell in head)
        body_html = "".join(
            "<tr>" + "".join(f"<td>{inline_html(cell, base)}</td>" for cell in row)
            + "</tr>"
            for row in body
        )
        return f"<table><tr>{thead}</tr>{body_html}</table>"
    if kind == "list":
        items = value  # type: ignore[assignment]
        numbered = bool(items) and bool(re.match(r"^\d+\.$", items[0][0]))
        tag = "ol" if numbered else "ul"
        inner = "".join(f"<li>{inline_html(text, base)}</li>" for _, text in items)
        return f"<{tag}>{inner}</{tag}>"
    return f"<p>{inline_html(str(value), base)}</p>"


def build_html(source: str) -> str:
    """Convert a markdown file into a standalone HTML document string."""
    base = os.path.dirname(os.path.abspath(source))
    with open(source, encoding="utf-8") as handle:
        blocks = parse_blocks(handle.read())
    body = "\n".join(block_html(kind, value, base) for kind, value in blocks)
    title = html.escape(os.path.splitext(os.path.basename(source))[0])
    return (
        '<!DOCTYPE html><html lang="tr"><head><meta charset="utf-8">'
        f"<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>"
    )


def convert_rtf(intermediate: str, output: str) -> None:
    """Convert the HTML intermediate to RTF with macOS textutil."""
    textutil = "/usr/bin/textutil"
    if not os.path.exists(textutil):
        raise RuntimeError("textutil not found: RTF conversion needs macOS textutil")
    subprocess.run([textutil, "-convert", "rtf", "-output", output, intermediate],
                   check=True, capture_output=True, text=True)


def verify_rtf(rtf_path: str, source: str) -> str:
    """Round-trip the RTF back to plain text and check headings + evidence."""
    with open(rtf_path, "rb") as handle:
        raw = handle.read()
    if raw[:5] != b"{\\rtf":
        raise AssertionError(f"{rtf_path}: not an RTF file (starts with {raw[:5]!r})")
    plain = subprocess.run(["/usr/bin/textutil", "-convert", "txt", "-stdout", rtf_path],
                           check=True, capture_output=True, text=True).stdout
    with open(source, encoding="utf-8") as handle:
        md = handle.read()
    missing = [t.strip("# ").strip() for t in re.findall(r"^##\s+.*$", md, re.M)
               if t.strip("# ").strip() not in plain]
    if missing:
        raise AssertionError(f"{rtf_path}: headings lost in RTF: {missing}")
    md_evidence = len(re.findall(r"`[\w./-]+:\d+", md))
    rtf_evidence = len(re.findall(r"[\w./-]+:\d+", plain))
    if rtf_evidence < md_evidence:
        raise AssertionError(
            f"{rtf_path}: evidence refs {rtf_evidence} < source {md_evidence}")
    sizes = {round(int(s) / 2, 1) for s in re.findall(r"\\fs(\d+)", raw.decode("latin-1"))}
    levels = {len(m) for m in re.findall(r"^(#{1,6})\s+", md, re.M)}
    expected = {float(BODY_PT)} | {float(HEADING_PT[l]) for l in levels}
    if re.search(r"^\|", md, re.M):
        expected.add(float(TABLE_PT))
    absent = sorted(t for t in expected if not any(abs(t - s) <= 0.6 for s in sizes))
    if absent:
        raise AssertionError(f"{rtf_path}: sizes {absent}pt missing (got {sorted(sizes)})")
    headings = len(re.findall(r"^##\s+", md, re.M))
    return (f"VERIFY        {rtf_path} | headings={headings}"
            f" | evidence={rtf_evidence}/{md_evidence}"
            f" | sizes={sorted(sizes)}pt"
            f" | bytes={len(raw)}")


def build(source: str) -> str:
    """Build one RTF document next to its markdown source."""
    os.makedirs(RENDER, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source))[0]
    intermediate = os.path.join(RENDER, f"{stem}.html")
    output = os.path.splitext(source)[0] + ".rtf"
    with open(intermediate, "w", encoding="utf-8") as handle:
        handle.write(build_html(source))
    convert_rtf(intermediate, output)
    print(verify_rtf(output, source))
    print("RTF           " + output)
    return output


def main() -> None:
    sources = sys.argv[1:] or [DEFAULT_SOURCE]
    for source in sources:
        path = source if os.path.isabs(source) else os.path.join(REPO, source)
        build(path)


if __name__ == "__main__":
    main()
