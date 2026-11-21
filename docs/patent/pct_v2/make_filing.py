#!/usr/bin/env python3
"""pct_v2 çalışma nüshalarından basılacak PCT metinlerini üretir.

Neden betik, neden elle kopyalama yok: Kural 11 ile biçimlendirilmiş metin,
çalışma nüshasından FARKLI olmak zorundadır — çalışma nüshası vekil için ölçüm
künyesi (HTML yorumu) ve Kural notları taşır, başvuru metni taşımaz. Elle
kopyalama rakam kaydırır; bu betik yalnız yorum/not atar, tek bir rakama
dokunmaz.

Kullanım:  python3 docs/patent/pct_v2/make_filing.py
Çıktı:     docs/patent/pct_v2/filing/*.txt  + stdout'ta üretim raporu (JSON)

Atma kuralları (yalnızca yorum/not, içerik değil):
  1. `<!-- ... -->` HTML yorumları (iç ölçüm künyesi)
  2. italik giriş notu bloğu (`*...*`)
  3. markdown başlık işareti `#` ve kalın `**`
  4. başlık satırındaki Kural parantezi -> "Title of the invention:" / "Buluşun başlığı:"
  5. başlıklardaki Kural/Rule parantezi ve "1." bölüm numaraları
Doğrulama: üretilen her metinde yorum/kalıntı taraması, istem sayımı, paragraf
numarası sürekliliği, özet kelime sayısı. Hata varsa çıkış kodu 1.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DST = HERE / "filing"

COMMENT_RE = re.compile(r"<!--.*?-->\s*", re.DOTALL)
HEAD_RE = re.compile(r"^#{1,6}\s+(.*)$")
RULE_PAREN_RE = re.compile(r"\s*\((?:PCT\s+)?(?:Kural|Rule)[^)]*\)?\s*", re.IGNORECASE)
SEC_NUM_RE = re.compile(r"^\s*\d+\.\s+")
TITLE_RE = re.compile(
    r"^\*\*(Title of the invention|Buluşun başlığı)[^:]*:\*\*\s*(.+?)\s*$", re.IGNORECASE
)
CLAIM_LINE_RE = re.compile(r"^\s{0,3}(\d{1,3})\.\s+")
PARA_RE = re.compile(r"^\[(\d{4})\]")


def strip_notes(text: str) -> str:
    """Yorum ve not katmanını atar; rakamlara ve kelime gövdelerine dokunmaz."""
    out = COMMENT_RE.sub("", text)

    lines = out.split("\n")

    # italik giriş notu bloğu: "*..." ile başlar, "*" ile biter
    kept: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("*") and not ln.startswith("**"):
            j = i
            while j < len(lines) and not lines[j].rstrip().endswith("*"):
                j += 1
            i = j + 1  # bloğu atla
            continue
        kept.append(ln)
        i += 1

    clean: list[str] = []
    for ln in kept:
        m = TITLE_RE.match(ln)
        if m:
            label = "Title of the invention" if m.group(1).lower().startswith("title") else "Buluşun başlığı"
            clean.append(f"{label}: {m.group(2)}")
            continue
        h = HEAD_RE.match(ln)
        if h:
            body = RULE_PAREN_RE.sub(" ", h.group(1)).strip()
            body = SEC_NUM_RE.sub("", body).strip()
            clean.append(body)
            continue
        clean.append(ln.replace("**", ""))

    text = "\n".join(clean)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def abstract_part(text: str, start_marker: str, end_marker: str) -> str:
    i = text.index(start_marker) + len(start_marker)
    j = text.index(end_marker, i)
    return " ".join(text[i:j].split()) + "\n"


def audit(name: str, body: str) -> dict:
    """Üretilen metnin saflığını ve sayısını ölçer; bulgu varsa AssertionError."""
    problems: list[str] = []
    if "<!--" in body:
        problems.append("HTML yorumu kalmış")
    for ln in body.split("\n"):
        if ln.startswith("#") or ln.startswith("**") or ln.startswith("*"):
            problems.append(f"markdown kalıntısı: {ln[:40]!r}")
        if re.search(r"\b(?:Kural|Rule)\s+\d", ln):
            problems.append(f"Kural notu kalmış: {ln[:60]!r}")

    claim_nums = sorted({int(m.group(1)) for ln in body.split("\n") if (m := CLAIM_LINE_RE.match(ln))})
    paras = [int(m.group(1)) for ln in body.split("\n") if (m := PARA_RE.match(ln))]
    return {
        "file": name,
        "bytes": len(body.encode()),
        "lines": len(body.strip().split("\n")),
        "words": len(body.split()),
        "claim_numbers": claim_nums,
        "claim_count": len(claim_nums),
        "paragraph_first": paras[0] if paras else None,
        "paragraph_last": paras[-1] if paras else None,
        "paragraph_count": len(paras),
        "paragraph_gapless": paras == list(range(1, len(paras) + 1)),
        "problems": problems,
    }


def main() -> int:
    DST.mkdir(exist_ok=True)
    jobs: list[tuple[str, str, str]] = []

    def emit(fname: str, body: str) -> None:
        (DST / fname).write_text(body, encoding="utf-8")
        jobs.append((fname, body, ""))

    emit("DESCRIPTION_FILING-COPY_EN.txt", strip_notes((HERE / "DESCRIPTION_EN.md").read_text(encoding="utf-8")))
    emit("CLAIMS_FILING-COPY_EN.txt", strip_notes((HERE / "CLAIMS_EN.md").read_text(encoding="utf-8")))
    emit("TARIFNAME_FILING-COPY_TR.txt", strip_notes((HERE / "DESCRIPTION_TR.md").read_text(encoding="utf-8")))
    emit("ISTEMLER_FILING-COPY_TR.txt", strip_notes((HERE / "ISTEMLER_TR.md").read_text(encoding="utf-8")))

    ozet = (HERE / "OZET_PCT.md").read_text(encoding="utf-8")
    emit("ABSTRACT_FILING-COPY_EN.txt", abstract_part(ozet, "## ÖZET (EN) — başvuru nüshası", "## ÖZET (TR)"))
    emit("OZET_FILING-COPY_TR.txt", abstract_part(ozet, "## ÖZET (TR) — başvuru nüshası", "\n---\n\n## İç künyeler"))

    reports = [audit(name, body) for name, body, _ in jobs]
    for r in reports:
        if r["file"].startswith(("CLAIMS", "ISTEMLER")) and r["claim_numbers"] != list(range(1, 16)):
            r["problems"].append(f"istem numaraları 1..15 değil: {r['claim_numbers']}")
        if r["file"].startswith(("DESCRIPTION", "TARIFNAME")) and not r["paragraph_gapless"]:
            r["problems"].append("paragraf numaraları 1..N kesintisiz değil")

    ok = not any(r["problems"] for r in reports)
    print(json.dumps({"ok": ok, "out": str(DST), "files": reports}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
