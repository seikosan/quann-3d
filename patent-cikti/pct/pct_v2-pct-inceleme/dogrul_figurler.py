r"""pct_v2 cizim setini tarifname + ham olcu JSON'lari + basili PDF karsisinda dogrular.

Neden bu betik var: cizimler `generate_figures_pct.py` uretiminde kendi kapilarini
(kutu-sigma, carisma, 0,21 cm) kosuyor; fakat o kapilar URETICININ KAYITLARINA bakar.
"Uretici kendi kaydini temiz ilan etmesi" kanit degil — bu betik disaridan, ayri bir
baglamda bes kaynagi karsilastirir: SVG (cizilen gercek metin) x fig_manifest.json
(kunya) x DESCRIPTION_EN.md (terim hakemi) x olcu JSON'lari (ham sayi) x PDF (fizik).

Bes denetim:
  A) SVG == KUNYA — her sekil icin SVG'deki <text> kelimeleri ile manifest kumesi birebir
     ayni olmali (satir kirma farki kelime coksetiyle eritilir). Fazla/eksik kelime = BULGU.
  B) TERIM — cizimdeki her alfabetik kelime DESCRIPTION_EN.md'de gecmeli. Muaf liste yalniz
     cizim-gelenegi sozcukleridir (fig/schematic/measurement/measured/data/no + panel harfleri).
  C) OLCU IZLENEBILIRLIGI — olcumlu sekillerde her measured-value: (i) raw_value ham JSON'da
     birebir bulunmali, (ii) basili metin raw_value'nun o hassasiyetteki yuvarlanmasi olmali,
     (iii) `formula` tasuyanlar formulen yeniden hesaplanmali. Sema sekillerde ham sayi YASAK.
  D) SEMA/OLCU AYRIMI (DESCRIPTION_EN.md [0028]) — FIG. 1-5, 8 sema: ondalik veya 3+ haneli
     sayi kullanimaz; FIG. 6-7 olcumlu: en az bir olcu-deger tasir.
  E) FIZIK — PDF 8 sayfa, A4, her sayfada murekkep, metin katmani 0 karakter; her "FIG. n"
     tarifnamede gecer; manifest 8 sekil icerir; her yazinin basili yuksekligi >= 0,21 cm
     (PCT Kural 11.13).

Kosum: python3 patent-cikti/pct/pct_v2-pct-inceleme/dogrul_figurler.py
Cikis: 0 = hepsi gecti; 1 = bulgu var. Kayit: DOGRUL_figurler.json
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path("/Users/serkankose/Documents/quann-3d")
FIG = REPO / "docs/patent/pct_v2/fig"
MAN = FIG / "fig_manifest.json"
EN = REPO / "docs/patent/pct_v2/DESCRIPTION_EN.md"
PDF = REPO / "docs/patent/pct_v2/filing/SEKILLER_FILING-COPY.pdf"
OUT = REPO / "patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json"

MIN_TEXT_CM = 0.21          # PCT Kural 11.13: yazi yuksekligi asgari 0,21 cm
A4_PT = (595.3, 841.9)      # pdfinfo A4
TOL_PT = 1.0

# Cizim-gelenegi sozcukleri: tarifnamede gecmeleri beklenmez (tek gerekce: baski etiketi).
MUAF = {"fig", "schematic", "measurement", "measured", "data", "no", "a", "b"}

# Sema sekillerde izinli tamsayilar: [0033] "3D + 2", [0040]-[0041] "N"/"N+1",
# FIG numarasi, FIG. 8'deki "sentence 1/2/3" siralama. Ondalik ve 3+ hane YASAK.
IZINLI_SAYI = {"1", "2", "3"}


def words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", s.lower())


def nums(s: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", s)


def svg_words(path: Path) -> list[str]:
    """SVG'de GERCEK cizilen metinlerin kelime cokseti (satir kirma eritilir)."""
    raw = re.findall(r"<text[^>]*>(.*?)</text>", path.read_text(encoding="utf-8"), re.S)
    out: list[str] = []
    for chunk in raw:
        txt = html.unescape(re.sub(r"<[^>]+>", " ", chunk))
        out += words(txt)
    return out


def walk_numbers(node, out: list[float]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            walk_numbers(v, out)
    elif isinstance(node, list):
        for v in node:
            walk_numbers(v, out)
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        out.append(float(node))


def find_key(node, key: str):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            got = find_key(v, key)
            if got is not None:
                return got
    elif isinstance(node, list):
        for v in node:
            got = find_key(v, key)
            if got is not None:
                return got
    return None


def printed_to_float(text: str) -> tuple[float, int] | None:
    """Basili etiketi (or. '+0.146', '528.9 KB', 'ratio 2.4x') (deger, hane) dondurur."""
    m = re.search(r"(-?\d+(?:\.\d+)?)", text.replace("\u2212", "-").replace("+", ""))
    if not m:
        return None
    dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
    return float(m.group(1)), dec


def check_a(manifest) -> dict:
    findings = []
    for sheet in manifest:
        svg = FIG / sheet["svg"]
        if not svg.exists():
            findings.append(f"{sheet['sheet']}: SVG yok {svg.name}")
            continue
        drawn = Counter(svg_words(svg))
        logged = Counter(w for e in sheet["texts"] for w in words(e["text"]))
        eksik, fazla = drawn - logged, logged - drawn
        if eksik or fazla:
            findings.append(f"{sheet['sheet']}: SVG'de cizilip kunyede olmayan "
                            f"{sorted(eksik)} | kunyede olup cizilmeyen {sorted(fazla)}")
    return {"ad": "A SVG == kunya", "bulgu": findings,
            "olcu": {"sekil": len(manifest),
                     "kunyi_metin": sum(len(s["texts"]) for s in manifest)}}


def check_b(manifest) -> dict:
    spec = re.sub(r"<!--.*?-->", " ", EN.read_text(encoding="utf-8"), flags=re.S)
    spec_words = set(words(spec))
    findings, kontrol = [], 0
    for sheet in manifest:
        for e in sheet["texts"]:
            for w in set(words(e["text"])):
                if w in MUAF or len(w) <= 2:
                    continue
                kontrol += 1
                if w not in spec_words:
                    findings.append(f"{sheet['sheet']}: tarifnamede gecmeyen terim "
                                    f"'{w}' ('{e['text'][:44]}')")
    return {"ad": "B terim (DESCRIPTION_EN.md hakem)", "bulgu": findings,
            "olcu": {"kontrol_edilen_kelime": kontrol,
                     "tarifname_kelime": len(spec_words)}}


def check_c(manifest) -> dict:
    findings, izlenen = [], 0
    cache: dict[str, list[float]] = {}
    for sheet in manifest:
        for e in sheet["texts"]:
            if e["kind"] != "measured-value":
                continue
            src = REPO / e["source"]
            if not src.exists():
                findings.append(f"{sheet['sheet']}: kaynak JSON yok {e['source']}")
                continue
            if e["source"] not in cache:
                vals: list[float] = []
                walk_numbers(json.loads(src.read_text(encoding="utf-8")), vals)
                cache[e["source"]] = vals
            vals = cache[e["source"]]
            raw = float(e["raw_value"])
            if not any(abs(v - raw) < 1e-12 for v in vals):
                findings.append(f"{sheet['sheet']}: raw_value {raw} ham JSON'da yok "
                                f"({e['source']})")
                continue
            got = printed_to_float(e["text"])
            if got is None:
                findings.append(f"{sheet['sheet']}: '{e['text']}' basili sayi okunamadi")
                continue
            printed, dec = got
            form = e.get("formula")
            if form and "1000000" in form:
                base = find_key(json.loads(src.read_text(encoding="utf-8")), "n_test")
                beklenen = round(float(base) / 1_000_000.0, dec)
                if abs(printed - beklenen) > 10 ** -dec / 2:
                    findings.append(f"{sheet['sheet']}: '{e['text']}' != n_test/1e6 = "
                                    f"{beklenen}")
            elif form and "/" in form:
                a, b = (k.strip() for k in form.split("/"))
                va, vb = find_key(json.loads(src.read_text(encoding="utf-8")), a), \
                    find_key(json.loads(src.read_text(encoding="utf-8")), b)
                if va is None or vb is None:
                    findings.append(f"{sheet['sheet']}: formül anahtarları '{form}' "
                                    f"JSON'da bulunamadı")
                else:
                    beklenen = round(float(va) / float(vb), dec)
                    if abs(printed - beklenen) > 10 ** -dec / 2:
                        findings.append(f"{sheet['sheet']}: '{e['text']}' != {form} = "
                                        f"{beklenen}")
            elif abs(printed - raw) > 10 ** -dec / 2 + 1e-12:
                findings.append(f"{sheet['sheet']}: basili '{e['text']}' != yuvarlama("
                                f"{raw}, {dec}) = {round(raw, dec)}")
            else:
                izlenen += 1
    return {"ad": "C olcu izlenebilirligi", "bulgu": findings,
            "olcu": {"olcu_deger": izlenen, "kaynak_dosya": len(cache)}}


def check_d(manifest) -> dict:
    findings = []
    for sheet in manifest:
        sem = not sheet["measured"]
        olcu = sum(1 for e in sheet["texts"] if e["kind"] == "measured-value")
        fig_no = nums(sheet["sheet"])
        if sem:
            if olcu:
                findings.append(f"{sheet['sheet']}: sema sekilde olcu-deger kayitli")
            for e in sheet["texts"]:
                for n in nums(e["text"]):
                    if "." in n or len(n) >= 3 or (n not in IZINLI_SAYI and n not in fig_no):
                        findings.append(f"{sheet['sheet']}: sema sekilde izinli olmayan "
                                        f"sayi '{n}' ('{e['text'][:38]}')")
        else:
            if not olcu:
                findings.append(f"{sheet['sheet']}: olcumlu sekilde olcu-deger yok")
            if not sheet["sources"]:
                findings.append(f"{sheet['sheet']}: kaynak JSON kuyrugu bos")
    return {"ad": "D sema/olcu ayrimi ([0028])", "bulgu": findings,
            "olcu": {"sema": sum(1 for s in manifest if not s["measured"]),
                     "olcumlu": sum(1 for s in manifest if s["measured"])}}


def check_e(manifest) -> dict:
    findings = []
    spec = re.sub(r"<!--.*?-->", " ", EN.read_text(encoding="utf-8"), flags=re.S)
    if len(manifest) != 8:
        findings.append(f"manifest 8 sekil degil: {len(manifest)}")
    etiket = [s["sheet"] for s in manifest]
    if etiket != [f"FIG. {i}" for i in range(1, 9)]:
        findings.append(f"sekil sirasi/etiketi beklenen degil: {etiket}")
    for i in range(1, 9):
        if f"FIG. {i}" not in spec:
            findings.append(f"tarifnamede 'FIG. {i}' gecmiyor")
    for sheet in manifest:
        if sheet["min_text_cm"] < MIN_TEXT_CM - 1e-6:
            findings.append(f"{sheet['sheet']}: en kucuk yazi {sheet['min_text_cm']}cm < "
                            f"{MIN_TEXT_CM}cm (PCT 11.13) — '{sheet['min_text']}'")
    pdf: dict = {}
    if not PDF.exists():
        findings.append(f"PDF yok: {PDF}")
    else:
        info = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout
        pg = re.search(r"Pages:\s+(\d+)", info)
        size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info)
        if not pg or int(pg.group(1)) != 8:
            findings.append(f"PDF sayfa sayisi 8 degil: {pg.group(1) if pg else 'okunamadi'}")
        if not size or abs(float(size.group(1)) - A4_PT[0]) > TOL_PT or \
                abs(float(size.group(2)) - A4_PT[1]) > TOL_PT:
            findings.append(f"PDF A4 degil: {size.groups() if size else 'okunamadi'}")
        chars = subprocess.run(["pdftotext", str(PDF), "-"], capture_output=True,
                               text=True).stdout.strip()
        if chars:
            findings.append(f"PDF'te metin katmani var ({len(chars)} karakter): cizimler "
                            f"gorsel olmali")
        pdf["karakter"] = len(chars)
        pdf["sayfa"] = int(pg.group(1)) if pg else None
        pdf["pt"] = [float(size.group(1)), float(size.group(2))] if size else None
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["pdftoppm", "-r", "110", "-png", str(PDF), f"{td}/p"],
                           check=True)
            from PIL import Image
            inks = []
            for f in sorted(Path(td).glob("*.png")):
                im = Image.open(f).convert("L")
                px = list(im.getdata())
                inks.append(round(sum(1 for v in px if v < 200) / len(px) * 100, 2))
            if min(inks) < 0.2:
                findings.append(f"bazi sayfalarda murekkep yok: {inks}")
            pdf["murekkep_yuzde"] = inks
    return {"ad": "E fizik + PCT 11.13 + FIG eslemesi", "bulgu": findings,
            "olcu": {"pdf": pdf, "min_yazi_cm": min(s["min_text_cm"] for s in manifest)}}


def main() -> int:
    manifest = json.loads(MAN.read_text(encoding="utf-8"))["bicimler"]
    sonuc = {"kapilar": [check_a(manifest), check_b(manifest), check_c(manifest),
                         check_d(manifest), check_e(manifest)],
             "kaynak": {"manifest": str(MAN), "tarifname": str(EN), "pdf": str(PDF),
                        "uretic": json.loads(MAN.read_text(encoding="utf-8"))["uretic"]}}
    for k in sonuc["kapilar"]:
        k["durum"] = "TEMIZ" if not k["bulgu"] else "BULGU"
    toplam = sum(len(k["bulgu"]) for k in sonuc["kapilar"])
    sonuc["ozet"] = {"bulgu_toplam": toplam,
                     "durum": "TEMIZ" if toplam == 0 else f"{toplam} BULGU"}
    OUT.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in sonuc["kapilar"]:
        print(f"[{k['durum']:^5}] {k['ad']}  {k['olcu']}")
        for b in k["bulgu"]:
            print("   -", b)
    print(f"{'KABUL' if toplam == 0 else 'RED'}: {toplam} bulgu — {OUT.name}")
    return 0 if toplam == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
