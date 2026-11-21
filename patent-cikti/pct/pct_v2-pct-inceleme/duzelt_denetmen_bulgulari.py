r"""Denetmenin ALTIL bulgusu için hedefli düzeltme (2026-11-19).

Her düzeltme bir iddiaya dayanır ve dogrulanarak uygulanir: bulgu ham metinde
gorunmuyorsa betik HICBIR SEY degistirmeden hata verir.

  F1 EN istem 1 (iii)  : "reference data set/comparison data set" -> "group"      (istem 8 + [0013]/[0046] "group" diyor)
  F2 TR bagimli istem  : bagimli istemlerde iki-parcali kalik yok -> "tanımlıdır."  (EN karsiligi "wherein")
  F3 TR istem 12       : "daraltılarak" kirpisi EN istem 12'de yok -> TR'den cikar
  F4 TR istem 13       : karakterize listesinde ikinci "bir okuyucu" -> "söz konusu okuyucu"
  F5 [0057] 4096       : ham kaynak "(proxy)" niteleyicisini söylüyor -> EN+TR'ye ekle
  F6 TR [0010]         : EN'deki "reader input" baglantisinin TR aynasi

Kosum: /Users/serkankose/patent-creator-venv/bin/python duzelt_denetmen_bulgulari.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PKG = Path("/Users/serkankose/Documents/quann-3d/docs/patent/pct_v2")
BAGIMLI = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14}   # EN'de "wherein" ile baglananlar


def patch(path: Path, edits: list[tuple[str, str, int]], tag: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    log: list[str] = []
    for old, new, expect in edits:
        n = text.count(old)
        if n != expect:
            raise SystemExit(f"{tag}: {path.name} — beklenen {expect} eslesme, bulunan {n}: {old[:70]!r}")
        text = text.replace(old, new)
        log.append(f"{tag} {path.name}: {old[:60]!r} -> {new[:60]!r} ({n} yer)")
    path.write_text(text, encoding="utf-8")
    return log


def dependent_tails(path: Path) -> list[str]:
    """Bagimli istem satirlarinin sondan 'karakterize edilir.' kaligini 'tanımlıdır.' yapar."""
    lines = path.read_text(encoding="utf-8").splitlines()
    log: list[str] = []
    for i, ln in enumerate(lines):
        m = re.match(r"^(\d+)\.\s", ln)
        if not m:
            continue
        no = int(m.group(1))
        if no not in BAGIMLI:
            continue
        if not ln.rstrip().endswith("karakterize edilir."):
            raise SystemExit(f"F2: istem {no} bagimli oldugu halde iki-parcali kuyrukla bitmiyor")
        if "İstem" not in ln:
            raise SystemExit(f"F2: istem {no} bagimlilik atfi tasimiyor")
        new = ln.rstrip()[: -len("karakterize edilir.")] + "tanımlıdır."
        lines[i] = new
        log.append(f"F2 istem {no:2}: ...{ln.rstrip()[-46:]}  ->  ...{new[-38:]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log


def main() -> int:
    out: list[str] = []

    # --- F1: EN istem 1 (iii) terminoloji kaymasi --------------------------------
    out += patch(
        PKG / "CLAIMS_EN.md",
        [("for a reference data set and a comparison data set",
          "for a reference data group and a comparison data group", 1)], "F1")

    # --- F2: TR bagimli istemlerde iki-parcali kalip -----------------------------
    out += dependent_tails(PKG / "ISTEMLER_TR.md")

    # --- F3: TR istem 12'deki EN'de olmayan kirp ---------------------------------
    out += patch(
        PKG / "ISTEMLER_TR.md",
        [("geri alma işlemleriyle daraltılarak güncellenmesini",
          "geri alma işlemleriyle güncellenmesini", 1)], "F3")

    # --- F4: TR istem 13 ikinci "bir okuyucu" ------------------------------------
    out += patch(
        PKG / "ISTEMLER_TR.md",
        [("küme gösteriminden bir görev çıktısı üreten bir okuyucu;",
          "küme gösteriminden bir görev çıktısı üreten söz konusu okuyucu;", 1)], "F4")

    # --- F5: [0057] proxy niteleyicisi (EN + TR) ---------------------------------
    out += patch(
        PKG / "DESCRIPTION_EN.md",
        [("was also measured over 4096 inputs:",
          "was also measured over 4096 randomly selected QUANN-point proxy inputs:", 1)], "F5")
    out += patch(
        PKG / "DESCRIPTION_TR.md",
        [("4096 girdi üzerinde en büyük olasılık farkı",
          "4096 adet rasgele seçilmiş QUANN-noktası vekil (proxy) girdi üzerinde en büyük "
          "olasılık farkı", 1)], "F5")

    # --- F6: TR [0010] "reader input" aynasi -------------------------------------
    out += patch(
        PKG / "DESCRIPTION_TR.md",
        [("ve okuyucudan küme gösteriminden görev çıktısını alır.",
          "ve okuyucudan, bağlamı kodlanmış gösterimlerden türetilen küme gösteriminin "
          "oluşturduğu bir okuyucu girdisi üzerinden görev çıktısını alır.", 1)], "F6")

    print("\n".join(out))

    # --- saglama ------------------------------------------------------------------
    tr = (PKG / "ISTEMLER_TR.md").read_text(encoding="utf-8")
    en = (PKG / "CLAIMS_EN.md").read_text(encoding="utf-8")
    bagimli_kalan = [int(m.group(1)) for m in re.finditer(r"(?m)^(\d+)\..*karakterize edilir\.\s*$", tr)
                     if int(m.group(1)) in BAGIMLI]
    assert not bagimli_kalan, f"bagimli istemlerde hala iki-parcali kalip: {bagimli_kalan}"
    assert tr.count("karakterize edilir") == 3, "bagimsiz 1/13/15 disinda karakterize edildi"
    assert "reference data set" not in en, "EN'de henuz 'reference data set' var"
    dtr = (PKG / "DESCRIPTION_TR.md").read_text(encoding="utf-8")
    assert "proxy inputs" in (PKG / "DESCRIPTION_EN.md").read_text(encoding="utf-8")
    assert "vekil (proxy) girdi üzerinde" in dtr, "TR [0057] proxy niteleyicisi yok"
    assert "bir okuyucu girdisi üzerinden" in dtr, "TR [0010] okuyucu girdisi aynasi yok"
    print(f"\nSAGLAMA OK: TR 'karakterize edilir' = {tr.count('karakterize edilir')} "
          f"(istem 1/13/15), bagimli iki-parcali kuyruk = 0, EN 'reference data group' = "
          f"{en.count('reference data group')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
