r"""`antecedent_basis` bulgularinin sinifi: gercek kusur mu, hat kalibi siniri mi.

Yerel hat kalibi (patentline/linter.py) bir tanitimi yalniz SU hallerde kabul ediyor:
  * artikel listesi: (a|an|at least one|one or more|a plurality of)              :202
    TR karsiligi   : (bir|bu|en az bir|cok sayida|en az iki|birden fazla|soz konusu)  :199
  * artikel ile terim arasinda EN FAZLA 2 ara kelime                             :223 `{0,2}`
  * veya `terim (sayi)` biciminde numarali tanim                                 :224

Bu iki sinir yuzunden GERCEKTE var olan tanitim gorulmez. Siniflar:

  B = tanitim var ama artikel ile ad arasinda 3+ niteleme kelimesi var
      or. "A computer-implemented machine learning method" -> "the method"
  C = belirli ifade, tanitima bir sifatin eklenmesiyle geliyor
      or. "a reader input" -> "the derived reader input"
  A = tanitim article-less cogul/kitlesel ad ile yapilmis (Ingilizce pratiode dogru)
      or. "... pair-geometry features derived from ..." -> "the pair-geometry features"
  D = hicbir yerde tanitim yok -> GERCEK oncul dayanak kusuru

Motorun `prior` hesabi (linter.py:331-332) ve tetik kalibi (:198-201) burada BIREBIR
tekrarlanir; farkli bir olcut icat edilmez.

Kosum: python3 kanit_antecedent.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HATTI = Path("/Users/serkankose/Documents/patent-hatti")
sys.path.insert(0, str(HATTI))

from patentline.linter import (EN_GENERIC_HEAD, EN_PHRASE_STOP, TR_ANTECEDENT,  # noqa: E402
                               parse_claims, tr_fold)

PKG = Path("/Users/serkankose/Documents/quann-3d/docs/patent/pct_v2")
OUT = Path(__file__).resolve().parent

ART_EN = r"(?:a|an|at least one|one or more|a plurality of)"
ART_TR = r"(?:bir|bu|en az bir|çok sayıda|en az iki|birden fazla|söz konusu)"


def trigger_for(locale: str) -> re.Pattern[str]:
    if locale == "tr":
        return re.compile(rf"(?i)\b{TR_ANTECEDENT}\s+((?:[\wÇĞİÖŞÜçğıöşü]+\s*){{1,3}})")
    return re.compile(r"(?i)\bthe\s+((?:[A-Za-z]+\s*){1,3})")


def words_of(locale: str, phrase: str) -> list[str]:
    """Motorun kelime suzmesi (linter.py:206-215) — ayni sirayla."""
    words = [w for w in tr_fold(phrase).split() if len(w) > 2][:3]
    if locale != "tr":
        while words and words[0] in {"same", "other", "such"}:
            words = words[1:]
        while words and words[-1] in EN_PHRASE_STOP:
            words = words[:-1]
    return words


def classify(hay: str, words: list[str], locale: str) -> tuple[str, str]:
    """Bir flagged ifadeyi mekanik olarak siniflandirir."""
    if not words:
        return ("bos", "")
    arts = ART_TR if locale == "tr" else ART_EN
    for n in (len(words), 2, 1):
        key = r"\s+".join(re.escape(w) for w in words[:n])
        # 1) motorun kabul esigi: 0-2 ara kelime -> bulgu yanlis olmaliydi
        near = re.search(rf"{arts}\s+(?:[\w'-]+\s+){{0,2}}{key}\b", hay, re.I)
        if near:
            return ("motor_kabul_etmeliydi", near.group(0))
        # 2) B: tanitim var ama arada 3+ niteleme kelimesi var
        far = re.search(rf"{arts}\s+(?:[\w'-]+\s+){{3,6}}{key}\b", hay, re.I)
        if far:
            return ("B_artikel-terim_arasi_3+", far.group(0))
    # 3) C: ifadenin govde kelimesi ayrica tanitilmis (ifadeye sifat eklenmis)
    stem = re.escape(words[-1])
    stem_near = re.search(rf"{arts}\s+(?:[\w'-]+\s+){{0,2}}{stem}\b", hay, re.I)
    if stem_near:
        return ("C_ifadeye_sifat_eklenmis", stem_near.group(0))
    # 4) A: article-less cogul/kitlesel tanim (baglaci edat/baglac sonrasi ilk gecis)
    for n in (len(words), 2, 1):
        key = r"\s+".join(re.escape(w) for w in words[:n])
        bare = re.search(rf"(?:[:,;(]|and|or|of|for|by|as|to|over|perform|applying|computing)\s+"
                         rf"(?:[\w'-]+\s+){{0,3}}{key}\b", hay, re.I)
        if bare:
            return ("A_articleless_cogul_tanim", re.sub(r"\s+", " ", bare.group(0))[-60:])
    return ("D_GECEK_KUSUR", "")


def run(path: Path, locale: str) -> dict:
    text = path.read_text(encoding="utf-8")
    claims = parse_claims(text)
    trig = trigger_for(locale)
    rows = []
    for i, (num, body) in enumerate(claims):
        prior = "\n".join(b for _, b in claims[:i])          # linter.py:332
        for m in trig.finditer(body):                         # linter.py:204
            words = words_of(locale, m.group(1))
            if not words:
                continue
            if locale != "tr":
                if words and words[-1] in EN_GENERIC_HEAD:
                    continue                                   # linter.py:214
            hay = prior + "\n" + body[: m.start()]            # linter.py:220
            # motorun kendi kabul testi: ayni kalip, 3->2->1 on ek (linter.py:221-227)
            motor_ok = False
            for n in (len(words), 2, 1):
                key = r"\s+".join(re.escape(w) for w in words[:n])
                if (re.search(rf"{(ART_TR if locale=='tr' else ART_EN)}\s+(?:[\w-]+\s+){{0,2}}{key}",
                              hay, re.I)
                        or re.search(rf"{key}\s*\(\s*\d+\s*\)", hay, re.I)):
                    motor_ok = True
                    break
            if motor_ok:
                continue          # motor bunu bulgu yapmadi
            cls, ev = classify(hay, words, locale)
            rows.append({"istem": num, "ifade": m.group(0).strip(), "sinif": cls,
                         "kanit_parca": re.sub(r"\s+", " ", ev)})
    # dort sinif de acikca sifirdan baslar: "D = 0" iddiasi JSON'da OKUNABILIR olmali,
    # anahtarin yoklugundan cikarilmamali (uydurma oradan sizar).
    ozet: dict[str, int] = {"A_articleless_cogul_tanim": 0, "B_artikel-terim_arasi_3+": 0,
                            "C_ifadeye_sifat_eklenmis": 0, "D_GECEK_KUSUR": 0}
    for r in rows:
        ozet[r["sinif"]] = ozet.get(r["sinif"], 0) + 1
    return {"dosya": str(path.relative_to(PKG.parent.parent)), "istem_sayisi": len(claims),
            "toplam_bulgu": len(rows), "ozet": ozet, "satirlar": rows}


def main() -> int:
    out = [run(PKG / "filing" / "CLAIMS_FILING-COPY_EN.txt", "en"),
           run(PKG / "filing" / "ISTEMLER_FILING-COPY_TR.txt", "tr")]
    (OUT / "KANIT_antecedent.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    for r in out:
        print(f"{r['dosya']:44} istem={r['istem_sayisi']} bulgu={r['toplam_bulgu']} {r['ozet']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
