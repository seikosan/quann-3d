#!/usr/bin/env python3
"""db_taramasi ham kayitlarini satir-k numarali tek kanit defterine cevirir.

Amac: ON_SANAT.md icindeki her sayi bir satira baglanabilsin (repo kurali 13/1:
kaynaksiz sayi yazilmaz). Bu betik sayi URETMEZ — index.jsonl'deki API alanlarini
(oldugu gibi) bir satira yazar; toplamlar da ayni dosyada, ayri satirlarda durur.

Cikti: KANIT_db_taramasi.txt
"""
from __future__ import annotations

import json
from pathlib import Path

BURADA = Path(__file__).resolve().parent
DOSYALAR = [("patent", BURADA / "index.jsonl"), ("npl", BURADA / "npl" / "index.jsonl"),
            ("ek", BURADA / "index2.jsonl")]


def oku(yol: Path) -> list[dict]:
    if not yol.exists():
        return []
    return [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines() if s.strip()]


def anahtar(h: dict) -> str:
    return h.get("yayim_no") or h.get("url") or h.get("id") or "?"


def main() -> int:
    satir = [
        "# KANIT — patent/NPL veritabani taramasinin sayim kaydi",
        "# uretim: db_kanit.py | ham: db_taramasi/index.jsonl + db_taramasi/npl/index.jsonl",
        "# her satir: <id> | <kanal> | HTTP | toplam-sayisi(API alani) | ilk-hit | sorgu",
    ]
    toplam_sorgu = 0
    tum_hit: list[str] = []
    for kanal, yol in DOSYALAR:
        kayitlar = oku(yol)
        satir.append(f"\n## {kanal} sorgulari ({len(kayitlar)})")
        for k in kayitlar:
            hitler = k.get("hitler") or []
            ilk = anahtar(hitler[0]) if hitler else "-"
            satir.append(f"{k['id']} | {k.get('kanal', kanal)} | {k.get('durum')} | "
                         f"{k.get('toplam') if k.get('toplam') is not None else k.get('toplam_sonuc')} | "
                         f"{ilk} | {k.get('sorgu','')[:70]}")
            toplam_sorgu += 1
            tum_hit += [anahtar(h) for h in hitler]
    benzersiz = sorted(set(tum_hit))
    sifir = [k["id"] for _, y in DOSYALAR for k in oku(y) if not (k.get("hitler") or [])]
    npl_hit = sum(len(k.get("hitler") or []) for k in oku(DOSYALAR[1][1]))
    satir += [
        "",
        "## Sayim (index.jsonl uzerinden hesaplandi)",
        f"toplam_sorgu | {toplam_sorgu}",
        f"hit_kayit_toplam | {len(tum_hit)}",
        f"benzersiz_hit | {len(benzersiz)}",
        f"bos_donen_sorgu | {len(sifir)} | {','.join(sifir)}",
        f"patent_sorgu | {len(oku(DOSYALAR[0][1])) + len(oku(DOSYALAR[2][1]))}",
        f"patent_hit | {len(tum_hit) - npl_hit}",
        f"npl_sorgu | {len(oku(DOSYALAR[1][1]))}",
        f"npl_hit | {npl_hit}",
    ]
    yol = BURADA / "KANIT_db_taramasi.txt"
    yol.write_text("\n".join(satir) + "\n", encoding="utf-8")
    print(f"-> {yol} ({len(satir)} satir) | sorgu {toplam_sorgu} | hit {len(tum_hit)} | benzersiz {len(benzersiz)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
