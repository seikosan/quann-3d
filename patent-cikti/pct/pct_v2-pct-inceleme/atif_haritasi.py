r"""UYGUNLUK_NOTU'ndeki her sayi icin dogrulanmis <dosya:satir> kuycesi uretir.

Mantik: sayinin gomulu oldugu JSON/txt satirini bulur; notta yazilacak atif
satirda GERCEKGE o sayiyi iceriyorsa gecerlidir. `rapor_denetle.py` de ayni
seyi kontrol eder, bu yuzde cikti dogrudan nota yapistirilir.

Kosum: python3 atif_haritasi.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path("/Users/serkankose/Documents/quann-3d/patent-cikti/pct/pct_v2-pct-inceleme")
PKG = Path("/Users/serkankose/Documents/quann-3d/docs/patent/pct_v2")

# (etiket, dosya, ayni satirda aranacak desen)
ISTEK: list[tuple[str, Path, str]] = [
    ("baslik EN kelime", OUT / "A_pct_en.json", r'"word_count": 6'),
    ("ozet EN kelime", OUT / "A_pct_en.json", r'"word_count": 137'),
    ("istem sayisi EN", OUT / "A_pct_en.json", r'"total_claims": 15'),
    ("baslik TR kelime", OUT / "B_pct_tr.json", r'"word_count": 7'),
    ("ozet TR kelime", OUT / "B_pct_tr.json", r'"word_count": 106'),
    ("D bulgu", OUT / "D_epo_claims_en.json", r'"total_issues": 1'),
    ("D skor", OUT / "D_epo_claims_en.json", r'"compliance_score": 99\.666'),
    ("E bulgu", OUT / "E_epo_claims_tr.json", r'"total_issues": 16'),
    ("F kritik", OUT / "F_epo_spec_support_en.json", r'"critical_issues": 0'),
    ("F onemli", OUT / "F_epo_spec_support_en.json", r'"important_issues": 47'),
    ("F coverage", OUT / "F_epo_spec_support_en.json", r'"percentage": 100'),
    ("G bulgu", OUT / "G_epo_formal_en.json", r'"warnings": 1'),
    ("H skor", OUT / "H_pkg_full.json", r'"package_integrity_score": 100'),
    ("I skor", OUT / "I_pkg_filing.json", r'"package_integrity_score": 100'),
    ("J bulgu", OUT / "J_us_claims_en.json", r'"total_issues": 0'),
    ("uretim EN paragraf", OUT / "uretim.json", r'"file": "DESCRIPTION_FILING-COPY_EN\.txt"'),
    ("KANIT kategori", OUT / "KANIT_artefaktlar.json", r'"15": "product"'),
    ("KANIT point cloud 13", OUT / "KANIT_artefaktlar.json", r'"tarifnamede_gecen_yer_sayisi": 13'),
    ("KANIT tek paragraf", OUT / "KANIT_artefaktlar.json", r'"motorin_atif_yaptigi_paragraf_sayisi": 1'),
    ("KANIT EN bagimli 12", OUT / "KANIT_artefaktlar.json", r'"EN_bağımlı_istem_sayisi": 12'),
    ("KANIT TR bagimli 0", OUT / "KANIT_artefaktlar.json", r'"TR_bağımlı_istem_sayisi": 0'),
    ("KANIT_antecedent ozet", OUT / "KANIT_antecedent.json", r'"B_artikel-terim_arasi_3\+"'),
    ("4096 EN", PKG / "DESCRIPTION_EN.md", r"\[0057\].*4096"),
    ("4096 ham kaynak", Path("/Users/serkankose/Documents/quann-3d/fraud/logs/run_full_scale_edge_eval.log"),
     r"4096 random QUANN-point inputs"),
    ("TR kuyruk duzeltme ornegi", PKG / "ISTEMLER_TR.md", r"^2\. .*tanımlıdır\.$"),
    ("EN group duzeltme", PKG / "CLAIMS_EN.md", r"reference data group and a comparison data group"),
]


def rel(p: Path) -> str:
    return str(p) if p.is_absolute() and "patent-creator-venv" in str(p) else \
        str(p).replace("/Users/serkankose/Documents/quann-3d/", "")


def main() -> int:
    eksik = []
    for etiket, path, desen in ISTEK:
        if not path.exists():
            print(f"| {etiket} | **DOSYA YOK** {rel(path)} | — |")
            eksik.append(etiket)
            continue
        hit = [(i, ln) for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
               if re.search(desen, ln)]
        if not hit:
            print(f"| {etiket} | {rel(path)} | **DESEN YOK** {desen} |")
            eksik.append(etiket)
            continue
        i, ln = hit[0]
        print(f"| {etiket} | `{rel(path)}:{i}` | `{ln.strip()[:78]}` |")
    print(f"\nEKSIK: {len(eksik)} {eksik}")
    return 1 if eksik else 0


if __name__ == "__main__":
    raise SystemExit(main())
