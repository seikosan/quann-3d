r"""OLCUM_KAYITLARI.md'nin her künyesini ve her sayısını ham kaynakta doğrular (2026-11-19).

Neden bu betik var: rapor kapısı `rapor_denetle.py` tablo satırlarını MUAF sayıyor
(~/.config/opencode/denetim/rapor_denetle.py:31 BASLIK + :72), ölçüm defterinin sayı
yoğunluğu ise tablolarda. Bağımsız doğrulayıcı tur bu boşluğu işaret etti:
"OLCUM_KAYITLARI'nın sayıları hiçbir kapıda değil". Bu betik o boşluğu kapatır.

Bes denetim:
  A) KÜNYE     — her `dosya:satır` / `:satır-satır` / `:satır, satır` adresi açılır; dosya yok,
                 satır aralık dışında veya satır boşsa BULGU.
  B) SAYI      — künye numarası ve paragraf etiketi (`[0057]`) sayı SAYILMAZ. Satırdaki her
                 uzun sayı (≥4 anlamlı rakam) ya atfedilen ham satırlarda BİREBİR geçmeli ya da
                 o satırlardaki bir sayının anlamlı-rakam yuvarlanması olmalı. İkisi de değilse
                 BULGU. (Sadece "en az biri" değil — HER sayı denetlenir.)
  C) EŞİTLİK   — "dosya:NN = DEGER" satırlarında DEGER o satırda birebir geçmeli.
  D) HESAP     — "hesap:" satırları: girdiler ham JSON'dan okunur, türev yeniden kurulur ve
                 defterin yazdığı değere eşitlenir.
  E) PARAGRAF  — defterin atıf yaptığı `[00NN]` tarifname paragrafları gerçekten DESCRIPTION_EN/TR
                 içinde var olmalı; sayısının o paragrafın ham satırında geçmesi zorunlu (UYARI).

Kosum: python3 patent-cikti/pct/pct_v2-pct-inceleme/dogrul_olcum.py
Cikis: 0 = hepsi geçti; 1 = bulgu var.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path("/Users/serkankose/Documents/quann-3d")
OLCUM = REPO / "docs/patent/pct_v2/OLCUM_KAYITLARI.md"
EN = REPO / "docs/patent/pct_v2/DESCRIPTION_EN.md"
TR = REPO / "docs/patent/pct_v2/DESCRIPTION_TR.md"
OUT = REPO / "patent-cikti/pct/pct_v2-pct-inceleme"

DOSYA_ADI = r"[A-Za-z0-9_./$~-]+\.(?:json|md|txt|log|py|csv|sh|svg|yaml|yml)"
# dosya:12 | dosya:12-14 | dosya:12, 16 | dosya:12-14, 20
ADRES = re.compile(rf"({DOSYA_ADI}):((?:\s*\d+(?:\s*-\s*\d+)?\s*,?\s*)+)")
# satir:  "manifest.json:28 = 1.2814998626708984e-06"
ESITLIK = re.compile(rf"({DOSYA_ADI}):(\d+)\s*=\s*([0-9eE.+-]+)")
PARAGRAF = re.compile(r"\[(\d{3,4})\]")
SAYI = re.compile(r"-?\d+\.?\d*(?:e[-+]?\d+)?")
UYGUN = re.compile(r"-?\d+\.\d+(?:e[-+]?\d+)?|-?\d{4,}")


def coz(yol: str) -> Path | None:
    y = Path(yol).expanduser()
    if y.is_absolute():
        return y if y.exists() else None
    return (REPO / yol) if (REPO / yol).exists() else None


def satirlar(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def satir_listesi(spec: str) -> list[int]:
    """`12-14, 20` -> [12,13,14,20]; `422, 838` -> [422,838]."""
    out: list[int] = []
    for parc in spec.split(","):
        parc = parc.strip()
        if not parc:
            continue
        if "-" in parc:
            a, b = parc.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(parc))
    return out


def rakam_sayisi(t: str) -> int:
    """anlamlı rakam sayısı: '1.2815e-06' -> 5, '0.3545' -> 4, '8392' -> 4."""
    govde = t.lstrip("-").split("e")[0].replace(".", "").lstrip("0")
    return len(re.findall(r"\d", govde))


def anlamli_yuvarla(x: float, n: int) -> float:
    return float(f"%.{n}g" % x)


SAYILAR = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")  # isaret korunur: -0.0086 != 0.0086


def ana_sayilar(metin: str) -> tuple[set[float], set[str]]:
    """ham satırlardan sayı kümesi: birebir metin + binlik ayraç temizlenmiş hâli."""
    metin_sade = metin.replace(",", "")
    return ({float(m) for m in SAYILAR.findall(metin_sade)}, set(SAYILAR.findall(metin)) |
            set(SAYILAR.findall(metin_sade)))


def main() -> int:
    bulgu: list[str] = []
    uyari: list[str] = []
    kunye = sayi_gecen = yuvarlanan = tanim = esitlik = hesap_satir = 0

    metin = OLCUM.read_text(encoding="utf-8").splitlines()
    for no, satir in enumerate(metin, 1):
        if not satir.startswith("|"):
            continue

        # --- C) "dosya:NN = DEGER" satirici doğrulama ----------------------------
        for yol, n, deger in ESITLIK.findall(satir):
            p = coz(yol)
            if p is None:
                bulgu.append(f"{no}: C DOSYA YOK {yol}")
                continue
            s = satirlar(p)
            if int(n) > len(s) or deger not in s[int(n) - 1]:
                bulgu.append(f"{no}: C EŞİTLİK TUTMUYOR {yol}:{n} != {deger!r} "
                             f"(satırda: {s[int(n) - 1][:70]!r})")
            else:
                esitlik += 1

        # --- A) her adresi ac -----------------------------------------------------
        adresler = []
        for yol, spec in ADRES.findall(satir):
            p = coz(yol)
            if p is None:
                bulgu.append(f"{no}: A DOSYA YOK {yol}")
                continue
            s = satirlar(p)
            for n in satir_listesi(spec):
                if not (1 <= n <= len(s)):
                    bulgu.append(f"{no}: A SATIR YOK {yol}:{n} (dosya {len(s)} satır)")
                    continue
                if not s[n - 1].strip():
                    bulgu.append(f"{no}: A BOŞ SATIR {yol}:{n}")
                    continue
                adresler.append((f"{yol}:{n}", s[n - 1]))
                kunye += 1

        # --- B) HER uzun sayi atfedilen satirlarda gecmeli -------------------------
        if "hesap:" in satir:      # turetilmis satir: D bolumu ham girdilerden dogrular
            hesap_satir += 1
            continue
        if not adresler:
            continue
        govde = ADRES.sub(" ", satir)
        govde = PARAGRAF.sub(" ", govde)          # [0057] tarifname paragrafı, ölçüm değil
        sayilar = [t for t in UYGUN.findall(govde) if rakam_sayisi(t) >= 4]
        if not sayilar:
            tanim += 1
            continue
        birlesen = "\n".join(txt for _, txt in adresler)
        ham_degerler, ham_metinler = ana_sayilar(birlesen)
        eksik = []
        for t in sayilar:
            if t in ham_metinler:
                sayi_gecen += 1
                continue
            try:
                x = float(t.replace(",", ""))
            except ValueError:
                eksik.append(t)
                continue
            if any(anlamli_yuvarla(r, rakam_sayisi(t)) == x for r in ham_degerler):
                yuvarlanan += 1
                continue
            eksik.append(t)
        if eksik:
            bulgu.append(f"{no}: B SAYI HAM KAYNAKTA YOK {sorted(set(eksik))} "
                         f"künye {[a for a, _ in adresler][:3]}")

        # --- E) paragraf etiketleri tarifnamede var mi -----------------------------
        for etiket in PARAGRAF.findall(satir):
            for ad, dosya in [("EN", EN), ("TR", TR)]:
                if not any(ln.startswith(f"[{etiket}]") for ln in satirlar(dosya)):
                    bulgu.append(f"{no}: E PARAGRAF [{etiket}] {ad} tarifnamesinde yok")

    print(f"A KÜNYE          : açılan {kunye} adres")
    print(f"B SAYI           : birebir {sayi_gecen} · yuvarlanmış {yuvarlanan} "
          f"· satır yok {tanim} · türetilmiş (hesap) {hesap_satir}")
    print(f"C EŞİTLİK        : doğrulanan {esitlik} satır")

    # --- D) "hesap:" satirlari: girdiler ham JSON'dan okunur, oran yeniden kurulur --
    ham = satirlar(REPO / "fraud/results/edge_eval_full_scale.json")
    girdi = []
    for n, etiket in [(29, '"k": 13'), (30, '"macs_per_tx": 2522200'),
                      (41, '"k": 50'), (42, '"macs_per_tx": 34091488')]:
        if etiket not in ham[n - 1]:
            bulgu.append(f"D GİRDİ edge_eval_full_scale.json:{n} içinde {etiket!r} yok "
                         f"(satır: {ham[n - 1][:60]!r})")
        else:
            girdi.append(f"{etiket}@:{n}")

    def sayi_cek(s: str) -> float:
        m = re.search(r"(-?[\d.]+)", s.split(":")[-1])
        if not m:
            raise ValueError(s)
        return float(m.group(1))

    hesaplar = []
    if len(girdi) == 4:
        k13, macs13, k50, macs50 = (sayi_cek(ham[n - 1]) for n in (29, 30, 41, 42))
        for sira, iddia, deger in [(51, "13.5166", macs50 / macs13),
                                   (52, "14.7929", (k50 / k13) ** 2)]:
            satir_metni = metin[sira - 1]
            if iddia not in satir_metni:
                bulgu.append(f"D DEFTER satır {sira} içinde {iddia!r} yok: {satir_metni[:70]!r}")
            elif abs(round(deger, 4) - float(iddia)) > 5e-4:
                bulgu.append(f"D HESAP satır {sira}: defter {iddia} <-> yeniden hesap {deger:.6f}")
            else:
                hesaplar.append(f"satır {sira}: {iddia} = {deger:.6f} ✓ "
                                f"(girdi {int(macs13)} / {int(macs50)})")
    print(f"D HESAP          : {' · '.join(hesaplar)}")

    print(f"UYARI            : {len(uyari)}")
    for s in uyari:
        print("  UYARI  " + s)
    print(f"BULGU            : {len(bulgu)}")
    for s in bulgu:
        print("  BULGU  " + s)

    (OUT / "DOGRUL_olcum.json").write_text(
        json.dumps({"kunye": kunye, "sayi_birebir": sayi_gecen, "sayi_yuvarlanan": yuvarlanan,
                    "satir_sayisiz": tanim, "hesap_satir": hesap_satir, "esitlik": esitlik,
                    "hesap": hesaplar, "uyari": uyari, "bulgu": bulgu, "temiz": not bulgu},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"-> {OUT.relative_to(REPO)}/DOGRUL_olcum.json")
    return 1 if bulgu else 0


if __name__ == "__main__":
    sys.exit(main())
