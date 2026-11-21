"""Koşum hattı — CLI ve web'in ORTAK tek kapısı (eval-only sözleşme burada).

Akış: CSV → VeriSeti → adapter.predict → kanıt bloğu → raporlar →
`runs/<UTC>_<adapter>/` altına `sonuc.json + kanit.json + ic_rapor.md +
dis_rapor.html`. Eğitim yok, checkpoint yazımı yok, `logs/`-a dokunulmaz
(AGENTS §3.3); tek yazma hedefi `runs_dir` içidir.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from kara_kutu import __version__
from kara_kutu.core.contract import Kayit, VeriSeti
from kara_kutu.core.registry import get_adapter
from kara_kutu.report.external import uret as uret_dis
from kara_kutu.report.internal import uret as uret_ic
from kara_kutu.report.provenance import kanit_olustur

REPO_KOK = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_KOK / "kara_kutu" / "runs"
ETIKET_KOLON = "Is Fraud?"  # IBM şeması; Yes/No → 1/0 (semantik: fraud/src/datasets.py:30-32)


def veri_seti(yol: Path, limit: int | None, etiket_kolon: str) -> VeriSeti:
    import pandas as pd  # yerel import: `liste` alt-komutu pandas istemez

    frame = pd.read_csv(yol, nrows=limit)
    kolonlar = [str(c) for c in frame.columns]
    kayitlar = [Kayit(i, rec) for i, rec in enumerate(frame.to_dict("records"))]
    etiketler: list[int] | None = None
    if etiket_kolon in kolonlar:
        etiketler = [
            1 if str(v).strip().lower() == "yes" else 0
            for v in frame[etiket_kolon]
        ]
    from kara_kutu.report.provenance import sha256_dosya

    return VeriSeti(ad=yol.name, kolonlar=kolonlar, kayitlar=kayitlar,
                    kaynak=str(yol), sha256=sha256_dosya(yol), etiketler=etiketler)


def adapter_kur(ad: str, esik: float | None, top_k: int | None):
    a = copy.copy(get_adapter(ad))  # registry örneğini mutasyona sokma
    if esik is not None:
        if not hasattr(a, "esik"):
            raise ValueError(f"{ad} adaptöründe eşik kavramı yok")
        a.esik = float(esik)
        if hasattr(a, "esik_kaynak"):  # override, varsayılan kaynağın yerini alır
            a.esik_kaynak = "kullanıcı override (CLI --esik / web formu)"
    if top_k is not None:
        if not hasattr(a, "top_k"):
            raise ValueError(f"{ad} adaptöründe top-k kavramı yok")
        a.top_k = int(top_k)
    return a


def kosum(ad: str, veri: Path, *, limit: int | None = None, esik: float | None = None,
          top_k: int | None = None, etiket_kolon: str = ETIKET_KOLON,
          raporlar: set[str] | None = None, runs_dir: Path = RUNS_DIR,
          komut: str | None = None) -> dict:
    """Tek koşum; `{dizin, sonuc, ozet}` döner. Rapor metinleri diske yazılır.

    `komut` kanıt bloğundaki komut alanıdır — CLI kendi argv'sini geçirir;
    web çağrısı üreten tarafı açıkça yazar (kayıt dürüstlüğü).
    """
    raporlar = {"md", "html"} if raporlar is None else {r.lower() for r in raporlar}
    adapter = adapter_kur(ad, esik, top_k)
    ds = veri_seti(veri, limit, etiket_kolon)

    verdictler = adapter.predict(ds)
    metrikler: dict[str, str] = {}
    if hasattr(adapter, "ek_metrikler"):
        metrikler.update(adapter.ek_metrikler(ds, verdictler))

    model_yolu = Path(adapter.model_dosyasi()) if hasattr(adapter, "model_dosyasi") \
        else Path(__import__(type(adapter).__module__.rsplit(".", 1)[0], fromlist=["x"]).__file__)
    kanit = kanit_olustur(
        komut=komut or f"kara_kutu.core.runner.kosum(adapter={ad})",
        veri_yolu=veri, model_yolu=model_yolu,
        ek_atiflar={"adapter": adapter.name, "surum": __version__},
    )
    # manifest ↔ disk sha doğrulaması — kanıt bloğunun kendi kendini denetimi
    meta = adapter.model_meta()
    manifest_sha = meta.ek_bilgi.get("paket_sha256_manifest", "")
    if manifest_sha:
        metrikler["model.onnx sha256 ↔ manifest"] = (
            "EVET ✓" if manifest_sha == kanit.model_sha256 else "HAYIR ✗ (DOĞRULANMADI)"
        )

    from kara_kutu.core.contract import KosumSonucu

    sonuc = KosumSonucu(
        adapter=adapter.name, esik=float(getattr(adapter, "esik", 0.0)),
        satir_sayisi=len(verdictler), verdictler=verdictler,
        baseline=tuple(adapter.baseline(ds)), meta=meta, kanit=kanit,
        metrikler=metrikler,
    )

    taban = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "_" + adapter.name
    dizin = Path(runs_dir) / taban
    carpisma = 1
    while dizin.exists():  # aynı saniyede ikinci koşum (web eşzamanlı) — üzerine yazma
        dizin = Path(runs_dir) / f"{taban}_{carpisma}"
        carpisma += 1
    dizin.mkdir(parents=True)
    (dizin / "sonuc.json").write_text(
        json.dumps(dataclasses.asdict(sonuc), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    (dizin / "kanit.json").write_text(
        json.dumps(dataclasses.asdict(kanit), ensure_ascii=False, indent=2),
        encoding="utf-8")
    uretilen: dict[str, str] = {}
    if "md" in raporlar:
        yol = dizin / "ic_rapor.md"
        yol.write_text(uret_ic(sonuc), encoding="utf-8")
        uretilen["md"] = str(yol)
    if "html" in raporlar:
        yol = dizin / "dis_rapor.html"
        yol.write_text(uret_dis(sonuc), encoding="utf-8")
        uretilen["html"] = str(yol)

    fraud_n = sum(1 for v in verdictler if v.karar == "FRAUD")
    return {"dizin": dizin, "sonuc": sonuc, "rapor_dosyalari": uretilen,
            "ozet": {"kosum": dizin.name, "adapter": adapter.name,
                     "satir_sayisi": len(verdictler), "fraud_sayisi": fraud_n,
                     "esik": sonuc.esik, "metrikler": metrikler,
                     "raporlar": uretilen}}
