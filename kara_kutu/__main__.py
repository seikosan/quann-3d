"""Kara Kutu CLI — `python3 -m kara_kutu run --adapter fraud --veri x.csv`.

Akış: CSV → VeriSeti → adapter.predict → kanıt bloğu → raporlar. Her koşum
`kara_kutu/runs/<UTC>_<adapter>/` altına `sonuc.json + kanit.json +
ic_rapor.md + dis_rapor.html` yazar. Eval-only; eğitim hatlarına dokunmaz.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from kara_kutu import __version__
from kara_kutu.core.contract import Kayit, VeriSeti
from kara_kutu.core.registry import get_adapter, kesif
from kara_kutu.report.external import uret as uret_dis
from kara_kutu.report.internal import uret as uret_ic
from kara_kutu.report.provenance import kanit_olustur

REPO_KOK = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_KOK / "kara_kutu" / "runs"
ETIKET_KOLON = "Is Fraud?"  # IBM şeması; Yes/No → 1/0 (semantik: fraud/src/datasets.py:30-32)


def _veri_seti(yol: Path, limit: int | None, etiket_kolon: str) -> "VeriSeti":
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


def _adapter_kur(ad: str, esik: float | None, top_k: int | None):
    a = copy.copy(get_adapter(ad))  # registry örneğini mutasyona sokma
    if esik is not None:
        if not hasattr(a, "esik"):
            raise ValueError(f"{ad} adaptöründe eşik kavramı yok")
        a.esik = float(esik)
    if top_k is not None:
        if not hasattr(a, "top_k"):
            raise ValueError(f"{ad} adaptöründe top-k kavramı yok")
        a.top_k = int(top_k)
    return a


def run_komutu(args: argparse.Namespace) -> int:
    veri = Path(args.veri).expanduser()
    if not veri.is_file():
        print(f"CSV bulunamadı: {veri}", file=sys.stderr)
        return 2
    adapter = _adapter_kur(args.adapter, args.esik, args.top_k)
    ds = _veri_seti(veri, args.limit, args.etiket_kolon)

    verdictler = adapter.predict(ds)
    metrikler: dict[str, str] = {}
    if hasattr(adapter, "ek_metrikler"):
        metrikler.update(adapter.ek_metrikler(ds, verdictler))

    model_yolu = Path(adapter.model_dosyasi()) if hasattr(adapter, "model_dosyasi") \
        else Path(sys.modules[type(adapter).__module__.rsplit(".", 1)[0]].__file__)
    kanit = kanit_olustur(
        komut=" ".join(["python3 -m kara_kutu", *sys.argv[1:]]),
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

    raporlar = {r.strip().lower() for r in args.rapor.split(",") if r.strip()}
    dizin = Path(args.runs_dir) / (
        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "_" + adapter.name)
    dizin.mkdir(parents=True, exist_ok=True)
    (dizin / "sonuc.json").write_text(
        json.dumps(dataclasses.asdict(sonuc), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    (dizin / "kanit.json").write_text(
        json.dumps(dataclasses.asdict(kanit), ensure_ascii=False, indent=2),
        encoding="utf-8")
    if "md" in raporlar:
        yol = dizin / "ic_rapor.md"
        yol.write_text(uret_ic(sonuc), encoding="utf-8")
        print(f"iç rapor : {yol}")
    if "html" in raporlar:
        yol = dizin / "dis_rapor.html"
        yol.write_text(uret_dis(sonuc), encoding="utf-8")
        print(f"dış rapor: {yol}")
    fraud_n = sum(1 for v in verdictler if v.karar == "FRAUD")
    print(f"koşum    : {dizin}")
    print(f"özet     : {len(verdictler)} kayıt, {fraud_n} FRAUD "
          f"(eşik {sonuc.esik:.4f}); kanıt {dizin / 'kanit.json'}")
    return 0


def liste_komutu(_: argparse.Namespace) -> int:
    for ad, a in sorted(kesif().items()):
        spec = a.input_spec()
        print(f"{ad}: {spec.aciklama}")
        print(f"  kolonlar: {', '.join(spec.kolonlar)}")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python3 -m kara_kutu",
                                description="Kara Kutu — tahmin + gerekçe + kanıt raporu")
    alt = p.add_subparsers(dest="komut", required=True)
    run = alt.add_parser("run", help="bir CSV'yi bir adapter'dan geçir")
    run.add_argument("--adapter", required=True, help="adapter adı (bkz. `liste`)")
    run.add_argument("--veri", required=True, help="girdi CSV dosyası")
    run.add_argument("--rapor", default="html,md", help="virgülle: html,md")
    run.add_argument("--esik", type=float, default=None,
                     help="karar eşiği (adapter varsayılanı; fraud: 0.9969)")
    run.add_argument("--top-k", type=int, default=None, dest="top_k",
                     help="kayıt başına gösterilen gerekçe sayısı")
    run.add_argument("--limit", type=int, default=None, help="en fazla N satır oku")
    run.add_argument("--etiket-kolon", default=ETIKET_KOLON, dest="etiket_kolon")
    run.add_argument("--runs-dir", default=str(RUNS_DIR), dest="runs_dir")
    run.set_defaults(f=run_komutu)
    alt.add_parser("liste", help="kayıtlı adapterları listele").set_defaults(
        f=liste_komutu)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.f(args))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"kara_kutu: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
