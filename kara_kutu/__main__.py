"""Kara Kutu CLI — `python3 -m kara_kutu run --adapter fraud --veri x.csv`.

Akış: CSV → VeriSeti → adapter.predict → kanıt bloğu → raporlar. Her koşum
`kara_kutu/runs/<UTC>_<adapter>/` altına `sonuc.json + kanit.json +
ic_rapor.md + dis_rapor.html` yazar. Eval-only; eğitim hatlarına dokunmaz.
Hattın kendisi `kara_kutu/core/runner.py`'de — web arayüzü ile ortak kapı.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kara_kutu.core.runner import ETIKET_KOLON, RUNS_DIR, kosum


def run_komutu(args: argparse.Namespace) -> int:
    veri = Path(args.veri).expanduser()
    if not veri.is_file():
        print(f"CSV bulunamadı: {veri}", file=sys.stderr)
        return 2
    raporlar = {r.strip().lower() for r in args.rapor.split(",") if r.strip()}
    try:
        cikti = kosum(
            args.adapter, veri, limit=args.limit, esik=args.esik, top_k=args.top_k,
            etiket_kolon=args.etiket_kolon, raporlar=raporlar,
            runs_dir=Path(args.runs_dir),
            komut=" ".join(["python3 -m kara_kutu", *sys.argv[1:]]),
        )
    except ValueError as exc:  # _adapter_kur sözleşme hataları
        print(f"kara_kutu: {exc}", file=sys.stderr)
        return 2
    for ad, yol in cikti["rapor_dosyalari"].items():
        print(f"{'iç rapor ' if ad == 'md' else 'dış rapor'}: {yol}")
    o = cikti["ozet"]
    print(f"koşum    : {cikti['dizin']}")
    print(f"özet     : {o['satir_sayisi']} kayıt, {o['fraud_sayisi']} FRAUD "
          f"(eşik {o['esik']:.4f}); kanıt {cikti['dizin'] / 'kanit.json'}")
    return 0


def liste_komutu(_: argparse.Namespace) -> int:
    from kara_kutu.core.registry import kesif

    for ad, a in sorted(kesif().items()):
        spec = a.input_spec()
        print(f"{ad}: {spec.aciklama}")
        print(f"  kolonlar: {', '.join(spec.kolonlar)}")
    return 0


def web_komutu(args: argparse.Namespace) -> int:
    from kara_kutu.web.app import calistir  # geç import: web bağımlılıkları

    return calistir(host=args.host, port=args.port, runs_dir=Path(args.runs_dir))


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
    web = alt.add_parser("web", help="FastAPI + tek sayfa arayüz (eval-only)")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8123)
    web.add_argument("--runs-dir", default=str(RUNS_DIR), dest="runs_dir")
    web.set_defaults(f=web_komutu)
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
