"""Adapter keşfi: `kara_kutu/adapters/*.py` içindeki modül düzeyli `ADAPTER`.

Ek kaynak: `KARA_KUTU_ADAPTERS` ortam değişkeni (os.pathsep ile ayrılmış .py
dosyaları/klasörleri) — dış plugin adayı, P2/P3'te kullanılacak (plan §3).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from kara_kutu.core.contract import Adapter

ADAPTERS_DIR = Path(__file__).resolve().parents[1] / "adapters"
ENV_VAR = "KARA_KUTU_ADAPTERS"


def _aday_dosyalar() -> list[Path]:
    adaylar = sorted(ADAPTERS_DIR.glob("*.py"))
    extra = os.environ.get(ENV_VAR, "").strip()
    if extra:
        for parca in extra.split(os.pathsep):
            p = Path(parca).expanduser()
            if p.is_dir():
                adaylar.extend(sorted(p.glob("*.py")))
            elif p.is_file():
                adaylar.append(p)
    return [a for a in adaylar if a.name != "__init__.py"]


def _modul_yukle(py: Path):
    mod_adi = f"kara_kutu._adapter_{py.stem}"
    if mod_adi in sys.modules:
        return sys.modules[mod_adi]
    spec = importlib.util.spec_from_file_location(mod_adi, py)
    if spec is None or spec.loader is None:
        raise ImportError(f"adapter modülü yüklenemedi: {py}")
    modul = importlib.util.module_from_spec(spec)
    sys.modules[mod_adi] = modul
    try:
        spec.loader.exec_module(modul)
    except Exception:
        sys.modules.pop(mod_adi, None)
        raise
    return modul


def kesif() -> dict[str, Adapter]:
    """ADAPTER nesnesi taşıyan tüm modülleri {name: adapter} olarak döndürür.

    Bozuk bir adapter diğerlerini engellemesin: yükleme hatası stderr'e düşer,
    keşif devam eder (testler bilinçli olarak fraud'un bulunmasını bekler).
    """
    bulunan: dict[str, Adapter] = {}
    for py in _aday_dosyalar():
        try:
            modul = _modul_yukle(py)
        except Exception as exc:  # noqa: BLE001 — kasıtlı geniş: izole başarısızlık
            print(f"kayıt-dışı ({py.name}): {exc}", file=sys.stderr)
            continue
        adapter = getattr(modul, "ADAPTER", None)
        if isinstance(adapter, Adapter):
            bulunan[adapter.name] = adapter
    return bulunan


def list_adapters() -> list[str]:
    return sorted(kesif())


def get_adapter(ad: str) -> Adapter:
    bulunan = kesif()
    if ad not in bulunan:
        raise KeyError(
            f"adapter bulunamadı: {ad!r}. Kayıtlılar: {', '.join(sorted(bulunan)) or '-'}"
        )
    return bulunan[ad]
