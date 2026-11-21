"""Kanıt bloğu üretimi — her koşumun makine imzası (plan §3 `report/provenance.py`).

`devir_yaz.py` ruh kardeşi: hiçbir alan elle yazılmaz; komut, git HEAD, sha256
ve UTC damgası koşulur. Raporlar yalnız bu bloğun değerlerini basar.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from kara_kutu.core.contract import KanitBlogu


def sha256_dosya(yol: Path) -> str:
    """Dosya içeriğinin sha256'sı (1 MiB bloklarla; büyük paketlere dayanıklı)."""
    h = hashlib.sha256()
    with Path(yol).open("rb") as f:
        for blok in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blok)
    return h.hexdigest()


def git_head(depo: Path) -> str:
    """Deponun tam HEAD hash'i; git yoksa/ağaç kirliyse dürüst etiket döner."""
    try:
        r = subprocess.run(["git", "-C", str(depo), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "GIT-YOK"


def kanit_olustur(*, komut: str, veri_yolu: Path, model_yolu: Path,
                  ek_atiflar: dict[str, str] | None = None) -> KanitBlogu:
    """KosumSonucu'na girecek KanitBlogu — ölçülen değerler, atıf değil."""
    depo = Path(__file__).resolve().parents[2]
    return KanitBlogu(
        komut=komut,
        git_head=git_head(depo),
        tarih=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        model_sha256=sha256_dosya(model_yolu),
        dataset_sha256=sha256_dosya(veri_yolu),
        ek_atiflar={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **(ek_atiflar or {}),
        },
    )
