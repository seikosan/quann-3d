"""Kara Kutu — gerçek hayat verisi için tek kapı: tahmin + gerekçe + kanıt raporu.

Tasarım sözleşmesi: `docs/KARA_KUTU_PLAN.md`. Kutu eval-only'dir; eğitim yapmaz,
checkpoint değiştirmez, `logs/` append-only kuralına dokunmaz. Her koşum
`kara_kutu/runs/<id>/` altına yazılır ve kanıt bloğu (git HEAD + sha256 + komut)
üretir (AGENTS §13 sayı-kanıtı kuralının kod hâli).
"""
from __future__ import annotations

__version__ = "0.1.0"  # P0: contract + fraud adapter + CLI + iki rapor
