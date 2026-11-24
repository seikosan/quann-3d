"""Kara Kutu sözleşmesi — tek veri/kanıt şeması (docs/KARA_KUTU_PLAN.md §3).

Hiçbir alan "elle sayı" taşımaz: ölçümse değer + "bu koşunun ölçümü" işareti,
değilse `dosya:satır` atfı zorunludur (AGENTS §13). Renderer'lar bu kuralı
`kaynakli()` yardımcısıyla mekanik uygular.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

OLCU = "bu koşunun ölçümü"


@dataclass(frozen=True)
class Kayit:
    """Girdi verisindeki tek kayıt (satır). `ham` ham kolon değerleridir."""

    kimlik: int
    ham: Mapping[str, Any]


@dataclass
class VeriSeti:
    """Adapter'a giren ham veri. `etiketler` varsa girdi-etiketli metrik hesaplanır
    (CSV'nin held-out olup olmadığı kutudan bakarak doğrulanamaz — bkz. plan §9)."""

    ad: str
    kolonlar: list[str]
    kayitlar: list[Kayit]
    kaynak: str = ""
    sha256: str = ""
    etiketler: list[int] | None = None

    def __len__(self) -> int:
        return len(self.kayitlar)


@dataclass(frozen=True)
class Gerekce:
    """Kayıt başına tek gerekçe girdisi: nokta-ablasyon etkisi (logit).

    Yön kelimesi ("destekliyor/karşıtı") dikey-geneli sözleşmede pozitif
    etiketi kullanılarak `yonu_metni` ile üretilir — gerekçe kendisi hangi
    pozitif sınıfın adına yön duyuracağını bilmez.
    """

    etiket: str
    delta_logit: float


def yonu_metni(delta_logit: float, pozitif_etiketi: str | None) -> str:
    """Gerekçenin pozitif yön kelimesini pozitif etiketi kullanarak üretir.

    pozitif_etiketi yoksa (çok sınıflı/dichotomy yok) nötr kelime basılır;
    pozitif-sınıf kelimesi UYDURULMAZ (AGENTS §13).
    """
    if delta_logit > 0:
        return (f"{pozitif_etiketi.lower()}-destekliyor"
                if pozitif_etiketi else "seçileni destekliyor")
    return (f"{pozitif_etiketi.lower()}-karşıtı"
            if pozitif_etiketi else "seçilenin karşıtı")


@dataclass
class Verdict:
    """Kayıt başına çıktı: skor + karar + en güçlü gerekçeler."""

    kimlik: int
    skor: float
    karar: str
    gerekceler: list[Gerekce] = field(default_factory=list)


@dataclass(frozen=True)
class Baseline:
    """Kıyas referansı. `kaynak` alanı `dosya:satır` atfıdır — boş bırakılamaz."""

    ad: str
    metrik: str
    deger: float
    kaynak: str


@dataclass(frozen=True)
class ModelMeta:
    """Model/paket künyesi. sha'lar diskteki dosyadan hesaplanır (kanıt bloğu)."""

    ad: str
    esik: float
    esik_kaynak: str
    olusturma: str
    ek_bilgi: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class InputSpec:
    """Adapter'ın beklediği girdi şeması (web UI ve CLI doğrulaması bunu kullanır)."""

    adapter: str
    kolonlar: tuple[str, ...]
    aciklama: str


@dataclass(frozen=True)
class KanitBlogu:
    """Her koşumun imzası: komut, git HEAD, sha256'lar, tarih."""

    komut: str
    git_head: str
    tarih: str
    model_sha256: str
    dataset_sha256: str
    ek_atiflar: Mapping[str, str] = field(default_factory=dict)


@dataclass
class KosumSonucu:
    """Bir koşumun tam çıktısı — rapor üreticileri yalnız bunu okur.

    `pozitif_etiket`/`negatif_etiket`/`skor_etiket` runner tarafından adapter'dan
    stamped edilir; rapor metinlerinde hangi sınıfa odaklanıldığı bu alanlarla
    taşınır. `pozitif_etiket` None ise dichotomy yoktur (çok sınıflı dikey,
    ör. complaints).
    """

    adapter: str
    esik: float
    satir_sayisi: int
    verdictler: list[Verdict]
    baseline: tuple[Baseline, ...]
    meta: ModelMeta
    kanit: KanitBlogu
    metrikler: dict[str, str] = field(default_factory=dict)
    pozitif_etiket: str | None = None
    negatif_etiket: str | None = None
    skor_etiket: str = "skor"


def kaynakli(deger: Any, kaynak: str | None) -> str:
    """Sayıyı metne çevir; kaynak yoksa `DOĞRULANMADI` etiketini mekanik bas.

    Renderer'lar hiçbir sayıyı bu yardımcısız basmamalı — AGENTS §13'ün kod
    kapısı budur (elle yazılmış DOĞRULANMADI değil, üretilmiş DOĞRULANMADI).
    """
    if kaynak:
        return f"{deger} (`{kaynak}`)"
    return f"{deger} (DOĞRULANMADI)"


@runtime_checkable
class Adapter(Protocol):
    """Dikey adaptör sözleşmesi — docs/KARA_KUTU_PLAN.md §3'teki Protocol'ün kodu."""

    name: str

    def input_spec(self) -> InputSpec: ...

    def predict(self, ds: VeriSeti) -> list[Verdict]: ...

    def baseline(self, ds: VeriSeti) -> tuple[Baseline, ...]: ...

    def model_meta(self) -> ModelMeta: ...
