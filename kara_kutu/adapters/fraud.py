"""Fraud adaptörü — ONNX deploy paketini (`fraud/results/deploy/`) sarar.

Girdi: IBM şemasında işlem CSV'si (kolon listesi `fraud/pilot/infer.py:28-32`).
Skor hattı `fraud/pilot/infer.py` ile birebir aynı sözleşmeyi uygular:
quantizer.transform → ONNX(ids, mask) → P(fraud); gerekçe = tek-puan ablasyon
(logit farkı, `fraud/pilot/infer.py:90-108` deseni). Eval-only: eğitim/ checkpoint
yazmaz. Kaynak paketin sha256'sı manifest'ten okunur, disktekiyle doğrulanır.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from kara_kutu.core.contract import (
    Baseline,
    Gerekce,
    InputSpec,
    ModelMeta,
    Verdict,
    VeriSeti,
)

REPO_KOK = Path(__file__).resolve().parents[2]
FRAUD_SRC = REPO_KOK / "fraud" / "src"
if str(FRAUD_SRC) not in sys.path:
    sys.path.insert(0, str(FRAUD_SRC))  # repo konvansiyonu (REFACTOR C7: sys.path KEEP)

# Gerekli IBM kolonları — kaynak: fraud/pilot/infer.py:28-32 (birebir aynı küme).
IBM_KOLONLARI: tuple[str, ...] = (
    "Year", "Month", "Day", "Time", "Amount", "Use Chip",
    "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
    "Errors?",
)

# Karar eşiği — KARAR A (kullanıcı, 2026-11-22, KARA_KUTU_PLAN §9): Ekim eşiği
# 0,9969 (`fraud/pilot/infer.py:40`) port-öncesi FP32 olasılıklarıyla kalibre
# edildi; kutu tam te'de maks p=0,993219 verdiğinden o eşik kutuda anlamsız
# (`kara_kutu/runs/alarm_farki_20261121-232405/ozet.json`). Kutu artık kendi
# best-F2 kalibrasyonunu `kara_kutu/artifacts/esik_fraud.json`'dan okur
# (üretici: `kara_kutu/tools/esik_kalibre.py`). Ekim sayısı yalnız referans:
ESIK_EKIM = 0.9969
ESIK_ARTEFACT = REPO_KOK / "kara_kutu" / "artifacts" / "esik_fraud.json"


def _esik_artefakttan() -> tuple[float, str]:
    """Artefakt yoksa patlar — sessiz Ekim eşiğine düşmek YASAK (Karar A)."""
    if not ESIK_ARTEFACT.is_file():
        raise RuntimeError(
            f"Kutu-yerel eşik artefaktı yok: {ESIK_ARTEFACT}. Üret: "
            "python3 -m kara_kutu.tools.esik_kalibre"
        )
    with ESIK_ARTEFACT.open(encoding="utf-8") as f:
        art = json.load(f)
    kaynak = (f"kara_kutu/artifacts/esik_fraud.json ({art['kural']}, "
              f"tohum {art['us']}; kanıt: {art['kanit_kosum']})")
    return float(art["esik"]), kaynak

# Baseline tavanları — kaynak: docs/market_priorities.md:8 ("RF .973 / GBT .981").
# Model referansı (kıyas için, baseline değil): FP32 AUC 0.9451 ve INT8 0.9367,
# fraud/logs/run_full_scale_edge_eval.log:52 ve :54.
_TOPK_VARSAYILAN = 5
_BATCH = 256


def _fraud_quantizer_sinifi():
    """`fraud/src/quantize.py` sınıfı — modül adı çakışmasına kapalı izolasyon.

    aml/tests `sys.path.insert(0, aml/src)` ile kendi `quantize`/`config`/
    `features`/`datasets_kaggle` modüllerini aynı üst-düzey adlarla önbelleğe
    alıyor (tam takım koşusunda `AttributeError: ... no attribute 'from_dict'`
    oradan geliyordu). Zinciri geçici olarak önbellekten düşürüp fraud/src'ten
    tazeler, sonra eski önbelleği geri koyarız: fraud.quantize'ın globals'ı
    kendi zincirine bağlandığı için geri yükleme güvenlidir.
    """
    import importlib

    zincir = ("config", "features", "datasets_kaggle", "quantize")
    hedef = (FRAUD_SRC / "quantize.py").resolve()
    eski = {ad: sys.modules.pop(ad, None) for ad in zincir}
    # fraud/src'i her koşulda listenin BAŞINA taşı: aml/src gibi aynı adlı
    # modül taşıyıcılar test sırasıyla öne geçmiş olabilir.
    eski_yol = list(sys.path)
    while str(FRAUD_SRC) in sys.path:
        sys.path.remove(str(FRAUD_SRC))
    sys.path.insert(0, str(FRAUD_SRC))
    try:
        importlib.invalidate_caches()
        import quantize  # noqa: PLC0415 — izolasyon bloğu içinde kasıtlı

        if Path(getattr(quantize, "__file__", "")).resolve() != hedef:
            raise RuntimeError(f"yanlış quantize yüklendi: {quantize.__file__}")
        return quantize.FeatureQuantizer
    finally:
        for ad, mod in eski.items():
            sys.modules.pop(ad, None)
            if mod is not None:
                sys.modules[ad] = mod
        sys.path[:] = eski_yol


class FraudAdapter:
    """Deploy paketini belleğe bir kez yükler; kayıt başına Verdict üretir."""

    name = "fraud"

    def __init__(self, deploy_dir: Path | None = None, top_k: int = _TOPK_VARSAYILAN,
                 esik: float | None = None) -> None:
        self.deploy = Path(deploy_dir) if deploy_dir else REPO_KOK / "fraud" / "results" / "deploy"
        self.top_k = top_k
        # Eşik tembel okunur: ADAPTER import'u artefakt yokken de kurulabilsin;
        # ilk erişimde artefakt yoksa RuntimeError (sessiz Ekim eşiğine düşülmez).
        self._esik_ov = float(esik) if esik is not None else None
        self._esik_kaynak_ov: str | None = None
        self._esik_cache: tuple[float, str] | None = None
        self._paket: tuple[Any, ...] | None = None

    # ------------------------------------------------------------------- eşik
    def _esik_artefakt(self) -> tuple[float, str]:
        if self._esik_cache is None:
            self._esik_cache = _esik_artefakttan()
        return self._esik_cache

    @property
    def esik(self) -> float:
        if self._esik_ov is not None:
            return self._esik_ov
        return self._esik_artefakt()[0]

    @esik.setter
    def esik(self, v: float) -> None:
        self._esik_ov = float(v)

    @property
    def esik_kaynak(self) -> str:
        if self._esik_kaynak_ov is not None:
            return self._esik_kaynak_ov
        return self._esik_artefakt()[1]

    @esik_kaynak.setter
    def esik_kaynak(self, v: str) -> None:
        self._esik_kaynak_ov = v

    # ------------------------------------------------------------------ paket
    def _yukle(self) -> tuple[Any, ...]:
        if self._paket is not None:
            return self._paket
        try:
            import numpy as np
            import onnxruntime as ort
            import pandas as pd
            FeatureQuantizer = _fraud_quantizer_sinifi()  # fraud/src/quantize.py
        except ImportError as exc:  # pragma: no cover — ortam hatası, test dışı
            raise RuntimeError(
                f"Eksik bağımlılık ({getattr(exc, 'name', exc)}). "
                "Kurulum: pip install -r fraud/pilot/requirements.txt"
            ) from exc
        yollar = {
            "manifest": self.deploy / "manifest.json",
            "model": self.deploy / "model.onnx",
            "quantizer": self.deploy / "quantizer.json",
            "decoder": self.deploy / "token_decoder.json",
        }
        eksik = [str(y) for y in yollar.values() if not y.is_file()]
        if eksik:
            raise RuntimeError("Deploy paketi eksik: " + ", ".join(eksik))
        with yollar["quantizer"].open(encoding="utf-8") as f:
            quantizer = FeatureQuantizer.from_dict(json.load(f))
        with yollar["decoder"].open(encoding="utf-8") as f:
            decoder = json.load(f)
        with yollar["manifest"].open(encoding="utf-8") as f:
            manifest = json.load(f)
        session = ort.InferenceSession(str(yollar["model"]),
                                       providers=["CPUExecutionProvider"])
        self._paket = (np, pd, quantizer, decoder, session, manifest)
        return self._paket

    # -------------------------------------------------------------- sözleşme
    def input_spec(self) -> InputSpec:
        return InputSpec(
            adapter=self.name,
            kolonlar=IBM_KOLONLARI,
            aciklama="IBM kart işlem şeması; etiket kolonu 'Is Fraud?' (Yes/No) "
                     "varsa AUC-ROC ölçülür (girdi CSV'nin held-out durumu doğrulanmaz).",
        )

    def model_dosyasi(self) -> Path:
        """Kanıt bloğunun sha256'sını alacağı model dosyası (packager API'si)."""
        return self.deploy / "model.onnx"

    def model_meta(self) -> ModelMeta:
        *_, manifest = self._yukle()
        art = manifest["artifacts"]
        return ModelMeta(
            ad="QUANN fraud hybrid (ONNX deploy)",
            esik=self.esik,
            esik_kaynak=self.esik_kaynak,
            olusturma=str(manifest["generated"]),
            ek_bilgi={
                "paket_sha256_manifest": str(art["model_onnx_sha256"]),
                "checkpoint_sha256_manifest": str(manifest["checkpoint"]["sha256"]),
                "checkpoint_yol": str(manifest["checkpoint"]["path"]),
                "model_boyut": f"D={manifest['model']['D']} d_model={manifest['model']['d_model']} "
                               f"K={manifest['model']['K']} quann_points={manifest['model']['quann_points']}",
                "manifest": "fraud/results/deploy/manifest.json",
            },
        )

    def baseline(self, ds: VeriSeti) -> tuple[Baseline, ...]:
        return (
            Baseline("Random Forest (standart tavan)", "AUC-ROC", 0.973,
                     "docs/market_priorities.md:8"),
            Baseline("Gradient Boosted Trees (standart tavan)", "AUC-ROC", 0.981,
                     "docs/market_priorities.md:8"),
        )

    # ---------------------------------------------------------------- tahmin
    def _frame(self, ds: VeriSeti):
        _, pd, *_ = self._yukle()
        kolonlar = list(ds.kolonlar)
        satirlar = [dict(k.ham) for k in ds.kayitlar]
        return pd.DataFrame(satirlar, columns=kolonlar)

    def predict(self, ds: VeriSeti) -> list[Verdict]:
        np, _, quantizer, decoder, session, _ = self._yukle()
        eksik = sorted(set(IBM_KOLONLARI) - set(ds.kolonlar))
        if eksik:
            raise ValueError("CSV'de gerekli IBM kolonları eksik: " + ", ".join(eksik))
        frame = self._frame(ds)
        if frame.empty:
            return []
        tokens = quantizer.transform(frame)
        kararlar: list[Verdict] = []
        for start in range(0, len(frame), _BATCH):
            dur = min(start + _BATCH, len(frame))
            probs, katkilar = self._chunk(np, session, tokens[start:dur])
            for yerel, prob in enumerate(probs):
                i = start + yerel
                token_satiri = tokens[i]
                sira = np.argsort(-np.abs(katkilar[yerel]), kind="stable")
                gerekceler: list[Gerekce] = []
                for j in sira:
                    delta = float(katkilar[yerel, j])
                    if abs(delta) < 1e-6:
                        continue
                    bilgi = decoder.get(str(int(token_satiri[j])))
                    etiket = bilgi["label"] if bilgi else f"token_id={int(token_satiri[j])}"
                    gerekceler.append(Gerekce(etiket=etiket, delta_logit=delta))
                    if len(gerekceler) == self.top_k:
                        break
                kararlar.append(Verdict(
                    kimlik=ds.kayitlar[i].kimlik,
                    skor=float(prob),
                    karar="FRAUD" if prob >= self.esik else "NORMAL",
                    gerekceler=gerekceler,
                ))
        return kararlar

    @staticmethod
    def _chunk(np, session, tokenlar):
        """(p, katkı) — desen: fraud/pilot/infer.py:90-108 (tek-puan ablasyon)."""
        masks = (tokenlar != 0) & (tokenlar != 1)
        probs = session.run(None, {"ids": tokenlar, "mask": masks})[0].reshape(-1)
        n, f = tokenlar.shape
        varyant = np.repeat(tokenlar[:, None, :], f, axis=1).reshape(n * f, f)
        vmask = (varyant != 0) & (varyant != 1)
        sat = np.repeat(np.arange(n), f)
        kol = np.tile(np.arange(f), n)
        vmask[sat, kol] = False
        ablate = session.run(None, {"ids": varyant, "mask": vmask})[0].reshape(n, f)
        katki = _logit(np, probs)[:, None] - _logit(np, ablate)
        return probs, katki

    # -------------------------------------------------------------- metrikler
    def ek_metrikler(self, ds: VeriSeti, verdictler: list[Verdict]) -> dict[str, str]:
        """Etiket varsa girdi CSV etiketleri üzerinde AUC-ROC ölçer. Girdinin
        held-out olduğu doğrulanmaz (kanıt: User0 paketi eğitim bölmesinde —
        tools/auc_farki ölçümü user0_fold=5, fraud/src/train_quann.py:58)."""
        if ds.etiketler is None or len(verdictler) < 2:
            return {}
        y = list(ds.etiketler)
        if len(set(y)) < 2:
            return {}
        p = [v.skor for v in verdictler]
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y, p))
        except ImportError:  # pragma: no cover
            return {"AUC-ROC (girdi CSV etiketleri)": "ölçülemedi: sklearn yok (DOĞRULANMADI)"}
        return {"AUC-ROC (girdi CSV etiketleri, bu koşunun ölçümü)": f"{auc:.4f}"}


def _logit(np, p):
    """Clipped logit — desen: fraud/pilot/infer.py:85-87."""
    q = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(q) - np.log1p(-q)


ADAPTER = FraudAdapter()
