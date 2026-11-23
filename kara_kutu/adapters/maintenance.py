"""Maintenance adaptörü — E0b binary edge ONNX (rul≤30) + train-fold mu/sd artefaktı.

Girdi: pencere CSV — bir satır = bir pencere; kolon `engine_id`, `rul` (int, window-end RUL)
ve 360 özellik kolonu `f{k}_{j}` (k = 0..9 pencere adımı, j = 0..35 özellik). Skor hattı:
mu/sd (`maintenance/results/e0b_artifacts/e0b_preprocessing.json`, train-fold-only) →
edge ONNX `maintenance/results/edge/model.onnx` → P(critical). Eşik: E0b val fold'u
(`maintenance/results/e0b_cmapss.json` cloud.backprop.val.threshold).

Gerekçe = pencere adım ablasyonu: adım k sıfırlanır (mu/sd standartlaştırmasında 0 =
standartlaştırılmış medyan, mu/sd zaten train-fold'tan fit edildi), P değişimi ölçülür;
5 en büyük |Δ| adım raporlanır. Eval-only: eğitim/checkpoint yazmaz; test fold hiç okunmaz.

NOT — bu adaptör, edge manifest'in söylediği gibi scaler'sızdır (manifest
`preprocessing_note` "scaler not included"); bu sebeple mu/sd artefaktı ayrıca üretilir
(`python3 -m kara_kutu.tools.maintenance_artefakt`). Multi-class/hybrid/stage edge ONNX
YOK (plan §7) — bu kusuru yutturmaz.
"""
from __future__ import annotations

import hashlib
import json
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
EDGE_DIR = REPO_KOK / "maintenance" / "results" / "edge"
ARTIFACT = REPO_KOK / "maintenance" / "results" / "e0b_artifacts" / "e0b_preprocessing.json"
E0B_JSON = "maintenance/results/e0b_cmapss.json"

_TOPK = 5
WINDOW = 10
FEATURE_DIM = 36
# E0b binary cloud/backprop val-fold eşiği — maintenance/results/e0b_cmapss.json:217
# (cloud.backprop.val.threshold; val 305 pencere, engine-disjoint 80/20 train-fold; seed 20261016).
# Adaptör bunu OKUYARAK varsayılan eşik yapar; elle yazılmaz/icat edilmez.
ESIK_VARSAYILAN = 0.019337554449836418
ESIK_KAYNAK = ("maintenance/results/e0b_cmapss.json tasks.binary.cloud.backprop.val.threshold "
               "(:217, val 305 pencere, engine-disjoint 80/20; seed 20261016)")


class MaintenanceAdapter:
    """Edge paketi + mu/sd artefaktını bir kez yükler; kayıt başına Verdict üretir."""

    name = "maintenance"

    def __init__(self, edge_dir: Path | None = None, artefakt: Path | None = None,
                 top_k: int = _TOPK, esik: float | None = None) -> None:
        self.edge = Path(edge_dir) if edge_dir else EDGE_DIR
        self.artefakt = Path(artefakt) if artefakt else ARTIFACT
        self.top_k = int(top_k)
        self.esik = ESIK_VARSAYILAN if esik is None else float(esik)
        self.esik_kaynak = ESIK_KAYNAK  # runner override'da metni değiştirir
        self._paket: tuple[Any, ...] | None = None

    # ------------------------------------------------------------------ paket
    def _yukle(self) -> tuple[Any, ...]:
        """(np, session, mu, sd, pencere, boyut, artefakt_sha, model_sha_manifest) — tembel."""
        if self._paket is not None:
            return self._paket
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover — ortam hatası
            raise RuntimeError(
                f"Eksik bağımlılık ({getattr(exc, 'name', exc)}). maintenance edge hattı "
                "numpy + onnxruntime ister."
            ) from exc
        model_yol = self.edge / "model.onnx"
        eksik = []
        if not model_yol.is_file():
            eksik.append(str(model_yol))
        if not self.artefakt.is_file():
            eksik.append(str(self.artefakt))
        if eksik:
            raise RuntimeError(
                "Maintenance edge paketi eksik: " + ", ".join(eksik) +
                " — artefakt üret: python3 -m kara_kutu.tools.maintenance_artefakt"
            )
        with self.artefakt.open(encoding="utf-8") as f:
            artefakt = json.load(f)
        mu = np.asarray(artefakt["mu"], dtype=np.float64)
        sd = np.asarray(artefakt["sd"], dtype=np.float64)
        if mu.shape != (int(artefakt["feature_dim"]),) or sd.shape != mu.shape:
            raise RuntimeError(f"Artefakt şekil hatası: mu={mu.shape} sd={sd.shape}")
        session = ort.InferenceSession(str(model_yol),
                                       providers=["CPUExecutionProvider"])
        with self.artefakt.open("rb") as f:
            artefakt_sha = hashlib.sha256(f.read()).hexdigest()
        with (self.edge / "manifest.json").open(encoding="utf-8") as f:
            manifest = json.load(f)
        model_sha_manifest = str(manifest["artifacts"]["model_onnx"]["sha256"])
        self._paket = (np, session, mu.astype(np.float32), sd.astype(np.float32),
                       int(artefakt["window"]), int(artefakt["feature_dim"]),
                       artefakt_sha, model_sha_manifest)
        return self._paket

    # ------------------------------------------------------------------ girdi
    @staticmethod
    def _pencere_matrisi(ham: dict, np, window: int, dim: int):
        vals = np.empty((window, dim), dtype=np.float64)
        for k in range(window):
            for j in range(dim):
                ham_j = ham.get(f"f{k}_{j}")
                if ham_j is None:
                    raise ValueError(
                        f"Pencere kolonu eksik: f{k}_{j} — kolon kümesi engine_id, rul + "
                        "f{k}_{j} (k=0..9, j=0..35)"
                    )
                try:
                    vals[k, j] = float(ham_j)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Pencere kolonu sayı değil: f{k}_{j}={ham_j!r}") from exc
        return vals

    def _onnx_kos(self, session, onnx_batch):
        return session.run(["probabilities"], {"cloud": onnx_batch})[0]

    # ----------------------------------------------------------------- sözleşme
    def input_spec(self) -> InputSpec:
        return InputSpec(
            adapter="maintenance",
            kolonlar=("engine_id", "rul"),
            aciklama=("Bir satır = bir 10×36 pencere. Zorunlu kolon: engine_id (kimlik), "
                      "rul (int, window-end RUL — binary etiket rul≤30), + 360 özellik "
                      "kolonu f{k}_{j} (k=0..9 pencere adımı, j=0..35 E0b özellik). "
                      "engine_id ve rul olmadan skor çalışır; 'rul' ile doğruluk ölçümü "
                      "yapılır (rul kolonu CSV'de yoksa ölçüm düşer)."),
        )

    def model_dosyasi(self) -> Path:
        return self.edge / "model.onnx"

    def model_meta(self) -> ModelMeta:
        _, _, _, _, window, dim, artefakt_sha, model_sha_manifest = self._yukle()
        with (self.edge / "manifest.json").open(encoding="utf-8") as f:
            manifest = json.load(f)
        return ModelMeta(
            ad="Maintenance E0b binary (backprop, real C-MAPSS FD001) — edge ONNX",
            esik=float(self.esik),
            esik_kaynak=self.esik_kaynak,
            olusturma=str(manifest.get("generated_utc", "")),
            ek_bilgi={
                "manifest": "maintenance/results/edge/manifest.json",
                "model_boyut": f"window={window} features={dim} classes=2 "
                               "(binary backprop edge paketi)",
                "paket_sha256_manifest": model_sha_manifest,
                "e0b_preprocessing sha256": artefakt_sha,
                "e0b edge checkpoint sha256": str(manifest["checkpoint"]["sha256"]),
                "rul_le_30 class": str(manifest["model"]["class_labels"]),
            },
        )

    def baseline(self, ds: VeriSeti) -> tuple[Baseline, ...]:
        """Kıyas referansları — maintenance/results/e0b_cmapss.json'a atıflıdır (AGENTS §13)."""
        return (
            Baseline("Edge paketi cloud/backprop — E0b test AUC-ROC (rul≤30)", "AUC-ROC",
                     0.885319401215355, f"{E0B_JSON}:229"),
            Baseline("E0b standart LR (baseline) — test AUC-ROC", "AUC-ROC",
                     0.9950348302949459, f"{E0B_JSON}:47"),
            Baseline("E0b standart GBT (baseline) — test AUC-ROC", "AUC-ROC",
                     0.99629464947384, f"{E0B_JSON}:105"),
        )

    def ek_metrikler(self, ds: VeriSeti, verdictler: list[Verdict]) -> dict[str, str]:
        """Binary rul≤30 ölçümü — girdi CSV'sinin rul etiketleri üzerinde.

        Etiket: rul ≤ 30 → critical(1). Metrik adı "bu koşunun ölçümü" ve "girdi CSV
        etiketleri" taşır; CSV'nin held-out olduğu kutudan bakarak doğrulanamaz.
        """
        çiftler: list[tuple[int, float]] = []
        for kayit, verdict in zip(ds.kayitlar, verdictler):
            try:
                rul = int(float(str(kayit.ham.get("rul")).strip()))
            except (TypeError, ValueError):
                continue
            çiftler.append((1 if rul <= 30 else 0, float(verdict.skor)))
        if not çiftler:
            return {}
        etiket = [e for e, _ in çiftler]
        skor = [s for _, s in çiftler]
        metrik = "rul≤30, girdi CSV etiketleri, bu koşunun ölçümü"
        bulgular: dict[str, str] = {}
        isabet = sum(1 for e, s in çiftler if (s >= self.esik) == bool(e)) / len(çiftler)
        bulgular[f"binary top-1 doğruluk ({metrik})"] = f"{isabet:.4f}"
        n_pos = sum(etiket)
        if n_pos:
            tp = sum(1 for e, s in çiftler if e == 1 and s >= self.esik)
            bulgular[f"rul_le_30 recall ({metrik})"] = f"{tp / n_pos:.4f}"
        if len(set(etiket)) > 1:
            try:
                from sklearn.metrics import roc_auc_score

                bulgular[f"AUC-ROC ({metrik})"] = f"{float(roc_auc_score(etiket, skor)):.4f}"
            except Exception:  # pragma: no cover — sklearn yoksa sessiz geç
                pass
        ortalama = sum(skor) / len(skor)
        bulgular["ortalama P(critical) (bu koşunun ölçümü)"] = f"{ortalama:.4f}"
        return bulgular

    # ------------------------------------------------------------- tahmin
    def predict(self, ds: VeriSeti) -> list[Verdict]:
        np, session, mu, sd, window, dim, _, _ = self._yukle()
        verdictler: list[Verdict] = []
        for kayit in ds.kayitlar:
            ham = kayit.ham
            vals = self._pencere_matrisi(ham, np, window, dim)
            scaled = ((vals - mu) / sd).astype(np.float32)
            out = self._onnx_kos(session, scaled[np.newaxis, :, :])
            p_kritik = float(out[0, 1])
            karar = "critical" if p_kritik >= self.esik else "ok"
            gerekceler = self._gerekceler(np, session, scaled, p_kritik)
            verdictler.append(Verdict(kimlik=kayit.kimlik, skor=p_kritik,
                                      karar=karar, gerekceler=gerekceler))
        return verdictler

    def _gerekceler(self, np, session, scaled, p_full):
        """Penzere adımı k sıfırlanınca (0 = standartlaştırılmış medyan) P değişimi."""
        delta: list[tuple[int, float]] = []
        for k in range(scaled.shape[0]):
            masked = scaled.copy()
            masked[k] = 0.0
            out = self._onnx_kos(session, masked[np.newaxis, :, :])
            delta.append((k, float(out[0, 1]) - p_full))
        delta.sort(key=lambda x: abs(x[1]), reverse=True)
        return [
            Gerekce(etiket=f"pencere adımı {k} maskelendi", delta_logit=float(d))
            for k, d in delta[: self.top_k]
        ]


ADAPTER = MaintenanceAdapter()
