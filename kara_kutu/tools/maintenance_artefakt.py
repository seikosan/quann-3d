"""E0b preprocessing artifacts — train-fold-only mu/sd + KMeans(24) serileştirici.

Neden: edge paketi (`maintenance/results/edge/manifest.json`) scaler'sız; manifest
`preprocessing_note` "scaler not included; raw sensor preprocessing is upstream"
diyor. Bu araç, E0b pipeline'ı (`maintenance/src/run_e0b_cmapss.py`) TRAIN fold'undan
fit ettiği mu/sd ve KMeans(24)'ı makinece okunabilir bir dosyada serileştirir.
Eval-only: eğitim/checkpoint yazmaz; test fold hiç okunmaz; gerçek FD001 kullanılır.

Çıktı: `maintenance/results/e0b_artifacts/e0b_preprocessing.json`
(mu/sd/KMeans merkezleri + config + kaynak künyesi; sha256'lar dosyanın içinde değil —
kanıt bloğu provenance.py ile hesaplanır).

Çalıştırma: `python3 -m kara_kutu.tools.maintenance_artefakt`
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_KOK = Path(__file__).resolve().parents[2]
MAINT_SRC = REPO_KOK / "maintenance" / "src"
if str(MAINT_SRC) not in sys.path:
    sys.path.insert(0, str(MAINT_SRC))
from config import SEED, FEATURE_COUNT, WINDOW_SIZE  # noqa: E402

SEED_INT = int(SEED)
K = 24  # E0b: MiniBatchKMeans(n_clusters=24, random_state=SEED, n_init=2, ...)
N_CLUSTERS = K
EDGE_ARTIFACTS = REPO_KOK / "maintenance" / "results" / "e0b_artifacts"
SOURCE_DIR = REPO_KOK / "maintenance" / "data" / "cmapss"
OUT_PATH = EDGE_ARTIFACTS / "e0b_preprocessing.json"


def _sha256(yol: Path) -> str:
    h = hashlib.sha256()
    with yol.open("rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def _train_fit() -> dict:
    """E0b train-fold: 80/20 engine-disjoint; windows stride 12; fit mu/sd + KMeans."""
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    from run_e0b_cmapss import _windows  # noqa: PLC0415 — E0b desenini birebir kullan
    # E0b (run_e0b_cmapss.py:183-204): engine 80/20 split; np.random.seed + torch
    # sıfırdan deterministik. Sadece numpy/sklearn'e dokunur, torch import edilmez.
    from loader_cmapss import load_cmapss  # noqa: PLC0415

    loaded = load_cmapss(SOURCE_DIR)
    train_data = loaded["train"]
    engines = np.unique(train_data["engine_ids"])
    rng = np.random.default_rng(SEED_INT)
    rng.shuffle(engines)
    cut = max(1, int(len(engines) * 0.8))
    train_engines = set(map(int, engines[:cut]))
    train_cloud, train_rul, train_groups = _windows(train_data, train_engines)

    mu = train_cloud.reshape(-1, FEATURE_COUNT).mean(axis=0).astype(np.float64)
    sd = train_cloud.reshape(-1, FEATURE_COUNT).std(axis=0).astype(np.float64)
    sd[sd < 1e-6] = 1.0

    cluster = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED_INT,
                              n_init=2, batch_size=512)
    cluster.fit(train_cloud.reshape(-1, FEATURE_COUNT))

    train_raw = sorted(p for p in SOURCE_DIR.rglob("train_FD001.txt"))
    test_raw = sorted(p for p in SOURCE_DIR.rglob("test_FD001.txt"))
    return {
        "seed": SEED_INT,
        "feature_dim": FEATURE_COUNT,
        "window": WINDOW_SIZE,
        "stride": 12,  # E0b STRIDE (run_e0b_cmapss.py:20)
        "n_clusters": int(cluster.n_clusters),
        "n_train_engines": int(len(train_engines)),
        "n_train_windows_used": int(len(train_cloud)),
        "mu": [float(x) for x in mu],
        "sd": [float(x) for x in sd],
        "kmeans_centers": [[float(x) for x in c] for c in cluster.cluster_centers_],
        "source": {
            "flow": "maintenance/src/run_e0b_cmapss.py:183-204 (E0b real FD001, engine-disjoint 80/20)",
            "train_file_sha256": _sha256(train_raw[0]) if train_raw else "DOGRULANMADI",
            "test_file_sha256": _sha256(test_raw[0]) if test_raw else "DOGRULANMADI",
            "official_test_note": "test fold hiç okunmadı; split yalnız train engine'lar üzerinden",
        },
        "not": "Eval-only re-emit: scaler+kmeans E0b flow'undan üretildi; synthetic E0/fallback kullanılmaz.",
    }


def main() -> int:
    art = _train_fit()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(art, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sha = _sha256(OUT_PATH)
    print(f"yazıldı: {OUT_PATH}")
    print(f"train_windows={art['n_train_windows_used']} mu={art['feature_dim']} sd={art['feature_dim']} "
          f"clusters={art['n_clusters']}")
    print(f"sha256: {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
