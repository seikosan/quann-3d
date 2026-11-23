"""Gerçek C-MAPSS FD001 üzerindebinary backprop checkpoint'i eğitir ve kaydeder.

E0b akışını (run_e0b_cmapss.run) birebir yansıtır: aynı seed, aynı
engine-disjoint 80/20 split'i, aynı pencere + standartlaştırma + KMeans
(token seti) seti, aynı eğitim döngüsü. Tek fark: binary backprop regime'ı
öĞITİLİR VE checkpoint olarak maintenance/results/quann_binary_backprop.pt
altına yazılır (E0b bu checkpoint'i hiç yazmaz; edge paketi synthetic-trained
checkpoint'ten export ediliyordu).

Bu, maintenance'nin gerçek veriyle yeniden eğitimini yapan araçtır — eval-only
DEĞİL. CLI/adaptör etkilenmez; köklü checkpoint maintenance/results/
quann_binary_backprop.pt'ye yazılır (eski synthetic checkpoint backup olarak
/tmp/quann_binary_backprop_synthetic.pt.bak).

Çıktı: saklanan checkpoint + e0b_cmapss.json ile karşılaştırılan val/test
metrikleri (stdout).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from loader_cmapss import load_cmapss
from run_e0b_cmapss import _windows, _scale, _labels, _fit_cloud, SEED, WINDOW, FEATURE_DIM

OUT_CHECKPOINT = ROOT / "maintenance" / "results" / "quann_binary_backprop.pt"


def main() -> int:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    loaded = load_cmapss(ROOT / "maintenance" / "data" / "cmapss")
    train_data = loaded["train"]
    official_test = loaded["test"]

    engines = np.unique(train_data["engine_ids"])
    rng = np.random.default_rng(SEED)
    rng.shuffle(engines)
    cut = max(1, int(len(engines) * 0.8))
    train_engines = set(map(int, engines[:cut]))
    val_engines = set(map(int, engines[cut:]))

    train_cloud, train_rul, _ = _windows(train_data, train_engines)
    val_cloud, val_rul, _ = _windows(train_data, val_engines)
    test_cloud, test_rul, _ = _windows(official_test)
    train_cloud, val_cloud, test_cloud = _scale(train_cloud, val_cloud, test_cloud)

    points_train = train_cloud.reshape(-1, FEATURE_DIM)
    cluster = MiniBatchKMeans(n_clusters=24, random_state=SEED, n_init=2, batch_size=512)
    cluster.fit(points_train)
    train_tokens = cluster.predict(points_train).reshape(-1, WINDOW)
    val_tokens = cluster.predict(val_cloud.reshape(-1, FEATURE_DIM)).reshape(-1, WINDOW)
    test_tokens = cluster.predict(test_cloud.reshape(-1, FEATURE_DIM)).reshape(-1, WINDOW)

    y_train = _labels(train_rul, "binary")
    y_val = _labels(val_rul, "binary")
    y_test = _labels(test_rul, "binary")

    print(f"train={len(train_cloud)} val={len(val_cloud)} test={len(test_cloud)} "
          f"rul<=30: train={int(y_train.sum())} val={int(y_val.sum())} test={int(y_test.sum())}")

    fitted, model = _fit_cloud("backprop", train_cloud, train_tokens, y_train,
                               val_cloud, val_tokens, y_val, test_cloud, test_tokens, y_test,
                               2, "binary", ROOT / "maintenance" / "logs" / "run_e0b_cmapss.log")

    OUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "token_weights": model.token_weights,
                "config": {"D": FEATURE_DIM, "n_classes": 2}}, OUT_CHECKPOINT)

    print("val metrics:", json.dumps(fitted, indent=2))

    # Doğrulama: e0b_cmapss.json ile aynı E0b binary backprop kayıdı.
    e0b = json.loads((ROOT / "maintenance" / "results" / "e0b_cmapss.json").read_text())
    e0b_val = e0b["tasks"]["binary"]["cloud"]["backprop"]["val"]
    e0b_test = e0b["tasks"]["binary"]["backprop"]["test"]
    checks = {
        "val_auc_roc": (fitted["auc_roc"], e0b_val["auc_roc"]),
        "val_threshold": (fitted["threshold"], e0b_val["threshold"]),
        "val_auc_pr": (fitted["auc_pr"], e0b_val["auc_pr"]),
        "test_auc_roc": (fitted["auc_roc"] if "auc_roc" in fitted else None, e0b_test["auc_roc"]),
    }
    print("\nkarşılaştırma (eğitilen ↔ e0b_cmapss.json binary/cloud/backprop):")
    for name, (new, ref) in checks.items():
        if new is None:
            continue
        print(f"  {name}: yeni={new:.6f} ↔ e0b={ref:.6f} (fark {new - ref:+.6f})")
    print(f"\ncheckpoint yazıldı: {OUT_CHECKPOINT}")
    print("model_state bytes:", (OUT_CHECKPOINT).stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
