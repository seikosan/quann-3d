"""Karar deneyi: 19 EKİM kodu (5f66aed worktree) ile aynı satırları skorla.

Koşturma: `git worktree add <tmp> 5f66aed` sonrası dosyayı worktree köküne
kopyala ve `python3 -u score_oct.py` koş (CKPT/BIG mutlak yolları kendi
makinen göre ayarla). Beklenen (2026-11-21 ölçümü): te=234,686 fraud=197
maks_p=1.0 alarm@0.99694=693 AUC=0.95971 — kutu aynı satırlarda
maks_p=0.990873 alarm=0 verir (kara_kutu/runs/alarm_farki_20261121-232335/ozet.json:33-48 — 2026-11-23 kazara silindi; DOĞRULANMADI, kalıntı).
Kutu ≠ Ekim fonksiyonu: mimari portu cloudlm/model.py:64,103-106.

Kutu duman taraması (kara_kutu, 2026-11-21) ilk 1M CSV satırında
te=234,686 fraud=197 maks_p=0.9908732175827026 alarm=0 vermişti
(kara_kutu/runs/alarm_farki_20261121-232335/ozet.json faz2.bolgeler[0] — 2026-11-23 kazara silindi; DOĞRULANMADI, kalıntı).
Bu script AYNI satır kümesini 19 Ekim'in QCloudFraudClassifier + checkpoint
bundled quantizer'ıyla skorlar; fark çıkarsa mimari-port kaynaklıdır.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fraud" / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

_orig_load = torch.load


def _load(*a, **k):  # torch>=2.6 weights_only varsayılanı — eski ckpt için gevşet
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)


torch.load = _load

import config  # noqa: E402
import datasets  # noqa: E402
import quantize  # noqa: E402
from train_quann import user_folds  # noqa: E402
from qcloud_fraud import QCloudFraudClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

BIG = "/Users/serkankose/Downloads/archive/credit_card_transactions-ibm_v2.csv"
CKPT = "/Users/serkankose/Documents/quann-3d/fraud/results/quann_full_hybrid.pt"  # results/ gitignored — worktree'de yok
THR = 0.9969438910484314  # audit_esik (false_alarm_audit.json population)
SATIR = 1_000_000  # kutu duman taramasıyla aynı kesit
USECOLS = ["User", "Is Fraud?", "Amount", "Time", "Month", "Day", "Year",
           "Zip", "Use Chip", "MCC", "Merchant City", "Merchant Name",
           "Merchant State", "Errors?"]  # 5f66aed audit_false_alarms.py:25-27 ile birebir


def main() -> None:
    ck = torch.load(CKPT, map_location="cpu")
    fq = quantize.FeatureQuantizer.from_dict(ck["quantizer"])
    model = QCloudFraudClassifier.load_quann(CKPT, torch.device("cpu"))

    # kullanıcı→fold eşlemesi (audit'nin user_folds'u, seed config.SEED)
    n_user = 2000  # ölçüldü: 24.4M satırda User ∈ [0,1999] (auc_farki ozet n_groups=2000)
    foldler = user_folds(n_user, n_splits=10)
    fold_dizi = np.asarray([foldler[int(u)] for u in range(n_user)], dtype=np.int8)
    print(f"[fold] n_user={n_user} user0_fold={int(fold_dizi[0])}", flush=True)

    toklar, yler = [], []
    pos = 0
    t0 = time.time()
    for parc in pd.read_csv(BIG, usecols=USECOLS, chunksize=250_000):
        g = parc[config.GROUP_COL].to_numpy()
        te = fold_dizi[g] >= 8
        if te.any():
            alt = parc.loc[te]
            yler.append(datasets.get_label(alt).astype(np.int8))
            toklar.append(np.asarray(fq.transform(alt)))
        pos += len(parc)
        if pos >= SATIR:
            break
    tok = np.concatenate(toklar)
    y = np.concatenate(yler)
    print(f"[veri] te={len(y):,} fraud={int(y.sum())} ({time.time()-t0:.0f}s)", flush=True)

    t = torch.tensor(tok, dtype=torch.long)
    p = model.predict_proba(t, use_quann_points=True, batch=8192).cpu().numpy()
    auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    print({
        "kaynak": "19 EKİM kodu (5f66aed) + ckpt bundled quantizer",
        "n_te": int(len(y)), "fraud": int(y.sum()),
        "maks_p": float(np.max(p)),
        "alarm_thr": int((p >= THR).sum()),
        "p_0_99_ustu": int((p >= 0.99).sum()),
        "auc": round(auc, 6),
    }, flush=True)


if __name__ == "__main__":
    main()
