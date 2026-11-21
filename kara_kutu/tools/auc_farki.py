"""AUC farkı teşhisi — kara kutu ONNX'unu IBM doğal held-out örneklemi üzerinde ölçer.

Soru (DEVIR 21 Kas kapanışı): aynı model neden User0 paketinde AUC-ROC 0,8346
(`kara_kutu/runs/.../ic_rapor.md`), deploy referansında ise INT8-channel 0,9367
(`fraud/logs/run_full_scale_edge_eval.log:54`) ölçtü?

Yöntem (uydurma yok, iki ölçüm):
1. IBM tam CSV'den (config.BIG_CSV) kullanıcı-fold'ları yeniden hesaplanır
   (`fraud/src/train_quann.py:58` user_folds, seed `fraud/src/config.py:55`);
   User0 paketinin tek kullanıcısı (User==0) hangi bölmede — eğitim mi held-out mu?
2. Gerçek held-out kullanıcılardan (fold>=8) doğal oranlı örneklem çekilir;
   deploy ONNX'u (kara kutu paketinin kendisi, ablation'sız skor yolu — parity
   `kara_kutu/tests/test_parity.py` bu yol için kanıtlı) ile AUC-ROC ölçülür.
3. Hanley–McNeil 95% CI: n1=27 pozitifli User0 ölçümünün istatistiksel marjı.

Koşum: python3 -m kara_kutu.tools.auc_farki [--n-ornek 400000] [--hizli]
Çıktı: kara_kutu/runs/auc_farki_<ts>/ozet.json (runs/ gitignore'da; sayısal
özet stdout'a basılır — kalıcı kanıt docs notuna işlenir).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fraud" / "src"))  # fraud/src kendi-kök import kalıbı

from kara_kutu.adapters.fraud import IBM_KOLONLARI, FraudAdapter  # noqa: E402

CHUNK = 1_000_000
SEED = 42  # örnekleme tohumu (raporda açıkça taşınır)


def _kaydir() -> Path:
    from datetime import datetime, timezone

    yol = REPO / "kara_kutu" / "runs" / f"auc_farki_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    yol.mkdir(parents=True, exist_ok=True)
    return yol


def _mcnemil_ci(auc: float, n1: int, n2: int) -> tuple[float, float]:
    """Hanley–McNeil (1982) AUC için %95 güven aralığı."""
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)) / (
        n1 * n2)
    se = math.sqrt(var)
    return auc - 1.96 * se, auc + 1.96 * se


def _foldlar(big_csv: Path) -> tuple[np.ndarray, int, int]:
    """(fold_of_group, n_groups, te_satir_sayisi) — iki geçişli, bellek-dostu."""
    import pandas as pd

    import config
    import datasets
    from train_quann import user_folds

    gruplar: set[int] = set()
    for parc in pd.read_csv(big_csv, usecols=[config.GROUP_COL], chunksize=CHUNK):
        gruplar.update(int(g) for g in parc[config.GROUP_COL].unique())
    uniq = np.array(sorted(gruplar))
    if not np.array_equal(uniq, np.arange(len(uniq))):
        raise RuntimeError("User id'leri 0..N-1 değil; fold eşlemi yeniden düşünülmeli")
    fold = user_folds(len(uniq))
    fold_of = {int(u): int(f) for u, f in zip(uniq, fold)}
    te = {u for u, f in fold_of.items() if f >= 8}
    te_mask = np.isin(uniq, list(te))
    n_te_satir = 0
    y_te = 0
    for parc in pd.read_csv(
        big_csv, usecols=[config.GROUP_COL, config.LABEL_COL], chunksize=CHUNK
    ):
        g = parc[config.GROUP_COL].to_numpy()
        keep = te_mask[np.searchsorted(uniq, g)]
        n_te_satir += int(keep.sum())
        y_te += int(datasets.get_label(parc)[keep].sum())
    print(f"[geçiş] n_groups={len(uniq)} user0_fold={fold_of[0]} "
          f"te_kullanici={len(te)} te_satir={n_te_satir} te_fraud={y_te}", flush=True)
    return fold_of, len(uniq), n_te_satir


def _ornekle(big_csv: Path, fold: dict[int, int], n_ornek: int, hizli: bool):
    """Held-out (fold>=8) kullanıcılardan doğal oranlı birleşik tesadüfi örneklem."""
    import pandas as pd

    import config
    import datasets

    fold_dizi = np.fromiter((fold[u] for u in range(len(fold))), dtype=np.int8,
                            count=len(fold))
    rng = np.random.default_rng(SEED)
    kolonlar = list(IBM_KOLONLARI) + [config.GROUP_COL, config.LABEL_COL]
    parcalar: list = []
    toplam = 0
    p = min(1.0, 4.0 * n_ornek / 6_000_000)  # te-alanını bolca aşan üst sınır, sonra kesilir
    for i, parc in enumerate(pd.read_csv(big_csv, usecols=kolonlar, chunksize=CHUNK)):
        if hizli and i >= 8:
            break
        g = parc[config.GROUP_COL].to_numpy()
        te = fold_dizi[g] >= 8
        al = te & (rng.random(len(parc)) < p)
        if al.any():
            parcalar.append(parc.loc[al])
            toplam += int(al.sum())
        if toplam >= n_ornek:
            break
    df = pd.concat(parcalar, ignore_index=True)
    if len(df) > n_ornek:
        df = df.iloc[rng.choice(len(df), n_ornek, replace=False)]
    y = datasets.get_label(df)
    df = df.drop(columns=[config.LABEL_COL, config.GROUP_COL])
    print(f"[örneklem] satır={len(df)} fraud={int(y.sum())} "
          f"base_rate={y.mean():.6f}", flush=True)
    return df, y


def _auc_olc(df, y, esik: float) -> dict[str, float | int]:
    """Kara kutu paketinin skor yolu: quantizer.json + model.onnx (ablation'sız)."""
    adapter = FraudAdapter()
    np_, _pd, fq, _dec, session, _man = adapter._yukle()
    tokenlar = fq.transform(df)
    probs = []
    t0 = time.time()
    for start in range(0, len(tokenlar), 8192):
        tok = tokenlar[start:start + 8192]
        mask = (tok != 0) & (tok != 1)
        probs.append(session.run(None, {"ids": tok, "mask": mask})[0].reshape(-1))
    p = np.concatenate(probs)
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, p))
    n1, n2 = int(y.sum()), int((y == 0).sum())
    lo, hi = _mcnemil_ci(auc, n1, n2)
    return {
        "auc": round(auc, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "n": len(y), "fraud": n1,
        "fraud_esik_ustu": int((p[y == 1] >= esik).sum()),
        "alarm_sayisi": int((p >= esik).sum()),
        "maks_p": round(float(p.max()), 6),
        "sure_sn": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-ornek", type=int, default=400_000)
    ap.add_argument("--hizli", action="store_true",
                    help="ilk ~8M satırla sınırlı örneklem (geliştirme kontrolü)")
    a = ap.parse_args()

    import config

    big = Path(config.BIG_CSV)
    if not big.is_file():
        print(f"IBM CSV yok ({big}) — teşhiş koşulamaz.", file=sys.stderr)
        return 2

    yonetmen = _kaydir()
    fold, n_grup, te_satir = _foldlar(big)
    df, y = _ornekle(big, fold, a.n_ornek, a.hizli)
    sonuc = _auc_olc(df, y, esik=0.9969)
    u0_lo, u0_hi = _mcnemil_ci(0.8346, 27, 19_936)
    ozet = {
        "us": SEED, "n_grup": n_grup, "user0_fold": fold[0],
        "te_kullanici_fold_esigi": 8, "te_satir_tam": te_satir,
        "heldout_ornek": sonuc,
        "user0_auc_ci95_hm": [round(u0_lo, 6), round(u0_hi, 6)],
        "referans": {
            "int8_channel_natural": 0.9367,
            "kaynak": "fraud/logs/run_full_scale_edge_eval.log:54",
        },
    }
    (yonetmen / "ozet.json").write_text(json.dumps(ozet, indent=2, ensure_ascii=False))
    print(json.dumps(ozet, indent=2, ensure_ascii=False), flush=True)
    print(f"kayıt: {yonetmen/'ozet.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
