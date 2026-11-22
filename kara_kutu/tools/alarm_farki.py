"""Alarm farkı teşhisi — kutu (FP32 ONNX) neden 400k held-out satırında 0 alarm verdi?

Çelişki (21-22 Kas ölçümleri): kara kutu ONNX'u 400k doğal held-out satırında
eşik 0,9969'da 0 alarm, maks p=0,989369 (kara_kutu/runs/auc_farki_*/ozet.json);
FP32 denetimi aynı modelin te popülasyonunda (4,7M) 11.784 alarm sayıyor
(fraud/results/false_alarm_audit.json "population.flagged").

Önceki yanılgı düzeltmesi: paket INT8 DEĞİL — export_onnx `DeployPath`'i (FP32)
dışa aktarır (fraud/src/export_deploy.py:47-53); INT8 sayıları edge_eval'in
torch simülasyonudur (edge_eval.py:12 "FX dynamic quant"). Kutu için doğru
referans FP32 AUC 0,9451 (fraud/logs/run_full_scale_edge_eval.log:52).

İki faz, iki ayrı soru (uydurma yok, ölçüm):
1. A/B (sadık sadakat): auc_farki'nin AYNI 400k örneklemi (seed 42) tek token
   matrisiyle hem FP32 torch hem kutu ONNX ile skorlanır → eşik çevresinde de
   kutu ≡ FP32 mi (parity kanıtı User0 p≤0,988 ile sınırlıydı, manifest.json)?
2. Bölge taraması: TEMA te satırları (fold>=8, ~4,9M) kutuyla skorlanır; alarm
   sayıları CSV bölgelarına göre raporlanır → 11.784 alarm nerede yoğunlaşıyor,
   ilk ~8M (örneklem bölgesi) neden boş?

Koşum: python3 -m kara_kutu.tools.alarm_farki [--n-ornek 400000] [--tarama-atla]
Çıktı: kara_kutu/runs/alarm_farki_<ts>/ozet.json + stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fraud" / "src"))  # fraud/src kendi-kök import kalıbı

from kara_kutu.adapters.fraud import ESIK_EKIM, IBM_KOLONLARI, FraudAdapter  # noqa: E402
from kara_kutu.tools.auc_farki import CHUNK, _ornekle  # noqa: E402  (örneklem sadakati)

BOLGE = 4_000_000  # CSV satır bölgesi genişliği (raporlama)


def _kaydir() -> Path:
    yol = REPO / "kara_kutu" / "runs" / f"alarm_farki_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    yol.mkdir(parents=True, exist_ok=True)
    return yol


def _fold_harita(big_csv: Path) -> tuple[dict[int, int], np.ndarray, int]:
    """(fold_of_user, uniq_users, n_users) — tek geçiş, fold = user_folds (train_quann.py:58)."""
    import pandas as pd

    import config
    from train_quann import user_folds

    gruplar: set[int] = set()
    for parc in pd.read_csv(big_csv, usecols=[config.GROUP_COL], chunksize=CHUNK):
        gruplar.update(int(g) for g in parc[config.GROUP_COL].unique())
    uniq = np.array(sorted(gruplar))
    if not np.array_equal(uniq, np.arange(len(uniq))):
        raise RuntimeError("User id'leri 0..N-1 değil; fold eşlemi yeniden düşünülmeli")
    fold_of = {int(u): int(f) for u, f in zip(uniq, user_folds(len(uniq)))}
    return fold_of, uniq, len(uniq)


def _esik_istat(p: np.ndarray, y: np.ndarray, esik: float) -> dict:
    """Eşikte alarm/TP/FP/duyarlılık/hassasiyet + maks p."""
    alarm = p >= esik
    tp = int((alarm & (y == 1)).sum())
    fp = int((alarm & (y == 0)).sum())
    n1 = int(y.sum())
    return {
        "alarm": int(alarm.sum()), "tp": tp, "fp": fp,
        "duyarli": round(tp / n1, 4) if n1 else None,
        "hassasiyet": round(tp / (tp + fp), 4) if tp + fp else None,
        "maks_p": round(float(p.max()), 6),
        "p_0_99_ustu": int((p >= 0.99).sum()),
    }


def _ab_ayni_satir(df, y, ckpt: Path) -> dict:
    """Faz 1: aynı tok → FP32 torch vs kutu ONNX; parity + eşik istatistikleri."""
    import torch
    from sklearn.metrics import roc_auc_score

    from qcloud_fraud import QCloudFraudClassifier
    from quantize import FeatureQuantizer

    raw = torch.load(ckpt, map_location="cpu")
    model = QCloudFraudClassifier.load_quann(str(ckpt), torch.device("cpu"))
    fq = FeatureQuantizer.from_dict(raw["quantizer"])
    tok = fq.transform(df)
    t = torch.tensor(tok, dtype=torch.long)
    with torch.no_grad():
        p32 = model.predict_proba(t, use_quann_points=True, batch=8192).cpu().numpy().reshape(-1)

    adapter = FraudAdapter()
    _np, _pd, fq_box, _dec, session, _man = adapter._yukle()
    tok_box = fq_box.transform(df)
    assert np.array_equal(tok, tok_box), "bundled quantizer ≠ deploy quantizer.json"
    probs = []
    for start in range(0, len(tok_box), 8192):
        tk = tok_box[start:start + 8192]
        probs.append(session.run(None, {"ids": tk, "mask": (tk != 0) & (tk != 1)})[0].reshape(-1))
    pbox = np.concatenate(probs)

    diff = np.abs(p32 - pbox)
    return {
        "n": int(len(y)), "fraud": int(y.sum()),
        "parity_max_mutlak_fark": float(diff.max()),
        "parity_fark_esik_cevresi": float(diff[p32 >= 0.9].max()) if (p32 >= 0.9).any() else 0.0,
        "auc_fp32": round(float(roc_auc_score(y, p32)), 6),
        "auc_kutu": round(float(roc_auc_score(y, pbox)), 6),
        "fp32_esikte": _esik_istat(p32, y, ESIK_EKIM),
        "kutu_esikte": _esik_istat(pbox, y, ESIK_EKIM),
    }


def _tam_tarama(big_csv: Path, fold_of: dict[int, int], satir_siniri: int | None = None) -> dict:
    """Faz 2: tüm te satırlarını kutuyla skorla; alarm yoğunluğunu bölge bazında ver."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    import config
    import datasets

    fold_dizi = np.fromiter((fold_of[u] for u in range(len(fold_of))), dtype=np.int8,
                            count=len(fold_of))
    adapter = FraudAdapter()
    _np, _pd, fq, _dec, session, _man = adapter._yukle()
    kolonlar = list(IBM_KOLONLARI) + [config.GROUP_COL, config.LABEL_COL]

    bolgeler: dict[int, dict] = {}
    tum_p: list[np.ndarray] = []
    tum_y: list[np.ndarray] = []
    pos = 0
    t0 = time.time()
    for parc in pd.read_csv(big_csv, usecols=kolonlar, chunksize=CHUNK):
        g = parc[config.GROUP_COL].to_numpy()
        te = fold_dizi[g] >= 8
        if te.any():
            alt = parc.loc[te].drop(columns=[config.GROUP_COL])
            yy = np.asarray(datasets.get_label(alt), dtype=np.int8)
            alt = alt.drop(columns=[config.LABEL_COL])
            tk = fq.transform(alt)
            pp = []
            for start in range(0, len(tk), 8192):
                k = tk[start:start + 8192]
                pp.append(session.run(None, {"ids": k, "mask": (k != 0) & (k != 1)})[0].reshape(-1))
            p = np.concatenate(pp)
            # parc içindeki konumlar → global CSV satırı → bölge
            yer = np.flatnonzero(te) + pos
            b = yer // BOLGE
            for bi in np.unique(b):
                al = b == bi
                s = bolgeset(bolgeler, int(bi * BOLGE))
                s["te"] += int(al.sum())
                s["fraud"] += int(yy[al].sum())
                s["alarm"] += int((p[al] >= ESIK_EKIM).sum())
                s["tp"] += int(((p[al] >= ESIK_EKIM) & (yy[al] == 1)).sum())
                s["maks_p"] = max(s["maks_p"], float(p[al].max()))
            tum_p.append(p.astype(np.float32))
            tum_y.append(yy)
        pos += len(parc)
        print(f"  [tarama] {pos/1e6:.0f}M satır okundu, te={sum(s['te'] for s in bolgeler.values())}"
              f" alarm={sum(s['alarm'] for s in bolgeler.values())}", flush=True)
        if satir_siniri and pos >= satir_siniri:
            print(f"  [tarama] satır sınırı {satir_siniri:,} — erken bitiş (duman modu)", flush=True)
            break

    p_all = np.concatenate(tum_p)
    y_all = np.concatenate(tum_y)
    toplam = _esik_istat(p_all.astype(np.float64), y_all, ESIK_EKIM)
    toplam["te_skorlanan"] = int(len(y_all))
    toplam["auc_kutu_tum_te"] = round(float(roc_auc_score(y_all, p_all.astype(np.float64))), 6)
    return {"bolgeler": [bolgeset(bolgeler, k) | {"bolge": k} for k in sorted(bolgeler)],
            "genel": toplam, "sure_sn": round(time.time() - t0, 1)}


def bolgeset(bolgeler: dict[int, dict], bolge: int) -> dict:
    return bolgeler.setdefault(bolge, {"te": 0, "fraud": 0, "alarm": 0, "tp": 0, "maks_p": 0.0})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-ornek", type=int, default=400_000)
    ap.add_argument("--tarama-atla", action="store_true", help="yalnız faz 1 (A/B)")
    ap.add_argument("--tarama-sinir", type=int, default=None,
                    help="faz 2 taramasını ilk N CSV satırıyla sınırla (duman testi)")
    a = ap.parse_args()

    import config

    big = Path(config.BIG_CSV)
    if not big.is_file():
        print(f"IBM CSV yok ({big})", file=sys.stderr)
        return 2
    ckpt = Path(config.RESULTS_DIR) / "quann_full_hybrid.pt"
    if not ckpt.is_file():
        print(f"checkpoint yok ({ckpt})", file=sys.stderr)
        return 2

    yonetmen = _kaydir()
    fold_of, uniq, n_user = _fold_harita(big)
    print(f"[fold] n_user={n_user} user0_fold={fold_of[0]}", flush=True)
    df, y = _ornekle(big, fold_of, a.n_ornek, hizli=False)
    ozet: dict = {"faz1_ab": _ab_ayni_satir(df, y, ckpt)}
    if not a.tarama_atla:
        ozet["faz2_tarama"] = _tam_tarama(big, fold_of, a.tarama_sinir)
    ozet["referans"] = {
        "fp32_auc": 0.9451, "fp32_kaynak": "fraud/logs/run_full_scale_edge_eval.log:52",
        "audit_alarm": 11784, "audit_kaynak": "fraud/results/false_alarm_audit.json:population.flagged",
        "audit_esik": 0.9969438910484314,
    }
    (yonetmen / "ozet.json").write_text(json.dumps(ozet, indent=2, ensure_ascii=False))
    print(json.dumps(ozet, indent=2, ensure_ascii=False), flush=True)
    print(f"kayıt: {yonetmen/'ozet.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
