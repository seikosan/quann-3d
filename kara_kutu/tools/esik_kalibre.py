"""Kutu-yerel eşik kalibrasyonu — Karar A (kullanıcı, 2026-11-22).

Soru (KARA_KUTU_PLAN §9): Ekim eşiği 0,9969 FP32 (port-öncesi) olasılıklarıyla
kalibre edildi; portlanmış kutu tam te'de maks p=0,993219 veriyor
(`kara_kutu/runs/alarm_farki_20261121-232405/ozet.json` genel.maks_p) → Ekim
eşiği kutuda 0 alarm üretiyor.

Yöntem (uydurma yok — Ekim'in KURALI kutu ölçeğinde yeniden koşulur):
1. Ekim işletim noktası "best-F2" eşiğiydi: 4,7M testte 1.842 TP / 9.942 FP
   (`docs/paper_draft.md:351`). Sayı kopyalanmaz, KURAL kopyalanır.
2. Kalibrasyon YALNIZ eğitim bölmesinde: kullanıcı-fold < 8 (`fraud/src/
   train_quann.py:58` user_folds); AGENTS §4.1 — test yalnız final değerlendirme.
   Örneklem doğal oranlı, tohum 42.
3. Skor = kutunun kendi ONNX yolu (ablation'sız; parity: `kara_kutu/tests/
   test_parity.py`). F2 = 5PR/(4P+R); tie grupları eşik sınır-değeriyle alınır.
4. Kanıt: runs/esik_kalibre_<ts>/ozet.json + `kara_kutu/artifacts/esik_fraud.json`
   (adapter eşikleri bu artefaktı okur — kara_kutu/adapters/fraud.py).
5. te (fold ≥ 8) kalibrasyonda hiç okunmaz. Tek seferlik final işletim noktası:
   `--te-dogrula` (artefakt varsa; ölçüm final rapora işlenir, tekrarlanmaz).

KARAR A1 (kullanıcı, 2026-11-22): eğitim-fold best-F2 tanısı iç-örnek bozuk
(AUC 0,941453 aynı ama precision 0,054 — bkz. §9); FİNAL eşik = Ekim ALARM
BÜTÇESİNİN quantili (`_butce()`: flagged/n, false_alarm_audit.json) — tek te
geçişinde hem eşik hem işletim sayıları ölçülür; best-F2 adayı artefakta
`kalibrasyon_tani` (REDDEDİLDİ) olarak işlenir.

Koşum:
    python3 -m kara_kutu.tools.esik_kalibre [--n-kal 1000000] [--hizli]
    python3 -m kara_kutu.tools.esik_kalibre --te-dogrula
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
ARTEFACT_VARSAYILAN = REPO / "kara_kutu" / "artifacts" / "esik_fraud.json"
BUTCE_DOSYA = REPO / "fraud" / "results" / "false_alarm_audit.json"
CHUNK = 1_000_000
SEED = 42  # kalibrasyon örneklem tohumu (raporda açıkça taşınır)
KURAL = "best-F2"
KURAL_KAYNAK = "docs/paper_draft.md:351 (Ekim işletim noktası kuralı)"
KURAL_A1 = "alarm-budgeti-esitleme"


def _butce() -> tuple[int, int, float]:
    """Ekim alarm bütçesi (canlı kanıt): population.flagged / population.n —
    `fraud/results/false_alarm_audit.json`. Sayı kopyalanmaz, dosyadan okunur."""
    pop = json.loads(BUTCE_DOSYA.read_text(encoding="utf-8"))["population"]
    alarm, n = int(pop["flagged"]), int(pop["n"])
    return alarm, n, alarm / n


def _kaydir(ad: str) -> Path:
    yol = REPO / "kara_kutu" / "runs" / f"esik_{ad}_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    yol.mkdir(parents=True, exist_ok=True)
    return yol


def _foldlar():
    """user→fold eşlemi (desen: kara_kutu/tools/auc_farki.py:59-88, tek geçiş)."""
    import pandas as pd

    sys.path.insert(0, str(REPO / "fraud" / "src"))
    import config
    from train_quann import user_folds

    gruplar: set[int] = set()
    for parc in pd.read_csv(Path(config.BIG_CSV), usecols=[config.GROUP_COL],
                            chunksize=CHUNK):
        gruplar.update(int(g) for g in parc[config.GROUP_COL].unique())
    uniq = np.array(sorted(gruplar))
    if not np.array_equal(uniq, np.arange(len(uniq))):
        raise RuntimeError("User id'leri 0..N-1 değil; fold eşlemi yeniden düşünülmeli")
    fold = user_folds(len(uniq))
    return config, {int(u): int(f) for u, f in zip(uniq, fold)}


def _ornekle(config, fold: dict[int, int], n: int, hizli: bool, *, kal: bool,
             p_ornek: float):
    """fold<8 (kal) ya fold>=8 (te) kullanıcılarından TAM DOSYA taramasıyla
    doğal oranlı örneklem — başa-yığılma zaman yanlılığını önlemek için break
    yok; hedefi aşarsa rastgele kesilir (tohum 42)."""
    import pandas as pd

    import datasets

    from kara_kutu.adapters.fraud import IBM_KOLONLARI

    fold_dizi = np.fromiter((fold[u] for u in range(len(fold))), dtype=np.int8,
                            count=len(fold))
    rng = np.random.default_rng(SEED)
    kolonlar = list(IBM_KOLONLARI) + [config.GROUP_COL, config.LABEL_COL]
    parcalar: list = []
    taranan = 0
    for i, parc in enumerate(pd.read_csv(Path(config.BIG_CSV), usecols=kolonlar,
                                         chunksize=CHUNK)):
        if hizli and i >= 8:
            break
        taranan += len(parc)
        g = parc[config.GROUP_COL].to_numpy()
        kat = (fold_dizi[g] < 8) if kal else (fold_dizi[g] >= 8)
        al = kat & (rng.random(len(parc)) < p_ornek)
        if al.any():
            parcalar.append(parc.loc[al])
    df = pd.concat(parcalar, ignore_index=True)
    if len(df) > n:
        df = df.iloc[rng.choice(len(df), n, replace=False)].reset_index(drop=True)
    y = datasets.get_label(df).astype(int)
    df = df.drop(columns=[config.LABEL_COL, config.GROUP_COL])
    print(f"[{'kal' if kal else 'te'}] taranan={taranan} "
          f"satır={len(df)} fraud={int(y.sum())} base_rate={y.mean():.6f}", flush=True)
    return df, y


def _skorla(df) -> "np.ndarray":
    """Kutu ONNX skoru (ablation'sız) — desen: auc_farki.py:126-137."""
    from kara_kutu.adapters.fraud import FraudAdapter

    np_, _pd, fq, _dec, session, _man = FraudAdapter()._yukle()
    tokenlar = fq.transform(df)
    ciktilar = []
    for start in range(0, len(tokenlar), 8192):
        tok = tokenlar[start:start + 8192]
        mask = (tok != 0) & (tok != 1)
        ciktilar.append(session.run(None, {"ids": tok, "mask": mask})[0].reshape(-1))
    return np.concatenate(ciktilar)


def _kari_esik(p: np.ndarray, y: np.ndarray, esik: float) -> dict:
    """Eşikte işletim sayıları (alarm/TP/FP/precision/recall/F2)."""
    alarm = p >= esik
    tp = int(((y == 1) & alarm).sum())
    fp = int(((y == 0) & alarm).sum())
    n_flag = tp + fp
    prec = tp / n_flag if n_flag else float("nan")
    rec = tp / int(y.sum()) if y.sum() else float("nan")
    f2 = (5 * prec * rec / (4 * prec + rec)) if n_flag and rec > 0 else 0.0
    return {"alarm": n_flag, "tp": tp, "fp": fp,
            "precision": round(prec, 6) if n_flag else None,
            "recall": round(rec, 6) if y.sum() else None,
            "f2": round(f2, 6)}


def _eniyi_f2(p: np.ndarray, y: np.ndarray) -> dict:
    """p >= t için F2'yi maksimize eden en yüksek bağ-sınırı eşiği (saf fonksiyon)."""
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("kalibrasyon örneklemde pozitif yok — örneklem büyütülmeli")
    sira = np.argsort(-p, kind="stable")
    s = p[sira]
    ysort = y[sira]
    tp_cum = np.cumsum(ysort)
    bag = np.flatnonzero(np.diff(s)) + 1
    sinir = np.append(bag, len(s) - 1)  # her bağ grubunun son indeksi
    k = sinir + 1
    tp = tp_cum[sinir]
    prec = tp / k
    rec = tp / n_pos
    pay = 4 * prec + rec
    f2 = np.where(pay > 0, 5 * prec * rec / np.where(pay > 0, pay, 1), 0.0)
    i = int(np.argmax(f2))
    return {"esik": float(s[sinir[i]]),
            "f2": round(float(f2[i]), 6),
            "esnek_not": "tie grubu eşikle birlikte alınır (p >= esik)"}


def _manifest_sha() -> str:
    import hashlib

    yol = REPO / "fraud" / "results" / "deploy"
    manifest = json.loads((yol / "manifest.json").read_text(encoding="utf-8"))
    disk = hashlib.sha256((yol / "model.onnx").read_bytes()).hexdigest()
    if disk != manifest["artifacts"]["model_onnx_sha256"]:
        raise RuntimeError("model.onnx sha256 manifest ile uyuşmuyor — paket bozuk")
    return disk


def kalibre(a) -> int:
    yonetmen = _kaydir("kalibre")
    config, fold = _foldlar()
    df, y = _ornekle(config, fold, a.n_kal, a.hizli, kal=True, p_ornek=a.p_ornek)
    t0 = time.time()
    p = _skorla(df)
    sec = round(time.time() - t0, 1)
    secim = _eniyi_f2(p, y)
    esik = secim["esik"]
    operasyon = _kari_esik(p, y, esik)
    ekim = _kari_esik(p, y, 0.9969438910484314)
    ozet = {
        "us": SEED, "n_kal": len(y), "fraud": int(y.sum()),
        "kural": KURAL, "kural_kaynak": KURAL_KAYNAK,
        "kalibrasyon_bolme": "user_fold < 8 (eğitim; te hiç okunmadı)",
        "esik": esik,
        "esikte": operasyon,
        "ekim_esigi_kutuda": ekim,
        "maks_p": round(float(p.max()), 6),
        "skor_suresi_sn": sec,
        "model_sha256": _manifest_sha(),
        "te_dokunuldu": False,
    }
    (yonetmen / "ozet.json").write_text(json.dumps(ozet, indent=2, ensure_ascii=False))
    artefakt = {
        "esik": esik,
        "kural": KURAL,
        "kural_kaynak": KURAL_KAYNAK,
        "olusturma": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "us": SEED,
        "kalibrasyon": {
            "tanım": "kutu ONNX olasılıkları, user_fold<8, doğal oran, tohum 42",
            "n": len(y), "fraud": int(y.sum()),
            "esikte": operasyon, "ekim_esigi_alarm": ekim["alarm"],
            "maks_p": round(float(p.max()), 6),
        },
        "model_sha256": ozet["model_sha256"],
        "kanit_kosum": f"kara_kutu/runs/{yonetmen.name}/ozet.json",
        "te_dogrulama": None,
    }
    hedef = Path(a.artefact_out)
    hedef.parent.mkdir(parents=True, exist_ok=True)
    hedef.write_text(json.dumps(artefakt, indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8")
    print(json.dumps(ozet, indent=2, ensure_ascii=False), flush=True)
    print(f"kayıt: {yonetmen/'ozet.json'} · artefakt: {hedef}", flush=True)
    return 0


def te_dogrula(a) -> int:
    """KARAR A1 (kullanıcı, 2026-11-22): eşik = Ekim ALARM BÜTÇESİNİN kutu
    ölçeğindeki quantili; te tek geçişte hem eşik hem işletim sayısı için okunur
    (AGENTS §4.1 tek kullanım — artefakta işlenir, tekrar reddedilir).

    Bütçe kaynağı: `fraud/results/false_alarm_audit.json` population —
    11.784 alarm / 4.700.000 satır (oran 0,0025072). Ekim'in sayıları
    kopyalanmaz; ORAN kopyalanır. Eğitim-fold best-F2 tanısı (A öncesi ölçüm)
    artefaktın `kalibrasyon_tani` bloğunda reddedilmiş aday olarak kalır:
    iç-örnek bölmede precision 0,054 ↔ te audit 0,156 (bölme farkı; mekanizma
    doğrulanmadı — §9)."""
    hedef = Path(a.artefact_out)
    if not hedef.is_file():
        print(f"artefakt yok: {hedef} — önce kalibrasyonu koş.", file=sys.stderr)
        return 2
    artefakt = json.loads(hedef.read_text(encoding="utf-8"))
    if artefakt.get("te_dogrulama") is not None:
        print("te ölçümü zaten kayıtlı — tekrar dokunmak yasak (AGENTS §4.1).",
              file=sys.stderr)
        return 3
    yonetmen = _kaydir("te")
    config, fold = _foldlar()
    from kara_kutu.adapters.fraud import FraudAdapter

    np_, _pd, fq, _dec, session, _man = FraudAdapter()._yukle()
    butce_alarm, butce_n, butce_oran = _butce()
    probs: list = []
    etiketler: list = []
    t0 = time.time()
    for parc in _te_parcalar(config, fold, a.hizli):
        tokenlar = fq.transform(parc[0])
        for start in range(0, len(tokenlar), 8192):
            tok = tokenlar[start:start + 8192]
            mask = (tok != 0) & (tok != 1)
            probs.append(session.run(None, {"ids": tok, "mask": mask})[0].reshape(-1))
        etiketler.append(parc[1])
    p = np.concatenate(probs)
    y = np.concatenate(etiketler)
    sec = round(time.time() - t0, 1)
    k = int(round(butce_oran * len(y)))
    s = np.sort(p)[::-1]
    esik = float(s[k - 1])  # bağ grubu eşikle birlikte alarm yakar (p >= esik)
    te = _kari_esik(p, y, esik)
    te.update({"n": len(y), "fraud": int(y.sum()), "esik": esik,
               "k_butce": k, "sure_sn": sec,
               "maks_p": round(float(p.max()), 6)})
    ozet = {"esik": esik, "te": te, "us": SEED,
            "butce": {"oran": round(butce_oran, 9), "tanım": f"{butce_alarm}/{butce_n}",
                      "kaynak": "fraud/results/false_alarm_audit.json:population"},
            "uyari": "tek seferlik final işletim noktası ölçümü (A1)"}
    (yonetmen / "ozet.json").write_text(json.dumps(ozet, indent=2, ensure_ascii=False))
    eski_esik = float(artefakt["esik"])  # eğitim-fold best-F2 adayı (tanı)
    artefakt["esik"] = esik
    artefakt["kural"] = KURAL_A1
    artefakt["kural_kaynak"] = ("fraud/results/false_alarm_audit.json:population "
                                f"({butce_alarm}/{butce_n}) — A1 kararı 2026-11-22")
    artefakt["kalibrasyon_tani"] = artefakt.pop("kalibrasyon")
    artefakt["kalibrasyon_tani"]["durum"] = (
        f"REDDEDİLDİ — iç-örnek bölme (eğitim-fold); eğitim-fold best-F2 eşik "
        f"adayı {eski_esik}, tanım gerekçesi §9")
    artefakt["te_dogrulama"] = {**te,
                                "kosum": f"kara_kutu/runs/{yonetmen.name}/ozet.json"}
    hedef.write_text(json.dumps(artefakt, indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8")
    print(json.dumps(ozet, indent=2, ensure_ascii=False), flush=True)
    print(f"kayıt: {yonetmen/'ozet.json'} · artefakt güncellendi: {hedef}", flush=True)
    return 0


def _te_parcalar(config, fold: dict[int, int], hizli: bool):
    """(df, y) parçaları — fold>=8 kullanıcıların TÜM satırları, bellek-dostu
    (desen: alarm_farki.py:143-166 chunk-taraması; burada örneklem yok)."""
    import pandas as pd

    import datasets

    from kara_kutu.adapters.fraud import IBM_KOLONLARI

    fold_dizi = np.fromiter((fold[u] for u in range(len(fold))), dtype=np.int8,
                            count=len(fold))
    kolonlar = list(IBM_KOLONLARI) + [config.GROUP_COL, config.LABEL_COL]
    for i, parc in enumerate(pd.read_csv(Path(config.BIG_CSV), usecols=kolonlar,
                                         chunksize=CHUNK)):
        if hizli and i >= 2:
            break
        g = parc[config.GROUP_COL].to_numpy()
        al = fold_dizi[g] >= 8
        if not al.any():
            continue
        alt = parc.loc[al].drop(columns=[config.GROUP_COL])
        y = datasets.get_label(alt).astype(int)
        alt = alt.drop(columns=[config.LABEL_COL])
        yield alt, y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-kal", type=int, default=1_000_000)
    ap.add_argument("--p-ornek", type=float, default=0.20,
                    help="tam dosya taramasında satırda kalma olasılığı (varsayılan 0,20)")
    ap.add_argument("--hizli", action="store_true",
                    help="geliştirme kontrolü: CSV'nin ilk ~8M satırıyla sınırlı")
    ap.add_argument("--te-dogrula", action="store_true",
                    help="TEK SEFERLİK final işletim noktası (fold>=8) — artefakt gerekir")
    ap.add_argument("--artefact-out", default=str(ARTEFACT_VARSAYILAN))
    a = ap.parse_args()
    if a.te_dogrula:
        return te_dogrula(a)
    return kalibre(a)


if __name__ == "__main__":
    raise SystemExit(main())
