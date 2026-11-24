"""Complaints adaptörü — CFPB şikâyet metnini edge paketiyle skorlar (P2, eval-only).

Girdi: tek metin kolonu `narrative` (CFPB ham kolonu `consumer_complaint_narrative` —
`complaints/src/datasets_cfpb.py:11-14`). Model `textcloud.ComplaintCloud`
(`complaints/src/textcloud.py:7-23`); edge paketi `complaints/results/edge/`,
girdileri `tokens` int64 (B,64) + `mask` bool (B,64) — `complaints/src/export_edge.py:49-50`.

Skor: edge ONNX yalnız `z` yayar (`complaints/src/export_edge.py:24` ZOnly → forward[1]);
135 sınıflı `issue_head` numpy'de burada uygulanır — ağırlıklar
`complaints/results/reader.pt` `model_state_dict["issue_head.*"]` ve export aynı
checkpoint'den yapılıyor (`complaints/src/export_edge.py:41`). Skor = argmax issue
sınıfının softmax olasılığı. Gerekçe = BPE token ablasyonu: tek pozisyon maskelenir,
tahmin edilen sınıfın log-softmax farkı alınır — tek sınıflı
`kara_kutu/adapters/fraud.py:261-272` (`_chunk`) deseninin çok sınıflı karşılığı.
Eval-only: eğitim yapmaz, checkpoint yazmaz.
"""
from __future__ import annotations

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
EDGE_DIR = REPO_KOK / "complaints" / "results" / "edge"
CHECKPOINT = REPO_KOK / "complaints" / "results" / "reader.pt"
STABILITE = REPO_KOK / "complaints" / "results" / "complaints_scale_stability.json"
STABILITE_YOL = "complaints/results/complaints_scale_stability.json"

METIN_KOLON = "narrative"   # complaints/src/datasets_cfpb.py:14 OUTPUT_COLUMNS
ETIKET_KOLON = "issue"      # complaints/src/datasets_cfpb.py:14
_OZEL = frozenset({"<pad>", "<eos>", "<unk>"})

# Kod penceresi: config.MAX_LEN == edge manifest config.max_len (complaints/src/config.py:23,
# complaints/results/edge/manifest.json:15). Geniş ekseni sabit 64 olarak export edildi
# (dynamic_axes yalnız batch — complaints/src/export_edge.py:51-52), doldurma 64'e yapılır.
MAKS_LEN = 64

# ONNX batch ekseni her çağrıda bu değerde sabit tutulur — neden: `_onnx_kos` yorumu.
SABIT_BATCH = 32

_TOPK = 5

# Eşik YOK: modelin kendi karar kuralı argmax (complaints/src/train_reader.py:114
# `logits.argmax(-1)`). kara_kutu/adapters/fraud.py:37-39'daki alarm eşiği emsali buraya UYDURULMAZ;
# 0.0 = "eşik uygulanmıyor" anlamıdır. CLI `--esik` verilirse karar etiketine not
# düşülür, argmax değiştirilmez (skor zaten P(issue) olasılığıdır).
ESIK = 0.0
ESIK_KAYNAK = "eşik yok — karar kuralı argmax: complaints/src/train_reader.py:114"


class ComplaintsAdapter:
    """Edge paketini (ONNX + BPE + issue head) bir kez yükler; kayıt başına Verdict üretir."""

    name = "complaints"
    # Çok sınıflı dikey: binary sınıflandırma söz konusu değildir —
    # karar = issue adı (argmax), eşik uygulanmaz; pozitif/negatif etiketi YOKtur.
    pozitif_etiket = None
    negatif_etiket = None
    skor_etiket = "P(issue)"
    etiket_kolon = ETIKET_KOLON  # "issue"; runner parse ETTİ — adaptör ham hamlende okur

    def __init__(self, edge_dir: Path | None = None, top_k: int = _TOPK,
                 esik: float | None = None) -> None:
        self.edge = Path(edge_dir) if edge_dir else EDGE_DIR
        self.top_k = int(top_k)
        self.esik = ESIK if esik is None else float(esik)
        self.esik_kaynak = ESIK_KAYNAK  # runner override'da metni değiştirir
        self._paket: tuple[Any, ...] | None = None

    # ------------------------------------------------------------------ paket
    def _yukle(self) -> tuple[Any, ...]:
        """(np, tokenizer, id2token, session, W, b, id→issue, manifest) — tembel yükleme.

        Modül-adı izolasyonu: `complaints/src` modüllerini (config / export_edge /
        textcloud / generator benzerleri) HİÇ import etmeyiz —
        kara_kutu/adapters/fraud.py:68-71'de anlatılan aynı-üst-düzey-ad
        önbellekleri (config, quantize, ...) bu adaptöre
        dokunamaz. Gerekli tek repo modülü `cloudlm.tokenizer` (paket-altı ad,
        çakışmasız). Head için `weights_only=True`: checkpoint yalnız tensör ve
        dict/str/int taşır, pickle kodu çalıştırmaz.
        """
        if self._paket is not None:
            return self._paket
        try:
            import numpy as np
            import onnxruntime as ort
            import torch
            from cloudlm.tokenizer import BPETokenizer
        except ImportError as exc:  # pragma: no cover — ortam hatası, test dışı
            raise RuntimeError(
                f"Eksik bağımlılık ({getattr(exc, 'name', exc)}). complaints edge hattı "
                "numpy + onnxruntime + torch + tokenizers ister."
            ) from exc
        yollar = {"model": self.edge / "reader.onnx",
                  "tokenizer": self.edge / "tokenizer.json",
                  "manifest": self.edge / "manifest.json"}
        eksik = [str(y) for y in yollar.values() if not y.is_file()]
        if not CHECKPOINT.is_file():
            eksik.append(str(CHECKPOINT))
        if eksik:
            raise RuntimeError("Edge paketi eksik: " + ", ".join(eksik))
        head = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        w = head["model_state_dict"]["issue_head.weight"].numpy().astype("float32")
        b = head["model_state_dict"]["issue_head.bias"].numpy().astype("float32")
        id2issue = {int(kimlik): str(etiket)
                    for etiket, kimlik in head["issue_vocab"].items()}
        with yollar["manifest"].open(encoding="utf-8") as f:
            manifest = json.load(f)
        tokenizer = BPETokenizer.load(str(yollar["tokenizer"]))
        id2token = {int(kimlik): str(parca) for parca, kimlik in tokenizer.vocab.items()}
        session = ort.InferenceSession(str(yollar["model"]),
                                       providers=["CPUExecutionProvider"])
        self._paket = (np, tokenizer, id2token, session, w, b, id2issue, manifest)
        return self._paket

    @staticmethod
    def _onnx_kos(np, session, tokens, mask):
        """ONNX koşusu — batch ekseni HİÇBİR çağrıda değişmez (daima SABIT_BATCH).

        Neden: edge grafiğinin ara gerilimi (B,64,64,128). Aynı seansta batch boyutu
        değişince onnxruntime ya "Shape mismatch attempting to re-use buffer
        {1,64,64,128} != {2,64,64,128}" ile patlıyor ya da (B=320'de denendi) 64
        ablasyon varyantının yalnız 2 satırını benzersiz üretiyor — SESSİZ yanlış.
        Batch sabitken grup koşumu ile tek satır koşumu birebir aynı `z`'yi veriyor;
        bu değişmez `kara_kutu/tests/test_complaints.py`de test edilir. Eksik satırlar
        son satırla tekrarlanır ve çıktıda atılır.
        """
        n = int(len(tokens))
        if n == 0:
            return np.zeros((0, 0), dtype="float32")
        bloklar = []
        for baslangic in range(0, n, SABIT_BATCH):
            tk = tokens[baslangic:baslangic + SABIT_BATCH]
            mk = mask[baslangic:baslangic + SABIT_BATCH]
            fazla = SABIT_BATCH - len(tk)
            if fazla > 0:
                tk = np.concatenate([tk, np.repeat(tk[-1:], fazla, axis=0)], axis=0)
                mk = np.concatenate([mk, np.repeat(mk[-1:], fazla, axis=0)], axis=0)
            bloklar.append(session.run(["z"], {"tokens": tk, "mask": mk})[0][:len(tk)])
        return np.concatenate(bloklar, axis=0)

    # -------------------------------------------------------------- sözleşme
    def input_spec(self) -> InputSpec:
        return InputSpec(
            adapter=self.name,
            kolonlar=(METIN_KOLON,),
            aciklama="CFPB şikâyet anlatımı; tek zorunlu kolon 'narrative' (ham CFPB "
                     "kolonu 'consumer_complaint_narrative' — complaints/src/datasets_cfpb.py:11-14). "
                     "'issue' kolonu varsa doğruluk girdi-CSV etiketleri üzerinde ölçülür "
                     "(CSV'nin held-out olduğu kutudan bakarak doğrulanamaz).",
        )

    def model_dosyasi(self) -> Path:
        """Kanıt bloğunun sha256'sını alacağı dosya (packager API'si)."""
        return self.edge / "reader.onnx"

    def model_meta(self) -> ModelMeta:
        _, tokenizer, _, _, w, _, _, manifest = self._yukle()
        konfig = manifest["model_config"]
        return ModelMeta(
            ad="ComplaintCloud edge (ONNX z + issue_head)",
            esik=self.esik,
            esik_kaynak=self.esik_kaynak,
            olusturma="yok — edge manifest'te üretim damgası anahtarı yok",
            ek_bilgi={
                "manifest": "complaints/results/edge/manifest.json",
                "model_boyut": (f"d_model={konfig['d_model']} d_cloud={konfig['d_cloud']} "
                                f"d_z={konfig['d_z']} vocab={konfig['vocab_size']} "
                                f"n_issue={konfig['n_issue']}"),
                "paket_sha256_manifest": str(manifest["sha256"]["reader.onnx"]),
                "tokenizer_sha256_manifest": str(manifest["sha256"]["tokenizer.json"]),
                "girdi_imzasi": f"tokens int64 (B,{MAKS_LEN}) + mask bool (B,{MAKS_LEN}) "
                                f"→ z float32 (B,{konfig['d_z']}) "
                                "[complaints/src/export_edge.py:49-50]",
                "issue_head": f"{w.shape[0]} sınıf, numpy'de uygulanıyor "
                              "[complaints/results/reader.pt; complaints/src/export_edge.py:24]",
                "sozluk_token": str(tokenizer.num_tokens),
                "batch_kisiti": f"ONNX batch ekseni sabit {SABIT_BATCH} (bkz. _onnx_kos)",
            },
        )

    def baseline(self, ds: VeriSeti) -> tuple[Baseline, ...]:
        """Kıyas referansları — hepsi sabitlik dosyasından, satır atıflı (AGENTS §13).

        Sayılar uydurma değil: `complaints_scale_stability.json` okunarak birebir
        alındı; her satır atfı `kara_kutu/tests/test_complaints.py`de hem biçim hem
        "sayı o satırda gerçekten var" açısından denetlenir.
        """
        return (
            Baseline("CFPB kullanılabilir anlatım oranı (tam korpus)", "pay",
                     0.2277, f"{STABILITE_YOL}:8"),
            Baseline("Bölge kümeleme seed kararlılığı (ARI, seed 1-4 ort.)", "ARI",
                     0.9938922500989028, f"{STABILITE_YOL}:30"),
            Baseline("Bölge payı sıralama kararlılığı (Spearman ρ, en düşük)", "Spearman ρ",
                     1.0, f"{STABILITE_YOL}:33"),
            Baseline("Sürükleme penceresi — en büyük bölge payı kayması (bölge 7)",
                     "Δ pay", 0.2068, f"{STABILITE_YOL}:72"),
        )

    # ------------------------------------------------------------ girdi + tahmin
    @staticmethod
    def _matris(tokenizer, np, metinler):
        """(tokens, mask) — şekil (len(metinler), MAKS_LEN).

    Doldurma 0 + `encode(t)[:max_len] or [2]` deseni:
    `complaints/src/train_reader.py:29-34`.
    """
        satirlar = [tokenizer.encode(metin)[:MAKS_LEN] or [2] for metin in metinler]
        tk = np.zeros((len(satirlar), MAKS_LEN), dtype="int64")
        for i, satir in enumerate(satirlar):
            tk[i, :len(satir)] = satir
        return tk, tk != 0

    def predict(self, ds: VeriSeti) -> list[Verdict]:
        np, tokenizer, id2token, session, w, b, id2issue, _ = self._yukle()
        if METIN_KOLON not in ds.kolonlar:
            raise ValueError(f"CSV'de '{METIN_KOLON}' kolonu yok; bulunan: {', '.join(ds.kolonlar)}")
        metinler = [_metin(kayit.ham.get(METIN_KOLON)) for kayit in ds.kayitlar]
        if not metinler:
            return []
        tk, mk = self._matris(tokenizer, np, metinler)
        baz = _log_softmax(np, self._onnx_kos(np, session, tk, mk) @ w.T + b)
        kararlar: list[Verdict] = []
        for i, metin in enumerate(metinler):
            sinif = int(np.argmax(baz[i]))
            olasilik = float(np.exp(baz[i, sinif]))
            karar = id2issue.get(sinif, f"issue_id={sinif}")
            if self.esik > 0 and olasilik < self.esik:
                # Yalnız kullanıcı override'ında (runner: esik_kaynak override metni)
                # devreye giren NOT; argmax kararı değiştirilmez — modelin kuralı argmax.
                karar = f"{karar} (eşik altı)"
            kararlar.append(Verdict(
                kimlik=ds.kayitlar[i].kimlik,
                skor=olasilik,
                karar=karar,
                gerekceler=self._gerekceler(np, session, tk, mk, baz, i, sinif,
                                            w, b, id2token, metin),
            ))
        return kararlar

    def _gerekceler(self, np, session, tk, mk, baz, i, sinif, w, b, id2token, metin):
        """Kayıt başına gerekçe: tek BPE pozisyonunu maskeleyip farkı ölçmek.

        delta = log P(seçilen sınıf | tam metin) − log P(seçilen sınıf | tokensız).
        Pozitif → token bu issue'yu destekliyor; negatif → karşı. Tek sınıflı
        kara_kutu/adapters/fraud.py:244-249 farkının çok sınıflı karşılığı; orada
        tek sigmoid logit vardır.
        Dürüstlük filtresi: özel token'lar ve girdi metninde (harf-kayıtsız) GERÇEKTEN
        geçmeyen BPE parçaları (byte-fallback vb.) basılmaz — gerekçe iddiası daima
        girdi metnine bağlı kalır. Okunabilirlik katmanı: `_okunur` etiketi tercih
        edilir (en az 3 karakter + bir harf — `complaints/src/explain_region.py:31-32`
        emsali); hiçbir aday bu eşiği geçmezse zeminsel adaylar korunur, böylece
        gerekçe sütunu hiç boşalmaz. Aynı dosyanın :16-27 stopword listesi
        BURADA uygulanmaz: o liste bölge ADLANDIRMA içindir; ablasyonun bir
        stopword'e yüksek hassasiyeti gerçek bir model bulgusudur, gizlenmez.
        """
        gecerli = np.flatnonzero(mk[i])
        if gecerli.size == 0:
            return []
        adet = int(gecerli.size)
        var = np.repeat(tk[i:i + 1], adet, axis=0)
        vmask = np.repeat(mk[i:i + 1], adet, axis=0)
        vmask[np.arange(adet), gecerli] = False
        ab = _log_softmax(np, self._onnx_kos(np, session, var, vmask) @ w.T + b)
        delta = baz[i, sinif] - ab[:, sinif]
        gorulen: dict[str, float] = {}
        for yer, j in enumerate(gecerli):
            etki = float(delta[yer])
            if abs(etki) < 1e-6:
                continue
            etiket = _etiket(id2token.get(int(tk[i, j]), ""), metin)
            if not etiket:
                continue
            if abs(etki) > abs(gorulen.get(etiket, 0.0)):
                gorulen[etiket] = etki
        okunur = {etiket: etki for etiket, etki in gorulen.items() if _okunur(etiket)}
        sirali = sorted((okunur or gorulen).items(), key=lambda kv: -abs(kv[1]))
        return [Gerekce(etiket=etiket, delta_logit=etki)
                for etiket, etki in sirali[:self.top_k]]

    # -------------------------------------------------------------- metrikler
    def ek_metrikler(self, ds: VeriSeti, verdictler: list[Verdict]) -> dict[str, str]:
        """'issue' kolonu varsa girdi-CSV etiketleri üzerinde doğruluk ölçer.

        Etiket sözlüğü checkpoint'in `issue_vocab`'udur; sözlükte olmayan seyrek
        etiketler ölçümden düşer (complaints/src/train_reader.py:72 `fillna(-1)`
        deseni). Girdi CSV'sinin held-out olduğu doğrulanamaz — metrik adı bunu taşır.
        """
        if ETIKET_KOLON not in ds.kolonlar or len(verdictler) < 2:
            return {}
        *_, id2issue, _ = self._yukle()
        vocab = {etiket: kimlik for kimlik, etiket in id2issue.items()}
        gercek: list[int] = []
        tahmin: list[int] = []
        for kayit, verdict in zip(ds.kayitlar, verdictler):
            sinif = vocab.get(_metin(kayit.ham.get(ETIKET_KOLON)))
            etiket = verdict.karar.split(" (eşik altı)")[0]
            if sinif is None or etiket not in vocab:
                continue
            gercek.append(sinif)
            tahmin.append(vocab[etiket])
        metrik = "girdi CSV etiketleri, bu koşunun ölçümü"
        bulgular: dict[str, str] = {}
        if not gercek:
            return bulgular
        isabet = sum(1 for g, t in zip(gercek, tahmin) if g == t) / len(gercek)
        bulgular[f"issue top-1 doğruluk ({metrik})"] = f"{isabet:.4f}"
        if len(set(gercek)) > 1:
            try:
                import warnings

                from sklearn.metrics import balanced_accuracy_score

                # sklearn 1.9.0'da `labels` parametresi yok: CSV'de görülmeyen bir
                # sınıfın tahmini uyarı üretir ama ölçümde yanlış sayılır — uyarı
                # bastırılır, ölçümün kendisi sklearn tanımına (y_true sınıfları
                # üzerinde makro recall) bırakılır.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dengeli = float(balanced_accuracy_score(gercek, tahmin))
                bulgular[f"issue dengeli doğruluk ({metrik})"] = f"{dengeli:.4f}"
            except ImportError:  # pragma: no cover
                bulgular[f"issue dengeli doğruluk ({metrik})"] = (
                    "ölçülemedi: sklearn yok (DOĞRULANMADI)")
        ortalama = sum(v.skor for v in verdictler) / len(verdictler)
        bulgular["ortalama P(seçilen issue) (bu koşunun ölçümü)"] = f"{ortalama:.4f}"
        return bulgular


def _metin(deger: Any) -> str:
    """CSV hücresi → metin. None/NaN boş sayılır (boş metin [2] ile kodlanır)."""
    if deger is None:
        return ""
    if isinstance(deger, float) and deger != deger:
        return ""
    return str(deger).strip()


def _etiket(parca: str, metin: str) -> str:
    """BPE parçasını okunur gerekçe etiketine çevir; metinde geçmeyeni ele."""
    if not parca or parca in _OZEL:
        return ""
    temiz = parca.replace("Ġ", " ").replace("Ċ", " ").strip()
    if not temiz:
        return ""
    return temiz if temiz.casefold() in metin.casefold() else ""


def _okunur(etiket: str) -> bool:
    """Okunabilirlik eşiği: en az 3 karakter + bir harf

    `complaints/src/explain_region.py:31-32` (`_is_interpretable`) aynı eşiği kullanır.
    """
    return len(etiket) >= 3 and any(harf.isalpha() for harf in etiket)


def _log_softmax(np, logits):
    """Satır bazlı log-softmax (maksimum kaydırmalı, sayısal stabil)."""
    kayan = logits - logits.max(axis=-1, keepdims=True)
    return kayan - np.log(np.exp(kayan).sum(axis=-1, keepdims=True))


ADAPTER = ComplaintsAdapter()
