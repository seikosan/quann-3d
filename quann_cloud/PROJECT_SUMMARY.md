# QUANN-Cloud: Backprop'suz Nokta Bulut Modeli — Proje Özeti

## 🎯 Hedef

**Dünyada bir ilk:** Backpropagation olmadan, nokta bulut tabanlı dil modeli.

- **Mimari:** CloudLM (nokta bulut) — sekans yok, geometri var
- **Eğitim:** QUANN'ın aralık daraltma + sigma adaptasyonu — backprop yok
- **Dust (dış referans):** Backprop'suz eğitim fikrinden ilham alır, sizin değil

> ⚠️ **GERÇEK VERİ SONUCU (2026-10-14):** QUANN (backprop'suz) gerçek Türkçe
> veride backprop'a **rekabet edemiyor** — PPL 1016 (random 1045, backprop 563).
> Mekanizma küçük ölçekte PPL'ı random'ın altına indiriyor ama 0th-order olduğu için
> LLM'e ölçeklenmiyor. Detay: **`FINDINGS.md`** · ölçülmüş, hype'sız.

## 📁 Proje Yapısı

```
quann_cloud/
├── README.md                  # Giriş
├── DESIGN.md                  # Detaylı mimari tasarım (pseudokod + ADR)
├── EXPERIMENTS.md             # Deney planı
├── PROJECT_SUMMARY.md         # Bu dosya (özet)
│
├── src/
│   ├── qcloud_weight.py       # QCloudWeight (her token'ın noktası) ✅ TEST
│   ├── qcloud_network.py      # QCloudNetwork (CloudLM + QUANN eğitim) ✅ TEST
│   └── qcloud_trainer.py      # QCloudTrainer (backprop'suz pipeline) ✅ TEST
│
├── experiments/
│   ├── xor_gate.py            # XOR doğrulama (v0 benzeri)
│   ├── tiny_corpus.py         # Küçük corpus (20K cümle)
│   └── full_corpus.py         # Büyük corpus (200K cümle)
│
├── tests/                     # test_weight.py, test_network.py
├── data/                      # turkish_sentences.txt corpus
└── logs/                      # Eğitim logları
```

## ✅ Test Sonuçları

| Modül | Sonuç | Özet |
|---|---|---|
| QCloudWeight | ✅ | reward → yön tersine döner; censure → aralık daralır (width 2.0→0.83); sigma 10 adımda 0.076; 20 adımda yakınsamadı (beklenen) |
| QCloudNetwork | ✅ | D=3, d_model=64, d_cloud=512, 2 katman, bidir; 2.48M param; forward → logits [1,100], point [1,3] |
| QCloudTrainer | ✅ | train_epoch (forward + reward/censure narrownning), evaluate_ppl, generate |

## 📐 Mimari Tasarım

Tam pseudokod `DESIGN.md`'de. Özet:

- **QCloudWeight** — her token'ın D boyutlu noktası: `low/high` (bağımsız aralık),
  `value`, `direction` (+1/-1/boyut), `sigma`, `outcomes`, `history`.
- **QCloudNetwork** — `context_encoder` (Transformer 4L, d_model=256) + `to_nd`
  (256→D) + `cloud_encoder` (CloudEncoderND) + `word_head`/`point_head` +
  `token_weights` (lazy `Dict[token_id, QCloudWeight]`). Standart CloudLM backprop
  günceller; QCloudNetwork reward/ceza ile bulut noktalarının aralığını daraltır.
- **QCloudTrainer** — epoch: forward → her token'ı reward/cense güncelle → sigma
  adaptasyonu → yakınsama kontrolü.

## 📊 Karşılaştırma: Backprop vs QUANN-Cloud

| Metrik | CloudLM (BP) | QUANN-Cloud |
|--------|--------------|-------------|
| **Backprop / Gradient** | ✅ Var | ❌ Yok |
| **Credit Assignment** | Chain rule | Token-level reward |
| **Search Space** | Weight space | Point cloud geometry |
| **Search Mechanism** | Gradient descent | Reward/cense + narrownning |
| **Learning Speed** | Hızlı (15-20 epoch) | Yavaş (15-25 epoch) |
| **Final PPL** | ~4-5 | ~5-10 (hedef) |
| **Interpretability / Edge** | ✅ Geometrik / potansiyel | ✅ Geometrik / potansiyel |
| **Memory** | KV cache + d_model | D boyutlu point cloud |

## 🧪 Deney Planı (durum: tamamı çalıştırıldı)

| Faz | Yapı | Sonuç |
|---|---|---|
| 1: XOR Gate | 2→4 gizli (D=16), 1 çıkış | ✅ %100 (FINDINGS §6) |
| 2: Tiny Corpus (20K) | d_model=256, D=16, ~21M | saf: NULL; hibrit: 4.271 PPL (aşağıda) |
| 3: Full Corpus (200K) | D=16/32 | backprop 2.94 vs QUANN random'a yakın |

## 📊 Deney Sonuçları (14-15 Ekim 2026)

### Saf tasarım (backprop yok) — Faz 2 tiny corpus

| Epoch | Held-out PPL (2K cümle) |
|---|---|
| 1 | 20622 |
| 2 | 20571 |
| 3 | 20601 |

**Sonuç: NULL.** PPL ≈ vocab (20000) = rastgele model. Kök nedenler:
1. Rastgele word_head (backprop yok) → 20K vocab'a anlamlı harita kurulamaz
2. Düz hata manzarası (Δ≈0) → yönsel sinyal yok
3. 1e-6 eşiği gürültü tabanının altında → %97 "reward" gürültü

### Hibrit tasarım (ağ backprop + QUANN noktaları arama)

Pilot (2K cümle, 2 epoch):
- Epoch 1: base_err 9.93→9.08 (backprop), PPL 20600→9264; arama katkısı ≈0 (Δ+0.0001)
- Epoch 2: base_err 7.10 (öğrenme sürüyor) ama held-out PPL 13794 (2K'da overfit)

**Tam 20K deneyi kuyrukta** (queue_hybrid.sh): control CloudLM 20K + hibrit 20K + karşılaştırma.

## 🔧 Yapılan Tasarım Düzeltmeleri (14 Ekim)

1. **Token kimliği bazlı ağırlıklar:** `token_weights` artık `Dict[token_id, QCloudWeight]`
   (lazy). Önceki konum bazlı (10 ağırlık) tasarım her cümle için aynı bulutu üretiyordu → sabit çıktı.
2. **Gerçek arama:** Aday noktalar forward'a `points_override` ile giriyor;
   batch'li değerlendirme (base + tüm adaylar tek forward).
3. **narrow_search():** düzeltmeli semantik — ödül→segmente daralt, ceza→kötü tarafı kes,
   value cezada sabit. (Orijinal `narrow()` ters mantıktı; korundu, hibrit `search` kullanıyor.)
4. **Encoder skip:** QUANN noktaları verildiğinde transformer encoder atlanır (hız).
5. **Eval düzeltmesi:** PPL hesabında token sayısını 10× şişiren hata giderildi.
6. **Gerçek BPE tokenizer** trainer'a entegre (placeholder'lar kaldırıldı).

## ✅ Gerçekleşenler (14 Ekim 2026)

1. **Faz 1 XOR:** QUANN, XOR'u backprop'suz **%100** çözüyor → mekanizma sağlam.
2. **Faz 2-3 Gerçek veri:** QUANN vs Backprop, aynı mimari/veri/BPE: QUANN random'a yakın
   (1016 vs 1045), backprop 563 → **0th-order ölçeklenmiyor** (FINDINGS.md).
3. **Edge:** CloudLM weight-only INT8 → 84MB→46MB, kalite 0.999x (kayıpsız).

## 📊 200K & 20K Karşılaştırmalar (15 Ekim — hibrit oturumu)

### 200K: CloudLM vs Causal Transformer (50ep, 200K cümle, 2000 held-out)

| Model | Params | Held-out PPL |
|---|---|---|
| CloudLM (32D) | 21.0M | **6.45** |
| Causal Transformer | 22.7M | **2.94** |

**Başlık bulgu:** Aynı ölçek/veri/epoch'ta standart causal Transformer bulutu 2.2×
geçiyor; üretim de belirgin iyi (bağlı Türkçe vs tekrarlayan gürültü).

### 20K: Adil karşılaştırma (aynı güncelleme bütçesi, 10ep)

| Model | Held-out PPL |
|---|---|
| Random | 20,000 |
| CloudLM — adil control (tek pencere) | 13,535 |
| CloudLM — tüm pencereler | 6,183 |
| **QUANN-Cloud hibrit** (best@epoch1) | **4,271** |

- Hibrit, adil control'ü 3.2× geçiyor — avantaj **temsil** farkından
  (token kimliği noktaları vs to_nd), **aramadan değil** (Δcand≈0).
- Hibrit epoch 1'den sonra overfit (→11236); üretim word-salad.

## 🚀 Sonraki Adımlar

1. **CloudLM'i (backprop'lu) tam ölçekli edge paketle:** tam INT8 (embedding dahil →
   ~21MB) + int8 GEMM (ONNX/CoreML) ile gerçek hızlanmayı ölç.
2. **CloudLM'i güçlü korpusla eğit** (temiz BPE + turkish_wiki) → gerçek PPL + üretim kalitesi.
3. **QUANN'ı doğru nişinde konumla:** küçük/edge/yorumlanabilir modeller için
   backprop'suz alternatif (XOR kanıtıyla).
4. **QUANN-Cloud v1.0:** Patent + yayın — iddiayı FINDINGS.md'deki savunulabilir
   çerçevede (mimari güçlü, QUANN küçük ölçek, edge uygun). Güncel patent sürümleri:
   `docs/patent/`.

## 💡 Özet

**QUANN-Cloud:** Backprop'suz, nokta bulut tabanlı dil modeli.

- **Sekans yok + Geometri var + Backprop yok + Arama-Tabanlı Öğrenme**
- **Sizin 2 sisteminiz:** QUANN (v0 backprop'suz) + CloudLM (mimari) ·
  **Dış referans:** Dust (Q-Labs)
- **Öğrenme = Yakınsama:** Aralık bir eşik altına düştüğünde token "öğrenmiş" sayılır.

**Kod durumu:** QCloudWeight ✅ · QCloudNetwork ✅ · QCloudTrainer ✅ ·
QUANN motoru (`qcloud_quann.py`) ✅ · Gerçek veri deneyi (`real_data_test.py`) ✅

> Gerçek veri bulgusu için **`FINDINGS.md`**'ye bak — QUANN'ın 0th-order sınırı,
> ölçülen PPL karşılaştırması ve savunulabilir iddia çerçevesi orada.
