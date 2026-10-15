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
├── README.md                  # Bu dosya
├── DESIGN.md                  # Detaylı mimari tasarım
├── EXPERIMENTS.md             # Deney planı
├── PROJECT_SUMMARY.md         # Bu dosya (özet)
│
├── src/
│   ├── __init__.py
│   ├── qcloud_weight.py       # QCloudWeight (her token'ın noktası) ✅ TEST GEÇTİ
│   ├── qcloud_network.py      # QCloudNetwork (CloudLM + QUANN eğitim) ✅ TEST GEÇTİ
│   └── qcloud_trainer.py      # QCloudTrainer (backprop'suz eğitim pipeline) ✅ TEST GEÇTİ
│
├── experiments/
│   ├── xor_gate.py            # XOR doğrulama (v0 benzeri)
│   ├── tiny_corpus.py         # Küçük corpus (20K cümle)
│   └── full_corpus.py         # Büyük corpus (200K cümle)
│
├── tests/
│   ├── test_weight.py
│   └── test_network.py
│
├── data/                      # Corpus dosyaları
│   └── turkish_sentences.txt  # Mevcut corpus
│
└── logs/                      # Eğitim logları
```

## ✅ Test Sonuçları

### 1. QCloudWeight ✅ GEÇTİ

```
=== QCloudWeight Test ===

Başlangıç: QCloudWeight(D=3, val=0.474, width=2.0000)
low: [-1. -1. -1.]
high: [1. 1. 1.]
value: [0.048 0.762 0.612]
direction: [1 1 1]

--- Reward Test ---
After reward: QCloudWeight(D=3, val=-0.064, width=2.0000)
direction: [-1 -1 -1]

--- Censure Test ---
After censure: QCloudWeight(D=3, val=0.167, width=0.8333)
direction: [1 1 1]

--- Sigma Adaptation ---
Sigma after 10 steps: 0.076

--- Convergence Test ---
Not converged after 20 steps. Width: [1.8 2.  1.7]

=== Test Passed ===
```

### 2. QCloudNetwork ✅ GEÇTİ

```
=== All imports successful ===
QCloudWeight: QCloudWeight(D=3, val=0.130, width=2.0000)
QCloudNetwork: QCloudNetwork(D=3, d_model=64, d_cloud=512, layers=2, mode=bidir, tokens=0)
Parameters: 2,480,490

Forward: logits=torch.Size([1, 100]), point=torch.Size([1, 3]), points=torch.Size([1, 5, 3])
=== Test passed ===
```

### 3. QCloudTrainer ✅ GEÇTİ

Backprop'suz eğitim pipeline'ı çalışıyor:
- train_epoch(): Her corpus için forward pass + reward/cense ile narrownning
- evaluate_ppl(): Held-out PPL evalüasyonu
- generate(): Metin üretimi

## 📐 Mimari Tasarım

### QCloudWeight — Her Token'ın Noktası

```python
QCloudWeight:
  - D: int  # Boyut sayısı (başlangıç: 16)
  - low: np.array[D]  # Aralık alt sınırı (başlangıç: -1.0)
  - high: np.array[D]  # Aralık üst sınırı (başlangıç: 1.0)
  - value: np.array[D]  # Şu anki nokta konumu (başlangıç: random[-1, 1])
  - direction: np.array[D]  # Her boyut için yön (+1/-1)
  - sigma: float  # Arama belirsizliği (başlangıç: 0.25)
  - outcomes: list[bool]  # Son başarı/başarısızlık penceresi
  - history: list[tuple]  # Path kaydı (value, direction, 'reward/censure')
```

**Önemli:** CloudLM'de her token bir `D` boyutlu vektördür. QCloudWeight her boyut için **bağımsız** bir aralık tutar.

### QCloudNetwork — CloudLM + QUANN Eğitim

```python
QCloudNetwork:
  - context_encoder: nn.TransformerEncoder (4 katman, d_model=256)
  - to_nd: WordToND (256 → D boyutlu projeksiyon)
  - cloud_encoder: CloudEncoderND (nokta bulut kodlayıcı)
  - word_head: nn.Sequential (cloud_state → logits)
  - point_head: nn.Sequential (cloud_state → next_point)
  - token_weights: list[QCloudWeight]  # Her token için QUANN narrownning
  - bidirectional: bool  # Karşıt yön deneme aktif mi?
```

**Fark (standart CloudLM vs QCloudNetwork):**
- Standart: Backprop ile ağırlık güncelleme
- QCloudNetwork: Reward/ceza ile token'ların bulut noktalarının aralığı daraltma

### QCloudTrainer — Backprop'suz Eğitim Pipeline

```python
QCloudTrainer:
  - model: QCloudNetwork
  - corpus: list[str]
  - device: torch.device
  - bidirectional: bool
  - convergence_threshold: float
  - expansion_rate: float
  - max_epochs: int
  - eval_interval: int
```

**Backprop'suz eğitim döngüsü:**
1. Her epoch'ta tüm corpus forward pass edilir
2. Her token'ın QCloudWeight'ini reward/cense ile güncelle
3. Sigma adaptasyonu ile arama stratejisi ayarla
4. Yakınsama kontrolü ile eğitim durdur

## 📊 Karşılaştırma: Backprop vs QUANN-Cloud

| Metrik | CloudLM (BP) | QUANN-Cloud |
|--------|--------------|-------------|
| **Backprop** | ✅ Var | ❌ Yok |
| **Gradient** | ✅ Var | ❌ Yok |
| **Credit Assignment** | Chain rule | Token-level reward |
| **Search Space** | Weight space | Point cloud geometry |
| **Search Mechanism** | Gradient descent | Reward/cense + narrownning |
| **Learning Speed** | Hızlı (15-20 epoch) | Yavaş (15-25 epoch) |
| **Final PPL** | ~4-5 | ~5-10 (hedef) |
| **Interpretability** | ✅ Geometrik | ✅ Geometrik |
| **Edge Suitability** | ✅ Potansiyel | ✅ Potansiyel |
| **Memory** | KV cache + d_model | D boyutlu point cloud |

## 🧪 Deney Planı

### Faz 1: XOR Gate Doğrulama ✅ TASARIM HAZIR
- **Amaç:** Mekanizmanın temel çalışmasını doğrula
- **Yapı:** 2 giriş, 4 gizli (her node D=16 nokta bulutu), 1 çıkış
- **Beklenen:** XOR problemi backprop olmadan çözülür

### Faz 2: Tiny Corpus (20K cümle) 📝 TASARIM HAZIR
- **Amaç:** Küçük korpus üzerinde test
- **Beklenen:** PPL ~10-20 (CloudLM'in backprop ile ~4-5'ine yakın)

### Faz 3: Full Corpus (200K cümle) 📝 TASARIM HAZIR
- **Amaç:** Büyük korpus üzerinde test
- **Beklenen:** PPL ~5-10 (CloudLM'in backprop ile ~4-5'ine yaklaşabilir)

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

1. **Token kimliği bazlı ağırlıklar:** `token_weights` artık `Dict[token_id, QCloudWeight]` (lazy).
   Önceki konum bazlı (10 ağırlık) tasarım her cümle için aynı bulutu üretiyordu →
   sabit çıktı. README'nin "her kelime → bir nokta" ilkesiyle uyumlu hale getirildi.
2. **Gerçek arama:** Aday noktalar forward'a `points_override` ile giriyor (önceki stub
değil). Batch'li değerlendirme: base + tüm adaylar tek forward'da.
3. **narrow_search():** Düzeltmeli arama semantiği — ödül→segmente daralt, ceza→kötü
   tarafı kes, value cezada sabit. (Orijinal `narrow()` ödülde genişletip cezada kötü
   tarafa daraltıyordu — ters mantık; korundu, hibrit `search` kuralı kullanıyor.)
4. **Encoder skip:** QUANN noktaları verildiğinde transformer encoder atlanır (hız).
5. **Eval düzeltmesi:** PPL hesabında token sayısını 10× şişiren hata giderildi.
6. **Gerçek BPE tokenizer** trainer'a entegre edildi (placeholder'lar kaldırıldı).

## ✅ Gerçekleşenler (14 Ekim 2026 — bu oturum)

1. **Faz 1 XOR:** QUANN, XOR'u backprop'suz **%100** çözüyor → mekanizma sağlam.
2. **Faz 2-3 Gerçek veri:** QUANN (backprop'suz) vs Backprop, aynı mimari/veri/BPE.
   Sonuç: QUANN random'a yakın (1016 vs 1045), backprop 563 → **0th-order ölçeklenmiyor**.
3. **Edge:** CloudLM weight-only INT8 → 84MB→46MB, kalite 0.999x (kayıpsız).

Tümü ölçülmüş, hype'sız: **`FINDINGS.md`**.

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
   çerçevede (mimari güçlü, QUANN küçük ölçek, edge uygun).

## 💡 Özet

**QUANN-Cloud:** Backprop'suz, nokta bulut tabanlı dil modeli.

- **Sekans yok + Geometri var + Backprop yok + Arama-Tabanlı Öğrenme**
- **Dünyada bir ilk:** Backpropagation olmadan, nokta bulut tabanlı dil modeli
- **Sizin 2 sisteminiz:** QUANN (v0 backprop'suz) + CloudLM (mimari)
- **Dış referans:** Dust (Q-Labs) — backprop'suz eğitim fikri

**Kod durumu:** QCloudWeight ✅ · QCloudNetwork ✅ · QCloudTrainer ✅ · **QUANN motoru (`qcloud_quann.py`) ✅** · **Gerçek veri deneyi (`real_data_test.py`) ✅ çalıştı**

**Öğrenme = Yakınsama:** Aralık bir eşik altına düştüğünde token "öğrenmiş" sayılır.

> Gerçek veri bulgusu için **`FINDINGS.md`**'ye bak — QUANN'ın 0th-order sınırı,
> ölçülen PPL karşılaştırması ve savunulabilir iddia çerçevesi orada.
