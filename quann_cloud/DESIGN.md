# QUANN-Cloud: Detaylı Mimari Tasarım

## 1. Temel Bileşenler

### 1.1 QCloudWeight — Her Token'ın Noktası

Her token'ın buluttaki noktası bir `QCloudWeight` nesnesidir:

```python
QCloudWeight:
  - low: np.array[D]  # Aralık alt sınırı (başlangıç: -1.0)
  - high: np.array[D]  # Aralık üst sınırı (başlangıç: 1.0)
  - value: np.array[D]  # Şu anki nokta konumu (başlangıç: random[-1, 1])
  - direction: np.array[D]  # Her boyut için yön (+1/-1)
  - sigma: float  # Arama belirsizliği (başlangıç: 0.25)
  - outcomes: list[bool]  # Son başarı/başarısızlık penceresi
  - history: list[tuple]  # Path kaydı (y, direction, 'reward/censure')
```

**Önemli:** CloudLM'de her token bir `D` boyutlu vektördür. QCloudWeight her boyut
için **bağımsız** bir aralık tutar: `low[i]/high[i]` aralık, `value[i]` anlık konum,
`direction[i]` öğrenme yönü.

### 1.2 QCloudNetwork — CloudLM + QUANN Eğitim

Mevcut CloudLM mimarisine QUANN eğitim döngüsü eklenir:

```python
QCloudNetwork:
  - context_encoder: Transformer (4 katman, d_model=256)
  - to_nd: WordToND (256 → D boyutlu projeksiyon)
  - cloud_encoder: CloudEncoderND (nokta bulut kodlayıcı)
  - word_head: Tahmin modülü (cloud_state → logits)
  - point_head: Yardımcı nokta tahmini (cloud_state → next_point)
  - token_weights: list[QCloudWeight]  # Her token için QCloudWeight
  - bidirectional: bool  # Karşıt yön deneme aktif mi?
```

**Fark (standart CloudLM vs QCloudNetwork):**
- Standart: Backprop ile ağırlık güncelleme
- QCloudNetwork: Reward/ceza ile token'ların bulut noktalarının aralığı daraltma

### 1.3 Backprop'suz Eğitim Döngüsü

```python
def train_epoch(self, corpus):
    for sentence in corpus:
        tokens = tokenize(sentence)
        # 1. Mevcut bulut konumlarıyla tahmin
        cloud_state = self.forward(tokens)
        prediction = self.predict_next_token(cloud_state)
        # 2. Gerçek next token'ı bul + 3. hata
        true_token = tokens[next_position]
        error = self.compute_error(prediction, true_token)
        # 4. Her token'ın QCloudWeight'ini güncelle
        for i, token_weight in enumerate(self.token_weights):
            candidate = token_weight.sample()                      # 4a rastgele deneme
            new_error = self.forward_with_point(i, candidate)      # 4b yeni hata
            if new_error < error:
                token_weight.narrow(candidate, reward=True)        # 4c ödül
            else:
                opposite = token_weight.opposite(candidate)        # 4d karşıt yön
                if self.forward_with_point(i, opposite) < error:
                    token_weight.narrow(opposite, reward=True)
                else:
                    token_weight.narrow(candidate, reward=False)   # ceza
        # 5. Sigma adaptasyonu + 6. yakınsama kontrolü
        for token_weight in self.token_weights:
            token_weight.update_sigma()
        if self.all_converged():
            break
```

## 2. Mevcut Koddan Dönüşüm

### 2.1 QWeight → QCloudWeight (D Boyutlu)

Mevcut `QWeight` tek boyutlu (skaler). CloudLM'de her token `D` boyutlu vektör:

```python
# V0: Tek boyut                     # Cloud: D boyutlu vektör
class QWeight:                      class QCloudWeight:
    low: float = 0.0                   low: np.array[D] = -1.0   # Her boyut için aralık
    high: float = 1.0                  high: np.array[D] = 1.0
    value: float = None                value: np.array[D] = None
                                       direction: np.array[D] = +1 # Her boyut için yön
```

### 2.2 QNetwork → QCloudNetwork (CloudLM + QUANN)

Mevcut `QNetwork` basit lineer katmanlar. CloudLM'de transformer tabanlı:

```python
# V0: Basit lineer ağ                    # Cloud: Transformer + Point Cloud
class QNetwork:                          class QCloudNetwork:
    layers: list[QLinear]                    context_encoder: nn.TransformerEncoder
                                               to_nd: WordToND   # Bağlam → nokta projeksiyonu
                                               cloud_encoder: CloudEncoderND
                                               word_head: nn.Linear
                                               token_weights: list[QCloudWeight]
```

### 2.3 Narrownning — Skaler → Vektör

Mevcut `narrow()` skaler aralık daraltır. Cloud'da **boyut bazında** daraltma:

```python
# V0: Tek boyutlu narrownning
def narrow(self, new_value, reward):
    if reward:
        if new_value > self.value:
            self.high = min(1.0, self.high + expansion_rate)
        else:
            self.low = max(0.0, self.low - expansion_rate)
    else:
        if new_value > self.value:
            self.low = new_value
        else:
            self.high = new_value

# Cloud: D boyutlu narrownning (her boyut için ayrı; sınırlar [-1, 1])
def narrow(self, new_point: np.array[D], reward: bool):
    for i in range(D):
        if reward:
            if new_point[i] > self.value[i]:
                self.high[i] = min(1.0, self.high[i] + expansion_rate)
            else:
                self.low[i] = max(-1.0, self.low[i] - expansion_rate)
        else:
            if new_point[i] > self.value[i]:
                self.low[i] = new_point[i]
            else:
                self.high[i] = new_point[i]
    self.value = new_point
```

## 3. Deney Planı (durum: tamamı çalıştırıldı — sonuçlar PROJECT_SUMMARY/FINDINGS)

| Faz | Yapı | Beklenen | Metrik |
|---|---|---|---|
| 1: XOR Gate | 2→4 gizli (D boyutlu nokta), 1 çıkış | XOR backprop'suz çözülür | yakınsama adımı, son aralık genişliği |
| 2: Tiny Corpus (20K) | d_model=256, D=16, 4L, ~21M | rekabetçi PPL (~10-20) | PPL, üretim, yakınsama süresi |
| 3: Full Corpus (200K) | d_model=256, D=16/32, 4L, ~21M | CloudLM'e yakın PPL | PPL, üretim, yorumlanabilirlik |

## 4. Riskler ve Azaltma

| Risk | Olasılık | Etki | Azaltma |
|------|----------|------|---------|
| Yavaş yakınsama | Yüksek | Yüksek | Adaptive expansion rate |
| Derin ağlarda imkansız | Yüksek | Yüksek | Layer-wise training |
| Erken yakınsama | Orta | Orta | Minimum aralık threshold kontrol |
| High-D narrownning | Orta | Orta | Boyut bazında narrownning (her boyut bağımsız) |
| Korpus ile BPE uyumsuzluğu | Yüksek | Yüksek | Aynı BPE ile eğit + test |

## 5. Karar Kayıtları (ADR)

### ADR-001: Backprop Tamamen Bırakılıyor
- **Tarih:** 2026-10-14
- **Karar:** QUANN-Cloud tamamen backprop'suz çalışır. Gradient tahmini veya gradyan benzeri hiçbir mekanizma kullanılmaz.
- **Gerekçe:** Projenin temel felsefesi backprop'suz öğrenme. Gradyan kullanmak felsefeyi zedeler.
- **Sonuç:** Kabul edildi.

### ADR-002: CloudLM Mimarisi Kullanılıyor
- **Tarih:** 2026-10-14
- **Karar:** Mevcut CloudLM mimarisi (context_encoder + to_nd + cloud_encoder + heads) korunur. Sadece eğitim döngüsü QUANN ile değiştirilir.
- **Gerekçe:** CloudLM'in geometrik temsil gücünü korumak, backprop'u atmak öğrenme yöntemidir.
- **Sonuç:** Kabul edildi.

### ADR-003: D Boyutu Belirlenir
- **Tarih:** 2026-10-14
- **Karar:** D=16 ile başlanır (3 bölgeli syntax ayrımı). 32D'ye geçiş için alt yapı hazırlanır.
- **Gerekçe:** D=16 ile 3 syntactic region keşfedildi (paper'da). Daha yüksek D için daha fazla corpus gerekir.
- **Sonuç:** Kabul edildi.
