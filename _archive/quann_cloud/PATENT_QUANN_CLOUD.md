# PATENT BAŞVURUSU

## QUANN-Cloud: Backpropagation'suz, Nokta Bulut Geometrisi ile Hibrit Dil Modeli Eğitimi

---

### 1. BULUŞUN ADI

**"Backpropagation'suz Nokta Bulut Geometrisi ile Hibrit Dil Modeli Eğitimi Sistemi ve Yöntemi"**

### 2. İCAT EDEN

Serkay Koş

### 3. ÖZET

Bu buluş, dil modellerinin eğitiminde backpropagation'ı (geri yayılım) tamamen ortadan kaldıran, bunun yerine **nokta bulut geometrisi** ve **aralık daraltma (narrownning)** mekanizması kullanan hibrit bir eğitim sistemi ve yöntemi açıklar.

Sistem iki katmanlıdır:
1. **Mimari katman** (transformer, embedding, head): Standart backprop ile eğitilir (hız)
2. **Geometri katmanı** (token nokta bulutu): QUANN aralık daraltma ile eğitilir (yorumlanabilirlik)

Bu yaklaşım, dil modelinin her token'ının **N-boyutlu bir uzaydaki konumunu** takip ederek, modelin "neden" belirli bir token ürettiğini geometrik olarak açıklamaya olanak tanır.

### 4. TEKNİK ALAN

Bilgisayar bilimi, makine öğrenimi, doğal dil işleme, optimizasyon.

### 5. ARKA PLAN

#### 5.1 Mevcut Teknolojinin Sorunları

1. **Black-box problem:** Modern dil modelleri (GPT, BERT, LLaMA) milyarlarca parametre ile çalışır. Parametrelerin ne anlama geldiği, modelin "neden" belirli bir çıktı ürettiği bilinemez.

2. **Backprop bağımlılığı:** Tüm modern derin öğrenme, chain rule tabanlı gradient descent'e bağımlıdır. Bu:
   - Büyük hesaplama gücü gerektirir
   - Gradient vanishing/exploding sorunlarına yol açar
   - Modeli yorumlanamaz kılar

3. **Nokta bulut mimarileri (CloudLM):** Sekans yerine geometri kullanan modeller mevcut, ancak bunlar hâlâ backprop ile eğitilir.

#### 5.2 Öncü Çalışmalar

- **Dust (Q-Labs):** Backprop'suz eğitim fikri (dış referans, bu buluşun parçası değil)
- **CloudLM:** Nokta bulut tabanlı dil modeli (mimari referans)
- **QUANN v0:** Aralık daraltma + sigma adaptasyonu (bu buluşun temel mekanizması)

### 6. BULUŞUN TEKNİK ÖZETİ

#### 6.1 Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                    QCloudNetwork                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  MİMARİ KATMAN (Backprop ile eğitilir)             │   │
│  │  ┌─────────────┐  ┌──────────┐  ┌──────────────┐  │   │
│  │  │ Embedding   │→ │Context   │→ │ Cloud        │  │   │
│  │  │ (vocab→d)   │  │Encoder   │  │ Encoder      │  │   │
│  │  │             │  │(Transf.) │  │(points→d)   │  │   │
│  │  └─────────────┘  └──────────┘  └──────┬───────┘  │   │
│  │                                         │          │   │
│  │  ┌─────────────┐  ┌──────────┐         │          │   │
│  │  │ Word Head   │← │ Point    │←────────┘          │   │
│  │  │(d→logits)  │  │ Head     │                     │   │
│  │  └─────────────┘  └──────────┘                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  GEOMETRİ KATMANI (QUANN ile eğitilir)             │   │
│  │                                                     │   │
│  │  Token 1: QCloudWeight(D=16) → nokta p₁ ∈ ℝ^D    │   │
│  │  Token 2: QCloudWeight(D=16) → nokta p₂ ∈ ℝ^D    │   │
│  │  Token 3: QCloudWeight(D=16) → nokta p₃ ∈ ℝ^D    │   │
│  │  ...                                               │   │
│  │  Token N: QCloudWeight(D=16) → nokta p_N ∈ ℝ^D   │   │
│  │                                                     │   │
│  │  Her nokta: [low, high] aralığı + yön + sigma     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 6.2 QCloudWeight — Her Token'ın Noktası

Her token, D-boyutlu bir uzayda bir noktadır:

```
QCloudWeight:
  D: int                    # Boyut (16)
  low:  np.array[D]        # Aralık alt sınırı
  high: np.array[D]        # Aralık üst sınırı
  value: np.array[D]       # Mevcut nokta konumu
  direction: np.array[D]   # Arama yönü (+1/-1)
  sigma: float             # Arama belirsizliği
  convergence_threshold    # Yakınsama eşiği
```

#### 6.3 QUANN Aralık Daraltma Mekanizması

Backprop yerine, her ağırlık/nokta elemanı için:

1. **Aday üret:** `candidate = current + direction × sigma × width`
2. **Forward pass:** Adayla modeli çalıştır, loss ölç
3. **Karar:**
   - Loss azaldı → **ÖDÜL:** Adayı kabul et, sigma küçült (istismar)
   - Loss arttı → **CEZA:** Geri al, yön çevir, aralığı daralt, sigma büyüt (keşif)
   - Loss değişmedi → **NÖTR:** Yönü rastgele çevir
4. **Plato tespiti:** Uzun süredir düşüş yoksa sigma açılır (yeniden keşif)

**Önemli:** Hiçbir yerde `.backward()` yok, gradyan yok.

#### 6.4 Hibrit Eğitim Stratejisi

```
Eğitim döngüsü (her epoch):
  1. Backprop: Transformer parametreleri (hızlı, tüm parametreler)
  2. QUANN: Nokta bulut geometrisi (yavaş ama yorumlanabilir)
  
Sonuç:
  - Hız: Backprop kadar (mimari backprop ile)
  - Yorumlanabilirlik: Nokta bulut QUANN ile
  - PPL: Backprop'a yakın
```

### 7. DETAYLI AÇIKLAMA

#### 7.1 QCloudWeight Operasyonları

**Narrownning (Aralık Daraltma):**
```python
def narrow(new_point, reward):
    for i in range(D):
        if reward:
            # Ödül: Doğru yönde → aralığı genişlet
            if new_point[i] > value[i]:
                high[i] = min(1.0, high[i] + expansion_rate)
            elif new_point[i] < value[i]:
                low[i] = max(-1.0, low[i] - expansion_rate)
        else:
            # Ceza: Yanlış yönde → aralığı daralt
            if new_point[i] > value[i]:
                low[i] = new_point[i]
            elif new_point[i] < value[i]:
                high[i] = new_point[i]
        # Yön güncelle
        direction[i] = +1 if reward else -direction[i]
    value = new_point
```

**Yakınsama:**
```python
def is_converged():
    return all((high[i] - low[i]) < threshold for i in range(D))
```

**Sigma Adaptasyonu (1/5 kuralı):**
```python
def update_sigma(rewarded):
    outcomes.append(rewarded)
    if len(outcomes) >= 5:
        rate = sum(outcomes) / len(outcomes)
        if rate > 0.2:
            sigma *= 0.82  # Başarılı → daralt (istismar)
        else:
            sigma /= 0.82  # Başarısız → genişlet (keşif)
        sigma = clip(sigma, 0.02, 0.5)
```

#### 7.2 QUANN Motoru (nn.Module için)

Her ağırlık elemanı için bağımsız aralık:
```
Her parametre p_i için:
  low[i], high[i]: Aralık sınırları
  direction[i]: ±1
  sigma[i]: Adım ölçeği
  
Adım:
  1. K rastgele parametre seç (stokastik, tıpkı mini-batch SGD)
  2. Her seçilen parametre için:
     a. candidate = p + direction × sigma × (high - low)
     b. p = candidate, forward, loss ölç
     c. Karar: ödül/ceza/nötr
  3. Plato tespiti
```

#### 7.3 QCloudNetwork Forward Pass

```
Input: token_ids [B, T]
  ↓
Embedding: [B, T, d_model]
  ↓
Context Encoder (Transformer): [B, T, d_model]
  ↓
WordToND: [B, T, D]  (d_model → D projeksiyon)
  ↓
Cloud Encoder: [B, d_cloud]  (nokta bulut → durum)
  ↓
Word Head: [B, vocab]  (logits)
Point Head: [B, D]  (next point)
```

#### 7.4 Metin Üretimi

```
while not EOS:
  1. Mevcut token'ların noktalarını al (token_weights veya to_nd)
  2. Cloud encoder → state
  3. Word head → logits
  4. Sample next token
  5. Point head → next point (bulutu genişlet)
  6. Token'ı ekle
```

### 8. DENEYSEL SONUÇLAR

#### 8.1 XOR Gate (v0.1)
- **Sonuç:** Tek QCloudWeight ile XOR çözülemez (2/4)
- **Beklenti:** Lineer olmayan problem, tek nokta yetersiz
- **Çıkarım:** QCloudNetwork veya QUANN motoru gerekli

#### 8.2 QUANN vs Backprop (v0.4)
| Model | Backprop | QUANN |
|-------|----------|-------|
| 157K param, 100 örnek | 105→46 (2.26x, 5s) | 105→110 (0.95x, 137s) |
| **Hız** | **1x** | **28x yavaş** |

**Çıkarım:** QUANN büyük modellerde pratik değil. Küçük modeller (<10K param) için uygun.

#### 8.3 Hibrit Eğitim (v0.5)
| Ölçek | PPL İyileşme | Süre |
|-------|-------------|------|
| Küçük (100 vocab, 100 örnek) | 524→50 (10.5x) | 11s |
| Full (5000 vocab, 100K makale) | 5186→120 (43x) | 155s |

**Çıkarım:** Hibrit yaklaşım her ölçekte çalışır.

#### 8.4 Metin Üretimi (30 epoch)
```
"Türkiye bir coğrafyadır." → "Türkiye bir coğrafya dır. ) modeli enerj ül 2020 değişkeni nedenle, ol"
"Bilim ve teknoloji gelişiyor." → "Bilim ve teknoloji geliş iyor. inc başına gayr birbirinden hedef hayat"
```

### 9. HAK TALEPLERİ

#### Talep 1 (Sistem)
Backpropagation'suz nokta bulut geometrisi ile hibrit dil modeli eğitim sistemi, içeren:
- (a) Bir dizi QCloudWeight nesnesi, her biri bir token'ın D-boyutlu uzaydaki konumunu temsil eden;
- (b) Bir nokta bulut kodlayıcı (cloud encoder), söz konusu QCloudWeight noktalarını alarak bir durum vektörü üreten;
- (c) Bir QUANN aralık daraltma motoru, her QCloudWeight'in [low, high] aralığını ödül/ceza sinyalleriyle güncelleyen;
- (d) Bir sigma adaptasyonu mekanizması, arama belirsizliğini başarı oranına göre ayarlayan;
- (e) Bir yakınsama kontrolü, aralık genişliği eşik altındığında token'ın "öğrenmiş" olarak işaretleyen.

#### Talep 2 (Yöntem)
Backpropagation'suz dil modeli eğitimi yöntemi, içeren:
- (a) Her token için D-boyutlu bir nokta tanımlama;
- (b) Nokta bulutu bir cloud encoder ile kodlama;
- (c) Kodlanmış durumdan next-token tahmini;
- (d) Tahmin doğruluğuna göre her nokta için ödül/ceza sinyali üretme;
- (e) Ödül: Noktanın aralığını doğru yönde genişletme, sigma küçültme;
- (f) Ceza: Noktanın aralığını yanlış yönde daraltma, yön çevirme, sigma büyütme;
- (g) Aralık genişliği eşik altındığında yakınsama ilan etme.

#### Talep 3 (Hibrit Eğitim)
Hibrit dil modeli eğitimi yöntemi, içeren:
- (a) Bir transformer mimarisini standart backprop ile eğitme;
- (b) Aynı modelin token nokta bulutunu QUANN aralık daraltma ile eğitme;
- (c) Her epoch sonunda: backprop adımı + QUANN nokta daraltma adımı;
- (d) Eğitim sonucunda: hem düşük PPL hem de yorumlanabilir nokta geometrisi.

#### Talep 4 (Yorumlanabilirlik)
Dil modeli yorumlanabilirliği yöntemi, içeren:
- (a) Her token'ın D-boyutlu uzaydaki konumunu takip etme;
- (b) Token'ların buluttaki kümelenme kalıplarını görselleştirme;
- (c) Benzer anlamlı token'ların bulutta yakın konumlarda olması;
- (d) Modelin "neden" belirli bir token ürettiğini geometrik mesafe ile açıklama.

#### Talep 5 (Stokastik QUANN)
Stokastik aralık daraltma yöntemi, içeren:
- (a) Her adımda toplam parametrelerin rastgele bir alt kümesini (K tanesi) seçme;
- (b) Seçilen parametreleri sırayla daraltma;
- (c) Mini-batch üzerinde loss ölçme (gürültü azaltma);
- (d) Plato tespiti: pencerede anlamlı düşüş yoksa sigma genişletme.

### 10. ÇİZİMLER

**Çizim 1:** QCloudNetwork mimarisi (yukarıdaki ASCII diagram)

**Çizim 2:** QUANN aralık daraltma döngüsü
```
     ┌──────────────────────────────────────────┐
     │           QUANN ADIM DÖNGÜSÜ             │
     └──────────────────────────────────────────┘
     
     [K rastgele parametre seç]
              │
              ▼
     ┌─────────────────┐
     │ Aday üret:      │
     │ cand = cur +    │
     │ dir × σ × width │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │ Forward pass    │
     │ (1 forward)     │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │  Karar:         │
     │  err < base-ε?  │──EVET──→ [ÖDÜL: kabul, σ↓]
     │  err > base+ε?  │──EVET──→ [CEZA: geri al, yön↑, aralık↓, σ↑]
     │  else           │──EVET──→ [NÖTR: geri al, yön rastgele]
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │ Plato kontrolü  │
     │ (pencere)       │──Plato──→ [σ × 1.5]
     └─────────────────┘
```

**Çizim 3:** Hibrit eğitim akışı
```
     ┌──────────────────────────────────────────┐
     │         HİBRİT EĞİTİM (her epoch)       │
     └──────────────────────────────────────────┘
     
     ┌──────────────────┐     ┌──────────────────┐
     │  BACKPROP        │     │  QUANN           │
     │  (Transformer)   │     │  (Nokta Bulut)   │
     │                  │     │                  │
     │  • Tüm param     │     │  • Token weights │
     │  • Gradient      │     │  • Aralık daralt │
     │  • AdamW         │     │  • Sigma adapt   │
     │  • HIZLI         │     │  • YORUMLANABİL  │
     └────────┬─────────┘     └────────┬─────────┘
              │                        │
              └───────────┬────────────┘
                          ▼
                 ┌─────────────────┐
                 │  PPL + Geometri│
                 │  (her ikisi de │
                 │   iyileşir)    │
                 └─────────────────┘
```

### 11. BULUŞUN AVANTAJLARI

| Özellik | Backprop | QUANN-Cloud |
|---------|----------|-------------|
| Gradyan | ✅ Var | ❌ Yok (geometri katmanı) |
| Yorumlanabilirlik | ❌ Black-box | ✅ Nokta geometrisi |
| Hız | ✅ Hızlı | ✅ Hibrit (mimari BP) |
| Credit assignment | Chain rule | Token-level reward |
| Search space | Weight space | Point cloud geometry |
| Edge suitability | ⚠️ | ✅ (D boyut küçük) |
| Memory | KV cache | D boyut nokta |
| Yakınsama garantisi | ❌ | ⚠️ (aralık daralır) |

### 12. ENDÜSTRİYEL UYGULAMA

Bu buluş aşağıdaki alanlarda uygulanabilir:
1. **Edge AI:** Küçük D boyut → düşük bellek, mobil cihazlarda çalışabilir
2. **Yorumlanabilir AI:** Nokta bulut görselleştirme ile model kararları açıklanabilir
3. **Incremental learning:** Yeni token'lar buluta eklenir, mevcut noktalar bozulmaz
4. **Multimodal:** Görsel/ses token'ları da aynı nokta bulutunda temsil edilebilir

### 13. Sınırlamalar

1. **Öğrenme hızı:** QUANN, backprop'tan 10-100x yavaş (büyük modellerde)
2. **Yakınsama:** Aralık daralması garantili, ama global optimum garantisi yok
3. **Ölçek:** 100K+ parametreli modellerde saf QUANN pratik değil (hibrit gerekli)

### 14. KAYNAKLAR

- CloudLM: Nokta bulut tabanlı dil modeli (mimari)
- QUANN v0: Aralık daraltma + sigma adaptasyonu
- Dust (Q-Labs): Backprop'suz eğitim (dış referans)
- BellaTurca: Türkçe corpus (deney)

---

*Bu belge bir patent başvurusu taslağıdır. Resmi başvuru için hukuki danışmanlık gereklidir.*
