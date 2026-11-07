# ContextCloudLM — Bağlamsal Nokta-Bulut Mimarisi

**Durum:** Planlama  
**Tarih:** 2026-11-07  
**Amaç:** Uzun metinlerde cümle-seviyesi ve pencere-seviyesi bağlamı birleştiren 2-katmanlı nokta-bulut geometrisi

---

## 1. Mevcut Durum Analizi

### CloudLM (`cloudlm/model.py`)
- Transformer + nokta-bulut hibrit
- Her cümle → 16D nokta bulutu
- Pair geometry: O(N²) mesafe + açı

### PureCloudLM (`purecloudlm/model.py`)
- Transformer-free, O(N) global geometry
- Sadece küme istatistikleri (mean, max, min, std)
- İsteğe bağlı sampled pair approximation

### Eksiklik
- Her iki model de **tek cümle** seviyesinde çalışıyor
- Uzun metinlerde **bağlam akışı** yok
- Cümleler arası ilişki geometrisi yok

---

## 2. Hedef Mimari: ContextCloudLM

### 2.1 İki-Katmanlı Yapı

```
[Metin: 50 cümle]
    │
    ├──→ Cümle Seviyesi (16D)
    │     C1 → [16D x N₁ nokta]
    │     C2 → [16D x N₂ nokta]
    │     ...
    │
    └──→ Pencere Seviyesi (32D)
          W1 (C1-C5) → [32D x 5 özet vektör]
          W2 (C3-C7) → [32D x 5 özet vektör]
          W3 (C5-C9) → [32D x 5 özet vektör]
```

### 2.2 Katmanlar

#### Katman 1: Cümle-Bulut (Mevcut purecloudlm)
```python
# Her cümle için
tokens → embedding → WordToND(16D) → GlobalGeometryEncoder
→ cloud_state_sentence (512D)
```

#### Katman 2: Pencere-Bulut (Yeni)
```python
# Komşu cümleleri birleştir
sentence_states[window_size] → WindowEncoder(32D)
→ cloud_state_window (512D)
```

#### Katman 3: Pencere Geometrisi (Yeni)
```python
# Pencere çiftleri arası
window_i vs window_j:
  - Mesafe: ||centroid_i - centroid_j||
  - Açı: cos(centroid_i, centroid_j)
  - Variance farkı
→ window_geometry_features
```

---

## 3. Teknik Tasarım

### 3.1 Yeni Sınıflar

#### `SentenceBulutEncoder`
```python
class SentenceBulutEncoder(nn.Module):
    """Tek bir cümleyi nokta-buluta dönüştürür (purecloudlm tabanlı)"""
    def __init__(self, vocab_size, d_model=256, d_cloud_dim=16, d_cloud=512):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.to_nd = WordToND(d_model, d_cloud_dim)
        self.sentence_encoder = GlobalGeometryEncoder(
            d_cloud_dim, d_cloud, pool_mode="all"
        )
    
    def forward(self, tokens, mask):
        # (B, N) → (B, d_cloud)
        x = self.embedding(tokens)
        points = self.to_nd(x)  # (B, N, 16)
        return self.sentence_encoder(points, mask)  # (B, 512)
```

#### `WindowBulutEncoder`
```python
class WindowBulutEncoder(nn.Module):
    """Komşu cümleleri pencere olarak birleştirir"""
    def __init__(self, sentence_dim=512, window_size=5, d_window_dim=32):
        self.window_size = window_size
        # Cümle vektörlerini pencere boyutuna projekte et
        self.sentence_proj = nn.Linear(sentence_dim, d_window_dim)
        self.window_encoder = GlobalGeometryEncoder(
            d_window_dim, d_cloud=512, pool_mode="all"
        )
    
    def forward(self, sentence_states, mask=None):
        """
        sentence_states: (B, NumSentences, 512)
        Returns: (B, NumWindows, 512)
        """
        # Kayan pencere ile örnek oluştur
        windowed = sliding_window(sentence_states, size=window_size, stride=2)
        # Her pencereyi projekte et
        projected = self.sentence_proj(windowed)  # (B, NumWindows, 5, 32)
        return self.window_encoder(projected, mask)  # (B, NumWindows, 512)
```

#### `ContextCloudLM`
```python
class ContextCloudLM(nn.Module):
    """2-katmanlı bağlamsal nokta-bulut LM"""
    def __init__(self, vocab_size, 
                 d_sentence=512, d_window=512,
                 window_size=5, window_stride=2):
        self.sentence_encoder = SentenceBulutEncoder(vocab_size)
        self.window_encoder = WindowBulutEncoder(d_sentence, window_size)
        
        # Final prediction: hem cümle hem pencere bağlamı kullan
        self.combined_head = nn.Sequential(
            nn.Linear(d_sentence + d_window, 1024), nn.GELU(),
            nn.Linear(1024, vocab_size)
        )
    
    def forward(self, tokens, sentence_mask=None, window_mask=None):
        # 1. Cümle seviyesi
        sentence_states = self.sentence_encoder(tokens, sentence_mask)
        
        # 2. Pencere seviyesi
        window_states = self.window_encoder(sentence_states)
        
        # 3. Birleştirilmiş tahmin
        combined = torch.cat([sentence_states, window_states.mean(dim=1)], dim=-1)
        return self.combined_head(combined)
```

### 3.2 Kayan Pencere Mekanizması

```python
def sliding_window(tensor, size=5, stride=2):
    """
    Tensor: (B, N, D)
    Returns: (B, num_windows, size, D)
    
    Örnek: N=10, size=5, stride=2
      W1: [0,1,2,3,4]
      W2: [2,3,4,5,6]
      W3: [4,5,6,7,8]
      W4: [6,7,8,9]
    """
    num_windows = max(1, (tensor.size(1) - size) // stride + 1)
    windows = []
    for i in range(num_windows):
        start = i * stride
        end = start + size
        windows.append(tensor[:, start:end, :])
    return torch.stack(windows, dim=1)
```

### 3.3 Pencere Geometrisi

```python
class WindowGeometryEncoder(nn.Module):
    """Pencereler arası geometriyi hesaplar"""
    def __init__(self, window_dim=512):
        # Pencere centroid'leri
        self.centroid_mlp = nn.Linear(window_dim, 128)
        # Pencere çiftleri arası ilişkiler
        self.pair_mlp = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.GELU(),
            nn.Linear(256, 512)
        )
    
    def forward(self, window_states):
        """
        window_states: (B, NumWindows, 512)
        Returns: (B, 512) - global metin durumu
        """
        # Her pencerenin centroid'ini hesapla
        centroids = self.centroid_mlp(window_states)  # (B, NumWindows, 128)
        
        # Pencere çiftleri arası mesafe + açı
        diff = centroids.unsqueeze(2) - centroids.unsqueeze(1)  # (B, N, N, 128)
        dist = diff.norm(dim=-1)  # (B, N, N)
        cos_ang = cosine_similarity(centroids, dim=-1)  # (B, N, N)
        
        # Pair features
        pair_in = torch.cat([diff, dist.unsqueeze(-1), cos_ang.unsqueeze(-1)], dim=-1)
        pair_feats = self.pair_mlp(pair_in)  # (B, N, N, 512)
        
        # Global pooling over pairs
        return pair_feats.mean(dim=(1, 2))  # (B, 512)
```

---

## 4. Eğitim Stratejisi

### 4.1 Veri Hazırlığı

```python
class ContextDataset(Dataset):
    """Uzun metinleri cümle + pencere formatında hazırlar"""
    def __init__(self, texts, sentence_tokenizer, window_size=5, window_stride=2):
        self.window_size = window_size
        self.stride = stride
        self.sentences = [sentence_tokenizer.sentences(t) for t in texts]
    
    def __getitem__(self, idx):
        sent_tokens = [tokenizer(s) for s in self.sentences[idx]]
        
        # Cümle seviyesi
        sentence_tokens = torch.stack(sent_tokens)
        
        # Pencere seviyesi
        windows = sliding_window(sent_tokens, self.window_size, self.stride)
        
        return {
            'sentence_tokens': sentence_tokens,
            'window_tokens': windows,
            'target': next_sentence
        }
```

### 4.2 Loss Fonksiyonu

```python
def context_loss(sentence_logits, window_logits, target, 
                 sentence_weight=0.5, window_weight=0.5):
    """
    Hem cümle hem pencere seviyesinde tahmin yap
    """
    sentence_loss = F.cross_entropy(sentence_logits, target)
    window_loss = F.cross_entropy(window_logits, target)
    
    return sentence_weight * sentence_loss + window_weight * window_loss
```

### 4.3 Hyperparametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `window_size` | 5 | Pencere boyutu (kaç cümle) |
| `window_stride` | 2 | Pencere kayma adımı |
| `d_sentence` | 512 | Cümle vektör boyutu |
| `d_window` | 512 | Pencere vektör boyutu |
| `sentence_weight` | 0.5 | Cümle loss ağırlığı |
| `window_weight` | 0.5 | Pencere loss ağırlığı |

---

## 5. Uygulama Adımları

### Adım 1: Temel Yapı
- [ ] `SentenceBulutEncoder` sınıfı (purecloudlm'den türetilmiş)
- [ ] `WindowBulutEncoder` sınıfı
- [ ] `sliding_window` utility fonksiyonu

### Adım 2: Geometri
- [ ] `WindowGeometryEncoder` sınıfı
- [ ] Pencere çiftleri arası mesafe + açı hesaplama

### Adım 3: Birleştirme
- [ ] `ContextCloudLM` ana model
- [ ] Combined head (sentence + window)

### Adım 4: Eğitim
- [ ] `ContextDataset` DataLoader
- [ ] Multi-level loss fonksiyonu
- [ ] Training script (`context_train.py`)

### Adım 5: Değerlendirme
- [ ] Cümle-seviyesi PPL
- [ ] Pencere-seviyesi PPL
- [ ] Uzun metin (≥500 kelime) test seti

---

## 6. Beklenen Sonuçlar

### 6.1 Performans
- **Cümle-seviyesi PPL:** purecloudlm ile benzer
- **Uzun metin PPL:** %10-15 iyileşme (bağlam sayesinde)
- **Hız:** %15-20 yavaşlama (ek pencere işlemesi)

### 6.2 Geometri Analizi
- Pencere centroid'leri arasında anlamlı mesafe pattern'leri
- Tema geçişlerinde açı değişiklikleri
- Uzun menzilli bağımlılıkların geometrik izlenebilirliği

---

## 7. Riskler ve Azaltma

| Risk | Etki | Azaltma |
|------|------|---------|
| Bellek artışı | Yüksek | Incremental training ile test |
| Hiperparametre hassasiyeti | Orta | Grid search (window_size: 3,5,7) |
| Overfitting | Orta | Early stopping + validation set |
| Training yavaşlığı | Yüksek | Batch size azalt, gradient accumulation kullan |

---

## 8. Başarı Kriterleri

1. **Teknik:** Model başarıyla eğitilir (loss düşer)
2. **Performans:** Uzun metin PPL, purecloudlm'den %5+ daha iyi
3. **Geometri:** Pencere centroid'leri anlamlı cluster'lar oluşturur
4. **Gerçekçilik:** Mevcut purecloudlm checkpoint'lardan transfer learning mümkün

---

## 9. Sonraki Adımlar

1. Plan onaylandıktan sonra kodlamaya başla
2. Önce küçük veri seti (10K cümle) ile prototip
3. Başarılı olursa tam corpus (1M+ cümle) ile genişlet
4. CloudLM ve purecloudlm ile karşıtırmalı experiment

---

**Onay:** Bu plan, purecloudlm ve CloudLM mimarilerini koruyarak yeni bağlam katmanı ekler.  
**Tarih:** 2026-11-07
