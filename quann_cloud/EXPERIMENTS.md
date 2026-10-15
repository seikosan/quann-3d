# QUANN-Cloud: Deney Planı

## 📋 Genel Bilgiler

- **Korpus:** data/turkish_sentences.txt (199K cümle)
- **Tokenizer:** BPE (20K vocab)
- **Model:** d_model=256, D=16, 4 katman, ~21M parametre
- **Batch size:** 64
- **Optimizer:** Backprop YOK, reward/ceza ile öğrenme
- **Sequance length:** 10 token

## 🧪 Faz 1: XOR Gate Doğrulama

### Amaç
Mekanizmanın temel çalışmasını doğrula.

### Yapı
- **Input:** 2 boyutlu (her giriş bir D=16 nokta bulutu)
- **Hidden:** 4 node (her node D=16 nokta bulutu)
- **Output:** 1 node (D=16 nokta bulutu)
- **Toplam parametre:** ~200 (her QCloudWeight ~2×16×2 = 64 değer)

### Beklenen Sonuç
- XOR problemi backprop olmadan çözülür
- Yakınsama: 100-500 epoch
- Son aralık genişliği: <0.01

### Metrikler
- Yakınsama adı sayısı
- Son aralık genişliği (her boyut için ayrı)
- Doğruluk (%)

## 🧪 Faz 2: Tiny Corpus (20K cümle)

### Amaç
Küçük korpus üzerinde test.

### Yapı
- **d_model:** 256
- **d_cloud:** 512
- **D:** 16
- **n_layers:** 4
- **Toplam parametre:** ~21M

### Eğitim Protocolü
1. **Corpus:** İlk 20K cümle
2. **Tokenizer:** 20K BPE vocab
3. **Epoch:** 10
4. **Batch:** 64
5. **Learning rate:** N/A (backprop yok)
6. **Expansion rate:** 0.1 (her boyut için)
7. **Convergence threshold:** 0.01 (her boyut için)

### Beklenen Sonuç
- **PPL:** ~10-20 (CloudLM'in backprop ile ~4-5'ine yakın)
- **Generation:** Gerçek Türkçe cümleler üretebilir
- **Yakınsama:** 5-10 epoch

### Metrikler
- PPL (held-out)
- Generation quality (exact match, word Jaccard)
- Yakınsama süresi (epoch cinsinden)

### Karşılaştırma
| Model | Params | PPL (20K) | Yakınsama |
|-------|--------|-----------|-----------|
| QUANN-Cloud | 21M | ~10-20 | 5-10 epoch |
| CloudLM (backprop) | 21M | ~4-5 | 15-20 epoch |
| Random (untrained) | 21M | 20,000 | — |

## 🧪 Faz 3: Full Corpus (200K cümle)

### Amaç
Büyük korpus üzerinde test.

### Yapı
- **d_model:** 256
- **d_cloud:** 512
- **D:** 16/32
- **n_layers:** 4
- **Toplam parametre:** ~21M

### Eğitim Protocolü
1. **Corpus:** 200K cümle
2. **Tokenizer:** 20K BPE vocab
3. **Epoch:** 20-50
4. **Batch:** 64
5. **Expansion rate:** 0.1 (adaptive)
6. **Convergence threshold:** 0.01 (her boyut için)

### Beklenen Sonuç
- **PPL:** ~5-10 (CloudLM'in backprop ile ~4-5'ine yaklaşabilir)
- **Generation:** Uzun, karmaşık cümleler üretebilir
- **Yakınsama:** 15-25 epoch

### Metrikler
- PPL (held-out, 5K sentence)
- Generation quality (exact match, word Jaccard, F1)
- Yakınsama süresi (epoch + token count)
- Yorumlanabilirlik (3D/ND görselleştirme)

### Karşılaştırma
| Model | Params | PPL (200K) | Yakınsama | Yorumlanabilir |
|-------|--------|------------|-----------|----------------|
| QUANN-Cloud | 21M | ~5-10 | 15-25 epoch | ✅ |
| CloudLM (backprop) | 21M | ~4-5 | 15-20 epoch | ✅ |
| GPT-2 small | 124M | ~15-20 | 100+ epoch | ❌ |

## 📊 Karşılaştırma: Backprop vs QUANN-Cloud

| Metrik | CloudLM (BP) | QUANN-Cloud |
|--------|--------------|-------------|
| **Backprop** | ✅ Var | ❌ Yok |
| **Gradient** | ✅ Var | ❌ Yok |
| **Credit Assignment** | Chain rule | Token-level reward |
| **Search Space** | Weight space | Point cloud geometry |
| **Search Mechanism** | Gradient descent | Reward/censure + narrowing |
| **Learning Speed** | Hızlı (15-20 epoch) | Yavaş (15-25 epoch) |
| **Final PPL** | ~4-5 | ~5-10 (hedef) |
| **Interpretability** | ✅ Geometrik | ✅ Geometrik |
| **Edge Suitability** | ✅ Potansiyel | ✅ Potansiyel |
| **Memory** | KV cache + d_model | D boyutlu point cloud |

## ⚠️ Riskler

1. **Yavaş yakınsama:** Backprop'dan 2-5x yavaş olabilir. Azaltma: adaptive expansion rate.
2. **High-D narrownning:** D=16'da 16 boyut bağımsız narrownning. Azaltma: boyut bazında narrownning.
3. **Corpus ile BPE uyumsuzluğu:** Aynı BPE ile eğit + test. Azaltma: aynı pipeline kullan.
4. **Small corpus generalization:** 20K cümlede overfit. Azaltma: 200K corpus ile test.

## 📝 Notlar

- Bu deneyler **düşük compute** ile yapılabilir (Apple Silicon MPS).
- Her epoch'ta tüm corpus forward pass edilir.
- Backprop olmadığı için optimizer parametreleri (lr, momentum) yok.
- Sadece expansion_rate, convergence_threshold, sigma gibi hyperparametreler var.
