# QUANN + CloudLM: İki Sistem, Bir Proje
### Dış Referans: Dust (Q-Labs) — Sahibi Bizim Değil

## 📜 V0 QUANN'ın 3 Sistemi (2026-09-20)

### 1. QNetwork — Aralık Daraltma (Interval Narrowing)

```
Her ağırlık → [0, 1] aralığında rastgele değer → hata ölç → ödül/ceza → aralık daralt
```

**Mekanizma:**
- Her `QWeight` bir `low/high` aralığı tutar (başlangıç: `[0.0, 1.0]`)
- `sample()` → aralıktan rastgele değer üret
- `narrow(new_value, reward=True)` → **ödül**: doğru yönde, aralık genişlet
- `narrow(new_value, reward=False)` → **ceza**: yanlış yönde, aralık daralt
- **Bidirectional mod**: yanlış yöndeysen, diğer ucu dene (karşıt yön)

**Öğrenme tanımı:** Aralık `convergence_threshold` altına düştü → ağırlık "öğrenmiş"
**Credit assignment:** İki forward pass (±perturbasyon) ile tahmini gradyan işareti

### 2. QNetworkBisect — Deterministik İkiye Bölme

```
Her ağırlık → sınıra kalan mesafenin TAM ORTASI → hata ölç → ödül/ceza
```

**Fark (V1 vs Bisect):**
- V1: `random.uniform(low, high)` — RASTGELE iç noktadan dene
- Bisect: `y + direction * (sınır-y)/2` — **Deterministik tam orta nokta**

**Mekanizma:**
- `reward`: hata düştü → aynı yönde devam (değer adaya taşınır)
- `censure`: hata arttı → **yön ters çevrilir**
- `sigma_adaptation`: Başarı penceresinden σ uyarlanır (başarısızsa büyür, başarılıysa küçülür)
- `censure_push`: Ceza durumunda denenen adayın **TERS yönüne itme** (2.7x iyi)
- `stuck`: adım sıfıra yaklaştığında plato → yön yeniden başlat (restart)

### 3. QUANN-NLP — Cümle Seviyesi Tahmin

```
Cümle → 3B küme uzayı → QUANN → Sonraki cümle (tek adımda)
```

**Sonuçlar (44 eğitim, 27 out-of-domain test):**
- QUANN cümle-seviyesi: **%100** test doğruluk (kalıbı geneliledi)
- Token-n-gram: **%0** test doğruluk (görmediği kelimeler)
- QUANN 5/5 tohumda kazandı.

**Avantajı:** Küme yapısını (kalıbı) yakalar, gördüğü kelimelerle çalışır. Backprop yok.

---

## 🎯 Sistemlerimiz (Sizin 2 Projeleriniz)

### QUANN (v0) — Backprop'suz Öğrenme
- **Soru:** "Backprop olmadan ağ nasıl öğrenir?"
- **Cevap:** Aralık daraltma + ödül/ceza + sigma adaptasyonu
- **Durum:** XOR, Tiny MNIST prototipler hazır. 3 farklı algoritma:
  1. QNetwork (aralık daraltma, rastgele deneme, bidirectional mod)
  2. QNetworkBisect (deterministik ikiye bölme, sigma adaptasyon)
  3. QUANN-NLP (cümle-seviyesi, %100 out-of-domain generalileme)

### CloudLM — Nokta Bulut Mimari
- **Soru:** "Sekans + tek vektör paradigmadan çıkınca ne olur?"
- **Cevap:** Cümle → nokta bulutu → geometri ile tahmin
- **Credit assignment:** Backprop (standart)
- **Durum:** PPL ~4-5, transformer'ın gerinde ama yorumlanabilirlik üstün.

### Dust — Dış Referans (Q-Labs)
- **Sahibi:** QUANN Research Team DEĞİL
- **Soru:** "Backprop olmadan transformer eğitilir mi?"
- **Cevap:** Evet — activation-space perturbation (zeroth-order)
- **Not:** Sizin proje değil. Sadece "backprop'suz eğitim" konusunda
  karşılaştırma referansı olarak kullanılıyor.

---

## 🔗 İlişki Haritası (Sizin 2 Sisteminiz + Dış Referanslar)

### QUANN-NLP → CloudLM (Mimari Köprü)

| | QUANN-NLP | CloudLM |
|---|---|---|
| **Temel fikir** | Cümle → küme dağılım vektörü | Cümle → nokta bulutu |
| **Tekil temsil** | ❌ Vektör | ❌ Nokta bulutu (N×D) |
| **Sekans** | Yok (cümle bütünü) | Yok (nokta kümesi) |
| **Yorumlanabilirlik** | Küme dağılımı | Mesafe, açı, bölge |

**Evrim:** QUANN'ın "tek vektör → cümle temsili" fikri → CloudLM'in "tek vektör → nokta bulutu" fikri.
Yapısal olarak benzer fikir (sekans yok, bütünsel temsil), ama CloudLM nokta bulutu ile daha zengin bir temsil sunar.

### QNetwork → Dust (Backprop'suz Eğitim — Dış Referans)

| | QUANN V0 | Dust |
|---|---|---|
| **Backprop** | ❌ Hayır | ❌ Hayır |
| **Ne sapanır?** | Ağırlık aralığı → reward/ceza | Aktivasyon → token-level reward |
| **Credit assignment** | İki forward pass | Population token-level |
| **Arama uzayı** | Parametre space (ağırlık değerleri) | Activation space (katman çıktısı) |
| **Adaptasyon** | Sigma uyarlaması (başarı penceresi) | Population scaling (daha büyük = daha iyi) |

**Evrim:** QUANN'ın "parametre space'de aralık daraltma" → Dust'ın "activation space'de token-level perturbation".
Credit assignment kalitesi yükseldi. Dust'ın popülasyon yaklaşımı, QUANN'ın tek-parametre denemesinden çok daha güçlü.

### QNetworkBisect → Dust (Deterministik → Stochastic — Dış Referans)

| | QUANN Bisect | Dust |
|---|---|---|
| **Aday seçimi** | Deterministik tam orta nokta | Rastgele Gaussian perturbation |
| **Ödül** | Aynı yönde devam, σ küçül | Reward-weighted noise |
| **Ceza** | Yön ters çevir + itme (push) | Noise-weighted outer product |
| **Adaptasyon** | Sigma başarı penceresi | Population büyüklüğü |

QUANN Bisect'in **deterministik** yaklaşımı ile Dust'ın **stochastic** yaklaşımı farklı uçlar.
Ama her ikisi de "search-based learning without backprop" fikrini paylaşıyor.

---

## 🏆 6 Paralel Yaklaşım Karşılaştırması

| Yaklaşım | Backprop | Gradient | Ödül/CE | Aralık Daralt | Geometri |
|---|---|---|---|---|---|
| **QUANN NLP** | ❌ | ❌ | ✅ | ❌ | ✅ (3B küme) |
| **QUANN Interval** | ❌ | ❌ | ✅ | ✅ | ❌ |
| **QUANN Bisect** | ❌ | ❌ | ✅ | ❌ (deterministik) | ❌ |
| **CloudLM** | ✅ | ✅ | ❌ | ❌ | ✅ (nokta bulutu) |
| **Dust** | ❌ | ❌ (zeroth) | ✅ | ❌ | ❌ |
| **Standart BP** | ✅ | ✅ | ❌ | ❌ | ❌ |

---

## 💡 Öneri: Üçlü Birleştirme

QUANN'ın V0 sistemi ile bugünkü CloudLM + Dust'ı birleştirmek mümkün:

### "QUANN-Cloud: Backprop'suz Nokta Bulut Modeli"

1. **Mimari:** CloudLM (nokta bulut) — sekans yok, geometri var — **sizin**
2. **Eğitim:** QUANN'ın aralık daraltma + sigma adaptasyonu — backprop yok, **sizin**
3. **Dust (dış referans):** Backprop'suz eğitim fikrinden ilham alır, **sizin değil**

**Sizin 2 sisteminiz birleşirse:** Sekans yok + geometri var + backprop yok + arama-based learning.
**Dünyada bir ilk:** Backprop'suz, nokta-bulut-tabanlı dil modeli.

Dust'ın katkıları burada sadece "backprop'suz eğitim" fikri konusunda referans;
asıl çalışma tamamen QUANN + CloudLM ile sizin projeniz olur.
