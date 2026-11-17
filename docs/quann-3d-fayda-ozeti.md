# Quann-3D: Fayda ve Kullanım Alanları Özeti

## Proje Özeti

Quann-3D, Transformer'ın "sıralı dikkat + tek vektör" paradigmasına alternatif olarak:
- **Kelime** → ND uzayda nokta
- **Cümle** → Nokta bulutu
- **Anlam** → Bulut geometrisi (mesafe, açı, küme formu)
- **Tahmin** → Bulut konfigürasyonundan sonraki kelime

İkinci katman **QUANN-Cloud**: Aynı mimariyi backprop'suz (aralık-daraltma + ödül/ceza) eğiten 0th-order yöntem.

---

## 1. Edge Deployment (Düşük Kaynak Ortamlar)

### Performans Metrikleri
| Özellik | Değer |
|---|---|
| Model Boyutu | 21M parametre |
| INT8 Paketi | 21MB (84MB → 21MB) |
| Inference Hızı | ~45.5K tx/s |
| Fraud ROC-AUC (FP32) | 0.9451 |
| Fraud ROC-AUC (INT8) | 0.9367 |
| INT8 Kaybı | -0.0084 |

### Fayda
- Büyük dil modellerinden (70B+ parametre) çok daha küçük boyut
- Üretim ortamında düşük bellek ve enerji gereksinimi
- Quantization kaybı minimal (%1'den az)

---

## 2. Açıklanabilirlik (Explainable AI)

### Native Açıklama Mekanizması
- **SHAP gerektirmez**: Nokta bulutu geometrisi doğrudan açıklama sağlar
- **False Alarm Audit**: Hangi token'ların flag'e yol açtığını gösterir
- **Counterfactual Analiz**: "Eğer bu token olmasaydı" simülasyonu

### Örnek Bulgular
- Fraud-koruyucu tokenlar: Merchant City=other, zip range, Merchant State=CA
- Pozitif sürücüler: Merchant Name kimliği, amount≈[30.79,46.26), saat=16
- False alarm olasılığı ~0.999, popülasyonu generic online tokenlar sürüyor

---

## 3. Sözdizimsel Yapı Yakalama

### Geometrik Bölgeleşme
| Boyut | Bulgular |
|---|---|
| 3D | 1 baskın ilişki (yer-fiil veya özne-fiil) |
| 7D | Hem özne-fiil hem yer-fiil ayrışır |
| 16D | Sözdizimsel roller uzayda bölgeleşir |

### Avantajlar
- **Pozitional encoding gerektirmez**: Kelime sırası önemli değil
- **Küme olarak anlam**: Noktaların birlikte duruşu anlamı taşır
- **Coğrafi yapı**: Edatlar merkez, belirsizlik batı gibi yapısal düzen

---

## 4. Küçük Veri ile Hızlı Adaptasyon

### QUANN-Cloud (Backprop'suz Eğitim)
- **XOR'da %100 başarı**: 0th-order yöntemle çözüldü
- **Gerçek LLM'de**: Random'a yakın (sınır bilinci)
- **Savunulabilir iddia**: Küçük/yorumlanabilir modeller için backprop gerektirmeyen alternatif

### Kullanım Senaryoları
- Eğitim verisi kısıtlı ortamlar
- Hesaplama kaynağı sınırlı edge cihazlar
- Hızlı prototipleme ve A/B testleri

---

## 5. Çok Dikey Kullanım Alanı

| Dikey | Fayda | Kanıt |
|---|---|---|
| **Fraud** | Edge deployment, açıklanabilirlik | ROC .9451, INT8 .9367 |
| **AML** | Tipoloji haritalama, bölge stability | ARI 0.994, Spearman 1.0 |
| **Endüstriyel Bakım** | Sensör bulutu değişim tespiti | C-MAPSS/CWRU testleri |

### AML Tipolojisi
- Region stability: ARI mean 0.994
- Product mix değişimi: Region 7 +21%, Region 1 -17%
- 3.77M usable kayıt üzerinde ölçüm

---

## 6. Geometrik Temsil Öğrenme

### Optimum Boyut
- **D=64 optimum**: daha büyük boyut tek başına fayda sağlamıyor
  - büyük boyut koşusu PPL 58.55 ölçtü — `logs/run_r2_cloud_attn_200k_d128.log:28`
  - daha büyük boyut koşusu PPL 58.27 ölçtü — `logs/run_r2_cloud_attn_200k_d256.log:28`
- **Bağlam penceresi (nokta = token)**: kontrollü ablation değil, gözlemdir
  - dar pencere koşusu PPL 57.04 ölçtü — `logs/run_r2_cloud_attn_200k_d64.log:26`
  - geniş pencere koşusu PPL 37.63 ölçtü — `logs/run_r2_cloud_attn_200k_d64_n20.log:26`
  - en geniş pencere koşusu PPL 134.31 ölçtü — `logs/run_r2_cloud_attn_200k_d64_n30.log:26`
  - eğitim cümlesi koşullar arasında farklı — `logs/run_r2_cloud_attn_200k_d64.log:9`
- **Kapasite sınırı**: Bulut (10×32=320 sayı), transformer gizli durumunun (10×256=2560) %12.5'ini taşıyor

### Performans (R2 D=64, pencere=20)
| Veri Seti | PPL |
|---|---|
| 200K | 37.63 (`logs/run_r2_cloud_attn_200k_d64_n20.log:28`) |
| 1.5M (100 epoch) | 15.60 |
| 1.5M (300 epoch) | 12.43 (train), 21.91 (test) |

---

## Ticari Değer Önermesi

### Temel Farklar
| Özellik | Transformer | Quann-3D |
|---|---|---|
| Temsil | Tek dev vektör (768D) | Nokta bulutu (N×D) |
| Anlam | Vektörün konumu | Noktaların birlikte duruşu |
| Sıralama | Kritik (positional encoding) | Yok — küme olarak |
| Dikkat | Q·K·V matris | Mesafe + açı (geometri) |
| Açıklanabilirlik | SHAP/LIME ek katman | Native geometri |
| Edge Boyutu | 100MB-GB | 21MB |

### Ticari Değer
1. **Denetlenebilir tahmin**: Neden fraud? Cevap geometride
2. **Edge boyutu**: Mobil/embedded'de çalışabilir
3. **Retrain'siz uyum**: Yeni tipolojiye hızlı adaptasyon

---

## Mevcut Durum

### En İyi Performans (Mevcut Best)
| Model | Config | Param | PPL | Checkpoint |
|---|---|---|---|---|
| CloudLM | D=32, 200K, 50ep | 21.0M | 6.45 | wiki_checkpoints_200k |
| Causal baseline | d=384, 4L, 200K, 50ep | 22.7M | 2.92 | baseline_200k |

Kaynak:
- CloudLM PPL 6.45 ölçtü — `logs/run_final_pipeline.log:45`
- Baseline PPL 2.92 ölçtü — `logs/run_final_pipeline.log:46`
- Erken bir koşu baseline'ı PPL 2.94 ölçmüştür — `logs/run_compare.log:18`; bu değer yukarıdaki baseline değerinden ayrı
  bir koşuya aittir, tek koşunun iki değeri değildir.
- Protokol uyarısı: `quann_cloud/REVISION.md:37` — 6.45 referansı `data/turkish_sentences.txt` üzerinde,
  karşılaştırmanın geniş boyut kolu wiki üzerinde eğitilmiş (korpus OOD artefaktı); bu yüzden değer strict held-out
  kanıtı olarak sunulmaz.

### Aktif Hipotezler
1. **R1**: Kapasite (D=64/128) — Holdout'ta eşit (Δ0.02)
2. **R2**: Cloud attention + derinlik — Tamamlandı
3. **R2b**: Bağlam penceresi 10→20 → PPL 57.04→37.63 (Δ19.41 = %34,0) — gözlem, kontrollü ablation değil (§6)
4. **R3**: Attention-weighted pooling — Max-pool daha stabil
5. **R4**: QUANN aramayı pair-kanalına yönlendir — Bekliyor

---

## Sonuç

Quann-3D, ham doğrulukta SOTA iddiası olmasa da:
- **Edge deployment** için optimize edilmiş küçük model
- **Native açıklanabilirlik** ile regülasyon uyumu
- **Çok dikey** kullanım (fraud, AML, bakım)
- **Geometrik temsil** ile sözdizimsel yapı yakalama

Ticari değeri: **açıklanabilirlik + uyum hızı + boyut**, ham doğruluk değil.

---

*Kaynak: RESEARCH.md, AGENTS.md, fraud/ results/*
