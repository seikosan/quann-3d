# TÜRKİYE CUMHURİYETİ — PATENT BAŞVURU METNİ
## v2.5: Nokta-Bulutu Geometrisi ile Birleşik Dil Modelleme

**Dosya:** `patent_application_v2.5.md`

**Başvuru Tarihi:** [2026]  
**Başvuru No:** [TR 2026/...]  
**İcat Başlığı:** Nokta-Bulutu Geometrisi ile Birleşik Dil Modelleme Sistemi, Yöntemi ve Bilgisayar Programı  
**İcatçı:** [Ad Soyad, adres, T.C. Kimlik]  
**Başvuru Sahibi:** [Unvan, adres]  

---

## İÇİNDEKİLER

1. **Teknik Alan**
2. **Ön Teknik (Prior Art)**
3. **Teknik Sorun**
4. **Buluşun Özü (Çözüm)**
5. **Teknik Etkiler**
6. **Şekillerin Kısa Açıklaması**
7. **Buluşun Detaylı Açıklaması**
8. **İddialar**
9. **Özet (Abstract)**

---

## 1. TEKNIK ALAN

Bu buluş, doğal dil işleme ve makine öğrenmesi sistemlerine ilişkindir; özellikle **düşük boyutlu (3 ≤ D ≤ 64) nokta-bulutu geometrisine** dayanan dil modellerine ilişkindir. Her kelime/alt-sözcük, D-boyutlu Öklid uzayında bir **pᵢ noktasına** projekte edilir; cümle ise bu noktaların oluşturduğu **nokta bulutu** olarak temsil edilir. Tahmin, bu bulutun **geometrisinden** (mesafe + açı + norm) üretilir.

Buluşun temel özelliği, **aynı nokta bulutunun hem hesaplama hem de native açıklama yüzeyi** olarak kullanılmasıdır. Nokta bulutu, transformer'ın "sıralı dikkat + tek vektör" paradigmasının yerine geçer.

---

## 2. ÖN TEKNİK (PRIOR ART)

### 2.1 Transformer ve Self-Attention

**Vaswani ve ark. (2017, NeurIPS 30)** "Attention Is All You Need" çalışması, sıralı girdiyi self-attention mekanizması ile işleyen transformer mimarisini önermiştir. Attention hesabı sekans uzunluğu N bakımından **O(N²)** karmaşıklığa sahiptir ve karar, yüksek boyutlu bağlamsal gösterimlerden (genellikle d_model=512-1024) okunur.

**Chung & Abbott (2021, Nature 592)** nöral popülasyon geometrisi çerçevesini ele almıştır; ancak bu çalışma, geometriyi tahmin substratı olarak kullanan yeni bir dil modeli mimarisini önermez.

### 2.2 Nokta-Kümesi Yöntemleri

- **PointNet (Qi ve ark., 2017, CVPR)** paylaşılan nokta MLP'si ve simetrik küme agregasyonu kullanır; ancak nokta koordinatları esasen ham 3B vizyon/sensör girdisidir.
- **Deep Sets (Zaheer ve ark., 2017, NeurIPS 30)** ve **Set Transformer (Lee ve ark., 2019, ICML)** kümelerden havuzlanmış temsil üretir; fakat öğrenilmiş düşük boyutlu noktaları ve çift-geometriyi birleştiren sistemle aynı değildir.

### 2.3 Kelime Vektörleri

**Word2Vec (Mikolov ve ark., 2013)** ve **GloVe (Pennington ve ark., 2014)** tekil kelime vektörlerini üretir; ancak cümle-level nokta-bulutu geometrisi ve bu geometriden tahmin üreten birleşik sistem önermezler.

### 2.4 Ön Teknikteki Boşluk

İncelenen yaklaşımlarda:
1. Nokta-bulutu geometrisi **tahmin substratı** olarak kullanılmaz
2. Geometri **native açıklama yüzeyi** olarak yeniden kullanılmaz
3. **Çift-geometri** (mesafe + açı + norm) tek bir temsilde birleştirilmez

Bu buluş, bu boşluğu doldurur.

---

## 3. TEKNİK SORUN

Bu buluş aşağıdaki teknik sorunlara çözüm sağlamayı amaçlar:

1. **Yüksek boyutlu opak temsil:** Transformer'ın d_model=256-1024 boyutlu gizli durumları insan yorumlamasına kapalıdır.
2. **Sıralı bağımlılık:** Self-attention mekanizması permütasyon-değildir; anlamın geometrik yapısı doğrudan incelenemez.
3. **Post-hoc açıklama:** SHAP/LIME gibi yöntemler model sonrası yaklaşık atıflar üretir; native açıklama sağlamaz.
4. **Edge dağıtım zorluğu:** Büyük model boyutları (22.7M+ parametre) kompakt edge dağıtımını zorlaştırır.

---

## 4. BULUŞUN ÖZÜ (ÇÖZÜM)

### 4.1 Temel Fikir

Her kelime/alt-sözcük → **D-boyutlu nokta** (3 ≤ D ≤ 64)  
Cümle → **Nokta bulutu** (N nokta)  
Tahmin → **Bulut geometrisi** (mesafe + açı + norm)

### 4.2 Mimari Bileşenler

| Modül | İşlev | Boyut |
|---|---|---|
| **Embedding** | Token → d_model bağlam vektörü | d_model=256 |
| **Transformer Encoder** | Bağlam işleme | 8 katman, 16 head |
| **WordToND** | Bağlam → D-boyutlu nokta | D=32/64 |
| **CloudEncoderND** | Nokta bulutu → küme durumu | 512 |
| **Word Head** | Küme durumu → vocab dağılımı | vocab=20K |
| **Point Head** | Küme durumu → bir sonraki nokta | D |

### 4.3 Çift-Geometri Kodlayıcı

Nokta çiftleri için **D+4 boyutlu vektör** kullanılır:

```
x_ij = [pᵢ - pⱼ, ‖pᵢ - pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖] ∈ ℝ^(D+4)
```

- **Fark vektörü (pᵢ - pⱼ):** Koordinat dönüşümü altında eşdeğişkendir
- **Mesafe (‖pᵢ - pⱼ‖):** Dönüşümde değişmez
- **Kosinüs açısı (cos θᵢⱼ):** Dönüşümde değişmez
- **Normlar (‖pᵢ‖, ‖pⱼ‖):** Dönüşümde değişmez

### 4.4 Simetrik Küme Agregasyonu

**Nokta özellikleri:** max-pool  
**Çift özellikleri:** mean-pool  
**Fusion:** İki pooled vektör birleştirilir → küme durumu

**Permütasyon-değişmezlik:** Nokta sırası tahmini etkilemez (sıralama yok).

---

## 5. TEKNİK ETKİLER

### 5.1 Mimari Avantajlar

1. **Düşük boyutlu temsil:** D=32/64 boyutu, geleneksel transformer'ın d_model=256-1024'e göre %75-94 daha az boyut
2. **Kompakt dağıtım:** INT8 kantizasyon ile ~5-6 MB
3. **Native açıklama:** Post-hoc değil, model içi açıklama
4. **Permütasyon-değişmezlik:** Nokta sırası tahmini etkilemez
5. **Çift-geometri kodlama:** Mesafe + açı + norm bileşenleri ile dönüşüm-eşdeğişken temsil

### 5.2 Edge Dağıtım

- **Model boyutu:** ~21M parametre (D=32) / ~22M (D=64)
- **INT8 kantizasyon:** ~5-6 MB (okuyucu)
- **Torch↔ONNX paritesi:** ~1.5e-6
- **Çıkartım hızı:** ~45K işlem/s

### 5.4 Native Açıklama

Aynı buluttan:
1. **Leave-one-point-out ablation:** Token noktası çıkarıldığında logit değişimi
2. **Bölge/prototip isimlendirme:** En yakın isimli prototip
3. **Bölge kayması/drift:** Zaman içinde bölge dağılımı değişimi

---

## 6. ŞEKİLLERİN KISA AÇIKLAMASI

- **Şekil 1:** Evrensel mimari ve modül akışı
- **Şekil 2:** Çift-geometri ve simetrik birleştirme
- **Şekil 3:** Aynı buluttan skor, ablation, prototip ve drift
- **Şekil 4:** Nokta-sayı optimizasyonu ve geometri zenginliği
- **Şekil 5:** Küme agregasyonu karşılaştırması
- **Şekil 6:** Bulut geometri vizualizasyonu (3D)

---

## 7. BULUŞUN DETAYLI AÇIKLAMASI

### 7.1 Veri Ön İşleme

**Türkçe Wikipedia + Turkish Sentences korpusu:**
- **1.5M cümle** (HuggingFace: NumanKaanKaratas/turkish-sentences)
- **200K cümle** filtre (min_len=20, max_len=300 karakter)
- **BPE tokenizer:** vocab_size=20,000

### 7.2 Model Yapısı

```python
class CloudLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, d_cloud=512,
                 d_cloud_dim=32, n_layers=8, nhead=16,
                 pool="max", cloud_attn=False):
        # Embedding: token → d_model
        # Transformer: 8 katman, 16 head
        # WordToND: d_model → D-boyutlu nokta (Tanh)
        # CloudEncoder: nokta bulutu → küme durumu
        # Word Head: küme durumu → vocab
        # Point Head: küme durumu → bir sonraki nokta
```

### 7.3 Eğitim Protokolü

**Kayıp fonksiyonu:**
```
loss = word_loss + 0.1 × point_loss
```

**word_loss:** Cross-entropy (bir sonraki kelime)  
**point_loss:** MSE (bir sonraki kelimenin bulut konumu)

**Hyperparametreler:**
- **Batch size:** 64
- **Learning rate:** 2e-4
- **Epochs:** 50
- **Gradient clipping:** 0.5
- **Weight decay:** 0.01
- **Scheduler:** Cosine annealing

### 7.4 Nokta-Sayı Optimizasyonu

Nokta sayısı bulut geometrisinin zenginliğini belirler:
- **Daha fazla nokta:** Daha detaylı geometrik temsil
- **Optimum nokta sayısı:** Uygulama ve veriye göre belirlenir
- **Aşırı nokta:** Geometrik gürültüye yol açabilir

### 7.5 Küme Agregasyonu

**Max-pool tercih nedeni:**
- Daha az parametre
- Daha hızlı hesaplama
- Daha stabil yakınsama

**Attention-pool alternatif:**
- Bağlam-ağırlıklı agregasyon
- Ek parametre gerektirir

---

## 8. İDDİALAR

### Bağımsız İddialar

**İddia 1 — Sistem:** Düşük boyutlu nokta-bulutu geometrisiyle bir girdi biriminden görev çıktısı üreten bilgisayar sistemi olup, **karakterize edici özellik olarak**:

(a) her bir token'ı veya girdi birimini **3 ≤ D ≤ 64** boyutlu Öklid uzayda bir pᵢ noktasına projekte eden bir projeksiyon modülü;

(b) nokta özelliklerine ek olarak nokta çiftleri için en az **mesafe ve açı bileşenlerini** kodlayan bir çift-geometri kodlayıcısı;

(c) nokta özellikleri ve çift-geometri özelliklerini **permütasyon-değişmez küme agregasyonu** ile bir küme durumuna birleştiren bir modül;

(d) adı geçen küme durumundan bir görev skoru veya çıktı üreten bir okuyucu; ve

(e) **aynı nokta bulutundan** native açıklama üreten bir yorumlama sistemi (leave-one-point-out ablation, bölge/prototip isimlendirme veya bölge-kayması/drift incelemesi)

içermekte olup, **aynı nokta-bulutu geometrisi hem görev skorunun hesaplama substratı hem de native açıklamanın kaynağıdır**.

**İddia 2 — Yöntem:** Nokta-bulutu geometrisiyle görev çıktısı ve native açıklama üreten bilgisayarla uygulanan bir yöntem olup, **karakterize edici özellik olarak**:

(a) token veya girdi birimlerini **3 ≤ D ≤ 64** boyutlu noktalar olarak projekte etme;

(b) nokta ve nokta-çifti özelliklerini hesaplama;

(c) nokta özellikleri ve çift-geometri özelliklerini **permütasyon-değişmez küme agregasyonu** ile bir küme durumuna birleştirme;

(d) küme durumundan okuyucu ile görev skoru/çıktısı üretme; ve

(e) **aynı bulut** üzerinde leave-one-point-out logit-ablation, bölge/prototip isimlendirme veya bölge-kayması/drift incelemesinden en az birini yürütme

adımlarını içermektedir.

**İddia 3 — Bilgisayar programı:** Bir işlemci üzerinde çalıştırıldığında İddia 2'deki yöntem adımlarını gerçekleştiren program talimatlarını içeren, geçici olmayan bilgisayar tarafından okunabilir bir kayıt ortamı.

### Bağımlı İddialar

**İddia 4:** İddia 1'deki sistem olup, çift-geometri kodlayıcısı her nokta çifti için **[pᵢ-pⱼ, ‖pᵢ-pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖] ∈ ℝ^(D+4)** vektörünü kullanır.

**İddia 5:** İddia 1'deki sistem olup, **D değeri 32 veya 64** olarak seçilir.

**İddia 6:** İddia 1'deki sistem olup, **nokta sayısı 20** olarak seçilir (R2b optimizasyonu).

**İddia 7:** İddia 1'deki sistem olup, **max-pool** küme agregasyonunda kullanılır (attention-pool yerine).

**İddia 8:** İddia 1'deki sistem olup, okuyucu **INT8 kantizasyon** ile temsil edilir.

**İddia 9:** İddia 1'deki sistem olup, **decoder tam sözlük** eşlemesi taşır ve hiçbir token/bucket kimliğini "other" kovasına atmaz.

**İddia 10 — Dil embodiment'i:** İddia 1'deki sistem olup, girdi birimleri token/alt-sözcük ve okuyucu çıktısı bir-sonraki-token dağılımıdır.

**İddia 11 — Nokta sayısı:** İddia 1'deki sistem olup, nokta sayısı 10 ile 30 arasında seçilir, optimum değer uygulama ve veriye göre belirlenir.

**İddia 12 — Küme agregasyonu:** İddia 1'deki sistem olup, nokta özellikleri için max-pool, çift-geometri özellikleri için mean-pool kullanılır.

---

## 9. ÖZET (ABSTRACT)

Sistem, token veya girdi birimlerini düşük boyutlu (3 ≤ D ≤ 64) noktalar olarak temsil eder; nokta özellikleri ile mesafe/açı temelli çift-geometriyi simetrik küme agregasyonuyla birleştirir ve küme durumundan okuyucuyla görev skoru üretir. Aynı nokta bulutu, leave-one-point-out ablation, bölge/prototip isimlendirme ve bölge-kayması/drift incelemesi için native açıklama nesnesidir.

Sistem, dil modelleme alanında uygulanabilir; doğruluk-üstünlüğü iddia edilmez, açıklama geometriye dayalı denetim yüzeyidir.

---

## ATIF LİSTESİ

1. Vaswani, A., et al., 2017. *Attention Is All You Need.* NeurIPS 30.
2. Qi, C., et al., 2017. *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* CVPR.
3. Zaheer, M., et al., 2017. *Deep Sets.* NeurIPS 30.
4. Lee, J., et al., 2019. *Set Transformer.* ICML.
5. Mikolov, T., et al., 2013. *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781.
6. Pennington, J., et al., 2014. *GloVe: Global Vectors for Word Representation.* EMNLP.
7. Chung, S., & Abbott, L. F., 2021. *Geometry of a neural population.* Nature 592.

---

## 10. VEKİL DEĞERLENDIRMESI (**DOSYAYA EKLENMEZ — strateji notu**)

### 10.1 Yenilik ve Barizlik (m.69)

**Yenilik göstergeleri:**
1. **PointNet ayrımı:** PointNet'te noktalar ham 3B sensör/görüntü girdisidir; burada nokta, farklı alan verilerinden **öğrenilmiş temsildir**.
2. **Çift-geometri ayrımı:** Tercih edilen D+4 vektör — fark, mesafe, kosinüs ve iki norm — burada belirginleştirilmiştir.
3. **Native açıklama ayrımı:** Aynı karar bulutundan ablation, prototip ve drift incelemesi yapılır; bu, SHAP özellik listesinin aynısı olarak sunulmaz.
4. **Nokta-sayı optimizasyonu:** Nokta sayısı bulut geometrisinin zenginliğini belirler; optimum değer uygulama ve veriye göre belirlenir.

**Barizlik riski:**
- Transformer + PointNet kombinasyonu bariz görünebilir
- **Savunma:** Nokta-bulutu geometrisinin **hem hesaplama hem açıklama substratı** olması yeni bir mimari karardır

### 10.2 Teknik Etki (m.69)

**Somut teknik etkiler:**
1. **Düşük boyutlu temsil:** D=32/64 boyutu, transformer'ın d_model=256-1024'e göre %75-94 daha az boyut
2. **Kompakt dağıtım:** INT8 kantizasyon ile ~5-6 MB
3. **Native açıklama:** Post-hoc değil, model içi açıklama
4. **Geometri zenginliği:** Nokta sayısı ile temsil kapasitesi kontrol edilebilir

### 10.3 İddia Kapsamı ve Sınırlamalar

**C1 — Doğruluk sınırlaması:** İddia 10'da PPL 37.63 değeri **bu embodiment'e özeldir**; evrensel performans garantisi değildir.

**C2 — Permütasyon:** İddia 1(c) simetrik küme agregasyonunu sınırlar; bağlam/tarihçeden üretilen noktaların sıra-duyarlı olabileceğini açıklar.

**C3 — Dönüşüm:** İddia 4 fark vektörünü eşdeğişken; mesafe/kosinüs/norm bileşenlerini değişmez olarak ayırır.

**C4 — Prior art:** 2.2'de incelenen kaynaklarda özgül birleşimin gösterilmediği söylenir; küresel "hiçbirinde yok" iddiası yapmaz.

### 10.4 Bölünmüş Başvuru Riski

**Tek icat fikri:** Dual-use geometri mekanizması (aynı bulut hem hesaplama hem açıklama).

**Dikey embodiment'ler:** Dil, fraud, AML, bakım — bunlar aynı icadın uygulamalarıdır, ayrı icatlar değildir.

**Öneri:** Eğer inceleme "çok icat" itirazı yaparsa, dikey embodiment'ları **bölünmüş (divisional)** başvurulara ayır.

### 10.5 Öncelik ve Tarih

**TR ulusal başvuru:** `TURKPATENT_basvuru_v2.3.md` (dil odaklı)  
**Bu başvuru:** v2.5 (claim odaklı, teknik özellik vurgusu)

**Öncelik ilişkisi:** Bu v2.5, v2.3'ten **genelleştirilmiş iddialar** içerir (nokta sayısı aralığı, küme agregasyonu varyantları). Öncelik tarihi T1 (v2.3) korunur, ancak yeni özellikler için T2 (v2.5) tarihi geçerlidir.

### 10.6 Şekil Hazırlığı

6 şekil hazır:
- fig1_mimari.png — Evrensel mimari
- fig2_ciftgeometri.png — Çift-geometri ve simetrik birleştirme
- fig3_aciklama.png — Aynı buluttan native açıklama
- fig4_nokta_optimizasyon.png — D=64 + Nokta=20 optimizasyonu
- fig5_max_vs_attn_pool.png — Max-pool vs attention-pool
- fig6_bulut_vizualizasyon.png — Bulut geometri vizualizasyonu

**TÜRKPATENT gereği:** Çizimler siyah-beyaz, referans numaralı, ayrı sayfalar.

---

**NOT:** Bu metin TÜRKPATENT formuna uygun olarak hazırlanmıştır. Resmi başvuru öncesi patent vekili değerlendirmesi gereklidir.

---

## DEĞİŞİKLİKLER (v2.4 → v2.5)

1. **İddia 10:** Spesifik PPL değerleri kaldırıldı, genel dil embodiment'i olarak güncellendi
2. **İddia 11 eklendi:** Nokta sayısı aralığı (10-30) ve optimum belirleme mekanizması
3. **İddia 12 eklendi:** Küme agregasyonu tercihleri (max-pool / mean-pool)
4. **Bölüm 5:** Deneysel sonuç tabloları kaldırıldı, mimari avantajlara odaklanıldı
5. **Bölüm 7.4-7.5:** Spesifik sayısal sonuçlar yerine teknik mekanizmalar açıklandı
6. **Özet:** Performans değerleri kaldırıldı, teknik özellikler korundu
