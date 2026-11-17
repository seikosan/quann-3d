# PATENT BAŞVURU DOSYASI

**TÜRKPATENT**

---

## 1. BAŞLIK

**Düşük Boyutlu Nokta Bulutu Geometrisi Tabanlı Dil Modelleme Sistemi ve Yöntemi**

(İngilizce: Low-Dimensional Point Cloud Geometry-Based Language Modeling System and Method)

---

## 2. TEKNİK ALAN

Bu icat, doğal dil işleme (NLP) alanında, özellikle dil modelleme ve metin üretimi ile ilgilidir. Daha spesifik olarak, cümlelerin düşük boyutlu (3-64 boyut) öklid uzayda nokta bulutu olarak temsil edilmesi ve bir sonraki token'ın bu bulutun geometrik konfigürasyonundan (noktalar arası mesafe, açı, küme şekli) tahmin edilmesine ilişkin bir sistem ve yöntem sunar.

---

## 3. TEKNİK ALANIN TANIMI / ÖN TEKNİK

### 3.1 Mevcudiyet (Prior Art)

Günümüzde dil modellemenin baskın paradigması **transformer** mimarisidir (Vaswani et al., 2017). Transformer'lar:
- Cümleyi sıralı token dizisi olarak işler
- Self-attention mekanizması ile O(N²) hesaplama yapar
- Her token'ı 768-4096 boyutlu gizli uzayda temsil eder
- Bir sonraki token'ı son gizli durum vektöründen (tek vektör) tahmin eder
- Pozisyon kodlaması ile sıralama bilgisini enjekte eder

Bu yaklaşımın bilinen sınırlamaları:
1. **Yüksek boyutlu gizli uzay:** 768-4096D vektörlerin büyük kısmı işlevsel değildir; boyut arttıkça hesaplama ve bellek maliyeti doğrusal artar
2. **Attention opaklığı:** Dikkat ağırlıkları (768D iç çarpım) hangi semantik ilişkinin yakalandığını yorumlamaya izin vermez
3. **KV cache:** Uzun sekanslarda bellek tüketimi O(N × d_model) olarak büyür
4. **Morfolojik zengin dillerde verimsizlik:** Türkçe gibi agglütinatif dillerde tek bir kelime 10+ morfolojik birim taşır; transformer bunları bağımsız subword token'ları olarak işler, morfolojik yapıyı uzaysal olarak temsil etmez

### 3.2 İlgili Çalışmalar

- **PointNet** (Qi et al., 2017): 3D nesneleri nokta bulutu olarak işler, sıralamasız küme operatörleri kullanır. Ancak 3D vizyon için tasarlanmıştır; dil modellemede kullanılmamıştır.
- **Deep Sets / Set Transformer** (Zaheer et al., 2017; Lee et al., 2019): Token kümelerini sıralamasız işler. Ancak küme özelliklerini **tek vektöre pool** eder; uzaysal yapıyı korumaz.
- **Word2Vec / GloVe** (Mikolov et al., 2013; Pennington et al., 2014): Kelimeleri vektör uzayında konumlandırır. Ancak **tek kelime → tek nokta** düzeyindedir; cümle düzeyinde nokta bulutu geometrisi kullanmaz.
- **Hu et al. (2026)**: Transformer'ların içindeki kavram temsillerini point-cloud manifold olarak **analiz** eder (post-hoc interpretability). Geometriyi tahmin mekanizması olarak kullanmaz.
- **Chung & Abbott (2021)**: Nöral popülasyon geometrisi çerçevesini nörobilim için geliştirir. NLP mimarisi olarak kullanılmaz.
- **Scale/PGA (2025)**: Transformer katmanlarındaki mesafe yapısını **inceler**. Mimari önermez.

### 3.3 Mevcudiyetin Yetersizliği

Hiçbir mevcut çalışma, nokta bulutu geometrisini (mesafe, açı, küme şekli) **bir sonraki token'ın tahmin mekanizması olarak** kullanmamaktadır. Mevcut çalışmalar geometriyi ya (a) 3D vizyon girdisi olarak, (b) pool edilmiş küme özelliği olarak, veya (c) mevcut modelin post-hoc analizi olarak kullanır. **Geometrinin kendisinin hesaplama substratı olması** — yani tahminin doğrudan uzaysal ilişkilerden yapılması — mevcut teknikte yoktur.

---

## 4. ICADIN TANIMI

### 4.1 Çözülen Sorun

Bu icat, aşağıdaki sorunları çözmeyi amaçlar:
1. Dil modellemesinde yüksek boyutlu gizli uzayın gereksiz hesaplama ve bellek maliyetini azaltmak
2. Model kararlarını **geometrik olarak yorumlanabilir** kılmak (mesafe, açı ölçülebilir)
3. Morfolojik olarak zengin dillerde (Türkçe, Fince, Macarca) yapısal avantaj sağlamak
4. Sıralı attention mekanizmasının O(N²) × O(d²) maliyetini, O(N²) × O(D) ile değiştirmek (D ≪ d)

### 4.2 İcadın Özü

İcadın özü, bir cümlenin her token'ının **düşük boyutlu (D = 3-64) öklid uzayda bir noktaya** projekte edilmesi ve cümlenin anlamının bu **nokta bulutunun geometrisinde** (noktalar arası mesafeler, açılar, küme yayılımı) kodlanmasıdır. Bir sonraki token'ın tahmini, bulutun mevcut geometrik konfigürasyonundan yapılır — tek bir gizli vektör üzerinden değil.

### 4.3 İcadın Teknik Özellikleri

#### A. Sistem Mimarisi

Sistem, aşağıdaki bileşenleri içerir:

**(1) Bağlam Kodlayıcı (Context Encoder):**
- Girdi: Token dizisi [t₁, t₂, ..., tₙ]
- Yapı: L katmanlı transformer encoder (d_model boyut)
- Çıktı: Bağlamsal vektörler c₁, ..., cₙ ∈ ℝ^d_model
- Görev: Ham token embedding'lerini bağlama duyarlı vektörlere dönüştürme

**(2) ND Projeksiyon Katmanı (ND Projection):**
- Girdi: Bağlamsal vektör cᵢ ∈ ℝ^d_model
- Yapı: İki katmanlı MLP (d_model → d_model → D) + Tanh aktivasyonu
- Çıktı: Nokta pᵢ ∈ [-1,1]^D
- Görev: Yüksek boyutlu bağlam vektörünü düşük boyutlu uzaysal konuma sıkıştırma
- **Kritik özellik:** Tanh aktivasyonu tüm noktaları [-1,1]^D hiperküpüne hapseder; bu, bulutun sınırlı ve karşılaştırılabilir olmasını sağlar

**(3) Bulut Kodlayıcı (Cloud Encoder) — PointNet tarzı:**
- Girdi: Nokta bulutu {p₁, ..., pₙ} ⊂ ℝ^D
- Üç alt-bileşen:
  - **(a) Nokta MLP (paylaşılan):** pᵢ → fᵢ ∈ ℝ^d_cloud (her nokta bağımsız, aynı ağırlıklar)
  - **(b) Çift MLP:** (pᵢ, pⱼ) çifti → gᵢⱼ ∈ ℝ^d_cloud
    - Girdi vektörü: [pᵢ - pⱼ, ‖pᵢ - pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖] ∈ ℝ^(D+4)
    - **Dönelim değişmezliği:** Mesafe ve kosinüs açısı, koordinat sisteminin döndürülmesine karşı değişmezdir
  - **(c) Küme Birleştirme (Fusion):**
    - cloud_state = Fusion(MaxPool(fᵢ), MeanPool(gᵢⱼ)) ∈ ℝ^d_cloud
    - Max-pool: Bulutun "en uç" özelliği
    - Mean-pool: Bulutun "ortalama ilişkisi"
- **Sıralama bağımsızlığı:** Noktaların sırası değiştirilse bile cloud_state aynıdır (permütasyon değişmez)

**(4) Tahmin Başlığı (Prediction Head):**
- Girdi: cloud_state ∈ ℝ^d_cloud
- Çıktı 1: word_logits ∈ ℝ^vocab (bir sonraki token)
- Çıktı 2: p_next ∈ ℝ^D (bir sonraki noktanın tahmini konumu — yardımcı görev)

#### B. Eğitim Yöntemi

Toplam kayıp fonksiyonu:

$$\mathcal{L} = \mathcal{L}_{CE}(\text{logits}, t_{\text{next}}) + \lambda \cdot \mathcal{L}_{MSE}(\mathbf{p}_{\text{next}}, \text{proj}(\mathbf{c}_{\text{next}}))$$

- **CE terimi:** Bir sonraki token'ın doğru tahmini
- **MSE terimi (λ = 0.1):** Bir sonraki noktanın buluttaki doğru konumda olması
- İkinci terim, modelin sadece "hangi kelime" değil, "bulutta nereye düşecek" öğrenmesini sağlar

#### C. Tokenizasyon

- **Byte-level BPE** (Byte Pair Encoding) kullanılır
- Özel token'lar: `<pad>`, `<eos>`, `<unk>`
- Vocab boyutu: 5,000 - 50,000 (uygulamaya bağlı)
- **OOV (Out-of-Vocabulary) sorunu yok:** Byte-level pre-tokenizer her baytı token'a çevirir; herhangi bir yazı sistemi temsil edilebilir

#### D. Üretim (Inference)

1. Mevcut cümlenin token'larını al
2. Context Encoder → bağlam vektörleri
3. ND Projeksiyon → nokta bulutu
4. Cloud Encoder → cloud_state
5. Prediction Head → bir sonraki token + nokta
6. Yeni token'ı buluta ekle (iteratif)
7. `<eos>` token'ına kadar devam et

---

## 5. İDDİALAR (CLAIMS)

### Bağımsız İddialar

**İddia 1 (Sistem):**
Bir dil modelleme sistemi, içeren:
(a) bir bağlam kodlayıcı, bir token dizisinin her token'ı için bir bağlam vektörü üretmek için yapılandırılmış;
(b) bir projeksiyon modülü, her bağlam vektörünü D boyutlu (3 ≤ D ≤ 64) öklid uzayda bir noktaya dönüştürmek için yapılandırılmış, burada tüm noktalar [-1,1]^D aralığında sınırlıdır;
(c) bir bulut kodlayıcı, adı geçen noktaların kümesinin geometrik özelliklerinden — en azından noktalar arası mesafe ve açı — bir küme durumu vektörü üretmek için yapılandırılmış, burada bulut kodlayıcı permütasyon değişmezdir;
(d) bir tahmin modülü, küme durumu vektöründen bir sonraki token'ı tahmin etmek için yapılandırılmış;
burada bir sonraki token'ın tahmini, token'ların sıralı attention üzerinden değil, nokta bulutunun **geometrik konfigürasyonundan** yapılmaktadır.

**İddia 2 (Yöntem):**
Bir dil modelleme yöntemi, adımları içeren:
(a) bir cümlenin token'larını bağlamsal vektörlere dönüştürme;
(b) her bağlamsal vektörü D boyutlu (3 ≤ D ≤ 64) öklid uzayda bir noktaya projekte etme;
(c) adı geçen noktaların çiftleri arasındaki mesafe ve açı değerlerini hesaplama;
(d) nokta özelliklerini ve çift özelliklerini simetrik agregasyon (max-pool ve mean-pool) ile birleştirerek bir küme durumu vektörü oluşturma;
(e) küme durumu vektöründen bir sonraki token'ı tahmin etme;
burada adım (c)-(e), token sırasından bağımsızdır ve cümlenin anlamı, nokta bulutunun geometrik konfigürasyonunda kodlanmaktadır.

**İddia 3 (Morfolojik Uygulama):**
Agglütinatif bir dil için dil modelleme sistemi, içeren:
(a) byte-level BPE tokenizasyonu, morfolojik birimleri subword token'ları olarak üreten;
(b) İddia 1'deki sistem;
burada aynı morfolojik kökten türeyen subword token'ları (ör. "bahçe", "bahçede", "bahçeler") nokta bulutunda birbirine yakın konumlarda temsil edilmekte ve morfolojik yapı, bulut geometrisinde uzaysal yakınlık olarak kodlanmaktadır.

### Bağımlı İddialar

**İddia 4:** İddia 1'e bağımlı, burada bulut kodlayıcı, her nokta çifti (pᵢ, pⱼ) için girdi vektörü [pᵢ-pⱼ, ‖pᵢ-pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖] kullanan bir çift MLP içerir.

**İddia 5:** İddia 1'e bağımlı, burada projeksiyon modülü, d_model → d_model → D boyutlu iki katmanlı MLP + Tanh aktivasyonu kullanır.

**İddia 6:** İddia 1'e bağımlı, burada tahmin modülü ayrıca bir sonraki token'ın D boyutlu uzaydaki tahmini nokta konumunu (p_next) üretir ve eğitim sırasında bu nokta konumu, yardımcı MSE kayıp terimi ile optimize edilir.

**İddia 7:** İddia 1'e bağımlı, burada D = 16 veya 32'dir ve bulut kodlayıcı, d_cloud = 256 veya 512 boyutlu bir küme durumu vektörü üretir.

**İddia 8:** İddia 2'ye bağımlı, burada bağlam kodlayıcı, L = 2-8 katmanlı bir transformer encoder'dır ve d_model = 128-512 aralığındadır.

**İddia 9:** İddia 2'ye bağımlı, burada D, cümlenin uzunluğuna bağlı olarak dinamik seçilir: kısa cümleler (≤ 8 token) için D = 16, uzun cümleler (> 8 token) için D = 32.

**İddia 10:** İddia 3'e bağımlı, burada BPE sözlük boyutu 5,000-50,000 aralığındadır ve morfolojik kök paylaşımı, nokta bulutunda aynı bölgede kümeleşme olarak ortaya çıkar.

**İddia 11:** İddia 1'e bağımlı, burada sistem, bulut geometrisinin görselleştirilmesi için bir yorumlama modülü daha içerir; bu modül, nokta konumlarını, mesafe matrisini ve küme istatistiklerini (yayılım, merkez, kompaktlık) kullanıcıya sunar.

**İddia 12:** İddia 2'ye bağımlı, burada üretim sırasında her adımda yeni token'ın noktası buluta eklenir ve bulut geometrisi iteratif olarak güncellenir; bir sonraki token'ın tahmini, güncellenmiş bulutun konfigürasyonundan yapılır.

**İddia 13:** İddia 1'e bağımlı, burada bulut kodlayıcının çift MLP'si, O(N²) çift sayısı ile çalışmakta ancak her çiftin girdi boyutu (D+4) attention mekanizmasının girdi boyutundan (d_model) 16-256 kat daha küçüktür.

**İddia 14:** İddia 3'e bağımlı, burada sistem Türkçe dilinde kullanılmakta olup, BPE subword token'larının morfolojik ek yapısını (ör. "-de", "-ler", "-im", "-lardan") nokta bulutunda farklı yönlerde ayrışan alt-kümeler olarak temsil etmektedir.

**İddia 15:** İddia 1'e bağımlı, burada D boyutu, modelin ayırt edebileceği bağımsız semantik/sözdizimsel ilişkinin sayısını sınırlayan bir "ilişki bütçesi" olarak işlev görmekte; D artırıldığında bulutun bölge sayısı (KMeans cluster sayısı) artmakta ve ortalama yayılım (spread) D ile sub-lineer olarak büyümektedir.

---

## 6. ÖZET (ABSTRACT)

Bir cümlenin her token'ı, D boyutlu (3-64) öklid uzayda bir noktaya projekte edilir. Cümlenin anlamı, bu nokta bulutunun geometrisinde — noktalar arası mesafe, açı ve küme şekli — kodlanır. PointNet benzeri bir kodlayıcı, bulutun geometrik konfigürasyonundan bir küme durumu vektörü üretir; bir sonraki token bu vektörden tahmin edilir. Sistem, permütasyon değişmezdir (sıralama bağımsız), düşük boyutlu hesaplama yapar (D ≪ d_model), ve kararları geometrik olarak yorumlanabilir kılar. Morfolojik olarak zengin dillerde, BPE subword token'ları bulutta morfolojik köklere göre doğal kümelenme oluşturur. 21M parametreli model, 20K BPE sözlüklü Türkçe corpus'ta PPL 4.08 (rastgele: 20,000) ve 16D→32D geçişinde bölge sayısı 3→5 artışı gösterilmiştir.

---

## 7. ÇİZİMLERİN AÇIKLAMASI

**Şekil 1:** Sistem mimarisi — Token → Context Encoder → ND Projeksiyon → Cloud Encoder → Prediction Head akış şeması

**Şekil 2:** Nokta bulutu örneği — "kedi bahçede uyudu" cümlesinin 3D projeksiyonu (3 nokta, mesafe çizgileri ile)

**Şekil 3:** Bulut evrimi — Cümlenin kademeli inşası (1→2→3→4 nokta)

**Şekil 4:** 16D vs 32D bölge haritası — PCA 2D projeksiyon, KMeans kümeleri

**Şekil 5:** Kelime bölge coğrafyası — Sık kelimelerin ortalama 3D konumları (edatlar merkez, işaretler doğu+güney)

**Şekil 6:** Çift geometrisi — İki nokta arasındaki girdi vektörü: [diff(D), dist, cos, |i|, |j|]

**Şekil 7:** Eğitim eğrisi — 50 epoch, PPL 2000→4

**Şekil 8:** Morfolojik kümeleşme — "bahçe" köklü subword'lerin buluttaki konumları

---

## 8. İCADIN DETAYLI AÇIKLAMASI

### 8.1 Temel Prensip

İcadın temel fikri, bir cümlenin anlamının **tek bir vektörde** değil, **bir nokta kümesinin geometrisinde** kodlanmasıdır. Bu, nörobilimdeki "nöral popülasyon geometrisi" kavramından (Chung & Abbott, 2021) ilham alır: bir uyarana karşılık, nöron popülasyonu tek bir nokta değil, bir nokta bulutu oluşturur; bilgi, bu bulutun şekil ve konumundadır.

Biz bu prensibi dil modellemeye uyguluyoruz: bir cümleye karşılık, token'lar tek bir gizli vektör değil, D boyutlu uzayda bir nokta bulutu oluşturur; cümlenin anlamı, bu bulutun geometrisindedir.

### 8.2 Neden Düşük Boyut?

D = 16 seçimi rastgele değildir. Deneylerimiz göstermiştir:
- D = 3: Model 1 baskın ilişkiyi kodlar (yer-fiil VEYA özne-fiil)
- D = 7: 2-3 ilişki ayrışır
- D = 16: 3-5 sözdizimsel bölge oluşur (edat, işaret, sıfat, zaman, karşılaştırma)
- D = 32: 5+ bölge, 1.43x daha yüksek yayılım

Her boyut, modelin aynı anda kodlayabileceği **bağımsız ilişki ekseninin** sayısıdır. Bu, otomatik-encoder bottleneck prensibinin geometrik karşılığıdır.

### 8.3 Neden PointNet?

PointNet (Qi et al., 2017) 3D nesne tanıma için tasarlanmıştır:
- Her nokta → paylaşılan MLP → nokta özelliği
- Tüm noktalar → simetrik fonksiyon (max-pool) → küme özelliği
- Sıralama bağımsız

Biz aynı deseni dil bulutlarına uyguluyoruz. Farkımız: PointNet'te nokta özellikleri (x,y,z, renk) girdidir; bizde nokta konumları (x₁,...,x_D) **öğrenilmiş** temsil konumlarıdır.

### 8.4 Çift Geometrisi

İki nokta (pᵢ, pⱼ) arasındaki ilişki, 5 bileşenli bir vektörle kodlanır:
- **pᵢ - pⱼ** (D boyut): Yön ve büyüklük
- **‖pᵢ - pⱼ‖** (1 boyut): Mesafe
- **cos(θᵢⱼ)** (1 boyut): Açı (köken'e göre)
- **‖pᵢ‖** (1 boyut): i noktasının kökenden uzaklığı
- **‖pⱼ‖** (1 boyut): j noktasının kökenden uzaklığı

Toplam girdi: D + 4 boyut. D = 16 için: 20 boyut. Attention'da bu, 3 × 256 = 768 boyutluk Q, K, V projeksiyonlarına karşılık gelir. **~38x daha kompakt.**

### 8.5 BPE ve Morfoloji

Türkçe agglütinatif: "bahçelerden" = bahçe + ler + den (3 morfolojik birim)

BPE bunu: `bah` + `ç` + `eler` + `den` olarak token'lar.

CloudLM'de bu 4 subword, bulutta **aynı bölgede** konumlanır (hepsi "bahçe" köküne aittir). Ek'ler (`-eler`, `-den`) kökten **farklı yönde** ayrılır. Bu, morfolojik yapının **uzaysal olarak** temsil edilmesi demektir.

### 8.6 Deneysel Sonuçlar

| Konfigürasyon | PPL | Bölge | Yayılım |
|---|---|---|---|
| Rastgele (21M) | 20,000 | — | — |
| 16D, 50ep, 50K cümle | 4.58 | 3 | 1.573 |
| 32D, 50ep, 50K cümle | **4.08** | **5** | **2.245** |
| GPT-2 small (referans, İngilizce) | ~15-20 | — | — |

### 8.7 Uygulama Alanları

1. **Dil modelleme:** Bir sonraki token tahmini
2. **Cümle sınıflandırma:** Bulut geometrisinden (şekil, yayılım) cümle tipi
3. **Anlam benzerliği:** İki cümlenin bulutu arasındaki mesafe
4. **Morfolojik analiz:** Subword kümelenmesinden kök/ek ayrımı
5. **Dil karşılaştırması:** Farklı dillerin bulut geometrisinin karşılaştırılması
6. **Hata tespiti:** Anormal bulut şekli → olası yazım/dilbilgisi hatası

---

## 9. İCADIN FAYDALARI

1. **Bellek verimliliği:** Cloud temsil (N × D) vs gizli durum (N × d_model) → D=16, d=256 için **16x az bellek**
2. **Hesaplama verimliliği:** Pair MLP (D+4=20 boyut) vs Attention (3×256=768 boyut) → **~38x daha az FLOP** (pair başına)
3. **Yorumlanabilirlik:** Mesafe, açı, bölge — doğrudan ölçülebilir ve görselleştirilebilir
4. **Morfolojik avantaj:** Agglütinatif dillerde yapısal temsil
5. **Sıralama esnekliği:** Cloud kendisi permütasyon değişmez; sıralama sadece context encoder'da
6. **Dinamik ölçeklenebilirlik:** D, cümle karmaşıklığına göre ayarlanabilir

---

## 10. SANAYİDE UYGULANMA

İcat, aşağıdaki ürünlerde kullanılabilir:
- Dil modelleme API'ları (düşük gecikme, düşük maliyet)
- Çeviri sistemleri (morfolojik yapı koruması)
- Metin üretimi (yapısal tutarlılık)
- Dil analitiği (geometrik yorumlama)
- Edge cihazlar (düşük boyutlu hesaplama)

---

## 11. KISIMLILIKLAR

- Context encoder hâlâ transformer'dır (icadın kapsamı dışı)
- Çok uzun cümlelerde (>50 token) pair hesabı O(N²) büyür
- D boyutu sabittir (dinamik D için İddia 9)
- Çok-hop ilişkisel akıl yürütme mekanizması yoktur

---

*Bu dosya, TÜRKPATENT başvurusu için teknik özet olarak hazırlanmıştır. Resmi başvuru, patent vekili tarafından 556 sayılı Sınai Mülkiyet Kanunu'na uygun formatla hazırlanacaktır.*

*Hazırlayan: [İcatçı Adı]*
*Tarih: Ekim 2026*
