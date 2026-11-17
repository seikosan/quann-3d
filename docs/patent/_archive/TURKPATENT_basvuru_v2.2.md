# TÜRKPATENT PATENT BAŞVURUSU

**556 sayılı Sınai Mülkiyet Kanunu'na uygun başvuru dosyası**

| | |
|---|---|
| **Başvuru No:** | [TÜRKPATENT tarafından verilecektir] |
| **Başvuru Tarihi:** | [başvuru tarihi = öncelik tarihi] |
| **İcatçı:** | [İcatçı Adı Soyadı, T.C. Kimlik No, adres] |
| **Başvuru Sahibi:** | [Sahip unvanı, adres] |
| **Vekil:** | [Patent vekili / vekil ofisi] |
| **İcat Başlığı:** | Düşük Boyutlu Nokta Bulutu Geometrisi Tabanlı Dil Modelleme Sistemi, Yöntemi ve Bilgisayar Programı |
| **Teknik Alan (IPC önerisi):** | G06F 18/00; G06N 3/04; G06N 5/02 |

> **GİZLİLİK / ÖNCELİK UYARISI:** Bu dosya patent başvurusu için hazırlanmıştır. Başvuru tarihi (öncelik hakkı) alınana kadar bu icata ilişkin hiçbir yayımlama (arXiv, konferans, dergi, blog, genel erişimli GitHub, podcast, sunum) yapılmamalıdır. Yayımlama, yenilik koşulu açısından yıkıcı etki (self-depublication) yaratır.

**Dosya içeriği:** 1. Başlık — 2. Açıklama — 3. İddialar — 4. Özet — 5. Çizimler (8 şekil, ayrı sayfalar).

---

## 1. BAŞLIK

**Düşük Boyutlu Nokta Bulutu Geometrisi Tabanlı Dil Modelleme Sistemi, Yöntemi ve Bilgisayar Programı**

(İngilizce: Low-Dimensional Point Cloud Geometry-Based Language Modeling System, Method and Computer Program)

---

## 2. AÇIKLAMA

### 2.1 Teknik Alan

Bu icat, doğal dil işleme (NLP) alanında, özellikle dil modelleme ve metin üretimi ile ilgilidir. Daha spesifik olarak, icat; bir cümlenin her token'ının düşük boyutlu (3 ≤ D ≤ 64) öklid uzayda bir noktaya projekte edilmesiyle oluşan **nokta bulutunun geometrik konfigürasyonundan** (noktalar arası mesafe, açı, küme yayılımı) bir sonraki token'ın tahmin edilmesine ilişkin bir sistem, bir yöntem ve bir bilgisayar programıdır.

İcat, özellikle morfolojik olarak zengin (agglütinatif) dillerin — Türkçe, Fince, Macarca, Japonca — dil modellemesinde, düşük gecikme ve düşük bellek gereksinimiyle çalışacak sistemlerde uygulanmak üzere tasarlanmıştır.

### 2.2 Ön Teknik (Prior Art)

#### 2.2.1 Baskın Paradigma: Transformer

Günümüzde dil modellemenin baskın paradigması transformer mimarisidir (Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I., 2017, *NeurIPS 30*). Transformer'lar:

- Cümleyi sıralı token dizisi olarak işler;
- Self-attention mekanizması ile O(N²) sekans karmaşıklığında hesaplama yapar;
- Her token'ı 768–4096 boyutlu gizli uzayda temsil eder;
- Bir sonraki token'ı son gizli durum vektöründen — yani **tek bir vektörden** — tahmin eder;
- Sıralama bilgisini pozisyon kodlaması ile enjekte eder.

Bu yaklaşımın bilinen sınırlamaları şunlardır:

1. **Yüksek boyutlu gizli uzay:** 768–4096 boyutlu vektörlerin büyük kısmı işlevsel değildir; boyut arttıkça hesaplama ve bellek maliyeti doğrusal olarak artar.
2. **Attention opaklığı:** Dikkat ağırlıkları (yüksek boyutlu iç çarpımlar) hangi semantik ilişkinin yakalandığını yorumlamaya izin vermez; karar mekanizması ölçülemez bir ağırlık matrisine gömülüdür.
3. **KV önbelleği (KV cache):** Uzun sekanslarda bellek tüketimi O(N × d_model) olarak büyür; bu, edge cihazlar ve düşük gecikme uygulamaları için belirleyici bir sınırlamadır.
4. **Morfolojik zengin dillerde verimsizlik:** Türkçe gibi agglütinatif dillerde tek bir sözcük 10+ morfolojik birim taşır ("bahçelerden" = bahçe + ler + den). Transformer bu birimleri bağımsız subword token'ları olarak işler; morfolojik yapı uzaysal olarak temsil edilmez, yalnızca dikkat ağırlıklarında dolaylı biçimde kodlanır.

#### 2.2.2 İlgili Çalışmalar

- **PointNet** (Qi, C., Yi, L., Su, H., & Guibas, L. J., 2017, *IEEE/CVF CVPR*, 77–80): Üç boyutlu nesneleri nokta bulutu olarak işler; paylaşılan MLP + simetrik (max-pool) küme operatörü ile sıralama bağımsız küme özelliği üretir. Ancak 3D vizyon girdisi için tasarlanmıştır; nokta konumları **ham sensör verisi** olarak girer ve dil modellemede kullanılmamıştır.
- **Deep Sets** (Zaheer, M., Kottur, M., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., & Smola, A. J., 2017, *NeurIPS 30*) / **Set Transformer** (Lee, J., Lee, J., Kim, M., Kosiorek, A. R., Choi, S., & Teh, Y. W., 2019, *ICML*, 3714–3723): Token kümelerini sıralamasız işler. Ancak küme özelliklerini **tek bir vektöre havuzlama (pool)** ile indirger; noktaların **uzaysal konum bilgisi girdi olarak var olmadığından** geometrik yapı korunmaz.
- **Word2Vec** (Mikolov, T., Chen, K., Corrado, G., & Dean, J., 2013, *arXiv:1301.3781*) / **GloVe** (Pennington, J., Socher, R., & Manning, C. D., 2014, *EMNLP*, 1532–1543): Kelimeleri vektör uzayında konumlandırır. Ancak **tek kelime → tek nokta** düzeyindedir; cümle düzeyinde, çok noktalı bir bulut geometrisi kullanmaz.
- **Hu ve ark. (2026)** (Hu, Z., Niu, L., & Varma, S., 2026, *arXiv:2607.04525* [cs.CL]; DOI 10.48550/arXiv.2607.04525): Transformer/LLM'lerin içindeki kavram temsillerini nöral popülasyon geometrisi çerçevesinde **point-cloud manifold** olarak biçimleştirir ve bağlamsal dönüşümleri vektör alanları olarak **analiz** eder (post-hoc yorumlanabilirlik). Geometriyi bir sonraki token'ın tahmin mekanizması olarak kullanmaz; yeni bir mimari önermez.
- **Chung & Abbott (2021)** (Chung, S., & Abbott, L. F., 2021, *Nature* 592(7853), 117–122): Nöral popülasyon geometrisi çerçevesini nörobilim için geliştirir. NLP mimarisi olarak kullanılmaz.
- **Xu (2026) — "Scale/PGA"** (Xu, W., 2026, *arXiv:2605.17084* [cs.LG]; DOI 10.48550/arXiv.2605.17084): *Subspace PGA* metriğiyle, bir katmanın **mesafe yapısının** unembedding matrisinin okuma alt-uzayına (readout subspace) ne ölçüde hizalandığını **inceler**; ölçek bağımlılığını gösterir. Yeni bir mimari önermez.

#### 2.2.3 Ön Teknikteki Boşluk

Hiçbir mevcut çalışma, nokta bulutu geometrisini (mesafe, açı, küme şekli) **bir sonraki token'ın tahmin mekanizması olarak** kullanmamaktadır. Mevcut çalışmalar geometriyi ya (a) 3D vizyon girdisi olarak (PointNet), (b) uzaysal konumu içermeyen pool edilmiş küme özelliği olarak (Deep Sets, Set Transformer), veya (c) mevcut bir modelin post-hoc analizi olarak (Hu ve ark., 2026; Xu, 2026) kullanır.

Ön teknikteki en yakın çalışmalar — Hu ve ark. (2026) kavram temsillerini point-cloud manifold olarak biçimleştirir; Xu (2026) katman mesafe yapısını *Subspace PGA* metriğiyle inceler — geometriyi yalnızca **mevcut bir modelin analizi** için kullanır; ikisi de geometriyi bir sonraki token'ın **tahmin mekanizması** olarak kullanmaz ve yeni bir mimari önermez.

Ön teknikteki temel boşluk şudur: **geometrinin kendisinin hesaplama substratı olması** — yani bir sonraki token tahmininin, öğrenilmiş düşük boyutlu nokta konumlarının uzaysal ilişkilerinden (mesafe, açı) doğrudan yapılması — mevcut teknikte bilinmemektedir.

### 2.3 Teknik Sorun

Bu icat, aşağıdaki teknik sorunları çözmeyi amaçlar:

1. Dil modellemesinde yüksek boyutlu gizli uzayın gereksiz hesaplama ve bellek maliyetini azaltmak;
2. Modelin tahmin kararlarını **geometrik olarak ölçülebilir ve yorumlanabilir** kılmak (mesafe, açı, bölge doğrudan hesaplanabilir);
3. Morfolojik olarak zengin dillerde (Türkçe, Fince, Macarca) morfolojik yapının uzaysal olarak temsil edilmesini sağlamak;
4. Bir sonraki token tahmininin, yüksek boyutlu ve opak attention ağırlıkları yerine düşük boyutlu ve geometrik olarak yorumlanabilir niceliklerden (mesafe, açı, bölge) yapılmasını sağlamak; bağlam kodlayıcı attention'sız (O(N)) seçildiğinde toplam hesaplama maliyetini de düşürmek.

### 2.4 İcadın Özü (Çözüm)

İcadın özü, bir cümlenin her token'ının **düşük boyutlu (3 ≤ D ≤ 64) öklid uzayda bir noktaya** projekte edilmesi ve cümlenin anlamının bu **nokta bulutunun geometrisinde** (noktalar arası mesafeler, açılar, küme yayılımı) kodlanmasıdır. Bir sonraki token'ın tahmini, bulutun mevcut geometrik konfigürasyonundan — permütasyon değişmez bir küme durumu vektörü üzerinden — yapılır; son gizli durumun tek bir vektörü üzerinden değil.

Bu yaklaşım, nörobilimdeki "nöral popülasyon geometrisi" ilkesinden (Chung & Abbott, 2021) ilham alır: bir uyarana karşılık bir nöron popülasyonu tek bir nokta değil, bir nokta bulutu oluşturur ve bilgi bu bulutun şekil ve konumundadır. İcat, bu ilkeyi dil modellemeye **hesaplama mekanizması olarak** uygulayan ilk sistemdir.

### 2.5 Teknik Etkiler

İcat, ön tekniğe kıyasla aşağıdaki teknik etkileri sağlar:

1. **Yorumlanabilirlik (birincil etki):** Mesafe, açı ve bölge (küme) istatistikleri doğrudan hesaplanabilir, ölçülebilir ve görselleştirilebilir; modelin bir sonraki token kararının hangi geometrik ilişkiden (hangi nokta çifti, hangi bölge) doğduğu denetlenebilir. Attention ağırlıklarının (yüksek boyutlu iç çarpımların) aksine, geometrik özellikler düşük boyutlu ve insan tarafından yorumlanabilir niceliklerdir.
2. **Bellek verimliliği (temsil düzeyi):** D boyutlu nokta temsil (N × D), d_model boyutlu bağlam temsiline (N × d_model) göre daha kompakttır; D = 16, d_model = 256 için nokta katmanı, bağlam katmanının 16'da 1'i kadar bellek gerektirir. (Sistemin toplam belleği bağlam kodlayıcı tarafından belirlenir; tasarruf, geometrik temsilin kendisindedir.)
3. **Morfolojik temsil:** Agglütinatif dillerde aynı kökten türeyen subword token'ları bulutta uzaysal olarak kümelenir; ekler kökten farklı yönlerde ayrışır.
4. **Sıralama esnekliği:** Bulut kodlayıcının agregasyonu permütasyon değişmezdir; ancak nokta konumları bağlam kodlayıcı aracılığıyla sıra-duyarlı olduğundan bulut geometrisi token sırasına dolaylı bağımlıdır (bkz. 2.7.1).
5. **Dinamik ölçeklenebilirlik:** D, cümlenin uzunluğuna veya ölçülen bulut karmaşıklığına göre ayarlanabilir.
6. **Hesaplama verimliliği (alternatif embodiment):** Bağlam kodlayıcı attention'sız (ör. lineer / Mamba benzeri, O(N)) seçildiğinde, toplam sistem O(N) bağlam + O(N²) düşük-boyutlu (D+4) çift hesabı ile, O(N²) yüksek-boyutlu (d_model) attention'a göre daha düşük maliyetlidir (bkz. 2.7.6.4, İddia 19).

### 2.6 Şekillerin Kısa Açıklaması

**Şekil 1:** İcadın sistem mimarisi — 100: bağlam kodlayıcı, 110: ND projeksiyon modülü, 120: bulut kodlayıcı (121: nokta MLP, 122: çift MLP, 123: küme birleştirme), 130: tahmin modülü (131: yardımcı nokta çıktısı), 140: tokenizasyon modülü, 150: yorumlama modülü.

**Şekil 2:** Nokta bulutu örneği — "kedi bahçede uyudu" cümlesinin 3D projeksiyonu; 200: token noktaları, 210: çift mesafe çizgileri.

**Şekil 3:** Üretim sırasında bulut evrimi — cümlenin kademeli inşası (1 → 2 → 3 → 4 nokta); 300: art arda bulut konfigürasyonları.

**Şekil 4:** 16D ve 32D bölge haritaları — PCA 2D projeksiyon, KMeans kümeleri; 400: 16D bölgeler (3 bölge), 410: 32D bölgeler (5 bölge).

**Şekil 5:** Kelime bölge coğrafyası — sık kelimelerin ortalama 3D konumları (edatlar merkezde, işaretler doğu-güneyde).

**Şekil 6:** Çift geometrisi — iki nokta arasındaki girdi vektörünün bileşenleri: 600: fark vektörü, 610: mesafe, 620: kosinüs açısı, 630: normlar.

**Şekil 7:** Eğitim eğrisi — 50 epoch; 700: PPL eğrisi (2000 → 4).

**Şekil 8:** Morfolojik kümeleşme — "bahçe" köklü subword token'larının buluttaki konumları; 800: kök alt-kümesi, 810: ek alt-kümeleri.

### 2.7 İcadın Detaylı Açıklaması

#### 2.7.1 Sistem Mimarisi

Şekil 1'de gösterildiği gibi, sistem 100 numaralı **bağlam kodlayıcıyı**, 110 numaralı **ND projeksiyon modülünü**, 120 numaralı **bulut kodlayıcıyı**, 130 numaralı **tahmin modülünü**, 140 numaralı **tokenizasyon modülünü** ve tercihen 150 numaralı **yorumlama modülünü** içerir.

**(1) Tokenizasyon Modülü (140)**

Modül 140, byte-level Byte Pair Encoding (BPE) kullanır. Özel token'lar `<pad>`, `<eos>`, `<unk>` tanımlıdır; sözlük boyutu 5.000–50.000 aralığındadır. Byte-level ön-tokenizer sayesinde herhangi bir yazı sistemi temsil edilebilir ve **OOV (sözlük dışı) sorunu oluşmaz**.

Agglütinatif dillerde bu modül kritik bir işlev görür: "bahçelerden" sözcüğü `bah` + `ç` + `eler` + `den` subword token'larına ayrılır. Bu subword'ler, aşağıda açıklanan bulutta aynı morfolojik köke aittirler ve uzaysal olarak kümelenirler (Şekil 8).

**(2) Bağlam Kodlayıcı (100)**

Modül 100, L katmanlı (tercihen L = 2–8) bir transformer encoder'dır; d_model boyutlu (tercihen 128–512) gizli uzay kullanır. Girdisi token dizisi [t₁, …, tₙ], çıktısı bağlamsal vektörler c₁, …, cₙ ∈ ℝ^d_model'dir. Görevi, ham token embedding'lerini bağlama duyarlı vektörlere dönüştürmektir.

Bu modül, ön teknikte bilinen herhangi bir bağlam kodlayıcı olabilir (transformer, RNN, lineer dikkatsiz mimariler); icadın karakterize edici özellikleri bu modülün ötesindedir.

**(3) ND Projeksiyon Modülü (110)**

Modül 110, her bağlamsal vektör cᵢ ∈ ℝ^d_model'ı D boyutlu (3 ≤ D ≤ 64) öklid uzayda bir noktaya pᵢ ∈ [-1,1]^D dönüştürür. Tercihen modül 110, d_model → d_model → D boyutlu iki katmanlı bir MLP ve Tanh aktivasyonu kullanır.

**Kritik teknik özellik:** Tanh aktivasyonu tüm noktaları [-1,1]^D hiperküpüne hapseder. Bu, (a) bulutun sınırlı ve cümleler arası karşılaştırılabilir olmasını, (b) çift geometrisi özelliklerinin (mesafe, açı) sınırlı aralıklarda ve sayısal olarak kararlı kalmasını, (c) farklı cümlelerin bulutlarının aynı referans çerçevesinde karşılaştırılmasını sağlar.

**(4) Bulut Kodlayıcı (120)**

Modül 120, nokta bulutu {p₁, …, pₙ} ⊂ ℝ^D'ünden bir küme durumu vektörü cloud_state ∈ ℝ^d_cloud (tercihen d_cloud = 256–512) üretir. Üç alt-bileşen içerir:

**(a) Nokta MLP (121):** Her nokta pᵢ, paylaşılan ağırlıklı bir MLP ile fᵢ ∈ ℝ^d_cloud nokta özelliğine dönüştürülür. Tüm noktalar aynı ağırlıkları kullanır.

**(b) Çift MLP (122):** Her nokta çifti (pᵢ, pⱼ) için bir girdi vektörü oluşturulur:

```
x_ij = [ p_i − p_j ,  ‖p_i − p_j‖ ,  cos(θ_ij) ,  ‖p_i‖ ,  ‖p_j‖ ]  ∈  ℝ^(D+4)
```

- `p_i − p_j` (D boyut): yön ve büyüklük;
- `‖p_i − p_j‖` (1 boyut): öklid mesafe;
- `cos(θ_ij)` (1 boyut): uzayın kökeninden iki noktaya çizilen vektörler arasındaki açının kosinüsü;
- `‖p_i‖`, `‖p_j‖` (2 boyut): noktaların kökenden uzaklıkları.

D = 16 için toplam girdi 20 boyuttur. Dikkat edilmelidir ki çift MLP, her nokta çifti için çalıştığından O(N²) karmaşıklığa sahiptir; bu, self-attention mekanizmasının asintotik karmaşıklığıyla aynı mertebededir. İkisi arasındaki fark, çift başına işlenen **temsil boyutundadır**: çift geometrisi girdisi (D + 4) boyutlu olup, dikkat ağırlıklarının yüksek boyutlu (d_model) iç çarpımlarına kıyasla düşük boyutlu ve geometrik olarak yorumlanabilir bir girdi taşır. Bağlam kodlayıcı attention'sız (O(N)) seçildiğinde (bkz. 2.7.6.4, İddia 19) toplam sistem maliyeti, yüksek-boyutlu attention'a göre düşer.

**Dönelim değişmezliği:** Mesafe ve kosinüs açısı, koordinat sisteminin döndürülmesine karşı değişmezdir. Dolayısıyla bulutun uzayda hangi yöne "baktığı" değil, noktaların **birbirine göre konumu** bilgidir; bu, temsilin rotasyon gürültüsüne karşı dayanıklılığını artırır.

**(c) Küme Birleştirme (123):**

```
cloud_state = Fusion( MaxPool(f_i) , MeanPool(g_ij) )  ∈  ℝ^d_cloud
```

Max-pool, bulutun "en uç" özelliğini; mean-pool, bulutun "ortalama ilişkisini" yakalar. Birleştirme, tercih olarak konkatenasyon + MLP ile yapılır.

**Permütasyon değişmezliği ve sıra bilgisi:** Nokta MLP paylaşıldığından ve simetrik (max/mean) agregasyon kullanıldığından, nokta **kümesi** içindeki sıralama değiştirilse bile cloud_state aynıdır; yani agregasyon adımının kendisi permütasyon değişmezdir. Bununla birlikte, nokta konumları pᵢ = proj(cᵢ) bağlam kodlayıcı (100) aracılığıyla üretildiğinden ve bağlam kodlayıcı pozisyon kodu taşıdığından, nokta konumları **token sırasına dolaylı bağımlıdır**: aynı kelimelerin farklı sıralarla yazıldığı iki cümle, farklı nokta konumlarına ve dolayısıyla farklı bulut geometrisine sahip olur. Böylece cümlenin anlamı, bulutun **geometrik konfigürasyonunda** kodlanır; agregasyon adımının kendisi ise sıraya duyarsızdır.

**(5) Tahmin Modülü (130)**

Modül 130, cloud_state vektöründen bir sonraki token'ı üretir:

- **Çıktı 1:** word_logits ∈ ℝ^vocab — bir sonraki token'ın sözlük dağılımı;
- **Çıktı 2 (131, tercihen):** p_next ∈ ℝ^D — bir sonraki token'ın buluttaki tahmini nokta konumu.

Sistemin modülleri (100–150), bir işlemci tarafından çalıştırılmak üzere bir geçici olmayan, bilgisayar tarafından okunabilir kayıt ortamında depolanan program talimatları olarak da gerçekleştirilebilir.

#### 2.7.2 Eğitim Yöntemi

Toplam kayıp fonksiyonu:

```
L = L_CE(logits, t_next) + λ · L_MSE(p_next, proj(c_next))
```

- **CE terimi:** bir sonraki token'ın doğru tahmini;
- **MSE terimi (tercihen λ = 0.1):** bir sonraki token'ın noktasının, bağlam kodlayıcının ürettiği doğru projeksiyonda (proj(c_next); burada proj(·), 110 numaralı projeksiyon modülüdür) olması.

İkinci terim, modelin yalnızca "hangi kelime gelecek" değil, aynı zamanda "bu kelime bulutta nereye düşecek" öğrenmesini sağlar. Bu, bulut geometrisinin anlamsal olarak tutarlı kalmasını — yani üretim sırasında bulutun dağılmamasını — zorlayan bir düzenleyici görevdir.

#### 2.7.3 Üretim (Inference) Yöntemi

1. Mevcut cümlenin token'larını al;
2. Bağlam kodlayıcı (100) → bağlam vektörleri c₁…cₙ;
3. ND projeksiyon (110) → nokta bulutu {p₁…pₙ};
4. Bulut kodlayıcı (120) → cloud_state;
5. Tahmin modülü (130) → bir sonraki token + p_next;
6. Yeni token'ın noktası p_next buluta eklenir; cloud_state yeniden hesaplanır;
7. Adımlar 4–6, `<eos>` token'ı üretilene kadar iteratif olarak tekrarlanır (Şekil 3).

Her adımda bulut geometrisi güncellenir; bir sonraki tahmin, **güncellenmiş bulutun konfigürasyonundan** yapılır.

#### 2.7.4 Neden Düşük Boyut? — İlişki Bütçesi

D seçimi rastgele değildir. Deneylerimiz aşağıdaki düzeni göstermiştir:

| D | Modelin kodlayabildiği bağımsız ilişki | Gözlenen bölge (KMeans) |
|---|---|---|
| 3 | 1 baskın ilişki (yer-fiil VEYA özne-fiil) | 1–2 |
| 7 | 2–3 ilişki | 2–3 |
| 16 | 3–5 sözdizimsel bölge (edat, işaret, sıfat, zaman, karşılaştırma) | 3 |
| 32 | 5+ bölge | 5 (16D'ye göre %67 artış) |

Her boyut, modelin aynı anda kodlayabileceği **bağımsız ilişki ekseninin** sayısıdır. D artırıldığında bölge sayısı ve ortalama yayılım (spread) **sub-lineer** olarak büyür (16D → 32D: yayılım 1.573 → 2.245, %43). Bu, D'nin bir "ilişki bütçesi" olarak işlev gördüğünün deneysel kanıtıdır ve D'nin cümle karmaşıklığına göre seçilmesine (bkz. İddia 10) temel oluşturur.

D = 3–64 aralığı, D = 3, 7, 16, 32'de doğrulanmış bu ilkenin genellemesidir; aralıktaki herhangi bir tamsayı D değeri icadın kapsamındadır.

#### 2.7.5 Örnek Uygulama (Spesifik Embodiment)

İcat, aşağıdaki spesifik parametrelerle gerçekleştirilmiştir:

- Bağlam kodlayıcı: L = 4 katman, d_model = 256, 8 head;
- Projeksiyon: 256 → 256 → D, Tanh; D = 16 veya 32;
- Bulut kodlayıcı: d_cloud = 256; nokta MLP D → 256, çift MLP (D+4) → 256;
- Tahmin modülü: 256 → 20.000 (BPE sözlük) + 256 → D (yardımcı nokta);
- Toplam: ~21M parametre;
- Korpus: 200.000 Türkçe cümle (Wikipedia), 50.000 cümlelik eğitim kümesi;
- Eğitim: 50 epoch, λ = 0.1.

Sonuçlar:

| Konfigürasyon | PPL | Bölge | Yayılım |
|---|---|---|---|
| Rastgele (21M) | 20.000 | — | — |
| Transformer (21M, aynı korpus) | **[X — başvuru öncesi ölçülüp doldurulmalıdır]** | — | — |
| 16D, 50 epoch | 4.58 | 3 | 1.573 |
| 32D, 50 epoch | **4.08** | **5** | **2.245** |

PPL 4.08, 20.000'lik sözlükte modelin etkin seçim uzayının ~4 token olduğunu — yani tahminlerinde yüksek güven taşıdığını — gösterir. **Karşılaştırma notu:** İcadın avantajı, eşdeğer PPL düzeyinde daha düşük boyutlu (D ≪ d_model) ve geometrik olarak yorumlanabilir bir temsil sağlamasıdır; bu nedenle aynı parametre bütçesi (~21M) ve aynı eğitim korpusuyla eğitilen bir transformer referansının PPL değeri tabloya eklenmelidir. Rastgele seçime göre 4.900 kat iyileşme yalnızca alt sınır olarak raporlanır.

#### 2.7.6 Alternatif Uygulamalar

İcadın kapsamı, aşağıdaki alternatifleri de kapsar:

1. **Alternatif agregasyon:** Max/mean-pool yerine, çift özellikleri üzerinde hafif bir attention (pair-attention) kullanılabilir; permütasyon değişmezlik, simetrik bir normalizasyon ile korunur.
2. **Alternatif çift özellikleri:** Girdi vektörüne çapraz çarpım (D(D−1)/2 boyut) veya Jacobi uzaklığı eklenebilir; iddialardaki "en azından mesafe ve açı" ifadesi bu genişlemeyi kapsar.
3. **Alternatif aktivasyon:** Tanh yerine, noktaları sınırlı bir küreye (‖p‖ ≤ 1) projekte eden normalize edici fonksiyonlar kullanılabilir.
4. **Alternatif (attention'sız) bağlam kodlayıcı:** Transformer yerine RNN, lineer (Mamba benzeri) veya dikkatsiz mimariler kullanılabilir; icadın karakterize edici özellikleri projeksiyon modülünden itibaren tanımlanır. Bağlam kodlayıcı attention'sız (O(N) karmaşıklık) seçildiğinde, toplam sistem — O(N) bağlam kodlama + O(N²) düşük-boyutlu (D+4) çift geometrisi hesabı — O(N²) yüksek-boyutlu (d_model) self-attention tabanlı bir sisteme göre daha düşük hesaplama maliyeti taşır; bu, icadın hesaplama verimliliği etkisinin geçerli olduğu konfigürasyondur (bkz. İddia 19).
5. **Çok cümleli bulut:** Paragraf düzeyinde, cümle bulutlarının bir üst-bulut (hierarchical cloud) oluşturması mümkündür.
6. **Zamansal boyut:** D+1 boyutlu uzayda, token'ların üretim sırasının ek boyut olarak kodlanması (4D bulut) mümkündür.
7. **Dinamik D:** D, cümlenin token sayısına veya eğitimde ölçülen bölge sayısı/yayılım değerine göre seçilir (bkz. İddia 10).

#### 2.7.7 Uygulama Alanları

1. **Dil modelleme:** bir sonraki token tahmini;
2. **Cümle sınıflandırma:** bulut geometrisinden (şekil, yayılım, kompaktlık) cümle tipi;
3. **Anlam benzerliği:** iki cümlenin bulut durumu vektörleri arasındaki mesafe;
4. **Morfolojik analiz:** subword kümelenmesinden kök/ek ayrımı;
5. **Dil karşılaştırması:** farklı dillerin bulut geometrisinin karşılaştırılması;
6. **Hata tespiti:** anormal bulut geometrisi (aykırı nokta, anormal yayılım) → olası yazım/dilbilgisi hatası.

İcat, dil modelleme API'ları, çeviri sistemleri, metin üretimi, dil analitiği ve edge cihazlar dahil olmak üzere sanayide uygulanabilir.

---

## 3. İDDİALAR

### Bağımsız İddialar

**İddia 1**

Ön teknikte bilinen, bir token dizisinin her token'ı için bir bağlam vektörü üreten bir bağlam kodlayıcıyı içeren bir dil modelleme sistemi,

**karakterize edici özellik olarak**

(a) her adı geçen bağlam vektörünü, 3 ≤ D ≤ 64 boyutlu öklid uzayda bir noktaya projekte eden ve tüm noktaları doygun (saturating) bir aktivasyon fonksiyonu ile sınırlı bir hiperküp içinde tutan bir projeksiyon modülü (110);

(b) permütasyon değişmez bir bulut kodlayıcı (120), bu kodlayıcı:
   - her nokta için paylaşılan ağırlıklı bir nokta MLP'si (121),
   - her nokta çifti için, adı geçen çiftin iki noktası arasındaki mesafesi ve adı geçen iki noktanın köken vektörleri arasındaki açının kosinüsü olmak üzere en azından çiftin geometrik özelliklerini içeren bir girdi vektöründen bir çift özelliği üreten bir çift MLP'si (122),
   - adı geçen nokta ve çift özelliklerini simetrik bir agregasyonla birleştirerek bir küme durumu vektörü üreten bir birleştirme modülü (123)
içeren;

(c) adı geçen küme durumu vektöründen bir sonraki token'ı tahmin eden bir tahmin modülü (130)

içermekte olup,

bir sonraki token'ın tahmini, token'ların sıralı attention hesabından değil, adı geçen noktaların geometrik konfigürasyonundan yapılmakta; adı geçen D boyutlu nokta temsilinin, d_model boyutlu bağlam vektörlerinden (D < d_model) daha düşük boyutlu olması sayesinde geometrik temsilin bellek maliyeti, bağlam temsiline göre azalmakta ve adı geçen mesafe, açı ve küme istatistiklerinin doğrudan hesaplanabilir olmasıyla model kararının geometrik olarak yorumlanabilir ve denetlenebilir olması sağlanmaktadır.

**İddia 2**

Ön teknikte bilinen, bir cümlenin token'larını bağlamsal vektörlere dönüştürme adımını içeren bir dil modelleme yöntemi,

**karakterize edici özellik olarak**

(a) her bağlamsal vektörü, 3 ≤ D ≤ 64 boyutlu öklid uzayda, sınırlı bir hiperküp içinde kalan bir noktaya projekte etme;

(b) her adı geçen noktayı, paylaşılan ağırlıklı bir fonksiyonla bir nokta özelliğine dönüştürme;

(c) her nokta çifti için, adı geçen çiftin iki noktası arasındaki mesafesi ve adı geçen iki noktanın köken vektörleri arasındaki açının kosinüsü olmak üzere en azından çiftin geometrik özelliklerini hesaplama;

(d) adı geçen nokta özelliklerini ve çift özelliklerini simetrik agregasyonla birleştirerek permütasyon değişmez bir küme durumu vektörü oluşturma;

(e) adı geçen küme durumu vektöründen bir sonraki token'ı tahmin etme

adımlarını içermekte olup, adı geçen (d) adımındaki simetrik agregasyon permütasyon değişmezdir; nokta konumlarının, bağlamsal vektörler aracılığıyla token sırasına dolaylı bağımlı olması sayesinde cümlenin anlamı, nokta bulutunun geometrik konfigürasyonunda kodlanmaktadır.

**İddia 3**

Bir bilgisayar programı, bir işlemci tarafından çalıştırıldığında İddia 2'deki yöntemin adımlarını gerçekleştiren program talimatlarını içermekte olup, adı geçen program talimatları, nokta bulutu geometrisinden bir sonraki token'ın tahmin edilmesini ve bulut katmanında düşük boyutlu temsil yoluyla bellek ve hesaplama tasarrufu sağlanmasını sağlamaktadır.

### Bağımlı İddialar

**İddia 4:** İddia 1'e bağımlı, burada çift MLP'si (122), her nokta çifti (pᵢ, pⱼ) için [pᵢ − pⱼ, ‖pᵢ − pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖] biçiminde D + 4 boyutlu bir girdi vektörü kullanmakta olup, bu girdinin mesafe ve kosinüs bileşenleri koordinat sisteminin döndürülmesine karşı değişmezdir.

**İddia 5:** İddia 1'e bağımlı, burada projeksiyon modülü (110), d_model → d_model → D boyutlu iki katmanlı bir MLP ve Tanh aktivasyonu kullanmakta olup, noktalar [-1,1]^D hiperkübünde sınırlıdır.

**İddia 6:** İddia 1'e bağımlı, burada birleştirme modülü (123), nokta özellikleri üzerinde max-pool ve çift özellikleri üzerinde mean-pool alarak bu iki özelliği birleştirmektedir.

**İddia 7:** İddia 1'e bağımlı, burada tahmin modülü (130) ayrıca bir sonraki token'ın D boyutlu uzaydaki tahmini nokta konumunu (p_next) üretmekte olup, sistem eğitim sırasında, bir sonraki token'ın doğru nokta konumuna olan ortalama kare hatayı minimize eden bir yardımcı kayıp terimini, çapraz entropi kayıp terimiyle λ (0 < λ ≤ 1) katsayısıyla birleştirmektedir.

**İddia 8:** İddia 7'ye bağımlı, burada 0.01 ≤ λ ≤ 1'dir, tercihen λ = 0.1'dir.

**İddia 9:** İddia 1'e bağımlı, burada üretim sırasında her adımda, yeni üretilen token'ın tahmini noktası buluta eklenmekte, küme durumu vektörü güncellenmekte ve bir sonraki token'ın tahmini güncellenmiş bulutun geometrik konfigürasyonundan yapılmaktadır.

**İddia 10:** İddia 1'e bağımlı, burada sistem, cümlenin token sayısı ve/veya ölçülen bulut karmaşıklığı — bölge sayısı, ortalama yayılım — esas alınarak D değerini önceden tanımlı bir değer kümesinden seçen bir boyut seçme modülü daha içermektedir.

**İddia 11:** İddia 1'e bağımlı, burada sistem, morfolojik birimleri subword token'ları olarak üreten byte-level BPE tabanlı bir tokenizasyon modülü (140) daha içermekte olup, projeksiyon modülü (110), aynı morfolojik kökten türeyen subword token'larının D boyutlu uzayda birbirine yakın konumlara projekte edilecek şekilde eğitilmiştir.

**İddia 12:** İddia 11'e bağımlı, burada tokenizasyon modülü (140) agglütinatif bir dil — Türkçe, Fince veya Macarca — için eğitilmiş olup, projeksiyon modülü (110), morfolojik ekleri bulutta kök alt-kümesinden farklı yönlerde ayrışan alt-kümeler olarak temsil edecek şekilde eğitilmiştir.

**İddia 13:** İddia 1'e bağımlı, burada sistem, bulut geometrisinin, önceden tanımlı eşik değerleri aşıldığında aykırı değerlerini — nokta düzeyinde kökenden anormal uzaklık veya küme düzeyinde anormal yayılım — tespit ederek dilbilgisi veya yazım hatası göstergesi üreten bir hata tespiti modülü daha içermektedir.

**İddia 14:** İddia 1'e bağımlı, burada sistem, iki cümle için üretilen küme durumu vektörleri arasındaki mesafeyi hesaplayarak anlam benzerliği ölçüsü üreten bir benzerlik modülü daha içermektedir.

**İddia 15:** İddia 1'e bağımlı, burada L 2–8, d_model 128–512, D {16, 32} ve d_cloud 256–512 aralığında/değerlerinde seçilmiştir.

**İddia 16:** İddia 1'e bağımlı, burada sistem, nokta konumlarını, çift mesafe matrisini ve küme istatistiklerini — yayılım, merkez, kompaktlık — işleyerek kullanıcıya sunan bir yorumlama modülü (150) daha içermektedir.

**İddia 17:** İddia 3'e bağımlı, burada program talimatları, geçici olmayan bir bilgisayar tarafından okunabilir kayıt ortamında depolanmıştır.

**İddia 18:** İddia 2'ye bağımlı, burada (e) adımından sonra, tahmin edilen token bağlam vektörüne dönüştürülüp (a) adımıyla bir noktaya projekte edilmekte, adı geçen nokta buluta eklenmekte ve adımlar (b)–(e), bir son token'ı üretilene kadar iteratif olarak tekrarlanmaktadır.

**İddia 19:** İddia 1'e bağımlı, burada bağlam kodlayıcı (100), self-attention mekanizması içermeyen — dikkatsiz (attention-free), lineer dikkat (linear attention) veya yeniden kullanımlı (recurrent) — O(N) karmaşıklıktaki bir mimaridir; böylece toplam sistem, O(N) bağlam kodlama ile O(N²) düşük-boyutlu çift geometrisi hesabından oluşmakta ve d_model boyutlu O(N²) self-attention tabanlı bir sisteme göre daha düşük hesaplama maliyeti taşımaktadır.

---

## 4. ÖZET (ABSTRACT)

Bir cümlenin her token'ı, 3–64 boyutlu öklid uzayda sınırlı bir hiperküp içinde bir noktaya projekte edilir. Cümlenin anlamı, bu nokta bulutunun geometrisinde — noktalar arası mesafe, açı ve küme yayılımı — kodlanır. Simetrik agregasyonlu (permütasyon değişmez) bir PointNet benzeri kodlayıcı, nokta ve çift geometrisinden bir küme durumu vektörü üretir; bir sonraki token bu vektörden tahmin edilir. Nokta konumları bağlam kodlayıcı aracılığıyla sıra-duyarlı olduğundan bulut geometrisi token sırasına dolaylı bağımlıdır; agregasyon adımının kendisi ise permütasyon değişmezdir. Sistem, düşük boyutlu (D ≪ d_model) ve geometrik olarak yorumlanabilir — mesafe, açı ve bölge doğrudan hesaplanabilir — bir temsil sağlar. Agglütinatif dillerde byte-level BPE subword token'ları bulutta morfolojik köklere göre doğal kümelenme oluşturur. 21M parametreli örnek uygulama, 20K BPE sözlüklü Türkçe korpusunda PPL 4.08 elde etmiştir; D = 16 → 32 geçişinde bölge sayısı 3 → 5, yayılım %43 artmıştır.

---

## 5. ÇİZİMLER (Ayrı Sayfalar)

TÜRKPATENT, her çizimi ayrı sayfada, siyah-beyaz çizgi, referans numaraları ve şekil başlığı ile ister. Aşağıdaki 8 çizim, `docs/figures/` klasöründe SVG olarak hazırlanmıştır; resmî dosyaya her biri ayrı sayfa olarak eklenir.

| Sayfa | Şekil | Başlık | Dosya |
|---|---|---|---|
| 1 | Şekil 1 | Sistem mimarisi (100–150) | `docs/figures/sekil_1_sistem_mimarisi.svg` |
| 2 | Şekil 2 | "kedi bahçede uyudu" 3D nokta bulutu (200, 210) | `docs/figures/sekil_2_3d_bulut.svg` |
| 3 | Şekil 3 | Bulut evrimi 1→4 nokta (300) | `docs/figures/sekil_3_bulut_evrimi.svg` |
| 4 | Şekil 4 | 16D ve 32D bölge haritaları (400, 410) | `docs/figures/sekil_4_bolge_haritalari.svg` |
| 5 | Şekil 5 | Kelime bölge coğrafyası (500) | `docs/figures/sekil_5_kelime_cografyasi.svg` |
| 6 | Şekil 6 | Çift geometrisi vektörü bileşenleri (600–630) | `docs/figures/sekil_6_cift_geometrisi.svg` |
| 7 | Şekil 7 | Eğitim eğrisi PPL (700) | `docs/figures/sekil_7_egitim_egrisi.svg` |
| 8 | Şekil 8 | Morfolojik kümeleşme (800, 810) | `docs/figures/sekil_8_morfolojik_kumelenme.svg` |

> **Not:** Çizimler, başvuru öncesi PDF çizim formatında (siyah-beyaz, referans numaralı) ayrı sayfalar halinde derlenmelidir. SVG dosyaları kaynak olarak saklanır.

---

*Bu dosya, TÜRKPATENT patent başvurusu için teknik metindir.*
*Hazırlayan: [İcatçı Adı]*
*Revizyon: v2.2*
*Tarih: Ekim 2026*
