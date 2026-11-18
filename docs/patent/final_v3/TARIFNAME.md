# TARİFNAME

**Buluşun başlığı:** Nokta bulutu temelli çift kullanımlı temsil üzerinden görev çıktısı ve gömülü açıklanabilirlik üreten yöntem, sistem ve kayıt ortamı

## 1. Buluşun teknik alanı

Buluş, makine öğrenmesi modellerinde görev çıktısı üretimi ile üretilen kararın açıklanmasına ilişkindir. Özellikle, bir bağlam kodlayıcısının ürettiği D-boyutlu nokta bulutu temsilinin hem görev çıktısının hem de o kararın açıklamasının tek kaynağı olarak kullanıldığı bilgisayarla uygulanan yöntemlere, bu yöntemleri yürüten bilgisayarlara ve üzerinde yürütülebilir komutlar taşıyan bilgisayarca okunabilir kayıt ortamlarına ilişkindir.

Uygulama alanları, açıklanan kararın denetlenmesinin ölçülebilir bir maliyet veya yükümlülük konusu olduğu uç (kenar) bilişim, finansal işlem sınıflandırması, makine sağlığı izleme ve metin işleme alanlarıdır. Buluş, bir sınıfın doğruluğunun başka bir sınıftan daha yüksek olduğu bir iddia taşımaz; iddia, aynı eğitimli temsil üzerinden görev çıktısı ile açıklama verisinin ayrı bir açıklama modeli olmadan ve taşınabilir bir paket içinde üretilebilmesidir.

## 2. Önceki teknik ve açığı

Açıklanabilir makine öğrenmesinde yerleşik yaklaşım, açıklamanın model-sonrası (post-hoc) bir araçla üretilmesidir: eğitimli bir sınıflandırıcının dışında, girdi özellikleri üzerinde çalışan ayrı bir atıf aracı çalıştırılır; görev çıktısı ile açıklama farklı sistemlerden gelir. Bu yaklaşımda üç teknik açık, işleyen prototiplerde ölçülebilir biçimde görülür.

**(a) Model-sonrası açıklamanın kendi kararlılığı, yeniden eğitimle sınırlıdır.** Beş ayrı eğitim tohumu ile (kaynak: aml/results/e5_stability.json:8-12) yeniden eğitilmiş aynı mimari üzerinde, model-sonrası ağaç atıf aracının (TreeExplainer, kurulu sürüm 0.54.1; kaynak: aml/results/e5_stability.json:25) sabit bir test alt-örneğinde (n=400; kaynak: aml/results/e5_stability.json:14) ürettiği atıf sıralamasının birinci-sıra kararlılığı 0.1025, ilk-üç kümesi jaccard benzerliği 0.43405 ve atıf sıralaması Spearman korelasyonu 0.65105 olarak ölçülmüştür (kaynak: aml/results/e5_stability.json:41-43)
Aynı veri ve aynı beş tohum üzerinde, modelin kendi geometrik temsilinden türeyen en-yakın-bölge etiketinin tohumlar arası kararlılığı 0.325 olarak ölçülmüştür (kaynak: aml/results/e5_stability.json:29; tanım: aml/results/e5_stability.json:30)
Bu iki sayı aynı protokol içinde ölçülmüş olsalar da aynı nicelik değildir; birinci sayı atıf sıralamasının, ikinci sayı bir etiketin yeniden eğitimler arası sabit kalma oranıdır ve aralarında oran veya üstünlük iddiası kurulmaz
Açık şudur: model-sonrası atıf, modelin kendisi kadar kararlı olmayan ayrı bir nicelik üretir ve bu niceliğin denetlenmesi ayrıca iş gerektirir.

**(b) Açıklama, uç aygıtta ayrı bir yığın gerektirir.** Model-sonrası atıf aracının çalıştırılması, dağıtılan modelin yanına ayrı bir araç ve ayrı bir hesaplma adımı ekler
Ölçülen örnek: aynı protokolde, model-sonrası ağaç atıf yolunun tohum başına ortalaması 0.7403 saniye (kaynak: aml/results/e5_stability.json:34057), geometrik temsilin kendi açıklama verisini üreten yolunun tohum başına ortalaması 0.3545 saniyedir (kaynak: aml/results/e5_stability.json:38)
Bu iki değer farklı iş kümelerinin süresidir ve genel bir karmaşıklık oranı iddiası taşımaz; ölçülen olgu, açıklamanın ayrı bir eğitim/artımlama adımı olarak süre ve yığın eklemesidir.

**(c) Sürüklenebilirlik (drift) denetimi de ayrı bir araç ister.** Bölge paylarının zaman içinde kaymasının denetimi, temsilin kendi yapısından değil, dışarıdan tanımlanmış özellik pencerelerinden türetilir; bu durumda kaymanın ölçülmesi için ikinci bir izleme hattı kurulması gerekir.

Bu üç açığın ortak nedeni, açıklama verisinin temsilin kendisinde bulunmayıp temsilin dışına ikinci bir sistem olarak eklenmesidir.

## 3. Buluşun kısa açıklaması ve teknik etkisi

Buluşta görev girdisi (sıralı token dizisi veya bir tabular kaydın birden çok ölçülen alan değeri) bir bağlam kodlayıcı ile D-boyutlu gösterimlere dönüştürülür ve N noktadan oluşan bir nokta bulutu kurulur. Noktalar için nokta özellikleri, nokta çiftleri için iki noktanın konumlarına bağlı çift-geometri özellikleri hesaplanır; her iki özellik kümesi girdi sırasından bağımsız simetrik bir küme agregasyonu ile tek bir küme gösteriminde birleştirilir ve bir okuyucu bu gösterimden görev çıktısını üretir.

Buluşun ayırt edici özelliği, aynı eğitimli modelin aynı nokta bulutu üzerinde üç açıklama işleminden en az birini uygulamasıdır: (i) nokta çıkarma etkisi, (ii) bölge isimlendirme, (iii) bölge kayması. Açıklama verisi, görev çıktısından ayrı bir model, ayrı bir eğitim veya ayrı bir girdi gerektirmez; görev çıktısı ile açıklama verisi aynı küme gösteriminden türeyen birlikte çıktılardır.

Teknik etki, ölçülebilir eksenler üzerinden ve işleyen prototipin ölçümleriyle ifade edilir:

- **Dağıtılan bellek ayak izi.** Okuyucunun INT8 kantize edilmesi ve nicelleştirme ile çözümleme tablolarının tek kendi kendine yeten pakette taşınmasıyla, uç aygıta indirilen paket 528.9 KB FP32'den 138.2 KB INT8'e inmiştir (kaynak: fraud/results/edge_eval_full_scale.json:15-16). Paket, görev çıktısının yanında insan-okunur açıklama etiketini de üretir: girdi değerlerini model girdi kimliklerine eşleyen kantize tablo 5.4 KB, model girdi kimliklerini insan-okunur etiketlere eşleyen çözümleme tablosu 16.5 KB ve 221 girdi kimliği bu pakette yer alır (kaynak: fraud/results/deploy/manifest.json:21-24; eşlemeler: fraud/results/deploy/manifest.json:97-98).
- **Gecikme ve geçiş hızı.** INT8 okuyucu üzerinde işlem başına gecikme p50 = 0.015913 ms, p99 = 0.017278 ms, toplam verim 62845 işlem/saniye olarak ölçülmüştür (kaynak: fraud/results/edge_eval_full_scale.json:23-25). Kantize ONNX yolunda işlem başına gecikme 0.025782 ms ve verim 39056 işlem/saniye ölçülmüştür (kaynak: fraud/results/edge_eval_full_scale.json:50-51); kendi kendine yeten paketin öz-testinde 19963 satır üzerinden 54063 işlem/saniye ölçülmüş ve FP32 referansla en büyük mutlak fark 1.4603e-06 olarak bulunmuştur (kaynak: fraud/results/deploy/manifest.json:27-30).
- **Açıklama maliyetinin sınırlı tutulması.** Nokta çıkarma etkisi N+1 ilerleyiş geçişiyle üretilir ve maskelenmiş bulutlar toplu girdi olarak okuyucuya birlikte verilir. Hesaplma yükü, nokta sayısının artmasıyla çift sayısının karesel artışı yoluyla büyüdüğü için N bağlam uzunluğu ile sınırlı tutulur; bu karesel büyüme ölçülmüştür: etkin nokta sayısı k=13 için işlem başına 2.522.200 çarp-topla işlemi, k=50 için 34.091.488 çarp-topla işlemi (kaynak: k değerleri ve MAC sayıları fraud/results/edge_eval_full_scale.json:29-30, fraud/results/edge_eval_full_scale.json:41-42).
- **Kantizasyon doğruluk maliyetinin bilinmesi.** INT8 geçişinin görev doğruluğundaki düşüşü ölçülmüş ve sınır olarak kabul edilmiştir: doğal oranlı 4.700.000 satırlık ayrık test kümesinde FP32 AUC 0.9450746 iken INT8-tensor 0.9364812 ve INT8-kanal 0.9367114 ölçülmüş, farklar tensor için -0.008593 ve kanal için -0.008363 olarak kaydedilmiştir (kaynak: fraud/results/edge_eval_full_scale.json:59, fraud/results/edge_eval_full_scale.json:62-66).
- **Denetim kararlılığı.** Açıklama verisi temsilin kendi bölgesinden türediği için, yeniden eğitimler arası bölge etiketi kararlılığı tek sayıyla denetlenebilir hâle gelir: beş tohumda 0.325 (kaynak: aml/results/e5_stability.json:29).
- **Eğitim yükü.** Nokta konumları gradyan hesaplanmaksızın, sıfırıncı mertebeden kabul/geri alma ile daraltılır; aday konumun görev yitimi değişimi eşik üzerinden kabul edilir (kaynak: quann_cloud/src/qcloud_quann.py:212, quann_cloud/src/qcloud_quann.py:220) ve okuyucu gradyan tabanlı geri yayılım ile eğitilir.

## 4. Şekillerin kısa açıklaması

- **Şekil 1** (fig1_sistem_mimarisi.svg) — şematiktir: sistem mimarisi; tek nokta bulutundan görev çıktısı ile açıklanabilirlik üçlüsü dalı. Çizimi üreten betik: docs/patent/final_v3/generate_figures_v3.py:63-81.
- **Şekil 2** (fig2_cift_geometri.svg) — şematiktir: nokta çifti ve çift-geometri özelliğinin bileşenleri [fark vektörü, çift mesafesi (Öklid normu), kosinüs benzerliği, birinci noktanın norm vektörü, ikinci noktanın norm vektörü]. Çizimi üreten betik: docs/patent/final_v3/generate_figures_v3.py:85-110.
- **Şekil 3** (fig3_simetrik_agregasyon.svg) — şematiktir: maksimum havuzlama (nokta) ile ortalama havuzlama (çift) birleştirmesi; kesikli yol: O(N) küresel-istatistik varyantı. Çizimi üreten betik: docs/patent/final_v3/generate_figures_v3.py:114-132.
- **Şekil 4** (fig4_nokta_cikarma.svg) — şematiktir: nokta çıkarma etkisi, K+1 ileri geçiş ve e_k = s_tam - s_-k katkı tanımı. Çizimi üreten betik: docs/patent/final_v3/generate_figures_v3.py:136-155.
- **Şekil 5** (fig5_bolge_isimlendirme.svg) — şematiktir: kümeleme (KMeans) ile bölge istatistiklerinden insan-okunur bölge etiketi üretimi. Çizimi üreten betik: docs/patent/final_v3/generate_figures_v3.py:159-182.
- **Şekil 6** (fig6_bolge_kaymasi.svg) — **ölçümlüdür**: (a) şikâyet korpusunda bölge nokta-payı kayması (kaynak veri: complaints/results/complaints_scale_stability.json:64 share_delta_drift_minus_train alanı, uç değerler complaints/results/complaints_scale_stability.json:66, complaints/results/complaints_scale_stability.json:72); (b) C-MAPSS ayar kaymasında alarm oranı 0.0117'den 0.0281'e, orantı 2.4x (kaynak veri: maintenance/results/e1_cmapss.json:189-192 drift_demo bloku). Çizimi üreten betik: docs/patent/final_v3/generate_figures_v3.py:186-221.
- **Şekil 7** (fig7_int8_paket.svg) — **ölçümlüdür**: (a) AUC: FP32 0.9451 / INT8-tensor 0.9365 / INT8-kanal 0.9367, n=4.7M; (b) paket ayak izi FP32 528.9 KB'dan INT8 138.2 KB'ye (kaynak veri: fraud/results/edge_eval_full_scale.json:15-16, fraud/results/edge_eval_full_scale.json:59, fraud/results/edge_eval_full_scale.json:62-66). Çizimi üreten betik: docs/patent/final_v3/generate_figures_v3.py:224-256.
- **Şekil 8** (fig8_window_bulut.svg) — şematiktir: cümle-bulut ile komşu pencere-bulutlarından ContextCloudLM hiyerarşisi. Çizimi üreten betik: docs/patent/final_v3/generate_figures_v3.py:259-274.

Şekil 1, 2, 3, 4, 5 ve 8 yapının tanımıdır ve ölçüm içermez; ölçüm taşıyan şekiller Şekil 6 ve Şekil 7 ile sınırlıdır.

## 5. Buluşun detaylı açıklaması

### 5.1 Sistem mimarisi (Şekil 1)

Şekil 1'de şematize edilen zincir sırasıyla şu bileşenlerden kuruludur: görev girdisi; görev girdisini işleyen bağlam kodlayıcı; bağlamı kodlanmış her gösterimi D-boyutlu uzaya izdüşüren nokta projeksiyonu; noktalar ve nokta çiftleri için geometri özellikleri hesaplayan çift-geometri kodlayıcısı; bu özellikleri tek bir küme gösteriminde birleştiren agregasyon modülü; küme gösteriminden görev çıktısını üreten okuyucu; aynı nokta bulutu üzerinden açıklama verisi üreten yorumlama motoru; ve görev çıktısı ile açıklama verisini birlikte sunan çıkış arayüzü (çizim kaynağı: docs/patent/final_v3/generate_figures_v3.py:63-81).

İki nokta yapısal olarak önemlidir. Birincisi, temsil bir küme olarak işlenir: girdi sıralaması agregasyona taşınmaz ve bağlamanın dolmadığı konumlara karşılık gelen noktalar ile bu noktalardan oluşan çiftler bir maske sinyali ile işlem dışı bırakılır (kaynak: cloudlm/model.py:47). İkincisi, yorumlama motoru ayrı bir model değildir; görev çıktısını üreten okuyucunun kendi ileri geçişini, içindeki noktalar değiştirilmiş hâlde yeniden çalıştırarak açıklama verisini üretir. Bu nedenle açıklama, temsilin dışına eklenen ikinci bir sistem değil, temsilin ikinci bir kullanımıdır.

### 5.2 Nokta bulutunun oluşması ve D boyutu (Şekil 1 ve Şekil 2)

Görev girdisi iki biçimde gelir. Sıralı bir token dizisinde bağlam kodlayıcının her konumda ürettiği gösterim D-boyutlu bir vektöre projekte edilir ve N nokta bağlam uzunluğu kadar olur. Bir tabular kayıtta ise kaydın birden çok ölçülen alan değeri nokta bulutunun noktalarını oluşturur; bu durumda alan sayısı N'i ve dolayısıyla çift sayısını belirler.

Ölçülen uygulama örneklerinde D boyutu görev alanına göre seçilmiştir: tabular dolandırılma görevinde D = 8 ve bulut durum boyutu 128 (kaynak: fraud/results/full_scale_hybrid.json:56-57), dil görevinde karşılaştırılan iki yapılandırma D = 32 ve D = 64 (kaynak: logs/run_r4_resume.log:67, logs/run_r4_resume.log:70), makine sağlığı görevinde gömme boyutu 48 (kaynak: maintenance/results/e1_cmapss.json:18). Seçilen D boyutu 3 ile 64 arasında kalır. Tabular embodimentda model girdi kimliği sayısı 221, daraltılmış nokta sayısı 214 ve etkin nokta sayısı K = 13 olarak kaydedilmiştir (kaynak: fraud/results/edge_eval_full_scale.json:12-13; fraud/results/full_scale_hybrid.json:61).

### 5.3 Çift-geometri özelliği (Şekil 2)

Şekil 2'de şematize edildiği gibi, bir nokta çifti (p_i, p_j) için çift-geometri özelliği beş bileşenin birleştirilmesinden oluşur: iki nokta vektörü arasındaki fark vektörü, fark vektörünün Öklid normu olan çift mesafesi, iki nokta vektörü arasındaki kosinüs benzerliği, birinci noktanın norm vektörü ve ikinci noktanın norm vektörü. Fark vektörü ve mesafe çiftin göreli konumunu, kosinüs benzerliği çiftin açısal ilişkisini, iki norm bileşeni ise her iki noktanın bulut merkezine göre mutlak konum bilgisini taşır (çizim kaynağı: docs/patent/final_v3/generate_figures_v3.py:85-110).

Bu bileşenlerin birleştirilmesiyle çift-geometri özelliğinin boyutu 3D + 2 olur; kodda çift girdi genişliği bu biçimde tanımlanmıştır (kaynak: cloudlm/model.py:63-64). Mesafe, kosinüs paydası ve norm bileşenlerinin hesaplanması: cloudlm/model.py:158, cloudlm/model.py:161, cloudlm/model.py:165, cloudlm/model.py:168. Buluş metninde bu genişlik yalnızca 3D + 2 cebirsel toplamı olarak anılır; "D + 4" biçimindeki sayısal etiket bu metinde kullanılmaz, çünkü kodda norm bileşenleri D boyutunda vektörler olarak taşınır (kaynak: cloudlm/model.py:63-64). "D + 4" sayısal etiketi çizim başlığından 2026-11-18 tarihinde çıkarılmıştır; Şekil 2 bileşen genişlikleri (D, 1, 1, D, D) ve R^{3D+2} toplamı kod tanımıyla birebir çizilmiştir (çizim kaynağı: docs/patent/final_v3/generate_figures_v3.py:85-110).

### 5.4 Simetrik küme agregasyonu ve maskeleme (Şekil 3)

Şekil 3'te şematize edilen agregasyon iki yoldan oluşur. Nokta yolunda nokta özellikleri noktalar üzerinden maksimum havuzlama ile indirgenir (kaynak: cloudlm/model.py:153). Çift yolunda çift-geometri özellikleri nokta çiftleri üzerinden ortalama havuzlama ile indirgenir (kaynak: cloudlm/model.py:177). İki yolun çıktısı bir birleştirme katmanında tek bir küme gösteriminde toplanır (kaynak: cloudlm/model.py:179).

Maskeleme her iki yolda da uygulanır. Nokta yolunda bağlamanın dolmadığı konumlara karşılık gelen noktalar havuzlama öncesi geçersiz değere çekilir ve havuzlama sonucu yine maskeyle temizlenir (kaynak: cloudlm/model.py:151-152, cloudlm/model.py:155). Çift yolunda bir çiftin geçerli sayılması için iki ucun da geçerli olması istenir; geçersiz çiftlerin özellikleri sıfırlanır ve ortalama, yalnızca geçerli çift sayısı üzerinden normalleştirilir (kaynak: cloudlm/model.py:171-174). Bu normalleştirme, dolmayan konumların küme gösterimine sayısal katkı vermemesini sağlar.

Her iki havuzlama da noktaların bir küme üzerinden işletildiği girdi sırasından bağımsız işlemler olduğundan, küme gösterimi noktaların numaralanma sırasından bağımsızdır. Bu özellik, tabular bir kaydın alanlarının bir sıralama taşımadığı durumda temsilin tanımlı kalmasını sağlar.

### 5.5 Okuyucu ve görev çıktısı

Küme gösterimi okuyucuya verilir ve görev çıktısı üretilir. Dil görevinde okuyucu bir sonraki tokenın olasılık dağılımını üretir; ileri geçiş maskesiyle birlikte bu biçimde tanımlanmıştır (kaynak: cloudlm/model.py:305) ve küme durumu bulut kodlayıcısından alınır (kaynak: cloudlm/model.py:313). Tabular görevde okuyucu bir sınıf atama skoru üretir; açıklama adımı da aynı okuyucunun logitini temel alır (kaynak: fraud/src/explain_pilot.py:103). Dolayısıyla dil ve tabular görevlerde açıklamanın dayandığı nicelik, görev çıktısını üreten aynı okuyucunun çıktısıdır.

### 5.6 Çok cümleli bağlam: cümle-bulut ve pencere-bulut hiyerarşisi (Şekil 8)

Şekil 8'de şematize edilen varyantta görev girdisi birden çok cümleden oluşan bir metindir ve bağlam kodlayıcı üç katmanda çalışır. Birinci katmanda her cümle bir cümle-bulut durumu olarak kodlanır (kaynak: contextcloud/model.py:20). İkinci katmanda ardışık cümle-bulut durumları sabit bir pencere boyutu ve sabit bir adımla kaydırılarak komşu pencere-bulutları türetilir; pencere boyutu 5 ve adım 2 olarak tanımlanmıştır (kaynak: contextcloud/model.py:121, contextcloud/model.py:127-128). Üçüncü katmanda pencereler arası geometrik büyüklükler bir pencere-geometri kodlayıcısı ile toplanır (kaynak: contextcloud/model.py:196) ve çoklu-cümle küme gösterimi okuyucuya verilir (kaynak: contextcloud/model.py:246) (çizim kaynağı: docs/patent/final_v3/generate_figures_v3.py:259-274).

Bu hiyerarşide pencerelerin sabit boyutu, çift sayısının artışını sınırlayan ikinci bir tasarım düğümüdür: pencere içindeki çift sayısı pencere boyutunun karesi ile, pencere sayısı ise cümle sayısının adım bölümü ile belirlenir.

### 5.7 Küresel-istatistik varyantı ve maliyet sınırı (Şekil 3, kesikli yol)

Şekil 3'ün kesikli yolu, çift-geometri hesabının yerini alan varyanttır. Bu varyantta çift-geometri özellikleri ve çiftler üzerinden ortalama havuzlama yerine nokta bulutunun küresel küme istatistikleri hesaplanır; istatistikler nokta boyutları üzerinden maksimum, minimum, ortalama ve standart sapma havuzlaması (kaynak: purecloudlm/model.py:62) ile bulut merkezi ve merkez sapmasını (kaynak: purecloudlm/model.py:63) içerir. Çift hesabı yapılmadığı veya yaklaşık yapıldığı için maliyet nokta sayısı ile doğrusal orantılıdır (kaynak: purecloudlm/model.py:8-9) (çizim kaynağı: docs/patent/final_v3/generate_figures_v3.py:114-132).

Maliyet sınırının ölçülebilir gerekçesi çift yolunda görülür: etkin nokta sayısı 13'ten 50'ye çıkarıldığında işlem başına çarp-topla işlemi 2.522.200'den 34.091.488'e çıkmıştır (kaynak: k değerleri ve MAC sayıları fraud/results/edge_eval_full_scale.json:29-30, fraud/results/edge_eval_full_scale.json:41-42)
Bu artış çift sayısının N^2 biçiminde artması ile uyumludur: ölçülen çarp-topla işlemi oranı 34091488 / 2522200 = 13.5x, nokta sayısı oranının karesi (50/13)^2 = 14.8x'tir (hesap: fraud/results/edge_eval_full_scale.json:30, fraud/results/edge_eval_full_scale.json:42 değerlerinden); buluşta N sayısı bu nedenle görev girdisinin bağlam uzunluğu ile sınırlı tutulur ve uç aygıt hedefinde çift yolu yerine küresel-istatistik varyantı seçilebilir.

### 5.8 Birinci açıklama işlemi: nokta çıkarma etkisi (Şekil 4)

Şekil 4'te şematize edilen işlemde, görev çıktısının üretildiği nokta bulutu olduğu gibi korunur ve seçilen bir nokta geçici olarak devre dışı bırakılır. Uygulama sırası şöyledir: girdi tokenlarından nokta bulutu üretilir (kaynak: fraud/src/explain_pilot.py:102); maskelenmemiş durumda okuyucu çalıştırılıp taban çıktı alınır (kaynak: fraud/src/explain_pilot.py:103); katkı verilecek nokta kopya bulutta sıfır değere çekilir (kaynak: fraud/src/explain_pilot.py:106-107) ve o noktanın maskesi kapatılır (kaynak: fraud/src/explain_pilot.py:109); okuyucu maskeli bulutla yeniden çalıştırılır (kaynak: fraud/src/explain_pilot.py:110); katkı, taban çıktıdan maskeli çıktının çıkarılmasıyla kaydedilir (kaynak: fraud/src/explain_pilot.py:111). Tanım, logit farkı biçiminde kayıtlıdır: taban logit eksi, ilgili noktası maskelenmiş logit (kaynak: fraud/results/explain_pilot.json:21).

Aynı işlem kantize dağıtımda da çalışır: maskelenmiş bulut türevleri tek bir toplu girdi olarak hazırlanır (kaynak: fraud/pilot/infer.py:97-98), maske her türev için ilgili konum kapatılarak kurulur (kaynak: fraud/pilot/infer.py:100) ve INT8 ONNX oturumu tek çağrıda çalıştırılır (kaynak: fraud/pilot/infer.py:104); katkı logit farkı olarak üretilir (kaynak: fraud/pilot/infer.py:107). Böylece N nokta için N+1 ilerleyiş geçişi tek toplu çağrıya indirgenir ve Şekil 4'teki e_k = s_tam - s_-k tanımı doğrudan ölçülür.

Bu işlemin yorumu, işlenen prototipte dürüstçe sınırlandırılmıştır: üretilen nicelik bir çıkarma duyarlılığıdır, tek-additif atıf değildir; çift-geometri hesabı nedeniyle katkılar toplamsal olmaz (kaynak: fraud/results/explain_pilot.json:23). Katkılar bu nedenle tek bir özelliğin payı olarak değil, o noktanın kaldırılması hâlinde kararda ölçülen değişim olarak sunulur.

### 5.9 İkinci açıklama işlemi: bölge isimlendirme (Şekil 5)

Şekil 5'te şematize edilen işlemde çok sayıda kaydın küme gösterimleri veya nokta bulutu durumları küme merkezli bir kümeleme algoritması ile geometrik bölgelere ayrılır; prototipte KMeans kullanılmıştır (kaynak: complaints/src/explain_region.py:47). Bölge sayısı şikâyet embodimentında 8 olarak kaydedilmiştir (kaynak: complaints/results/region_map.json:2). Sınıflandırma embodimentında bölgeler, sınıfların eğitim kümesindeki gösterim ortalamalarından oluşan prototipler olarak da tanımlanabilir; bu tanım yalnızca eğitim verisi üzerinden kurulur (kaynak: aml/src/e1_explain.py:297).

Her bölgeye insan-okunur etiket, o bölgeye atanan kayıtların girdi sembollerinin sıklık sıralamasından seçilen en az bir sembolün birleştirilmesiyle verilir; prototipte sıralı sembollerin ilk sekizi etiket üretiminde kullanılır (kaynak: complaints/src/explain_region.py:66). Ölçülen bölge etiketlerinden örnekler: "credit report information" ve "credit report consumer" (kaynak: complaints/results/region_map.json:422, complaints/results/region_map.json:1462; etiketler BPE parça göstergeleriyle kayıtlıdır).

Bölge isimlendirmenin denetlenebilir sınırı da kayıtlıdır: geometrik olarak ayrı iki bölge aynı insan-okunur etiketi taşıyabilir; ölçülen örnekte iki ayrı bölgenin etiketi aynıdır (kaynak: complaints/results/region_map.json:422, complaints/results/region_map.json:838). Bu nedenle bölge etiketi tek başına kimlik sayılmaz; denetim, bölge indeksi ile birlikte yapılır.

### 5.10 Üçüncü açıklama işlemi: bölge kayması (Şekil 6)

Şekil 6'da ölçümleri gösterilen işlemde, referans veri grubu ile karşılaştırma veri grubu için aynı geometrik bölgelerdeki nokta payları karşılaştırılır ve pay dağılımının farkı bir olasılık dağılımı yakınsaklık ölçütüyle sayısallaştırılır
Kullanılan iki ölçüt şunlardır: bölgeler arası pay farklarının mutlak değerleri toplamının yarısı olarak tanımlanan tam değişim ölçütü (kaynak: complaints/src/audit_drift.py:36) ve Kullback-Leibler yakınsaklığı (kaynak: complaints/src/audit_drift.py:40)
Ayrıca referans dağılımda payı bir alt eşik altında olan, karşılaştırma dağılımında ise payı bir üst eşiği aşan bölgeler yeni ortaya çıkan bölge olarak işaretlenir (kaynak: complaints/src/audit_drift.py:44).

Ölçülen hafta örnekleri: 2025-02-03/2025-02-09 haftasında tam değişim 0.090370 ve Kullback-Leibler 0.034140 (kaynak: complaints/results/drift_report.json:21, complaints/results/drift_report.json:33-34); izleyen 2025-02-10/2025-02-16 haftasında tam değişim 0.024691 ve Kullback-Leibler 0.001897 (kaynak: complaints/results/drift_report.json:37, complaints/results/drift_report.json:49-50). Yöntem tanımı gereği sürükleme etiketi kullanılmaz; referans, eğitim penceresinde sabitlenmiş bölge merkezleridir (kaynak: complaints/results/drift_report.json:2).

Şekil 6'nın (a) panelinde çizilen nicelik, eğitim ve sürüklenme pencerelerinin bölge nokta payları farkıdır (kaynak: complaints/results/complaints_scale_stability.json:64). Ölçüm 20.000 kayıtlık bir alt-örnekte 14.000 eğitim ve 6.000 sürüklenme kaydıyla yapılmıştır (kaynak: complaints/results/complaints_scale_stability.json:2-4). Bölge bazında pay farkı ölçülen uçlar: bölge 1 için -0.17 ve bölge 7 için +0.2068 (kaynak: complaints/results/complaints_scale_stability.json:66, complaints/results/complaints_scale_stability.json:72). Bölge atamasının alt-örneklemeye kararlılığı da ölçülmüştür: eşleşme düzeltilmiş Rand indeksi ortalaması 0.99389 ve bölge payları sıra korelasyonu ortalaması 1.0 (kaynak: complaints/results/complaints_scale_stability.json:30-32).

Şekil 6'nın (b) panelinde, ayar kaymasının alarm oranına etkisi ölçülmüştür
Değişmemiş durumda ayrıktaki sağlıklı kayıtların alarm oranı 0.011723, üç ölçülen alanda sabit bir kayma vektörü uygulandıktan sonra 0.028136 olarak ölçülmüş ve oran 2.4x olarak kaydedilmiştir (kaynak: maintenance/results/e1_cmapss.json:190-192)
Kayma vektörü ve eşik tanımı da kayıtlıdır: uygulanan kayma [6.0, 6.0, -6.0] değerlerindedir (kaynak: maintenance/src/e1_real.py:293), eşik, eğitim dağılımının 0.95 çeyreğinden hesaplanan mesafe değeridir: 8.4605 (kaynak: maintenance/results/e1_cmapss.json:186-187) ve ayrıktaki sağlıklı kayıt sayısı 853'tür (kaynak: maintenance/results/e1_cmapss.json:189).

### 5.11 INT8 kantizasyon ve kendi kendine yeten paket (Şekil 7)

Şekil 7'nin (b) panelinde ölçümü gösterilen dağıtımda okuyucunun parametrik katmanları INT8 tam sayı biçimine kantize edilir ve dağıtım tek bir kendi kendine yeten paket içinde üç şeyi taşır: kantize edilmiş model ağırlıkları, girdi değerlerini model girdi kimliklerine eşleyen nicelleştirme tablosu ve model girdi kimliklerini insan-okunur girdi etiketlerine eşleyen çözümleme tablosu. Kayıtlı paket bileşenleri: kantize ONNX modeli 543.5 KB, nicelleştirme tablosu 5.4 KB, çözümleme tablosu 16.5 KB ve 221 çözümleme girdisi (kaynak: fraud/results/deploy/manifest.json:18, fraud/results/deploy/manifest.json:21-24). Eşleme tanımları kayıtlıdır: ham satırlar kimliklere nicelleştirme tablosuyla, kimlikler insan-okunur etiketlere çözümleme tablosuyla dönüşür (kaynak: fraud/results/deploy/manifest.json:97-98).

Böylece uç aygıtta görev çıktısı ile insan-okunur açıklama aynı paketten çıkar: nokta çıkarma etkisi INT8 oturumu üzerinde çalıştırılır (kaynak: fraud/pilot/infer.py:104) ve maskelenen konunun etiketi çözümleme tablosundan gelir. Paketin kendi öz-testinde 19963 satır üzerinden FP32 referansla en büyük mutlak fark 1.4603e-06, p99 mutlak fark 8.8483e-07 ve verim 54063 işlem/saniye ölçülmüştür (kaynak: fraud/results/deploy/manifest.json:27-30).

Ayak izi bütçesi, kullanılmayan parametrelerin sayılmasıyla da izlenmiştir: tabular dağıtımda dil modeline ait olup pakette kullanılmayan parametre sayısı 8392 olarak kayıtlıdır (kaynak: fraud/results/edge_eval_full_scale.json:11)
Bu kayıt, aynı mimarinin hangi kısmının hangi görev için taşındısının denetlenebilir biçimde belirlendiğini gösterir; çoklu görev başarımına ilişkin bir ölçüm içermez.

### 5.12 Eğitim

Eğitim iki kısımdan oluşur
Okuyucunun ağırlıkları gradyan tabanlı geri yayılım ile güncellenir
Nokta bulutunun konumları ise gradyan hesaplanmaksızın güncellenir: bir nokta konumu için bir aday konum üretilir, aynı mini-batch üzerinde adayla görev yitimi ölçülür ve adayın yitimi taban yitimin bir eşik altında olması hâlinde aday kabul edilir (kaynak: quann_cloud/src/qcloud_quann.py:212); adayın yitimi tabanın eşik üstüne çıkması hâlinde konum önceki değerine geri alınır (kaynak: quann_cloud/src/qcloud_quann.py:220) ve arama aralığı aday tarafında daraltılır (kaynak: quann_cloud/src/qcloud_quann.py:223, quann_cloud/src/qcloud_quann.py:226)
Bu sıfırıncı mertebeden kabul ve geri alma, nokta konumlarının güncellenmesi için geri yayılım gerektirmez.

## 6. Ölçümlü örnek: tabular sınıflandırma görevi

Tabular embodiment, kredi kartı işlem kaydının 8 boyutlu alan değerleri üzerinden dolandırılma skorunun üretildiği görevdir (D = 8, bulut durum boyutu 128, K = 13; kaynak: fraud/results/full_scale_hybrid.json:56-57, fraud/results/full_scale_hybrid.json:61). Eğitim 100.000 kayıtlık, 130 dolandırılma kaydı içeren, taban oranı 0.0013 olan bir alt-küpe üzerinde 2852.3 saniyede tamamlanmış ve bu yapılandırmada AUC 0.9351452 ile hatırlama 0.73077 ölçülmüştür (kaynak: fraud/results/full_scale_hybrid.json:5-7, fraud/results/full_scale_hybrid.json:10, fraud/results/full_scale_hybrid.json:16, fraud/results/full_scale_hybrid.json:23).

Ölçümün sınırı açıkça kaydedilir
Aynı veri ailesinin tam ölçekli karşılaştırmasında test kümesi 4.716.078 satırdır (kaynak: fraud/results/duels.json:113) ve iki karşılaştırma modeli daha yüksek AUC üretmiştir: gradyan artışı temelli ağaç topluluğu 0.9933922 (kaynak: fraud/results/duels.json:4; model tanımı: fraud/src/train_duels.py:208) ve dönüştürücü tabanlı FT-Transformer 0.9694542 (kaynak: fraud/results/duels.json:50; model tanımı: fraud/src/train_duels.py:89, fraud/src/train_duels.py:119)
Diğer karşılaştırma modeli, gradyan artışı temelli bir diğer ağaç topluluğu, 0.9364403 üretmiştir (kaynak: fraud/results/duels.json:26; model tanımı: fraud/src/train_duels.py:233)
Buluş, doğruluk üstünlüğü iddia etmez
Bu örnekte iddia, yüksek doğruluklu bir ağaç modelinin yanında uç aygıta indirgenebilen, kendi açıklamasını ve bölge denetimini üreten bir temsilin aynı görev çıktısını üretmesidir.

## 7. Ölçümlü örnek: INT8 uç dağıtım (Şekil 7)

Şekil 7'nin (a) panelinde ölçülen AUC değerleri, taban oranı korunmuş doğal oranlı ayrık test kümesinde alınmıştır: 4.700.000 satır, 6110 dolandırılma kaydı, taban oran 0.0013 (kaynak: fraud/results/edge_eval_full_scale.json:59-61). FP32 AUC 0.9450746, INT8-tensor AUC 0.9364812 ve INT8-kanal AUC 0.9367114 ölçülmüş; kantizasyonun AUC maliyeti tensor için -0.008593 ve kanal için -0.008363 olarak kaydedilmiştir (kaynak: fraud/results/edge_eval_full_scale.json:62-66). Aynı ölçüm çalışma günlüğünde yuvarlanmış biçimiyle FP32 0.9451, INT8 tensor 0.9365, INT8 channel 0.9367 olarak yazılıdır (kaynak: fraud/logs/run_full_scale_edge_eval.log:52-54).

Kantize modelin çıktı sürüklenmesi de ölçülmüştür: 4096 girdi üzerinde (ölçümün girdi sayısı: fraud/logs/run_full_scale_edge_eval.log:46) olasılık farkı en büyük 0.3433, ortalama 0.00892 ve p99 0.1457 olarak kayıtlıdır (kaynak: fraud/results/edge_eval_full_scale.json:54-56). Bu değer, kantizasyonun tek tek kararlarda taşınabilirlik için kabul edilmiş bir çıktı farkı yarattığının ölçüsüdür.

Şekil 7'nin (b) panelinde ölçülen ayak izi, uç aygıta indirilen paketin boyutudur: FP32 528.9 KB'dan INT8 138.2 KB'ye inmiştir (kaynak: fraud/results/edge_eval_full_scale.json:15-16); kantize ONNX dosyası 543.5 KB'tır ve FP32 referansla en büyük mutlak fark 1.0058e-07'dir (kaynak: fraud/results/edge_eval_full_scale.json:48-49). Ölçülen işlem hızı, INT8 okuyucuda işlem başına 0.015913 ms (p50) ve 62845 işlem/saniye (kaynak: fraud/results/edge_eval_full_scale.json:23-25), ONNX çalıştırma katmanında işlem başına 0.025782 ms ve 39056 işlem/saniyedir (kaynak: fraud/results/edge_eval_full_scale.json:50-51).

## 8. Ölçümlü örnek: dil görevi ve çok cümleli bağlam

Boyut seçiminin ölçüldüğü karşılaştırmada, aynı eğitim düzeneğinin D = 32 ve D = 64 yapılandırmaları ayrı bir held-out kümesinde perplexity bakımından denenmiştir: D = 32 için held-out PPL 61.94, D = 64 için 61.19 ölçülmüş, fark 0.76 ve kazanan D = 64 olarak kaydedilmiştir (kaynak: logs/run_r4_resume.log:67, logs/run_r4_resume.log:70-71). Ölçüm protokolü aynı satırda kayıtlıdır: n = 2000, seed = 0, satır aralığı [200000, 202000), dizi uzunluğu 10 (kaynak: logs/run_r4_resume.log:71). Bu değerler tek bir boyut seçiminin ölçümüdür; başka bir protokolde ölçülen perplexity değerleriyle karşılaştırılmaz.

Sınır, açıkça kaydedilir
Ortak bir held-out kümesinde (2000 cümle) ölçülen perplexity CloudLM için 6.45, nedensel (causal) dil modeli temel çizgisi için 2.94 bulunmuş ve temel çizgi daha iyi ölçülmüştür (kaynak: logs/run_compare.log:16-18); karşılaştırılan model boyutları 20.999.648 ve 22.724.768 parametredir (kaynak: logs/run_compare.log:13-14)
Buluş, dil görevinde doğruluk üstünlüğü iddia etmez; dil görevindeki iddia, görev çıktısının ve üç açıklama işleminin aynı temsil üzerinden üretilabilir olmasıdır.

Çok cümleli hiyerarşinin (Şekil 8) ölçüldüğü eğitimde veri 80/10/10 oranında ve seed = 42 ile bölünmüştür (kaynak: contextcloud/task_utils.py:17-19) ve bölme 159888 eğitim / 19986 doğrulama / 19986 test cümle olarak kaydedilmiştir (kaynak: logs/run_r5_cc_stable.log:2). Sözcük dağarcığı 10000, model 12.260.784 parametre olmuştur (kaynak: logs/run_r5_cc_stable.log:6-7). Eğitimde en düşük doğrulama yitimi 3.2305 ve ayrık held-out TEST yitimi 3.2318 ölçülmüştür (kaynak: logs/run_r5_cc_stable.log:1235-1236). Bu sayılar çapraz-entropi yitimidir; perplexity değildir ve 8. paragraftaki PPL değerleriyle aynı metrik olarak sunulmaz.

## 9. Ölçümlü örnek: açıklama çıktılarının üretimi

**Nokta çıkarma etkisi (birinci işlem).** Katkıların üretildiği pilot, 827 kayıtlık ve 27 dolandırılma kaydı içeren tabakalı bir geliştirme örneğidir; taban oran 0.03265 olarak kaydedilmiş ve örneğin doğal "4.7M" ayrık popülasyon olmadığı değil tabakalı bir geliştirme örneği olduğu açıkça uyarılmıştır (kaynak: fraud/results/explain_pilot.json:6-10)
Bu örnekte AUC-ROC 0.82861 ölçülmüştür (kaynak: fraud/results/explain_pilot.json:32)
Pilotun kanıtladığı iddia üretim doğruluğu değil, katkı verisinin ayrı bir açıklama modeli olmadan üretilebilirliğidir: katkı tanımı logit farkıdır (kaynak: fraud/results/explain_pilot.json:21), işaretin anlamı kayıtlıdır (kaynak: fraud/results/explain_pilot.json:22) ve toplamsal olmadığının uyarısı düşülmüştür (kaynak: fraud/results/explain_pilot.json:23)
Girdi birimlerinin sıralanma kuralı da tanımlıdır: ortalama mutlak logit katkısı ile gözlem sayısının karekökünün çarpımı (kaynak: fraud/results/explain_pilot.json:24).

**Bölge/prototip etiketleme (ikinci işlem).** Grafik işlem ailesinde, eğitimde oturtulmuş sınıf prototiplerine en-yakın-prototip atamasının ayrık test doğruluğu 0.75 olarak ölçülmüştür; test kümesi 400 düğüm, 40 usulsüz düğüm içerir ve sınıf bazında doğruluk 0.7833 ve 0.45'tir (kaynak: aml/results/e2_explain.json:14-19). Prototipler yalnızca eğitim düğümlerinin ortalaması olarak tanımlanır (kaynak: aml/src/e1_explain.py:297). Ölçümün sınırı dosyada kayıtlıdır: tek atımlık grafik özellikleri transdüktiftir ve kaynak veri hesap kimliği değil işlem düğümü taşır (kaynak: aml/results/e2_explain.json:11).

**Bölge saflığının durum bazlı sınırı.** Makine sağlığı görevinde ayrık test 1064 kayıt üzerinde en-yakın-prototip doğruluğu 0.81673 ölçülmüştür (kaynak: maintenance/results/e1_cmapss.json:23-24). Doğruluk, bozulma kademesi ile düşmüştür: sağlıklı kademe 853 kayıtta 0.93787, kritik kademe 25 kayıtta 0.64, ara kademe 114 kayıtta 0.18421 (kaynak: maintenance/results/e1_cmapss.json:27-28, maintenance/results/e1_cmapss.json:43-44, maintenance/results/e1_cmapss.json:51-52). Bu, bölge etiketinin bozulmuş durumlarda tek başına denetim kanıtı olarak kullanılamayacağının ölçülü ifadesidir.

## 10. Ölçümlü örnek: açıklamanın kararlılığının denetimi

Açıklama verisinin denetimi, temsilin kendi bölgesinden türediği için tek sayıya indirgenebilir. Beş ayrı eğitim tohumunda (kaynak: aml/results/e5_stability.json:8-12) ve sabit 400 kayıtlık test alt-örneğinde (kaynak: aml/results/e5_stability.json:14) ölçülen bölge etiketi kararlılığı 0.325'tir; tanım, atıf tohumundaki etiketle tüm yeniden eğitimlerde eşleşen test kayıtlarının oranıdır (kaynak: aml/results/e5_stability.json:29-30).

Aynı protokollerde model-sonrası atıf yolunun kararlılığı da ölçülmüştür ve üç ayrı nicelik olarak kayıtlıdır: atıf sıralaması birinci-sıra kararlılığı 0.1025, ilk-üç kümesi jaccard benzerliği 0.43405, atıf sıralaması Spearman korelasyonu 0.65105 (kaynak: aml/results/e5_stability.json:41-43); atıf aracı TreeExplainer sürüm 0.54.1'dir (kaynak: aml/results/e5_stability.json:25)
Bu üç nicelik ile 0.325 değeri (kaynak: aml/results/e5_stability.json:29) aynı nicelikler olmadığından aralarında oran veya üstünlük kurulmaz; buluşun kullandığı sonuç, açıklama kararlılığının eğitimden bağımsız ikinci bir araç gerektirmeden tek sayıyla izlenebilir hâle gelmesidir.

Aynı dosyada ölçülen tohum başına süreler, açıklamanın ayrı bir iş adımı olarak maliyetini karşılaştırmalı olarak gösterir: geometrik temsilin kendi açıklama verisini üreten yolunun oturtma ve kodlama süresi 0.3545 saniye (kaynak: aml/results/e5_stability.json:38), model-sonrası ağaç atıf yolunun yeniden eğitim ve atıf süresi 0.7403 saniyedir (kaynak: aml/results/e5_stability.json:34057). Bu iki süre farklı iş kümeleridir; aralarındaki fark "açıklama daha hızlıdır" biçiminde genellenmez, yalnızca model-sonrası atıf yolunun ayrı bir yeniden eğitim adımı taşıdığına işaret eder.

## 11. Varyantlar


#### Varyant 1 — küresel-istatistik (çiftsiz) temsil

Bu varyantta Çift-geometri hesabı yerine nokta bulutunun küresel küme istatistikleri kullanılır; dönüşüm kodlayıcısı ile merkezi kodlayıcı arasındaki tüm çift hesabı kaldırılmış veya yaklaşık hâle getirilmiştir (kaynak: purecloudlm/model.py:8-9) ve merkezi kodlayıcı O(N) havuzlama ile bulut merkezi/variance hesaplar (kaynak: purecloudlm/model.py:57, purecloudlm/model.py:62-63). Uç aygıtta nokta sayısının büyümesi gereken durumda çift yolunun yerine bu yol seçilir.


#### Varyant 2 — çok cümleli bağlam hiyerarşisi

Bu varyantta Bölüm 5.6'da tanımlanan cümle-bulut ve pencere-bulut yapısı kullanılır (kaynak: contextcloud/model.py:20, contextcloud/model.py:121, contextcloud/model.py:196, contextcloud/model.py:246).


#### Varyant 3 — sıfırıncı mertebeden nokta güncellemesi ve gradyan tabanlı okuyucu

Bu varyantta Nokta konumları Bölüm 5.12'deki kabul ve geri alma ile güncellenirken okuyucu geri yayılım ile eğitilir (kaynak: quann_cloud/src/qcloud_quann.py:212, quann_cloud/src/qcloud_quann.py:220)
Bu varyantın maliyet sınırı da ölçülmüştür: sıfırıncı mertebe yaklaşımı aday başına bir ileri geçiş gerektirdiğince büyük modelde uygulanabilir bulunmamış ve ölçüm aynı mimarinin küçük ölçeğinde yapılmıştır (kaynak: quann_cloud/FINDINGS.md:22)
Küçük ölçekli ölçümde held-out perplexity değerleri eğitilmemiş temel çizgi için 1045.03, geri yayılımla eğitim için 563.45 ve sıfırıncı mertebe eğitim için 1016.30 olmuştur (kaynak: ölçülen tablo başlığı quann_cloud/FINDINGS.md:27, değerler quann_cloud/FINDINGS.md:28, quann_cloud/FINDINGS.md:29, quann_cloud/FINDINGS.md:30)
Sıfırıncı mertebe varyantı geri yayılımla eğitime göre daha yüksek perplexity üretmiştir; bu varyant doğruluk iddiasıyla değil, geri yayılım gerektirmeyen bir nokta güncellemesi sağlamasıyla istenir
Ölçüm 6000 eğitim ve 500 değerlendirme cümlesi, dizi uzunluğu 8 ve 1024 parçalı sözcük dağarcığı ile yapılmıştır (kaynak: quann_cloud/FINDINGS.md:18-19).

## 12. Ölçülmemiş iddialar ve doğrulanamayan kayıtlar

Buluş metnine alınmayan veya metinde bilinçli olarak yumuşatılan noktalar, gerekçeleriyle kaydedilir:

1. **Çift-geometri genişliğinin adı ve çizim başlığında çelişki.** Kodda genişlik 3D + 2 olarak tanımlıdır (kaynak: cloudlm/model.py:63-64). Bu toplamı "D + 4" diye adlandıran etiket kodda veya ölçüm dosyalarında doğrulanamamış, yalnızca Şekil 2 çiziminin başlık metninde bulunmuştur (kaynak: docs/patent/final_v3/generate_figures_v3.py:87). Bu nedenle metin bileşen adlarını kullanmış, sayısal etiket kullanılmamıştır. Çelişki GİDERİLMİŞTİR (2026-11-18): etiket çizim başlığından çıkarılmış, çizim kod tanımıyla (R^{3D+2}) yeniden üretilmiştir.
2. **Sabit nokta sayısı.** Eski taslaklarda geçen sabit nokta sayısı iddiasına dair kod veya yapılandırma izi bulunamamıştır; kodda geçen 20 değeri üretilen token sayısıdır (kaynak: cloudlm/model.py:316) ve bu durum vekil soru-cevap kaydında "doğrulanmadı" olarak düşülmüştür (kaynak: docs/patent/VEKIL_SORU_CEVAP.md:91)
Nokta sayısı bu nedenle bağlam uzunluğuna bağlı tanımlanmıştır.
3. **Tek modelle çoklu görev başarımı.** Aynı ağırlıkların hem dil hem tabular görevde eşzamanlı başarımı ölçülmemiştir. Metinde bu konuda yalnız yapısal gözlem vardır: tabular dağıtımda kullanılmayan dil parametresi sayısı 8392 olarak sayılmıştır (kaynak: fraud/results/edge_eval_full_scale.json:11).
4. **Doğruluk üstünlüğü.** Tabular görevde iki karşılaştırma modeli daha yüksek AUC üretmiştir (kaynak: fraud/results/duels.json:4, fraud/results/duels.json:50); dil görevinde nedensel temel çizgi daha düşük perplexity üretmiştir (kaynak: logs/run_compare.log:17-18)
Metin hiçbir alanında doğruluk üstünlüğü iddia etmez.
5. **Bölge kaymasında uç değerler.** Haftalık tam değişim ölçütünün en yüksek değeri tek kayıtlık bir haftaya aittir (kaynak: complaints/results/drift_report.json:1157-1158, complaints/results/drift_report.json:1169-1170) ve tek gözlem olduğu için buluş metnine alınmamıştır; metinde iki ardışık haftanın ölçümleri kullanılmıştır.
6. **Yeni bölge işareti.** Yeni ortaya çıkan bölge işareti mekanizma olarak tanımlanmıştır, ancak ölçülen şikâyet sürüklenmesinde yeni bölge olayı kaydedilmemiştir (kaynak: complaints/results/drift_report.json:1174)
Bu nedenle işaret, iddia olarak değil ölçüt olarak istemlenmiştir.
7. **Açıklama kararlılığı kıyası.** Bölge etiketi kararlılığı ile model-sonrası atıf kararlılığı farklı niceliklerdir; metinde oran veya üstünlük kurulmamıştır.
8. **Protokolü satırla bağlı olmayan perplexity değerleri.** Protokolü (örneklem, satır aralığı, dizi uzunluğu) aynı satırda kayıtlı olmayan perplexity değerleri metne alınmamıştır.
9. **Şekillerde ölçüm.** Yalnız Şekil 6 ve Şekil 7 ölçüm taşır; diğer şekiller şematiktir ve metindeki sayılar bu iki şeklin kaynak dosyalarından alınmıştır. Bu iki şeklin metindeki sayıları satır düzeyinde de bağlıdır: bölge payı farkı complaints/results/complaints_scale_stability.json:64-66, complaints/results/complaints_scale_stability.json:72; alarm oranları ve oran maintenance/results/e1_cmapss.json:190-192; AUC değerleri fraud/results/edge_eval_full_scale.json:62-64 ile çalışma günlüğünde yuvarlanmış biçimiyle fraud/logs/run_full_scale_edge_eval.log:52-54; paket ayak izi fraud/results/edge_eval_full_scale.json:15-16.

## 13. Endüstriyel uygulanabilirlik

Buluş, uç aygıtta çalışan ve kararı denetlenebilir olması gereken sistemlerde doğrudan uygulanabilir. Finansal işlem sınıflandırmasında INT8 paket 138.2 KB ayak izi ile dağıtılır (kaynak: fraud/results/edge_eval_full_scale.json:16), işlem başına 0.015913 ms gecikme ve 62845 işlem/saniye üretilir (kaynak: fraud/results/edge_eval_full_scale.json:23-25) ve reddedilen işlemin hangi girdi konularına dayandığı çözümleme tablosundan insan-okunur etiketle çıkar (kaynak: fraud/results/deploy/manifest.json:98). Makine sağlığı izlemesinde bölge kayması, ayar değişikliğinin alarm oranını 0.011723'ten 0.028136'ya taşıdığını etiket gerektirmeden gösterir (kaynak: maintenance/results/e1_cmapss.json:190-191). Metin işlemede çok cümleli hiyerarşi 80/10/10 ayrık bölme ile eğitilir (kaynak: contextcloud/task_utils.py:17-19) ve bölge etiketleri şikâyet konularının insan-okunur sıralı sembollerinden üretilir (kaynak: complaints/src/explain_region.py:66).

Üretim için gereken bileşenler standarttır: bir işlemci ve bellek, ağırlıkları INT8 biçiminde saklanan bir okuyucu, iki eşleme tablosu ve ileri geçişi maskeli girdilerle yeniden çalıştıran bir yorumlama adımı. Ayrı bir açıklama eğitimi, ayrı bir açıklama modeli veya ayrı bir açıklama girdisi gerekmediğinden, açıklama yeteneği modelle birlikte sürümlenir ve denetlenir.
