# TARİFNAME — Vekil Taslağı v1 (çalışma belgesi)

**Buluşun başlığı:** Nokta bulutu temelli çift kullanımlı temsil üzerinden görev çıktısı ve gömülü açıklanabilirlik üreten yöntem, sistem ve kayıt ortamı

Bu belge `ISTEMLER.md:86` satırında künyesi verilen istem setinin dayanağıdır; her sayının yanında ham kaynak
`dosya:satır` verilir. Sayı, kaynak satırındaki ham değerin kopyasıdır; türetilmiş değer
parantez içinde ham değerin yanında yazılır. `docs/patent/final_v3/` değiştirilmemiştir.

## 1. Buluşun teknik alanı

Buluş, makine öğrenmesi modellerinde görev çıktısı üretimi ile üretilen kararın açıklanmasına ilişkindir. Özellikle, bir bağlam kodlayıcısının ürettiği D-boyutlu nokta bulutu temsilinin hem görev çıktısının hem de o kararın açıklamasının tek kaynağı olarak kullanıldığı bilgisayarla uygulanan yöntemler, bu yöntemleri yürüten sistemler ve üzerinde yürütülebilir komutlar taşıyan bilgisayarca okunabilir kayıt ortamları ile ilgilidir.

Uygulama alanları, açıklanan kararın denetlenmesinin ölçülebilir bir maliyet veya yükümlülük konusu olduğu uç (kenar) bilişim, finansal işlem sınıflandırması, makine sağlığı izleme ve metin işleme alanlarıdır. Buluş, bir model sınıfının doğruluğunun başka bir sınıftan yüksek olduğu iddiasını taşımaz.

## 2. Önceki teknik ve açığı

Yerleşik yaklaşım açıklamanın model-sonrası (post-hoc) bir araçla üretilmesidir: eğitimli bir sınıflandırıcının dışında, girdi özellikleri üzerinde çalışan ayrı bir atıf aracı çalıştırılır; görev çıktısı ile açıklama farklı sistemlerden gelir. Üç teknik açık, işleyen prototiplerde ölçülebilir biçimde görülür.

**(a) Model-sonrası açıklamanın kendi kararlılığı yeniden eğitimle sınırlıdır.** Beş ayrı eğitim tohumu (`aml/results/e5_stability.json:8-12`) ve sabit 400 kayıtlık test alt-örneğinde (`aml/results/e5_stability.json:14`) model-sonrası ağaç atıf aracının (TreeExplainer, kurulu sürüm 0,54.1; `aml/results/e5_stability.json:25`) atıf sıralaması birinci-sıra kararlılığı 0,1025, ilk-üç kümesi Jaccard benzerliği 0,43405 ve atıf sıralaması Spearman korelasyonu 0,65105 ölçülmüştür (`aml/results/e5_stability.json:41-43`). Aynı veri ve aynı tohumlar üzerinde, temsilin kendi geometrisinden türeyen en-yakın-bölge etiketinin tohumlar arası kararlılığı 0,325'tir (`aml/results/e5_stability.json:29`; tanım `aml/results/e5_stability.json:30`). Bu iki ölçüm aynı nicelik değildir; aralarında oran veya üstünlük kurulmaz. Açık şudur: model-sonrası atıf, modelin kendisi kadar kararlı olmayan ayrı bir nicelik üretir ve bu niceliğin denetimi ayrı bir iş gerektirir.

**(b) Açıklama, uç aygıtta ayrı bir yığın ve ayrı bir iş adımı gerektirir.** Ölçülen örnek: aynı protokolde model-sonrası ağaç atıf yolunun tohum başına ortalaması 0,7402620668290183 s (`aml/results/e5_stability.json:34057`), temsilin kendi açıklama verisini üreten yolunun tohum başına ortalaması 0,35445423298515377 s'dir (`aml/results/e5_stability.json:38`). Bu iki süre farklı iş kümeleridir; "açıklama daha hızlıdır" biçiminde genellenmez. Ölçülen olgu, açıklamanın ayrı bir eğitim/artımlama adımı olarak süre ve yığın eklemesidir.

**(c) Sürüklenebilirlik (drift) denetimi de ayrı bir araç ister.** Bölge paylarının zaman içinde kaymasının denetimi temsilin kendi yapısından değil, dışarıdan tanımlanmış özellik pencerelerinden türetilir; kaymanın ölçülmesi için ikinci bir izleme hattı kurulması gerekir.

Ortak neden, açıklama verisinin temsilin kendisinde bulunmayıp temsilin dışına ikinci bir sistem olarak eklenmesidir.

## 3. Buluşun kısa açıklaması ve teknik etkisi

Görev girdisi (sıralı token dizisi veya bir tabular kaydın birden çok ölçülen alan değeri) bir bağlam kodlayıcı ile D-boyutlu gösterimlere dönüştürülür ve N noktadan oluşan bir nokta bulutu kurulur. Noktalar için nokta özellikleri, nokta çiftleri için iki noktanın konumlarına bağlı çift-geometri özellikleri hesaplanır; her iki özellik kümesi girdi sırasından bağımsız simetrik bir küme agregasyonu ile tek bir küme gösteriminde birleştirilir ve okuyucu bu gösterimden görev çıktısını üretir. Ayırt edici özellik, aynı eğitimli modelin aynı nokta bulutu üzerinde üç açıklama işleminden en az birini uygulamasıdır: nokta çıkarma etkisi, bölge isimlendirme, bölge kayması. Görev çıktısı ile açıklama verisi, aynı küme gösteriminden türeyen birlikte çıktılardır.

Teknik etki, ölçülebilir eksenler üzerinden ve işleyen prototipin ölçümleriyle ifade edilir (doğruluk üstünlüğü ekseni yoktur):

- **Dağıtılan bellek ayak izi (İst. 11; künye `ISTEMLER.md:55`).** Uç aygıta indirilen paket FP32 528,9 KB'dan INT8 138,2 KB'ye inmiştir (`fraud/results/edge_eval_full_scale.json:15-16`). Paket görev çıktısının yanında insan-okunur açıklama etiketini de üretir: nicelleştirme tablosu 5,4 KB, çözümleme tablosu 16,5 KB ve 221 çözümleme girdisi tek pakettedir (`fraud/results/deploy/manifest.json:21-24`).
- **Gecikme ve geçiş hızı (İst. 11-12; künyeler `ISTEMLER.md:55` ve `ISTEMLER.md:58`).** INT8 okuyucuda işlem başına gecikme p50 0,015912843984551728 ms, p99 0,017278228622744793 ms, verim 62.845 işlem/saniye (`fraud/results/edge_eval_full_scale.json:23-25`). Kantize ONNX yolunda 0,025782 ms ve 39.056 işlem/saniye (`fraud/results/edge_eval_full_scale.json:50-51`).
- **Açıklama maliyetinin sınırlı tutulması (İst. 6; künye `ISTEMLER.md:40`).** Nokta çıkarma etkisi N + 1 ilerleyiş geçişine indirgenir ve maskeli bulutlar tek toplu girdi verilir (`fraud/pilot/infer.py:97-104`). Yük, çift sayısının artışıyla karesel biçimde büyür ve bu büyüme ölçülmüştür: etkin nokta sayısı k = 13 için işlem başına 2.522.200 çarp-topla işlemi (`fraud/results/edge_eval_full_scale.json:29-30`) ve k = 50 için 34.091.488 çarp-topla işlemi (`fraud/results/edge_eval_full_scale.json:41-42`); aynı iki yapılandırmada işlem başına gecikme sırasıyla 0,019058348584621854 ms (`fraud/results/edge_eval_full_scale.json:31`) ve 0,22592594427806034 ms ile 4439,0513682986775 işlem/saniye (`fraud/results/edge_eval_full_scale.json:43-44`) olarak kaydedilmiştir.
- **Kantizasyon doğruluk maliyetinin bilinmesi (İst. 11; künye `ISTEMLER.md:55`).** Doğal oranlı 4.700.000 satırlık ayrık testte (6.110 dolandırılma kaydı, taban oran 0,0013; `fraud/results/edge_eval_full_scale.json:59-61`) FP32 AUC 0,9450745846153957, INT8-tensor 0,9364812440872093, INT8-kanal 0,9367113790742326 ölçülmüş; kantizasyon maliyeti −0,008593340528186388 ve −0,008363205541163099 AUC olarak kaydedilmiştir (`fraud/results/edge_eval_full_scale.json:62-66`). Düşüş ölçülüp sınır olarak kabul edilmiştir.
- **Denetim kararlılığının tek sayıya indirgenmesi (İst. 8-9; künyeler `ISTEMLER.md:46` ve `ISTEMLER.md:49`).** Açıklama verisi temsilin kendi bölgesinden türediği için yeniden eğitimler arası kararlılık tek sayıyla izlenir: beş tohumda 0,325 (`aml/results/e5_stability.json:29`).
- **Etiketsiz sürüklenebilirlik işareti (İst. 10; künye `ISTEMLER.md:52`).** C-MAPSS ayar kaymasında ayrıktaki sağlıklı kayıtların alarm oranı 0,011723329425556858'ten 0,02813599062133646'ya çıkmış olarak kaydedilmiştir (`maintenance/results/e1_cmapss.json:190-192`); yöntem sürükleme etiketi kullanmaz (`complaints/results/drift_report.json:2`).
- **Eğitim yükü ekseni (İst. 15; künye `ISTEMLER.md:67`).** Nokta konumları gradyan hesaplanmaksızın kabul/geri alma ile daraltılır (`quann_cloud/src/qcloud_quann.py:212-226`); bu eksen doğruluk değil, geri yayılım gerektirmeyen nokta güncellemesidir.

## 4. Şekillerin kısa açıklaması

Çizim künyeleri ve `OZET.md` §3 çizim listesiyle birebir ortaktır. Yalnız Şek. 6 ve Şek. 7 ölçüm taşır; diğerleri şematiktir.

- **Şek. 1** — şematiktir: sistem mimarisi; görev girdisi (100) → bağlam kodlayıcı (110) → nokta projeksiyonu (112) → nokta bulutu (120) → çift-geometri kodlayıcısı (130) → simetrik agregasyon (140; nokta yolu 141, çift yolu 142, birleştirme 143, maske 144) → küme gösterimi (150) → okuyucu (160) → görev çıktısı (162); aynı buluttan yorumlama motoru (170; 171 nokta çıkarma, 172 bölge isimlendirme, 173 bölge kayması) → çıkış arayüzü (180); eğitim bloğu (250 sıfırıncı mertebe nokta güncelleyici, 260 geri yayılımlı okuyucu eğitici).
- **Şek. 2** — şematiktir: nokta çifti ve çift-geometri bileşenleri [fark vektörü (D), çift mesafesi (1), açısal benzerlik (1), birinci norm (D), ikinci norm (D)] ∈ R^(3D+2).
- **Şek. 3** — şematiktir: maksimum havuzlama (141) ile ortalama havuzlama (142) ve birleştirme (143); kesikli yol: O(N) küresel-istatistik agregatörü (240).
- **Şek. 4** — şematiktir: nokta çıkarma etkisi; maskeli türevlerin tek toplu girdi verilmesi ve e_k = s_tam − s_−k tanımı.
- **Şek. 5** — şematiktir: kümeleme ile bölge ayrımı ve bölge istatistiklerinden insan-okunur bölge etiketi üretimi.
- **Şek. 6** — **ölçümlüdür**: (a) şikâyet korpusunda bölge nokta-payı kayması, ölçülen uçlar −0,17 ve +0,2068 (`complaints/results/complaints_scale_stability.json:66` ve `complaints/results/complaints_scale_stability.json:72`); (b) C-MAPSS ayar kaymasında alarm oranı 0,011723329425556858 → 0,02813599062133646 (`maintenance/results/e1_cmapss.json:190-192`).
- **Şek. 7** — **ölçümlüdür**: (a) ayrık test AUC FP32 0,9450745846153957 / INT8-tensor 0,9364812440872093 / INT8-kanal 0,9367113790742326 (`fraud/results/edge_eval_full_scale.json:62-64`); (b) paket ayak izi 528,9 KB → 138,2 KB (`fraud/results/edge_eval_full_scale.json:15-16`).
- **Şek. 8** — şematiktir: cümle-bulut (210) → komşu pencere-bulutları (220) → pencere-geometri kodlayıcısı (230) hiyerarşisi.

## 5. Buluşun detaylı açıklaması

### 5.1 Sistem mimarisi (Şek. 1) — İst. 1, 17

Zincir Şek. 1'de şematize edildiği şekilde kuruludur. İki nokta yapısal olarak önemlidir. Birincisi, temsil bir küme olarak işlenir: girdi sıralaması agregasyona taşınmaz ve bağlamanın dolmadığı konumlara karşılık gelen noktalar ile bu noktalardan oluşan çiftler bir maske sinyali ile işlem dışı bırakılır (`cloudlm/model.py:47`). İkincisi, yorumlama motoru (170) ayrı bir model değildir (§4 çizim künyesindeki işaretler, `TARIFNAME.md:45`): görev çıktısını üreten okuyucunun (160) kendi ileri geçişini, içindeki noktalar değiştirilmiş hâlde yeniden çalıştırarak açıklama verisini üretir (`fraud/src/explain_pilot.py:103` ve `fraud/src/explain_pilot.py:110`). Açıklama bu nedenle temsilin dışına eklenen ikinci bir sistem değil, temsilin ikinci bir kullanımıdır.

### 5.2 Nokta bulutunun oluşması ve D boyutu (Şek. 1) — İst. 1(a), 2

Görev girdisi iki biçimde gelir. Sıralı token dizisinde bağlam kodlayıcının her konumda ürettiği gösterim D-boyutlu bir vektöre projekte edilir (`cloudlm/model.py:312`) ve N nokta bağlam uzunluğu kadar olur. Tabular kayıtta kaydın ölçülen alan değerleri noktaları oluşturur (`fraud/src/explain_pilot.py:102`); alan sayısı N'i ve dolayısıyla çift sayısını belirler.

Ölçülen uygulama örneklerinde D yapılandırma dosyalarında kayıtlıdır: tabular dolandırılma görevinde D = 8 ve bulut durum boyutu 128 (`fraud/results/full_scale_hybrid.json:56-57`); küçük ölçekli dil yapılandırmasında D = 8 (`quann_cloud/FINDINGS.md:17`); dil görevinde ölçülen iki yapılandırma D = 32 ve D = 64 (`logs/run_r4_resume.log:67` ve `logs/run_r4_resume.log:70`). İst. 2 (`ISTEMLER.md:28`)'de sayılan değerler bu kayıtlardır. D = 64 boyutunun D = 32'ye göre başarım hükmü bu taslağın kapsamı dışındadır ve yinelenmez (mevcut kayıt: `docs/patent/final_v3/TARIFNAME.md:167`). Önceki taslaklarda kullanılan "3 ≤ D ≤ 64" ifadesi bir seçim aralığıdır; alt uç (D = 3) depoda ölçümle doğrulanmamıştır (bkz. `OZET.md` §5).

### 5.3 Çift-geometri özelliği (Şek. 2) — İst. 3

Bir nokta çifti (p_i, p_j) için çift-geometri özelliği beş bileşenin birleştirilmesinden oluşur: iki nokta vektörü arasındaki fark vektörü, fark vektörünün Öklid normu olan çift mesafesi, iki nokta vektörü arasındaki açısal benzerlik, birinci noktanın normu ve ikinci noktanın normu (`cloudlm/model.py:157-168`). Genişlik 3D + 2 olarak tanımlıdır (`cloudlm/model.py:63-64`). "D + 4" sayısal etiketi kodda veya ölçüm dosyalarında doğrulanamadığı için bu metinde kullanılmaz (`docs/patent/final_v3/TARIFNAME.md:220`).

### 5.4 Simetrik küme agregasyonu ve maskeleme (Şek. 3) — İst. 1(c), 4, 5

Nokta yolunda nokta özellikleri noktalar üzerinden maksimum havuzlama ile indirgenir (`cloudlm/model.py:153`). Çift yolunda çift-geometri özellikleri nokta çiftleri üzerinden ortalama havuzlama ile indirgenir (`cloudlm/model.py:177`). İki yolun çıktısı birleştirme katmanında tek küme gösteriminde toplanır (`cloudlm/model.py:179`).

Maskeleme her iki yolda uygulanır: nokta yolunda dolmayan konumlar havuzlama öncesi geçersiz değere çekilir ve sonuç maskeyle temizlenir (`cloudlm/model.py:151-152` ve `cloudlm/model.py:155`); çift yolunda bir çiftin geçerli sayılması için iki ucun da geçerli olması istenir, geçersiz çiftlerin özellikleri sıfırlanır ve ortalama geçerli çift sayısı üzerinden normalize edilir (`cloudlm/model.py:171-177`). Her iki havuzlama girdi sırasından bağımsız olduğundan küme gösterimi noktaların numaralanma sırasından bağımsızdır; bu, alanları sıralama taşımayan tabular kayıtlarda temsilin tanımlı kalmasını sağlar.

### 5.5 Okuyucu ve görev çıktıları (Şek. 1) — İst. 1(d), 16

Küme gösterimi okuyucuya verilir. Dil görevinde okuyucu bir sonraki tokenın olasılık dağılımını üretir ve küme durumu bulut kodlayıcısından alınır (`cloudlm/model.py:313-314`). Tabular görevde okuyucu bir sınıf atama skoru üretir ve açıklama adımı da aynı okuyucunun logitini temel alır (`fraud/src/explain_pilot.py:103`). Çıktı başlığının görev türüne göre seçilmesi İst. 16 (`ISTEMLER.md:70`)'nın konusudur; aynı ağırlıklarla iki görevin eşzamanlı başarımı ölçülmemiştir (§8.3).

### 5.6 Birinci açıklama işlemi: nokta çıkarma etkisi (Şek. 4) — İst. 1(e)(i), 6, 7

Uygulama sırası: girdi birimlerinden nokta bulutu üretilir (`fraud/src/explain_pilot.py:102`); maskelenmemiş durumda okuyucu çalıştırılıp taban çıktı alınır (`fraud/src/explain_pilot.py:103`); katkı verilecek nokta kopya bulutta sıfıra çekilir (`fraud/src/explain_pilot.py:106-107`) ve o noktanın maskesi kapatılır (`fraud/src/explain_pilot.py:109`); okuyucu maskeli bulutla yeniden çalıştırılır (`fraud/src/explain_pilot.py:110`); katkı taban çıktıdan maskeli çıktının çıkarılmasıyla kaydedilir (`fraud/src/explain_pilot.py:111`). Tanım logit farkı biçiminde kayıtlıdır (`fraud/results/explain_pilot.json:21`).

Aynı işlem kantize dağıtımda çalışır: maskeli türevler tek toplu girdi olarak hazırlanır (`fraud/pilot/infer.py:97-98`), maske her türev için ilgili konum kapatılarak kurulur (`fraud/pilot/infer.py:100`), INT8 oturumu tek çağrıda çalışır (`fraud/pilot/infer.py:104`) ve katkı logit farkı olarak üretilir (`fraud/pilot/infer.py:107`). Böylece N nokta için N + 1 ilerleyiş geçişi tek toplu çağrıya indirgenir.

Yorum sınırı prototipte kayıtlıdır: üretilen nicelik bir çıkarma duyarlılığıdır, tek-additif atıf değildir; çift-geometri hesabı nedeniyle katkılar toplamsal olmaz (`fraud/results/explain_pilot.json:23`). Girdi birimlerinin sıralanma kuralı da kayıtlıdır: ortalaması alınmış mutlak logit katkısı ile gözlem sayısının karekökünün çarpımı (`fraud/results/explain_pilot.json:24`).

### 5.7 İkinci açıklama işlemi: bölge isimlendirme (Şek. 5) — İst. 1(e)(ii), 8, 9

Çok sayıda kaydın küme gösterimleri küme merkezli bir kümeleme algoritması ile geometrik bölgelere ayrılır; prototipte KMeans kullanılmıştır (`complaints/src/explain_region.py:47`) ve ölçülen bölge sayısı 8'dir (`complaints/results/region_map.json:2`). Her bölgeye insan-okunur etiket, o bölgeye atanan kayıtların girdi sembollerinin sıklık sıralamasından seçilen sembollerin birleştirilmesiyle verilir; prototipte sıralı ilk sekiz sembol kullanılır (`complaints/src/explain_region.py:66`). Ölçülen etiket örneği `complaints/results/region_map.json:422` satırındadır.

Sınıflandırma embodimentında bölgeler, sınıfların eğitim kümesindeki gösterim ortalamalarından oluşan prototipler olarak tanımlanır (`aml/src/e1_explain.py:296-299`). Denetlenebilir sınır: geometrik olarak ayrı iki bölge aynı insan-okunur etiketi taşıyabilir (`complaints/results/region_map.json:422` ve `complaints/results/region_map.json:838`); bu nedenle denetim bölge indeksi ile birlikte yapılır.

### 5.8 Üçüncü açıklama işlemi: bölge kayması (Şek. 6) — İst. 1(e)(iii), 10

Referans ve karşılaştırma veri gruplarının aynı geometrik bölgelerdeki nokta payları karşılaştırılır. Kullanılan ölçütler: bölgeler arası pay farklarının mutlak değerleri toplamının yarısı olarak tanımlanan tam değişim ölçütü (`complaints/src/audit_drift.py:36`) ve Kullback-Leibler yakınsaklığı (`complaints/src/audit_drift.py:40`). Ayrıca referans dağılımda payı bir alt eşik altında, karşılaştırma dağılımında payı bir üst eşiği aşan bölgeler yeni ortaya çıkan bölge olarak işaretlenir (`complaints/src/audit_drift.py:44-45`).

Ölçülen ardışık haftalar: 2025-02-03/2025-02-09 için tam değişim 0,090370 ve KL 0,0341397185589417 (`complaints/results/drift_report.json:33-34`), hafta anahtarı `complaints/results/drift_report.json:21`; 2025-02-10/2025-02-16 için tam değişim 0,024691 ve KL 0,0018965315544816688 (`complaints/results/drift_report.json:49-50`), hafta anahtarı `complaints/results/drift_report.json:37`. Ölçüm 20.000 kayıtlık alt-örnekte 14.000 eğitim ve 6.000 sürüklenme kaydıyla yapılmıştır (`complaints/results/complaints_scale_stability.json:2-4`); bölge bazında pay farkı uçları −0,17 ve +0,2068'dir (`complaints/results/complaints_scale_stability.json:66` ve `complaints/results/complaints_scale_stability.json:72`). Bölge atamasının alt-örneklemeye kararlılığı eşleşme düzeltilmiş Rand indeksi ortalaması 0,99389 ve bölge payları sıra korelasyonu ortalaması 1,0 olarak ölçülmüştür (`complaints/results/complaints_scale_stability.json:30-32`).

Kontrollü ortam kayması: değişmemiş durumda ayrıktaki 853 sağlıklı kaydın alarm oranı 0,011723329425556858, üç ölçülen alanda sabit kayma vektörü uygulandıktan sonra 0,02813599062133646 ölçülmüş ve olarak kaydedilmiştir (`maintenance/results/e1_cmapss.json:189-192`). Kayma vektörü [6,0; 6,0; −6,0] (`maintenance/src/e1_real.py:293`), alarm eşiği eğitim dağılımının 0,95 çeyreğinden hesaplanan 8,460468292236328 mesafe değeridir (`maintenance/results/e1_cmapss.json:186-187`).

### 5.9 INT8 kantizasyon ve kendi kendine yeten paket (Şek. 7) — İst. 11, 12, 18

Dağıtım tek pakette üç şeyi taşır: kantize model ağırlıkları, girdi değerlerini model girdi kimliklerine eşleyen nicelleştirme tablosu ve model girdi kimliklerini insan-okunur etiketlere eşleyen çözümleme tablosu. Kayıtlı bileşenler: kantize ONNX modeli 548,4 KB (`fraud/results/deploy/manifest.json:18`), nicelleştirme tablosu 5,4 KB (`fraud/results/deploy/manifest.json:21`), çözümleme tablosu 16,5 KB (`fraud/results/deploy/manifest.json:23`), 221 çözümleme girdisi (`fraud/results/deploy/manifest.json:24`). Eşleme tanımları `fraud/results/deploy/manifest.json:97-98` satırlarındadır. Uç aygıta indirilen ayak izi 528,9 KB FP32'den 138,2 KB INT8'e inmiştir (`fraud/results/edge_eval_full_scale.json:15-16`).

Paketin kendi öz-testi 19.963 satır üzerinden yürütülmüş; FP32 referansla en büyük mutlak fark 1,2814998626708984e-06, p99 mutlak fark 8,401274840252881e-07 ve verim 26.473 işlem/saniye ölçülmüştür (`fraud/results/deploy/manifest.json:27-30`). Kantize ONNX dosyası 543,5 KB ve Torch↔ORT en büyük mutlak farkı 1,0058e-07'dir (`fraud/results/edge_eval_full_scale.json:48-49`).

Ayak izi bütçesi kullanılmayan parametrelerin sayılmasıyla da izlenmiştir: tabular dağıtımda dil modeline ait olup kullanılmayan parametre sayısı 8.392'dir (`fraud/results/edge_eval_full_scale.json:11`). Bu kayıt yapısal bir gözlemdir; çoklu görev başarımı ölçümü içermez.

### 5.10 Küresel-istatistik varyantı (Şek. 3, kesikli yol) — İst. 13

Çift-geometri hesabının yerini nokta bulutunun küresel küme istatistikleri alır: nokta boyutları üzerinden maksimum, minimum, ortalama ve standart sapma havuzlaması (`purecloudlm/model.py:62`) ile bulut merkezi ve merkez sapması (`purecloudlm/model.py:63`); çift hesabı yapılmadığı veya çiftlerin bir alt kümesi üzerinden örneklenerek yapıldığı için maliyet nokta sayısı ile doğrusal orantılıdır (`purecloudlm/model.py:8-9`; kodlayıcı tanımı `purecloudlm/model.py:57`). Bu varyant için doğruluk veya perplexity üstünlüğü ileri sürülmez: ayrık testte geçerli bir başarım ölçülmemiştir (`docs/patent/final_v3/BLUS_NOTU.md:36`).

### 5.11 Çok cümleli bağlam hiyerarşisi (Şek. 8) — İst. 14

Birinci katmanda her cümle bir cümle-bulut durumu olarak kodlanır (`contextcloud/model.py:20`). İkinci katmanda ardışık cümle-bulut durumları sabit bir pencere boyutu ve sabit bir adımla kaydırılarak komşu pencere-bulutları türetilir; pencere boyutu 5 ve adım 2 olarak tanımlanmıştır (`contextcloud/model.py:121`, `contextcloud/model.py:127` ve `contextcloud/model.py:128`). Üçüncü katmanda pencereler arası geometrik büyüklükler bir pencere-geometri kodlayıcısı ile toplanır (`contextcloud/model.py:196`) ve çoklu-cümle küme gösterimi okuyucuya verilir (`contextcloud/model.py:246`).

Bu hiyerarşinin eğitildiği görevde veri 80/10/10 oranında ve seed = 42 ile bölünmüştür (`contextcloud/task_utils.py:16-19`) ve bölme 159.888 eğitim / 19.986 doğrulama / 19.986 test cümle olarak kaydedilmiştir (`logs/run_r5_cc_stable.log:2`). Sözcük dağarcığı 10.000 (`logs/run_r5_cc_stable.log:6`), model 12.260.784 parametre olmuştur (`logs/run_r5_cc_stable.log:7`). En düşük doğrulama yitimi 3,2305 (`logs/run_r5_cc_stable.log:1235`) ve ayrık held-out TEST yitimi 3,2318 çapraz-entropi yitimi olarak ölçülmüştür (`logs/run_r5_cc_stable.log:1236`). Bu sayı bir perplexity değildir ve başka protokollerin perplexity değerleriyle karşılaştırılmaz.

### 5.12 Eğitim — İst. 15

Okuyucunun ağırlıkları gradyan tabanlı geri yayılım ile güncellenir. Nokta konumları gradyan hesaplanmaksızın güncellenir: aday konum üretilir, aynı mini-batch üzerinde adayla görev yitimi ölçülür ve adayın yitimi taban yitimin bir eşik altında olması hâlinde aday kabul edilir (`quann_cloud/src/qcloud_quann.py:212-217`); yitim tabanın eşik üstüne çıkması hâlinde konum önceki değerine geri alınır, arama aralığı aday tarafında daraltılır (`quann_cloud/src/qcloud_quann.py:218-226`).

Bu varyantın maliyet sınırı ve başarım sınırı kayıtlıdır: sıfırıncı mertebe yaklaşımı aday başına bir ileri geçiş gerektirdiğinden büyük modelde uygulanabilir bulunmamıştır (`quann_cloud/FINDINGS.md:22`). Küçük ölçekli ölçümde (6.000 eğitim ve 500 değerlendirme cümlesi, dizi uzunluğu 8, 1024 parçalı sözcük dağarcığı; `quann_cloud/FINDINGS.md:18-19`) held-out perplexity eğitilmemiş temel çizgi için 1045,03 (`quann_cloud/FINDINGS.md:28`), geri yayılımlı eğitim için 563,45 (`quann_cloud/FINDINGS.md:29`), sıfırıncı mertebe eğitim için 1016,30 olmuştur (`quann_cloud/FINDINGS.md:30`). Sıfırıncı mertebe varyantı geri yayılımlı eğitime göre daha yüksek perplexity üretmiştir; bu varyant doğruluk iddiasıyla değil, geri yayılım gerektirmeyen nokta güncellemesi sağlamasıyla istenir.

## 6. Alternatif embodiments

- **Agregasyon türü.** Nokta yolunda maksimum havuzlama yerine ortalama ya da dikkat ağırlıklı havuzlama; çift yolunda ortalama yerine maksimum havuzlama kullanılabilir. İst. 1(c) havuzlama türünü değil, girdi sırasından bağımsız simetrik bir küme agregasyonunu sınırlandırır; İst. 4 ölçülen ikili (maksimum + ortalama) için bir bağımlı seçenektir.
- **Geometrik büyüklük kümesi.** Çift-geometri özelliği §5.3'teki beş bileşenle sınırlı değildir; açısal benzerlik yerine ya da ona ek olarak noktalar arası açının tangenti, normalize edilmiş iç çarpım konuma bağlı ölçütleri kullanılabilir. İst. 1(b) "konumlarına bağlı geometrik büyüklükler" der; bileşen listesi İst. 3'tedir.
- **Çift hesabından kaçınma.** Çift sayısı N² ile büyüdüğünden, İst. 13'ün (künye `ISTEMLER.md:61`) küresel-istatistik varyantı çift hesabını tümüyle kaldırır (max/min/mean/std + centroid + varyans; `purecloudlm/model.py:62-63`, `purecloudlm/model.py:66`). Bu, büyük N'li tabular kayıtlar için alternatif yoldur.
- **Kümeleme türü.** Bölge isimlendirmede küme merkezli KMeans ölçülmüştür (`complaints/src/explain_region.py:47`); alternatif olarak yoğunluk tabanlı ya da hiyerarşik kümeleme ve §5.7'deki prototip-temelli bölgesel tanım kullanılabilir.
- **Bölge etiketi kaynağı.** Etiket, girdi sembollerinin sıklık sıralamasından türetildiği yanı sıra, bölgedeki kayıtların ham metin öbeklerinden (prototipte bölge başına 350 karakterlik örnekler saklanır; `complaints/src/explain_region.py:60`) de türetilebilir.
- **Okuyucu türü.** Okuyucu sınıflandırma başlığı (`fraud/src/explain_pilot.py:103`) ya da dil modeling başlığı (`cloudlm/model.py:314`) taşır; ikiden fazla başlığın aynı küme gösterimine bağlanması da mümkündür.
- **Kantizasyon.** INT8 ölçülmüştür; daha düşük bit genişliğinde ya da ağırlıkları kantize eden şemalarda da aynı paket yapısı çalışır. Ölçülen iki INT8 şeması (tensor ve kanal ölçekli) ayrık testte farklı maliyet vermiştir (`fraud/results/edge_eval_full_scale.json:63-66`).
- **Nokta güncelleme kuralı.** İst. 15'teki (künye `ISTEMLER.md:67`) kabul/geri alma kuralı ölçülen kuraldır (`quann_cloud/src/qcloud_quann.py:212-226`); arama aralığının daraltılması yerine adım büyüklüğünün küçültülmesi eşdeğer daraltma kuralı kullanılabilir.

## 7. Ölçülmemiş hususlar (iddia edilmez)

Aşağıdaki nicelikler depoda ölçülmemiştir; bu taslakta hiçbir sayıyla desteklenmez ve hiçbiri için üstünlük ileri sürülmez:

1. **D boyutunun alt sınırı.** "3 ≤ D ≤ 64" bir seçim aralığı olarak tanımlanır; D = 3 için hiçbir ölçüm yoktur. Ölçülen D değerleri yalnız §5.2'de sayılanlardır.
2. **Aynı ağırlıklarla çoklu görev başarımı.** Dil başlığı ve tabular başlığın aynı parametrelerle eşzamanlı başarımı tek bir ölçüm dosyasında kayıtlı değildir (§5.5). İst. 16 yapısal bir seçimi tanımlar, eşzamanlı başarım iddia etmez.
3. **Küresel-istatistik varyantının başarımı.** İst. 13'ün O(N) varyantı için geçerli bir ayrık-test başarımı ölçülmemiştir; yalnız maliyet yapısı (doğrusal) tanımlıdır.
4. **Açıklama üretmenin ayrı bir araca göre toplam maliyet kazancı.** §2(b)'deki iki süre farklı iş kümelerine aittir; bunlardan oran veya maliyet kazancı türetilmez.
5. **Bölge sayısının seçimi.** Ölçülen bölge sayısı 8'dir; 8'in optimal olduğuna ya da başka bölge sayılarına göre üstünlüğüne dair ölçüm yoktur.
6. **Şek. 6 ve Şek. 7 dışındaki çizimlerde ölçüm yoktur.** Şek. 1-5 ve 8 şematiktir.
7. **Yenilik/buluş basamağı hükmü.** Bu belge bir ön araştırma ya da hukuki yeterlilik görüşü değildir; §2'deki "yerleşik yaklaşım" nitelendirmesi vekilin teknik kabulüdir ve incelemede doğrulanmalıdır.

## 8. Sınırlar ve olumsuz ölçümler (kayıt)

Buluşun uygulanabilirliğini sınırlayan, depoda kayıtlı olumsuz ölçümler şunlardır. Bunlar gizlenmez; buluşun kapsamı bu sınırların ötesine genişletilmez.

- **§8.1 Dil görevinde temel çizgiye göre başarım farkı.** Ortak held-out perplexity'de (2000 cümle) nokta-bulut modeli 6,45, nedensel (causal) temel çizgi 2,94 ölçülmüştür (`logs/run_compare.log:17-18`). Nokta bulutu temsilinin dil görevinde perplexity üstünlüğü yoktur; buluşun açıklanabilirlik etkisi dil görevinde bu farktan bağımsızdır ve perplexity ekseninde bir iddia taşımaz.
- **§8.2 Kantizasyon doğruluk maliyeti.** INT8 kantizasyonu ayrık testte AUC'yi düşürür: tensor ölçekli için −0,008593 ve kanal ölçekli için −0,008363 (`fraud/results/edge_eval_full_scale.json:65-66`). Bu maliyet §3'te etkiyle birlikte ölçülmüş bir sınır olarak kaydedilmiştir.
- **§8.3 Sıfırıncı mertebe nokta eğitiminin başarım maliyeti.** Küçük ölçekte held-out perplexity 1016,30 (sıfırıncı mertebe) ile 563,45 (geri yayılımlı) ölçülmüştür (`quann_cloud/FINDINGS.md:29-30`); eğitilmemiş temel çizgi 1045,03'tür (`quann_cloud/FINDINGS.md:28`). Ayrıca aday başına ek ilerleyiş geçişi nedeniyle büyük modelde uygulanabilir bulunmamıştır (`quann_cloud/FINDINGS.md:22`). İst. 15 (`ISTEMLER.md:67`) geri yayılım gerektirmeyen güncelleme mekanizmasını ister, başarımla değil.
- **§8.4 Bölge etiketlerinin tek-olmayışı.** Geometrik olarak ayrı iki bölge aynı insan-okunur etiketi taşıyabilir: bölge anahtarı 2 ve bölge anahtarı 4'ün ölçülen etiketi aynıdır (`complaints/results/region_map.json:422` ve `complaints/results/region_map.json:838`). Denetim bu yüzden etiketle değil, bölge indeksiyle birlikte yapılır.
- **§8.5 Model-sonrası karşılaştırmanın niteliği.** §2(a)'daki iki sayı farklı niceliklerdir ve aynı veri/tohum kümesinde ölçülmüş olsalar bile birebir karşılaştırılabilir ölçütler değildir; oran kurulmaz.
- **§8.6 Bağlam-belleği mekanizması için ölçülen olumsuzluk (istemlenmedi).** Çok cümleli hiyerarşide bağlamı kodlayan bellek-bağlamsal varyant, görev-sabit ablation'a göre ayrık TEST yitiminin daha yüksek ölçüldüğü iki bağımsız örnekte bulunmuştur: 3,5425 (`logs/run_r5_ablate_stable_on_memorytask.log:1332`) ve 3,6271 (`logs/run_r5_cc_memory_stablesz.log:894`) ile 3,5581 (`logs/run_r5_seed2_ablate_s43.log:982`) ve 3,6336 (`logs/run_r5_seed2_memory_stablesz_s43.log:631`). Bu nedenle İst. 14 (`ISTEMLER.md:64`) yalnız pencere-bulut yapısını içerir; "bağlam belleği" biçiminde bir üst kavram istemi yazılmamıştır. Bu kayıt bir önceki teknik / başvuru stratejisi notudur, buluşun işleyişi olarak sunulmaz.

## 9. Buluşun özeti

Nokta bulutu temelli bir makine öğrenmesi temsilinde görev çıktısı ile o kararın açıklaması aynı eğitimli modelin aynı nokta bulutundan üretilir. Görev girdisi (sıralı token dizisi veya bir tabular kaydın ölçülen alan değerleri) bir bağlam kodlayıcı ile D-boyutlu noktalara dönüştürülür; noktalar için nokta özellikleri, nokta çiftleri için konuma bağlı çift-geometri özellikleri hesaplanır ve her ikisi girdi sırasından bağımsız simetrik bir küme agregasyonu ile tek küme gösteriminde birleştirilir. Okuyucu bu gösterimden görev çıktısını üretir. Aynı nokta bulutu üzerinde nokta çıkarma etkisi, bölge isimlendirme ve bölge kayması işlemlerinden en az biri uygulanarak açıklama verisi üretilir; ayrı bir açıklama modeli, ayrı açıklama eğitimi veya ayrı açıklama girdisi gerekmez. Ölçülen teknik etkiler bellek ayak izinin 528,9 KB'dan 138,2 KB'ye inmesi, uç aygıtta p50 0,015912843984551728 ms işlem gecikmesi, açıklama maliyetinin N + 1 ilerleyiş geçişine indirgenmesi ve etiketsiz sürüklenebilirlik işaretinin 2,4× orantıyla ölçülmesidir; doğruluk üstünlüğü iddia edilmez.

## 10. İstemler künyesi (tam metin `ISTEMLER.md` içindedir)

| İstem | Tür | Konu | Bu belgede dayanağı |
|---|---|---|---|
| 1 | bağımsız · yöntem | Çift kullanımlı nokta bulutu: görev çıktısı + aynı buluttan açıklama | §3, §5.1-§5.8 |
| 2 | bağımlı | D boyutu; ölçülen D = 8, 32, 64 | §5.2 |
| 3 | bağımlı | Çift-geometri bileşenleri, genişlik 3D + 2 | §5.3 |
| 4 | bağımlı | Maksimum + ortalama havuzlama, birleştirme | §5.4 |
| 5 | bağımlı | Maskeleme ve geçerli-çift normalizasyonu | §5.4 |
| 6 | bağımlı | Nokta çıkarma etkisi; N + 1 geçişin tek toplu çağrıya indirgenmesi | §5.6 |
| 7 | bağımlı | Karşı-olgusal duyarlılık olarak raporlama ve sıralama kuralı | §5.6 |
| 8 | bağımlı | Kümeleme ile bölge isimlendirme, sembol sıklığından etiket | §5.7 |
| 9 | bağımlı | Eğitim-kümesi prototipleri ile bölgesel atama | §5.7 |
| 10 | bağımlı | Tam değişim / KL ile bölge kayması ve yeni bölge işareti | §5.8 |
| 11 | bağımlı | INT8 okuyucu + kendi kendine yeten paket (191, 192, 193) | §5.9 |
| 12 | bağımlı (11) | Açıklamanın kantize oturumda üretilmesi, etiket eşlemesi | §5.9 |
| 13 | bağımlı | O(N) küresel-istatistik varyantı | §5.10 |
| 14 | bağımlı | Cümle-bulut → pencere-bulutu hiyerarşisi | §5.11 |
| 15 | bağımlı | Gradyansız nokta güncelleme + geri yayılımlı okuyucu | §5.12 |
| 16 | bağımlı | Görev türüne göre seçilen çıktı başlığı | §5.5 |
| 17 | bağımsız · sistem | Modül olarak aynı zincir + yorumlama motoru | §5.1 |
| 18 | bağımlı (17) | Dağıtım deposu olarak kendi kendine yeten paket | §5.9 |
| 19 | bağımsız · kayıt ortamı | 1 numaralı yöntemin (a)-(f) adımlarını yürüten komutlar | §3 |

Künye `ISTEMLER.md` içindeki sırayla birebirdir; istem sayısı oradan okunmuştur (bu belgede sayı iddiası olarak tekrarlanmaz).
