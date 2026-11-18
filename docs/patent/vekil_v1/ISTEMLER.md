# İSTEMLER — Vekil Taslağı v1 (çalışma belgesi; başvuru metni değildir)

**Buluşun başlığı:** Nokta bulutu temelli çift kullanımlı temsil üzerinden görev çıktısı ve gömülü açıklanabilirlik üreten yöntem, sistem ve kayıt ortamı

Bu taslak `docs/patent/final_v3/` metnini değiştirmeyen, bağımsız bir vekil hazırlığıdır;
önceki taslaklardan cümle kopyalanmamıştır. Her istemin altında yalnız vekil dayanağı
(`dosya:satır`) düşülmüştür; dayanak satırları başvuruya girmez. Bağımsız istemlerde sayı
yoktur: sayısal sınırlar bilinçli olarak bağımlı istemlere alınmıştır. Referans numaraları
`OZET.md` §3 çizim listesiyle birebir ortaktır.

## A. Bağımsız istemler

> **Vekil açıklaması (istem metni değildir):** parantez içindeki üç basamaklı sayılar (100, 110, 120, 160, 170, …) ölçüm değil, `TARIFNAME.md:45` satırındaki çizim künyesinde tanımlı referans işaretleridir; 'İst. N' biçimindeki iç başvurular bu dosyadaki istem numaralarıdır. Ölçüm değeri taşıyan her cümle parantez içinde ham kaynak satırını (dosya:satır) taşır.

**İstem 1 — Yöntem.** Bilgisayarla uygulanan bir makine öğrenmesi yöntemi olup, aşağıdaki adımları içermesiyle karakterize edilir:
(a) bir görev girdisini (100) bir bağlam kodlayıcı (110) ile işleyerek bağlamı kodlanmış gösterimler üretme; görev girdisinin sıralı bir token dizisi veya bir tabular kaydın birden çok ölçülen alan değeri olması ve her bir bağlamı kodlanmış gösterimin D-boyutlu bir uzayda bir nokta olarak temsil edilmesiyle, N sayıda noktadan oluşan bir nokta bulutunun (120) oluşturulması; böylece tabular bir kayıtta ölçülen alan değerlerinin o kaydın nokta bulutundaki noktaları olarak temsil edilmesi;
(b) nokta bulutundaki noktalar için nokta özelliklerinin ve nokta çiftleri için, çifti oluşturan iki noktanın konumlarına bağlı geometrik büyüklüklerden türetilen çift-geometri özelliklerinin hesaplanması (130);
(c) nokta özellikleri ile çift-geometri özelliklerinin, noktaların ve nokta çiftlerinin girdi sırasından bağımsız olan simetrik bir küme agregasyonu (140) ile tek bir küme gösteriminde (150) birleştirilmesi;
(d) küme gösteriminin bir okuyucu (160) ile işlenerek bir görev çıktısının (162) üretilmesi;
(e) görev çıktısının üretildiği eğitimli modelin kendi nokta bulutu (120) üzerinden, ayrı bir açıklama modeli, ayrı bir açıklama eğitimi veya ayrı bir açıklama girdisi kullanılmaksızın, (i) nokta çıkarma etkisi, (ii) bölge isimlendirme ve (iii) bölge kayması işlemlerinden en az birinin uygulanmasıyla açıklama verisinin üretilmesi (170); ve
(f) görev çıktısı ile açıklama verisinin, aynı eğitimli modelin aynı küme gösteriminden türeyen birlikte çıktılar olarak bir çıkış arayüzüne (180) verilmesi;
burada nokta bulutu (120), görev çıktısının hesaplandığı temsil ile açıklama verisinin türetildiği temsil olarak, eğitimli modelin dışına ikinci bir açıklama modeli eklenmeksizin ortaklaşa kullanılır.

> Dayanak: (a) `cloudlm/model.py:312-313`; `fraud/src/explain_pilot.py:102`. (b) `cloudlm/model.py:157-168`. (c) `cloudlm/model.py:153`, `cloudlm/model.py:177` ve `cloudlm/model.py:179`; sıra bağımsızlığı `cloudlm/model.py:47`. (d) `cloudlm/model.py:313-314`; `fraud/src/explain_pilot.py:103`. (e) `fraud/src/explain_pilot.py:93-112`; `aml/src/e1_explain.py:296-299`; `complaints/src/audit_drift.py:36-45`. (f) `fraud/pilot/infer.py:107`; `fraud/results/deploy/manifest.json:96-98`.

## B. İstem 1'e bağımlı istemler (yöntem)

**İstem 2.** İstem 1'e göre yöntem olup, D boyutunun görev alanına göre seçilen bir tamsayı olması ve yapılandırma dosyalarında D = 8, D = 32 ve D = 64 değerlerinin ölçülmüş uygulama örnekleri olarak kaydedilmesiyle karakterize edilir.
> Dayanak: D = 8 `fraud/results/full_scale_hybrid.json:56`; D = 8 (ikinci yapılandırma) `quann_cloud/FINDINGS.md:17`; D = 32 ve D = 64 yapılandırmaları `logs/run_r4_resume.log:67` ve `logs/run_r4_resume.log:70`. Kapasite/yarış hükmü bu taslağın konusu değildir ve tekrarlanmaz (bkz. `docs/patent/final_v3/TARIFNAME.md:167`, `BLUS_NOTU.md:31`). Seçim aralığının alt ucu (D = 3) ölçülmemiştir: bkz. `OZET.md` §5.

**İstem 3.** İstem 1'e göre yöntem olup, çift-geometri özelliklerinin, bir nokta çifti için iki nokta vektörü arasındaki fark vektörünü, bu fark vektörünün Öklid normu olan çift mesafesini, iki nokta vektörü arasındaki açısal benzerliği ve çiftteki her iki noktanın norm değerlerini içermesi ve çift-geometri özelliği genişliğinin bu bileşenlerin birleştirilmesiyle 3D + 2 olarak belirlenmesiyle karakterize edilir.
> Dayanak: bileşen sırası ve toplam genişlik `cloudlm/model.py:63-64`; hesaplama `cloudlm/model.py:157-168`. "D + 4" etiketi kodda doğrulanamadığı için kullanılmaz: `docs/patent/final_v3/TARIFNAME.md:220`.

**İstem 4.** İstem 1'e göre yöntem olup, simetrik küme agregasyonunun, nokta özellikleri üzerinde noktalar üzerinden maksimum havuzlama, çift-geometri özellikleri üzerinde nokta çiftleri üzerinden ortalama havuzlama uygulanması ve iki havuzlama sonucunun bir birleştirme katmanı ile tek küme gösteriminde toplanması ile gerçekleştirilmesiyle karakterize edilir.
> Dayanak: `cloudlm/model.py:153`; `cloudlm/model.py:177`; `cloudlm/model.py:179`.

**İstem 5.** İstem 1'e göre yöntem olup, bağlamanın dolmadığı konumlara karşılık gelen noktaların ve bu noktalardan oluşan çiftlerin bir maske sinyali (144) ile agregasyon dışında bırakılması, bir çiftin geçerli sayılması için iki ucun da geçerli olması ve ortalama havuzlamanın geçerli çift sayısı üzerinden normalize edilmesiyle karakterize edilir.
> Dayanak: nokta maskesi `cloudlm/model.py:151-152,155`; çift maskesi ve geçerli-çift normalizasyonu `cloudlm/model.py:171-177`.

**İstem 6.** İstem 1'e göre yöntem olup, nokta çıkarma etkisinin, seçilen noktanın kopya bir nokta bulutunda değerinin sıfıra çekilmesi ve o noktanın maskesinin kapatılması, okuyucunun maskelenmemiş ve maskeli durumlarda çalıştırılması ve iki durum arasındaki görev çıktısı farkının logit farkı olarak kaydedilmesiyle; ve maskeli nokta bulutu türevlerinin tek bir toplu (batch) girdi olarak okuyucuya birlikte verilmesiyle, böylece N nokta için N + 1 ilerleyiş geçişinin tek bir toplu çağrıya indirgenmesiyle karakterize edilir.
> Dayanak: `fraud/src/explain_pilot.py:102-111`; toplu çağrı `fraud/pilot/infer.py:97-104`; logit farkı tanımı `fraud/results/explain_pilot.json:21`.

**İstem 7.** İstem 1'e göre yöntem olup, nokta çıkarma etkisiyle üretilen açıklama verisinin, katkılar toplamı görev çıktısını veren toplamsal bir atıf olarak değil, o noktanın kaldırılması hâlinde görev çıktısında ölçülen karşı-olgusal duyarlılık olarak raporlanması ve girdi birimlerinin, ortalaması alınmış mutlak logit farkı ile gözlem sayısının karekökünün çarpımı üzerinden sıralanmasıyla karakterize edilir.
> Dayanak: `fraud/results/explain_pilot.json:21-24`; kod `fraud/src/explain_pilot.py:111`.

**İstem 8.** İstem 1'e göre yöntem olup, bölge isimlendirmenin, çok sayıda kaydın küme gösterimleri veya nokta bulutu durumları üzerinde küme merkezli bir kümeleme algoritmasının çalıştırılması ve her bir bölgeye, o bölgeye atanan kayıtların girdi sembollerinin sıklık sıralamasından seçilen en az bir sembolün birleştirilmesiyle insan-okunur bir etiket atanması ile gerçekleştirilmesiyle karakterize edilir.
> Dayanak: `complaints/src/explain_region.py:47` ve `complaints/src/explain_region.py:66`; ölçülen bölge sayısı 8 `complaints/results/region_map.json:2`; ölçülen etiketler `complaints/results/region_map.json:422`.

**İstem 9.** İstem 1'e göre yöntem olup, bölgelerin, sınıfların eğitim kümesindeki gösterim ortalamalarından oluşan prototipler olarak tanımlanması ve bir gözleme, o gözlemin küme gösterimine en yakın olan prototipin bölgesinin atanmasıyla karakterize edilir.
> Dayanak: `aml/src/e1_explain.py:296-299`; ölçülen en-yakın-prototip doğruluğu `aml/results/e2_explain.json:16`.

**İstem 10.** İstem 1'e göre yöntem olup, bölge kaymasının, referans ve karşılaştırma veri gruplarındaki bölge nokta payları arasında, tam değişim ölçütü ve Kullback-Leibler yakınsaklığından en az biriyle sayısallaştırılmasını ve ayrıca referans dağılımda payı bir alt eşik altında olup karşılaştırma dağılımında payı bir üst eşiği aşan bölgelerin yeni ortaya çıkan bölge olarak işaretlenmesini içerdiğiyle karakterize edilir.
> Dayanak: tam değişim `complaints/src/audit_drift.py:36`; KL `complaints/src/audit_drift.py:40`; yeni bölge işareti `complaints/src/audit_drift.py:44-45`.

**İstem 11.** İstem 1'e göre yöntem olup, okuyucunun en az bir parametrik katmanının INT8 tam sayı kantizasyonu ile temsil edilmesi ve görev çıktısı ile açıklama verisini üreten dağıtımın, (i) kantize edilmiş model ağırlıklarını (191), (ii) girdi değerlerini model girdi kimliklerine eşleyen bir nicelleştirme tablosunu (192) ve (iii) model girdi kimliklerini insan-okunur girdi etiketlerine eşleyen bir çözümleme tablosunu (193) tek bir kendi kendine yeten pakette (190) taşıması; böylece aynı paketin hem görev çıktısını hem de açıklama verisinin insan-okunur biçimini uç bir aygıt üzerinde ürettiğiyle karakterize edilir.
> Dayanak: paket bileşenleri `fraud/results/deploy/manifest.json:20-24`; eşleme tanımları `fraud/results/deploy/manifest.json:97-98`; ayak izi `fraud/results/edge_eval_full_scale.json:15-16`.

**İstem 12.** İstem 11'e göre yöntem olup, nokta çıkarma etkisinin kantize edilmiş okuyucu oturumu üzerinde çalıştırılması ve maskelenen girdi biriminin insan-okunur etiketinin çözümleme tablosundan alınmasıyla karakterize edilir.
> Dayanak: INT8 oturumda ablation `fraud/pilot/infer.py:104` ve `fraud/pilot/infer.py:107`; etiket eşlemesi `fraud/results/deploy/manifest.json:98`.

**İstem 13.** İstem 1'e göre yöntem olup, çift-geometri özelliklerinin hesaplanması ve çiftler üzerinden ortalama havuzlama yerine, nokta bulutunun bir küresel küme istatistikleri kümesinin hesaplanması; küresel küme istatistikleri kümesinin nokta boyutları üzerinden maksimum, minimum, ortalama ve standart sapma havuzlaması ile bulut merkezi ve merkez sapmasını içermesi ve küme gösteriminin hesaplanan küresel küme istatistiklerinden türetilmesi (240) ile nokta sayısı ile doğrusal orantılı bir hesaplama maliyetinin elde edilmesiyle karakterize edilir.
> Dayanak: `purecloudlm/model.py:8-9,57,62-63`. Bu varyant için hiçbir doğruluk/perplexity üstünlüğü iddia edilmez; geçerli bir ayrık-test başarımı ölçülmemiştir (`docs/patent/final_v3/BLUS_NOTU.md:36`).

**İstem 14.** İstem 1'e göre yöntem olup, görev girdisinin birden çok cümleden oluşan bir metin olması; bağlam kodlayıcının her cümleyi bir cümle-bulut durumu olarak kodlaması (210), ardışık cümle-bulut durumlarının sabit bir pencere boyutu ve sabit bir adımla kaydırılarak birden çok komşu pencere-bulutunun türetilmesi (220), pencereler arası geometrik büyüklüklerin bir pencere-geometri kodlayıcısı ile toplanması (230) ve bu surette elde edilen çoklu-cümle küme gösteriminin okuyucuya verilmesiyle karakterize edilir.
> Dayanak: `contextcloud/model.py:20`, `contextcloud/model.py:121`, `contextcloud/model.py:127-128`, `contextcloud/model.py:196` ve `contextcloud/model.py:246`. Ayrık bölme `contextcloud/task_utils.py:16-19`; ölçülen ayrık TEST yitimi `logs/run_r5_cc_stable.log:1236`.

**İstem 15.** İstem 1'e göre yöntem olup, eğitimin, okuyucu parametrelerinin gradyan tabanlı geri yayılım ile güncellenmesini (260) ve nokta bulutundaki nokta konumlarının ise gradyan hesaplanmaksızın, aday nokta konumunun görev yitimi üzerindeki değişiminin bir eşikle ölçülmesine dayalı kabul ve geri alma işlemleriyle daraltılarak güncellenmesini (250) içerdiğiyle karakterize edilir.
> Dayanak: kabul `quann_cloud/src/qcloud_quann.py:212-217`; geri alma ve aralık daraltma `quann_cloud/src/qcloud_quann.py:218-226`. Doğruluk iddiası yoktur; küçük ölçekli ölçümde sıfırıncı mertebe eğitimin held-out perplexity'si geri yayılımlı eğitimden yüksektir: `quann_cloud/FINDINGS.md:28-30`.

**İstem 16.** İstem 1'e göre yöntem olup, görev çıktısının, görev girdisinin türüne göre seçilen bir çıktı başlığı ile üretilmesi; sıralı token dizisi girdisinde çıktının bir sonraki tokenın olasılık dağılımı, tabular kayıt girdisinde çıktının bir sınıf atama skoru olması ve her iki durumda da aynı küme gösteriminden türemesiyle karakterize edilir.
> Dayanak: dil başlığı `cloudlm/model.py:313-314`; tabular skor ve açıklamanın aynı logite dayanması `fraud/src/explain_pilot.py:103`. Aynı ağırlıkların iki görevde eşzamanlı başarımı ölçülmemiştir; bu istem yapısal bir seçim iddiası taşır (`docs/patent/final_v3/TARIFNAME.md:223`).

## C. Bağımsız sistem ve kayıt ortamı istemleri

**İstem 17 — Sistem.** En az bir işlemci ve işlemci tarafından yürütülebilir komutları saklayan bir bellek içeren bir bilgisayar sistemi olup, komutların yürütülmesi sırasında sistemin: bir görev girdisini (100) işleyen bir bağlam kodlayıcı modülü (110); bağlamı kodlanmış gösterimleri D-boyutlu vektörlere dönüştürerek N noktalı bir nokta bulutu (120) oluşturan bir nokta projeksiyon modülü; noktalar için nokta özellikleri ve nokta çiftleri için iki noktanın konumlarına bağlı geometrik çift-geometri özellikleri hesaplayan bir çift-geometri kodlayıcısı (130); bu özellikleri girdi sırasından bağımsız bir küme agregasyonu (140) ile tek bir küme gösteriminde (150) birleştiren bir agregasyon modülü; küme gösteriminden bir görev çıktısı (162) üreten bir okuyucu (160); görev çıktısının üretildiği aynı nokta bulutu üzerinden, nokta çıkarma etkisi, bölge isimlendirme ve bölge kayması işlemlerinden en az birini uygulayarak açıklama verisi üreten ve ayrı bir açıklama modeli gerektirmeyen bir yorumlama motoru (170); ve görev çıktısı ile açıklama verisini birlikte sunan bir çıkış arayüzü (180) olarak yapılandırılmasıyla karakterize edilir.
> Dayanak: İstem 1 ile aynı kanıt kümesi; modül ayrıştırmaya örnek `fraud/pilot/infer.py:97-107` (kantize oturum üzerinde açıklama üretimi).

**İstem 18.** İstem 17'ye göre bilgisayar sistemi olup, okuyucunun INT8 kantize edilmiş parametrik katmanlarını (191), girdi değerlerini model girdi kimliklerine eşleyen bir nicelleştirme tablosunu (192) ve model girdi kimliklerini insan-okunur girdi etiketlerine eşleyen bir çözümleme tablosunu (193) tek bir kendi kendine yeten pakette (190) taşıyan bir dağıtım deposunu; ve paketteki kantize katmanlar üzerinden nokta çıkarma etkisini hesaplayacak şekilde yapılandırılmış yorumlama motorunu içerdiğiyle karakterize edilir.
> Dayanak: `fraud/results/deploy/manifest.json:20-24,97-98`; `fraud/pilot/infer.py:104`.

**İstem 19 — Kayıt ortamı.** Üzerinde bilgisayar tarafından yürütülebilir komutlar saklayan bilgisayarca okunabilir, geçici olmayan bir kayıt ortamı olup, bu komutların yürütülmesi sırasında sistemin İstem 1'in (a)-(f) adımlarını gerçekleştirmesiyle karakterize edilir.
> Dayanak: İstem 1 ile aynı kanıt kümesi.

## D. Vekil notları (başvuruya girmez)

1. **İstem sayısı:** 19 istem; bağımsızlar 1 (yöntem), 17 (sistem), 19 (kayıt ortamı). Ek ücret eşikleri vekil tarifesinden doğrulanmalıdır — bu taslakta ücret sayısı iddia edilmez.
2. **Birlik (unity):** tek buluş fikri "aynı eğitimli nokta bulutunun hem görev çıktısının hem açıklamanın tek kaynağı olması"dır (İstem 1(e)-(f)); dikeyler bağımlı istemdedir. İncelemeci birlik itirazı yaparsa İstem 11-12 (uç paket) ve İstem 14 (çoklu-cümle) bölünmüş başvuruya ayrılabilir.
3. **İstem 1'e alınmayan özellikler (kasıtlı):** D aralığı, INT8 paket, kümeleme türü, TV/KL ölçütü, N + 1 toplama — hepsi bağımlı istemde; bağımsız istemin kapsamı ölçümle gereksiz yere daraltılmadı.
4. **R5 mekanizma bulgusu (NEGATİF) istemlenmedi:** bellek-bağlamsal varyantın ayrık TEST yitimi, görev-sabit ablation'da temel mekanizmadan kötü ölçülmüştür (`logs/run_r5_ablate_stable_on_memorytask.log:1332` = 3,5425 ile `logs/run_r5_cc_memory_stablesz.log:894` = 3,6271; ikinci örnek `logs/run_r5_seed2_ablate_s43.log:982` = 3,5581 ile `logs/run_r5_seed2_memory_stablesz_s43.log:631` = 3,6336). Bu yüzden İstem 14 yalnız pencere-bulut yapısını içerir; "bağlam belleği" biçiminde bir üst kavram istemi yazılmadı. Ayrıntı: `OZET.md` §4.

**E. Tarifname künyesi (tam metin `TARIFNAME.md` içindedir).** Bölüm adları, oradaki sırayla: teknik alan · önceki teknik ve açığı · buluşun kısa açıklaması ve teknik etkisi · şekillerin kısa açıklaması · detaylı açıklama · alternatif embodimentlar · ölçülmemiş hususlar · sınırlar ve olumsuz ölçümler · buluşun özeti · istemler künyesi.
