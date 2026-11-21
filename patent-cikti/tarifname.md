# Nokta-Bulutu Geometrisi ile Birleşik Dil Modelleme

## Teknik Alan

Buluş, yüksek boyutlu dil modellerinin verimliliğini artırmak ve model içi açıklanabilirliği sağlamak amacıyla, kelime dizilerini düşük boyutlu Öklid uzayındaki nokta-bulutlarına dönüştüren bir yöntemdir.

## Önceki Teknik

Geleneksel dil modelleri genellikle yüksek boyutlu ve sıralı bağımlılıklara sahip olup, açıklamaları post-hoc yöntemlerle sınırlıdır. Önceki teknikler ya simgesel cebir kullanır ya da tahmin ve açıklama için ayrı, birbirinden bağımsız yapılar gerektirir.

## Buluşun Kısa Açıklaması

Buluş, her kelimeyi D-boyutlu bir noktaya projekte ederek cümleyi bir nokta-bulutu olarak temsil eder. Bu geometrik yapıdaki mesafe, açı ve norm değerleri hem tahmin hem de native açıklama için ortak bir substrat olarak kullanılır.

## Şekillerin Kısa Açıklaması

Şekil 1: Nokta-bulutu temsilindeki kelime projeksiyonunu ve çift-geometri özelliklerini gösterir.

## Detaylı Açıklama

Buluş, girdi metnindeki her kelimeyi D-boyutlu bir fiziksel veri bloğuna projekte ederek cümleyi bir nokta-bulutu olarak modelleyerek, geleneksel dil modellerinin yüksek boyutlu ve post-hoc açıklamaya bağımlı yapısını aşar. Bu yaklaşım, nokta çiftleri arasındaki mesafe, açı ve norm değerlerini hesaplayarak çift-geometri özelliklerini tüter ve bu özellikleri simetrik bir küme agregasyonu ile birleştirir.

Projeksiyon modülü (10), girdi dizisinden gelen tokenları bellek adres uzayında sınırlı bir tamsayı aralığında tanımlanmış D-boyutlu fiziksel veri bloklarına hariteler. Çift-geometri kodlayıcı (12) ise bu bloklar arasındaki fiziksel mesafeleri, açısal ayrımları ve Öklid normlarını hesaplayarak geometrik bir zenginlik sağlar. Bu adımlar, kelimeyi soyut bir sembol yerine uzaysal bir nesne haline getirir.

Küme agregasyonu modülü (14), elde edilen fiziksel veri bloklarını ve çift-geometri özelliklerini paralel işlemci çekirdeklerinde simetrik olarak birleştirerek permütasyon-değişmez bir küme durumu oluşturur. Bu modül, nokta sıralarının değişmesinin tahmin sonuçlarını etkilemesini engeller ve bağlam-ağırlıklı dikkat mekanizması veya maksimum havuzlama gibi alternatifler kullanılabilir. Okuyucu (16), bu küme durumundan INT8 kantizasyon birimi üzerinden işleyerek düşük bit derinliğinde temsil edilen bir görev skoru üretir.

Yorumlama sistemi (18), aynı fiziksel veri bloğu üzerinden sistem içi hata analizi ve drift izleme verisi üretir. Bu sistem, nokta-bulutundan bir noktanın çıkarılması (leave-one-point-out) veya bölge kayması analizi yoluyla, modelin kararlarını doğrudan geometrik değişimlerle açıklayan native bir açıklama verisi sağlar. Böylece, açıklama işlemi için ayrı bir model veya ek aşamaya ihtiyaç kalmaz.

İlk uygulama biçiminde, D değeri 32 veya 64 olarak seçilir ve nokta sayısı 20 ile optimize edilerek R2b benzeri bir performans sağlanır. İkinci uygulama biçiminde ise nokta sayısı 10 ile 30 arasında sınırlanır ve Çift-geometri kodlayıcı (12), fiziksel veri bloklarını donanım önbelleği hizalamasına uygun veri blokları olarak düzenler. Bu yapı, hem kompakt bir dağıtım (INT8 ile ~5-6 MB) hem de yüksek hız sağlar.

Alternatif olarak, küme agregasyonu için bağlam-ağırlıklı bir dikkat mekanizması kullanılabilir ve Yorumlama sistemi (18), nokta-bulutundaki bölgeleri prototip isimlendirmesi ile etiketleyerek daha anlamlı açıklamalar üretebilir. Bu esneklik, farklı donanım ve kullanım senaryolarına uygunluğun garanti altına alınmasını sağlar. Sonuç olarak, buluş dil modellemesinde boyut, hız ve açıklanabilirlik sorunlarını aynı anda çözer.

## İstemler

1. Bilgisayarla uygulanan bir dil modelleme yöntemi olup, bir girdi metni dizisinden gelen her bir kelimeyi, bellek adres uzayında sınırlı bir tamsayı aralığında tanımlanmış D-boyutlu bir fiziksel veri bloğuna projekte ederek bir nokta bulutu oluşturmak; söz konusu nokta bulutundaki nokta çiftleri arasındaki fiziksel mesafe, açı ve norm değerlerini hesaplayarak çift-geometri özelliklerini türetmek; söz konusu fiziksel veri bloklarını ve çift-geometri özelliklerini, paralel işlemci çekirdeklerinde simetrik olarak birleştirerek permütasyon-değişmez bir küme durumu oluşturmak; söz konusu küme durumundan bir görev skoru üretmek; ve aynı fiziksel veri bloğu üzerinden sistem içi hata analizi ve drift izleme verisi üretmek adımlarını içerir, D-boyutlu fiziksel veri bloğuna projekte etme ve çift-geometri özelliklerini türetme adımları ile karakterize eden.

2. İstem 1'e göre, söz konusu D-boyutlu fiziksel veri bloğunun boyutunun 3 ile 64 arasında değişen bir tamsayı değere sahip olması ile karakterize eden.

3. İstem 1'e göre, söz konusu nokta bulutundaki nokta çiftleri arasındaki fiziksel mesafe, açı ve norm değerlerini hesaplayarak türetilen çift-geometri özelliklerini türetme adımının, söz konusu nokta bulutundaki her bir noktanın Öklid normunu ve söz konusu nokta çiftleri arasındaki açısal ayrımı hesaplamayı içerdiği ile karakterize eden.

4. İstem 1'e göre, söz konusu fiziksel veri bloklarını ve çift-geometri özelliklerini birleştirerek oluşturulan küme durumunu oluşturma adımının, söz konusu fiziksel veri bloklarını ve çift-geometri özelliklerini maksimum havuzlama (max-pool) işlemi ile birleştirmeyi içerdiği ile karakterize eden.

5. İstem 1'e göre, söz konusu görev skoru üretme adımının, söz konusu küme durumunu INT8 kantizasyon birimi üzerinden işleyerek düşük bit derinliğinde temsil edilen bir çıktı üretmeyi içerdiği ile karakterize eden.

6. İstem 1'e göre, sistem içi hata analizi ve drift izleme verisi üretme adımının, söz konusu nokta bulutundan bir noktanın çıkarılması (leave-one-point-out) veya bölge kayması analizi yoluyla native açıklama verisi türetmeyi içerdiği ile karakterize eden.

7. En az bir donanım işlemcisi ve işlemciye bağlı bir bellek içeren bilgi işlem sistemi olup, söz konusu işlemci, bir girdi metni dizisinden gelen her bir kelimeyi, bellek adres uzayında sınırlı bir tamsayı aralığında tanımlanmış D-boyutlu bir fiziksel veri bloğuna projekte eden Projeksiyon modülü (10); söz konusu nokta bulutundaki nokta çiftleri arasındaki fiziksel mesafe, açı ve norm değerlerini hesaplayarak çift-geometri özelliklerini türeten Çift-geometri kodlayıcı (12); söz konusu fiziksel veri bloklarını ve çift-geometri özelliklerini, paralel işlemci çekirdeklerinde simetrik olarak birleştirerek permütasyon-değişmez bir küme durumu oluşturan Küme agregasyonu modülü (14); söz konusu küme durumundan bir görev skoru üreten Okuyucu (16); ve aynı fiziksel veri bloğu üzerinden sistem içi hata analizi ve drift izleme verisi üreten Yorumlama sistemi (18); içermesi ile karakterize eden.

8. İstem 7'ye göre, İstem 7'deki Projeksiyon modülünün, girdi metni dizisindeki kelime sayısını 10 ile 30 arasında sınırlayarak söz konusu nokta bulutunu oluşturması ile karakterize eden.

9. İstem 7'ye göre, söz konusu Çift-geometri kodlayıcısının, söz konusu fiziksel veri bloklarını donanım önbelleği hizalamasına uygun veri blokları olarak düzenlemesi ile karakterize eden.

10. İstem 7'ye göre, söz konusu Küme agregasyonu modülünün, söz konusu fiziksel veri bloklarını ve çift-geometri özelliklerini bağlam-ağırlıklı bir dikkat mekanizması (attention-pool) ile birleştirmesi ile karakterize eden.

11. İstem 7'ye göre, Okuyucunun, söz konusu küme durumundan türetilen görev skorunu INT8 kantizasyon ile temsil edilen bir veri bloğu olarak dışa aktarması ile karakterize eden.

12. İstem 7'ye göre, Yorumlama sisteminin, söz konusu nokta bulutundaki bölgeleri prototip isimlendirmesi ile etiketleyerek native açıklama verisi üretmesi ile karakterize eden.

13. Geçici olmayan bir bilgisayarca okunabilir depolama ortamı olup, bir donanım işlemcisi tarafından yürütüldüğünde, bir girdi metni dizisinden gelen her bir kelimeyi, bellek adres uzayında sınırlı bir tamsayı aralığında tanımlanmış D-boyutlu bir fiziksel veri bloğuna projekte ederek bir nokta bulutu oluşturmayı; söz konusu nokta bulutundaki nokta çiftleri arasındaki fiziksel mesafe, açı ve norm değerlerini hesaplayarak çift-geometri özelliklerini türetmeyi; söz konusu fiziksel veri bloklarını ve çift-geometri özelliklerini, paralel işlemci çekirdeklerinde simetrik olarak birleştirerek permütasyon-değişmez bir küme durumu oluşturmayı; söz konusu küme durumundan bir görev skoru üretmeyi; ve aynı fiziksel veri bloğu üzerinden sistem içi hata analizi ve drift izleme verisi üretmeyi gerçekleştiren komutlar barındırması ile karakterize eden.

## Özet

Buluş, her kelimeyi D-boyutlu bir noktaya projekte ederek cümleyi bir nokta-bulutu olarak temsil eder. Bu geometrik yapıdaki mesafe, açı ve norm değerleri hem tahmin hem de native açıklama için ortak bir substrat olarak kullanılır.
