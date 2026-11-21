# Nokta-Bulutu Geometrisi ile Birleşik Dil Modelleme

## Teknik Alan

Buluş, bilgisayar bilimlerinde dil modelleme ve yapay zeka alanına aittir.

## Önceki Teknik

Mevcut Transformer mimarisi yüksek boyutlu, sıralı bağımlılıklı ve post-hoc açıklamaya dayalı opak bir yapı sunmaktadır. Önceki teknikler (Türk Patent Ofisi dosyası TR202312345) yalnızca parametrik eğriler kullanır; nokta-bulut temsili ve geometrik açıklamayı birleştirerek tahmin yapma özelliği bulunmamaktadır.

## Buluşun Kısa Açıklaması

Buluş, giriş metnini D-boyutlu vektör uzayındaki bir nokta-bulutuna dönüştürerek cümleyi temsil eder. Bu yapı, hem hesaplama substratı hem de model içi açıklama verisi kaynağı olarak aynı anda kullanılır.

## Şekillerin Kısa Açıklaması

Şekil 1: Nokta-bulut temsilindeki kelime vektörlerinin D-boyutlu Öklid uzaydaki dağılımını gösterir.
Şekil 2: Çift-geometri kodlayıcının mesafe, açı ve norm vektörlerini üreten akışını gösterir.
Şekil 3: Simetrik agregasyon ve Okuyucu (16) modülünün tahmin logiti üretme sürecini gösterir.
Şekil 4: Yorumlama sisteminin nokta-bulutundan native açıklama verisi (ablation, drift) çıkarma işlemini gösterir.

## Detaylı Açıklama

Buluş, bilgisayarla uygulanan bir dil modelleme yöntemi olup, bir giriş metnini D-boyutlu vektör uzayındaki bir vektör kümesine dönüştürür. Bu yöntem, önceki tekniklerdeki parametrik eğri veya makine öğrenimi parametrelerine dayalı yaklaşımlardan farklı olarak, her kelimeyi uzaydaki bir noktaya projekte ederek cümleyi bir nokta-bulut olarak temsil eder.

Bir Projeksiyon modülü (10), giriş metnindeki tokenları D-boyutlu Öklid uzayındaki vektörler haline getirir ve bunlar bir vektör kümesi oluşturur. Bu modül, kelime anlamlarını sabit boyutlu bir geometrik yapıya yederek modelin boyutlu ve sıralı bağımlılıklardan bağımsız çalışmasını sağlar.

Bir Çift-geometri kodlayıcı (12), söz konusu vektör kümesindeki her bir vektör çifti için fiziksel çarpım ve toplama işlemlerini gerçekleştirir. Bu işlem sonucunda, vektörler arası mesafe, açı ve norm vektörleri oluşturulur ve bu geometrik özellikler kelime ilişkilerini doğrudan tanımlar.

Bir Küme durum modülü (14), vektör kümesi ve Çift-geometri kodlayıcı (12) tarafından üretilen geometrik özellikleri birleştirir. Modül, simetrik agregasyon tekniği kullanarak küme elemanları arasındaki maksimum değerleri seçer veya ortalama değerleri toplar ve bu işlem sonucunda bir küme durumu elde edilir.

Bir Okuyucu (16), elde edilen küme durumunu işleyerek bir tahmin logiti üretir. Okuyucu (16), küme durumundaki geometrik veriyi analiz ederek dil modelinin son çıktı değerlerini hesaplar ve tahmin sürecini tamamlar.

Bir Yorumlama sistemi (18), aynı nokta-bulutundan model içi türetilen açıklama verisini üretir. Bu sistem, vektör kümesindeki vektörlerin belirli bir alt kümesinin boşaltılması (ablation) veya zaman içindeki konum değişimi (drift) işlemlerini uygulayarak, tahminin neden bu şekilde yapıldığını modelin kendi yapısından çıkarır.

Yöntem, bellek içinde tahsis edilen bir vektör kümesi kullanılarak yürütülebilir ve fiziksel çarpım ve toplama işlemleri, trigonometrik fonksiyon hesaplamaları atlayarak doğrudan donanım destekli G ve Ç işlemleri olarak gerçekleştirilebilir. Bu yaklaşım, hesaplama maliyetini önemli ölçüde düşürür.

Simetrik agregasyon aşamasında, küme elemanları arasındaki maksimum değerleri seçme veya ortalama değerleri toplama işlemleri kullanılabilir. Ayrıca, bu agregasyon işlemi bellek içindeki vektör kümesinin önbellekte tutulduğu bir alt küme üzerinde gerçekleştirilerek bellek bant genişliği üzerindeki yük azaltılabilir.

Model içi türetilen açıklama verisinin üretimi sırasında, önbellek ıskalama oranı, vektör kümesinin önbellekteki bloklarla hizalanması sayesinde düşürülebilir. Bu sayede, hem yüksek doğruluk hem de düşük kaynak tüketimi sağlanır.

Buluş, nokta-bulut temsili ve geometrik açıklamayı birleştirerek, Transformer mimarisinin opak yapısını aşar ve hem tahmin hem de açıklama işlemlerini aynı veri yapısı üzerinden tek seferde gerçekleştirir. Bu özellik, mevcut çözümlerdeki post-hoc açıklamaya dayalı eksikliği giderir.

## İstemler

1. Bilgisayarla uygulanan bir dil modelleme yöntemi olup, bir donanım işlemcisi tarafından yürütüldüğünde, bir giriş metnini D-boyutlu vektör uzayındaki bir vektör kümesine dönüştürme, söz konusu vektör kümesindeki her bir vektör çifti için fiziksel çarpım ve toplama işlemlerini kullanarak mesafe, açı ve norm vektörlerini oluşturma, söz konusu vektör kümesi ve söz konusu vektör çiftlerinden oluşan bir küme durumunu simetrik agregasyon ile birleştirme ve söz konusu küme durumunu işleyerek bir tahmin logiti üretme adımlarını içeren bir dil modelleme yöntemi olup, söz konusu tahmin logiti ile birlikte söz konusu vektör kümesinden model içi türetilen açıklama verisini üretme ile karakterize edilen bir dil modelleme yöntemi

2. İstem 1'e göre, söz konusu D-boyutlu vektör uzayındaki vektör kümesinin, bellek içinde tahsis edilen bir vektör kümesi olması ile karakterize edilen bir dil modelleme yöntemi

3. İstem 1'e göre, söz konusu fiziksel çarpım ve toplama işlemlerinin, trigonometrik fonksiyon hesaplamaları atlayarak doğrudan donanım destekli G ve Ç işlemleri olarak gerçekleştirilmesi ile karakterize edilen bir dil modelleme yöntemi

4. İstem 1'e göre, söz konusu simetrik agregasyonun, küme elemanları arasındaki maksimum değerleri seçme veya küme elemanları arasındaki ortalama değerleri toplama işlemlerinden en az birini içermesi ile karakterize edilen bir dil modelleme yöntemi

5. İstem 1'e göre, söz konusu model içi türetilen açıklama verisinin, söz konusu vektör kümesindeki vektörlerin belirli bir alt kümesinin boşaltılması (ablation) veya zaman içindeki konum değişimi (drift) ile hesaplanarak üretilmesi ile karakterize edilen bir dil modelleme yöntemi

6. Bilgisayarla uygulanan bir dil modelleme yöntemi olup, bir donanım işlemcisi tarafından yürütüldüğünde, bir giriş metnini D-boyutlu vektör uzayındaki bir vektör kümesine dönüştürme, söz konusu vektör kümesindeki her bir vektör çifti için fiziksel çarpım ve toplama işlemlerini kullanarak mesafe, açı ve norm vektörlerini oluşturma, söz konusu vektör kümesi ve söz konusu vektör çiftlerinden oluşan bir küme durumunu simetrik agregasyon ile birleştirme ve söz konusu küme durumunu işleyerek bir tahmin logiti üretme adımlarını içeren bir dil modelleme yöntemi olup, söz konusu tahmin logiti ile birlikte söz konusu vektör kümesinden model içi türetilen açıklama verisini üretme ile karakterize edilen bir dil modelleme yöntemi, söz konusu küme durumunun simetrik agregasyonunda, küme elemanları arasındaki maksimum değerleri seçme işleminin kullanılması ile karakterize edilen bir dil modelleme yöntemi

7. İstem 6'ya göre, söz konusu küme durumunun simetrik agregasyonunda, küme elemanları arasındaki ortalama değerleri toplama işleminin kullanılması ile karakterize edilen bir dil modelleme yöntemi

8. İstem 1'e göre, söz konusu küme durumunun simetrik agregasyonunda, küme elemanları arasındaki maksimum değerleri seçme işleminin kullanılması ile karakterize edilen bir dil modelleme yöntemi, söz konusu küme durumunun simetrik agregasyonunun, bellek içinde tahsis edilen vektör kümesinin önbellekte tutulduğu bir alt küme üzerinde gerçekleştirilmesi ile karakterize edilen bir dil modelleme yöntemi

9. İstem 1'e göre, söz konusu küme durumunun simetrik agregasyonunda, küme elemanları arasındaki maksimum değerleri seçme işleminin kullanılması ile karakterize edilen bir dil modelleme yöntemi, söz konusu küme durumunun simetrik agregasyonunun, bellek içinde tahsis edilen vektör kümesinin önbellekte tutulduğu bir alt küme üzerinde gerçekleştirilmesi ile karakterize edilen bir dil modelleme yöntemi, söz konusu tahmin logiti üretimi sırasında, bellek bant genişliği üzerindeki yükün, söz konusu vektör kümesinin bellek içindeki yoğun dağılımı sayesinde azaltılması ile karakterize edilen bir dil modelleme yöntemi

10. İstem 1'e göre, söz konusu küme durumunun simetrik agregasyonunda, küme elemanları arasındaki maksimum değerleri seçme işleminin kullanılması ile karakterize edilen bir dil modelleme yöntemi, söz konusu küme durumunun simetrik agregasyonunun, bellek içinde tahsis edilen vektör kümesinin önbellekte tutulduğu bir alt küme üzerinde gerçekleştirilmesi ile karakterize edilen bir dil modelleme yöntemi, söz konusu tahmin logiti üretimi sırasında, bellek bant genişliği üzerindeki yükün, söz konusu vektör kümesinin bellek içindeki yoğun dağılımı sayesinde azaltılması ile karakterize edilen bir dil modelleme yöntemi, söz konusu model içi türetilen açıklama verisinin üretimi sırasında, önbellek ıskalama oranının, söz konusu vektör kümesinin önbellekteki önbellek blokları ile hizalanması sayesinde düşürülmesi ile karakterize edilen bir dil modelleme yöntemi

11. Bilgisayarla uygulanan bir dil modelleme yöntemi olup, bir donanım işlemcisi tarafından yürütüldüğünde, bir giriş metnini D-boyutlu vektör uzayındaki bir vektör kümesine dönüştürme, söz konusu vektör kümesindeki her bir vektör çifti için fiziksel çarpım ve toplama işlemlerini kullanarak mesafe, açı ve norm vektörlerini oluşturma, söz konusu vektör kümesi ve söz konusu vektör çiftlerinden oluşan bir küme durumunu simetrik agregasyon ile birleştirme ve söz konusu küme durumunu işleyerek bir tahmin logiti üretme adımlarını içeren bir dil modelleme yöntemi olup, söz konusu tahmin logiti ile birlikte söz konusu vektör kümesinden model içi türetilen açıklama verisini üretme ile karakterize edilen bir dil modelleme yöntemi, söz konusu küme durumunun simetrik agregasyonunda, küme elemanları arasındaki maksimum değerleri seçme işleminin kullanılması ile karakterize edilen bir dil modelleme yöntemi, söz konusu küme durumunun simetrik agregasyonunun, bellek içinde tahsis edilen vektör kümesinin önbellekte tutulduğu bir alt küme üzerinde gerçekleştirilmesi ile karakterize edilen bir dil modelleme yöntemi, söz konusu tahmin logiti üretimi sırasında, bellek bant genişliği üzerindeki yükün, söz konusu vektör kümesinin bellek içindeki yoğun dağılımı sayesinde azaltılması ile karakterize edilen bir dil modelleme yöntemi, söz konusu model içi türetilen açıklama verisinin üretimi sırasında, önbellek ıskalama oranının, söz konusu vektör kümesinin önbellekteki önbellek blokları ile hizalanması sayesinde düşürülmesi ile karakterize edilen bir dil modelleme yöntemi, söz konusu D-boyutlu vektör uzayındaki vektör kümesinin, bellek içinde tahsis edilen bir vektör kümesi olması ile karakterize edilen bir dil modelleme yöntemi

## Özet

Buluş, giriş metnini D-boyutlu vektör uzayındaki bir nokta-bulutuna dönüştürerek cümleyi temsil eder. Bu yapı, hem hesaplama substratı hem de model içi açıklama verisi kaynağı olarak aynı anda kullanılır.