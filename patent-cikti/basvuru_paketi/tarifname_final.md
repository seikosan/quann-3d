# NOKTA-BULUTU GEOMETRİSİ İLE BİRLEŞİK DİL MODELLEME SİSTEMİ VE YÖNTEMİ

## Teknik alan

Buluş, bilgisayar uygulamalı doğal dil işleme ve makine öğrenmesi sistemleriyle ilgilidir. Özellikle, bağlamı kodlanmış tokenların D-boyutlu Öklid uzayında vektörlerle temsil edilmesi, vektörler arasındaki geometrik özelliklerin hesaplanması ve bu özelliklerin dil modelleme çıktısının üretilmesinde ve model içi açıklamada kullanılmasıyla ilgilidir.

## Tekniğin bilinen durumu

Dil işleme sistemlerinde tokenlar çoğunlukla yüksek boyutlu vektörlerle temsil edilir ve bağlam, sıralı kodlayıcılar veya dikkat katmanları aracılığıyla işlenir. Model çıktılarının açıklanması için kullanılan bazı yöntemler, eğitilmiş modelin dışında ek analizler yürütür. Nokta-kümesi yöntemleri ise girdi noktalarını birleştirebilmekle birlikte, dil tokenları için öğrenilen vektörleri, nokta çiftlerinin geometrik özelliklerini ve aynı geometrik temsil üzerinden üretilen model içi açıklamayı tek bir iş akışında birleştirmeyebilir.

Bu nedenle, token bağlamını koruyarak geometrik bir temsil oluşturabilen ve bu temsili hem görev çıktısı hem de açıklama verisi için kullanan bir bilgisayar uygulamalı düzene ihtiyaç bulunmaktadır.

## Buluşun çözmeyi amaçladığı teknik sorun

Buluşun amacı, bağlamı kodlanmış token temsillerinden nokta bulutu oluşturmak, nokta ve nokta çiftlerine ait geometrik özellikleri görev çıktısına taşımak ve model içi açıklama verisini aynı bulut üzerinden türetmektir. Sıralı dil bağlamı projeksiyondan önce bağlam kodlayıcı tarafından işlenir; bulut üzerinde uygulanan simetrik agregasyon ise nokta sıralamasından bağımsız bir küme temsili üretir.

## Buluşun kısa açıklaması

Sistem, girdi metnini tokenlara ayırır ve tokenların bağlamsal gösterimlerini bir kodlayıcıda işler. Projeksiyon modülü, her bir bağlamsal gösterimi D-boyutlu bir vektöre dönüştürerek nokta bulutunu oluşturur. Çift-geometri kodlayıcısı, seçilen nokta çiftleri için fark vektörü, Öklid mesafesi, açısal benzerlik ve nokta normlarını içeren özellik vektörünü hesaplar. Küme agregasyonu modülü nokta özellikleriyle çift-geometri özelliklerini simetrik biçimde birleştirir. Okuyucu, elde edilen küme gösteriminden görev çıktısı üretir. Yorumlama sistemi ise aynı nokta bulutundan nokta çıkarma etkisini veya farklı veri grupları arasındaki bölge kayması analizini hesaplayarak açıklama verisi üretir.


## Şekillerin kısa açıklaması

- Şekil 1, dil modelleme sisteminin modüllerini ve veri akışını gösterir.
- Şekil 2, girdinin üç boyutlu nokta bulutu olarak temsil edilmesine örnek verir.
- Şekil 3, tokenlar işlendiğinde nokta bulutunun oluşumunu gösterir.
- Şekil 4, farklı boyutlardaki bölge gösterimlerini örnekler.
- Şekil 5, kelime gösterimlerinin bölge dağılımını örnekler.
- Şekil 6, çift-geometri özellik vektörünün bileşenlerini gösterir.
- Şekil 7, eğitim sırasında izlenen örnek eğriyi gösterir.
- Şekil 8, morfolojik grupların geometrik gösterimdeki dağılımını örnekler.

## Buluşun detaylı açıklaması

Bir uygulama biçiminde, girdi metni tokenizör tarafından token dizisine dönüştürülür. Bağlam kodlayıcı, token dizisinin sıralı bilgisini kullanarak her token için bağlamsal gösterim üretir. Böylece token sıralaması, nokta bulutu agregasyonundan önce işlenen bağlam kodlamasında korunur.

Projeksiyon modülü, her bağlamsal gösterimi D bileşenli bir vektöre dönüştürür. Her vektör nokta bulutunda bir noktayı temsil eder. D değeri uygulama gereksinimine göre seçilebilir; açıklanan uygulamalarda 3 ile 64 arasındaki değerler ve 32 veya 64 boyutlu örnekler kullanılabilir.

Çift-geometri kodlayıcısı, nokta bulutundaki bir nokta çifti için aşağıdaki bileşenleri hesaplayabilir: fark vektörü, Öklid mesafesi, açısal benzerlik ve iki noktanın normları. Bu bileşenler birlikte D+4 bileşenli bir özellik vektörü oluşturur:

`x_ij = [p_i - p_j, ||p_i - p_j||, cos(theta_ij), ||p_i||, ||p_j||]`.

Burada `p_i` ve `p_j`, nokta bulutundaki D-boyutlu vektörleri; `i` ve `j` ise seçilen nokta çiftinin indislerini belirtir. Açısal benzerlik, nokta vektörleri arasındaki kosinüs benzerliği olarak hesaplanabilir.

Küme agregasyonu modülü, nokta özelliklerini ve çift-geometri özelliklerini simetrik birleştirme işlemlerinden geçirerek küme gösterimi oluşturur. Bir uygulamada nokta özellikleri maksimum havuzlama ile, çift-geometri özellikleri ortalama havuzlama ile birleştirilir; birleştirilmiş gösterimler okuyucuya iletilir. Alternatif bir uygulamada bağlam-ağırlıklı agregasyon kullanılabilir. Bu agregasyonun permütasyon-değişmez olması, bağlam kodlayıcının token sıralamasını önceden işlemesiyle birlikte uygulanır.

Okuyucu, küme gösterimini görev türüne uygun çıktı uzayına dönüştürür. Dil modelleme uygulamasında çıktı, bir sonraki tokena ilişkin skorlar veya olasılık dağılımı olabilir. Okuyucunun sayısal gösterimi, dağıtım gereksinimlerine bağlı olarak INT8 kantizasyonla temsil edilebilir.

Yorumlama sistemi, görev çıktısında kullanılan aynı nokta bulutunu açıklama kaynağı olarak kullanır. Nokta çıkarma analizinde bir nokta geçici olarak buluttan çıkarılır, okuyucu yeniden çalıştırılır ve görev skorundaki değişim ölçülür. Bölge kayması analizinde noktalar tanımlı geometrik bölgelere atanır; referans ve karşılaştırma veri gruplarında her bölgenin nokta payları hesaplanır ve paylardaki değişim açıklama verisi olarak raporlanır. Böylece açıklama, temsil veya görev çıktısından bağımsız ikinci bir model kurmayı gerektirmeden aynı geometrik ara gösterim üzerinden elde edilir.

Nokta sayısı, token dizisi uzunluğu ve uygulama gereksinimlerine göre belirlenebilir. Simetrik küme birleştirmesi nokta sayısından bağımsız bir işlem sırası sağlar; nokta sayısı ile D boyutu ayrı tasarım parametreleridir. Buluş, dil modelleme dışında, girdileri token benzeri birimlere ayırıp D-boyutlu vektörlerle temsil edebilen bilgisayar uygulamalarında da kullanılabilir.

## Sanayiye uygulanabilirlik

Buluş, işlemci ve belleğe sahip bilgi işlem sistemlerinde yazılım, donanım veya bunların birleşimiyle uygulanabilir. Projeksiyon, çift-geometri kodlama, küme agregasyonu, okuyucu ve yorumlama işlevleri bir veya daha fazla işlemci üzerinde yürütülebilir.
