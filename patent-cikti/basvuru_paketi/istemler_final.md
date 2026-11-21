# İSTEMLER

1. Bilgisayarla uygulanan bir dil modelleme yöntemi olup, aşağıdaki adımları içermesiyle karakterize edilir: girdi metnindeki tokenların sıralı bağlamını bir bağlam kodlayıcı ile işlemek; bağlamı kodlanmış her bir token gösterimini D-boyutlu bir vektöre projekte ederek bir nokta bulutu oluşturmak; nokta bulutundaki nokta özelliklerini ve nokta çiftlerine ait mesafe, açısal benzerlik ve norm özelliklerini hesaplamak; nokta özellikleri ile nokta çiftlerine ait özellikleri simetrik bir küme agregasyonu ile bir küme gösteriminde birleştirmek; küme gösteriminden bir okuyucu aracılığıyla görev çıktısı üretmek; ve aynı nokta bulutundan nokta çıkarma etkisini veya veri grupları arasındaki bölge kayması analizini yürüterek açıklama verisi üretmek.

2. İstem 1'e göre yöntem olup, D boyutunun 3 ile 64 arasında seçilmesiyle karakterize edilir.

3. İstem 1'e göre yöntem olup, bir nokta çifti için fark vektörü, Öklid mesafesi, açısal benzerlik ve çiftteki her bir noktanın normunu içeren D+4 bileşenli bir özellik vektörünün hesaplanmasıyla karakterize edilir.

4. İstem 1'e göre yöntem olup, nokta özelliklerinin maksimum havuzlama ve nokta çiftlerine ait özelliklerin ortalama havuzlama ile birleştirilmesiyle karakterize edilir.

5. İstem 1'e göre yöntem olup, okuyucunun en az bir parametresinin INT8 kantizasyonla temsil edilmesiyle karakterize edilir.

6. İstem 1'e göre yöntem olup, nokta çıkarma etkisinin belirlenmesi için bir noktanın nokta bulutundan çıkarılması, okuyucunun yeniden çalıştırılması ve görev çıktısındaki değişimin ölçülmesiyle karakterize edilir.

7. En az bir işlemci ve işlemci tarafından yürütülebilir komutları saklayan bir bellek içeren bilgisayar sistemi olup, komutların yürütülmesi sırasında: tokenların sıralı bağlamını işleyen bir bağlam kodlayıcı; bağlamı kodlanmış token gösterimlerini D-boyutlu vektörlere dönüştürerek nokta bulutu oluşturan bir projeksiyon modülü; nokta çiftlerine ait geometrik özellikleri hesaplayan bir çift-geometri kodlayıcısı; nokta ve çift özelliklerini küme gösteriminde birleştiren bir küme agregasyonu modülü; küme gösteriminden görev çıktısı üreten bir okuyucu; ve aynı nokta bulutundan nokta çıkarma etkisi veya bölge kayması analizi hesaplayan bir yorumlama sistemi oluşturulacak şekilde yapılandırılmasıyla karakterize edilir.

8. İstem 7'ye göre bilgisayar sistemi olup, İstem 7'de tanımlanan projeksiyon modülünün D-boyutlu vektörlerin boyutunu 3 ile 64 arasında seçmesiyle karakterize edilir.

9. İstem 7'ye göre bilgisayar sistemi olup, çift-geometri kodlayıcısının bir nokta çifti için fark vektörü, Öklid mesafesi, açısal benzerlik ve iki noktanın normlarını içeren D+4 bileşenli özellik vektörünü hesaplamasıyla karakterize edilir.

10. İstem 7'ye göre bilgisayar sistemi olup, küme agregasyonu modülünün nokta özelliklerini maksimum havuzlama ve nokta çiftlerine ait özellikleri ortalama havuzlama ile birleştirmesiyle karakterize edilir.

11. İstem 7'ye göre bilgisayar sistemi olup, okuyucunun en az bir parametresinin INT8 kantizasyonla temsil edilmesiyle karakterize edilir.

12. İstem 7'ye göre bilgisayar sistemi olup, yorumlama sisteminin bir noktayı geçici olarak nokta bulutundan çıkarıp okuyucuyu yeniden çalıştırarak görev çıktısındaki değişimi hesaplaması veya referans ve karşılaştırma veri gruplarındaki geometrik bölge nokta paylarını karşılaştırarak bölge kayması analizi yürütmesiyle karakterize edilir.

13. Bilgisayarca okunabilir, geçici olmayan bir kayıt ortamı olup, üzerinde saklanan komutların bir işlemci tarafından yürütülmesi sırasında: girdi metnindeki tokenların sıralı bağlamını kodlamak; bağlamı kodlanmış token gösterimlerini D-boyutlu vektörlere projekte ederek bir nokta bulutu oluşturmak; nokta özellikleri ile nokta çiftlerine ait mesafe, açısal benzerlik ve norm özelliklerini hesaplamak; bu özellikleri simetrik küme agregasyonu ile bir küme gösteriminde birleştirmek; küme gösteriminden görev çıktısı üretmek; ve aynı nokta bulutundan nokta çıkarma etkisini veya veri grupları arasındaki bölge kayması analizini hesaplayarak açıklama verisi üretmek adımlarını gerçekleştirmesiyle karakterize edilir.
