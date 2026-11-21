# Teknik etki

## Teknik problem

Dil modeli çıktısını ve model içi açıklama verisini aynı bilgisayar iş akışında üretirken, açıklama kaynağının görev çıktısında kullanılan ara gösterimle izlenebilir biçimde ilişkilendirilmesi gerekir.

## Teknik çözüm

Bağlam kodlayıcı token sıralı bağlamını işler. Projeksiyon modülü bağlamı kodlanmış token gösterimlerini D-boyutlu vektörlere çevirir. Çift-geometri kodlayıcısı nokta çiftlerinin fark, mesafe, açısal benzerlik ve norm bileşenlerini üretir. Küme agregasyonu ve okuyucu görev çıktısını oluştururken yorumlama sistemi aynı nokta bulutunu kullanarak nokta çıkarma etkisini veya geometrik bölge paylarındaki değişimi hesaplar.

## Teknik etki

- Görev çıktısı, nokta ve nokta çifti özelliklerinden oluşturulan hesaplanabilir bir küme gösterimine bağlanır.
- Nokta çıkarma analizinde, aynı okuyucunun nokta çıkarılmadan önceki ve sonraki görev çıktıları karşılaştırılabilir.
- Bölge kayması analizinde, referans ve karşılaştırma gruplarının bölge başına nokta payları aynı geometrik gösterim üzerinden hesaplanabilir.
- Okuyucuda INT8 kantizasyon uygulanması, kantize edilmiş parametre gösteriminin kullanılmasına imkân verir; belirli bir bellek veya hız kazancı bu pakette ölçülmüş teknik etki olarak ileri sürülmemektedir.

Bu açıklama, teknik işleyişi tanımlar; yenilik, buluş basamağı veya patentlenebilirlik hakkında hukuki hüküm oluşturmaz.
