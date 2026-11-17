# VEKİLE SORU–CEVAP — Kısa Yanıtlar

Beş sorunun birkaç cümlelik yanıtları. Her yanıtın ayrıntılı kod kanıtı
[`VEKIL_SORU_CEVAP.md`](VEKIL_SORU_CEVAP.md) dosyasında.

## Soru 1 — Transformer ve QUANN birlikte mi?

Hayır, birlikte değil. Dil modelinde standart Transformer katmanı var ve eğitim
geri-yayılımla yapılıyor; QUANN'ın geri-yayılımsız güncellemesi yalnız
tabular/fraud tarafında çalışıyor. İstemde Transformer "gereksiz" diye
yazılmamalı; alternatif bir temsil katmanı ancak eşdeğerlik cümlesi olarak geçebilir.

## Soru 2 — Birim noktaya nasıl dönüşüyor; 20 nokta; sıra

Her birim gömme alıyor, Transformer'dan geçiyor, iki katmanlı küçük bir ağla
D boyutlu noktaya indirgeniyor. Nokta sayısı sabit değil, girdi birimi sayısına
eşit; "20 nokta" kodda doğrulanmadı, metinden çıkarılmalı. Sıra bilgisi şu an
zorunlu ekleniyor ve kapatma bayrağı yok — "pozisyon zorunlu değil" cümlesi kodla
çelişiyor.

## Soru 3 — Tüm çiftler mi, ağlar, birleştirme

Nokta sayısının karesi kadar tüm çiftler kullanılıyor, alt seçim yok. Bir nokta
ağı ve bir çift ağı var; çıktıları birleştiriliyor, özetleme toplam/ortalama ile
yapıldığı için nokta sırası sonucu değiştirmiyor. "Açı" terimi geçerse bu iki
vektör arasındaki açının kosinüsüdür; açının kendisi hesaplanmıyor.

## Soru 4 — Aralık daralıyor mu, genişliyor mu

Kodda üç ayrı güncelleme kuralı var. Motor varyantı kabulde aralığı
değiştirmez, yalnız adım ölçeği (sigma) küçülür; classic'te kabulde aralık
genişler; hybrid'de kabul, kabul edilen segmente daralır. Bu yüzden "hata azalınca
aralık daralır" tek cümle olarak yazılamaz; istem varyant adı vererek yazılmalı ve
önerilen hybrid olmalı (AML ve fraud deneyleri onu kullanıyor). Sigma ile arama
aralığı metinde iki ayrı kavram olarak tanımlanmalı.

## Soru 5 — Nokta çıkarılınca ne oluyor; prototip, isim, eşik

Nokta temsil uzayından çıkarılıyor: gömme yeniden hesaplanmıyor, kodlayıcı +
özetleme + başlık yeniden koşuyor ve logit farkı ölçülüyor. Prototipler eğitim
verisinin sınıf merkezleri (veya KMeans merkezleri), atama en yakın merkeze göre
yapılıyor. Ayrı bir belirsizlik skoru ve merkez kayması ölçüsü kodda yok, istem
olarak yazılmamalı; drift olarak ölçülen şey bölge payları dağılımının değişimi
(TV/KL) veya prototipe uzaklığın eşik aşımıdır.

## Vekilin dikkat etmesi gereken üç nokta

1. Şekillerdeki sayılar (-2.34, 0.42, %78 gibi) ölçüm değil elle yazılmış
   şematik örnektir; ölçülmüş sonuç gibi sunulmamalı.
2. Dil ucunda açıklama kodu yok; native açıklanabilirlik iddiası ya dil ucunda
   kod ve ölçümle desteklenir ya evrensel dosyanın çekirdeğine bırakılır.
3. Sabit "20 nokta" ve "pozisyon zorunlu değil" ifadeleri metinden çıkarılmalı.
