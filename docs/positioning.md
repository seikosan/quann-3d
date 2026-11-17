# Ürün Konumlandırması

> **Omurga:** Cloud geometry, dil, işlem, sensör ve hesap verisini ortak bir
> geometrik temsil olarak ele alır. Ticari değer; denetlenebilir tahmin,
> edge boyutu ve retrain'siz uyumdur — ham doğrulukta SOTA iddiası değildir.

## Farklılaştırıcılar

### 1. Küme / temsil

Cloud geometry tek bir vektör yerine birbiriyle ilişkili noktaları ve küme
şeklini temsil eder; aynı yaklaşım metne, işlem akışına, sensör penceresine ve
hesap davranışına uygulanabilir. **Değer:** AML'de tipoloji, fraud'da işlem
örüntüsü, bakımda makine davranışı; ortak temsil, bu alanlar arasında aktarımın
kanıtlandığı iddiası değildir.

### 2. Çift kanal: mesafe + açı

Noktalar arasındaki uzaklık ile yön/açı bilgisi birlikte işlenir; böylece
yakınlık kadar ilişkinin biçimi de tahmine katılır. **Değer:** fraud'da hız ve
korelasyon örüntüleri, AML'de işlem tipolojileri, bakımda sensörler arası
değişen ilişkiler.

### 3. Açıklanabilir geometri: bölge haritaları

Tahmin, buluttaki bölgeler, kümeler ve prototiplere göre incelenebilir; uzman
kararı bir haritaya ve örneğin hangi bölgeye düştüğüne bağlanabilir. Bu,
modelin her kararı otomatik olarak doğru açıkladığı anlamına gelmez.
**Değer:** AML denetimi, fraud alarm incelemesi ve bakım mühendisine
arıza-öncülü değişimin gösterilmesi.

### 4. QUANN: backprop'suz uyum

QUANN, parametre aramasını gradyan/backpropagation olmadan yürütür; küçük ve
sınırlı modellerde yerel uyum için araştırılır. Büyük dil modeli eğitimi için
ölçeklenebilir bir yöntem olduğu gösterilmemiştir. **Değer:** tipoloji veya
makine rejimi değişirken küçük edge modellerinde retrain'siz uyum hipotezini
pilotla sınamak.

## Öncelik dikeyleri — ürün pilotları

Kaynak pazar haritası: [`market_priorities.md`](market_priorities.md).

### 1. AML / para aklama tipolojisi

- **Hedef müşteri:** Bankalar, ödeme kuruluşları ve uyum ekipleri.
- **Kazanma mantığı:** Tipolojiler işlem şekli olarak temsil edilir; uzman,
  bölge/prototip haritasını inceleyip SAR gerekçesine bağlayabilir. Yeni
  tipolojiye uyum süresini ve denetim izini ölçmek, ham skor yarışından daha
  değerli olabilir.
- **Asla iddia etmeyeceklerimiz:** Daha yüksek tespit doğruluğu veya daha az
  yanlış alarmın kanıtlandığını; bölge haritasının tek başına SAR gerekçesi ya
  da düzenleyici onay olduğunu.
- **Pilot başarı kriteri:** Zaman-ayrık veride, önceden seçilmiş tipolojiler
  için uzmanların karar/itiraz uyumu, karar başına inceleme süresi ve her
  alarmın bölge/prototip izinin yeniden üretilebilirliği birlikte ölçülür;
  eşikler pilot öncesi müşteriyle belirlenir.

### 2. İşlem fraud / kart otorizasyonu (edge)

- **Hedef müşteri:** Kart şebekeleri, acquirer'lar, e-ticaret ve neobank'ler.
- **Kazanma mantığı:** Küçük INT8 paket ve otorizasyon yolunda ölçülen gecikme;
  çift kanalın hız/korelasyon sinyalleri ve alarmın geometrik incelemesi,
  özellikle edge veya operasyonel maliyetin belirleyici olduğu dar akışlarda
  değer yaratabilir.
- **Asla iddia etmeyeceklerimiz:** XGBoost'tan daha yüksek PR-AUC ya da SOTA
  fraud doğruluğu. Mevcut eşleşmiş ölçümde QUANN strong-reader PR-AUC **0.462**,
  XGBoost **0.8047**: yaklaşık **%43 daha düşük**; bu fark açıkça raporlanır.
- **Pilot başarı kriteri:** Aynı zaman-ayrık testte mevcut incumbent ile PR-AUC,
  sabit yanlış-alarm bütçesinde yakalama, gecikme ve paket boyutu ölçülür.
  Başarı, müşteriyle önceden kararlaştırılan bir akışta kalite/maliyet
  dengesinin ve alarm inceleme faydasının operasyonel eşiği karşılamasıdır;
  mevcut ölçüm tek başına üretim kazanımı göstermez.

### 3. Endüstriyel bakım / makine sağlığı

- **Hedef müşteri:** Üretim, enerji, maden ve yiyecek-içecek tesisleri.
- **Kazanma mantığı:** Sensör penceresi bulut biçimine çevrilir; bölge kayması
  mühendise gösterilir. Tesis içi edge çalışması, bağlantı kısıtında boyut
  avantajı; erken uyarı ise önlenen plansız duruş üzerinden ölçülebilir.
- **Asla iddia etmeyeceklerimiz:** Arıza teşhisinde SOTA doğruluk, tüm makineler
  arasında genelleme veya haritanın mühendis incelemesi olmadan arıza nedeni
  kanıtladığı.
- **Pilot başarı kriteri:** Seçili makine ve arıza türünde zaman-sıralı
  doğrulama; yanlış alarm bütçesi altında uyarı öncül süresi, plansız duruş
  etkisi ve mühendislerin bölge açıklamasını doğrulama oranı ölçülür. Başarı
  eşikleri bakım ekibiyle pilot öncesi sabitlenir.

## Asla iddia etmeyeceklerimiz — genel

- LLM'lerin yerini almak veya genel amaçlı yapay zekâ olmak.
- SOTA PPL, accuracy ya da ham doğrulukta rakipleri geçmek.
- Büyük modelleri QUANN ile eğitmenin uygulanabilir olduğu: 111K parametrede
  QUANN PPL'i backprop'ın yaklaşık **1.8×**'i ve random'dan yalnızca **%3**
  daha iyidir; **21M** parametrede QUANN araması infeasible olarak
  raporlanmıştır.
- Bölge haritalarının nedensellik, eksiksiz açıklama veya düzenleyici uyum
  garantisi verdiği.

> Pilot metrikleri ve eşikleri başlamadan önce müşteriyle sabitlenir; her sonuç
> aynı veri bölmesi, incumbent ve operasyonel koşullarla raporlanır.
