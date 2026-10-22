# Pilot Plan — Edge fraud score with transaction explanation

## Tez ve sınır

Bu pilotun iddiası **“daha doğru” değildir**. Test edilecek değer önerisi:
“aynı sınıf doğruluk hedefi içinde 1/40 boyut + işlem başına explanation +
INT8 edge.” Mevcut kanıt doğruluk eşitliğini göstermiyor; matched XGBoost PR-AUC
0.8047, paket hibrit INT8 PR-AUC 0.2573. Bu fark açıktır ve kapanmasına bağlı
değiliz. Boyut karşılaştırması müşteri incumbent'ının gerçek deploy artifact'ı
ile sahada doğrulanır.

## A/B tasarımı

1. Mevcut rule/XGB karar sistemi **A**, pilot model **B** olur. İlk aşamada aynı
   auth olaylarını paralel gölge-modda skorla; B authorization sonucunu,
   müşteri yanıtını veya müşteri görünümünü değiştirmez.
2. Aynı event ID, özellik anı ve zaman damgasını iki sisteme kaydet. PII yerine
   müşteri onaylı takma kimlik kullan; explanation ve eşik değişikliklerini
   denetlenebilir şekilde sakla. Mevcut sistemin çıktısı hiçbir zaman B ile
   overwrite edilmez.
3. Skor, explanation ve pipeline overhead'ini edge hedef cihazda ölç. Önerilen
   pilot bütçesi: uçtan uca p95 ≤10 ms/işlem, p99 ≤25 ms; müşteri auth SLA'sı
   daha sıkıysa o sınır geçerlidir. Skor-only 45–54k tx/s laptop ölçümü bu
   explanation dahil SLA'yı kanıtlamaz.
4. Eşik .9969'u başlangıç referansı olarak kaydet. A/B karşılaştırmasında
   precision, recall, PR-AUC, ROC-AUC, FPR, kaçan fraud tutarı, yanlış-red sayısı,
   explanation coverage ve latency'yi aynı zaman penceresi ve etiket olgunlaşma
   süresiyle hesapla. B eşiğini aynı etiketli doğrulama dönemi üzerinde seç; test
   döneminde yeniden ayarlama yapma.

## Yanlış-red maliyeti ve karar çerçevesi

Gölge-modda yanlış-red (normal işlemi reddetme) sayısı ve müşteri tarafından
verilen işlem başına kayıp `C_FR` ölçülür; aylık tahmini maliyet:

```text
yanlış-red maliyeti = normal işlem hacmi × normal işlemlerde red oranı × C_FR
kaçan fraud maliyeti = fraud adedi × kaçırılan fraud oranı × ortalama net fraud kaybı
```

Müşteri mevcut sistemin ek inceleme, challenge ve decline maliyetlerini de
aynı tabloya koyar. .9969 puan eşiği tek başına bir yanlış-red oranı değildir;
oran etiketli müşteri verisi ve gerçek politika akışı üzerinde ölçülür. Pilot
boyunca B yalnızca alarm/explanation üretir; otomatik red için yetki vermez.

## Başarı kriteri

**Başarı:** müşteri authorization yolunu edge cihazında B'nin explanation'ı ile
uçtan uca çalıştırır ve uyum ekibi explanation biçimini/kanıt izini kabul eder;
aynı zamanda p95/p99 SLA'sı ve müşteri tarafından tanımlanan yanlış-red maliyet
kapısı geçilir. Bu, doğrulukta incumbent'ı geçtiğimiz iddiası değildir.

Ölçülebilir kapılar pilot öncesi müşteriyle imzalanır: yeterli etiket olgunluğu,
kararlaştırılmış fraud/normal kapsaması, tam explanation ve audit kaydı, latency
bütçesi, ve yanlış-red maliyet tavanı. Sağlam bir matched karşılaştırma için
aynı transaction kohortu, aynı etiket penceresi, aynı cost-sensitive threshold
seçim protokolü kullanılır.

## Exit / durdurma kriterleri

- **İlerlet:** compliance kabulü + edge/SLA + maliyet kapısı birlikte geçer;
  shadow sonuçları müşteri mevcut akışına göre anlamlı bir operasyonel kullanım
  gösterir. Bir sonraki aşama insan onaylı review/challenge denemesidir.
- **Düzeltip tekrarla:** açıklama anlaşılıyor ancak kategori eşleşmesi, veri
  kayması veya latency sorunu giderilebilir ve modelin incumbent yanında
  güvenli bir review rolü kalır.
- **Durdur:** compliance explanation'ı reddeder; gerekli kolon/semantik yoktur;
  latency/maliyet kapısı kalıcı olarak aşılır; veya olgunlaşmış etiketlerde B,
  hiçbir onaylı review kullanımında kabul edilebilir fayda göstermeden ek yanlış
  red/maliyet üretir. Olay oranı düşükse yetersiz örneklemi başarı sayma; pilotu
  uzat veya sonuçsuz kapat.

## Güvenli kapsam

Pilot CSV istemcisi `fraud/pilot/infer.py`; model paketi `fraud/results/deploy/`.
İlk sürüm yalnızca gölge skor ve açıklama içindir. Müşteri auth sistemi,
onaylı eşik, gecikme ölçümü ve compliance kabulü olmadan karar entegrasyonu
yapılmaz.
