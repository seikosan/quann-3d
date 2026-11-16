# PATENT BAŞVURU DOSYASI — Tabular Kimlik-Bulut Yorumlama (v1, Yeni Buluş)

> **Kapsam notu:** Bu başvuru, dil modelleme buluşunun (`../dil/patent_application_v2.3.md`,
> cümle → nokta-bulut → bir sonraki token) **ayrık** bir uygulamasıdır. Buluş, aynı
> nokta-bulut (CloudEncoder) çekirdeğini **tabloları/kayıtları** (ör. kredi kartı
> işlemleri) hedef alan, **kimlik-tabanlı özellik noktaları + native set-ablation
> düz-metin açıklama + karşı-faktürel hatalı-uyarı denetimi + self-contained kenar
> dağıtımı** bileşenleriyle birleştiren bağımsız bir sistem/yöntem olarak sunulur.
> Bulut kodlayıcı + 0. sıra daraltma çekirdeği, bitişik (bilinen) buluş olarak
> referans verilir; burada yeni olan **tablolar arası kimlik-bucket projeksiyonu,
> çözümlenen ablation açıklaması, karşı-faktürel denetim ve self-contained paket**tir.

> **Kapsam notu (2026-10-22):** Bu dosya, `geometry_dual_use_core_v1.md`'deki evrensel çekirdeğin **tabular/fraud** uygulamasıdır. Bağımsız İstem 1'in (a)-(c) çekirdek paragrafları genel dosyaya çapraz-referans taşır; bu dosyaya özgü yeni olan kimlik-bucket projeksiyon + karşı-faktürel hatalı-uyarı denetimi + self-contained edge embodiment'idir.

---

## 1. BAŞLIK

**Tabular kayıtlar için kimlik-tabanlı nokta-bulutu ile native set-ablation açıklama,
karşı-faktürel hatalı-uyarı denetimi ve self-contained kenar dağıtımı gerçekleştiren
sistem ve yöntemi**

---

## 2. AÇIKLAMA

### 2.1 Teknik Alan

Bu buluş, makine öğrenmesi alanına; özellikle **tablolar veri** (sayısal + kategorik
özellik satırları) üzerindeki **aykırı değer / dolandırıcılık (fraud) tespiti** ve
**yorumlanabilir makine öğrenmesi (explainable ML)** alanlarına ilişkindir.
Özellikle, bir kararın **kayıt düzeyinde düz-metin olarak açıklanabilmesini**,
**popülasyon düzeyinde hatalı-uyarı (false-alarm) denetiminin** otomatik yapılmasını
ve tümünün **tek bir self-contained, kenar-cihaza taşınabilir artifact** olarak
dağıtılmasını sağlayan bir sistem ve yöntemi ileri sürer.

### 2.2 Önceki Teknik (Prior Art)

#### 2.2.1 Baskın Paradigma: Dereceli Ağaç Aileleri + SHAP

Üretimdeki standart dolandırıcılık/aykırı-değer dedektörleri, dereceli ağaç
ensamblesıdır (Gradient Boosting, XGBoost, LightGBM), ayrık orman (isolation
forest), otocodek (autoencoder) veya dönüşümlü sinir ağlarıdır. Bu dedektörler bir
risk skoru üretir ve uygun araçlarla **özellik düzeyinde** atfı (GBM için SHAP
katkıları) sunar: *hangi özellik* satırı riskli kıldı.

Bu yaklaşımın sınırlamaları:

1. **Özellik düzeyi, kayıt düzeyi değil:** SHAP bir *lineer katkı toplamı* varsayar
   (SHAP'ın additivity aksiyomu). Nokta-bulut çift-geometri gibi çiftleşen
   (non-additive) temsillerde, tek-özellik ablationları toplam skora eşitlenmez;
   "hangi özellik" sorusu yanıltıcıdır.
2. **Düz-metin değil:** SHAP çıktısı özellik-adi + sayısal ağırlıktır; bir
   incelenen (analist/denetçi) tarafından doğrudan okunabilen bir *cümle*
   değildir.
3. **Popülasyon denetimi yoktur:** SHAP tek-kayıt atfıdır; *hangi genel token*
   hatalı-uvari popülasyonunu sürüyor (ör. tüm çevrimiçi alımlar sahtekarca
   okunuyor mu) sorusunu yanıtlamaz.
4. **Self-contained değil, kenara taşınmaz:** Bir GBM + SHAP, model + özellik
   sözlüğü + atfı koduyla birlikte, büyük ve ortam-bağımlı bir yığınıdır; tek
   dosya, bit-kesin, kenar-cihazda düz-metin üreten bir artifact değildir.

#### 2.2.2 İlgili Çalışmalar

- **Nokta-bulut / set tabanlı tablolar temsilleri** (FT-Transformer, TabTransformer,
  set-encoding ailesi): Tablolar satırları özellik-vektörleri olarak işler;
  sıralamasız kümeyi tek vektöre havuzlar. Noktaların **uzaysal konumu girdi
  olarak var değildir** — geometri korunmaz, dolayısıyla nokta-düzeyinde
  ablation açıklaması tanımlanamaz.
- **QUANN nokta-bulut çekirdeği** (bitişik buluş, `../dil/patent_application_v2.3.md`):
  token'ları düşük-boyutlu noktalara projekte edip bulut geometrisinden karar
  alan ve 0. sıra (backprop'suz) daraltma ile eğiten çekirdek. Bu çekirdek burada
  **bilinen** olarak referans verilir; yeni olan onun **tablolar kaydına**
  (kimlik-bucket) uygulanması ve **açıklama/denetim/dağıtım** bileşenleridir.
- **0. sıra (backprop-free) optimizasyon:** Fonksiyon değerlerinden (gradyansız)
  ağırlık/nokta güncelleyen yöntemler; buluşta nokta daraltması için tercihen
  kullanılır (okuyucu yine de backprop ile eğitilebilir — hibrit rejim).

#### 2.2.3 Önceki Teknikteki Boşluk

Hiçbir mevcut yöntem, aşağıdaki dört bileşeni **tek bir, self-contained** sistemde
birleştirmemektedir: (a) tablolar kaydın özelliklerinin **kimlik-tabanlı
(özellik, değer-bucket) noktalarına** projekte edilerek bir nokta-bulutu
oluşturulması; (b) bu buluttan **native leave-one-point-out logit-ablation** ile
üretilip **tam-sözlük bir decoder** üzerinden **düz-metin**e çözümlenen bir
açıklama; (c) tek-ve çift-maske **karşı-faktürel** testlerle yapılan
popülasyon-düzeyinde hatalı-uyarı denetimi; ve (d) bunların tamamını,
bit-kesinlik doğrulaması ve provenans (sha256) kaydıyla birlikte **tek artifact**
olarak kenara taşıyan bir dağıtım paketi.

### 2.3 Teknik Sorun

Buluş, aşağıdaki teknik sorunları çözmeyi amaçlar:

1. Bir kaydın skoru, **özellik-ağırlığı listesi** yerine, **kayıt düzeyinde ve
   doğrudan okunabilen düz-metin** (ör. "amount>100, MCC=5311, Merchant
   State=ONLINE") olarak açıklanmak;
2. Çiftleşen (non-additive) temsillerde, tek-özellik ablationlarının toplam
   skoru yeniden üretmemesi sorunu, **karşı-faktürel** (token yoksa kayıt
   bayraklanır mı) testlerle dürüst biçimde ele alınmak;
3. Hatalı-uyarı **popülasyonunun** sürücü token'larının (genel vs nadir) otomatik
   belirlenmesi;
4. Modelin, **sub-megabyte**, self-contained, kenar-cihaza taşınabilir ve
   düz-metin üreten tek bir artifact olarak dağıtılması.

### 2.4 Buluşun Kısa Açıklaması (Çözüm)

İcadın özü: bir tablolar kaydın her özelliği, ölçülen değer-bucket'ına göre
bir **kimlik token'ına** `(özellik, değer-bucket)` dönüştürülür; her kimlik
token'ı, düşük boyutlu (3 ≤ D ≤ 64) öklid uzayda bir **noktaya** projekte edilir;
kayıt, bu noktaların **permütasyon-değişmez bulutu** olur. Bulut, bilinen
CloudEncoder (nokta MLP + max-pool, çift geometrisi [fark, mesafe, kosinüs-
açısı] + mean-pool) ile bir küme durumuna indirgenir ve küçük bir okuyucu P(
aykırı) üretir.

Yeni üç bileşen:

- **Açıklama modülü (260):** Kaydın skoru, buluttan **tek nokta çıkarıldığında**
  logit'in değişimi olarak (leave-one-point-out ablation) ölçülür; en büyük
  katkılı noktalar **tam-sözlük token decoder (270)** üzerinden düz-metne
  çözümlenir. Bu, SHAP'tan farklıdır: çift-geometri terimi katkıları
  **non-additive** kılar, dolayısıyla bir *karşı-faktürel duyarlılık*, lineer
  katkı toplamı olarak sunulmaz.
- **Karşı-faktürel denetim modülü (280):** Bir token'ın (tek-maske) veya bir
  token çiftinin (çift-maske) **yok** sayılması durumunda kaydın eşik üstünde
  kalıp kalmadığı sorulur. Bu, "kararı tek başına o token veriyor mu / başka
  hangi nokta ile birlikte veriyor" sorusunu yanıtlar; hatalı-uyarı popülasyonunun
  genel vs nadir token'lara ayrışmasını ölçer.
- **Dağıtım paketleyici (290):** Model, `model.onnx` + `quantizer.json` (ham
  satır → token id) + `token_decoder.json` (id → düz-metin etiket) + `manifest.json`
  (kaynak checkpoint sha256, her artifact'ın sha256/boyutu, yapılandırma, self-test)
  olarak tek bir self-contained paket olur; ham satırlarda torch↔ONNX bit-kesinlik
  (max |Δp|) ve kenar gecikmesi doğrulanır.

### 2.5 Teknik Etkiler

1. **Kayıt-düzeyi düz-metin açıklama (birincil etki):** Bir aykırı kaydın skoru,
   incelenenin doğrudan okuyabileceği özellik-cümlelerine (ör. "amount>100;
   MCC=5311; Merchant State=ONLINE") çözümlenir; SHAP'taki sayısal ağırlık
   listesi yerine *okunabilir bir ifade* elde edilir.
2. **Dürüst non-additive atıf:** Katkı, leave-one-point-out logit-değişimi +
   karşı-faktürel maske olarak verilir; toplam skoru yeniden üretmeyen tek-
   özellik değerleri, "karar bu bulut konfigürasyonundan doğdu" biçiminde
   dürüstçe raporlanır.
3. **Popülasyon denetimi:** Tek/çift-maske karşı-faktürelleri, hatalı-uyarı
   popülasyonunu sürüren *genel* token'ları (vs nadir, satır-başına güçlü token'lar)
   otomatik ayırır — modelin "çevrimiçi alışveriş mi?" sorusuna sahtekarca
   yanıt verip vermediğini ölçülebilir kılar.
4. **Self-contained kenar dağıtımı:** Tam sistem (okuyucu + girdi-kantizörü +
   decoder + manifest) tek pakettir; ham satırlarda bit-kesin, düz-metin üreten,
   sub-megabyte bir artifact olarak kenar-cihaza taşınır.
5. **Tercihi 0. sıra geometri:** Nokta daraltması 0. sıra (backprop'suz) yapılabilir;
   okuyucu backprop ile eğitilerek hibrit rejim elde edilir (saf 0. sıra gerçek
   görevlerde rastgele yakın kalır — buluş hibrit rejimi iddia eder).

### 2.6 Şekillerin Kısa Açıklaması

- **Şekil 1:** Sistem mimarisi — 200: girdi kaydı, 210: özellik-kantizasyon,
  220: kimlik-nokta projeksiyonu, 230: bulut kodlayıcı (231: nokta MLP, 232:
  çift MLP, 233: birleştirme), 240: 0. sıra nokta daraltma, 250: okuyucu, 260:
  ablation açıklama, 270: tam-sözlük decoder, 280: karşı-faktürel denetim,
  290: dağıtım paketleyici.
- **Şekil 2:** Kimlik-bucket projeksiyonu — bir kayıdın `(özellik,bucket)`
  token'larının buluttaki noktaları.
- **Şekil 3:** Leave-one-point-out ablation — bir nokta çıkarıldığında logit
  değişimi.
- **Şekil 4:** Karşı-faktürel maske (tek vs çift token) — eşik üstü/kalığı.
- **Şekil 5:** Self-contained dağıtım paketi (onnx + quantizer + decoder +
  manifest) ve ham-satır bit-kesinlik kontrolü.

### 2.7 Buluşun Detaylı Açıklaması

#### 2.7.1 Girdi Kaydı (200)

Bir kayıt, N özelliğin bir satırıdır (ör. kredi kartı işlemi: Amount, Use Chip,
Merchant Name/City/State, Zip, MCC, Errors, Is Fraud). Sayısal özellikler,
kategorik özellikler ve etiket olarak ayrılır.

#### 2.7.2 Özellik-Kantizasyon Modülü (210)

Her sayısal özellik, eğitimde (leak-safe) belirlenen bucket'lara bölünür; her
kategorik özellik, gözlenen değerlerine göre bucket'lanır. Sonuç, her özellik için
bir **kimlik token'ı** `(özellik, değer-bucket)`'tır. Bir kayıt, K özelliği için K
kimlik token'ı üretir. Kantizör, eğitim veri setine `fit` edilir ve test
veri setine uygulanır; checkpoint içine gömülür (bit-kesin AUC için).

#### 2.7.3 Kimlik-Nokta Projeksiyonu (220)

Her kimlik token'ı, 3 ≤ D ≤ 64 boyutlu uzayda bir noktaya projekte edilir (tercihe
göre 0. sıra daraltma ile, İstem 12). D=8 güçlü-okuyucu rejimi, d_cloud=128,
d_model=32, 214 yığılmış nokta örneklemi kullanır.

#### 2.7.4 Bulut Kodlayıcı (230)

Bilinen CloudEncoder: (231) her nokta için paylaşılan nokta MLP + max-pool;
(232) her nokta çifti için `[pᵢ−pⱼ, ‖pᵢ−pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖]` girdili çift
MLP + mean-pool; (233) ikisinin birleştirilmesiyle permütasyon-değişmez küme
durumu. Çift-geometri terimi, katkılarının **non-additive** olmasının kaynağıdır
(İstem 13).

#### 2.7.5 0. Sıra Nokta Daraltma (240) — tercihe bağlı

Nokta konumları, gradyansız (0. sıra) QUANN daraltmasıyla öğrenilebilir; okuyucu
(250) yine backprop ile eğitilir. Saf 0. sıra rejim gerçek görevlerde rastgele
yakın kalır; icadın iddia ettiği **hibrit** rejim, nokta daraltması 0. sıra +
okuyucu backprop'tür.

#### 2.7.6 Okuyucu (250)

Küçük bir okuyucu (MLP), küme durumundan P(aykırı) üretir; tercihen INT8'e
kantizlenir (529 KB FP32 → 138 KB INT8).

#### 2.7.7 Açıklama Modülü (260) + Tam-Sözlük Decoder (270)

Her nokta için, nokta buluttan çıkarıldığında logit değişimi
(leave-one-point-out) hesaplanır. En büyük |değişim|li noktalar, **tam-sözlük**
token decoder (270) — 221 id = tam özellik sözlüğü; her bucket/değer bir ad alır,
"diğer" bucket'ı atılmaz — üzerinden düz-metne çözümlenir. Çıktı, bir özellik-
cümlesi listesidir (ör. "amount>100; MCC=5311; Merchant State=ONLINE").

#### 2.7.8 Karşı-Faktürel Denetim Modülü (280)

- **Tek-maske:** Token T yok sayılırsa kayıt hâlâ eşik üstünde mi? (popülasyonun
  kaçında "evet" → T karar-tek-başına mı).
- **Çift-maske:** (T, U) birlikte yok sayılırsa? Tek başına U etkisiz, birlikte
  etkili ise → karar bulut-konfigürasyonuna (non-additive) bağlıdır.
Bu, hatalı-uyarı popülasyonunun **genel** (büyük pay) vs **nadir** (satır-başına
güçlü) token'lara ayrışmasını ölçer.

#### 2.7.9 Dağıtım Paketleyici (290)

Paket = `model.onnx` (dinamik batch) + `quantizer.json` + `token_decoder.json` +
`manifest.json`. Manifest: kaynak checkpoint sha256, her artifact'ın sha256/boyutu,
model yapılandırması, self-test sonucu. Ham satırlarda torch↔ONNXRuntime bit-
kesinlik (max |Δp| 1.5e-6) ve kenar gecikmesi (~45.5k kayıt/sn) doğrulanır.
Girdi sözleşmesi: `ids (B,K) int64` + `mask (B,K) bool` → `p (B) = P(aykırı)`.

---

## 3. İSTEMLER

### Bağımsız İstemler

**İstem 1**

Ön teknikte bilinen, bir tablolar kaydın özelliklerini işleyerek bir risk skoru
üreten bir makine öğrenmesi dedektörünü içeren bir sistem,

**karakterize edici özellik olarak**

(a) adı geçen kaydın her sayısal ve/veya kategorik özelliğini, bir değer-bucket'ına
işleyerek bir `(özellik, değer-bucket)` kimlik token'ı üreten bir özellik-kantizasyon
modülü (210);

(b) her kimlik token'ını, 3 ≤ D ≤ 64 boyutlu öklid uzayda bir noktaya
projekte eden ve bu noktaları bir permütasyon-değişmez bulut olarak
birleştiren bir kimlik-nokta modülü (220);

(c) bu buluttan bir küme durumu üreten ve bu küme durumundan bir
skor veren bir okuyucu (250);

(d) bu buluttan, **tek nokta çıkarıldığında** skorun logit-değişimini ölçen
ve bu noktaları bir token decoder (270) üzerinden **düz-metin etiketlere**
çözen bir açıklama modülü (260);

(e) işlenen kayıtların bir alt-kümesi üzerinde, bir veya birden fazla token'ın
**yok** sayılması durumunda skorun eşik üstünde kalıp kalmadığını test eden bir
karşı-faktürel denetim modülü (280); ve

(f) bu okuyucuyu, özellik-kantizasyon modülünü, token decoder'ı ve her
artifact'ın kimlik/kod-bütünlüğü kaydını birlikte içeren **tek bir self-contained
dağıtım paketi** (290) olarak derleyen bir modül

içermekte olup, adı geçen sistem, bir kaydın skorunu bir **okunabilir
özellik-ifadesi** olarak, popülasyon düzeyinde bir **karşı-faktürel** ölçümle ve
kenara taşınabilir **tek artifact** olarak birlikte sunmaktadır.

**İstem 2**

Ön teknikte bilinen, tablolar kayıtlardan risk skoru üreten bir yöntem,

**karakterize edici özellik olarak**

(a) her kaydın özelliklerini `(özellik, değer-bucket)` kimlik token'larına
dönüştürme;

(b) her kimlik token'ını D boyutlu uzayda bir noktaya projekte edip bulut
oluşturma;

(c) buluttan permütasyon-değişmez bir küme durumu üreterek skor tahmini;

(d) skorun logit-değişimini, buluttan tek nokta çıkarılarak ölçme ve en büyük
katkılı noktaları bir decoder üzerinden düz-metne çözme;

(e) bir token veya token çiftinin yok sayılması durumunda skorun eşik üstünde
kalıp kalmadığını test etme

adımlarını içermekte olup, adı geçen (d) ve (e) adımları, çiftleşen nokta-geometri
temsilinde katkıların non-additive olduğunu **karşı-faktürel** olarak raporlayarak,
özellik-ağırlığı listesi yerine kayıt-düzeyi okunabilir bir açıklama
sağlamaktadır.

**İstem 3**

Bir bilgisayar programı, bir işlemci tarafından çalıştırıldığında İstem 2'deki
yöntemin adımlarını gerçekleştiren program talimatlarını içermekte olup, adı
geçen program talimatları, tablolar kayıt açıklamasını düz-metin olarak ve
self-contained bir artifact olarak üretmektedir.

### Bağımlı İstemler

**İstem 4:** İstem 1'e bağımlı, burada çift-geometri girdi vektörü, her nokta
çifti için `[pᵢ − pⱼ, ‖pᵢ − pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖]` biçimindedir ve
döndürmeye karşı değişmezdir.

**İstem 5:** İstem 1'e bağımlı, burada okuyucu (250), küçük bir MLP'dir ve INT8'a kantizlenmiştir.

**İstem 6:** İstem 1'e bağımlı, burada token decoder (270), **tam-sözlük**tür: her
`(özellik, değer-bucket)` kimliği bir insan-okunur etikete eşlenir ve hiçbir
bucket "diğer" olarak atılmaz.

**İstem 7:** İstem 1'e bağımlı, burada açıklama modülü (260), leave-one-point-out
logit-değişimini, tek-özellik değerlerinin toplam skoru yeniden üretmediği
(non-additive) durumda bir *karşı-faktürel duyarlılık* olarak raporlar.

**İstem 8:** İstem 1'e bağımlı, burada karşı-faktürel denetim modülü (280), hem
tek-maske hem de çift-maske (iki token birlikte) testlerini çalıştırır.

**İstem 9:** İstem 8'e bağımlı, burada bir token'ın tek başına kararı
etkilemediği, ancak ikinci bir token ile birlikte etkilediği durumlar
belirlenerek kararın bulut-konfigürasyonuna bağımlılığı (non-additivity)
kanıtlanır.

**İstem 10:** İstem 1'e bağımlı, burada karşı-faktürel denetim modülü (280),
eşik-üstü hatalı-uyarı popülasyonunu, **genel** (büyük popülasyon-payı) token'larla
**nadir** (satır-başına büyük katkı) token'lara ayırır.

**İstem 11:** İstem 1'e bağımlı, burada kimlik-nokta modülü (220), nokta
konumlarını 0. sıra (gradyansız) bir daraltmayla üretmekte olup, okuyucu (250)
backprop ile eğitilerek hibrit bir rejim oluşturulur.

**İstem 12:** İstem 11'e bağımlı, burada D = 8, d_cloud = 128, d_model = 32 ve 214
yığılmış nokta kullanılmaktadır.

**İstem 13:** İstem 1'e bağımlı, burada bulut kodlayıcı, nokta özellikleri üzerinde
max-pool ve çift özellikleri üzerinde mean-pool alarak ikisini birleştirir.

**İstem 14:** İstem 1'e bağımlı, burada dağıtım paketi (290), `model.onnx`,
`quantizer.json`, `token_decoder.json` ve bir `manifest.json` içerir; manifest,
kaynak checkpoint'in ve her artifact'ın kimlik (sha256) ve boyut kaydını, model
yapılandırmasını ve bir self-test sonucunu taşır.

**İstem 15:** İstem 14'e bağımlı, burada ham satırlarda `model.onnx` çıktısı,
referans (torch) çıktısıyla **bit-kesin** olarak (max |Δp| eşiğinin altında)
doğrulanır.

**İstem 16:** İstem 1'e bağımlı, burada girdi sözleşmesi, `ids (B,K) int64` +
`mask (B,K) bool` → `p (B)` biçimindedir ve dinamik batch taşır.

**İstem 17:** İstem 3'e bağımlı, burada program talimatları, geçici olmayan, bilgisayarca okunabilir bir kayıt ortamında depolanmıştır.

**İstem 18:** İstem 1'e bağımlı, burada D, d_cloud ve d_model değerleri, görevin
ölçülen karmaşıklığı (bölge sayısı, bulut yayılımı) esas alınarak önceden
tanımlı bir değer kümesinden seçilir.

**İstem 19:** İstem 1'e bağımlı, burada sistem, kredi kartı işlemleri, bankacılık
işlemleri veya herhangi bir `(sayısal, kategorik)` tablolar kaydı alanı
için uygulanabilir.

---

## 4. ÖZET (ABSTRACT)

Bir tablolar kaydın her özelliği, ölçülen değer-bucket'ına göre bir
`(özellik, değer-bucket)` kimlik token'ına ve düşük boyutlu (3–64) bir noktaya
dönüştürülür; kayıt, bu noktaların permütasyon-değişmez bulutudur. Bilinen bir
nokta-bulut kodlayıcı küme durumunu, küçük bir okuyucu bir risk skorunu üretir.
Skor, buluttan **tek nokta çıkarıldığında** logit-değişimi olarak ölçülür ve en
büyük katkılı noktalar **tam-sözlük** bir decoder üzerinden **düz-metne**
çözümlenir — böylece özellik-ağırlığı listesi yerine okunabilir bir kayıt-
açıklaması elde edilir. Çift-geometri, katkıları non-additive kılar; bu, tek-ve
çift-token **karşı-faktürel** maskelerle dürüstçe raporlanır ve hatalı-uyarı
popülasyonunun genel vs nadir token'lara ayrışması ölçülür. Model, `onnx` +
`quantizer` + `decoder` + `manifest` (sha256, config, self-test) olarak
**self-contained** bir paket olur; ham satırlarda bit-kesin, kenar-cihazda düz-
metin üreten, sub-megabyte bir artifact olarak dağıtılır.

---

## 5. ŞEKİLLER (Ayrı Sayfalar Olarak Sunulmalıdır)

| Sayfa | Şekil | İçerik | Kaynak |
|---|---|---|---|
| 1 | Şekil 1 | Sistem mimarisi (200–290 referans numaralarıyla) | Yeni çizim (blok diyagram) |
| 2 | Şekil 2 | Kimlik-bucket projeksiyonu — kayıt noktaları bulutta | `fig/fig1_fraud_bulut.png` |
| 3 | Şekil 3 | Leave-one-point-out ablation (logit değişimi) | Yeni çizim (şema) |
| 4 | Şekil 4 | Tek-ve çift-maske karşı-faktürel | Yeni çizim (şema) |
| 5 | Şekil 5 | Self-contained paket + ham-satır bit-kesinlik | `fraud/results/deploy/` |

> **Vekil notu:** Şekiller siyah-beyaz çizgi, referans numaralarıyla, ayrı sayfalarda
> PDF çizim formatında hazırlanmalıdır. Bu dosya yeni bir buluş (v1) olup, dil
> modelleme patentinden (`../dil/patent_application_v2.3.md`) ayrık bir başvuru olarak
> sunulur; ikisi arasındaki bulut kodlayıcı çekirdeği çakışması, vekil tarafından
> çapraz-referans / öncelik (priority) açısından değerlendirilmelidir.

---

## 6. VEKİL DEĞERLENDİRMESİ (Bu bölüm başvuru dosyasına EKLENMEZ — strateji notu)

### 6.1 Neden ayrı başvuru?

Dil modelleme buluşu (`../dil/patent_application_v2.3.md`) **cümle → word-token → bulut →
bir sonraki token** idi. Bu buluş, aynı çekirdeği **tablolar kayıt** hedefine
kaydırır ve asıl yeniliği **açıklama yüzeyinde** (kimlik-bucket + çözümlenen
ablation + karşı-faktürel denetim + self-contained dağıtım) taşır. Ayrı dosya:
(a) iddia kapsamını dil alanından koparıp **tablolar/uyum (compliance)** alanına
sıkar, (b) SHAP/GBM önceliğiyle daha net bir boşluk iddiası kurar, (c) tek bir
başvuruyu aşırı genişletip (dil + tabular) reddedilirlik riskini düşürür.

### 6.2 Yenilik (novelty) riski

- **Güçlü:** (d) çözümlenen ablation + (e) karşı-faktürel + (f) self-contained
  paket *bileşimi* tek artifact olarak — SHAP/GBM ve set-tablolar edebiyatında
  bu dörtlü birlikte bilinmemektedir.
- **Dikkat:** Bulut kodlayıcı + 0. sıra daraltma, bitişik patentin (v2.3)
  kapsamındadır; bağımsız İstem 1, bu çekirdeği **bilinen** gibi referans
  vermektedir. Vekil, İstem 1'deki (b)(c) paragraflarının v2.3'le çakışıp
  çakışmadığını ve gerekirse (b)(c)'yi bağımlı'ya alıp bağımsızı salt
  açıklama/denetim/dağıtım üçlüsüne çekmesini değerlendirmelidir.
- **Veri:** IBM kredi kartı (24.4M), eş-ayrım 1.02M/100K/4.7M; AUC-PR 0.256
  vs XGB 0.80 (honest gap); deploy parite 1.5e-6, 45.5k kayıt/sn; audit 1842/9942/
  4268. Patent metni *performans üstünlüğü* iddia etmez; iddia **açıklama +
  denetim + dağıtım** bileşimidir.

### 6.3 Sonraki adımlar

- EN çevirisi (`patent_tabular_explain_v1_EN.md`) — `_archive/docs_patent/patent_application_v2.3_EN.md` ile aynı bölüm
  şemasında.
- Şekiller 1, 3, 4 yeni çizim gerektirir; 2 ve 5 mevcut görsel/artifact
  temellidir.
- Vekile: bağımsız İstem 1'in (b)(c) paragraflarının v2.3 çakışması +
  öncelik (priority date) stratejisi.
