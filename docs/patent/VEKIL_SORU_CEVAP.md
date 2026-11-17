# VEKİLE SORU–CEVAP — Başvuru Sırasında Gelen Sorular

**Konu:** Düşük-Boyutlu Nokta-Bulutu Geometrisi buluşu — vekilin başvuru
hazırlığı sırasında sorduğu sorular ve kod-kanıtlı yanıtları.
**Tamamlayıcı doküman:** ana yapı anlatımı için [`VEKIL_ANLATIM.md`](VEKIL_ANLATIM.md);
paket listesi ve strateji için [`VEKIL_GONDERIM_NOTU.md`](VEKIL_GONDERIM_NOTU.md).

## Yanıt biçimi (kural)

Her yanıt üç parça taşır: **kısa cevap** (jargonsuz), **teknik ayrıntı**
(hangi modül yapıyor), **kanıt** (`dosya:satır`). Kaynak gösterilemeyen ifade
"kanıtsız" diye etiketlenir; sayı uydurulmaz.

**Hakem kuralı:** kod ile metin/şekil çelişirse **kod hakemdir**; metin koda göre
düzeltilir. Her sorunun sonunda `METNE ETKİSİ` bunu yazar.

---

## Soru 1 — Çalışan mimaride Transformer ve QUANN birlikte mi?

**Kısa cevap:** Hayır, birlikte değil. Dil yolunda Transformer katmanı var, QUANN
yok. Tabular/fraud yolunda QUANN'ın geri-yayılımsız güncellemesi var, Transformer
yok.

**Teknik ayrıntı:**

- Dil modeli standart PyTorch Transformer katmanı kullanır; dikkat elle
  yazılmamıştır: `cloudlm/model.py:234-239` (`nn.TransformerEncoderLayer` +
  `nn.TransformerEncoder`).
- Dil eğitimi geri-yayılımla: `training/train_wiki.py:196` (`torch.optim.AdamW`),
  `training/train_wiki.py:255-256` (`loss.backward()` / `optimizer.step()`).
- `grep -rni "quann" cloudlm/ training/` hiçbir QUANN fonksiyon/modül çağrısı
  vermez; eşleşmeler yalnız başlık/yol/docstring metnidir, ör.
  `training/train_wiki.py:462`.
- Geri-yayılımsız yol tabular tarafta: `fraud/src/train_quann.py:158`
  (`update_token_weights`).
- Bu yolun tanımı: `fraud/src/qcloud_fraud.py:20` ("0th-order, no .backward()").
- **Kodda zorunlu bileşenler** (forward'da koşulsuz, `cloudlm/model.py:306-314`):
  gömme → konum eklemesi → Transformer kodlayıcı → LayerNorm → birimden-noktaya
  izdüşüm (`to_nd`) → bulut kodlayıcı (`cloud`) → görev başlığı.
- **Kodda opsiyonel olanlar:** dolgu maskesi `None` geçilebilir
  (`cloudlm/model.py:305`).
- Özetleme tipi `pool="max"|"attn"` seçeneği `cloudlm/model.py:215` satırında
  tanımlı, uygulaması `cloudlm/model.py:144` satırında.
- Bulut-dikkati `cloud_attn` varsayılan kapalı
  (`cloudlm/model.py:216`, kullanım yeri `cloudlm/model.py:77`).

**METNE ETKİSİ:**

- **Birinci düzeltme:** "Transformer alternatif/opsiyonel" ifadesi **mevcut dil kodunda
   doğrulanmadı**: kodlayıcı koşulsuz (`cloudlm/model.py:239`, `:310`); onu kapatıp
   noktaları doğrudan üreten bir varyant kodda yok. İstemde Transformer
   "gereksiz" diye yazılmamalı; "farklı bir temsil katmanı da kullanılabilir"
   biçiminde eşdeğerlik cümlesi olarak yazılmalı.
- **İkinci düzeltme:** İstemde kullanılabilecek gerçek hiper-değişken adları: `d_model`, `d_cloud`,
   `d_cloud_dim`, `n_layers`, `nhead`, `max_pos_len`, `hidden`, `point_hidden`,
   `pool`, `cloud_attn` (`cloudlm/model.py:211-217`).

---

## Soru 2 — Birim nasıl noktaya dönüşüyor; 20 nokta; uzun girdi; sıra/zaman

**Kısa cevap:** Her girdi birimi gömme vektörü alır, Transformer'dan geçer, iki
katmanlı küçük bir ağla D boyutlu noktaya indirgenir. Nokta sayısı sabit değil;
girdi birimi sayısına eşittir. Sıra bilgisi şu an zorunlu bir bileşen olarak
ekleniyor.

**Teknik ayrıntı:**

- Hat: gömme `nn.Embedding(vocab_size, d_model)` (`cloudlm/model.py:232`) →
  Transformer → LayerNorm → `WordToND.proj`: `Linear → GELU → Linear → Tanh`
  (`cloudlm/model.py:30-33`), çıkış boyutu `d_cloud_dim`.
- Gömme genişliği `d_model` varsayılanı 256 (`training/train_wiki.py:436`).
- Nokta boyutu D: modelde varsayılan 16 (`cloudlm/model.py:212`); eğitim
  betiğinde `--d-dim` varsayılanı 32 (`training/train_wiki.py:438-439`).
- Nokta sayısı: `points = self.to_nd(x)` → şekil `(B, N, D)`
  (`cloudlm/model.py:312-313`); **N = girdi birimi sayısı**. Sabite indirme, top-k seçim
  veya örnekleme yok; özetleme yalnız çıktı tarafında (Soru 3).
- Uzun girdi: pencere `--seq-len` (varsayılan 10, `training/train_wiki.py:435`);
  kaydırılmış pencere şeridi `training/train_wiki.py:45-83`; kısa girdi filtresi
  `training/train_wiki.py:172`. Kesmenin hangi uçtan yapıldığına dair kod satırı
  **bulunamadı** → metne "kesme ucu" yazılmaz.
- Sıra bilgisi: sinüzoidal, öğrenilemeyen konum kodlaması
  (`cloudlm/model.py:182-198`; tampon olarak kayıtlı, eğitilmez).
- Konum kodlaması forward'da **koşulsuz** uygulanıyor
  (`cloudlm/model.py:307`); kapatma bayrağı yok.
- Tabular/fraud yolunda zaman özellik olarak girer: saat sin/cos
  `fraud/src/features.py:78-81`.
- Gün numarası ve hafta-sonu bayrağı: `fraud/src/features.py:93-95`.

**"20 nokta":** kod veya config'de `num_points=20` **yok**. Kodda geçen tek 20,
üretimde üretilen token sayısı (`cloudlm/model.py:316`, `max_new: int = 20`) — nokta
sayısı değil. `n20` geçen log adında pencere değeri açık yazmıyor
(`logs/run_r2_cloud_attn_200k_d64_n20.log:16`) → **"20 nokta" DOĞRULANMADI**.
Başvuru metni nokta sayısını zaten bağlam penceresine bağlıyor
(`docs/patent/dil/TURKPATENT_basvuru_v2.3.md:274`).

**METNE ETKİSİ:**

- **Birinci düzeltme:** `VEKIL_ANLATIM.md:59-60` "pozisyon bilgisi zorunlu değil,
  ek bir bileşen olarak taşınır" **mevcut dil koduyla çelişiyor**
  (`cloudlm/model.py:307` koşulsuz). Seçenekler: madde metinden çıkarılır; ya
  "kapatılabilir konum kodlaması" ayrıca kodlanıp ölçülür; ya istem "uygulanmış
  hâli sıralı konum bilgisiyle" yazar ve sırasız varyant yalnız tercihli seçenek
  olarak anlatılır.
- **İkinci düzeltme:** Sabit "20 nokta" hiçbir yerde iddia edilmemeli; "girdi
  birimi sayısı kadar nokta" yazılmalı.

---

## Soru 3 — Tüm çiftler mi seçilenler mi; ağ katmanları, boyutlar, birleştirme

**Kısa cevap:** Tüm nokta çiftleri kullanılıyor, alt seçim yok. İki küçük ağ var
(nokta ağı ve çift ağı); çıktıları birleştirilip sıradan bağımsız bir özetde
toplanıyor.

**Teknik ayrıntı:**

- Çift kümesi: `diff = points.unsqueeze(2) - points.unsqueeze(1)`
  (`cloudlm/model.py:157`) → **N×N tüm çiftler**. Komşuluk/top-k maskelemesi yok; maske
  yalnız dolgu için (`cloudlm/model.py:171-173`).
- Mesafe: `dist = diff.norm(dim=-1)` — Öklid (`cloudlm/model.py:158`).
- "Açı": iki nokta vektörünün **kosinüs benzerliği** (`cloudlm/model.py:159-161`);
  `arccos`/`atan2` yok.
- Çift özelliği = `[fark, mesafe, kosinüs, |i|, |j|]` birleşimi
  (`cloudlm/model.py:167-168`).
- Bu birleşimin boyutu `3·D + 2` olarak formülle tanımlı
  (`cloudlm/model.py:64`).
- Nokta ağı `point_mlp`: girdi D boyutunda, iki gizli katman, çıkış `d_cloud`
  (`cloudlm/model.py:58-62`).
- Çift ağı `pair_mlp`: girdi çift-özelliği boyutunda, iki gizli katman, çıkış
  `d_cloud` (`cloudlm/model.py:65-69`).
- İki ağın gizli katman genişliği varsayılan olarak 256 (`cloudlm/model.py:50`).
- Birleştirme: `Linear(2·d_cloud → 2·d_cloud) → GELU → Linear → d_cloud` +
  LayerNorm (`cloudlm/model.py:70-74`).
- Özetleme: nokta tarafı `max` (`cloudlm/model.py:153`) veya dikkat-ağırlıklı toplam
  (`cloudlm/model.py:148-149`); çift tarafı `toplam / çift sayısı` = ortalama
  (`cloudlm/model.py:177`). **Sıradan bağımsızlığı sağlayan işlem bu toplam/ortalama**;
  dolgu noktaları sıfırlanır (`cloudlm/model.py:172-173`).
- Görev başlığı: `Linear(d_cloud → d_cloud) → GELU → Linear(d_cloud → vocab)`
  (`cloudlm/model.py:247-250`); dil yolunda çıktı boyutu = sözcük dağarcığı
  (ör. 20000, `training/train_wiki.py:431`).

**METNE ETKİSİ:** "açı" terimi geçiyorsa tanımı **"iki nokta vektörü arasındaki
açının kosinüsü"** olarak netleştirilmeli (kod açının kendisini değil kosinüsünü
hesaplıyor). Hesaplama yüküyle ilgili bir cümle yazılacaksa doğru ifade "tüm
çiftler, N²".

---

## Soru 4 — Aralık güncelleme kuralları (metin "daralıyor", şema "genişliyor")

**Kısa cevap:** İkisi de doğru ama **farklı varyantları** anlatıyor; kodda üç ayrı
kural var. Tek cümlelik "hata azalınca aralık daralır" genellemesi kodla çelişir
ve isteme yazılamaz.

**Üç varyant (kod hakem):**

- **QUANN motoru** — kabul koşulunda aralık **sabit** kalır, yalnız adım ölçeği
  sigma küçülür; reddedilende aday geri alınır, yön tersine döner, aralığın aday
  tarafı ortadan kesilir ve sigma büyür.
- Motor kabul koşulu (hata farkı eşiğiyle karar):
  `quann_cloud/src/qcloud_quann.py:212`.
- Motor kabul dalında sigma küçültme: `quann_cloud/src/qcloud_quann.py:217`.
- Motor ceza dalında geri alma + ortadan kesme + sigma büyütme:
  `quann_cloud/src/qcloud_quann.py:220-228`.
- Motor küçültme çarpanı 0.82: `quann_cloud/src/qcloud_quann.py:62`.
- Motor sigma alt sınırı 0.02: `quann_cloud/src/qcloud_quann.py:60`.
- Motor sigma üst sınırı 0.6: `quann_cloud/src/qcloud_quann.py:61`.
- **classic** — kabulde aralık **genişler**; reddedilende kötü tarafın ötesi
  kesilir, değer sabit kalır.
- Classic kabul dalı ("ÖDÜL: aralığı genişlet"):
  `quann_cloud/src/qcloud_weight.py:115-122`.
- Classic ceza dalı ("CEZA: daralt", "value değişmez!"):
  `quann_cloud/src/qcloud_weight.py:125-133`.
- Classic genişleme adımı 0.1: `quann_cloud/src/qcloud_weight.py:53`.
- **hybrid** — kabulde aralık **kabul edilen segmente daralır**; reddedilende
  reddedilen tarafın ötesi kesilir, değer sabit.
- Hybrid kabul dalı (segment `[v, c]`): `quann_cloud/src/qcloud_weight.py:173-177`.
- Hybrid ceza dalı: `quann_cloud/src/qcloud_weight.py:179-181`.
- Kural seçimi (classic / hybrid anahtarı): `quann_cloud/src/qcloud_trainer.py:199`.

**Çelişkinin tam kaynakları:**

- "daralıyor" diyen: `docs/patent/VEKIL_ANLATIM.md:47-49` ("arama aralığı zamanla
  daralır") — kabul dalı için bu yalnız hybrid'de doğru.
- "genişliyor" diyen: patent şekli değil, tasarım notu
  `quann_cloud/DESIGN.md:108-115` (ödül dalında genişleme).
- Aynı tasarım notunun vektör hâli: `quann_cloud/DESIGN.md:120-128`.
- Aynı kuralın özet satırı: `docs/quann_evolution_analysis.md:15`.
- Özet hüküm: kabul dalında aralık **hybrid**'de daralır, **classic**'te genişler,
  **motor**da değişmez (yalnız sigma küçülür); motor aralığı yalnız ceza
  dalında keser.
- Patent çizimlerinde ödül/aralık ağacı **yok**: `docs/patent/generate_patent_figures.py`
  şekilleri mimari bloğu ve eğri çizimidir; `docs/patent/dil/fig/` SVG'lerinde
  "aralık/ödül/reward" eşleşmesi yok.

**Bu yöntemle hangi parametreler öğreniliyor:**

- QUANN motoru: `requires_grad` işaretli **tüm** model parametreleri
  (`quann_cloud/src/qcloud_quann.py:112-116`).
- Her adımda rastgele seçilen ağırlık elemanı sayısı 60
  (`quann_cloud/src/qcloud_quann.py:65`); seçim mantığı
  `quann_cloud/src/qcloud_quann.py:142`.
- QCloudWeight yolu: yalnız **birim→nokta tablosu**
  (`quann_cloud/src/qcloud_network.py:288`); geri-yayılım o tabloya dokunmaz
  (`quann_cloud/src/qcloud_trainer.py:197`).

**Hangi deney hangi varyantı kullandı:**

- hybrid → AML'de kullanım: `aml/src/train_aml.py:94` (`narrow_search`).
- hybrid → AML sonucu kaydı: `aml/results/e3_prime.json:85`.
- hybrid → fraud'ta kullanım: `fraud/src/train_strong_reader.py:7`
  ("full hybrid checkpoint").
- fraud checkpoint adı: `fraud/src/train_strong_reader.py:38`.
- QUANN motoru → deney: `quann_cloud/experiments/real_data_test.py:283`.
- QUANN motoru → sonuç satırında held-out PPL 1016.30
  (`quann_cloud/logs/full_run.log:46`).
- classic → canlı deney kanıtı **bulunamadı**; yalnız kod varsayılanı
  (`quann_cloud/src/qcloud_network.py:285`).

**METNE ETKİSİ** (bu madde `DEVIR.md:54-61`'de açık iş olarak kayıtlıydı):

- **Birinci düzeltme:** Yol B istemi **varyant-spesifik** yazılmalı. Öneri **hybrid** (hem AML hem
   fraud deneyleri onu kullanıyor): *"kabul edilen deneme ile mevcut değer
   arasındaki aralık korunur, aralığın reddedilen tarafı kesilir"*.
- **İkinci düzeltme:** "Hata azalınca aralık daralır" cümlesi `VEKIL_ANLATIM.md:47-49`'dan
   kaldırılmalı veya varyant adı verilmeli.
- **Üçüncü düzeltme:** Sigma (adım ölçeği) ile arama aralığı (alt/üst sınır) metinde **iki ayrı
   kavram** olarak tanımlanmalı; motor varyantında kabulde değişen şey aralık
   değil sigmadır.

---

## Soru 5 — Nokta çıkarılınca ne yeniden hesaplanıyor; prototip, isim, eşikler

**Kısa cevap:** Nokta **temsil uzayından** çıkarılıyor; gömme yeniden
hesaplanmıyor, kodlayıcı + özetleme + başlık yeniden koşuyor. Prototipler eğitim
verisinin sınıf merkezleri (veya KMeans merkezleri); atama en yakın merkeze göre.
Ayrı bir "belirsizlik skoru" kodda yok.

**Teknik ayrıntı:**

- Nokta çıkarma fonksiyonu: `fraud/src/explain_pilot.py:93-112`.
- Nokta tablosu **bir kez** üretilir: `fraud/src/explain_pilot.py:102`.
- Her nokta için ilgili satır sıfırlanır ve maskesi kapatılır:
  `fraud/src/explain_pilot.py:106-109`.
- `points_override` verildiğinde gömme ve birimden-noktaya izdüşüm **atlanır**:
  `fraud/src/qcloud_fraud.py:129-135`.
- Yeniden hesaplanan aşamalar bulut kodlayıcı + özetleme (`cloudlm/model.py:130-179`); başlık
  `cloudlm/model.py:314` satırında uygulanıyor (tanım
  `cloudlm/model.py:247-250`).
- Nokta tarafı özetleme (max): `cloudlm/model.py:153`.
- Çift tarafı özetleme (ortalama): `cloudlm/model.py:177`.
- Örnek başına K+1 ileri geçiş (bir taban + K çıkarma), `torch.no_grad()` altında:
  `fraud/src/explain_pilot.py:98`.
- Ölçülen büyüklük logit farkı (`base - ablated`): `fraud/src/explain_pilot.py:111`.
- ONNX sürümünde olasılık logite çevrilip fark alınıyor: `fraud/pilot/infer.py:107`.
- Karar değişimi eşikle ayrıca sayılıyor: `fraud/src/audit_false_alarms.py:144`.
- Ham girdiden birim silip tüm modeli yeniden çalıştıran bir yol **yok**.
- Karıştırmama uyarısı: AML'deki "ablation" **özellik-seti** karşılaştırmasıdır,
  nokta çıkarma değil: `aml/src/e3_chain.py:283`.
- Aynı AML dosyasında delta hesabı: `aml/src/e3_chain.py:327-338`.
- Prototip üretimi (eğitim-only sınıf merkezi): `aml/src/e1_explain.py:296-299`.
- Aynı tanım AML zincir deneyinde: `aml/src/e3_chain.py:241-244`.
- Prototip tanımının JSON metni: `aml/src/e3_chain.py:362`.
- Bakım dikeyinde sınıf merkezi: `maintenance/src/e1_real.py:251-252`.
- KMeans merkezi alternatifi: `complaints/src/explain_region.py:47-48`.
- Otomatik isimlendirme (bölgedeki en sık alt-birimler, duran sözcük filtresi,
  ilk üçü ad): `complaints/src/explain_region.py:65-71`.
- Gerçek isimli bölge çıktısı: `complaints/results/region_map.json`.
- AML/bakımda ad = sınıf etiketi: `aml/src/e1_explain.py:204`.
- Elle yazılmış tek ad örneği: `aml/src/e4_adapt.py:547`.
- Atama en yakın prototipe (Öklid argmin): `aml/src/e1_explain.py:316-323`.
- Aynı atama şikayet dikeyinde: `complaints/src/audit_drift.py:95`.
- Olasılıksal veya eşikli atama yok.
- Belirsizlik: ikinci-en-yakın prototip mesafesi kaydediliyor
  (`aml/src/e1_explain.py:342`) ama bir belirsizlik skoruna dönüştürülmüyor;
  entropi/margin formülü yok.
- Drift — bölge payı değişiminde toplam-varyasyon:
  `complaints/src/audit_drift.py:36`.
- Drift — KL karşılığı: `complaints/src/audit_drift.py:40`.
- Drift — yeni bölge bayrağı: `complaints/src/audit_drift.py:44-45`.
- Bakımda mesafe tabanlı alarm: `maintenance/src/e1_real.py:288-303`.
- AML'de mesafe tabanlı alarm: `aml/src/e4_adapt.py:296-315`.
- Prototip **merkez yer değiştirmesi/açısı** ölçüsü yok.
- Eşik — bakım alarm eşiği yüzde 95 quantile: `maintenance/src/e1_real.py:299`.
- Eşik — AML işleyiş noktası yüzde 99 quantile: `aml/src/e4_adapt.py:178`.
- Eşik — bölge yarıçapı yüzde 95 quantile: `aml/src/e4_adapt.py:306`.
- Eşik — fraud'ta doğrulama kümesinde F2 maksimizasyonu:
  `fraud/src/threshold_utils.py:12-30`.
- Eşik — sabit sayılar 0.01 ve 0.05: `complaints/src/audit_drift.py:20-21`.

**METNE ETKİSİ — üç uyarı:**

- **Birinci uyarı — şekillerdeki sayılar ölçüm değil.**
   `docs/patent/generate_patent_figures.py:173-204` içindeki `logit değişimi: -2.34`,
   `Mesafe: 0.42`, `Destek: %78`, `%21→%26` değerleri şekle elle yazılmış örnek
   metindir; hiçbir log veya JSON çıktısında karşılığı yok. Vekil bunları ölçülmüş
   sonuç gibi sunmamalı: ya şekle "şematik örnek" ibaresi konur ya gerçek çıktı
   değerleriyle yeniden üretilir.
- **İkinci uyarı — dil embodiment'ında açıklama kodu yok.** Üç açıklama mekanizması
   fraud/AML/bakım/şikayet dikeylerinde kodlu; `cloudlm/` içinde açıklama modülü
   yok. **Dosya A (yalnız dil)** içinde native açıklanabilirlik iddiası
   yazılacaksa ya dil ucunda kod + ölçüm üretilir ya bu iddia Dosya B (evrensel)
   çekirdeğine bırakılır; aksi hâlde uygulanabilirlik (enablement) itirazı riski
   doğar.
- **Üçüncü uyarı:** "Belirsizlik ölçüsü" ve "merkez kayması biçiminde drift" **istem olarak
   yazılmamalı**; kodda karşılığı yok. Drift istemi yazılacaksa ölçülen büyüklük
   açıkça "bölge payları dağılımının zaman içinde değişimi (TV/KL)" veya
   "prototipe uzaklığın eşik aşımı" olarak tanımlanmalı.

---

## Vekilin soruları sonrası açılan düzeltme listesi

| # | Konu | Kaynak | Eylem |
|---|---|---|---|
| 1 | "pozisyon bilgisi zorunlu değil" ifadesi kodla çelişiyor | `cloudlm/model.py:307` | `VEKIL_ANLATIM.md:59-60` düzelt veya kapatılabilir konum kodlaması kodla |
| 2 | "arama aralığı zamanla daralır" genellemesi | `qcloud_quann.py:217` · `qcloud_weight.py:115-122` | `VEKIL_ANLATIM.md:47-49` varyant adı vererek yaz |
| 3 | Şekillerde elle yazılmış sayılar | `generate_patent_figures.py:173-204` | "şematik örnek" ibaresi veya gerçek değerlerle yeniden üretim |
| 4 | Dil ucunda açıklama kodu yok | `cloudlm/` (dosya yok) | Dosya A'da açıklama iddiasının kapsamı kararlaştırılmalı |
| 5 | Sabit "20 nokta" sayısı | kodda `num_points` yok | Metinlerde sabit nokta sayısı iddiası kaldırıldı/kaldırılmalı |
| 6 | Paket sürüm başlığı uyuşmazlığı | `DEVIR.md:55-56` | Gönderim listesiyle senkronlanmadan gönderim yapılmaz |
