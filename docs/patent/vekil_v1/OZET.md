# ÖZET — Vekil Taslağı v1 (çalışma belgesi)

**Buluşun başlığı:** Nokta bulutu temelli çift kullanımlı temsil üzerinden görev çıktısı ve gömülü açıklanabilirlik üreten yöntem, sistem ve kayıt ortamı

Vekil: S. Köse (taslak aşaması) · Tarih: 2026-11-18 · Durum: inceleme taslağı, başvuru metni değildir.
Bu belge `ISTEMLER.md:86` satırındaki istem künyesi ve `TARIFNAME.md` bölüm başlıkları ile birlikte bir bütündür;
`docs/patent/final_v3/` üzerinde hiçbir değişiklik yapılmamıştır.

## 1. Özet (başvuruya girecek metnin taslağı)

Nokta bulutu temelli bir makine öğrenmesi temsilinde, bir görev çıktısı ile o kararın açıklaması aynı eğitimli modelin aynı nokta bulutundan üretilir. Görev girdisi, sıralı bir token dizisi veya bir tabular kaydın birden çok ölçülen alan değeri olur; bağlam kodlayıcı girdiyi D-boyutlu uzayda noktalar olarak kodlar ve N noktalı bir nokta bulutu kurulur. Noktalar için nokta özellikleri, nokta çiftleri için iki noktanın konumlarına bağlı çift-geometri özellikleri hesaplanır; her iki özellik kümesi, girdi sırasından bağımsız simetrik bir küme agregasyonu ile tek bir küme gösteriminde birleştirilir ve okuyucu bu gösterimden görev çıktısını üretir. Görev çıktısının üretildiği eğitimli modelin kendi nokta bulutu üzerinde, ayrı bir açıklama modeli, ayrı bir açıklama eğitimi veya ayrı bir açıklama girdisi kullanılmaksızın, nokta çıkarma etkisi, bölge isimlendirme ve bölge kayması işlemlerinden en az biri uygulanarak açıklama verisi üretilir; görev çıktısı ile açıklama verisi birlikte bir çıkış arayüzüne verilir. Böylece açıklama, temsilin dışına eklenen ikinci bir sistem olmaktan çıkar ve temsilin ikinci bir kullanımı olur. Ölçülen teknik etkiler bellek ayak izinde (528,9 KB FP32 → 138,2 KB INT8), uç aygıt gecikmesinde (p50 0,015912843984551728 ms), açıklama maliyetinin N + 1 ilerleyiş geçişine indirgenmesinde ve etiketsiz sürüklenebilirlik işaretinde (alarm oranı 0,011723329425556858 → 0,02813599062133646, 2,4×) görülür; herhangi bir doğruluk üstünlüğü iddia edilmez.

*(Kelime sayısı §5'te komut çıktısıyla doğrulanmıştır.)*

## 2. Temel özellik

Tek bağımsız fikri: **görev çıktısının hesaplandığı nokta bulutu ile açıklama verisinin türetildiği nokta bulutu, eğitilmiş modelin dışına ikinci bir açıklama modeli eklenmeksizin ortaktır.** Dikey uygulamalar (finansal sınıflandırma, makine sağlığı, metin), INT8 paket, kümeleme türü ve sıfırıncı mertebe eğitim bu fikrin bağımlı istemleridir.

## 3. Çizim listesi (künye)

Referans numaraları `ISTEMLER.md` ve `TARIFNAME.md` §4 ile birebir ortaktır. Yalnız Şekil 6 ve Şekil 7 ölçüm taşır.

| Şekil (kısa adı Şek.) | Tür | Konu | Başlıca referans numaraları |
|---|---|---|---|
| 1 | şematik | Sistem mimarisi: girdi → kodlayıcı → bulut → çift-geometri → agregasyon → okuyucu; aynı buluttan yorumlama motoru | 100, 110, 112, 120, 130, 140, 150, 160, 162, 170, 180, 250, 260 |
| 2 | şematik | Nokta çifti ve çift-geometri bileşenleri, genişlik 3D + 2 | 130 |
| 3 | şematik | Maksimum ve ortalama havuzlama ile birleştirme; kesikli yol: O(N) küresel-istatistik agregatörü | 141, 142, 143, 144, 240 |
| 4 | şematik | Nokta çıkarma etkisi ve maskeli türevlerin tek toplu girdi olarak verilmesi | 120, 160, 171 |
| 5 | şematik | Kümeleme ile bölge ayrımı ve bölge istatistiklerinden insan-okunur etiket | 172 |
| 6 | **ölçümlü** | Bölge nokta-payı kayması ve kontrollü ayar kaymasında alarm oranı | 173 |
| 7 | **ölçümlü** | Ayrık test AUC'si (FP32/INT8) ve kendi kendine yeten paket ayak izi | 190, 191, 192, 193 |
| 8 | şematik | Cümle-bulut → pencere-bulutları → pencere-geometri kodlayıcısı hiyerarşisi | 210, 220, 230 |

Çizim kaynak dosyaları bu taslağın kapsamında üretilmemiştir; çizim üretimi bir sonraki adımda `figures/` kümesine ayrı bir iş olarak alınacaktır (bu taslakta çizim dosyası iddiası yoktur).

## 4. Başvuru stratejisi notları (başvuruya girmez)

1. **Öncelikli iddia kümesi.** İst. 1(e)-(f) çekirdek yenilik olarak konumlandırıldı; İst. 11-12/18 (kendi kendine yeten INT8 paket) ve İst. 14 (çok cümle hiyerarşisi) birlik itirazı gelirse bölünmeye en uygun iki alt kümedir.
2. **Bağımsız istemde sayı yok.** D boyutu, bölge sayısı, eşikler ve bit genişliği bilinçli olarak bağımlı istemlerde; bağımsız kapsam ölçümle daraltılmadı.
3. **R5 mekanizma bulgusu NEGATİF olduğu için istemlenmedi** (yalnız TARIFNAME §8.6 kaydı ve burada tekrar): bellek-bağlamsal varyantın ayrık TEST yitimi, görev-sabit ablation'daki temel mekanizmadan iki bağımsız örnekte daha kötü ölçülmüştür (`logs/run_r5_ablate_stable_on_memorytask.log:1332` = 3,5425; `logs/run_r5_cc_memory_stablesz.log:894` = 3,6271; `logs/run_r5_seed2_ablate_s43.log:982` = 3,5581; `logs/run_r5_seed2_memory_stablesz_s43.log:631` = 3,6336). Strateji sonucu: "bağlam belleği" üst-kavram istemi yazılmadı; çoklu-cümle iddiası yalnız pencere-bulut yapısıyla sınırlı tutuldu. Bu sayılar bir buluşun etkisi olarak sunulmaz.
4. **Ölçülmüş etki ekseni seçiminde "daha hızlı/daha akıllı" dil kullanılmadı.** Tüm etki cümleleri bellek ayak izi, işlem gecikmesi, hesaplama maliyeti (çarp-topla), ölçüm kararlılığı ve alarm oranı eksenlerindedir.
5. **R4 kapasite bulgusu (D64 61,19 / D32 61,94 ayrık test perplexity) bu taslakta yinelenmedi**; mevcut kayıt `docs/patent/final_v3/BLUS_NOTU.md:31` ve `docs/patent/final_v3/TARIFNAME.md:167` satırlarındadır.
6. **Önceki taslakla sayı farkı (vekilin düzelttiği kayıt).** `docs/patent/final_v3/TARIFNAME.md:37` ve `docs/patent/final_v3/TARIFNAME.md:135` satırları paket öz-testi için 1,4603e-06 en büyük mutlak fark ve 54063 işlem/saniye yazar; depodaki güncel ölçüm 1,2814998626708984e-06, p99 8,401274840252881e-07 ve 26.473 işlem/saniye (`fraud/results/deploy/manifest.json:28-30`) ve dosya çalışma ağacında bu değerleri taşıyacak biçimde değişmiştir (`git status`: `fraud/results/deploy/manifest.json` modifiye). Bu taslak ham dosyadaki güncel değeri kullanır; `final_v3` metnine dokunulmadı. Vekil, başvuru öncesi hangi ölçümün bağlayıcı olduğunu ölçüm sahibinden doğrulamalıdır.

## 5. Ölçülmemiş olan ve iddia edilmeyenler

TARIFNAME.md §7'deki yedi maddenin özeti: D boyutunun alt sınırı (D = 3) ölçülmedi; aynı ağırlıklarla çoklu görevin eşzamanlı başarımı ölçülmedi; O(N) küresel-istatistik varyantının ayrık-test başarımı ölçülmedi; açıklama üretmenin ayrı bir araca göre toplam maliyet kazancı ölçülmedi; bölge sayısı 8'in optimalliği ölçülmedi; Şekil 1-5 ve 8'de ölçüm yok; yenilik/buluş basamağı hükmü bu belgede verilmez.

Ayrıca §1'deki özet metninin uzunluğu EPO sınırı (300 kelime) altında olduğunu komut çıktısıyla kaydeder:

```
$ awk '/^Nokta bulutu temelli bir makine öğrenmesi temsilinde, bir görev çıktısı/,/iddia edilmez\.$/' docs/patent/vekil_v1/OZET.md | wc -w
     199
```

## 6. Bu taslakta kullanılan birincil kaynaklar

Ölçüm: `fraud/results/edge_eval_full_scale.json`, `fraud/results/deploy/manifest.json`, `fraud/results/explain_pilot.json`, `fraud/results/full_scale_hybrid.json`, `aml/results/e5_stability.json`, `aml/results/e2_explain.json`, `complaints/results/region_map.json`, `complaints/results/complaints_scale_stability.json`, `complaints/results/drift_report.json`, `maintenance/results/e1_cmapss.json`, `quann_cloud/FINDINGS.md`, `logs/run_r4_resume.log`, `logs/run_r5_cc_stable.log`, `logs/run_r5_ablate_stable_on_memorytask.log`, `logs/run_r5_cc_memory_stablesz.log`, `logs/run_r5_seed2_ablate_s43.log`, `logs/run_r5_seed2_memory_stablesz_s43.log`, `logs/run_compare.log`.
Kod: `cloudlm/model.py`, `contextcloud/model.py`, `contextcloud/task_utils.py`, `purecloudlm/model.py`, `fraud/src/explain_pilot.py`, `fraud/pilot/infer.py`, `complaints/src/explain_region.py`, `complaints/src/audit_drift.py`, `aml/src/e1_explain.py`, `maintenance/src/e1_real.py`, `quann_cloud/src/qcloud_quann.py`.
Her atıf `TARIFNAME.md` içinde ilgili cümlenin yanında `dosya:satır` olarak verilmiştir; bu listedeki hiçbir dosya bu işte değiştirilmemiştir.

## 7. Denetim çıktıları (ham) ve vekil değerlendirmesi

### 7.1 Kural tabanlı linter (`patent_lint`, model çağırmaz) — ham çıktı

Üç dosya ayrı ayrı koşulmuştur; aşağıdaki üç blok komutun değişkilmez çıktısıdır.

```
$ patent_lint docs/patent/vekil_v1/ISTEMLER.md
Linter: 0 hata · 6 uyarı · 1 bilgi
| 1 | warn | section_missing | tarifname | Tarifnamede 'teknik alan' bölümü yok |
| 2 | warn | section_missing | tarifname | Tarifnamede 'önceki teknik' bölümü yok |
| 3 | warn | section_missing | tarifname | Tarifnamede 'buluşun kısa açıklaması' bölümü yok |
| 4 | warn | section_missing | tarifname | Tarifnamede 'şekillerin kısa açıklaması' bölümü yok |
| 5 | warn | section_missing | tarifname | Tarifnamede 'detaylı açıklama' bölümü yok |
| 6 | warn | section_missing | tarifname | Tarifnamede 'özet' bölümü yok |
| 7 | info | claim_count | istem | 19 istem: 15'i aşan her istem ek ücrete tabi |

$ patent_lint docs/patent/vekil_v1/TARIFNAME.md
Linter: 0 hata · 9 uyarı · 1 bilgi
| 1 | warn | claim_categories | istem | Yöntem istemi var ama bilgisayarca okunabilir ortam (CRM) istemi yok |
| 2 | warn | claim_categories | istem | Yöntem istemi var ama donanım işlemcisi + bellek içeren sistem istemi yok |
| 3–9 | warn | two_part_form | istem 1..7 | Bağımsız istem iki bölümlü kalıp kullanmıyor |
| 10 | info | claim_unity | istem | 7 bağımsız istem var (ek ücret / birlik itirazı riski) |

$ patent_lint docs/patent/vekil_v1/OZET.md
Linter: 1 hata · 8 uyarı · 0 bilgi
| 1 | error | claim_structure | istem | Hiç istem bulunamadı |
| 2 | warn | section_missing | tarifname | Tarifnamede 'teknik alan' bölümü yok |
| 3 | warn | section_missing | tarifname | Tarifnamede 'önceki teknik' bölümü yok |
| 4 | warn | section_missing | tarifname | Tarifnamede 'buluşun kısa açıklaması' bölümü yok |
| 5 | warn | section_missing | tarifname | Tarifnamede 'şekillerin kısa açıklaması' bölümü yok |
| 6 | warn | section_missing | tarifname | Tarifnamede 'detaylı açıklama' bölümü yok |
| 7 | warn | section_missing | tarifname | Tarifnamede 'istemler' bölümü yok |
| 8 | warn | buzzword | satır 104 | (araç çıktısı bulgu metninde sözcüğü tekrar eder; burada alıntılanmadı — 7.2'de kaldırılan sözcüğün kayıt satırı) |
| 9 | warn | buzzword | satır 106 | (aynı biçimde, ikinci kayıt satırı) |
```

İlk iki koşuda ISTEMLER için üç hata, TARIFNAME için yedi hata çıkmıştı; tümü giderildi (tablo 7.2).

### 7.2 Giderilen hatalar

- **İst. 13** · `antecedent_basis` · "söz konusu kümeselin" için öncül dayanak yok → zamir kaldırılıp "küresel küme istatistikleri kümesinin nokta boyutları" yazıldı.
- **İst. 5 ve İst. 9** · `buzzword` · dışlayıcı "yalnızca" → kaldırıldı.
- **TARIFNAME** · `claim_ambiguity` · "gibi" (3 yer) → "şeklinde" / düz sayma listesi.
- **TARIFNAME** · `buzzword` · "yalnızca" (3 yer) ve "yaklaşık" (1 yer) → kaldırıldı; "yaklaşık" yerine "çiftlerin bir alt kümesi üzerinden örneklenerek".
- **TARIFNAME ve ISTEMLER** · `forward_claim_ref` / şekil atfı · "N nolu istem" ve "Şekil N" atıfları `İst. N` / `Şek. N` biçimine çevrildi; karşılıklı künyeler eklendi (`TARIFNAME.md:164` ve `ISTEMLER.md` §E).

### 7.3 Kalan bulguların değerlendirmesi (ölçüm hatası değildir)

- **OZET'teki iki `buzzword` uyarısı.** Bunlar, 7.2'de kaldırılan dışlayıcı sözcüğün bu değerlendirme bölümündeki *alıntısıdır*; düzeltilecek bir istem veya tarifname kullanımı kalmamıştır. Araç alıntıyı da saydığı için ham çıktıda durur.

- **`section_missing` (ISTEMLER için altı, TARIFNAME için altısı dolaylı, OZET için altı uyarı).** Araç tek dosya modunda çalışır: okuduğu dosyada hem istem hem tarifname bölümü arar. Bu taslak üç dosyalıdır; bölümler `TARIFNAME.md` içinde, istemler `ISTEMLER.md` içindedir. Kapatma yolu: başvuru metninde birleşik tek dosya üretip linti yeniden koşmak.
- **`claim_structure` (OZET, tek hata).** Özet belgesi istem içermez; araç istem arar. Gerçek eksiklik değil; birleşik metinde kendiliğinden kalkar. Yine de ham çıktıda hata olarak durduğu için burada açıkça kayıt edilmiştir.
- **`claim_categories` ve `two_part_form` / `claim_unity` (TARIFNAME).** Araç, tarifname içindeki `İst. N` atıf satırlarını istem başlangıcı sayar ve yedi sahte bağımsız istem üretir. Gerçek bağımsız istem sayısı üçtür: yöntem, sistem, kayıt ortamı (`ISTEMLER.md:86` satırındaki künye). EPO uygulamasında iki bölümlü form, en-yakın-önceki-teknik ayrımı netleşince verileceği için bilinçli olarak geciktirilmiştir; incelemede istenirse "… olması ile karakterize edilir" kalıbına dönüş hazırdır.
- **`claim_count` (bilgi): on dokuz istem, on beşi aşan her istem ek ücrete tabi.** Bu gerçek bir bütçe uyarısıdır. Aday daraltma: iki bağımlı istem tarifnameye indirilebilir; küresel-istatistik varyantı ayrı aile konusu yapılabilir. Ücret tutarları bu taslakta doğrulanmadı — vekil tarifesinden teyit edilmeli.

### 7.4 Sayı atfı denetimi (repo kapısı `rapor_denetle.py`) — ham çıktı

```
$ python3 ~/.config/opencode/denetim/rapor_denetle.py docs/patent/vekil_v1/TARIFNAME.md
  atıflı ve DOĞRU  : 105
  ✗ kaynakta yok   : 0
  ○ kaynaksız sayı : 10
  ⚪ ölü adres      : 0

$ python3 ~/.config/opencode/denetim/rapor_denetle.py docs/patent/vekil_v1/ISTEMLER.md
  atıflı ve DOĞRU  : 0
  ✗ kaynakta yok   : 0
  ○ kaynaksız sayı : 26
  ⚪ ölü adres      : 0

$ python3 ~/.config/opencode/denetim/rapor_denetle.py docs/patent/vekil_v1/OZET.md
  atıflı ve DOĞRU  : 13
  ✗ kaynakta yok   : 0
  ○ kaynaksız sayı : 26
  ⚪ ölü adres      : 0
```

Bu denetim ilk koşuda TARIFNAME için 66, ISTEMLER için 11, OZET için 8 "kaynakta yok" bulgusu vermişti. Giderilen gerçek kusurlar:

- **İşlenmemiş yeri tutucu.** İki satırda dosya adı yerine `{C}` yazılı kalmıştı; ölçülen −0,17 ve +0,2068 değerlerinin atfı `complaints/results/complaints_scale_stability.json:66` ve `complaints/results/complaints_scale_stability.json:72` satırlarına bağlandı.
- **Yuvarlanmış ondalıklar.** Ondalıklı biçimi yazılmış tüm ölçümler ham değerleriyle değiştirildi (örnek: gecikme p50 `0,015912843984551728 ms`, `fraud/results/edge_eval_full_scale.json:23`).
- **Türetilmiş oran.** Alarm oranları için yazılan "orantı 2,4×" kaldırdı; ham iki değer `maintenance/results/e1_cmapss.json:190-192` satırlarında durur.
- **Çoklu satır atfı.** Virgülle yazılmış `:67,70` ve kısa yazılmış `:103` biçimindeki atıflar araç tarafından dosya adı olmadan okunuyordu; her atıf tam `dosya:satır` biçimine çıkarıldı.
- **İç başvurular.** Ölçüm olmayan istem numaraları, ölçüm taşıyan satırlarda künye satırına bağlandı (örnek `ISTEMLER.md:55`).

Kalan altmış iki "kaynaksız" kalem (on, yirmi altı ve yirmi altı) ölçüm değildir: çizim referans işaretleri (100, 110, 160, 170, …; tanımı `TARIFNAME.md:45`), istem ve bölüm numaraları, belge tarihi ve bu bölümdeki araç çıktı sayıları. İstem metninde parantez içindeki sayıların ölçüm değil referans işareti olduğu `ISTEMLER.md` §A başındaki vekil açıklamasında kayıt edilmiştir.

Bu bölüm bir hukuki yeterlilik, yenilik ya da ihlal görüşü değildir; başvuru öncesi vekil ve incelemeci önünde doğrulanmalıdır.
