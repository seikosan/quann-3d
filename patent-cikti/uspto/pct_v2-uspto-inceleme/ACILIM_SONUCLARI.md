# AÇILIM SONUÇLARI — PCT v2 tarifname adayı (14 paragraf) ve ölçülen etki

*Durum: TASLAK / ölçüm raporu. Bu dosya hukuki hüküm kurmaz; "yenidir, patentlenebilir,
112(a) sağlandı" demez. Araç çıktıları bulgu listesi olarak aktarılır; nihai değerlendirme
patent vekili + incelemeci işidir.*
*Oturum tarihi: 2026-11-20. Önceki rapor: `INCELEME_RAPORU.md` (önce durum, aynı araçlar).*

---

## 1. Ne yaptık

Planlanan 14 açıklama paragrafını (`ACILIM_PLANI.md` kartları) İngilizce olarak yazıp,
orijinal tarifnameye **dokunmadan** bir aday kopya ürettik: `DESCRIPTION_EN_ADAY.md`.
Ekleme ve paragraf numaralandırması elle değil, tek seferde koşan deterministik bir Python
scriptiyle yapıldı. Sonra aynı denetim araçlarını (`patent-creator` MCP sunucusu, yerel) hem
orijinal hem aday üzerinde yeniden koşup, "yazılı tanım" bulgu sayısındaki değişimi ölçtük.

---

## 2. Sonuçlar

| Ne | Değer | Anlamı |
|---|---|---|
| Yazılan paragraf | **14 adet / 2 087 kelime** (`insert_and_renumber.py` paragraf kelime toplamı; bkz. §3.2) | Plan kartlarının tamamı karşılandı |
| Aday paragraf sayısı | **84** (orijinal 70) — etiketler `0001`–`0084` kesintisiz | Yeniden numaralandırma doğrulandı |
| Kelime artışı (tarifname) | 6 049 → 8 398 kelime = **+2 349** (`wc -w`, §3.1) | +2 087 gövde + 222 künye yorumu + başlık yorumu/boş satır |
| `review_specification` — ORİJİNAL (bu oturum taze) | **47 IMPORTANT** / 0 CRITICAL / 0 enablement / kapsam 15-15 (`mcp_spec_before.json`) | `INCELEME_RAPORU.md`'deki 47 ile birebir aynı → ölçüm karşılaştırılabilir |
| `review_specification` — ADAY (biçimsizleştirilmiş kopya) | **36 IMPORTANT** / 0 CRITICAL / 0 enablement / kapsam 15-15 (`mcp_spec_after_neutral.json`) | **47 → 36 = −11 bulgu (%23 azalma)** |
| Giden bulgular | **15** (bkz. §3.4) — istem 3, 4, 8 tamamen temiz; 10 ve 13 yarıya indi | En çok fayda: öğe tanımı + sistem çerçevesi paragrafları |
| Kalan bulgular | **36** (bkz. §3.5) | Tamamı "only Para. N" (tek paragraf) kalıbı; **yeni bir CRITICAL yok** |
| ADAY'da sayıca ilk kez beliren öğeler | 4 (`task output from the set representation` ×3, `context encoder` ×1) | Araç öğe-çıkarımının metne bağlı etiket farkı; metin aynı öğeyi zaten anlatıyor (§3.5 notu) |
| `review_patent_claims` (istemler değişmedi) | `compliance_score = 100.0`, 0 bulgu, 15 istem; kapsam `checks_run_only`; atlanan: `antecedent_basis` (`mcp_claims_aday.json`) | Skor yalnız **koşan** kontrollerin skoru; hukuki uygunluk değil |
| `check_formalities` (ADAY) | 2 CRITICAL: `Missing required section: SUMMARY` / `DETAILED DESCRIPTION`; atlanan: `figure_numeral_coverage` (`mcp_form_aday2.json`) | Orijinalde de **aynı 2 CRITICAL** (`mcp_form_orig2.json`) → **fark yok**; bölüm adları bilerek değiştirilmedi |
| Orijinal dosyalara dokunma | `DESCRIPTION_EN.md` sha1 `507561bebdf1d59eb90d3c76920209bc42f888c3`, mtime `2026-11-19T11:46`; `CLAIMS_EN.md` sha1 `064616b77e12618096c9fc96390cf41c7aa6508f` | Oturum başındaki değerlerle aynı → salt-okur kuralına uyuldu |

**Ölçüm uyarısı (önemli, gizlenmiyor):** `review_specification` paragrafları `[NNNN]` köşeli
etiketiyle böldüğü için (`specification_analyzer.py:130`), **işaretli** aday dosya araçta
bozuk okundu: 84 paragraf bulundu ama kapsam 2/15'e düştü ve 3 CRITICAL "not found in
specification" üretildi (`mcp_spec_after.json`). Bu üç CRITICAL gerçek eksiklik değil:
"output interface" adayı metinde 4 satırda, "multi-sentence set representation" 2 satırda
geçiyor (`grep -c`, §3.1). Bu yüzden adil karşılaştırma **işaretler biçimsizleştirilmiş**
kopyalarla yapıldı (§3.3). İşaretli aday dosyasının kendisi başvuru metni olarak tasarlandı;
bozuk okuma aracın ayrıştırma sınırıdır.

---

## 3. Teknik yanıtlar

### 3.1 Üretilen dosyalar ve dokunmama kanıtı

| Dosya | Ne | Kanıt |
|---|---|---|
| `docs/patent/pct_v2/ACILIM_METINLERI_EN.md` | 14 paragrafın taslağı + her paragrafa tek satır künye (hedef konum, grup/istem, OLC/RESEARCH kaynakları) | 19 181 B, 2026-11-20 00:37 (`ls -la`) |
| `docs/patent/pct_v2/DESCRIPTION_EN_ADAY.md` | Orijinal + 14 paragraf + yeniden numaralandırma; başlıklar aynen 6 adet | 55 210 B, 282 satır, 8 398 kelime |
| `patent-cikti/uspto/pct_v2-uspto-inceleme/mcp_ham/` | 8 ham araç çıktısı (JSON) + `insert_and_renumber.py` kopyası | `ls -la` (bu klasörde, 2026-11-20 01:46) |

```
$ ls -la docs/patent/pct_v2/DESCRIPTION_EN.md docs/patent/pct_v2/CLAIMS_EN.md
-rw-r--r--  1 serkankose  staff  11426 Nov 19 11:46 docs/patent/pct_v2/CLAIMS_EN.md
-rw-r--r--  1 serkankose  staff  39987 Nov 19 11:46 docs/patent/pct_v2/DESCRIPTION_EN.md
$ shasum ...
507561bebdf1d59eb90d3c76920209bc42f888c3  docs/patent/pct_v2/DESCRIPTION_EN.md
064616b77e12618096c9fc96390cf41c7aa6508f  docs/patent/pct_v2/CLAIMS_EN.md
$ wc -w -l docs/patent/pct_v2/DESCRIPTION_EN.md docs/patent/pct_v2/DESCRIPTION_EN_ADAY.md
     225    6049   DESCRIPTION_EN.md
     282    8398   DESCRIPTION_EN_ADAY.md
```

Araç notu: `git status --porcelain docs/patent/pct_v2/` çıktısı `?? docs/patent/pct_v2/` —
yani klasör git tarafından izlenmiyor; bu yüzden dokunmama kanıtı sha1 + mtime ile verildi.

### 3.2 Script doğrulaması (canlı çıktı, `insert_and_renumber.py`)

```
toplam paragraf: 84
eklenen blok sayısı: 14
=== DOĞRULAMA: etiketler 0001'den kesintisiz/artan mı ===
sayı: 84 | 0001 başlangıç: True | kesintisiz-artan: True
yinelenen yok: True
=== metin-içi tek çapraz atıf ===  paragraph [0065] → paragraph [0079]  (grep, 2 eşleşme)
BAŞLIKLAR AYNI ✓      (her iki dosyada 6 `## ` başlığı, diff boş)
```

Ekleme haritası ve yeni numaralar (script + dosya taraması ile ölçüldü; kelime sayıları
paragraf gövdesi):

| Grup | Yeni no | Kelime | Ne anlatıyor |
|---|---|---|---|
| G1 | [0032] | 147 | task input / ordered token sequence / tabular record tanımı |
| G1 | [0033] | 168 | input order, permütasyon-bağımsızlığı, point cloud, D-boyutlu nokta |
| G2 | [0034] | 150 | context-encoded representation; 3D+2; tek kodlayıcı alternatifi; 20999648 parametre ( zincirin tamamı) |
| G4 | [0037] | 116 | Öklid normu aritmetiği = çift mesafesi; beş bileşenin anlamı |
| G3 | [0039] | 153 | max/average-pooling sonuçları, bulut merkezi = nokta konumlarının ortalaması |
| G5 | [0041] | 123 | context length = görev düzeninin kabul ettiği girdi birimi sayısı |
| G7 | [0043] | 164 | reader'ın okuma adımları; iki görevde tek yapısal okuyucu |
| G7 | [0044] | 142 | bir sonraki token olasılık dağılımı; class assignment score + karar sınırı |
| G7 | [0045] | 110 | INT8 self-test ve p50 gecikme değerleri (reader başlığı altında toplama) |
| G9 | [0047] | 192 | cümle → pencere adımlı hiyerarşi; stride < pencere ⇒ örtüşme; multi-sentence temsil |
| G6 | [0057] | 134 | emerging region işaretinin kayıttaki üçlü yapısı; eşik değerleri kaydedilmedi |
| G11 | [0062] | 181 | dağıtım deposunun fiziksel tek-depo olması; okuma sırası; sistem–modül eşlemesi |
| G10 | [0066] | 181 | gradyan tabanlı geri yayılım tanımı + gradyansız döngünün adım adımı anlatımı |
| G10 | [0067] | 126 | gradyansız eğitimin ikinci çalışma örneği (R8) — sınır cümleleriyle |

Sayı disiplini: yeni paragraflardaki her sayı ya tarifnamede zaten yazılıydı
(10000 · 3D+2 · 20999648 · K=13 · 221 · 214 · 5/2 · 8 alan · seq 10 · seq 8 · 19963 ·
1.2814998626708984e-06 · 26473.303851652894 · 0.015912843984551728 ms · 2000 — hepsi mevcut
paragraflara atıfla bağlandı), ya yalnız **bir** yeni kaynaktan geldi: [0067]'deki
**5233.4086 / 86080 / 19770.644** değerleri `RESEARCH.md:23` ve `RESEARCH.md:107` künyeli.
Bu paragrafta sınır aynen yazıldı: *tek seed, çift tekrar koşumu yok, tekrar düşünülmekte;
bu nedenle hiçbir iyileşme/iddia edilmez.* Plan "kaydedilmemiş" dediği yerlerde (maske
değeri, eşikler, eps, aralık genişliği, okuyucu katman sayıları, donanım tanımlaması) metin
aralık vermiyor, "kaydedilmedi" diyor.

### 3.3 Ölçüm protokolü ve dört koşu

Araç kodu: paragraf bölme `re.compile(r"\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)")`
(`specification_analyzer.py:130`). Bu yüzden üç kopya türü koşulları:

| Kopya | Nasıl üretildi | Ne ölçer |
|---|---|---|
| işaretli | dosya olduğu gibi (satır-başı ve metin-içi `[00NN]`) | başvuru metninin gerçek biçimi — ama araçta ayrıştırma bozuk |
| neutral | satır-başı etiketler kalır, **metin-içi** `[00NN]` → düz sayı | eklenen içeriğin etkini ayırmak için (esasa en yakın) |
| untagged | tüm `[00NN]` işaretleri sökülür | biçim etkisini sıfırlayan kontrol |

| Koşu | wd bulgu | en | CRITICAL | paragraf | terim | kapsam | özet (araç, verbatim) |
|---|---|---|---|---|---|---|---|
| ORİJİNAL işaretli | **47** | 0 | 0 | 70 | 1305 | 15/15 | `[WARNING] Found 47 IMPORTANT issues in specification support` |
| ORİJİNAL untagged | 50 | 0 | 0 | 113 | 1310 | 15/15 | `[WARNING] Found 50 IMPORTANT issues…` |
| ADAY işaretli | 55 | 0 | **3** | 84 | 1740 | **2/15** | `[WARNING] Found 3 CRITICAL issues, 52 IMPORTANT issues…` (toplam 55 = 3 CRITICAL + 52 IMPORTANT, hepsi `written_description`) |
| **ADAY neutral** | **36** | 0 | 0 | 84 | 1723 | 15/15 | `[WARNING] Found 36 IMPORTANT issues in specification support` |
| ADAY untagged | 36 | 0 | 0 | 142 | 1753 | 15/15 | `[WARNING] Found 36 IMPORTANT issues…` |

Adil okuma: **aynı biçimde** karşılaştırınca — işaretli↔işaretli güvenilir değil
(ayrıştırma bozuk); untagged↔untagged **50 → 36 (−14)**; neutral aday **36**. Yani içerik
etkisi yönü iki bağımsız biçimde de aynı: **bulgu sayısı düşüyor, CRITICAL/enablement
artmıyor, istem kapsamı 15/15 kalıyor.**

### 3.4 ADAY'da giden 15 bulgu (`mcp_spec_before.json` − `mcp_spec_after_neutral.json`)

| İstem | Öğe | Önceki araç problemi | İlacı |
|---|---|---|---|
| 1 | class assignment score | only Para. 36 | [0044] |
| 1 | input order of the points | only Para. 10 | [0033] |
| 1 | probability distribution of a next | only Para. 36 | [0044] |
| 3 | Euclidean norm of the difference vector | only Para. 33 | [0037] |
| 4 | result of the average-pooling | only Para. 34 | [0039] |
| 4 | result of the max-pooling | only Para. 34 | [0039] |
| 8 | emerging region | only Para. 13 | [0057] |
| 10 | fixed stride | only Para. 37 | [0047] |
| 10 | multi-sentence set representation | only Para. 37 | [0047] |
| 10 | sentence cloud state | only Para. 65 | [0047] |
| 13 | at least one processor | only Para. 16 | [0062] |
| 13 | computer system | only Para. 16 | [0062] |
| 13 | input order of the points | only Para. 10 | [0033] |
| 13 | instructions configures the system | only Para. 16 | [0062] |
| 15 | input order | only Para. 10 | [0033] |

İstem bazında değişim (aynı iki dosyadan): 1: 11→9 · 3: 1→**0** · 4: 2→**0** · 5: 1→1 ·
6: 1→1 · 8: 1→**0** · 9: 1→1 · 10: 6→3 · 11: 2→2 · 12: 3→3 · 13: 9→6 · 14: 2→2 · 15: 7→8.

### 3.5 Kalan 36 bulgu ve mekanizma neden

Kalan 36'nın tamamı aynı kalıp: `Claim element "X" has limited description (only Para. N)`.
Tam liste: istem 1 (9), 5 (1), 6 (1), 9 (1), 10 (3), 11 (2), 12 (3), 13 (6), 14 (2), 15 (8).

Bu bulguların bir kısmı, eklenen paragrafların **kelimesi kelimesine** tanımladığı öğeler
için hâlâ duruyor. Bunun sebebi metin eksikliği değil, aracın arama sırası; kod satırlarıyla:

1. `_extract_technical_terms` (`specification_analyzer.py:174`) terimleri ~6 kelimelik
   kayan diziler olarak çıkarıyor. Ölçülen örnek — benim `[0032]` paragrafımdan çıkan
   terimler: `the task input is defined by the`, `structure it presents to the`,
   `context encoder` … Yani düz `task input` terimi bu paragrafta **üretilmiyor**;
   orijinal `[0029]`da ise `a task input,` biçiminde noktalama ile sınırlandığı için
   düz `task input` terimi üretiliyor.
2. `_find_specification_support` (`:339`) önce terim indeksine bakıyor ve **indekste bulunan
   öğe için yalnız o paragrafları döndürüyor** (`:344`); alt-dize araması bu durumda hiç
   çalışmıyor. Ölçülen: `task input → index=[29]`, `tabular record → [10]`,
   `point cloud → [20]`, `reader → [42]`, `cloud centre → [36]`.
3. Tek paragraf dönerse eşik `elif len(support_paras) == 1:` (`:277`) IMPORTANT üretiyor.

İçeriğin gerçekten var olduğu bağımsız sayımla doğrulandı (`grep -o -i … | wc -l`,
orijinal → aday): `task input` 10→15 · `tabular record` 3→7 · `point cloud` 13→15 ·
`reader input` 2→5 · `cloud centre` 3→5 · `context encoder` 7→11 ·
`D-dimensional space` 2→4 · `context length of the task input` 2→3.
Bazı öğeler ise gerçekten tek yerde: `same trained model` 1→1, `candidate point position`
1→1, `distribution store holding` 1→1, `plurality of sentences` 2→2 — bunlar için
bundan sonraki kazanç, öğeyi ikinci bir paragrafta **ayrı cümle halinde** geçirmekle gelir.

Etiket farkı notu: aday listesinde 4 öğe ilk kez görünüyor (`task output from the set
representation` ×3, `context encoder` ×1). Sebep doğrulanmadı; tek gözlenen, aracın öğe
dizelerini istem metninden regular expression ile çıkarıp (`_extract_claim_elements`,
`specification_analyzer.py:295`, desen `:300`) metne göre eşleştirmesi. Yeni bir **CRITICAL** üretmiyorlar, IMPORTANT kalıbındalar.

### 3.6 İstem ve formalite araçlarının çıktıları (bu oturum, aday üzerinden)

`review_patent_claims` — `mcp_claims_aday.json` (istem metni değiştirilmedi):
`claim_count 15`, `independent_claims 3`, `dependent_claims 12`,
`compliance_score 100.0`, `total_issues 0`,
`checks_run = [definiteness, subjective_terms, internal_references, structure]`,
`compliance_score_scope = checks_run_only`,
`checks_skipped = antecedent_basis (opt-in: … disabled by default; PATENT_ENABLE_ANTECEDENT_CHECK=1)`.
Özet verbatim: *"[INCOMPLETE] Checks skipped: antecedent_basis (…); No issues found in the
checks that ran on 15 claims; this is not a legal compliance determination."*
Araç atıfları: MPEP §2173 (s.475), §2173.02 (s.479, 481), §2161.01 (s.411), §2181 (s.519).

`check_formalities` — `mcp_form_aday2.json` (özet metni `OZET_PCT.md`'deki ÖZET (EN) nüshası,
137 kelime; başlık 64 karakter; 8 şekil atfı):
`checks_run = [abstract, title, sections, drawing_references, abstract_title_overlap]`,
`checks_skipped = figure_numeral_coverage (no detailed description supplied)`,
`overall_compliant false`, `ready_to_file false`, 2 CRITICAL:
`Missing required section: SUMMARY`, `Missing required section: DETAILED DESCRIPTION`
(mpep alanı: `37 CFR 1.77`). Aynı iki CRITICAL orijinalde de var (`mcp_form_orig2.json`) →
**formalitede bozma yok, iyileşme de yok**. Bölüm adları (`## SUMMARY OF THE INVENTION`,
`## DETAILED DESCRIPTION OF EMBODIMENTS`) araçla eşleşmiyor; plan uyarınca metin başlıkları
**değiştirilmedi** — karar vekile ait (başlık uydurmak yerine ya araçla eşleşen ad ya da
37 CFR 1.77 biçiminin baştan kabulü).

Yerel kural-linter (`patent_lint`, model çağırmayan bağımsız hat):
orijinal `1 hata · 9 uyarı`, aday `1 hata · 10 uyarı`. Hata iki dosyada da aynı ve biçimsel:
`claim_structure: Hiç istem bulunamadı` (tek dosya okunduğu için beklenen). Uyarı farkı
+1: aday satır 106 = `[0034]` içindeki "**The only** encoder alternative recorded…" —
linter bunu *dışlayıcı ifade* sayıyor. İki seçenek vekil kararı: (a) sınır cümlesi olarak
bırakmak, (b) "The present description records one encoder alternative for the language
body: …" biçimine çevirmek (aynı olguyu korur, dışlama sözcüğünü kaldırır). Bu rapor
metni değiştirmedi; ölçümler mevcut hâli anlatıyor.

### 3.7 Koşulmayan/koşulamayan kontroller (gizlenmiyor)

- `review_patent_claims` → `antecedent_basis` **koşmadı** (varsayılan kapalı, opt-in).
  Skor 100 bu yüzden yalnız koşan 4 kontrolün skorudur (`checks_run_only`).
- `check_formalities` → `figure_numeral_coverage` **atlandı** ("no detailed description
  supplied"); şekil referans-numarası kapsaması hiç ölçülmedi (PCT tarafında referans
  işareti yokluğu zaten `UYGUNLUK_NOTU.md#D3`'te notlu).
- `review_specification` yalnız metin sezgileri çalıştırır; `enablement` 0 bulunması
  "uygulama yeterli" anlamına gelmez, araç bu yönde ince kontrolleri koşmuyor.
- Ön sanat araması bu turda **koşulmadı** (görev kapsamı dışı); Google Patents/EPO durum
  kontrolü yapılmadı, o yüzden burada arama sonucu yok.
- ADAY işaretli dosya üzerinde `review_specification` güvenilir değil (§2 uyarısı).

---

## 4. HÜKÜM

**İyi yan.** Planlandığı gibi 14 paragraf yazıldı ve aday dosya deterministik üretildi:
84 paragraf, kesintisiz `0001–0084` numaralandırma, tek çapraz atıf doğru kaydırıldı,
6 bölüm başlığına dokunulmadı, iki kaynak dosyanın sha1/mtime'ı değişmedi. Araç ölçümü
yön olarak net: aynı biçimde karşılaştırıldığında yazılı-tanım IMPORTANT bulguları
**47 → 36** (işaretli olmayan koşumda 50 → 36), istem kapsamı 15/15 kalıyor, yeni
CRITICAL ve enablement bulgusu yok; istem 3, 4, 8'de bulgu tamamen kalktı, istem 10 ve 13
yarıya indi. Formalitede bozma yok. Sayı disipleni korundu: tek yeni sayı kümesi (R8)
kaynağıyla ve "iyileştirme iddia edilmez" sınırıyla yazıldı.

**Kötü / sınırlı yan.** Kalan 36 bulgunun önemli kısmı, eklenen paragrafların tanımladığı
öğeler için hâlâ "only Para. N" diyor — ve bunun sebebi kısmen **aracın kendisi**: terim
indeksi tek paragraf döndürdüğünde alt-dize araması hiç çalışmıyor, benim paragraflarım
ise terimleri 6-kelimelik diziler olarak indeksleniyor. Yani bu sayaç, eklenen açıklamanın
hukuki değerini ölçmüyor; "sayaç 0 oldu" demek de "hâlâ 36 var, kötü" demek de ölçütü
yanlış okumak olur. Ayrıca formalitedeki 2 CRITICAL (SUMMARY / DETAILED DESCRIPTION bölüm
adları) ve linter'ın işaretlediği 1 dışlayıcı ifade olduğu gibi duruyor; `antecedent_basis`
ve `figure_numeral_coverage` hiç ölçülmedi; ön sanat araması yapılmadı.

**Sonraki adım (ölçülebilir).** (1) Kalan öğelerden gerçekten tek yerde geçenler için
(`same trained model`, `candidate point position`, `distribution store holding`) ikinci bir
cümle/paragraf; (2) bölüm adlarının 37 CFR 1.77 biçimine alınması vekil kararı;
(3) `PATENT_ENABLE_ANTECEDENT_CHECK=1` ile istem denetiminin tam kapsamla yeniden koşulması;
(4) başvuru öncesi ön sanat araması. Hiçbiri bu raporda yapılmış sayılmaz.

**Hukuk notu:** Yukarıdakiler otomatik metin sezgilerinin bulgularıdır; yenilik, buluş
basamağı veya 35 U.S.C. 112(a)/(b) / EPC 83-84 yeterliliği hakkında hüküm içermez.
**Denetçi/vekil görüşü ve onayı olmadan başvuru metni olarak kullanılmaz.**
