# DEĞİŞİKLİK KAYDI — pct_v3 TARİFNAME ve ÖZET (EN + TR)

- **Künye:** id pctv3-tarifname-01 · konu pct_v2 tarifname/özet → pct_v3 filing copy · durum taslak (vekil onayı bekliyor) · sahiplik patent-vekil · güncellendi 2026-11-20 §
- **Kapsam:** yalnız dört dosya yazıldı — `filing/DESCRIPTION_FILING-COPY_EN.txt`, `filing/TARIFNAME_FILING-COPY_TR.txt`, `filing/ABSTRACT_FILING-COPY_EN.txt`, `filing/OZET_FILING-COPY_TR.txt`. `docs/patent/pct_v2/**` salt-okunur: kanıt §9 ve §7 (sha1 künyesi `patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:29-36`).
- **Güdülen bulgu:** `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/C_us_spec_support_en.json:7` → `total_issues: 47` (`issues[]` uzunluğu da 47); aynı kayıtta `critical_issues: 0` ve `important_issues: 47` §. Bu iz 28 ayrı unsura iner (§3 eşleme). Amaç: her istem unsuruna başvuru düzeyinde (hangi modül, girdi/çıktı, sayısal sınır) dayanak paragrafı.

## 1. Taban (ölçülen, pct_v2)

| Nicelik | Değer | Kaynak |
|---|---|---|
| tarifname paragraf sayısı | 70 | `grep -c '^\[00' docs/patent/pct_v2/filing/DESCRIPTION_FILING-COPY_EN.txt` |
| tarifname satır sayısı | 155 | `wc -l` aynı dosya |
| özet kelime (EN / TR) | 137 / 106 | `wc -w` pct_v2 filing |
| tarifnmede referans işareti | yok | `grep -oE '\([0-9]{2,3}\)'` eşleşme yok (bkz. `REFERANS_NUMARALARI.md:5`) |

- **Karşılaştırma notu § (2026-11-20 akşamı, kullanıcı kararı (a)):** v2 tabanı başka bir oturumda bilerek değişti;
  aynı grep artık 84 döndürür (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama_v2kabul_2026-11-20.txt:18`); EN/TR
  kopyalar v2'de artık uyuşmuyor (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama_v2kabul_2026-11-20.txt:20`).
  Bu tablo sabahki anlık ölçümü kaydeder; pct_v3 türetmesi o sabahki tabana dayanır (§ hedef).

## 2. Ek bölüm

- **Yer:** metnin **sonuna**, `[0070]` ve SANAYİDE UYGULANABİLİRLİK sonrası (KURULUM_PLAN.md §4 "sona eklenen").
- **Eski paragraflar:** `[0001]–[0070]` numaraları, sırası ve bölüm yapısı aynen korundu; **ancak** referans işareti eklemesi nedeniyle bu aralıkta **33 satır** değişti (bkn. §9). İşaretleme gereği (ilk anışta parantezli numara) ile "metin dokunulmadan" koşulu çeliştiği için üstün tutulan: işaret zorunluluğu; anlam/dizim değişmedi, yalnız `(...)` eklemesi ve şekil betimlemelerinde numaralı yeniden kurma yapıldı.
- **Başlık:** `ADDITIONAL ENABLING DETAIL FOR CLAIMED ELEMENTS` / TR `İSTEM UNSURLARI İÇİN EK TAMAMLAYICI AYRINTI (…)`.
- **Hacim:** `[0071]–[0098]` = **28 paragraf** (hedef 28±5, üst sınır 40 — ikisi de sağlanıyor).
- **Bütünlük:** EN ve TR ayrı 98 paragraf, numara dizisi 1–98 kesintisiz ve tekil (`python3` kontrolü, §7).

## 3. 47 iz → 28 unsur → paragraf eşlemesi

| # | Unsur (iz kaydı) | pct_v2 paragraf | Yeni paragraf | İstem |
|---|---|---|---|---|
| 1 | task input | [0065] | [0071] | 1,10,13,15 |
| 2 | context-encoded representations | [0010] | [0072] | 1,13 |
| 3 | point in a D-dimensional space | [0010] | [0073] | 1,13 |
| 4 | ordered sequence of (tokens) | [0010] | [0074] | 1,13 |
| 5 | tabular record | [0010] | [0075] | 1,13 |
| 6 | point cloud | [0020] | [0076] | 1,11,12,13,15 |
| 7 | input order of the points | [0010] | [0077] | 1,13 |
| 8 | reader | [0036] | [0078] | 1,6,9,10,14,15 |
| 9 | probability distribution of a next (token) | [0036] | [0079] | 6 |
| 10 | class assignment score | [0036] | [0080] | 9 |
| 11 | same trained model | [0011] | [0081] | 1 |
| 12 | Euclidean norm of the difference vector | [0033] | [0082] | 3 |
| 13 | result of the max-pooling | [0034] | [0083] | 4 |
| 14 | result of the average-pooling | [0034] | [0084] | 4 |
| 15 | context length of the task input | [0039] | [0085] | 5 |
| 16 | emerging region | [0013] | [0086] | 8 |
| 17 | plurality of sentences | [0065] | [0087] | 10 |
| 18 | sentence cloud state | [0065] | [0088] | 10 |
| 19 | fixed stride | [0037] | [0089] | 10 |
| 20 | multi-sentence set representation | [0037] | [0090] | 10 |
| 21 | cloud centre | [0033] | [0091] | 11 |
| 22 | reader by gradient-based backpropagation | [0015] | [0092] | 12 |
| 23 | candidate point position | [0015] | [0093] | 12 |
| 24 | computer system | [0016] | [0094] | 13 |
| 25 | at least one processor | [0016] | [0095] | 13 |
| 26 | instructions configures the system | [0016] | [0096] | 13 |
| 27 | distribution store holding | [0050] | [0097] | 14 |
| 28 | input order | [0010] | [0098] | 15 |

## 4. Numara künyesi senkronu

- **Kaynak:** `docs/patent/pct_v3/DEGISIKLIK_KAYDI_SEKIL.md` (şekiller tamamlandı, tablo şekillerle birebir eşleşecek şekilde daraltıldı) + güncel `docs/patent/pct_v3/REFERANS_NUMARALARI.md`: 118 satır (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:30`), 56 tablo satırı (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:31`).
- **Kaldırılan işaretler (hiçbir dosyada yok):** 108, 114, 220, 900, 902 §. Karşılık: `D-dimensional space`, `set representation` (FIG. 3 dışı), `max-pooling`, `model input identities`, `distribution store` **işaretsiz**; `set representation` yalnız FIG. 3 bağlamında `one set representation (310)`, FIG. 1'deki modül `aggregation module (112)` §.
- **Anlam düzeltmeleri:** 302 artık `pair-geometry features` (maske sinyali işaretsiz); 214 `beş bileşenin birleştirilmesi`; 216 `çift özelliği genişliği (3D + 2)`; 304/306 havuzlama yolları; 308 birleştirme katmanı; 312 küresel küme istatistikleri varyantı §. FIG. 3'teki "cost proportional to the number of points" ve FIG. 5'teki "centre" işaretleri için **uydurma numara yok** (504 kapsamı / işaretsiz etiket) §.
- **Doğrulama:** EN tarifnamede tablo numarası kapsaması 56/56, eksik listesi boş (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:14`); EN ve TR'de tablo-dışı işaret yok; EN–TR paragraf-bazlı işaret kümeleri eşit (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:12`).

## 5. Özet değişikliği (ham `wc -w` çıktısı)

| Dosya | pct_v2 | pct_v3 | Fark |
|---|---|---|---|
| `ABSTRACT_FILING-COPY_EN.txt` | 137 | **148** | +11 (yalnız parantezli işaretler) |
| `OZET_FILING-COPY_TR.txt` | 106 | **132** | +26 (işaretler + 1:1 ayna tümce) |

- Üst sınır 150 kelime (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:16`); EN 148 kelime (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:17`), TR 132 kelime (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:18`).
- **Yeni sayı yok §:** ölçümlü tek sayı `138.2 KB` EN özette 1 kez (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:25`) ve TR özette 1 kez (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:38`) geçer; komutu `patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:37`; N, D, INT8 ve 150 sınırı gibi çerçeve ifadeleri pct_v2 ile aynı.
- Açılış biçimi korundu: EN `A computer-implemented method ...`; TR `Bilgisayarla uygulanan bir yöntem ...`.
- Eklenen işaretler (PCT Rule 47(4), tabloda geçerli numaralar): 100, 106, 116, 118, 122, 124, 126, 300, 302, 704, 706, 708, 904 §. Kaldırılmış numara kullanılmadı (§4 sayacı).

## 6. Sayı kaynak doğrulaması (uydurma sayı taraması)

- Ek bölümdeki benzersiz sayısal token: 59 (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:27`; paragraf numaraları ve `(N)` referans işaretleri çıkarıldıktan sonra). Sabit-katar kontrolü `docs/patent/pct_v2/OLCUM_KAYITLARI.md` üzerinde: kaynaksız token 0 (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:40`).
- Satır kanıtları (`docs/patent/pct_v2/OLCUM_KAYITLARI.md` satır no'ları):

    | OLCUM satırı | İçerik |
    |---|---|
    | :22 | p50 0.015912843984551728 ms |
    | :23 | p99 0.017278228622744793 ms |
    | :36 | D/d_cloud/K = 8/128/13 |
    | :37 | 100000/130/0.0013 |
    | :38 | AUC 0.9351451502337654, hatırlama 0.7307692307692307 |
    | :49-52 | 2522200, 34091488, oran 13.5, kare 14.8 (ham 14.7929) |
    | :67 | 221 vocab / 214 points / K 13 |
    | :116,119 | hafta etiketleri ve TV/KL değerleri |
    | :150 | 61.94 / 61.19 |
    | :151 | PPL protokolü n=2000, seed 0, [200000,202000), seq_len 10 |
    | :154 | 0.8/0.1/0.1 seed 42 |
    | :155-156 | 159888/19986/19986 ve sözlük 10000 |
    | :190 | 6000+500, seq_len 8, BPE 1024 |
    | :191 | 1045.03 / 563.45 / 1016.30 |

- Ölçülmemiş hiçbir değer sayı olarak yazılmadı; ortaya çıkan bölge olayı için "kaydedilmedi" deyip ölçüt düzeyinde kaldık (§2 ek bölüm [0086]).

## 7. Bilerek açık bırakılanlar (bu turda düzeltilmedi)

| Bulgu | Kaynak | Neden açık |
|---|---|---|
| `missing_sections: SUMMARY, DETAILED DESCRIPTION` (2 CRITICAL, `overall_compliant: false`, `ready_to_file: false`) | `.../motor_2026-11-20/D_us_formal_abstract_filingcopy.json` → `sections` | Başlık pratiği **ABD (MPEP 608.01)** kuralı; hedef PCT başvurusu olduğu için kullanıcı kararıyla açık. PCT Rule 5 düz başlıkları korundu. |
| `claims.total_claims: 0`, `issues: ["No claims found"]` (1 CRITICAL) | `.../K_pct_formal_en.json` → `claims` | Araç, tarifname dosyasına istem metni karıştırılmadığında istem gövdesini göremiyor: **araç girdi sınırı**, tarifname hatası değil. İstemler ayrı filing copy. |
| `checks_skipped: claims_fees` ("no numbered claims in an identifiable claims section") | `.../H_epo_formal_en.json` | Aynı girdi sınırı; EPO ücret hesabı istem sayısına bağlı olduğundan **ölçülmedi**. |
| `checks_skipped: physical`, `abstract figure designation` | `.../K_pct_formal_en.json` | Metinden doğrulanamaz; fiziki muayene ve istem formundaki şekil seçimi vekilin işi. |

## 8. Ücret ve senkron notu

- Ek 28 paragraf § ile değişen 33 satır (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:13`) sonucu tarifname 213 satıra çıktı (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:9`); bu, PCT 30 sayfa üzeri ek ücret doğurabilir — eşik kuralı §, güncel tutar DOĞRULANMADI (resmî tarifeye bakılmalı).
- **Numara künyesi senkronu:** `docs/patent/pct_v3/DEGISIKLIK_KAYDI_SEKIL.md` kaynaklı daraltma işlendi; kaldırılan numaralar filing nüshalarında 0 kez geçiyor (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:17-21`).

## 9. Doğrulama (ham)

- `grep -c '^\[00'` → EN ve TR ayrı 98 paragraf; numara dizisi kesintisiz ve tekil (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:10` EN, `patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:11` TR).
- Taban pct_v2 paragraf sayısı 70 (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:15`); ek bölüm eski bölüm sonrası kesintisiz aralıkta 28 paragraf §.
- `wc -l` → EN 213 · TR 213 satır (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:9` EN, `patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:11` TR).
- `diff` pct_v2→pct_v3 (EN): değişen 33 · yeni satır 91 (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:13`); 91'in dökümü 33 değişmiş + 58 yeni bölüm satırı § (bölüm başlığı + paragraflar + boş satırlar; TR'de aynı yapı §).
- Kapsama: EN tarifnamede tablo numarası kapsaması 56/56, eksik listesi boş (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:14`); EN=TR paragraf sayısı eşit (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:12`); EN ve TR'de tablo-dışı işaret yok §.
- **pct_v2 dokunulmazlığı:** bu depo `docs/patent/pct_v2` dizinini git izlemiyor (`git ls-files docs/patent/pct_v2 | wc -l` çıktısı 0 §) ve `git status --porcelain` boş üretilemez. Kanıt zaman damgası: yedi filing copy'nin hepsi 19 Kas 23:45 (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:43-49`); bu oturumun yazımları 20 Kas saat 10:43–10:44 sonrasıdır §. `find docs/patent/pct_v2 -type f -newermt '2026-11-20 00:00'` → 2 dosya (`patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt:50-52`: `ACILIM_METINLERI_EN.md`, `DESCRIPTION_EN_ADAY.md`); bunlar bu oturumun ilk yazımından önce değişmişti, **sebep doğrulanmadı**.

**Bu dosya ve üretilen metinler taslaktır; başvuru kararı, istem kapsamı ve resmî ücret hesabı için vekil onayı gerekir — vekil onayı gerekli.**

## 10. Linter çıktısı (aynı gün, ham)

`patent_lint docs/patent/pct_v3/filing/DESCRIPTION_FILING-COPY_EN.txt` → 1 hata · 18 uyarı · 0 bilgi (`patent-cikti/pct/pct_v3-pct-inceleme/lint_v3.txt:2`, ham kayıt aynı dizinde).

- Hata `claim_structure: Hiç istem bulunamadı` (`patent-cikti/pct/pct_v3-pct-inceleme/lint_v3.txt:6`) — tek dosya girdisinde istem gövdesi yok; §7'deki `K`/`H` bulgularıyla aynı araç girdi sınırı, tarifname hatası değil.
- Uyarı `section_missing` 7 adet (`patent-cikti/pct/pct_v3-pct-inceleme/lint_v3.txt:7-13`): technical field / background / summary / brief description / detailed description / claims / abstract — linter ABD başlık adlarını arıyor; dosyada PCT Rule 5 düz başlıkları (`TECHNICAL FIELD`, `BACKGROUND ART`, `DISCLOSURE OF THE INVENTION`, `BRIEF DESCRIPTION OF THE DRAWINGS`, `BEST MODE FOR CARRYING OUT THE INVENTION`, `INDUSTRIAL APPLICABILITY`) duruyor. §7'deki açık karar aynı: PCT hedefi için başlık değişmedi.
- Uyarı `buzzword` 11 adet (`patent-cikti/pct/pct_v3-pct-inceleme/lint_v3.txt:14-24`): `best` satır 69 ve 71 = `BEST MODE` başlığı (PCT kuralı, kasıtlı); `only` yedi satır; `critical` satır 139; `never` satır 203 — bunlar kapsamı daraltabilecek ifadeler; **vekil gözden geçirmeli**, bu turda metin değiştirilmedi (sessiz düzeltme yerine kayıt).
