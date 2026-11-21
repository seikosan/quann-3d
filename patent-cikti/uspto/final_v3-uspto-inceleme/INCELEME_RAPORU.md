# USPTO İNCELEME RAPORU — `docs/patent/final_v3/` (vekil/inceleme nüshası)

**Tarih:** 2026-11-19 · **Çerçeve:** ABD (USPTO) — 35 USC 112(a)/(b) + MPEP 608 / 37 CFR 1.77 biçim kontrolleri
**Denetlenen metin:** `docs/patent/final_v3/ISTEMLER.md` (10.510 bayt) · `TARIFNAME.md` (46.455 bayt) · `OZET.md` (2.641 bayt) — bayt sayıları `wc -c` çıktısı
**Başlık:** "Nokta bulutu temelli çift kullanımlı temsil üzerinden görev çıktısı ve gömülü açıklanabilirlik üreten yöntem, sistem ve kayıt ortamı" (`OZET.md:3`, 132 karakter)
**Çizimler:** 8 SVG (`docs/patent/final_v3/fig/fig1..8_*.svg`) + `sekiller_v3.pdf` → `drawings_present = true`

Bu dosya bir **inceleme taslağıdır**, başvuru metni değildir. Kaynak metinlerde hiçbir değişiklik yapılmadı; yalnız öneri veriliyor.
Kanıt dosyaları (ham araç çıktıları, bu klasörde): `claims_A_default.json`, `claims_B_antecedent.json`,
`spec_C_112a.json`, `formal_D_abstract_short.json`, `formal_E_abstract_full.json`, `_giris.json`,
`kanit_check_package.json` (paket denetimi ham çıktısı), `kanit_linter_ISTEMLER.txt`,
`kanit_linter_TARIFNAME.txt`, `kanit_linter_OZET.txt`, `kanit_linter_BIRLESIK.txt` (yerel linter ham
çıktıları; komut: `cd ~/Documents/patent-hatti && .venv/bin/python -m patentline lint <dosya>`).
Bağımsız doğrulama turunun günlüğü: `DOGRULAMA_GUNLUGU.md` (49 iddia satırı → 46 ✓ · 1 ✗ · 2 kanıtsız);
✗ ve kanıtsız kalemler bu nüshada düzeltildi — bkz. §10.

## Koşu biçimi (şeffaflık notu)

`review_patent_claims` / `review_specification` / `check_formalities` araçlarının gövdesi okundu
(`mcp_server/tools/analyzer_tools.py:71-121`, `:158-219`, `:253-…`): üçü de aynı üç sınıfı çağırıp
sonucu olduğu gibi geçiriyor — `ClaimsAnalyzer().analyze_claims()`,
`SpecificationAnalyzer().analyze_specification_support()`, `FormalitiesChecker().check_all_formalities()`.
Metin 59.606 bayt olduğu için (elle yeniden yazım = rakam kayması riski) analizler **aynı sınıflarla ve
dosyadan ham bayt okunarak** koşuldu (`run_analizi.py`, 5 görev paralel `ThreadPoolExecutor`).
Fark: MCP sarmalayıcısının eklediği MPEP atıf bloğu bu koşuda koşmadı; MPEP alıntıları §7'de
`search_mpep` / `get_mpep_section` ile ayrıca alındı.

---

## 0. KAPSAM UYARISI — hiçbir "0 bulgu" bu raporda "temiz" anlamına gelmez

1. **İstem/tarifname kalıpları İngilizce metin için yazılmış.** Kanıt (kod):
   `mcp_server/claims_analyzer.py:101` `\bthe\s+([a-z][a-z\s-]+?)…` ve `:106` `\bsaid\s+…` (öncül dayanak),
   `:111` `SUBJECTIVE_TERMS` listesi (subtle/minimal/significant/high/low…), `RELATIVE_TERMS`
   (about/approximately/substantially…). Türkçe istemlerde bu tokenlar geçmediği için
   `definiteness` ve `subjective_terms` kontrolleri **ateş alamaz**.
   ⇒ `total_issues = 0` sonucu metnin temizliğinin değil, kalıpların Türkçeyi görmemesinin sonucudur.
2. **Bağımlılık tespiti de İngilizce/EP kalıbında.** Motor `independent_count = 15`, `dependent_count = 0`
   verdi (`claims_A_default.json`); oysa dosyada 12 bağımlı istem var ("İstem N'e göre…" — grep sayımı 12;
   bağımsızlar 1, 13, 15). Motorun tanımı: bağımlılık, açılıştaki "The method of claim 1, wherein…" tipi
   göndermedir (`claims_analyzer.py:194-202`, 37 CFR 1.75(c)'ye atıf).
3. **Biçim kontrolü ABD başlık adlarıyla eşleştiriyor.** Türkçe karşılıkları var olduğu halde
   BACKGROUND/SUMMARY/BRIEF DESCRIPTION/DETAILED DESCRIPTION "yok" sayıldı (§4.3).
4. **Motorun kendi kapsam notları** (araç çıktısı, `claims_A_default.json` / `formal_D_abstract_short.json`):
   `compliance_score_scope = "checks_run_only"` · summary: *"No issues found in the checks that ran on 15
   claims; this is not a legal compliance determination."* · `verdict_scope`: *"Automated text and
   drawing-reference checks only; drawings_present is a declaration, not inspection of drawing files.
   This is not a filing clearance."*

---

## 1. SONUÇLAR — ne koşuldu, ne çıktı

| Ne | Değer | Anlamı |
|---|---|---|
| İstem sayısı (motor) | 15 (`claims_A_default.json: claim_count`) | Motor 15 istemi buldu; dosyadakiyle uyumlu |
| Bağımsız / bağımlı (motor) | 15 / 0 | Motor Türkçe bağımlılığı tanımadı → sayı bu pakette **geçersiz** |
| Bağımsız / bağımlı (dosya, grep) | 3 / 12 | Gerçek yapı: bağımsız 1, 13, 15 |
| İstem uyum skoru | 100,0 · `total_issues 0` | Kapsam: yalnız koşan kontroller (`checks_run_only`), hukuki hüküm değil |
| Öncül dayanak (opt-in kapalı) | atlandı | Varsayılan kapalı; ayrı koşu denendi → bkz. satır 6 |
| Öncül dayanak (opt-in AÇIK) | 0 bulgu, `checks_skipped []` | Kalıplar İngilizce ("the X", "said X") → Türkçede ateş alamaz |
| Tarifname 112(a) desteği | `spec_coverage 100 %`, 15/15, `compliant true`, 143 paragraf, 918 terim | Kelime-örtüşmesi sezgiseli; 0 issue → tarama "destek gördü", hukuken yeterlilik değil |
| Biçim: özet (yalnız `OZET.md:5`) | 138 kelime, tek paragraf → **uyumlu** | 150 kelime eşiğinin altında |
| Biçim: özet (`OZET.md:5+7+9`) | 243 kelime, 5 paragraf → **UYGUMSUZ** | 150 kelimeyi aşıyor + çok paragraflı |
| Biçim: başlık | 132 karakter → araç INFO "kısalt" | Kural tavanı 500 karakter (37 CFR 1.72(a)) → 132 yasal sınırın içinde |
| Biçim: zorunlu bölüm | 5 × CRITICAL "section missing" | ABD başlık adlarıyla eşleşme; TR karşılıkları var (§4.3) |
| Çizim atıfları | 8 şekil bulundu, `drawings_provided true`, ama "Brief Description" bölümü yok → CRITICAL | TR "Şekillerin kısa açıklaması" §4'te mevcut (`TARIFNAME.md:43`) |
| `ready_to_file` | **false** | Biçim denetleyicisinin kendi kararı |
| Paket tutarlılığı (`check_package`) | 7 belge, 0 bulgu, puan 100, `stamps []` | Belgeler arası çelişki/istey sayı uyuşmazlığı bulunmadı |
| Yerel linter — `ISTEMLER.md` | 0 hata · 6 uyarı · 1 bilgi | Uyarılar dosya-tek-olmasından; bilgi: 3 bağımsız istem |
| Yerel linter — `TARIFNAME.md` | 3 hata · 16 uyarı · 1 bilgi | 3 "hata" + 9 two-part uyarısı **artefakt** (§5) |
| Yerel linter — birleşik nüsha | 4 hata · 14 uyarı · 2 bilgi | 24 istem saydı → §12 listesi istem sanıldı (§5) |
| Ön sanat araması | **yapılmadı** | Bu komutta ön sanat adımı yok; motor anahtar durumu da sorgulanmadı |

Kanıt dosyaları: satırların ham çıktıları bu klasördeki `claims_*` / `spec_C_112a` / `formal_D/E` JSON'ları,
`kanit_check_package.json` ve `kanit_linter_*.txt` dosyalarıdır (linter komutu:
`cd ~/Documents/patent-hatti && .venv/bin/python -m patentline lint <dosya>`).

---

## 2. Araç 1 — `review_patent_claims` (35 USC 112(b) sezgiselleri)

`checks_run: definiteness · subjective_terms · internal_references · structure`
`checks_skipped: antecedent_basis` · `issues: []` (0) · özet cümle: *"[INCOMPLETE] Checks skipped:
antecedent_basis … No issues found in the checks that ran on 15 claims; this is not a legal compliance
determination."*

Açıkça belirtilmesi gereken **pozitif olmayan** gözlemler (bunlar araç bulgusu değil, dosya içeriğinden
benim notlarımdır; her biri satır kanıtlıdır):

| Gözlem | Nerede | Neden önemli |
|---|---|---|
| Motor 15 istemin 15'ini bağımsız saydı | `claims_A_default.json` vs `ISTEMLER.md:7-33` | Bağımlılık zinciri otomatik denetlenemedi: "İstem 13'e göre" (satır 31) doğru kurulsa da araç bunu görmüyor |
| İstem 1 uzunluğu 308 kelime | `ISTEMLER.md:5` (`wc -w` = 308) | Motorun uzunluk eşiği 500 kelime (`claims_analyzer.py:459`) → eşik altı, bu yüzden MINOR da üretmedi. Uzunluk tek başına kusur değildir |
| İstem 13 = 138 kelime, İstem 15 = 106 kelime | `ISTEMLER.md:29,33` | Eşik altında |
| İstem 5 içinde gerekçe cümlesi | `ISTEMLER.md:13`: "…nokta sayısının artırılmasının çift sayısının karesel artışı yoluyla hesapma yükünü artırması nedeniyle N … sınırlı tutulması" | İstem içinde "neden" anlatımı (teknik gerekçe) sınırı belirsizleştirebilir; öneri: gerekçeyi tarifnameye taşıyıp istemde "N, görev girdisinin bağlam uzunluğu ile sınırlıdır" biçiminde bırakmak. (Araç bulgusu değildir; manuel inceleme önerisi) |
| İstem 9 ve 14'te "kendi kendine yeten paket" | `ISTEMLER.md:21,31` | Terim tarifnamede tanımlı (`TARIFNAME.md:133`); istemde tanımsız işlevsel terim bırakılmadığı manuel teyit edilmeli |
| Yazım: "N noktali" (İstem 13), "hesapma yükü" (İstem 5) | `ISTEMLER.md:29`, `ISTEMLER.md:13` | "N noktalı", "hesaplama yükü" olmalı. Tarifnamede ayrıca `hesaplma`/`Hesaplma` **iki yerde**: `TARIFNAME.md:20` ve `:38` (`grep -in "hesaplma"` çıktısı = 2 satır) — başvuru nüshasında yazım hatası bırakılmamalı. (Doğrulama turunda: bu satırda daha önce ":98" de yazıyordu; satır 98 okundu, orada yazım hatası **yok** → kaldırıldı. Bkz. §10.) |

## 3. Araç 2 — `review_specification` (35 USC 112(a): yazılı tanımlama + canlandırma)

`specification_paragraphs 143` · `indexed_terms 918` · `spec_coverage {percentage 100, supported_claims 15,
total_claims 15, unsupported_claims []}` · `written_description_issues 0` · `enablement_issues 0` ·
`input_warnings []` · summary: *"[OK] Specification provides adequate support for all claims"*

Kapsam notu: bu kontrol istem terimlerinin tarifnamedeki kelime örtüşmesine bakar; **best mode
kontrol edilmiyor** (araç tanımının kendi ifadesi: "Best mode is not checked; manual review with inventor
input is required" — `analyzer_tools.py:159`).

Dosya içeriğinden, aracın görmediği iki manuel inceleme noktası:

1. **Tarifnamede kaynak kodu/veri dosyası atıfları metnin içinde** — örn. `TARIFNAME.md:36`
   `(kaynak: fraud/results/edge_eval_full_scale.json:15-16)`, `:62` `(kaynak: cloudlm/model.py:47)`,
   `:200` `(kaynak: purecloudlm/model.py:8-9)`. Başvuru metninde bu künye satırlarının kalıp kalmayacağı
   vekil kararı; ölçümlerin **tablolaştırılıp** dipnota/alana taşınması pratik bir alternatif.
2. **Tarifnamede "çalışma belgesi" türü ifadeler ve tarih notları** — örn. `TARIFNAME.md:74`
   "'D + 4' sayısal etiketi çizim başlığından 2026-11-18 tarihinde çıkarılmıştır", `:220` "Çelişki
   GİDERİLMİŞTİR (2026-11-18)". Bunlar inceleme izidir; §12 başlığının tamamı
   (`TARIFNAME.md:216-231`) başvuru nüshasından çıkarılması aday bölümdür. Ayrıca bu bölüm **numaralı
   liste** içerdiği için linter tarafından istem kümesi sanıldı (§5).

## 4. Araç 3 — `check_formalities` (MPEP 608 / 37 CFR) — iki özet varyantı

Başvuru paketi üç ayrı dosya olduğundan "özet" metninin sınırı belirsizdi; ikisi de koşuldu.

**Varyant D — özet = `OZET.md:5` (anlati paragrafı, 138 kelime):** 6 bulgu.

| Şiddet | Nerede | Ne (araç metni) | Aracın önerisi | Atıf |
|---|---|---|---|---|
| CRITICAL | specification | Missing required section: BACKGROUND | BACKGROUND bölümü ekle | 37 CFR 1.77 |
| CRITICAL | specification | Missing required section: SUMMARY | SUMMARY bölümü ekle | 37 CFR 1.77 |
| CRITICAL | specification | Missing required section: BRIEF DESCRIPTION OF DRAWINGS | bölüm ekle | 37 CFR 1.77 |
| CRITICAL | specification | Missing required section: DETAILED DESCRIPTION | bölüm ekle | 37 CFR 1.77 |
| CRITICAL | drawings | Figures referenced but "Brief Description of Drawings" section missing (8 şekil bulundu: 1-8) | şekil listesi olan bölüm ekle | 37 CFR 1.77(b)(3) |
| INFO | title | Title is longer than recommended — 132 karakter, "<= 100 characters (recommended)" | 32 karakter kısalt | MPEP 606 |

`abstract: {word_count 138, line_count 1, compliant true}` · `compliance_summary.ready_to_file false` ·
`overall_compliant false`.

**Varyant E — özet = `OZET.md:5+7+9` (ölçüm paragrafı dâhil, 243 kelime):** yukarıdaki 6 bulguya ek 2 bulgu:

| Şiddet | Nerede | Ne | Öneri | Atıf |
|---|---|---|---|---|
| WARNING | abstract | "Abstract exceeds the 150-word limit" — 243 kelime | 150 kelimeye indir | 37 CFR 1.72(b); MPEP 608.01(b) |
| INFO | abstract | "Abstract should be a single paragraph" — çok paragraflı | tek paragrafa birleştir | MPEP 608.01(b) |

**Önemli nüans (kural aracın şiddetinden farklı):** 37 CFR 1.77(b) ve MPEP 608.01(a) metni bölümler için
"should include … in order" / "preferred layout … suggested for the applicant's use" diyor (§7'de alıntı);
aracın "Must include" ve CRITICAL derecesi aracın kendi eşik tercihidir. Başlık için ise ölçülen kural
**500 karakter** — 132 karakter bu tavanın içinde; aracın 100-karakter eşiği kural değil öneridir.

## 5. Yerel linter (`patent_lint`, model çağırmayan kural denetimi) — bulgular ve artefaktlar

| Koşu | Özet | Yorum |
|---|---|---|
| `ISTEMLER.md` | 0 hata · 6 uyarı (`section_missing` ×6) · 1 bilgi (3 bağımsız istem) | 6 uyarı, istem dosyasının tek başına lint edilmesinden; paket tek dosyada değil |
| `TARIFNAME.md` | 3 hata · 16 uyarı · 1 bilgi | Aşağıdaki 13 bulgu **artefakt** |
| `OZET.md` | 1 hata (`claim_structure: hiç istem bulunamadı`) · 6 uyarı | Özet dosyasında istem araması artefaktı |
| `birlesik_inceleme.md` (OZET→TARIFNAME→İSTEMLER, bu inceleme için üretildi) | 4 hata · 14 uyarı · 2 bilgi (`claim_count 24`, `claim_unity 12 bağımsız`) | İstem sayısı 24 değil 15; §12 listesi istem sayılıyor |

**Artefakt kanıtı (üç adımda doğrulandı):**
1. `grep -c "Şekil" docs/patent/final_v3/ISTEMLER.md` = **0** → gerçek istemlerde şekil atfı yok;
   linterin "İstem 1 / İstem 9 şekle atıf yapıyor" hataları gerçek istemlere ait değil.
2. Kuralın kendisi: `patentline/linter.py:292` — `(şekil|figure|fig\.)\s*\d` ifadesini "istem gövdesi"nde
   arıyor; linter taranan dosyada numaralı satırları istem sayıyor.
3. `TARIFNAME.md` §12 numaralı listesi 9 madde (satır 220-231, `awk` sayımı = 9). Madde 1 (satır 220)
   "Şekil 2", madde 9 (satır 231) "Şekil 6 ve Şekil 7" içeriyor → "claim_figure_ref istem 1 / istem 9",
   9 × `two_part_form`, `claim_unity 9 bağımsız` bulguları buradan geliyor. `buzzword "istem 220"` yine
   satır 220'deki "yalnızca" kelimesi.

**Artefakt olmayan, elle bakılmaya değer linter bulguları:**
- `buzzword "yalnızca"` — tarifnamede 6 yerde: `TARIFNAME.md:74, 80, 111, 182, 193, 220`
  (`ISTEMLER.md`'de 0 kez; grep sayımı). İstemlerde geçmediği için kapsam daraltma riski düşük;
  tarifnamedeki kullanımlar çoğunlukla tanımlayıcı ("yalnızca geçerli çift sayısı üzerinden").
- `claim_numbering kopuk: 1 (beklenen 2)` (birleşik nüsha) — tek gerçek nedeni §12 listesinin
  numaralandırması; gerçek istem dizisi 1-15 kesintisiz (`ISTEMLER.md:5-33`).
- `claim_count 24` / `claim_unity 12 bağımsız` sayıları **yalnızca birleşik inceleme nüshası** için
  geçerli; gerçek küme 15 istem / 3 bağımsız.

---

## 6. ATLANAN / KOŞMAYAN KONTROLLER (gizlenmez)

| Kontrol | Durum | Neden (araç çıktısındaki ifade) |
|---|---|---|
| `antecedent_basis` (öncül dayanak) | **varsayılan KAPALI** → Varyant A'da atlandı | `checks_skipped.reason`: "opt-in: pattern-based antecedent checking has a high false-positive rate, so it is disabled by default"; `enable_with: PATENT_ENABLE_ANTECEDENT_CHECK=1` |
| `antecedent_basis` (opt-in AÇIK koşu) | koştu, **0 bulgu** | `claims_B_antecedent.json`: `checks_run` içinde, `checks_skipped []`. Kalıplar İngilizce (`the X` / `said X`, `claims_analyzer.py:101,106`) → Türkçe metinde doğrulama değeri sınırlı |
| `figure_numeral_coverage` (şekil referans numarası kapsamı) | **atlandı** | `checks_skipped.reason`: "no detailed description supplied" — biçim denetimine detaylı açıklama ayrı argüman olarak verilmedi |
| Best mode | **hiçbir koşulda kontrol edilmedi** | Araç tanımı: "Best mode is not checked; manual review with inventor input is required" (`analyzer_tools.py:159`) |
| İlgililik/yeterlilik (112(b)) hukuki hükmü | verilmedi | Araç: "this is not a legal compliance determination"; biçim aracı: "This is not a filing clearance" |
| Çizim dosyalarının kendisi | incelenmedi | `verdict_scope`: "drawings_present is a declaration, not inspection of drawing files" |
| EPO (EPC Art. 84/83, Rule 42-49) ve PCT biçimi | **koşulmadı** | **Oturum içi karar** (kullanıcı bu oturumda yalnız USPTO üçlüsünü seçti — kanıt dosyası yok, kanıtsız kalem olarak işaretlendi: `DOGRULAMA_GUNLUGU.md` 7b2). Motor kodunda TR/TÜRKPATENT yargı yetkisi **yok**: `patent_law_tools.py:14-18` `JURISDICTION_SOURCES = {US, EPO, PCT}`; `grep -rni "turkpatent\|\"TR\"" mcp_server --include="*.py"` çıktısı boş |
| Ön sanat araması (Google Patents / EPO / BigQuery) | **yapılmadı** | Bu komut adım listesinde ön sanat yok; motor anahtar durumları da bu oturumda sorgulanmadı → "arama yapıldı" denemez |
| Tarifnamede paragraf numaralandırması | araç kontrol etmedi | Dosya gerçekliği: `grep -c "^\[00" TARIFNAME.md` = **0** → `[0001]` tarzı numaralandırma yok (kural §7'de) |
| Türkçe dil duyarlılığı | motor desteklemiyor | §0.1-0.2 kanıtları |

## 7. KURAL ALINTILARI (yalnız bu oturumda alınan motor çıktıları)

| Konu | Kaynak (bölüm · dosya · sayfa) | Alıntı |
|---|---|---|
| Özet uzunluğu/biçimi | MPEP 608.01(b) · `mpep-0600.pdf` · s.135 (`get_mpep_section`) | "The abstract must be as concise as the disclosure permits, **preferably not exceeding 150 words** in length." ve 37 CFR 1.72(b): "A brief abstract … must commence on a separate sheet, preferably following the claims, under the heading 'Abstract' or 'Abstract of the Disclosure.'" |
| Özet tek paragraf | MPEP 608.01(a) · `mpep-0600.pdf` · s.134 (`search_mpep`) | "The abstract is a brief narrative of the disclosure as a whole … in a **single paragraph** preferably not exceeding 150 words, commencing on a separate sheet following the claims." |
| Başlık tavanı | MPEP 605.02, 606 · `mpep-0600.pdf` · s.115 | 37 CFR 1.72(a): "The title of the invention **may not exceed 500 characters** in length and must be as short and specific as possible." |
| Tarifname bölüm sırası | 37 CFR 1.77(b) · `mpep-9020-appx-r.pdf` · s.108 | "(b) The specification **should** include the following sections in order: (1) Title of the invention…" |
| Aynı kuralın MPEP yorumu | MPEP 608.01(a) · `mpep-0600.pdf` · s.133 | "The following guidelines illustrate the **preferred layout** … These guidelines are **suggested** for the applicant's use. As provided in 37 CFR 1.77(b), the specification of a utility application should include the following sections in order." |
| İstemde referans işaretleri | MPEP 608.01(m) "Form of Claims" · `mpep-0600.pdf` · s.143 | "Reference characters corresponding to elements recited in the detailed description of the drawings and used in conjunction with the recitation of the same element … **should be enclosed within parentheses** so as to avoid confusion." (İnceleme notu: linter kuralı şekle *metin* atfını yasaklıyor; işaret kullanımı parantez koşuluyla bu pasajda ayrıca düzenlenmiş.) |
| Paragraf numaralandırma | MPEP 9020 (Ek R) · `mpep-9020-appx-r.pdf` · s.73 (aynı konunun ikinci isabeti: MPEP 608.01 · s.124) | "…may be numbered at the time the application is filed, and **should be individually and consecutively numbered using Arabic numerals** … The number should consist of at least four numerals enclosed in square brackets, including leading zeros (e.g., **[0001]**)." |
| 112(b) iki gereklilik | MPEP 2103 "Patent Examination Process" · `mpep-2100.pdf` · s.9 | "35 U.S.C. 112(b) contains two separate and distinct requirements: (A) that the claim(s) set forth the subject matter the inventor … regards as the invention, and (B) that the claim(s) **particularly point out and distinctly claim** the invention." |

Not (kapsam satırlarının **doğru** atfı — doğrulama turunda düzeltildi):
- *"Indexed passages only; not a completeness or currency check."* cümlesi `search_patent_law` çıktısının
  `coverage.scope` alanındadır — kod kanıtı: `mcp_server/tools/patent_law_tools.py:163`. Aynı cümleyi
  `search_mpep` üretmiyor: `tools/mpep_tools.py` içinde `scope` alanı yok (grep çıktısı boş).
- `get_mpep_section` ayrı bir kapsam cümlesi veriyor (`mpep_tools.py:196`): "Indexed content from
  <bölüm>. … Chunk boundaries are approximate and can include adjacent text; … **This is not guaranteed
  full section text.**"

## 8. ÖNERİLEN DÜZELTMELER (metin değiştirilmedi; karar vekile ait)

1. **Özet sınırı:** başvuruya girecek özet olarak `OZET.md:5` (138 kelime) esas alınmalı; `OZET.md:7`'deki
   ölçüm paragrafı (243 kelimeye taşıyor) tarifname §3'e ait. Ölçüm kaynak künyeleri (`json:15-16` vb.)
   özetten çıkarılmalı.
2. **Başlık:** 132 karakter 37 CFR 1.72(a) tavanının (500) içinde; araç önerisi 100 karakter. Kısaltma
   kararı tercih meselesi — hukuki zorunluluk değil.
3. **Bölüm başlıkları:** ABD'ye gidecek nüsha için `TARIFNAME.md:5 / :11 / :28 / :43 / :56` başlıkları
   BACKGROUND OF THE INVENTION / SUMMARY / BRIEF DESCRIPTION OF THE DRAWINGS / DETAILED DESCRIPTION
   eşleşmesine çevrilmeli (mevcut TR yapısı korunarak iki nüsha tutmak mümkün). TÜRKPATENT nüshasında
   mevcut TR başlıkları kalabilir; bu 4 CRITICAL ABD-şablonu eşleşmesinden kaynaklanıyor.
4. **§12 numaralı listesi** (`TARIFNAME.md:216-231`) başvuru nüshasından çıkarılmalı veya madde
   başlıkları numaralı listeden düz paragrafa çevrilmeli: otomatik araçlar bu listeyi istem kümesi sayıyor.
5. **Paragraf numaralandırma:** tarifnameye `[0001]…` numaralandırma eklenmesi (MPEP 608.01, s.124) hem
   otomatik araçlar hem inceleme yazışması için faydalı; şu an yok (`grep` = 0).
6. **İstem içi gerekçe cümleleri** (örn. `ISTEMLER.md:13` "…nedeniyle…") tarifnameye taşınmalı.
7. **Yazım:** "N noktali" → "N noktalı" (`ISTEMLER.md:29`), "hesapma" → "hesaplama" (`ISTEMLER.md:13`),
   "hesaplma"/"Hesaplma" → "hesaplama" (`TARIFNAME.md:20` ve `:38`; `grep -in "hesaplma"` = 2 satır).
8. **Sonraki denetim adımları:** (a) İngilizce nüsha üzerinden aynı üçlü (`dil/` veya `evrensel/…_EN.md`)
   — motor kalıpları ancak İngilizce metinde gerçek sinyal üretir; (b) aynı final_v3 için **EPO üçlüsü**
   (başlık/bölüm eşleşmesi TR pratiğine daha yakın); (c) opt-in öncül dayanak İngilizce nüshada.

## 8a. BAĞIMSIZ DOĞRULAMA (raporu yazandan ayrı tur)

Bu nüsha, `reviewer` alt-ajanının 49 iddia satırını ham kaynağa karşı işaretlediği turdan **sonra**
güncellendi. Günlük: `DOGRULAMA_GUNLUGU.md`. Sonuç: **46 ✓ · 1 ✗ · 2 kanıtsız**.

| Kalem | Durum | Bu nüshada yapılan |
|---|---|---|
| "hesaplma" yazım hatası `TARIFNAME.md:20, 38, 98` | ✗ | Satır 98 okundu: hata yok. İki yer bırakıldı (`:20`, `:38`) + grep kanıtı eklendi (§2, §8.7) |
| §6'daki "yalnız USPTO koşuldu" gerekçesi | kanıtsız | "oturum içi karar" olarak işaretlendi; kanıt dosyası olmadığı yazıldı. Motor kodunda TR yargı yetkisi olmadığı ayrıca kod kanıtına bağlandı (`patent_law_tools.py:14-18`) (§6) |
| "kapsam satırı `search_mpep` çıktısında" atfı | kanıtsız | Atıf düzeltildi: cümle `search_patent_law`'ın (`patent_law_tools.py:163`); `search_mpep`'te scope alanı yok; `get_mpep_section` ayrı bir kapsam cümlesi veriyor (`mpep_tools.py:196`) (§7 notu) |
| `check_package` ve linter sayıları | doğrulandı ama ham çıktı klasörde yoktu | Ham çıktılar klasöre yazıldı: `kanit_check_package.json`, `kanit_linter_{ISTEMLER,TARIFNAME,OZET,BIRLESIK}.txt`; değerler birebir aynı (7 belge / 0 bulgu / 100,0 / `stamps []`; 0-6-1, 3-16-1, 1-6, 4-14-2) |

Doğrulayıcının **değiştirmeye gerek görmediği** ana bulgular (motor çıktıları, biçim bulguları, dosya
gerçekleri, kod satır atıfları, MPEP sayfa atıfları, TR bölüm başlığı eşleşmesi, "yapılmadı/atlandı"
tutarlılığı) ham kaynakta birebir karşılandı. Rapor metninde hukuki hüküm yok; §9'daki çerçeve geçerli.

## 9. HÜKÜM

**İyi yan:** Paket kendi içinde tutarlı çıkıyor (`check_package`: 7 belge, 0 bulgu, puan 100); özet
anlati paragrafı biçimce ABD ölçütünde (138 kelime, tek paragraf); 15 istem / 3 bağımlı-olmayan yapı
dosyada doğru kurulmuş; tarifname 112(a) sezgisel taramasında 15/15 istem için destek görüyor ve Ölçüm
künyeleri (ölçülmüş teknik etki + sınırların açıkça yazılması) vekil için alışılmadık derecede denetlenebilir.

**Kötü / eksik yan:** Bu koşu **ABD ölçütlerinde bir temizlik kanıtı üretmedi.** Üç içerik analizinin
0 bulgu vermesi, motorun kalıplarının İngilizce olması nedeniyle beklenen sonuçtur (§0.1); bağımlılık
sayımı bile hatalı çıktı (15/0). Biçim denetimi `ready_to_file = false` verdi ve 5 CRITICAL üretti — bu
CRITICAL'lerin dördü bölüm adı eşleşmesi (TR karşılıkları mevcut), biri şekil-bölümü eşleşmesi; ancak
özetin 243 kelimeye taşma riski gerçek ve tek satırlık bir kararla giderilebilir. Öncül dayanak, şekil
numarası kapsamı, best mode ve ön sanat hiç denenmedi; şekil dosyaları incelenmedi.

**Çerçeve:** Bu rapor, yalnız motorun koştugu sezgisel kontrollerin ve yerel kural linterinin çıktısıdır;
yenilik/buluş basamağı/ihlal değerlendirmesi içermez ve hukuki sonuç içermez. Türkçe bir TÜRKPATENT
paketinin ABD araçlarıyla taranması **kapsam dışı bir çerçeve** taşır; İngilizce nüsha ve/veya EPO üçlüsü
koşulmadan "ABD'ye hazır" denemez. **Vekil onayı gerekli.**


