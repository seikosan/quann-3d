# INCELEME_RAPORU_2026-11-20 — PCT v2 gönderim nüshaları: ABD (USPTO) odaklı otomatik denetimler

- **Girdi:** `docs/patent/pct_v2/filing/` gönderim nüshaları (EN asıl + TR eş kopya). Kaynak defterler
  (`docs/patent/pct_v2/*.md`) bu turda **değiştirilmedi**; hiçbir bulgu bu raporda düzeltilmedi.
- **Motor:** `patent-creator` MCP sunucusu. Koşum künyesi, girdi sha256'ları ve ham kayıt dosyaları:
  `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/MOTOR_KAYITLARI_2026-11-20.md`.
  Sayı künyesi (ham kayıttan üretilen tablo): aynı dizinde `METRIKLER_2026-11-20.md`.
- **Kapsam:** otomatik metin sezgileri; hukuki hüküm değildir. Kapanışta vekil onayı zorunludur.

## 1. Ne yaptık

Gönderime hazırladığımız İngilizce istem, tarifname ve özet metnini olduğu gibi (bayt-bayt, elle yazmadan)
üç ABD denetiminden geçirdik: istem netliği (35 USC 112(b) sezgileri), tarifname desteği (35 USC 112(a)
yazılı-tanım/uygulama sezgileri) ve MPEP 608 / 37 CFR 1.77 biç denetimi. Ardından aynı metni EPO ve PCT
biç kontrollerinden, paket bütünlüğü denetiminden ve Türkçe nüshanın EPO kontrollerinden geçirdik; bir de
istem denetiminin **varsayılan kapalı** olan öncül-dayanak (antecedent basis) kontrolünü elle açtık.

## 2. Sonuçlar

| Ne | Değer (araç çıktısı) | Anlam |
|---|---|---|
| ABD istem denetimi — skor | `compliance_score 100` · `0 kritik / 0 önemli / 0 minör` (kayıt `A`) | Koşan 4 kontrolde bulgu yok; **ama bir kontrol hiç koşmadı** (§4) |
| ABD istem sayacı | 15 istem · 3 bağımsız · 12 bağımlı (kayıt `A`) | Taslağın kendi künyesiyle tutarlı |
| ABD tarifname desteği | `47` bulgu · `0 kritik` · `47 önemli` · `enablement 0` · kapsama `100%` (kayıt `C`) | Hepsi aynı kalıp: "istem öğesi tek paragraftan türüyor" |
| ABD biç denetimi | `overall_compliant false` · `ready_to_file false` · **2 KRİTİK** (kayıt `D`) | ABD başlıkları SUMMARY ve DETAILED DESCRIPTION bulunamadı |
| Özet / başlık / şekiller | 137 kelime · 64 karakter · FIG. 1–8 (kayıt `D`) | Üçü de araçça uyumlu |
| Öncül-dayanak (elle açık) | `43` bulgu · hepsi `IMPORTANT` · hepsi güven `LOW` · skor `85,67` (kayıt `B`) | Kalıp temelli, araç kendisi "manuel doğrula" diyor |
| EPO istem | `1 önemli` · `0 kritik` (kayıt `E`) | Kural 43(2) EPC: birden çok bağımsız istem |
| EPO tarifname | `47 önemli` (kayıt `F`) — ABD `C` kaydıyla birebir aynı 47 öğe | EPC 83. madde yeterliliği açısından da aynı iz |
| EPO biç | `1 önemli` (kayıt `H`) | Kural 47(4) EPC: özetde parantez içinde şekil numaraları yok |
| PCT biç | `1 kritik` (kayıt `K`) | Araç "istem yok" diyor; **araç istem girdisi almıyor** — §3.4'te açıklanıyor |
| Paket bütünlüğü | 6 dosya tarandı · `0 bulgu` · skor `100` (kayıt `L`) | Dosyalar arası tutarlılık denetimi temiz |
| TR nüsha EPO | istemde `16 önemli` (kayıt `E-TR`) · tarifnamede `0` (kayıt `F-TR`) | TR sayacı 15 bağımsız/0 bağımlı okudu — §3.5'te araç sınırı |
| Dün → bugün karşılaştırma | 47 ↔ 47 ve 2 KRİTİK ↔ 2 KRİTİK | Dünün bulguları duruyor; yeni olan yalnız B/E/F/H/K/L/TR satırları |

## 3. Teknik yanıtlar (bulgular araç çıktısından olduğu gibi)

### 3.1 `review_patent_claims` — 35 USC 112(b) sezgileri (kayıt `A`)

Araç özeti (verbatim): «[INCOMPLETE] Checks skipped: antecedent_basis (opt-in: pattern-based antecedent checking has a high false-positive rate, so it is disabled by default); enable with PATENT_ENABLE_ANTECEDENT_CHECK=1 (LOW confidence; manual review required). No issues found in the checks that ran on 15 claims; this is not a legal compliance determination.»

- Koşan kontroller (`checks_run`): `definiteness` · `subjective_terms` · `internal_references` · `structure`
- `compliance_score_scope`: `checks_run_only` — skor yalnız koşan kontrollerin skorudur.
- Araç MPEP atıfları: `MPEP § 2173.02` (s.481) · `MPEP § 2161.01` (s.411) · `MPEP § 2181` (s.519) · `MPEP § 2173` (s.475) · `MPEP § 2173.02` (s.479)
- Araç tanımı (verbatim): "Corresponding specification structure is not checked."

### 3.2 `review_specification` — 35 USC 112(a) yazılı tanım / uygulama (kayıt `C`)

Araç özeti (verbatim): «[WARNING] Found 47 IMPORTANT issues in specification support»

- Sayaçlar: `specification_paragraphs 70` · `indexed_terms 1269` · `written_description_issues 47` ·
  `enablement_issues 0` · `spec_coverage` 100% (15/15, `unsupported_claims: []`) · `compliant true`.
  **Not:** 47 "önemli" bulgu ile `compliant: true` alanı araç çıktısında yan yana duruyor; hangisinin esas
  olduğu denetçi/vekil değerlendirmesi ister.
- Araç MPEP atıfları: `MPEP § 2181` (s.522) · `MPEP § 2161.01` (s.411) · `MPEP § 2163` (s.417) · `MPEP § 2136.05(a)` (s.231) · `MPEP § 2161` (s.410)
- İz listesi mekanik olarak bağlanır: aşağıdaki satırlar `rapor_doldur.py` tarafından bu bölüm başlığında
  adı verilen kaydın `issues` dizisinden **sırayla ve elle yazılmadan** kopyalandı; alan adları
  `claim` / `element` / `spec_refs` / `severity` / `type` / `confidence` / `mpep` ve
  `legal_ref` alanlarıdır. Sayı künyesi: `METRIKLER_2026-11-20.md` tablosu. Bağımsız doğrulayıcı turda
  bu izlerin tüm kayıt için (istem, öğe, paragraf) üçlüsünün ham kayıtla birebir olduğunu kontrol etti.
- 47 bulgunun izleri (araç `claim` / `element` / `spec_refs` alanları, sırasıyla):

1. istem **1** · `task input` · [0065] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
2. istem **1** · `context-encoded representations` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
3. istem **1** · `point in a D-dimensional space` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
4. istem **1** · `ordered sequence of` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
5. istem **1** · `tabular record` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
6. istem **1** · `point cloud` · [0020] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
7. istem **1** · `input order of the points` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
8. istem **1** · `reader` · [0036] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
9. istem **1** · `probability distribution of a next` · [0036] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
10. istem **1** · `class assignment score` · [0036] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
11. istem **1** · `same trained model` · [0011] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
12. istem **3** · `Euclidean norm of the difference vector` · [0033] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
13. istem **4** · `result of the max-pooling` · [0034] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
14. istem **4** · `result of the average-pooling` · [0034] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
15. istem **5** · `context length of the task input` · [0039] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
16. istem **6** · `reader` · [0036] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
17. istem **8** · `emerging region` · [0013] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
18. istem **9** · `reader` · [0036] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
19. istem **10** · `task input` · [0065] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
20. istem **10** · `plurality of sentences` · [0065] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
21. istem **10** · `sentence cloud state` · [0065] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
22. istem **10** · `fixed stride` · [0037] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
23. istem **10** · `multi-sentence set representation` · [0037] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
24. istem **10** · `reader` · [0036] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
25. istem **11** · `point cloud` · [0020] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
26. istem **11** · `cloud centre` · [0033] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
27. istem **12** · `reader by gradient-based backpropagation` · [0015] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
28. istem **12** · `point cloud` · [0020] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
29. istem **12** · `candidate point position` · [0015] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
30. istem **13** · `computer system` · [0016] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
31. istem **13** · `at least one processor` · [0016] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
32. istem **13** · `instructions configures the system` · [0016] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
33. istem **13** · `task input` · [0065] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
34. istem **13** · `context-encoded representations` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
35. istem **13** · `ordered sequence of` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
36. istem **13** · `tabular record` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
37. istem **13** · `point cloud` · [0020] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
38. istem **13** · `input order of the points` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
39. istem **14** · `distribution store holding` · [0050] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
40. istem **14** · `reader` · [0036] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
41. istem **15** · `task input` · [0065] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
42. istem **15** · `context-encoded representations` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
43. istem **15** · `ordered sequence of` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
44. istem **15** · `tabular record` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
45. istem **15** · `point cloud` · [0020] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
46. istem **15** · `input order` · [0010] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163
47. istem **15** · `reader` · [0036] · IMPORTANT · `written_description` · güven MEDIUM · araç atfı MPEP 2163

### 3.3 `check_formalities` — MPEP 608 / 37 CFR 1.77 biç denetimi (kayıt `D`)

Araç özeti (verbatim): «[INCOMPLETE] Checks skipped: figure_numeral_coverage (no detailed description supplied). [WARNING] Found 2 CRITICAL formality issue(s)»

Bulgular (araç `issues` alanından, verbatim):

  1. `section`: specification · `severity`: CRITICAL · `problem`: Missing required section: SUMMARY · `current`: Not found · `required`: Must include all required sections · `fix`: Add "SUMMARY" section to specification · `mpep`: 37 CFR 1.77 · `confidence`: HIGH
  2. `section`: specification · `severity`: CRITICAL · `problem`: Missing required section: DETAILED DESCRIPTION · `current`: Not found · `required`: Must include all required sections · `fix`: Add "DETAILED DESCRIPTION" section to specification · `mpep`: 37 CFR 1.77 · `confidence`: HIGH

- Alt kontroller: abstract `word_count 137` (dosyanın kendi `wc -w` ölçümüyle aynı) · title
  `character_count 64` · drawings `figure_count 8` (FIG. 1–8), `drawings_provided true`.
- `found_sections`: BACKGROUND ✓ · SUMMARY ✗ · BRIEF DESCRIPTION OF DRAWINGS ✓ · DETAILED DESCRIPTION ✗
  (kayıt `D` `sections` alt alanı). Beklenen desen: tarifname PCT Rule 5 tipinde başlıklarla yazılı
  ("DISCLOSURE OF THE INVENTION", "BEST MODE FOR CARRYING OUT THE INVENTION" —
  `docs/patent/pct_v2/filing/DESCRIPTION_FILING-COPY_EN.txt:27` ve
  `docs/patent/pct_v2/filing/DESCRIPTION_FILING-COPY_EN.txt:69`); ABD denetleyicisinin regex'i
  `SUMMARY` ve `DETAILED DESCRIPTION` arıyor
  (`/Users/serkankose/patent-creator-venv/lib/python3.13/site-packages/mcp_server/formalities_checker.py:379-384`).
  İki aşamanın başlık kümesi
  farklıdır; ABD'ye girişte başlık eşlemesi yapılıp yapılmayacağı **vekil kararıdır**.

### 3.4 EPO ve PCT katmanı (kayıtlar `E`, `F`, `H`, `K`, `L`)

- `review_epo_claims` özeti (verbatim): «[WARNING] Found 1 IMPORTANT issue requiring attention under Art. 84 EPC» — tek bulgu, verbatim:

  1. `severity`: IMPORTANT · `type`: conciseness · `claim`: 1 · `location`: claim set · `term`: 3 independent product claims · `problem`: Multiple independent claims (1, 13, 15) in the same category (product). Rule 43(2) EPC generally allows only one independent claim per category. · `fix`: Consider consolidating into a single independent claim per category with dependent claims for alternatives, or justify under Rule 43(2)(a)-(c) EPC. · `legal_ref`: Rule 43(2) EPC · `confidence`: HIGH

  **Araç içi tutarsızlık (olduğu gibi aktarılıyor):** kaydın `summary` alanı
  (`H_epo_formal_en.json:21`) "Found 1 WARNING EPO formality issue(s)" derken, aynı kaydın `severity`
  alanı (`H_epo_formal_en.json:76`) `IMPORTANT` diyor; şiddet düzeyi **vekil yorumu** ister.

  **Aktarım notu (gözlem, düzeltme değil):** araç 1., 13. ve 15. istemleri "aynı kategori (product)" saydı;
  taslak künyesi 1–12 metod, 13–14 sistem, 15 "non-transitory recording medium" diyor
  (`docs/patent/pct_v2/CLAIMS_EN.md:5`). Kategorinin gerçekten aynı sayılıp sayılmayacağı **vekil kararıdır**.

- `review_epo_specification` özeti (verbatim): «[WARNING] Found 47 IMPORTANT issues in Art. 83 EPC sufficiency analysis» — 47 bulgunun (istem, öğe) çiftleri ABD `C`
  kaydıyla **birebir aynı** (küme karşılaştırması: 47/47 örtüşme, fark 0). Yani aynı destek izi hem
  112(a) hem EPC 83 açısından işaretlenmiş durumda.
- `check_epo_formalities` özeti (verbatim): «[INCOMPLETE] Checks skipped: claims_fees (no numbered claims in an identifiable claims section; supply a CLAIMS heading and claim text). [WARNING] Found 1 WARNING EPO formality issue(s)» — tek bulgu, verbatim:

  1. `section`: abstract · `severity`: IMPORTANT · `problem`: Abstract does not contain reference signs in parentheses · `current`: No reference signs found · `required`: Reference signs corresponding to drawings, in parentheses · `fix`: Add reference numerals from drawings in parentheses, e.g., 'processor (10)' · `legal_ref`: Rule 47(4) EPC · `confidence`: HIGH

- `check_pct_formalities` özeti (verbatim): «[INCOMPLETE] Checks skipped: physical (physical requirements cannot be verified from text; manual inspection required); abstract figure designation (the figure to publish with the abstract is chosen on the request form (PCT Rules 3.3(a)(iii) and 8.2); check the request names one). [WARNING] Found 1 CRITICAL PCT formality issue(s)» — tek kritik bulgu, verbatim:

  1. `section`: claims · `severity`: CRITICAL · `problem`: No claims detected in the application · `current`: 0 claims · `required`: At least one independent claim · `fix`: Add at least one independent claim · `legal_ref`: PCT Rule 6.2(a) · `confidence`: HIGH

  **Okuma uyarısı (araç sınırı):** `check_pct_formalities` aracının girdi imzasında istem metni **yoktur**
  — tanım yalnız `abstract`, `title`, `specification`, `drawings_present` alıyor
  (`/Users/serkankose/patent-creator-venv/lib/python3.13/site-packages/mcp_server/tools/epo_analyzer_tools.py:352-356`).
  Araç bu yüzden istem sayısını `0` görüyor;
  `claims_fees` alt kontrolü de EPO koşumunda aynı nedenle atlandı (`H` kaydı). Bu, gönderim nüshamızda 15
  istem olmadığı anlamına gelmez (`A` ve `E` kayıtları 15 istem saydı); **yanlış-pozitif adayıdır** ve
  vektörün Kural 6 değerlendirmesi ayrıdır.
- `check_package` özeti (verbatim): «[OK] Package is internally consistent: 6 documents scanned, no issues found.»

### 3.5 TR eş kopya — EPO kontrolleri (kayıtlar `E-TR`, `F-TR`)

- `review_epo_claims` (TR): «[WARNING] Found 16 IMPORTANT issues requiring attention under Art. 84 EPC» — tip dağılımı `two_part_form 15` + `conciseness 1`; araç
  sayacı `15 bağımsız / 0 bağımlı` okudu.
- **Araç sınırı (gizlenmez, motor koduyla gösteriliyor):** denetleyici iki-parçalılığın karakterize
  kısmını yalnız İngilizce anahtar sözcüklerle arıyor — `characteri[sz]ed in that` ve
  `wherein the improvement comprises`
  (`/Users/serkankose/patent-creator-venv/lib/python3.13/site-packages/mcp_server/epo_claims_analyzer.py:210-213`);
  bağımlılık tespiti de aynı biçimde İngilizce `claim`/`claims` sözcüğünü istiyor
  (`/Users/serkankose/patent-creator-venv/lib/python3.13/site-packages/mcp_server/epo_claims_analyzer.py:167`).
  TR gönderim nüshası "karakterize edilir" ve "istem 1'e göre" kalıplarını kullanıyor; "karakterize"
  ifadesi TR nüshasının şu satırlarında geçiyor:
  `docs/patent/pct_v2/filing/ISTEMLER_FILING-COPY_TR.txt:5`,
  `docs/patent/pct_v2/filing/ISTEMLER_FILING-COPY_TR.txt:29`,
  `docs/patent/pct_v2/filing/ISTEMLER_FILING-COPY_TR.txt:33` (ölçüm: `grep -n karakterize`).
  Bağımlılık kalıbı İngilizce beklediğinden TR koşumunda hiçbir istem bağımlı sayılmadı
  (`ETR_epo_claims_tr.json` `independent_claims` / `dependent_claims` alanları). Motorun bu okumasına
  uygun bir girdi gözlemi: TR gönderim nüshasında İngilizce "claim" kelimesi hiç geçmiyor
  (`grep -i -w claim` sayaç olarak sıfır veriyor; komut `MOTOR_KAYITLARI_2026-11-20.md` içinde). Aynı sayı
  `conciseness` bulgusuna da yansımış (araç `term` alanı, verbatim): "15 independent product claims" —
  on beş istemin tamamı aynı kategoriye sayılmış. Bu 16 bulgu **ölçüm hatası adayıdır**; TR metninin
  yapısı hakkında hüküm taşımaz.
- `review_epo_specification` (TR): «[OK] Specification provides sufficient disclosure under Art. 83 EPC» — `indexed_terms 880` (EN koşumu `1269`),
  bulgu `0`, kapsama `100%`. İki dildeki terim-indeks farkının nedeni bu turda **doğrulanmadı**;
  hangi dilin daha doğru okunduğu **ölçülmedi**.

### 3.6 Öncül-dayanak kontrolünün elle açılmış hâli (kayıt `B`)

- Koşum: motorun opt-in ortam bayrağı açık (`/Users/serkankose/Documents/quann-3d/patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/B_antecedent_kos.py:12`
  satırındaki komut); aynı motor sınıfı `B_antecedent_kos.py:30` satırından çağrıldı.
  `checks_run`: `antecedent_basis`, `definiteness`, `subjective_terms`, `internal_references`, `structure` ·
  `checks_skipped`: boş liste.
- Araç özeti (verbatim): «[WARNING] Found 43 IMPORTANT issues requiring attention»
- `43` bulgu · `0 kritik` · hepsi `IMPORTANT` · hepsi `confidence LOW` · `compliance_score 85,67`.
  Araç her bulguya verbatim şu düzeltmeyi yazıyor: "MANUAL VERIFICATION REQUIRED: Check if "X" is properly
  introduced earlier. Tool has high false positive rate." (kayıt `B`, `fix` alanı)
- İstem dağılımı (araç `claim` alanı): istem 1: 14 · istem 3: 1 · istem 4: 3 · istem 5: 1 · istem 6: 2 · istem 7: 2 · istem 8: 1 · istem 9: 3 · istem 10: 2 · istem 11: 1 · istem 13: 6 · istem 14: 2 · istem 15: 5
- İzler mekanik olarak bağlanır: aşağıdaki satırlar `rapor_doldur.py` tarafından `B` kaydının `issues`
  dizisinden sırayla ve elle yazılmadan kopyalandı (`claim` / `term` / `location` / `severity` /
  `confidence` alanları). Sayı künyesi: `METRIKLER_2026-11-20.md` tablosu.
- İzler (araç `claim` / `term` / `location` alanları):

1. istem **1** · `the method` · konum `preamble` · IMPORTANT · güven LOW
2. istem **1** · `the task input` · konum `preamble` · IMPORTANT · güven LOW
3. istem **1** · `the trained model` · konum `preamble` · IMPORTANT · güven LOW
4. istem **1** · `the context-encoded representations and processing the derived reader input` · konum `preamble` · IMPORTANT · güven LOW
5. istem **1** · `the trained model` · konum `preamble` · IMPORTANT · güven LOW
6. istem **1** · `the task input consists of one of an ordered sequence of` · konum `preamble` · IMPORTANT · güven LOW
7. istem **1** · `the measured field values of` · konum `preamble` · IMPORTANT · güven LOW
8. istem **1** · `the point cloud of` · konum `preamble` · IMPORTANT · güven LOW
9. istem **1** · `the point features and the pair-geometry features into a single set representation as the reader input by a symmetric set aggregation` · konum `preamble` · IMPORTANT · güven LOW
10. istem **1** · `the task output from the set representation` · konum `preamble` · IMPORTANT · güven LOW
11. istem **1** · `the task output` · konum `preamble` · IMPORTANT · güven LOW
12. istem **1** · `the point cloud of the trained model itself` · konum `preamble` · IMPORTANT · güven LOW
13. istem **1** · `the task output as contribution data of` · konum `preamble` · IMPORTANT · güven LOW
14. istem **1** · `the records` · konum `preamble` · IMPORTANT · güven LOW
15. istem **3** · `the pair-geometry features` · konum `preamble` · IMPORTANT · güven LOW
16. istem **4** · `the symmetric set aggregation` · konum `preamble` · IMPORTANT · güven LOW
17. istem **4** · `the point features and average-pooling over the point pairs` · konum `preamble` · IMPORTANT · güven LOW
18. istem **4** · `the max-pooling and a result of the average-pooling` · konum `preamble` · IMPORTANT · güven LOW
19. istem **5** · `the task input at which context` · konum `preamble` · IMPORTANT · güven LOW
20. istem **6** · `the point-removal effect` · konum `preamble` · IMPORTANT · güven LOW
21. istem **6** · `the reader` · konum `preamble` · IMPORTANT · güven LOW
22. istem **7** · `the region naming` · konum `preamble` · IMPORTANT · güven LOW
23. istem **7** · `the records assigned` · konum `preamble` · IMPORTANT · güven LOW
24. istem **8** · `the region shift comprises quantifying point shares of regions in a reference data group and in a comparison data group` · konum `preamble` · IMPORTANT · güven LOW
25. istem **9** · `the reader` · konum `preamble` · IMPORTANT · güven LOW
26. istem **9** · `the model input identities` · konum `preamble` · IMPORTANT · güven LOW
27. istem **9** · `the same package produces both the task output and a human-readable` · konum `preamble` · IMPORTANT · güven LOW
28. istem **10** · `the task input` · konum `preamble` · IMPORTANT · güven LOW
29. istem **10** · `the multi-sentence set representation` · konum `preamble` · IMPORTANT · güven LOW
30. istem **11** · `the set of global set statistics` · konum `preamble` · IMPORTANT · güven LOW
31. istem **13** · `the instructions configures the system` · konum `preamble` · IMPORTANT · güven LOW
32. istem **13** · `the instructions further configures the system` · konum `preamble` · IMPORTANT · güven LOW
33. istem **13** · `the context-encoded representations into d-dimensional vectors` · konum `preamble` · IMPORTANT · güven LOW
34. istem **13** · `the task input consists of one of an ordered sequence of` · konum `preamble` · IMPORTANT · güven LOW
35. istem **13** · `the point cloud of` · konum `preamble` · IMPORTANT · güven LOW
36. istem **13** · `the same point cloud from which the task output` · konum `preamble` · IMPORTANT · güven LOW
37. istem **14** · `the model input identities` · konum `preamble` · IMPORTANT · güven LOW
38. istem **14** · `the interpretation engine` · konum `preamble` · IMPORTANT · güven LOW
39. istem **15** · `the set of instructions causes the computer` · konum `preamble` · IMPORTANT · güven LOW
40. istem **15** · `the task input consists of one of an ordered sequence of` · konum `preamble` · IMPORTANT · güven LOW
41. istem **15** · `the point cloud of` · konum `preamble` · IMPORTANT · güven LOW
42. istem **15** · `the point cloud and` · konum `preamble` · IMPORTANT · güven LOW
43. istem **15** · `the task output from the set representation` · konum `preamble` · IMPORTANT · güven LOW

## 4. ATLANAN / KOŞMAYAN KONTROLLER (gizlenmez)

| Kayıt | Atlanan kontrol | Gerekçe (araç çıktısı, verbatim) |
|---|---|---|
| `A` | `antecedent_basis` | opt-in: pattern-based antecedent checking has a high false-positive rate, so it is disabled by default |
| `D` | `figure_numeral_coverage` | no detailed description supplied |
| `H` | `claims_fees` | no numbered claims in an identifiable claims section; supply a CLAIMS heading and claim text |
| `K` | `physical` | physical requirements cannot be verified from text; manual inspection required |
| `K` | `abstract figure designation` | the figure to publish with the abstract is chosen on the request form (PCT Rules 3.3(a)(iii) and 8.2); check the request names one |
| `C` / `F` | (çıktıda `checks_skipped` alanı yok) | araç tanımı verbatim: "Best mode is not checked; manual review with inventor input is required." |
| `D` / `H` / `K` | şekillerin fiziksel/baskı denetimi | araç `verdict_scope` verbatim: "drawings_present is a declaration, not inspection of drawing files" |
| hepsi | ön-sanat / yenilik taraması | Bu turda **yapılmadı**. Kanal durumlarının ham kayıtları bu dizinde: `Z_durum_google_patents.json` (`ready: false`), `Z_durum_bigquery.json` (`available: false`), `Z_durum_epo_api.json` (`ready: false`); biçim/şekil aracı `Z_durum_diagram.json` (`ready: true`). Önceki turun tek kanallı taramasının kaydı `docs/patent/pct_v2/ON_SANAT.md` §5'te duruyor. |

**Okuma uyarısı:** `A` kaydındaki "0 bulgu" ve `100` skoru yalnız `checks_run` listesi içindir —
`compliance_score_scope: checks_run_only` alanını araç kendisi de veriyor. Boş bulgu listesi "temiz" demek
değildir; `B` kaydı bunun ölçülü örneğidir: aynı istem metni, opt-in kontrol açılınca `43` LOW-güven bulgu
üretti.

## 5. Yapılmayanlar (kapsamın dışı)

- Şekil dosyalarının baskı/PDF/A kalitesi, sayfa düzeni, puntolar: araç denetlemiyor (yukarıdaki satırlar).
- PCT/RO/101 formu, ücretler, vekâletname, ISA seçimi, öncül beyanı: idari işler, metin denetimi dışında.
- İstem daraltma / ek istem stratejisi, eşdeğer kapsam, ihracat/devir işlemleri: hukuki değerlendirme.
- Yenilik/buluş basamağı değerlendirmesi: bu raporda **yok**; yapılmış değil.

## 6. HÜKÜM (denetçi-vekil köprüsü; hukuki hüküm değildir)

**İyi yan (ölçülü):** koşan ABD istem kontrollerinde bulgu yok (kayıt `A`), 15 istemin tamamı tarifnamede
destekli işaretlendi (`C` `spec_coverage` 100%), paket bütünlüğü temiz (`L` `0` bulgu), özet/başlık/şekil
atıf sayıları üç yargı alanında da sınır içinde (`D`, `H`, `K`). EPO ve PCT biç katmanlarında tek gerçek
bulgu özetdeki şekil numaraları eksikliği (`H`) — araç bunu `ÖNEMLİ/UYARI` düzeyinde veriyor. Dünkü turla
karşılaştırınca metinde yeni bir kırık emergence yok: 47 ↔ 47 ve 2 ↔ 2.

**Kötü / dikkat:** (i) ABD biç denetimi `ready_to_file false` — `SUMMARY` ve `DETAILED DESCRIPTION`
başlıkları yok (`D`); PCT başlıklarıyla yazılmış metinde bu bir aşama uyumsuzluğu, ama ABD'ye giriş
dosyası için iş duruyor. (ii) 47 "yazılı tanım" bulgusu küçümsenecek hacimde değil ve hepsi çekirdek
öğelere değiyor: `task input` ([0065]), `point cloud` ([0020]), `reader` ([0036]) gibi. (iii) Öncül-dayanak
kontrolü kapalıyken "0 bulgu" okuması yanıltıcıdır: açınca 43 LOW-güven bulgu çıkıyor (`B`) — bunların
kaçı gerçek, kaçısı kalıp hatası, **ölçülmedi**; vekilin satır satır bakması gerekir. (iv) PCT koşumundaki
"istem yok" kritiği araç girdi sınırından kaynaklanıyor; bu rapor ona dayanarak hüküm kurmuyor. (v) TR
nüshadaki 16 EPO bulgusu araç sınırından gelen yanlış-pozitif adayıdır; TR metnine ilişkin hüküm kurulmadı.

Sonuç: otomatik sezgi düzeyinde dosya ciddi bir kırık taşımıyor, ancak `ready_to_file false` durumu ve 47+43
izlenmesi gereken madde ile "gönderime hazır" demek değildir. **Başvuru kararı ve metin değişiklikleri için
vekil onayı gerekli.**
