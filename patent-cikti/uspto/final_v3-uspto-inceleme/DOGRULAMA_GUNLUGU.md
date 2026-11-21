# DOĞRULAMA GÜNLÜĞÜ — `INCELEME_RAPORU.md` (bağımsız doğrulama turu)

**Tarih:** 2026-11-19 · **Doğrulayan:** `tester` subagent — raporu yazan oturumdan bağımsız bağlam.
**Kural:** Rapor DÜZELTİLMEDİ. Her iddia ya ham kanıt dosyasına (`dosya:satır`) ya da bu turda yeniden
koşulan komuta bağlandı. ✓ = ham kaynakta doğrulandı · ✗ = ham kaynak iddiayı çürüttü ·
KANITSIZ = ham çıktı korunmadığı için doğrulanamadı (yanlış anlamına gelmez).
**Yeniden koşulanlar:** `grep`/`sed`/`wc`/`awk` (dosya gerçekleri), `patent_lint` (4 dosya),
`check_package`, `search_mpep`/`get_mpep_section` (MPEP alıntı + sayfa doğrulaması).

---

## 1. DOĞRULAMA TABLOSU

| # | İddia (rapor) | Sonuç | Kanıt (dosya:satır veya komut + ham değer) |
|---|---|---|---|
| 1a | `claims_A`: claim_count=15 | ✓ | `claims_A_default.json:12` → `"claim_count": 15` |
| 1b | `claims_A`: independent_count=15, dependent_count=0 | ✓ | `claims_A_default.json:13-14` → `"independent_count": 15`, `"dependent_count": 0` |
| 1c | `claims_A`: compliance_score=100.0, total_issues=0 | ✓ | `claims_A_default.json:11` → `"compliance_score": 100.0`; `:5` → `"total_issues": 0` |
| 1d | `claims_A`: checks_run = definiteness · subjective_terms · internal_references · structure; checks_skipped = antecedent_basis | ✓ | `claims_A_default.json:23-28` (4 kontrol) ve `:16-22` (antecedent_basis + reason + enable_with) |
| 1e | `claims_B`: antecedent_basis checks_run İÇİNDE, checks_skipped boş | ✓ | `claims_B_antecedent.json:16` → `"checks_skipped": []`; `:18` → `"antecedent_basis"` checks_run dizisinde; `:3` → `"antecedent_env": "1"` |
| 1f | `spec_C`: paragraphs=143, indexed_terms=918, coverage 100/15/15, compliant=true, issues=[], input_warnings=[] | ✓ | `spec_C_112a.json:5-6` (143, 918); `:16-21` (percentage 100, supported 15, total 15, unsupported []); `:15` compliant true; `:13` issues []; `:12` input_warnings [] |
| 2a | `formal_D`: 6 bulgu (5 CRITICAL + 1 INFO) | ✓ | `formal_D_abstract_short.json:53-114` → issues dizisinde 6 kayıt (title INFO + 4×spec CRITICAL + drawings CRITICAL); `:116-118` → `"critical_issues": 5, "warnings": 0, "info": 1` |
| 2b | `formal_D`: abstract word_count=138, compliant=true | ✓ | `formal_D_abstract_short.json:6,8` → `"word_count": 138`, `"compliant": true` |
| 2c | `formal_D`: ready_to_file=false; checks_skipped'de figure_numeral_coverage | ✓ | `formal_D_abstract_short.json:119` → `"ready_to_file": false`; `:130-134` → check figure_numeral_coverage, reason "no detailed description supplied" |
| 2d | `formal_E`: 8 bulgu | ✓ | `formal_E_abstract_full.json:56-137` → 8 kayıt; `:139-141` → critical 5 + warnings 1 + info 2 = 8 |
| 2e | `formal_E`: abstract 243 kelime; "exceeds the 150-word limit" ve "single paragraph" bulguları | ✓ | `formal_E_abstract_full.json:6` → 243; `:60` → "Abstract exceeds the 150-word limit"; `:70` → "Abstract should be a single paragraph". (Not: rapor "5 paragraf" der — JSON alanı `line_count: 5` (:7); etiket gevşek, değer doğru) |
| 2f | Başlık 132 karakter | ✓ | `formal_D_abstract_short.json:12` ve `formal_E_abstract_full.json:15` → `"character_count": 132`; `_giris.json:3` → title_chars 132; `wc -m` (OZET.md:3, önek sizdirilmiş) → 132 |
| 3a | ISTEMLER.md'de "Şekil" geçmiyor | ✓ | `grep -c "Şekil" ISTEMLER.md` → `0` (exit 1) |
| 3b | Bağımlı istem 12; bağımsızlar 1/13/15 | ✓ | `grep -cE "^[0-9]+\. İstem [0-9]+'e göre" ISTEMLER.md` → `12`; numara listesi: 5:1., 29:13. ("En az bir işlemci…"), 33:15. ("Üzerinde bilgisayar…") |
| 3c | İstem 1 = 308, İstem 13 = 138, İstem 15 = 106 kelime | ✓ | `sed -n '5p' ISTEMLER.md \| wc -w` → 308; `'29p'` → 138; `'33p'` → 106 |
| 3d | TARIFNAME §12: 9 madde; satır 220 "Şekil 2", satır 231 "Şekil 6" | ✓ | `sed -n '216,231p' TARIFNAME.md \| grep -cE '^[0-9]+\.'` → 9; `awk 'NR>=220 && NR<=231 && /^[0-9]+\./' \| wc -l` → 9; `sed -n '220p' \| grep -o "Şekil [0-9]"` → "Şekil 2"; `'231p'` → "Şekil 6" ve "Şekil 7" |
| 3e | "yalnızca" TARIFNAME'de 6 kez (74/80/111/182/193/220); ISTEMLER'de 0 | ✓ | `grep -n "yalnızca" TARIFNAME.md` → tam olarak satır 74, 80, 111, 182, 193, 220 (`grep -c` → 6); `grep -c "yalnızca" ISTEMLER.md` → 0 |
| 3f | "^\[00" paragraf numarası yok | ✓ | `grep -c "^\[00" TARIFNAME.md` → `0` (exit 1) |
| 3g | wc -c: 10.510 / 46.455 / 2.641 | ✓ | `wc -c ISTEMLER.md TARIFNAME.md OZET.md` → 10510 / 46455 / 2641 (toplam 59606) |
| 3h | "hesaplma" TARIFNAME.md:20, 38, 98 | **✗ (kısmi)** | `grep -in "hesaplma" TARIFNAME.md` → yalnız **20** ve **38**. Satır 98'in tam metni okundu: "Maliyet sınırının ölçülebilir gerekçesi çift yolunda görülür: … çıkmıştır (kaynak: …)" — yazım hatası yok ("hesabı" doğru). Raporun `:88` ve §8.7'deki **":98" bileşeni ham kaynakta yok**; 20 ✓ 38 ✓ 98 ✗ |
| 3i | TARIFNAME:36/62/200 kaynak künyeleri; :74 "D + 4" notu; :220 "Çelişki GİDERİLMİŞTİR (2026-11-18)"; :133 paket tanımı | ✓ | `sed -n '36p' \| grep -o …` → "kaynak: fraud/results/edge_eval_full_scale.json:15-16"; `'62p'` → "kaynak: cloudlm/model.py:47"; `'200p'` → "kaynak: purecloudlm/model.py:8-9"; `'74p'` → "'D + 4' sayısal etiketi çizim başlığından 2026-11-18 tarihinde çıkarılmıştır"; `'220p'` → "Çelişki GİDERİLMİŞTİR (2026-11-18)"; `'133p'` → "tek bir kendi kendine yeten paket içinde üç şeyi taşır" |
| 3j | 8 SVG (fig1..8) + sekiller_v3.pdf | ✓ | `ls fig/ \| wc -l` → 8 (fig1_sistem_mimarisi.svg … fig8_window_bulut.svg); `ls sekiller_v3.pdf` → mevcut |
| 3k | OZET:3 başlık 132 karakter; OZET:5 = 138 kelime; 5+7+9 = 243 | ✓ | `sed -n '3p' OZET.md \| sed 's/^\*\*Buluşun başlığı:\*\* //' \| tr -d '\n' \| wc -m` → 132; `sed -n '5p' \| wc -w` → 138; `sed -n '5p;7p;9p' \| wc -w` → 243 |
| 3l | ISTEMLER:13 "hesapma yükü"; :29 "N noktali"; :21 ve :31 "kendi kendine yeten paket"; :31 "İstem 13'e göre" | ✓ | `sed -n '13p' \| grep -o "hesapma yükünü"` → bulundu; `'29p' \| grep -o "N noktali"` → bulundu; satır 21 ve 31 metinlerinde "kendi kendine yeten paket" mevcut; `31:14. İstem 13'e göre bilgisayar sistemi…` |
| 4a | claims_analyzer.py:101 "the X" ve :106 "said X" regex'leri İngilizce | ✓ | `claims_analyzer.py:100-103` → `\bthe\s+([a-z][a-z\s-]+?)…`; `:105-108` → `\bsaid\s+…` |
| 4b | :111 SUBJECTIVE_TERMS İngilizce kelimeler | ✓ | `claims_analyzer.py:111-132` → subtle, minimal, substantial, significant, …, high, low, large, small, strong, weak; RELATIVE_TERMS `:135-144` → about/approximately/substantially… |
| 4c | :194-202 yorumu bağımlılığı İngilizce/EP açılış ifadesi olarak tanımlıyor | ✓ | `claims_analyzer.py:194-204` → "A dependency is a REFERENCE PHRASE in the claim's opening (\"The method of claim 1, wherein...\"), per 37 CFR 1.75(c)… The phrase forms cover US and EP styles"; regex `:206-208` yalnız İngilizce kalıplar (of/according to/as claimed in…claims) |
| 4d | :459'da uzunluk eşiği word_count > 500 | ✓ | `claims_analyzer.py:458-460` → `word_count = len(claim_text.split()); if word_count > 500:` |
| 4e | linter.py:292 "(şekil\|figure\|fig\.)\s*\d" arıyor | ✓ | `linter.py:292` → `re.search(r"(?i)(şekil\|figure\|fig\.)\s*\d\|şekilde gösteril", body)`. (Not: rapor parça alıntı — `(?i)` öneki ve "\|şekilde gösteril" alternatifi alıntı dışı; içerik doğru) |
| 4f | analyzer_tools.py:159 "Best mode is not checked" | ✓ | `analyzer_tools.py:159` docstring → "Best mode is not checked; manual review with inventor input is required." (birebir) |
| 4g | analyzer_tools.py:71-121 ClaimsAnalyzer().analyze_claims çağrısını geçirir | ✓ | `analyzer_tools.py:82-83` → `analyzer = ClaimsAnalyzer(); analysis_results = analyzer.analyze_claims(claims_text)` |
| 4h | Koşum biçimi: aynı sınıflar, dosyadan okuma, 5 paralel görev; MCP MPEP bloğu koşmadı | ✓ | `run_analizi.py:32-34` (aynı 3 sınıf import), `:74` (analyze_claims), `:81-82` (_parse_claims + analyze_specification_support), `:89-94` (check_all_formalities), `:99-106` (ThreadPoolExecutor max_workers=5, 5 görev); `analyzer_tools.py:86-98/187-199/292-304` mpep_refs bloğu run_analizi.py'da YOK ✓. Not: `run_analizi.py:43` `read_text` kullanır ("ham bayt okuma" deyişi gevşek; bayt sayımı `:112-114` `.encode()` ile — işlev, elle yeniden yazım olmadığı, doğru) |
| 4i | "TÜRKPATENT modu motorda yok" | ✓ | Araç kataloğu araması ("turkpatent/türk patent"): yalnız check_formalities (US), check_epo_formalities, check_pct_formalities var; TR ofis aracı yok |
| 5a | compliance_score_scope = "checks_run_only" | ✓ | `claims_A_default.json:29`; `claims_B_antecedent.json:24` |
| 5b | "…this is not a legal compliance determination." | ✓ | `claims_A_default.json:10` ve `claims_B_antecedent.json:10` summary metninde birebir |
| 5c | verdict_scope "…This is not a filing clearance." | ✓ | `formal_D_abstract_short.json:136`; `formal_E_abstract_full.json:159` |
| 5d | checks_skipped.reason: "opt-in: pattern-based antecedent checking has a high false-positive rate, so it is disabled by default"; "no detailed description supplied" | ✓ | `claims_A_default.json:19`; `formal_D_abstract_short.json:133`; `formal_E_abstract_full.json:156` |
| 6a | TARIFNAME:5/:11/:28/:43/:56 ilgili TR bölüm başlıklarını içeriyor | ✓ | `:5` "## 1. Buluşun teknik alanı" (→ BACKGROUND); `:11` "## 2. Önceki teknik ve açığı" (→ BACKGROUND); `:28` "## 3. Buluşun kısa açıklaması ve teknik etkisi" (→ SUMMARY); `:43` "## 4. Şekillerin kısa açıklaması" (→ BRIEF DESCRIPTION OF DRAWINGS); `:56` "## 5. Buluşun detaylı açıklaması" (→ DETAILED DESCRIPTION). 5 TR başlık ↔ 4 ABD bölümü eşlemesi (BACKGROUND ← :5+:11) metinle uyumlu |
| 7a | Ön sanat araması "yapılmadı" — ve başka yerde "yapıldı" izi yok | ✓ | INCELEME_RAPORU.md:67 ("yapılmadı"), :183 ("yapılmadı… 'arama yapıldı' denemez"), :236-237 ("hiç denenmedi"); raporun hiçbir bölümünde ön sanat sonucu/funduk listesi yok |
| 7b | EPO/PCT koşulmadı | ✓ | INCELEME_RAPORU.md:182 ("koşulmadı"), :221-223, :241-242; kanıt klasöründe EPO/PCT çıktısı yok (`ls` ile teyit) |
| 7b2 | 7b'nin gerekçesi: "Kullanıcı tercihi: USPTO üçlüsü" | **KANITSIZ** | Oturum içi bağlam iddiası; kanıt dosyalarında (JSON/py/md) doğrulanabilir iz yok. İddianın kendisi dışlanmıyor — sadece bu gerekçe kanıtsız |
| 7c | Çizim dosyaları incelenmedi; üst bilgideki "8 SVG + sekiller_v3.pdf" gerçek dosya listesi (inceleme değil) | ✓ | INCELEME_RAPORU.md:6 (dosya listesi — `ls` ile birebir doğrulandı), :181 ("incelenmedi"), :237; :41 "drawings_present is a declaration, not inspection of drawing files" ↔ formal_D:136 birebir |
| 7d | Best mode hiçbir koşulda kontrol edilmedi | ✓ | INCELEME_RAPORU.md:96-98 ve :179; `analyzer_tools.py:159` |
| 7e | "Yapıldı" denilen koşuların (check_package, linter, MPEP) ham çıktısı klasörde yoktu → bu turda yeniden koşuldu ve eşleşti | ✓ | §2'nin yeniden koşum bölümü: check_package birebir; 4 lint koşusu birebir; MPEP 8 alıntı + 9 sayfa atfı (aşağıda) |
| 8a | Atıfsız sayı kümesi BULUNDU: §1 check_package satırı "7 belge, 0 bulgu, puan 100, stamps []" — raporda dosya/komut atfı yok, ham çıktı klasörde yok | ✓ (bulgu doğrulandı) | INCELEME_RAPORU.md:63 (satırda atıf yok); kanıt klasörü `ls` → 10 dosya, check_package çıktısı yok. Yeniden koşum: `files_scanned: 7, total_issues: 0, package_integrity_score: 100, stamps: []` → sayılar DOĞRU ama raporda kaynağı yazılmamış |
| 8b | Linter sayıları ham çıktısız (0/6/1; 3/16/1; 1/6; 4/14/2; claim_count 24; claim_unity 12; "kopuk: 1 (beklenen 2)") | ✓ (bulgu doğrulandı) | `patent_lint` yeniden koşum (4 dosya) — aşağıda ham çıktı: ISTEMLER "0 hata · 6 uyarı · 1 bilgi"; TARIFNAME "3 hata · 16 uyarı · 1 bilgi"; OZET "1 hata · 6 uyarı · 0 bilgi"; birleşik "4 hata · 14 uyarı · 2 bilgi" + "12 bağımsız" + "24 istem" + "kopuk: 1 (beklenen 2)". Sayılar birebir doğru; raporda yalnız aracın adı var, ham çıktı korunmamış |
| 8c | MPEP sayfa/alıntı atıfları ham çıktısız (kaynak adı yazılmış: araç + PDF + sayfa) | ✓ (bulgu doğrulandı; 9 sayfa atfından 9'u uyumlu) | Yeniden koşum: 608.01(b)→mpep-0600.pdf s.135 ✓; 608.01(a)→s.134 ✓; 605.02,606→s.115 ✓; 608.01(a) preferred layout→s.133 ✓; 608.01(m) parentheses→s.143 ✓; 9020 numaralandırma→s.73 ✓; 608.01→s.124 ✓; 2103 112(b)→mpep-2100.pdf s.9 ✓; § 1.77 mpep-9020-appx-r.pdf s.108'de BAŞLIYOR ✓ ("§ 1.77 Arrangement of application elements. (a) …" p.108 hit); 1.77(b) alıntı metni mpep-0600.pdf:193 ve consolidated_rules.pdf:110'da "(b) The specification should include the following sections in order" olarak birebir bulundu. 8 alıntının tamamı içerik olarak doğrulandı (aşağıda) |
| 8d | "Metin 59.606 bayt" (rapor :17) — ayrı atfı yok ama türetilebilir | ✓ | 10510 + 46455 + 2641 = 59606 (wc -c çıktısının toplamı) |
| 8e | "search_mpep çıktılarının kapsam satırı hepsinde aynı: 'Indexed passages only; not a completeness or currency check.'" | **KANITSIZ** | Yeniden koşulan search_mpep çıktısında böyle bir alan/metin YOK (sonuç yapısı: rank, source, section, mpep_section, section_title, file, page, has_*, relevance_score, text). get_mpep_section'ın coverage metni farklı: "Indexed content from MPEP 600… This is not guaranteed full section text." Alıntı ham çıktıyla eşleşmedi; yanlışlığı da kanıtlanamadı |

---

## 2. YENİDEN KOŞUM KANITLARI (ham çıktı)

### 2.1 Dosya gerçekleri — koşu 1

```
=== A) grep -c 'Şekil' ISTEMLER.md
0
grep_exit=1
=== B) bagimli istem sayisi (satir basi 'N. İstem M\'e göre')
12
=== C) tum istem numaralari (grep -nE "^[0-9]+\. ")
5:1. Bilgisayarla uygulanan bir makine öğrenmesi yöntemi olu
7:2. İstem 1'e göre yöntem olup, D boyutunun 3 ile 64 arasın
9:3. İstem 1'e göre yöntem olup, çift-geometri özelliklerini
11:4. İstem 1'e göre yöntem olup, simetrik küme agregasyonun
13:5. İstem 1'e göre yöntem olup, N sayısının görev girdisin
15:6. İstem 1'e göre yöntem olup, nokta çıkarma etkisinin, s
17:7. İstem 1'e göre yöntem olup, bölge isimlendirmenin, küm
19:8. İstem 1'e göre yöntem olup, bölge kaymasının, referans
21:9. İstem 1'e göre yöntem olup, okuyucunun en az bir param
23:10. İstem 1'e göre yöntem olup, görev girdisinin birden ç
25:11. İstem 1'e göre yöntem olup, çift-geometri özellikleri
27:12. İstem 1'e göre yöntem olup, eğitim adımının, okuyucun
29:13. En az bir işlemci ve işlemci tarafından yürütülebilir
31:14. İstem 13'e göre bilgisayar sistemi olup, okuyucunun I
33:15. Üzerinde bilgisayar tarafından yürütülebilir komutlar
=== D) istem kelime sayilari
istem1_line5:      308
istem13_line29:      138
istem15_line33:      106
=== E) wc -c uc dosya
   10510 ISTEMLER.md
   46455 TARIFNAME.md
    2641 OZET.md
   59606 total
```

### 2.2 "yalnızca" ve paragraf numarası — koşu 1 devamı

```
=== F) TARIFNAME 'yalnizca' grep -n   (6 uzun satır; her birinin başı:)
74:Bu bileşenlerin birleştirilmesiyle çift-geometri özelliğinin boyutu 3D + 2 olur; … "D + 4" sayısal etiketi çizim başlığından 2026-11-18 tarihinde çıkarılmıştır; …
80:Maskeleme her iki yolda da uygulanır. … ortalama, yalnızca geçerli çift sayısı üzerinden normalleştirilir (kaynak: cloudlm/model.py:171-174). …
111:Şekil 5'te şematize edilen işlemde … bu tanım yalnızca eğitim verisi üzerinden kurulur (kaynak: aml/src/e1_explain.py:297).
182:**Bölge/prototip etiketleme (ikinci işlem).** … Prototipler yalnızca eğitim düğümlerinin ortalaması olarak tanımlanır (kaynak: aml/src/e1_explain.py:297). …
193:Aynı dosyada ölçülen tohum başına süreler … yalnızca model-sonrası atıf yolunun ayrı bir yeniden eğitim adımı taşıdığına işaret eder.
220:1. **Çift-geometri genişliğinin adı ve çizim başlığında çelişki.** … yalnızca Şekil 2 çiziminin başlık metninde bulunmuştur … Çelişki GİDERİLMİŞTİR (2026-11-18): …
yalnizca_count: 6
=== G) ISTEMLER 'yalnizca' count → 0 (grep_exit=1)
=== H) TARIFNAME paragraf numarasi '^\[00' → 0 (grep_exit=1)
=== I) TARIFNAME 'hesaplma' grep -n →
20:**(b) Açıklama, uç aygıtta ayrı bir yığın gerektirir.** … ayrı bir hesaplma adımı ekler
(38. satır büyük harfle "Hesaplma yükü…" — case-sensitive aramada çıkmadı, aşağıda case-insensitive koşuldu)
```

Not: F bloğundaki 6 satırın gövdeleri bu günlükte satır başı + ilk/son bölüm gösterilerek kırpıldı;
tam satırlar doğrulama oturumunun ham çıktısındadır ve `grep -c` sayısı (6) tam satırlar üzerinden
alınmıştır. Satır numaraları (74/80/111/182/193/220) raporla birebir aynıdır.

### 2.3 TARIFNAME §12 ve künyeler — koşu 2

```
=== K) TARIFNAME 216-231 (§12 bolgesi)
## 12. Ölçülmemiş iddialar ve doğrulanamayan kayıtlar
Buluş metnine alınmayan veya metinde bilinçli olarak yumuşatılan noktalar, gerekçeleriyle kaydedilir:
1. **Çift-geometri genişliğinin adı ve çizim başlığında çelişki.** …
2. **Sabit nokta sayısı.** …
Nokta sayısı bu nedenle bağlam uzunluğuna bağlı tanımlanmıştır.
3. **Tek modelle çoklu görev başarımı.** …
4. **Doğruluk üstünlüğü.** …
Metin hiçbir alanında doğruluk üstünlüğü iddia etmez.
5. **Bölge kaymasında uç değerler.** …
6. **Yeni bölge işareti.** …
Bu nedenle işaret, iddia olarak değil ölçüt olarak istemlenmiştir.
7. **Açıklama kararlılığı kıyası.** …
8. **Protokolü satırla bağlı olmayan perplexity değerleri.** …
9. **Şekillerde ölçüm.** Yalnız Şekil 6 ve Şekil 7 ölçüm taşır; …
=== L) sed 216-231 | grep -cE '^[0-9]+\.' → 9
=== M) awk 'NR>=220 && NR<=231 && /^[0-9]+\./' | wc -l → 9
=== N) satir 220 → "Şekil 2" ; satir 231 → "Şekil 6", "Şekil 7"
=== O) 36p → "kaynak: fraud/results/edge_eval_full_scale.json:15-16"
       62p → "kaynak: cloudlm/model.py:47"
      200p → "kaynak: purecloudlm/model.py:8-9"
=== P) 133p → "Şekil 7'nin (b) panelinde … tek bir kendi kendine yeten paket içinde üç şeyi taşır: …"
=== Q) grep -in "hesaplma" TARIFNAME.md | cut -d: -f1 → 20, 38   (98 YOK)
=== R) OZET baslik (wc -m, önek sizdirilmiş) → 132
=== S) ozet_line5 (wc -w) → 138 ; ozet_5+7+9 (wc -w) → 243
```

### 2.4 Satır 98 kontrolü (3h ✗ kanıtı)

```
=== TARIFNAME satir 98 tam metin:
Maliyet sınırının ölçülebilir gerekçesi çift yolunda görülür: etkin nokta sayısı 13'ten 50'ye
çıkarıldığında işlem başına çarp-topla işlemi 2.522.200'den 34.091.488'e çıkmıştır (kaynak: k
değerleri ve MAC sayıları fraud/results/edge_eval_full_scale.json:29-30, …:41-42)
=== "esaplma" aramasi (büyük/küçük harf duyarsız çekirdek) → yalnız 20 ve 38
```

### 2.5 `patent_lint` yeniden koşumu (4 dosya) — 8b kanıtı

```
ISTEMLER:  "Linter: 0 hata · 6 uyarı · 1 bilgi"
           warn section_missing ×6 (tarifnamede teknik alan/önceki teknik/kısa açıklama/şekiller/detaylı/özet yok)
           info claim_unity: "3 bağımsız istem var"
TARIFNAME: "Linter: 3 hata · 16 uyarı · 1 bilgi"
           error claim_figure_ref istem 1; error claim_figure_ref istem 9; error buzzword istem 220 'yalnızca'
           warn two_part_form istem 1..9 (9 adet); warn section_missing ×2 (istemler, özet);
           warn buzzword satır 74, 80, 111, 182, 193 (5 adet)
           info claim_unity: "9 bağımsız istem var"
OZET:      "Linter: 1 hata · 6 uyarı · 0 bilgi"
           error claim_structure: "Hiç istem bulunamadı"; warn section_missing ×6
birlesik:  "Linter: 4 hata · 14 uyarı · 2 bilgi"
           error claim_numbering "İstem numaralandırması kopuk: 1 (beklenen 2)";
           error claim_figure_ref istem 1 ve istem 9; error buzzword istem 232 'yalnızca';
           warn two_part_form istem 1..9; warn buzzword satır 86, 92, 123, 194, 205
           info claim_unity: "12 bağımsız istem var"; info claim_count: "24 istem: 15'i aşan her istem ek ücrete tabi"
```

Raporun §1 ve §5 linter satırlarıyla birebir eşleşme: ✓ (raporun "13 bulgu artefakt" dökümü de
2 figure_ref + 9 two_part + 1 claim_unity + 1 buzzword(istem 220) = 13 ile tutarlı).

### 2.6 `check_package` yeniden koşumu — 8a kanıtı

```
tools["patent-creator"].check_package({directory: "docs/patent/final_v3"}) →
{"total_issues":0,"critical_issues":0,"important_issues":0,"minor_issues":0,"issues":[],
 "summary":"[OK] Package is internally consistent: 7 documents scanned, no issues found.",
 "package_integrity_score":100,"files_scanned":7,"stamps":[]}
```

Rapor §1: "7 belge, 0 bulgu, puan 100, stamps []" → birebir ✓. (Raporda bu sayıların atfı yok — 8a bulgusu.)

### 2.7 MPEP yeniden koşumu — 8c kanıtı

Alıntı metinleri (search_mpep / get_mpep_section, bu turda koşuldu):

1. "The abstract must be as concise as the disclosure permits, preferably not exceeding 150 words in length." → mpep-0600.pdf s.135, mpep_section 608.01(b) ✓
2. "The abstract is a brief narrative of the disclosure as a whole, as concise as the disclosure permits, in a single paragraph preferably not exceeding 150 words, commencing on a separate…" → mpep-0600.pdf s.134, 608.01(a) ✓
3. "§ 1.72 Title and abstract. (a) The title of the invention may not exceed 500 characters in length and must be as short and specific as possible." → mpep-0600.pdf s.115, mpep_section "605.02, 606" ✓ (raporun bölüm etiketi de birebir)
4. "(b) The specification should include the following sections in order: (1) Title of the invention…" → mpep-0600.pdf s.193 (608.05) ve consolidated_rules.pdf s.110'da birebir "(b)" ile; § 1.77 bölümü mpep-9020-appx-r.pdf s.108'de başlıyor ("§ 1.77 Arrangement of application elements. (a) …" hit) ✓
5. "The following guidelines illustrate the preferred layout for the specification of a utility application. These guidelines are suggested for the applicant's use. … As provided in 37 CFR 1.77(b), the specification of a utility application should include the following…" → mpep-0600.pdf s.133, 608.01(a) ✓
6. "Reference characters … should be enclosed within parentheses so as to avoid confusion…" → mpep-0600.pdf s.143, 608.01(m) ✓
7. "…may be numbered at the time the application is filed, and should be individually and consecutively numbered using Arabic numerals … at least four numerals enclosed in square brackets, including leading zeros (e.g., [0001])" → mpep-9020-appx-r.pdf s.73 ✓ ve mpep-0600.pdf s.124 (608.01) ✓
8. "35 U.S.C. 112(b) contains two separate and distinct requirements: (A) that the claim(s) set forth the subject matter the inventor or a joint inventor regards as the invention, and (B) that the claim(s) particularly point out and distinctly claim the invention." → mpep-2100.pdf s.9, 2103 ✓

Örnek ham metadata (get_mpep_section "608.01(b)"):

```
{"section":"608.01(b)","chapter":"MPEP 600","boundary":"approximate",
 "coverage":"Indexed content from MPEP 600. … This is not guaranteed full section text.",
 "chunks[1].metadata":{"source":"MPEP","file":"mpep-0600.pdf","page":135,
   "mpep_section":"608.01(b)","section_title":"Abstract of the Disclosure", ...}}
```

Doğrulamayan tek MPEP kalemi: raporun "kapsam satırı" alıntısı (8e — KANITSIZ; yukarıdaki yapıda
"Indexed passages only; not a completeness or currency check" metni görünmedi).

---

## 3. KAPANIS

**49 iddia satırı doğrulandı: 46 ✓ · 1 ✗ · 2 KANITSIZ.**

- Tek ✗: **3h** — "hesaplma" yazım hatasının TARIFNAME.md:**98**'de geçtiği iddiası yanlış; hata yalnız
  satır 20 ve 38'de (satır 98'de "hesabı" doğru yazılmış). Raporun `:88` ve §8.7 satırlarındaki ":98"
  ham kaynakta yok. Diğer tüm sayılar ve satır atıfları ham kaynakta birebir doğrulandı.
- KANITSIZ (1): **7b2** — "Kullanıcı tercihi: USPTO üçlüsü" gerekçesi oturum bağlamı, kanıt dosyasında iz yok.
- KANITSIZ (2): **8e** — "kapsam satırı" alıntısı yeniden koşulan search_mpep çıktısında görünmedi.
- Sayı uydurma avının cevabı (8. iddia): raporda atıfsız sayı kümesi **1 adet** — §1'deki check_package
  satırı (7 belge / 0 bulgu / puan 100 / stamps []); değerleri bu turda yeniden koşumla DOĞRU çıktı ama
  raporda kaynağı yazılmamış. Linter sayıları ve MPEP sayfa atıfları da ham çıktısız kalmış; hepsi
  yeniden koşumla birebir doğrulandı (değer uydurma yok, kanıt saklama eksikliği var).
- "Yapılmadı" denilen dört kalem (ön sanat, EPO/PCT, çizim dosyası incelemesi, best mode) raporda
  tutarlı şekilde "yapılmadı/atlandı" işaretli; istemeden "yapıldı" izi bırakılmamış (7a-7e).

Rapor metninde değişiklik YAPILMADI. Bu günlük yalnız `patent-cikti/uspto/final_v3-uspto-inceleme/`
içinde yeni dosya olarak bırakıldı.
