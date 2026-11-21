# INCELEME RAPORU — PCT v2 İngilizce Taslak: USPTO Otomatik Kontrolleri

- **Tarih:** 2026-11-19
- **Girdiler (araç koşumu sırasında okunan ham dosyalar):**
  - İstemler: `docs/patent/pct_v2/CLAIMS_EN.md` (araç girişine aktarılan uzunluk: 11.424 karakter — execute ölçümü)
  - Tarifname: `docs/patent/pct_v2/DESCRIPTION_EN.md` (39.979 karakter — execute ölçümü)
  - Özet (EN nüsha): `docs/patent/pct_v2/OZET_PCT.md` içinden çıkarıldı (boşluk tekilleştirme sonrası 863 karakter — execute ölçümü)
  - Başlık: "Dual-use point-cloud representation with embedded explainability" (`CLAIMS_EN.md:3`, `DESCRIPTION_EN.md:3`)
  - Şekiller: `docs/patent/pct_v2/fig/` — 8 SVG + `fig_manifest.json` mevcut (`drawings_present=true` verildi)
- **Koşan araçlar (MCP `patent-creator`, paralel):** `review_patent_claims` · `review_specification` · `check_formalities`
- **Kapsam notu:** Bunlar otomatik metin sezgileridir (heuristic); hukuki hüküm değildir. Hiçbir bulgu bu raporda düzeltilmedi — yalnız aktarıldı.

---

## 1. Ne yaptık

PCT v2 taslağının İngilizce istem, tarifname ve özet metinlerini olduğu gibi üç USPTO otomatik denetiminden geçirdik: istem netliği (35 USC 112(b) sezgileri), tarifname desteği (35 USC 112(a) yazılı-tanım/uygulama sezgileri) ve MPEP 608 / 37 CFR 1.77 biç denetimleri.

## 2. Sonuçlar

| Ne | Değer (araç çıktısı) | Anlam |
|---|---|---|
| İstem denetimi — compliance_score | **100** (kapsam: `checks_run_only`) | Koşan 4 kontrolde hiç bulgu yok; ama bir kontrol atlandı (§4) |
| İstem sayacı | 15 istem · 3 bağımsız · 12 bağımlı | Araç sayacı; taslak notuyla tutarlı |
| İstem bulguları | 0 kritik / 0 önemli / 0 minör | Koşan kontrollerde temiz |
| Tarifname desteği — toplam bulgu | **47** · 0 kritik · **47 önemli** · enablement 0 | Hepsi "yazılı tanım zayıf" uyarısı (MPEP 2163, MEDIUM güven) |
| İstem→tarifname kapsaması | **100%** (15/15 istem destekli) | Hiçbir istem "desteksiz" işaretlenmedi |
| Biç denetimi — genel | `overall_compliant=false` · `ready_to_file=false` | **2 KRİTİK** biç bulgusu (§3.3) |
| Özet | 137 kelime · 1 satır → compliant | Dosyanın kendi ölçümüyle (137 kelime) tutarlı |
| Başlık | 64 karakter · 6 kelime → compliant | ≤500 sınırının altında |
| Şekiller | 8 şekil atıf bulundu · `drawings_provided=true` → compliant | Atıf listesi FIG.1–8 |

## 3. Teknik yanıtlar (bulgular araç çıktısından olduğu gibi)

### 3.1 `review_patent_claims` — 112(b) sezgileri

Araç özeti (verbatim): *"[INCOMPLETE] Checks skipped: antecedent_basis (opt-in: pattern-based antecedent checking has a high false-positive rate, so it is disabled by default); enable with PATENT_ENABLE_ANTECEDENT_CHECK=1 (LOW confidence; manual review required). No issues found in the checks that ran on 15 claims; this is not a legal compliance determination."*

- Koşan kontroller (`checks_run`): `definiteness`, `subjective_terms`, `internal_references`, `structure`
- `compliance_score_scope`: `checks_run_only` — skor yalnız koşan kontrollerin skorudur
- Araç atıfları: MPEP § 2173 (s.475), § 2173.02 (s.479, 481), § 2161.01 (s.411), § 2181 (s.519)
- Araç açıklaması (verbatim): *"Corresponding specification structure is not checked."*

### 3.2 `review_specification` — 112(a) yazılı-tanım/uygulama sezgileri

Araç özeti (verbatim): *"[WARNING] Found 47 IMPORTANT issues in specification support"*. Araç ayrıca `"compliant": true` alanı döndürdü — 47 önemli bulguyla birlikte; iki alan araç çıktısında yan yana duruyor, hangisinin esas olduğu **denetçi/vekil değerlendirmesi gerektirir**.

- Sayaçlar: `specification_paragraphs: 70` · `indexed_terms: 1305` · `written_description_issues: 47` · `enablement_issues: 0` · `spec_coverage: 100% (15/15, unsupported_claims: [])`
- 47 bulgunun TAMAMI aynı kalıp: `severity=IMPORTANT` · `type=written_description` · `mpep=MPEP 2163` · `confidence=MEDIUM` · problem: *"Claim element \"X\" has limited description (only Para. N)"* · fix: *"Consider adding more detailed description of \"X\" for stronger support"*

Bulgular (istem · öğe · işaretlenen paragraf — araç `spec_refs` alanından):

| # | İstem | Öğeler (paragraf) |
|---|---|---|
| 1 | 1 | task input ([0065]) · context-encoded representations ([0010]) · point in a D-dimensional space ([0010]) · ordered sequence of ([0010]) · tabular record ([0010]) · point cloud ([0020]) · input order of the points ([0010]) · reader ([0036]) · probability distribution of a next ([0036]) · class assignment score ([0036]) · same trained model ([0011]) |
| 3 | 3 | Euclidean norm of the difference vector ([0033]) |
| 4 | 4 | result of the max-pooling ([0034]) · result of the average-pooling ([0034]) |
| 5 | 5 | context length of the task input ([0039]) |
| 6 | 6 | reader ([0036]) |
| 8 | 8 | emerging region ([0013]) |
| 9 | 9 | reader ([0036]) |
| 10 | 10 | task input ([0065]) · plurality of sentences ([0065]) · sentence cloud state ([0065]) · fixed stride ([0037]) · multi-sentence set representation ([0037]) · reader ([0036]) |
| 11 | 11 | point cloud ([0020]) · cloud centre ([0033]) |
| 12 | 12 | reader by gradient-based backpropagation ([0015]) · point cloud ([0020]) · candidate point position ([0015]) |
| 13 | 13 | computer system ([0016]) · at least one processor ([0016]) · instructions configures the system ([0016]) · task input ([0065]) · context-encoded representations ([0010]) · ordered sequence of ([0010]) · tabular record ([0010]) · point cloud ([0020]) · input order of the points ([0010]) |
| 14 | 14 | distribution store holding ([0050]) · reader ([0036]) |
| 15 | 15 | task input ([0065]) · context-encoded representations ([0010]) · ordered sequence of ([0010]) · tabular record ([0010]) · point cloud ([0020]) · input order ([0010]) · reader ([0036]) |

**Aktarım notu (gözlem, düzeltme değil):** Kalıp "yalnız tek paragraftan türetiliyor" sezgisidir; ör. `reader` için [0036] işaretlenmişken [0029]-[0031], [0053] gibi pasajlar da aracı okur — aracın birebir terim-eşleme sezgisi mi kaçırdığı yoksa gerçekten tek kaynak mı olduğu **vekil incelemesi** ister. Bulgular olduğu gibi bırakılmıştır.

Araç atıfları: MPEP § 2161 (s.410), § 2161.01 (s.411), § 2163 (s.417), § 2181 (s.522), § 2136.05(a) (s.231).

### 3.3 `check_formalities` — MPEP 608 / 37 CFR 1.77 biç denetimi

Araç özeti (verbatim): *"[INCOMPLETE] Checks skipped: figure_numeral_coverage (no detailed description supplied). [WARNING] Found 2 CRITICAL formality issue(s)"*

Kritik bulgular (araç `issues` alanından, verbatim):

1. **CRITICAL** · specification · *"Missing required section: SUMMARY"* · current: "Not found" · fix: *Add "SUMMARY" section to specification* · mpep: **37 CFR 1.77** · confidence HIGH
2. **CRITICAL** · specification · *"Missing required section: DETAILED DESCRIPTION"* · current: "Not found" · fix: *Add "DETAILED DESCRIPTION" section to specification* · mpep: **37 CFR 1.77** · confidence HIGH

Temiz alt kontroller: abstract `word_count=137` (dosyanın kendi `wc -w` ölçümüyle aynı) · title 64 karakter · drawings: FIG. 1–8 atfı, `drawings_provided=true` · `warnings=0` · `info=0`.

**Bağlam gözlemi (hukuki hüküm değil):** Tarifname PCT Rule 5.1(a) başlıklarıyla yazılmış ("DISCLOSURE OF THE INVENTION", "BEST MODE FOR CARRYING OUT THE INVENTION"); denetleyici ise ABD Ulusal Aşama başlıkları olan "SUMMARY" ve "DETAILED DESCRIPTION" arıyor (`found_sections`: BACKGROUND ✓, SUMMARY ✗, BRIEF DESCRIPTION OF DRAWINGS ✓, DETAILED DESCRIPTION ✗). İki aşamanın başlık kümesi farklıdır; ABD'ye girişte başlık eşlemesi gerekip gerekmediği **vekil kararıdır**.

Araç atıfları: MPEP § 1302.01 (s.2), MPEP 600 dizini (608.01 düzeni, s.1), MPEP § 608.01(d) (s.138, 37 CFR 1.74 alıntısıyla), MPEP 600 (s.2).

Araç verdict_scope (verbatim): *"Automated text and drawing-reference checks only; drawings_present is a declaration, not inspection of drawing files. This is not a filing clearance."*

## 4. ATLANAN KONTROLLER (checks_skipped — gizlenmez)

| Araç | Atlanan | Gerekçe (araç çıktısı) | Nasıl açılır |
|---|---|---|---|
| `review_patent_claims` | `antecedent_basis` | opt-in: "pattern-based antecedent checking has a high false-positive rate, so it is disabled by default" | ortam değişkeni `PATENT_ENABLE_ANTECEDENT_CHECK=1` |
| `check_formalities` | `figure_numeral_coverage` | "no detailed description supplied" | şekil-numara kapsam kontrolü bu koşuda yapılmadı |
| `review_specification` | (çıktıda `checks_skipped` alanı yok) | araç tanımı (verbatim): *"Best mode is not checked; manual review with inventor input is required."* | best-mode manuel inceleme şart |

**Okuma uyarısı:** "0 bulgu" yalnız `checks_run` listesi içindir; `compliance_score_scope: checks_run_only` alanı bunu araç kendisi de söylüyor. Boş liste "temiz" demek değildir.

## 5. Bağlam — makale ön-sanat taraması ve kapsam etkisi (YORUM)

Aynı buluş için bu oturumda yapılan makale ön-sanat taraması: `patent-cikti/uspto/onart-quann-cloud.md`. En yakın iki referans ailesi:

1. **arXiv 2607.16612 — "Backpropagation-Free Trunk Training via the Split Forward Gradients" (Split-FG, 2026)**: dil modelinde geri-yayılımsız eğitim + donuk-trunk/donanım-kısmen-eğitme bulgusu (onart dosyası satır 1'de en yüksek risk).
2. **PointLLM / 3D-LLM ailesi** (onart dosyası satır 26-27): LLM'e nokta-bulutu modal girdisi kazandırma tarafı.

**Yorum (2-3 cümle, hukuki hüküm değildir):** Bu iki referans ailesi, istem 12'nin (nokta konumlarının gradyansız zeroth-order kabul/geri-al güncellemesi) ve istem 1'in "nokta-bulut temsilli LM" çerçevesinin tek başına genişliğine itiraz zemini oluşturabilecek komşu öğretimler olarak görünmektedir; taslağın bu referanslardan ayrıştırdığı potansiyel özellikler, onart notunda belirtildiği üzere "aynı bulut üzerinde üç açıklama işlemi + ortak sunum" ve "tek kendi-kendine-yeten INT8 paket" kombinasyonlarıdır. Vekilin ön-sanat karşısında kapsamı bu kombinasyon özelliklerine doğru daraltıp daraltmayacağı tamamen **vekil değerlendirmesi gerektirir**.

**Arama kapsamı uyarısı (onart dosyasından aktarım):** Patent veritabanı araması (Google Patents/EPO OPS/BigQuery/USPTO API) **yapılmadı** — dördü de bu oturumda anahtar istedi; boş patent tarafı "önsanat yok" kanıtı değildir. Bu USPTO metin kontrolleri de ön-sanat denetimi içermez.

## 6. HÜKÜM (denetçi-vekil köprüsü; hukuki hüküm değildir)

**İyi yan:** Koşan istem kontrollerinde (definiteness, subjektif terim, iç-atıf, yapı) sıfır bulgu ve 15/15 istemin tarifnamede destek bulması, metnin otomatik sezgi düzeyinde sağlam bir iskelete işaret ettiğini gösteriyor. Özet (137 kelime), başlık (64 karakter) ve şekil atıfları (FIG.1–8) biç sınırlarında. 47 yazılı-tanım bulgusunun tamamı "önemli" düzeyinde ve tek bir kalıba iniyor: birebir terim-eşleme destek izi — bu tür bulgular genellikle paragraf zenginleştirmesiyle izlenebilir nitelikte.

**Kötü yan / dikkat:** (i) Biç denetimi `ready_to_file=false` verdi: 2 KRİTİK eksik bölüm (ABD aşaması başlıkları SUMMARY, DETAILED DESCRIPTION) — PCT başlıklarıyla yazılmış taslakta bu beklenen bir aşama uyumsuzluğu olsa da ABD'ye giriş metni için başlık eşlemesi işi duruyor. (ii) 47 IMPORTANT bulgu küçümsenecek hacimde değil; özellikle `task input` ([0065]) ve `point cloud` ([0020]) gibi çekirdek öğelerin tek paragrafa indirgenmiş destek izi, vekilin ilk bakacağı yerlerdendir. (iii) `antecedent_basis` ve `figure_numeral_coverage` koşmadı — "100" ve "0 bulgu" bu nedenle nihai temizlik kanıtı değildir. Sonuç: otomatik düzeyde ciddi bir kırık yok ama dosya "hazır" değil; **denetçi/vekil görüşü ve onayı gerekli**.

## 7. Kanıt — "Dosya yazıldı"
```
total 24
drwxr-xr-x  3 serkankose  staff     96 Nov 19 23:19 .
drwxr-xr-x  6 serkankose  staff    192 Nov 19 23:14 ..
-rw-r--r--  1 serkankose  staff  12134 Nov 19 23:22 INCELEME_RAPORU.md
```
