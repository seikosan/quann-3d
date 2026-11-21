# İNCELEME RAPORU — pct_v3 başvuru dosyası (USPTO + PCT araçları, taze koşum)

- Tarih: 2026-11-20 · koşan: `patent-denetmen` oturumu, motor `patent-creator` MCP (yerel)
- Girdiler (sha1, bu tur `shasum`): CLAIMS `883405033e…` · DESCRIPTION `4ef0d32a3e…` · ABSTRACT `b398c213ae…`
  (üç dosya § `docs/patent/pct_v3/filing/`) · title = "Dual-use point-cloud representation with embedded explainability"
  · `drawings_present=true` beyanı (`filing/SEKILLER_FILING-COPY.pdf` mevcut)
- Metinler araca **ham bayt** verildi (kayıt sunucusu `kayit_sunucusu.py`, GET /claims|/description|/abstract);
  elle yeniden yazım yok. Ham yanıtlar: `motor_2026-11-20/*.json` (A, C, D, K, K2, L).
- **Önemli: `antecedent_basis` bu turda KOŞTU.** A kaydında `checks_run` beş kalem içeriyor,
  `checks_skipped: []` → config'e eklenen `PATENT_ENABLE_ANTECEDENT_CHECK=1` bu MCP sürecine yüklenmiş.
  10:52'deki eski A kaydı hâlâ skipped gösteriyordu (o süreç bayrağı almamıştı) — iki koşul farkı kayıtta ayrışıyor.

## 1. Sonuç tablosu

| Araç (ham kayıt) | Skor / durum | CRITICAL | IMPORTANT | Yorum |
|---|---|---|---|---|
| `review_patent_claims` + antecedent ON (A) | 86,33 (kapsam: checks_run_only) | 0 | 41 (hepsi `antecedent_basis`) | 15 istem, 3 bağımsız; araç kendi güvenliğini LOW+FP uyarısıyla veriyor |
| `review_specification` (C) | kapsam %100 (15/15 istem destekli) | 0 | 17 (`written_description`, MPEP 2163) | enablement 0; 98 tarifname paragrafı indekslendi |
| `check_formalities` — US (D) | overall False | 2 | 0 | eksik `SUMMARY` / `DETAILED DESCRIPTION` başlığı — KURULUM_PLAN kararı 1 uyarınca **bilinçli açık** (PCT Rule 5.2 başlıkları esas) |
| `check_pct_formalities` (K) | overall False | 1 | 0 | "No claims detected" — AŞAĞIDA artifakt olduğu deney ile gösterildi |
| K2 DENEY (K2) | critical **0** | 0 | 0 | aynı araç, specification=tarifname+`CLAIMS`+istemler → 15 istem [1–15] algılandı, claims bölümü `compliant:true` |
| `check_package` (L) | **100 / [OK]** | 0 | 0 | 6 belge tarandı, sorun yok |

Eşzamanlı teyit: bu turun sayıları 10:52'deki bağımsız kayıtla birebir aynı — A(antecedent)=41 ↔
`B_us_claims_en_antecedentON_v3.json` `total_issues:41`; C=17/kapsam 100 ↔ eski C=17; D=2 ↔ eski D=2; K=1 ↔ eski K=1.
Terim kümeleri de aynı (karşılaştırma bu turda yapıldı, fark listesi boş).

## 2. Bulgular (araç çıktısı, aynen)

### 2.1 A — istemler: 41 × IMPORTANT `antecedent_basis`
- Tümü: `severity IMPORTANT · confidence LOW · mpep MPEP 2173.05(e)`; fix alanı aynen:
  "MANUAL VERIFICATION REQUIRED … Tool has high false positive rate."
- Tüm `location` alanları "preamble" görünüyor oysa işaretli dizeler gövdede — segment ayracı
  sınırlaması (DEGISIKLIK_KAYDI_ISTEM.md bu notu taşıyor); FP değerlendirmesinde kanıt olarak kullanıldı.
- Örnek (A kaydı issues[0]): `claim 1 · "the method" · "POSSIBLE antecedent basis issue … may lack proper introduction"`.
- Mekanik ön-eleme (bu tur scripti, ham istem metninde tanitim araması): 41'den **15**'inde tanıtım
  kalıbı metinde bulundu (muhtemel yanlış-pozitif); 26'sı işaretli kaldı — ancak işaretlenen "terim"
  dizelerinin bir kısmı kırpılmış eylem dizileri ("the task input consists of one of an ordered sequence of")
  yani terim-çıkarımı artığı; **ön-eleme vekil incelemesinin yerine geçmez**.
- Seyir: v2 aynı kontrol **43** (`motor_2026-11-20/B_us_claims_en_antecedentON.json:9`, v2 dizini) → v3 **41**.

### 2.2 C — tarifname desteği: 17 × IMPORTANT `written_description`
- Araç mesajı kalıbı: `Claim element "X" has limited description (only Para. N)` · `mpep MPEP 2163` · `confidence MEDIUM`.
- Bağımsız ham sayım (bu tur; tarifnamede büyük/küçük duyarsız tam-deyim taraması, 11 benzersi öğe):

| Öğe | Araç iddiası | Ham sayım (paragraf / toplam geçiş) |
|---|---|---|
| context encoder | tek [0001] | **13 / 15** |
| interpretation engine | tek [0097] | **10 / 11** |
| pair-geometry features | tek [0091] | **9 / 10** |
| sentence cloud state | tek [0065] | **7 / 9** |
| tabular record | tek [0010] | **6 / 6** |
| window-geometry encoder | tek [0014] | **4 / 4** |
| context length of the task input | tek [0039] | **3 / 3** |
| clustering operation | tek [0043] | **2 / 2** |
| point in a D-dimensional space | tek [0010] | **1 / 1** — gerçekten tek |
| reader by gradient-based backpropagation | tek [0092] | **1 / 1** — gerçekten tek |
| "point cloud of" | tek [0011] | **1 / 1** (kırpılmış terim dizesi; "point cloud" bağımsız olarak çok geçiyor) |

- Okuma: 11 benzersi öğeden **8'i** aracın "yalnız tek paragrafta" iddiasıyla çelişiyor — bilinen
  indeksleme sınırlamasıyla uyumlu (v2 kaydı: `_extract_technical_terms` terimleri ~6 kelimelik
  dizeler üretiyor, `specification_analyzer.py:174`, ve işaretli paragraf bölünmesi
  `specification_analyzer.py:130` — `ACILIM_SONUCLARI.md:169,117`). Gerçekten tek-deyiş olan iki tam deyim
  ("point in a D-dimensional space", "reader by gradient-based backpropagation") vekilin
  destek açısından ayrıca bakacağı adaylar. Hüküm değil, bulgu sırası.
- Seyir: v2 orijinal **47** (`mcp_ham/mcp_spec_before.json` text[0]: `total_issues:47`, 70 paragraf)
  → v2 aday **36** (`mcp_ham/mcp_spec_after_neutral.json` text[0]: `total_issues:36`, 84 paragraf)
  → v3 **17** (bu tur C kaydı). Kapsam üç turda da %100 (15/15).

### 2.3 D — US formaliteleri: 2 CRITICAL (bilinçli açık)
- "Missing required section: SUMMARY" · "Missing required section: DETAILED DESCRIPTION" (37 CFR 1.77 beklentisi).
- KURULUM_PLAN.md §2 karar 1: hedef yalnız PCT; başlıklar PCT Rule 5.2 sözleşmesinde kalıyor.
  USPTO ulusal aşama girişinde vekil kararı gerektiren **kayıtlı açık madde**.
- Diğer alanlar (abstract, title, drawings, sections) OK; abstract-titulo örtüşme kontrolü koşuldu.

### 2.4 K — PCT formaliteleri: 1 CRITICAL + artifakt gösterimi
- Bulgu: `CRITICAL · section "claims" · "No claims detected in the application" · PCT Rule 6.2(a)`.
- Kanıt ki girdi-sekli artifaktı:
  1. Araç signature'ı istem metni **almıyor** (`check_pct_formalities({abstract,title,specification,drawings_present})`);
  2. kaynak `pct_formalities_checker.py:430-434` istemi yalnız `specification` metni içinde arıyor
     (`CLAIMS` başlığı veya `N.` satırları), :454'te CRITICAL'i basıyor;
  3. K2 deneyi: specification = tarifname + `CLAIMS` başlığı + istemler verilince aynı araç
     `total_claims:15, claim_numbers:[1..15], compliant:true`, critical=0.
- fiili başvuru pratiğinde istemler ayrı yaprak (PCT Rule 8/9); araç tek-metin varsayıyor. Bulgu,
  aracın göremediği girdiden kaynaklı; "istem yok" **sonucu çıkarılmaz**.
- `ready_to_file:false` ayrıca iki **her zaman atlanan** kontolden de kaynaklanıyor
  (`pct_formalities_checker.py:651`: `ready = not checks_skipped and …`).

### 2.5 L — paket bütünlüğü: 100 · "[OK] 6 documents scanned, no issues found."

## 3. Atlanan kontroller ( gizlenmez )

| Koşum | Atlanan | Gerekçe (araç metni) |
|---|---|---|
| A (claims) | — | `checks_skipped: []` — antecedent bu turda **koştu** |
| C (spec) | — | kayıtta atlanan listesi boş |
| D (US formal) | `figure_numeral_coverage` | "no detailed description supplied" — US `DETAILED DESCRIPTION` başlığı olmadığı için rakam kapsaması taranamadı (2.3 ile aynı kök) |
| K / K2 (PCT formal) | `physical` | "cannot be verified from text; manual inspection required" — fiziksel kâğıt şartları metinden denetlenemez |
| K / K2 | `abstract figure designation` | özet şekli başvuru formunda seçilir (PCT Rules 3.3(a)(iii), 8.2) — form kontrolü manuel |

## 4. Bu turda KOŞMAYANLAR

- EPO üçlüsü (`review_epo_claims/specification`, `check_epo_formalities`) ve TR kopya kontrolleri —
  komut US+PCT hedefli; EPO/TR kayıtları 10:52 koşumunda mevcut (`patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/`).
- Patent veritabanı ön-sanat araması: yapılmadı (SerpApi/BigQuery anahtarları yok) — değişen bir şey yok.

## 5. Hüküm (vekil onayı gerekli)

**İyi:** antecedent kontrolü ilk kez MCP hattında koştu ve beşinci kontrol olmasına rağmen istem
bulgusu 43→41'e indi; tarifname desteği üç turda 47→36→17; kapsam %100 sabit; paket bütünlüğü 100;
K-PCT "istem yok" CRITICAL'ının araç girdi sınırlandığı olduğu deneyle gösterildi. Sayılar 10:52
bağımsız koşumla birebir örtüştü — ölçüm tekrarlanabilir.

**Kötü / açık:** 41 antecedent bulgusu LOW güven + FP uyarısıyla geldi ve location alanları güvenilmez;
bunların gerçek hukuki ayıklanması vekil işi. İki CRITICAL kayıtta duruyor: US başlıkları (bilinçli
PCT tercihi, ulusal aşamada vekil kararı) ve K'nın artifakt CRITICAL'ı (raporda gerekçeli; araçtan
bağımsız giderim istem sayfalarının birleştirilmesidir ki bu PCT pratiğine aykırı). Kalan 17
yazılı-tanım uyarısından 8'i araç indeksleme artığıyla uyumlu; iki tam deyim gerçekten tek-deyiş —
destek güçlendirmesi vekile sorulacak. "Patentlenebilir / başvurmaya hazır" hükmü kurulmaz.
