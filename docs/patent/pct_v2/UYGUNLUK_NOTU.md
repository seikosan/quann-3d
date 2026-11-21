# UYGUNLUK NOTU — pct_v2 PCT paketi (iç denetim defteri, başvuru metni değildir)

künye: bu dosya §1'deki kural matrisine bağlıdır · tarih 2026-11-19 · yazan ana oturum
Denenen paket: `docs/patent/pct_v2/filing/` altındaki altı başvuru kopyası + çalışma kopyaları
`DESCRIPTION_EN.md` / `DESCRIPTION_TR.md` / `CLAIMS_EN.md` / `ISTEMLER_TR.md` / `OZET_PCT.md`.
Sayıların ham kaynağı: `OLCUM_KAYITLARI.md` (S0–S11). Motor ham çıktıları:
`patent-cikti/pct/pct_v2-pct-inceleme/*.json`. Kapı komutu: `bash patent-cikti/pct/pct_v2-pct-inceleme/kapilar.sh`.

Motorun kendi kapsam satırı (A_pct_en.json `verdict_scope`): *"Automated text checks only …
This is not a filing clearance."* Bu not da bir başvuru izin belgesi değil, vekile giden iç listedir.

## 1. Kural matrisi

Her satır bir kural maddesi; **durum** sütununda `eksik`/`bekleniyor` yazan madde bu paketin
karşılamadığı veya karşılayamayacağı işdir ve §5'te karşılığı vardır. Kanıt hücresinde
`dosya:satır = DEĞER` biçimindeki her künye yazım anında ham kaynakta doğrulandı; aynı
adresları her koşumda yeniden açan kapı `patent-cikti/pct/pct_v2-pct-inceleme/dogrul_tablolar.py`
(§2'deki motor değerlerini de yeniden okur).

| Kural | İstenen | Paketin durumu | Kanıt (bu oturumda okundu) |
|---|---|---|---|
| PCT Rule 4.3 | Başlık kısa ve teknik alanla uyumlu | EN altı · TR yedi kelime; iki dilde de `compliant: true` | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:12 = 6` · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:13 = true` · `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:12 = 7` · `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:13 = true` |
| PCT Rule 5.1(a) | Teknik alan · önceki teknik · buluşun açıklanması · şekillerin kısa açıklanması · en iyi biçim (+ sanayide uygulanabilirlik) | Altı başlık iki dilde de sırayla; motorun denetlediği beş alanın eksik listesi boş | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:24 = []` · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:25 = true` · `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:24 = []` · `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:25 = true` · `docs/patent/pct_v2/DESCRIPTION_EN.md:11,21,39,67,87,217` (EN başlıkları) · `docs/patent/pct_v2/DESCRIPTION_TR.md:11,21,39,67,87,217` (TR başlıkları) |
| PCT Rule 6.1(b) | İstemler ardışık numaralı | On beş istem; numaralar 1–15 kesintisiz | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:28 = 15` · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:29-45` (`claim_numbers` 1…15) · `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:28 = 15` · `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:40 = 15` · `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:82 = 15` |
| PCT Rule 6.2(a) | İstem gövdesinde tarifname/şekil atfı yok | EN gövdesinde tarama boş; TR'deki tek eşleşme "hesaplayacak **şekilde**" sözcüğü — şekil atfı değil | `docs/patent/pct_v2/CLAIMS_EN.md:10-66` gövde taraması (`grep -c "FIG\."` = 0) · `docs/patent/pct_v2/ISTEMLER_TR.md:11-39` taraması; eşleşme `docs/patent/pct_v2/ISTEMLER_TR.md:37` |
| PCT Rule 6.2(b) | Referans işaretleri, anlaşmayı hızlandırmıyorsa zorunlu değil | Bilinçli tasarım kararı: hiç işaret yok (numara şeması uydurulmadı); kural bu eksikliği uyarıya çeviriyor | `docs/patent/pct_v2/CLAIMS_EN.md:7` tasarım notu · `patent-cikti/pct/pct_v2-pct-inceleme/G_epo_formal_en.json:11 = false` (`has_reference_signs`) · `patent-cikti/pct/pct_v2-pct-inceleme/G_epo_formal_en.json:9` ("Missing reference signs") · `patent-cikti/pct/pct_v2-pct-inceleme/G_epo_formal_en.json:73 = true` (`overall_compliant`) · → §4.4 |
| PCT Rule 5.2(a) / 11.13(m) | Paragraf numaralandırması | İki tarifnede de yetmiş paragraf, aralıksız | `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:14 = 70` · `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:15 = true` · `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:56 = 70` · `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:57 = true` |
| PCT Rule 8.1(b) | Özet elli–yüz elli kelime | EN yüz otuz yedi · TR yüz altı kelime; iki dilde `compliant: true` | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:6 = 137` · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:7 = true` · `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:6 = 106` · `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:7 = true` · `docs/patent/pct_v2/OZET_PCT.md:48-49` (`wc -w` ölçümü) |
| PCT Rule 9.1 (birlik) | Tek genel konsept; çokluk bağımlı istemle toplanır | Üç bağımsız (yöntem 1 · sistem 13 · kayıt ortamı 15) + on iki bağımlı; yerel lint `claim_unity` **info** | `patent-cikti/pct/pct_v2-pct-inceleme/lint_CLAIMS_EN_md.txt:26 = info` · `patent-cikti/pct/pct_v2-pct-inceleme/lint_ISTEMLER_TR_md.txt:11 = info` · `docs/patent/pct_v2/CLAIMS_EN.md:10` `docs/patent/pct_v2/CLAIMS_EN.md:52` `docs/patent/pct_v2/CLAIMS_EN.md:66` · `docs/patent/pct_v2/ISTEMLER_TR.md:11` `docs/patent/pct_v2/ISTEMLER_TR.md:35` `docs/patent/pct_v2/ISTEMLER_TR.md:39` |
| PCT Rule 11.13(l)–(n) | Metin temizliği: birim/terim/sıcaklık tutarlılığı | Terim denetimi iki dilde de temiz: metrik dışı birim ve aykırı sıcaklık listeleri boş | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:50 = []` · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:51 = []` · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:52 = true` · `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:50 = []` · `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:51 = []` · `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:52 = true` |
| AGDD §205 / Kural 5.2(a) | Şekillerin kısa açıklanması gerçek şekil listesine bağlı | Şekil 1–8 için ayrı paragraf; ölçüm taşıyanlar Şekil 6–7 olarak ayrıca işaretli | `docs/patent/pct_v2/DESCRIPTION_EN.md:83` (FIG. 8 tarif)  · `docs/patent/pct_v2/DESCRIPTION_EN.md:85` (ölçüm taşıyan şekiller) · `docs/patent/pct_v2/DESCRIPTION_TR.md:83` · `docs/patent/pct_v2/DESCRIPTION_TR.md:85` |
| AGDD 5.2(b) — özetle okunacak şekil | Özetle birlikte yayınlanacak şekil talep formunda işaretlenir | **eksik/bekleniyor** — özet metninde şekil atfı yok; motor bu kontrolü çalıştırmıyor, işaret PCT/RO/101'de verilecek | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:9 = not checked` · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:106-107` (atlanan kontrol + gerekçe) · `~/patent-creator-venv/lib/python3.13/site-packages/mcp_server/pct_formalities_checker.py:238-241` (motor kaynağı) · `patent-cikti/pct/pct_v2-denetmen/DENETIM_OZET.md:59` (kapsam dışı listesi) |
| AGDD 5.2(b) — fiziksel vurgu | Kalın/altı çizili/tasarım tarifi kuralları | **eksik/bekleniyor** — başvuru metinlerinde (`.txt` nüshalar ve EN/TR kaynaklarında kalın-italikayrımı yok) kalın ve altı çizili yok; fiziksel çıktı vekilde basılacak | `grep -c '\*\*\|__' docs/patent/pct_v2/filing/*.txt` = 0/0/0/0/0/0 (bu oturumda koştu, başvuru metinleri) · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:102-103` (`physical` atlandı) · `~/patent-creator-venv/lib/python3.13/site-packages/mcp_server/pct_formalities_checker.py:231-232` (motor kaynağı) |
| PCT Rule 13.1 | Şekiller ayrı sayfalarda, `FIG.` etiketli; şekil listesi tarifnamede | **karşılanıyor (üçüncü tur)** — sekiz şekil `filing/SEKILLER_FILING-COPY.pdf` içinde ayrı sayfalarda, her sayfada tek `FIG. n` etiketi; içerik denetimi bağımsız kapıyla yapıldı (bkz. §5 sonu ve §7) | `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json` (A–E, `bulgu_toplam: 0`) · `docs/patent/pct_v2/DESCRIPTION_EN.md:67` (şekil listesi) · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:9 = not checked` (motor hâlâ metinden denetlemiyor) |
| PCT Rule 11.2–11.16 | Fiziksel biçim: A4, marj, satır numarası, punto, sayfa üstü | **çizim seti için ölçüldü / metin sayfaları bekleniyor** — şekil PDF'i A4 (595,304 × 841,89 pt), sekiz sayfa, her sayfada mürekkep, metin katmanı 0 karakter, en küçük yazı 0,2175 cm ≥ 0,21 cm (Kural 11.13); tarifname/istem metinlerinin dizgisi (satır numarası, sayfa üstü) vekilin Word/LaTeX nüshasında üretilir | `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json` (E kapısı: `pdf.sayfa = 8`, `pdf.pt`, `min_yazi_cm = 0,2175`) · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:102-103` (`physical` + gerekçe) · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:110` (`verdict_scope`) · `~/patent-creator-venv/lib/python3.13/site-packages/mcp_server/pct_formalities_checker.py:231-232` · `~/patent-creator-venv/lib/python3.13/site-packages/mcp_server/pct_formalities_checker.py:252-255` ("This is not a filing clearance.") |
| PCT Rule 19.2 | Değiştirilmiş sayfa gönderimi (istemin tarife göre düzeltilmesi) | **bekleniyor** — uluslararası arama raporundan sonra; bu oturumda gönderilecek bir değişiklik sayfası yok | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:110` (`verdict_scope`) · `patent-cikti/pct/pct_v2-denetmen/DENETIM_OZET.md:57-60` (denetlenemedi listesi) |
| PCT/RO/101 talep formu | Başvuru formu: mucitler, vekâletname, beyanlar, şekil atfı | **eksik** — paket metinlerden oluşuyor, form hazırlanmadı | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:110` ("forms … require manual review") · `patent-cikti/pct/pct_v2-denetmen/DENETIM_OZET.md:59` |
| PCT Rule 6.4(b) / 3.2 — ücret ve vekâletname | İade ücreti ve vekâletname beyanı | **eksik** — ücret hesabı ve vekâletname vekil/ofis işi; bu paket ücret iddiası taşımıyor | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:110` ("fee payment require manual review") · `patent-cikti/pct/pct_v2-denetmen/DENETIM_OZET.md:59` |
| EPO UygK 45(2) / 46(1) | İstemler tarifnameyle destekli; tarifname buluşu açıklar | On beş istemin on beşi destekli, kapsama yüzde yüz, karşılanmayan istem yok | `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:629 = true` · `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:631 = 100` · `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:632 = 15` · `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:634 = []` |
| EPO UygK 43(1)–(2) · İBK F-II 3.7 | İstem açık ve öz; kategori açık | EN'de on beş `Rule 43(1)` eşleşmesinin on beşi de İngilizce-belirteç artefaktı; TR istem linti **0 hata**; gerçek kusur sayacı sıfır | `patent-cikti/pct/pct_v2-pct-inceleme/D_epo_claims_en.json:4 = 1` · `patent-cikti/pct/pct_v2-pct-inceleme/D_epo_claims_en.json:5 = 0` · `patent-cikti/pct/pct_v2-pct-inceleme/E_epo_claims_tr.json:4 = 16` · `patent-cikti/pct/pct_v2-pct-inceleme/E_epo_claims_tr.json:5 = 0` · `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_antecedent.json:10 = 0` (`D_GECEK_KUSUR`, EN) · `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_antecedent.json:113 = 0` (TR) · → §4.1–§4.2 |
| İBK F-IV 5.1 | Bağımsız istem iki parçalı: ön izleme + ayırt edici bölüm | Üç bağımsız istemin her birinde tam bir `characterized in that` / `karakterize edilir`; bağımlı kuyrukları `tanımlıdır.` | `docs/patent/pct_v2/CLAIMS_EN.md:10` `docs/patent/pct_v2/CLAIMS_EN.md:52` `docs/patent/pct_v2/CLAIMS_EN.md:66` (grep toplamı 3) · `docs/patent/pct_v2/ISTEMLER_TR.md:11` `docs/patent/pct_v2/ISTEMLER_TR.md:35` `docs/patent/pct_v2/ISTEMLER_TR.md:39` (grep toplamı 3) · `docs/patent/pct_v2/ISTEMLER_TR.md:13` (örnek bağımlı kuyruk; grep toplamı 12) |
| İBK H-V 1 | Teknik sorun + teknik etki ölçülü biçimde belirtilmiş | Etki yalnız ölçülen eksenler üzerinde ve ölçüm koşuluyla yazıldı; doğruluk üstünlüğü iddiası yok | `docs/patent/pct_v2/DESCRIPTION_EN.md:123` (k=13→50 maliyet sınırı) · `docs/patent/pct_v2/DESCRIPTION_TR.md:123` · `docs/patent/pct_v2/OLCUM_KAYITLARI.md:51-52` (oranların ham künyesi) |
| İki dillilik | TR metin EN aslının birebir karşılığı; terimler tek biçimli | Yetmiş–yetmiş paragraf, on beş–on beş istem; bölüm başlıkları iki dilde paralel — **birebir çeviri denetimi yapılmadı**, yalnız yapı + ana terim listesi karşılaştırıldı | `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:14-15` · `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:56-57` · `docs/patent/pct_v2/DESCRIPTION_TR.md:11,21,39,67,87,217` · → §5 |

Toplam 22 satır · 6 satırda durum açıkça `eksik`/`bekleniyor` · 1 satır `kısmen` — bu üç sayı `patent-cikti/pct/pct_v2-pct-inceleme/yeniden_bolum1.py:39-147` `SATIRLAR` listesindeki durum metinlerinden üretilir.

## 2. Motor bataryası — ham sonuç (koşum 2026-11-19 12:10, `run_pct_analizi.py`)

| Koşu | Kapsam | Değerler — her biri kendi JSON satırında | Yorum |
|---|---|---|---|
| A_pct_en | PCT biçimi, EN | Kritik `0` — `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:86`<br>Uyarı `0` — `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:87` | `issues: []`; `overall_compliant: false` **yalnız** atlanan iki kontrol yüzünden (§5) |
| B_pct_tr | PCT biçimi, TR | Kritik `0` — `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:86`<br>Uyarı `0` — `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:87` | aynısı; `overall_compliant: false` yine yalnız atlanan kontrollerden |
| C_pct_en_desc_only | Yalnız tarifname | Kritik `1` — `patent-cikti/pct/pct_v2-pct-inceleme/C_pct_en_desc_only.json:83`<br>Uyarı `0` — `patent-cikti/pct/pct_v2-pct-inceleme/C_pct_en_desc_only.json:84` | „No claims detected“ — beklenen; motor istemi görmek için tarifname+istem **birleşik** ister (A bunu gösterir) → §4.6 |
| D_epo_claims_en | EPO istem biçimi, EN | Bulgu `1` — `patent-cikti/pct/pct_v2-pct-inceleme/D_epo_claims_en.json:4`<br>Kritik `0` — `patent-cikti/pct/pct_v2-pct-inceleme/D_epo_claims_en.json:5`<br>Önemli `1` — `patent-cikti/pct/pct_v2-pct-inceleme/D_epo_claims_en.json:6`<br>Skor `99.66666666666667` — `patent-cikti/pct/pct_v2-pct-inceleme/D_epo_claims_en.json:22` | kalan tek bulgu Rule 43(2) kategori → §4.1 artefaktı |
| E_epo_claims_tr | EPO istem biçimi, TR | Bulgu `16` — `patent-cikti/pct/pct_v2-pct-inceleme/E_epo_claims_tr.json:4`<br>Kritik `0` — `patent-cikti/pct/pct_v2-pct-inceleme/E_epo_claims_tr.json:5`<br>Önemli `16` — `patent-cikti/pct/pct_v2-pct-inceleme/E_epo_claims_tr.json:6`<br>Skor `94.66666666666667` — `patent-cikti/pct/pct_v2-pct-inceleme/E_epo_claims_tr.json:187` | 15× Rule 43(1) + 1× Rule 43(2) → §4.2, İngilizce-only kalıp |
| F_epo_spec_support_en | İstem–tarifname desteği | Kritik `0` — `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:10`<br>Önemli `47` — `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:11`<br>Coverage `100` — `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:631`<br>Desteklenen `15` — `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:632`<br>Karşılanmayan `[]` — `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:634` | `compliant: true`; 47 tavsiye → §4.3 |
| G_epo_formal_en | EPO biçimsel | Kritik `0` — `patent-cikti/pct/pct_v2-pct-inceleme/G_epo_formal_en.json:67`<br>Uyarı `1` — `patent-cikti/pct/pct_v2-pct-inceleme/G_epo_formal_en.json:68`<br>Genel uygunluk `true` — `patent-cikti/pct/pct_v2-pct-inceleme/G_epo_formal_en.json:73` | kalan uyarı: özetlerde parantezli işaret (tasarım kararı) → §4.4 |
| H_pkg_full | 12 dokümanlık paket | Bulgu `0` — `patent-cikti/pct/pct_v2-pct-inceleme/H_pkg_full.json:4`<br>Kritik `0` — `patent-cikti/pct/pct_v2-pct-inceleme/H_pkg_full.json:5`<br>Skor `100.0` — `patent-cikti/pct/pct_v2-pct-inceleme/H_pkg_full.json:10` | `filing_copy_purity` CRITICAL **kalktı** |
| I_pkg_filing | 6 dosyalık kopya | Bulgu `0` — `patent-cikti/pct/pct_v2-pct-inceleme/I_pkg_filing.json:4`<br>Kritik `0` — `patent-cikti/pct/pct_v2-pct-inceleme/I_pkg_filing.json:5`<br>Skor `100.0` — `patent-cikti/pct/pct_v2-pct-inceleme/I_pkg_filing.json:10` | `stamps: []` |
| J_us_claims_en | US istem denetimi | Bulgu `0` — `patent-cikti/pct/pct_v2-pct-inceleme/J_us_claims_en.json:4`<br>Kritik `0` — `patent-cikti/pct/pct_v2-pct-inceleme/J_us_claims_en.json:5`<br>Skor `100.0` — `patent-cikti/pct/pct_v2-pct-inceleme/J_us_claims_en.json:10` | `antecedent_basis` opt-in kapalı (§5) |

Not: A/B/C/G satırlarında motor **`warnings`** alanını bildiriyor (Kritik =
`critical_issues`); D/E/F/H/I/J satırlarında **`important_issues`** var. Her hücredeki sayı,
kendi künyesinin açıldığı satırda birebir geçiyor; F'nin „Karşılanmayan" hücresi boş listedir
(`[]`). Skorlar motorun ham ondalık değeri, yuvarlanmadı. Bu tablodaki hiçbir sayı elle
yazılmadı: değerler ve satır numaraları `yeniden_bolum2.py` tarafından JSON'dan okundu.

## 3. Bu turda yapılan düzeltmeler (hangisi hangi kapıyı yeşile çevirdi)

Denetmen turunda eklenen düzeltmeler ayrıca §6'da; aşağıdaki liste şekil kuralı bulguları içindir.

| Düzeltme | Metindeki karşılığı | Kapıya etkisi |
|---|---|---|
| EN bağımsız istem 1/13/15 iki parçalı biçime getirildi ("characterized in that"); TR karşılıklarının ön yükü önceki tekniği anlatan gövde + "karakterize edilir" olarak yeniden yazıldı | `docs/patent/pct_v2/CLAIMS_EN.md:10` · `docs/patent/pct_v2/CLAIMS_EN.md:52` · `docs/patent/pct_v2/CLAIMS_EN.md:66` | D'nin Rule 43(1) bulguları kalktı; bugün tek bulgu Rule 43(2) (kural §1, ölçüm §2) |
| [0063]'teki saha ifadesi istem terimiyle eşleştirildi: "0.64 on 25 critical records" → "0.64 on 25 records of the critical stage" (TR: "kritik kademedeki 25 kayıtta") | `docs/patent/pct_v2/DESCRIPTION_EN.md:199` (TR aynası aynı satırda) | H/I `filing_copy_purity` CRITICAL'i kalktı; paket bütünlüğü tam not (§2 H/I) |
| [0065] istem on'un terimini aldı: "a text consisting of a plurality of sentences" | `docs/patent/pct_v2/DESCRIPTION_EN.md:207` | F'nin CRITICAL'lerinden biri kalktı (§2 F) |
| [0050] istem on dört'ün terimini aldı: "a distribution store holding" (TR: "bir dağıtım deposu taşır") | `docs/patent/pct_v2/DESCRIPTION_EN.md:159` | F'nin ikinci CRITICAL'i kalktı (§2 F) |
| [0034] istem dört'ün terimleriyle genişletildi: "a result of the max-pooling" / "a result of the average-pooling" | `docs/patent/pct_v2/DESCRIPTION_EN.md:105` · `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:631-633` | F kritikleri sıfırlandı, coverage tam (§2 F) |
| Bölüm başlığı "BEST MODE(S) FOR CARRYING OUT THE INVENTION" → "BEST MODE FOR CARRYING OUT THE INVENTION" | §1 Rule 5.1(a) satırı; G koşusu ölçümü §2'de | G'deki "Missing required description section" CRITICAL'i kalktı |
| Öncül dayanak ve anlatım netleştirmeleri (EN + TR aynası); [0028] ve [0068]'deki "only/must", "yalnız/zorunda" ifadeleri çıkarıldı (yerel `buzzword` uyarıları) | `docs/patent/pct_v2/CLAIMS_EN.md:32-66` arası istem satırları | TR istem dosyalarında hata kalmadı (`patent-cikti/pct/pct_v2-pct-inceleme/lint_ISTEMLER_TR_md.txt:1`, §2 son satır) |

## 4. Düzeltilmeyen bulgular — araç artefaktı (her biri kanıt satırıyla)

Artefakt demek "metni düzeltmiyorum" demektir; o yüzden her iddia motorun kendi satırına bağlıdır.
Kanıt üretici: `patent-cikti/pct/pct_v2-pct-inceleme/kanit_artefaktlar.py` → `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_artefaktlar.json`
(üretici betik motor kalıplarını olduğu gibi kopyalar, yorum katmaz).

| No | Bulgu | Neden metinde düzeltilmedi | Kanıt satırı |
|---|---|---|---|
| §4.1 | **D — Rule 43(2): "Multiple independent claims (1, 13, 15) in the same category (product)"** | Motor kategoriyi yalnız İngilizce kalıplardan çıkarıyor (`^a\s+method\b`, `^a\s+system\b`, aksi hâlde `return "product"`). Oysa istem 1 = *method*, 13 = *computer system*, 15 = *recording medium*: fiilen üç kategori. | `~/patent-creator-venv/lib/python3.13/site-packages/mcp_server/epo_claims_analyzer.py:186-192` · `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_artefaktlar.json:7-9` (üçünün değeri de `product`) |
| §4.2 | **E — TR'de on beş "bağımsız" istem ve "iki parçalı biçim yok"** | Motor kalıpları İngilizce (`characteri[sz]ed\s+in\s+that`, `wherein\s+the\s+improvement\s+comprises`); bağımlılık atfını İngilizce `claims?\s+\d` ile arıyor. TR metni "karakterize edilir" ve "İstem 1'e göre" yazıyor. | `~/patent-creator-venv/lib/python3.13/site-packages/mcp_server/epo_claims_analyzer.py:194-236` · `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_artefaktlar.json:24-27` (TR'de kararlı ifade var, İngilizce eşleşme `false`) · `:42-44` (TR bağımlı sayımı 0, EN bağımlı sayımı 12) |
| §4.3 | **F — tavsiyeler: "has limited description (only Para. N)"** | Motor sayılandırılmış paragraf bulamazsa bölüm dizinine düşüyor (`_index_by_sections`) ve terimi tek paragrafla eşliyor. Örnekte "point cloud" tarifnamede on üç yerde geçiyor, motor bir paragraf sayıyor. | `~/patent-creator-venv/lib/python3.13/site-packages/mcp_server/epo_specification_analyzer.py:245-250` · `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_artefaktlar.json:47-49` |
| §4.4 | **G — Rule 47(4): "Abstract does not contain reference signs in parentheses"** | Pakette bilinçli olarak referans işareti yok (§1, Rule 6.2(b) satırı); bulgu EPO bölgesel pratiğine dair tasarım kararı. | `patent-cikti/pct/pct_v2-pct-inceleme/G_epo_formal_en.json:62` (kural) · `:68` (uyarı sayısı) · `:73` (`overall_compliant: true`) |
| §4.5 | **Yerel hat — `antecedent_basis` hataları, 0 gerçek kusur** | Hat kabul eşiği "artikel + en fazla iki ara kelime"; sınıflandırıcı motorun `prior`/tetik hesabını birebir tekrarlıyor ve hataları sekizi **B** (artikel ile ad arasında üçten çok niteleme), beşi **A** (belirteçsiz çoğul tanım), ikisi **C** (ifadeye sıfat eklenmiş), sıfırı gerçek kusur buldu. Motorun kendisi de bu denetimi opt-in kapalı tutuyor. | `~/Documents/patent-hatti/patentline/linter.py:198-223` · `patent-cikti/pct/pct_v2-pct-inceleme/kanit_antecedent.py` (üretici betik) · `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_antecedent.json:5` · `:7` A · `:8` B · `:9` C · `:10` D=0 · `patent-cikti/pct/pct_v2-pct-inceleme/J_us_claims_en.json:9` (`false-positive` gerekçesi) |
| §4.6 | **Tarifne lint'inde "claim_structure: Hiç istem bulunamadı"; istem lint'inde `section_missing`** | Tek doküman lint edildiğinde kaçınılmaz: tarifnede istem, istem dosyasında bölüm başlıkları yok. Birleşik metin koşusu temiz. | `patent-cikti/pct/pct_v2-pct-inceleme/C_pct_en_desc_only.json:73` (CRITICAL bu yüzden) · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:86-87` (birleşik metinde sıfır) |
| §4.7 | **`package_checker` saflık kalıbı `\b\d+\s+critical\b`** | Kalıp meşru ölçüm düz yazısını kir sayabiliyor; §3'ün [0063] satırındaki yeniden yazımla bulgu gerçek düzeltmeye dönüştü, kalıbın kendisi sezgisel kaldı. | `~/patent-creator-venv/lib/python3.13/site-packages/mcp_server/package_checker.py:55-61` · `patent-cikti/pct/pct_v2-pct-inceleme/H_pkg_full.json:10` (bütünlük tam) |
## 5. Koşulmayan kontroller (atlananlar — "temiz" denmeyen yerler)

Her satır motorun kendi kaydına bağlıdır; bu bölümde hiçbir kontrol "temiz" sayılmaz.

| Koşulmayan kontrol | Neden koşulmadı | Kanıt satırı |
|---|---|---|
| Fiziksel biçim — **yalnız metin sayfaları** (PCT Rule 11.2–11.16: satır numarası, satır arası, sayfa üstü, yazı tipi) | Motor düz metinden fiziksel gereklilikleri denetleyemediğini kendisi bildiriyor: "physical requirements cannot be verified from text; manual inspection required". Kapsam daraldı: **çizim seti artık bu satırın dışında** — sekiz şekil A4 PDF olarak üretildi ve E kapısıyla ölçüldü (§1 Rule 11.2–11.16 satırı). | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:102-103` (`"check": "physical"` + gerekçe) · `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json` (E kapısı) |
| Özette şekil tarifi (Rule 8.1(b)) | Özetle hangi şeklin yayımlanacağı başvuru formunda seçilir; metinden bilinemez: "the figure to publish with the abstract is chosen on the request form". | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:105-107` (`"check": "abstract figure designation"`) |
| Yenilik / ön sanat araması — **3. turda YAPILDI** (1.–2. turda yapılamamıştı) | Patent + NPL veritabanı taraması koşuldu: `42` sorgu, `161` hit kaydı, `160` benzersiz hit, `9` kayıt 1. istem düzeyinde; her sorgunun ham yanıtı diskte. Rule 33 (ISRS) anlamında resmî arama raporu yine yok; Espacenet/PATENTSCOPE + CPC sınıf taraması vekil işi. | sayım: `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_db_taramasi.txt:54-56` · künye: `docs/patent/pct_v2/ON_SANAT.md:185` · anahtarlı araç tanımı: `~/patent-creator-venv/lib/python3.13/site-packages/mcp_server/tools/bigquery_tools.py:96-97` |
| ~~Şekil dosyalarının içeriği (Rule 11.12–11.14)~~ — **KOŞULDU** | Motorun `drawings_present` beyanı hâlâ sadece beyan: kendi kapsam satırı "a declaration, not inspection of drawing files" diyor. Onun yerine **harici muayene koşullu**: beş kapılı bağımsız doğrulayıcı SVG × künye × tarifname × ham ölçüm JSON'u × basılı PDF'i karşılaştırır; `0 bulgu`. Motor kaydındaki `overall_compliant: false` bundan sonra **metin-dışı idari katman** içindir, şekiller için değil. | `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json` (`ozet.bulgu_toplam = 0`) · betik `patent-cikti/pct/pct_v2-pct-inceleme/dogrul_figurler.py` · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:110` (`verdict_scope`) |
| US öncül dayanak denetimi | Motor bu denetimi opt-in kapalı tutuyor; `PATENT_ENABLE_ANTECEDENT_CHECK=1` hiç koşulmadı. Yerine yerel hat koşusu ve §4.5 sınıflandırması var. | `patent-cikti/pct/pct_v2-pct-inceleme/J_us_claims_en.json:17-19` (kural + `enable_with`) |
| İdari katman | Form PCT/RO/101, ücret ve indirim hesabı, vekaletname, RO TÜRKPATENT başvuru dili tercihi, Rule 26bis beyanı, dizim listesi (ST.26) kapsam dışı. | idari veri bu depoda üretilmiyor; §8 vekil onayı satırı |
| Bağımsız onay | Bu notun bulguları ana oturumdan; vekil ve bağımsız doğrulayıcı turu olmadan başvuru kararı verilmez. | §8 karar satırı |

## 6. Denetmen turunda düzeltilen kusurlar (ikinci tur, 2026-11-19)

Bağımsız denetmenin (`patent-denetmen`) bulgu listesi ham metne karşı sınandı; gerçek olanlar
uygulandı. Düzeltici betik `patent-cikti/pct/pct_v2-pct-inceleme/duzelt_denetmen_bulgulari.py`
her düzeltmeyi ham metinde beklenen eşleşme sayısı kadar bulursa uygular, değilse betik hiçbir
şeyi değiştirmeden çıkar. Koşum çıktısı aynı dizinde duruyor.

| Kusur | Yapılan | Doğrulanan kaynak |
|---|---|---|
| TR bağımlı istemlerin sonu ikinci kez "karakterize edilir." ile bitiyordu (aynı istemde iki kez iki-parçalı kalıp) | bağımlı istemlerin kuyruğu EN karşılığı "wherein"e uygun şekilde "…tanımlıdır." ile bağlandı; bağımsız istemlerin birinde birer "karakterize edilir" kaldı | `docs/patent/pct_v2/ISTEMLER_TR.md:25` · `patent-cikti/pct/pct_v2-pct-inceleme/lint_ISTEMLER_TR_md.txt:1` |
| TR istem 13'te ikinci bir "karakterize edilir" bloğu daha vardı | kaldırıldı | `docs/patent/pct_v2/ISTEMLER_TR.md:35` |
| EN istem 1 (iii) "reference data set / comparison data set" diyordu; istem 8 ve tarifname "group" diyor | "for a reference data group and a comparison data group" | `docs/patent/pct_v2/CLAIMS_EN.md:26` |
| TR istem 12'de EN karşılığı olmayan "daraltılarak" kırpısı vardı | çıkarıldı; aralık daraltması tarifnamede ayrı bir seçenek olarak duruyor | `docs/patent/pct_v2/ISTEMLER_TR.md:33` ↔ `docs/patent/pct_v2/DESCRIPTION_TR.md:169` |
| TR [0010] ile EN [0010] arasındaki "okuyucu girdisi" bağı kopuktu | TR aynası yazıldı | `docs/patent/pct_v2/DESCRIPTION_TR.md:41` ↔ `docs/patent/pct_v2/DESCRIPTION_EN.md:41` |
| [0057] "4096 girdi" örnekemin niteliğini söylemiyordu | ham kaynağın niteleyicisi metne taşındı: "4096 adet rasgele seçilmiş QUANN-noktası vekil (proxy) girdi" | `fraud/logs/run_full_scale_edge_eval.log:46` → `docs/patent/pct_v2/DESCRIPTION_TR.md:181` |
| TR başlığı motorun başlık sınırını aşıyordu | başlık yeniden kuruldu | `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:12` |

**Üçüncü tur — şekil seti ve ön sanat (tarih dosyanın başındaki künyededir).**
Koşulan bağımsız kapı: `patent-cikti/pct/pct_v2-pct-inceleme/dogrul_figurler.py`

| Kusur | Yapılan | Doğrulanan kaynak |
|---|---|---|
| Çizimlerde tarifnamede geçmeyen terimler (`illustrative`, `unchanged/shifted setting`, `sliding`) geçersiz sayıldı; B kapısı kelime düzeyine baktığı için söz öbeklerini kaçırdı | çizim metinleri tarifnamenin kendi sözcüklerine çekildi: `centre` · `geometric regions` · `setting shift` · `slid by a fixed window size and stride` · `centroid-based clustering` | çizim künyesi `docs/patent/pct_v2/fig/fig_manifest.json:402` · `:438` · `:989` ↔ tarifname `docs/patent/pct_v2/DESCRIPTION_EN.md:49` (`cloud centre`) · `:137` (`geometric regions`, `centroid-based clustering`) · `:79` (`before and after an applied setting shift`) · `:115` (`slid by a fixed window size and a fixed stride`) · B kapısı `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:16` (`kontrol_edilen_kelime: 339`) |
| FIG. 6 panel (a) ölçüleri yanlış JSON'a atfen künyelenmişti (üretici kaynak etiketini üretim sonunda okuyordu) | her ölçü etiketi üretildiği andaki kaynağını taşır; FIG. 6 künyesi iki kaynağı ayrı sayar (`complaints_scale_stability.json` 8 · `e1_cmapss.json` 3) | C kapısı `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:25` (`olcu_deger: 15`) · `docs/patent/pct_v2/fig/fig_manifest.json` (`texts[].source`) |
| `axis("off")` sayfalarda çizilmeyen eksen tikleri künyeye sızıyordu | künyeye yalnız çizilen yazılar girer | A kapısı `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:8` (`kunyi_metin: 142`, SVG ile birebir) |
| ON_SANAT §1.2'de iki hatalı DOI, §1.3'te kendi `BULUNAMADI` etiketiyle çelişen iki "döndü" cümlesi | DOI'ler 2. tur ham araç kaydından; "döndü" → "**dönmedi**"; 1. turun ham yanıtının diskte olmadığı §0'a yazıldı, vekile V6 kartı açıldı | `docs/patent/pct_v2/ON_SANAT.md:92-97` · `patent-cikti/pct/pct_v2-pct-inceleme/oncesanat_ham_tur2_2026-11-19.json` |

Kapının kanıtı iki parçalıdır. (i) **Canlılık, diskte doğrulanabilir:** künyeye tarifnamede geçmeyen
bir terim enjekte edildiğinde kapı RED verip hem A (künye↔SVG) hem B (terim) denetiminde bulgu
üretiyor; künye geri yüklendiğinde yeniden KABUL veriyor. Geri yükleme iddiası kapalı döngüyle
ölçüldü: enjeksiyon öncesi ve sonrası SHA-256 aynı (`5955a82c9670383424143b19b28d6758ef46b47474b990cf9c09fc0b6863cb08`),
enjekte edilmiş hâlin karması ayrı; kırmızı koşum `CIKIS=1 / RED: 2 bulgu`, geri yükleme sonrası
`CIKIS=0 / KABUL: 0 bulgu`. Ham çıktı `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_kirmizi_kapali_donunum.txt`. Bu kayıt aynı
zamanda şu düzeltmeyi taşır: `patent-cikti/pct/pct_v2-pct-inceleme/dogrul_figurler.py` her koşumda `DOGRUL_figurler.json`
dosyasını YENİDEN YAZAR — kapı bir beyan değil, çıktı üreten bir koşumdur.
(ii) **İlk koşumdaki bulgu sayısı KANITSIZ:** kapının ilk turda bulduğu tarifname-dışı terimler
oturumda görüldü ve metne uygulandı, ancak o bulgu listesi diskte kaydedilmedi; bu yüzden "kaç tane"
sorusunun yanıtı bu defterde sayı olarak verilmiyor. Temiz kabulü yalnız (i)'e ve güncel `0 bulgu`
kaydına dayanır: `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:73`.
Üreticinin kendi "0 taşma" beyanına karşılık bu kapı dışarıdan bakmıştır.

**Vekile bırakılanlar (metin değişikliği yapılmadı):**

| Madde | Neden burada değiştirilmedi | Kanıt satırı |
|---|---|---|
| TR bağımsız istemin tek cümle kuralına takılan çok sıralı yapısı | EN'de de aynı uzunlukta; kapsam daraltmadan tek cümleye indirmek bağımsız istemin korunmasını riske atıyor | `docs/patent/pct_v2/ISTEMLER_TR.md:35` ↔ `docs/patent/pct_v2/CLAIMS_EN.md:52` |
| Çizim referans numaraları şeması | Sayı uydurmamak için konmadı (§1, Rule 6.2(b) satırı); tarifname şekilleri `FIG. N` olarak anıyor | `docs/patent/pct_v2/DESCRIPTION_EN.md:105` (FIG. 3 anışı) |
| TR [0057]'deki "vekil (proxy)" niteleyicisinin kapsam iddiasına etkisi | Niteleyici ham örnekemin kendi adlandırması; metne taşındı ama kapsam yorumu hukuki karar | `fraud/logs/run_full_scale_edge_eval.log:46` → `docs/patent/pct_v2/DESCRIPTION_TR.md:181` |

## 7. Sayıların ham kaynakları (bu oturumda doğrulandı)

Üretici: `patent-cikti/pct/pct_v2-pct-inceleme/atif_haritasi.py` — her satır, atfedilen satırda
sayının gerçekten geçtiğini gösterir (rapor kapısının istediği şey).

| İddia | Kaynak | Ham satır |
|---|---|---|
| Başlık EN kelime sayısı | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:12` | `"word_count": 6` |
| Özet EN kelime sayısı | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:6` | `"word_count": 137` |
| İstem sayısı EN | `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:28` | `"total_claims": 15` |
| Başlık TR kelime sayısı | `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:12` | `"word_count": 7` |
| Özet TR kelime sayısı | `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:6` | `"word_count": 106` |
| D bulgu sayısı / skoru | `patent-cikti/pct/pct_v2-pct-inceleme/D_epo_claims_en.json:4` · `:22` | `"total_issues": 1` · `"compliance_score": 99.66…` |
| E bulgu sayısı | `patent-cikti/pct/pct_v2-pct-inceleme/E_epo_claims_tr.json:4` | `"total_issues": 16` |
| F kritik / önemli sayısı | `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:10` · `:11` | `"critical_issues": 0` · `"important_issues": 47` |
| F coverage ve desteklenen istem | `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:631` · `:632` | `"percentage": 100` · `"supported_claims": 15` |
| G bulgusu (Rule 47(4)) | `patent-cikti/pct/pct_v2-pct-inceleme/G_epo_formal_en.json:62` · `:68` · `:73` | `"legal_ref": "Rule 47(4) EPC"` · `"warnings": 1` · `"overall_compliant": true` |
| H ve I paket skoru | `patent-cikti/pct/pct_v2-pct-inceleme/H_pkg_full.json:10` · `patent-cikti/pct/pct_v2-pct-inceleme/I_pkg_filing.json:10` | `"package_integrity_score": 100.0` |
| J bulgu sayısı / skoru | `patent-cikti/pct/pct_v2-pct-inceleme/J_us_claims_en.json:4` · `:10` | `"total_issues": 0` · `"compliance_score": 100.0` |
| Filing kopyalarında paragraf sayısı | `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:14` · `:56` | `"paragraph_count": 70` |
| Filing kopyalarında istem sayısı | `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:40` · `:82` | `"claim_count": 15` |
| Yerel lint: EN istem / TR istem | `patent-cikti/pct/pct_v2-pct-inceleme/lint_CLAIMS_EN_md.txt:1` · `lint_ISTEMLER_TR_md.txt:1` | `Linter: 15 hata` · `Linter: 0 hata` |
| Yerel antecedent sınıflandırması | `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_antecedent.json:5` · `:7` · `:8` · `:9` · `:10` | `15 · A=5 · B=8 · C=2 · D=0` |
| Motorun "tek paragraf" iddiası | `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_artefaktlar.json:48-49` | 13 geçişe karşı `motorin_atif_yaptigi_paragraf_sayisi: 1` |
| Ölçüm defterindeki yeni künyeler | `docs/patent/pct_v2/OLCUM_KAYITLARI.md:28` · `:116` · `:117` | `4096 girdi` · hafta etiketleri |
| Şekil künyesindeki yazı adedi (SVG ile birebir) | `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:8` | `"kunyi_metin": 142` |
| Çizim terim denetiminden geçen kelime | `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:16` | `"kontrol_edilen_kelime": 339` |
| Ham JSON'da doğrulanan ölçü etiketi / kaynak dosya | `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:25` · `:26` | `"olcu_deger": 15` · `"kaynak_dosya": 3` |
| Şekil PDF'i sayfa sayısı / metin katmanı | `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:45` · `:44` | `"sayfa": 8` · `"karakter": 0` |
| En küçük çizim yazısı (Kural 11.13 alt sınırı 0,21 cm) | `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:61` | `"min_yazi_cm": 0.2175` |
| Defter düzenlemesi sonrası taze kapı kaydı | `patent-cikti/pct/pct_v2-pct-inceleme/KAPI_KARARI_2026-11-19.md` | kapı 5 `EKSIK: 0` · kapı 6 `kaynakta yok 0` · kapı 7 `ölü adres 0` · kapı 8 `BULGU: 0` |
| Bağımsız doğrulayıcı turları | `patent-cikti/pct/pct_v2-pct-inceleme/KAPI_KARARI_2026-11-19.md` | tam tarama `26 ✓ / 5 ✗ / 3 KANITSIZ` → kapanış `7 ✓ / 0 ✗ / 0 KANITSIZ` |
| Talep formu eksik alan grepleri | `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_form_eksik_grep.txt` | applicant/ISA/priority `0 eşleşme`, mucit yalnız yokluk ilanı |
| Şekil kapısı toplam bulgu | `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:73` | `"bulgu_toplam": 0` |
| NPL ön-tarama kaydı | `docs/patent/pct_v2/ON_SANAT.md:21` · `:113` | `21 hedefli sorgu … 21 kaynak kaydı` · §1.3'te 3 satır `BULUNAMADI`; bunlardan 2'si 3. turda bağımsız doğrulandı (`:117`, `:118`) |
| Patent veritabanı taraması (3. tur) | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_db_taramasi.txt:54-61` · `docs/patent/pct_v2/ON_SANAT.md:185` | `toplam_sorgu 42` · `hit_kayit_toplam 161` · `benzersiz_hit 160` · `bos_donen_sorgu 6` · 1.–2. turda dört anahtarlı araç `ready:false` / `available:false` (`patent-cikti/pct/pct_v2-pct-inceleme/oncesanat_ham_tur2_2026-11-19.json`) |

## 8. Karar

Metin kapıları bu hâliyle temiz, her ölçüm kendi künyesiyle:

- PCT biçim denetimi EN'de bulgu yok: `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:86-87`.
- PCT biçim denetimi TR'de bulgu yok: `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:86-87`.
- İstem–tarifname desteğinde karşılanmayan istem listesi boş (`"unsupported_claims": []`),
  kapsam tam: `patent-cikti/pct/pct_v2-pct-inceleme/F_epo_spec_support_en.json:631-635`.
- Paket bütünlüğü tam not: `patent-cikti/pct/pct_v2-pct-inceleme/H_pkg_full.json:10` ve `patent-cikti/pct/pct_v2-pct-inceleme/I_pkg_filing.json:10`.
- US istem denetiminde bulgu yok, not tam: `patent-cikti/pct/pct_v2-pct-inceleme/J_us_claims_en.json:4` ve
  `patent-cikti/pct/pct_v2-pct-inceleme/J_us_claims_en.json:10`.

Kalan D/E/F/G bulguları ile yerel `antecedent_basis` hatalarının tamamı §4 ve §6'da
kanıtlanmış araç sınırı ya da tasarım kararıdır.

Üçüncü turdan sonra metin-dışı katmanın durumu: **sekiz şekil üretildi, A4 PDF olarak basıldı ve
üreticiden bağımsız kapıyla denetlendi** (`patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:73` alanı `bulgu_toplam: 0`); **NPL ön-taraması
yapıldı** (kaynak künyeleri `ON_SANAT.md:21`); **patent+NPL veritabanı taraması üçüncü turda koşuldu**
(§5 künyesi `docs/patent/pct_v2/ON_SANAT.md:185`). Denenmemiş/eksik olanlar artık şunlardır ve
hiçbiri "temiz" sayılmamıştır: (i) **resmî CPC / Espacenet / PATENTSCOPE taraması** — dört anahtarlı
araç hâlâ yapılandırılmamış (`ON_SANAT.md:43`); (ii) PCT talep formu, ücret ve vekâletname
(§5 idari satırları);
(iii) tarifname/istem metinlerinin fiziksel dizgisi (§5, "yalnız metin sayfaları" satırı).
**Başvuru kararı için vekil onayı gerekli.**
