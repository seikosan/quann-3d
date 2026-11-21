# KURULUM PLANI — pct_v3 (PCT uluslararası başvuru dosyası, yeni paket)

## 1. Amaç

`docs/patent/pct_v2/` gönderim nüshalarını **temel alan**, bu turun otomatik denetim bulgularının
mekanik olanlarını gidermiş **yeni bir PCT başvuru dosyası**: istemler + tarifname + özet + şekiller §,
İngilizce asıl + Türkçe eş kopya, üretime hazır derleme betikleri ve kapılarıyla.

## 2. Kullanıcı kararları (2026-11-20, bu oturum)

| # | Karar | Sonuç |
|---|---|---|
| 1 | Hedef: **yalnız PCT uluslararası başvuru** | Başlıklar PCT Rule 5.2 sözleşmesinde kalır; USPTO `SUMMARY`/`DETAILED DESCRIPTION` başlık bulgusu **bilinçli açık** bırakılır ve §4 satırında kayıt altına alınır |
| 2 | **Mekanik bulgular işlenir, hukuki olanlar alternatif olarak durur** | Özet işaretleri + destek paragrafları + öncül-dayanak düzeltmeleri metne girer; istem kümesinin küçültülmesi (Rule 43(2)) **alternatif dosyada**, ana küme 15 istem |
| 3 | Şekiller **gerçek ölçüm verisinden** yeniden üretilir + tek PDF | Sayılar yalnız `docs/patent/pct_v2/OLCUM_KAYITLARI.md` defterindeki ham kaynaklardan; uydurma sayı yok |

## 3. Girdiler (salt-okunur)

- Metin tabanı: `docs/patent/pct_v2/filing/` (altı dosya §: istem, tarifname ve özet EN+TR filing copy'leri —
  adları `patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:29-36` sha1 künyesinde sayılı) — bu pakette **yalnız okundu, kopyalandır**.
- **Güncelleme § (2026-11-20 akşamı, kullanıcı kararı (a)):** v2 tabanı başka bir oturumda bilerek değişti ve
  kabul edildi — master artık 84 paragraflık aday (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama_v2kabul_2026-11-20.txt:25`),
  filing DESCRIPTION aynı adaydan üretildi (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama_v2kabul_2026-11-20.txt:18`). Sabahki
  sha1 künyesi tarihsel kayıttır; yeni DESCRIPTION sha'sı `patent-cikti/pct/pct_v3-pct-inceleme/dogrulama_v2kabul_2026-11-20.txt:9`.
  pct_v3 metinleri sabahki tabandan türedi ve değişmedi (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama_v2kabul_2026-11-20.txt:31`).
- Ölçüm defteri: `docs/patent/pct_v2/OLCUM_KAYITLARI.md` (her sayı ham JSON satırına bağlı).
- Şekil tabanı: `docs/patent/pct_v2/fig/` (sekiz SVG § + manifest),
  üreticiler `docs/patent/pct_v2/generate_figures_pct.py`, `build_drawings.py`.
- Denetim kayıtları (giderilecek bulguların kaynağı):
  `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/` — `A`,`B`,`C`,`D`,`E`,`F`,`H`,`K`,`L`,
  `ETR`,`FTR` + `INCELEME_RAPORU_2026-11-20.md`.
- Ortak künye: `docs/patent/pct_v3/REFERANS_NUMARALARI.md` (üç dosya aynı tabloyu kullanır).

## 4. Bulgu → giderim eşlemesi

| Bulgu (kayıt) | Giderim | Dosya |
|---|---|---|
| `H` Rule 47(4) EPC — özetde parantez içi şekil işaretleri yok | Özet, anıştığı unsurlar için parantezli numara ile yeniden yazılır (≤150 kelime, PCT Rule 8.1) | `filing/ABSTRACT_FILING-COPY_EN.txt`, `filing/OZET_FILING-COPY_TR.txt` |
| `C` + `F` — 47 "öğe tek paragraftan türüyor" izi | Tarifnameye **sona eklenen** "additional enabling detail" bölümü: her farklı unsur için ayrı paragraf; mevcut `[0001]–[0070]` numaraları **korunur**, yeni paragraflar `[0071]`-den devam eder (denetim izi bozulmasın diye) | `filing/DESCRIPTION_FILING-COPY_EN.txt`, `filing/TARIFNAME_FILING-COPY_TR.txt` |
| `B` — 43 LOW öncül-dayanak izi | İstemlerde `the X` yalnız aynı istemde/dizin önce anılmışsa kalır; şüpheli her düzeltme ayrı satırda gerekçeli, vekilin geri alabileceği biçimde | `filing/CLAIMS_FILING-COPY_EN.txt`, `filing/ISTEMLER_FILING-COPY_TR.txt` + `DEGISIKLIK_KAYDI_ISTEM.md` |
| `E` Rule 43(2) EPC — 1/13/15 aynı kategori | **Hukuki — işlenmez.** Alternatif küme ayrı dosyada (kategori başına bir bağımsız), ana 15 istem kümesi durur | `ALTERNATIF_ISTEM_KUMESI.md` |
| `D` 2 CRITICAL — US başlıkları | **Karar gereği işlenmez** (yalnız PCT hedefi). EPO bölüm denetimi ve PCT bölüm denetimi pct_v2'de geçmişti | `DEGISIKLIK_KAYDI_TARIFNAME.md` "bilinçli açık" satırı |
| `K` "No claims detected" + `H` `claims_fees` atlaması | Araç girdi sınırı (araca istem metni verilemiyor) → metin değişikliği **yok**, not düşülür | `DEGISIKLIK_KAYDI_TARIFNAME.md` |
| `A` `antecedent_basis` kapalı | Opt-in kontrolü açılmış hâli (`B`) esas alınır; kapalıyken "0 bulgu" okuması raporda tekrarlanmaz | — |

## 5. Dosya planı

```
docs/patent/pct_v3/
  KURULUM_PLAN.md                     (bu dosya)
  REFERANS_NUMARALARI.md              (ortak künye — önce yazılır)
  README.md                           (künye: ne/var nasıl derlenir/kapılar)
  filing/CLAIMS_FILING-COPY_EN.txt    filing/ISTEMLER_FILING-COPY_TR.txt
  filing/DESCRIPTION_FILING-COPY_EN.txt  filing/TARIFNAME_FILING-COPY_TR.txt
  filing/ABSTRACT_FILING-COPY_EN.txt  filing/OZET_FILING-COPY_TR.txt
  filing/SEKILLER_FILING-COPY.pdf
  fig/fig1..fig8.svg + fig_manifest.json
  generate_figures_pct3.py  build_drawings_pct3.py  dogrulama.sh
  DEGISIKLIK_KAYDI_ISTEM.md  DEGISIKLIK_KAYDI_TARIFNAME.md  DEGISIKLIK_KAYDI_SEKIL.md
  ALTERNATIF_ISTEM_KUMESI.md          (vekil kararı bekliyor)
```

## 6. Kapılar (kabul ölçütü)

1. § **Yapısal özdenetim** `docs/patent/pct_v3/dogrulama.sh` — ham çıktı `patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt` §:
   istem sayaçları 15 ve 3 (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:2` ve `patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:3`), paragraf numaraları kesintisiz
   artan; her şekil numarası tarifnamede en az bir kez geçiyor; PDF sayfa sayısı 8 (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:20`);
   SVG metin yüksekliği PCT Kural 11.13 eşik sınırının altında değil.
2. § **Araç denetimi:** pct_v3 nüshaları aynı motor koşumundan geçer (kayıtlar
   `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/`), pct_v2 sayılarıyla yan yana `MOTOR_KARSI_KARSTIRMA.md`'de yazılır.
3. § **Sayı atıf kapısı:** `python3 ~/.config/opencode/denetim/rapor_denetle.py` → eşik: `kaynakta yok 0 · ölü adres 0`.
4. § **Bağımsız doğrulayıcı tur** (`reviewer`): her iddia ham kayda karşı ✓/✗/KANITSIZ.
5. § `bash scripts/check_docs_lines.sh` yeşil (`docs/patent/**` bütçe istisnasında).

## 7. Yasaklar

- `docs/patent/pct_v2/` içinde hiçbir dosya değişmez/silinmez (yalnız okunur).
- Ölçüm defterinde olmayan sayı metne/şekle giremez; birim çevirimi ham+türetilmiş yan yana yazılır.
- Hukuki hüküm cümlesi yazılmaz ("yenidir", "patentlenebilir", "ihlal yok") — her metnin sonunda
  **vekil onayı gerekli** notu durur.
- `state_dict` ana adları ve `logs/` append-only kuralları (repo AGENTS.md §3) bu pakete de uygulanır.
