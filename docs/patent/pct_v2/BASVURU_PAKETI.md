# BAŞVURU PAKETİ — pct_v2 gönderim kontrol listesi (iç defter, başvuru metni değildir)

künye: tarih 2026-11-19 · yazan ana oturum · bu dosya **başvurunun yapıldığı anlamına gelmez**;
hiçbir resmî ofise gönderim yapılmadı, form doldurulmadı, ücret hesaplanmadı.
Bu paketin tümü **vekil onayı olmadan başvuru konusu yapılmaz** (bkz. §6).

**Kanıt biçimi kuralı:** aşağıdaki her ölçü bir dosya satırına ya da bu dosyada yazılı komuta
bağlıdır. Rakamı doğrulanamayan hiçbir alan "hazır" işaretlenmemiştir; resmî ofis verisi
(ücret tutarları, form kutu numaraları, e-PCT teknik sınırları) bu depoda **yoktur** ve
bilerek **boş bırakılmıştır** — bellekten doldurulmamıştır.

---

## §1 Gönderilecek nüsha envanteri (basılabilir kopyalar)

Üretim komutu: `python3 docs/patent/pct_v2/make_filing.py` → `docs/patent/pct_v2/filing/`
(başarı: `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json` `"ok": true`, altı dosyanın
`problems` toplamı `0`). Dosya kimliği komutu: `shasum -a 256 docs/patent/pct_v2/filing/*`

| Nüsha | Rol (PCT) | Dil | Ölçü | Kanıt |
|---|---|---|---|---|
| `DESCRIPTION_FILING-COPY_EN.txt` | Tarifname (Kural 5) | EN | 5.733 kelime · 70 paragraf, aralıksız | kelime `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:9` · paragraf `:14` |
| `CLAIMS_FILING-COPY_EN.txt` | İstemler (Kural 6) | EN | 1.722 kelime · 15 istem | kelime `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:22` · istem `:40` |
| `ABSTRACT_FILING-COPY_EN.txt` | Özet (Kural 8) | EN | 137 kelime (50–150 aralığında) | kelime `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:93` · motor `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:6` |
| `TARIFNAME_FILING-COPY_TR.txt` | Tarifname eş nüsha | TR | 4.316 kelime · 70 paragraf | kelime `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:51` · paragraf `:56` |
| `ISTEMLER_FILING-COPY_TR.txt` | İstemler eş nüsha | TR | 1.358 kelime · 15 istem | kelime `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:64` · istem `:82` |
| `OZET_FILING-COPY_TR.txt` | Özet eş nüsha | TR | 106 kelime | kelime `patent-cikti/pct/pct_v2-pct-inceleme/uretim.json:106` · motor `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:6` |
| `SEKILLER_FILING-COPY.pdf` | Şekiller (Kural 7) | EN `FIG. n` | 8 sayfa A4 · metin katmanı 0 karakter · en küçük yazı 0,2175 cm | `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:45` · `:44` · `:61` |

Boyutlar: şekil PDF'i `930301 B`; altı metin nüshası `828` – `37279` B. Ham çıktı diske
yazıldı: `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_dosya_boyutlari.txt` (`ls -l` +
`os.path.getsize` döngüsü).

**Başlık (Kural 4.3) — iki dilde, motorca uyumlu bulunmuş:**

| Dil | Başlık | Kelime | Kanıt |
|---|---|---|---|
| EN | Dual-use point-cloud representation with embedded explainability | 6 | `docs/patent/pct_v2/DESCRIPTION_EN.md:3` · `patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:12` |
| TR | Çift kullanımlı nokta bulutu temsilinde gömülü açıklanabilirlik | 7 | `docs/patent/pct_v2/DESCRIPTION_TR.md:3` · `patent-cikti/pct/pct_v2-pct-inceleme/B_pct_tr.json:12` |

---

## §2 Şekil eşlemesi (talep formunda ve incelemede kullanılacak köprü)

Kaynak: `docs/patent/pct_v2/fig/fig_manifest.json` + tarifname paragraf taraması.
Şekillerde modül referans numarası **bilinçli olarak yok** (Kural 6.2(b) — şema uydurulmadı;
bkz. `docs/patent/pct_v2/UYGUNLUK_NOTU.md:26`).

| Şekil | SVG dosyası (`docs/patent/pct_v2/fig/`) | Nitelik | Tarifname paragrafları | Ham ölçüm kaynağı |
|---|---|---|---|---|
| FIG. 1 | `fig1_system_architecture.svg` | şematik | [0020], [0029] | — |
| FIG. 2 | `fig2_pair_geometry.svg` | şematik | [0021], [0033] | — |
| FIG. 3 | `fig3_symmetric_aggregation.svg` | şematik | [0022], [0034], [0038], [0065] | — |
| FIG. 4 | `fig4_point_removal.svg` | şematik | [0023], [0040] | — |
| FIG. 5 | `fig5_region_naming.svg` | şematik | [0024], [0043] | — |
| FIG. 6 | `fig6_region_shift.svg` | **ölçümlü** | [0025], [0028], [0046], [0048], [0049] | `complaints/results/complaints_scale_stability.json` · `maintenance/results/e1_cmapss.json` |
| FIG. 7 | `fig7_int8_package.svg` | **ölçümlü** | [0026], [0028], [0050], [0056] | `fraud/results/edge_eval_full_scale.json` |
| FIG. 8 | `fig8_window_cloud.svg` | şematik | [0027], [0028], [0037], [0060], [0065] | — |

Şekil kapısının beş denetimi — A künye↔SVG eşitliği, B tarifname terimi, C ölçüm izlenebilirliği,
D şema/ölçüm ayrımı, E fiziksel ölçü + Kural 11.13 + FIG eşlemesi — `0 bulgu`:
`patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:73` (`bulgu_toplam: 0`).

**Öneri (karar vekile ait):** özeti birlikte yayımlanacak şekil olarak **FIG. 1** işaretlenmesi.
Gerekçe: sistem mimarisini tek başına gösteren tek şema; ölçüm taşımıyor, bu yüzden yayımlandığında
yanlış okunma riski yok. Motor bu seçimi metinden denetleyemiyor:
`patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:9` = `not checked: verify the request form names a figure`.

---

## §3 PCT/RO/101 talep formu — alan kontrol listesi

Durum kodları: **HAZIR** (metin bu depoda mevcut ve kapıdan geçti) · **EKSİK·KULLANICI**
(bilgi yalnız başvuru sahibinden gelir) · **EKSİK·VEKİL** (hukuki/idari karar veya resmî veri) ·
**UYGULANABİLİR?** (vekil karar verir, bu paket iddia kurmaz).

| # | Form alanı | Durum | Dayanak / eksikliğin kanıtı |
|---|---|---|---|
| 1 | Buluşun başlığı (EN ve TR) | HAZIR | §1 başlık tablosu |
| 2 | Başvuru sahibi(lar) — ad/unvan, adres, uyruk, ikametgâh | EKSİK·KULLANICI | Kanıt komutu (bu defterin kendi satırları sayılmasın diye `--exclude BASVURU_PAKETI.md` ile): `grep -rniE "applicant|başvuru sahibi" docs/patent/pct_v2/*.md --exclude BASVURU_PAKETI.md` → **0 eşleşme**; basılabilir nüshalarda da 0 eşleşme. Ham çıktı: `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_form_eksik_grep.txt` §1 ve §5 (betiği: `patent-cikti/pct/pct_v2-pct-inceleme/form_eksik_grep.sh`) |
| 3 | Mucit(lar) — ad, adres, vatandaşlık | EKSİK·KULLANICI | `grep -rniE "mucit|inventor" docs/patent/pct_v2/*.md --exclude BASVURU_PAKETI.md` yalnız `docs/patent/pct_v2/UYGUNLUK_NOTU.md:37` satırındaki *eksik* notunu döndürüyor — kimlik alanı değil, yokluğun ilanı. Kimlik verisi: 0 kayıt (`KANIT_form_eksik_grep.txt` §2) |
| 4 | Mucitluk beyanı / atıf (Rule 4.4(b) beyanı) | EKSİK·VEKİL | Metinde beyan yok; hangi ülkede isteneceği vekilin kararı |
| 5 | Vekil / temsilci kimliği ve vekâletname (Rule 9.3) | EKSİK·VEKİL | `docs/patent/pct_v2/UYGUNLUK_NOTU.md:38` — ücret ve vekâletname satırı |
| 6 | Tebligat adresi | EKSİK·KULLANICI | Depoda adres verisi yok |
| 7 | Ekli belgelerin listesi (tarifname · istem · özet · şekiller) | HAZIR | §1 envanteri; dört belge türü de basılabilir kopya olarak mevcut |
| 8 | Şekillerle birlikte yayımlanacak şeklin işaretlenmesi (Kural 8) | EKSİK·VEKİL | §2 önerisi FIG. 1; motor metinden doğrulayamıyor (`A_pct_en.json:9`) |
| 9 | Başvuru dili ve çeviri rejimi | EKSİK·VEKİL | EN ve TR nüshalar birlikte hazır; RO TÜRKPATENT pratiğinde hangisinin "başvuru dili" olacağı ofis uygulaması |
| 10 | Arama otoritesi seçimi (ISA) | EKSİK·VEKİL | `grep -rniE "arama otoritesi|ISA seçildi" docs/patent/pct_v2/*.md --exclude BASVURU_PAKETI.md` → **0 eşleşme** (`KANIT_form_eksik_grep.txt` §3) |
| 11 | Ön inceleme talebi (İkinci Bölüm, Kural 69/II) | EKSİK·VEKİL | Talep yok; strateji kararı |
| 12 | Öncül (rüçhan) talebi | EKSİK·KULLANICI | `grep -rniE "priority|rüçhan" docs/patent/pct_v2/*.md --exclude BASVURU_PAKETI.md` → **0 eşleşme** (`KANIT_form_eksik_grep.txt` §4). Bu, *öncül yoktur* iddiası **değildir**: metinde kayıt olmaması ile başvuru sahibinin beyanı aynı şey değil |
| 13 | Dizi listesi (ST.26) | UYGULANABİLİR? | Tarifenmede nükleik/amino asit dizisi yok; "uygulanmaz" kararını vekil onaylar |
| 14 | Biyolojik materyal tevdiatı | UYGULANABİLİR? | Konu alanı dışı görünüyor; kararı vekil verir |
| 15 | Gönderim (transmittal) ücreti · uluslararası başvuru ücreti · arama ücreti · sayfa ücreti | EKSİK·VEKİL | **Bu depoda ücret verisi yoktur ve hesaplanmadı.** Resmî tutarlar WIPO ücret cetvelinden alınmalı; bu oturumda cetvel okunmadı → tutarlar **DOĞRULANMADI** |
| 16 | e-PCT indirimleri ve resmî indirim uygunluğu (düşük gelirli ülke vb.) | EKSİK·VEKİL | Uygunluk ülke/ödemeye bağlı; veri yok |
| 17 | İmza / e-PCT sertifika ve kimlik doğrulama | EKSİK·VEKİL | Bu oturumda hiçbir resmî sisteme giriş yapılmadı |
| 18 | Yayın için isteğe bağlı beyanlar (gecikme talebi, sekizinci yıl vb.) | EKSİK·VEKİL | Tercih kararı |

**Form kutu numaraları bilinçli olarak yazılmadı:** bu depoda PCT/RO/101 şablonu yok; kutu
numarası bellekten yazılsaydı doğrulanamaz sayı olurdu. Vekil güncel form sürümünü esas alır.

---

## §4 Teknik teslim (e-PCT / ofis yüklemesi) kontrol listesi

| Madde | Durum | Kanıt / eksik |
|---|---|---|
| Şekil PDF'i A4, 8 sayfa, dikey, kenar boşlukları 2,5 cm | HAZIR | `docs/patent/pct_v2/build_drawings.py:42` (`MARGIN_CM = 2.5`) · `:43` (`SAFE_W_CM` 16,0 cm) · `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:45` (`sayfa: 8`) · `:46` (`pt: [595.304, 841.89]`) |
| Şekillerde metin katmanı yok (tarama/görüntü olarak basıldı) | HAZIR | `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:44` (`karakter: 0`); aynı kapı her sayfada mürekkep ölçüyor (`:47` `murekkep_yuzde`, 8 değer) |
| Yazı yüksekliği Kural 11.13 alt sınırı ≥ 0,21 cm | HAZIR | `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:61` (`min_yazi_cm: 0.2175`) |
| Siyah-beyaz / gri tonlama, renkli öğe yok | HAZIR | Üretim hattı her sayfayı gri ölçeğe çevirir: `docs/patent/pct_v2/build_drawings.py:55` (`convert("L")`) |
| Şekil başına tek sayfa, `FIG. n` etiketi sayfada bir kez | HAZIR | E kapısı SVG yazılarını sayfa sayısıyla birlikte denetliyor: `patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:45` (`sayfa: 8`) ve `:38` (E kapısı `durum: TEMIZ`) |
| PDF/A-1 uyumu | **DOĞRULANMADI** | `pdfinfo` künyesi: `Producer: LibreOffice 26.8.2.2`, `PDF version: 1.7`. PDF/A doğrulayıcısı (veraPDF/PDF*A-1 validator) bu makinede yok; ofis reddederse LibreOffice'ten PDF/A dışa aktarımı yeniden denenmeli |
| Şeffaflık / gömülü font riski | **DOĞRULANMADI** | Doğrulayıcı koşulu değil; yükleme öncesi ofisin ön-denetimine bırakıldı |
| Metin nüshalarının fiziksel dizgisi (satır numarası, sayfa üstü, punto) | EKSİK·VEKİL | `.txt` nüshalar ham metin; Kural 11 dizgisi vekilin Word/LaTeX nüshasında üretilir (`docs/patent/pct_v2/UYGUNLUK_NOTU.md` §5 "yalnız metin sayfaları" satırı) |
| Dosya adı/boyut sınırları | **DOĞRULANMADI** | Resmî sınırlar bu depoda yok; okunmadı |

---

## §5 Gönderim öncesi sıralı kapılar (bu depoda koşulabilir)

Sıra önemlidir: bir önceki kırmızıysa sonraki anlamsızdır.

| Sıra | Koşulacak kapı | Kabul ölçütü | Son durum |
|---|---|---|---|
| 1 | `python3 docs/patent/pct_v2/generate_figures_pct.py` | çıkış `0`, son satır `KUTU-SIGMA: 0 taşma` | geçtik (üretim kaydı `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_kutu_sigma_yesil.txt`) |
| 2 | `python3 docs/patent/pct_v2/build_drawings.py` | `KABUL: 8 sayfa A4, her sayfada mürekkep, metin katmanı yok` | geçtik (`patent-cikti/pct/pct_v2-pct-inceleme/build_drawings_konsol.txt`) |
| 3 | `python3 patent-cikti/pct/pct_v2-pct-inceleme/dogrul_figurler.py` | çıkış `0`, `bulgu_toplam: 0` | geçtik (`patent-cikti/pct/pct_v2-pct-inceleme/DOGRUL_figurler.json:73`) |
| 4 | `python3 docs/patent/pct_v2/make_filing.py` | `"ok": true`, tüm `problems` boş | geçtik (`patent-cikti/pct/pct_v2-pct-inceleme/uretim.json`) |
| 5 | `bash patent-cikti/pct/pct_v2-pct-inceleme/kapilar.sh` | dokuz kapı (0–8), altısı KARAR kapısı; karar satırı `SONUC: yeşil` | geçtik — tam hattın ham çıktısı `patent-cikti/pct/pct_v2-pct-inceleme/kapilar_taze.txt:51`. Bu satırın 1–4. sıraları da aynı koşuda üretildi. **Defterler doğrulayıcı turundan sonra düzenlendiği için markdown okuyan dört kapı yeniden koşuldu:** tek kayıt `patent-cikti/pct/pct_v2-pct-inceleme/KAPI_KARARI_2026-11-19.md` |
| 6 | `python3 patent-cikti/pct/pct_v2-pct-inceleme/dogrul_olcum.py` ve `dogrul_tablolar.py` | çıkış `0` | geçtik — taze ham çıktı `patent-cikti/pct/pct_v2-pct-inceleme/dogrul_olcum_konsol.txt:6` (`BULGU: 0`) ve `dogrul_tablolar_konsol.txt:1` (`ölü adres 0`) |
| 7 | Bağımsız doğrulayıcı turu (`reviewer`/`tester`) | her iddia ham kaynağa karşı ✓ ya da "kanıtsız" | **geçtik** — iki tur: tam tarama `26 ✓ / 5 ✗ / 3 KANITSIZ`, kapanış turu `7 ✓ / 0 ✗ / 0 KANITSIZ` (kayıt `patent-cikti/pct/pct_v2-pct-inceleme/KAPI_KARARI_2026-11-19.md`). Kapanış turu **vekil onayının yerine geçmez**; §6'daki KANITSIZ satırlar açık duruyor |

## §6 Bu paketin İÇİNDE OLMAYANLAR (kapalı diye işaretlenmez)

1. **Patent veritabanı yenilik taraması 3. turda YAPILDI; resmî CPC taraması hâlâ eksik.**
   Google Patents tam-metin kanalı vekil proxy üzerinden 42 sorguyla koşuldu: `161` hit kaydı,
   `160` benzersiz hit, `9` kayıt 1. istem düzeyinde derinleşti (kayıt sayısı
   `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_kayit_sayisi.txt:4`); ham
   yanıtlar diskte (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/`, sayım
   `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_db_taramasi.txt:54-56`, künye
   `docs/patent/pct_v2/ON_SANAT.md:185`). En yakın iki öncül `docs/patent/pct_v2/ON_SANAT.md:238`
   satırında. Kalan: Espacenet/PATENTSCOPE + CPC sınıf taraması (dört resmî araç hâlâ
   yapılandırılmamış — araç durum tablosu `docs/patent/pct_v2/ON_SANAT.md:43`), atıf ağları ve
   X/Y/P sınıflaması. Birinci ve ikinci turun erişim engeli kaydı `docs/patent/pct_v2/ON_SANAT.md:24`,
   anahtarlı araçların ham durum kaydı
   `patent-cikti/pct/pct_v2-pct-inceleme/oncesanat_ham_tur2_2026-11-19.json` (`durum_cagrilari`;
   `jq '.durum_cagrilari | keys | length' <kayıt>`).
2. **NPL ön-taraması KANITSIZ KISIMLAR TAŞIR** — 1. turun 21 araç yanıtı diskte saklanmadı; yalnız
   2. turun 8 sorgusu ham kayıt: `docs/patent/pct_v2/ON_SANAT.md` §0 ham-kayıt notu ve §4 V6 kartı.
3. **Resmî ücret hesabı ve indirim uygunluğu yok** — §3 satır 15–16.
4. **PCT/RO/101 doldurulmadı, vekâletname yok** — §3 satır 2–6, 17.
5. **Uluslararası arama raporu / yazılı görüş yok** — ISA araması henüz yapılmadı (kendi
   veritabanı ön-taramamız bundan ayrıdır, bkz. madde 1); Kural 19.2 değişiklik
   sayfası bu aşamada konu dışı (`docs/patent/pct_v2/UYGUNLUK_NOTU.md:36`).
6. **Metin sayfalarının fiziksel dizgisi yok** — §4 son satırı.
7. **Başvuru tarihi/numarası yok** — hiçbir ofise gönderim yapılmadı.

**Kapanış:** metin ve şekil katmanı üretilmiş ve bağımsız kapılardan geçmiştir; idari ve resmî-veri
katmanı üretilmemiştir. **Başvuru kararı ve gönderim için vekil onayı gereklidir.**
