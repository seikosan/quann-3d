# KAPI_KARARI_2026-11-19 — defter düzenlemeleri SONRASI taze kapılar (tek kayıt)

Bu dosya elle yazılmadı; aşağıdaki komutların ham çıktıları `patent-cikti/pct/pct_v2-pct-inceleme/`
altındaki dosyalara yönlendirildi ve buraya özetlendi. Toplu hattın (9 kapı) son tam koşumu
`patent-cikti/pct/pct_v2-pct-inceleme/kapilar_taze.txt` dosyasındadır (rapor kapısı o koşuda
`doğru 6 · kaynaksız 40`). Bu kayıt SONRADIR: `docs/patent/pct_v2/` defterleri doğrulayıcı turundan
sonra düzenlendi ve markdown okuyan dört kapı yeniden koşuldu.

| # | Kapı | Karar | Ham çıktı |
|---|---|---|---|
| 0 | şekil hattı (üretim + A4 derleme + harici kapı) | `0/0/0` | `KANIT_kirmizi_kapali_donunum.txt:16` `KABUL: 0 bulgu` · `build_drawings_konsol.txt` |
| 5 | sayı-atıf haritası | `EKSIK: 0` | `atif_haritasi_konsol.txt:28` |
| 6 | rapor kapısı (4 defter) | `doğru 8 · kaynakta yok 0 · ölü adres 0 · kaynaksız 43` → çıkış 2 | `rapor_kapisi_yeni_defterler.txt:20` |
| 7 | tablo içi künye + motor değer | açılan adres `167` · ölü adres `0` · uyuşmazlık `0` | `dogrul_tablolar_konsol.txt:1` |
| 8 | ölçüm defteri (OLCUM_KAYITLARI) | `UYARI 0` · `BULGU 0` | `dogrul_olcum_konsol.txt:6` |

## Doğrulayıcı turları (bağımsız bağlam, salt-okunur)

| Tur | Kapsam | Hüküm |
|---|---|---|
| 1 | BASVURU_PAKETI · UYGUNLUK_NOTU · ON_SANAT, tüm iddialar | `26 doğru / 5 yanlış / 3 kanıtsız` |
| 2 | 1. turun 5 yanlış + 3 kanıtsız maddesinin kapanışı | `7 doğru / 0 yanlış / 0 kanıtsız` |

Beş yanlış madde ve kapanışı: üç öz-atıflı grep (`--exclude BASVURU_PAKETI.md` ile yeniden koşuldu →
`KANIT_form_eksik_grep.txt`), çizimle birebir olmayan `slid…` alıntısı ve `centroid` çelişkisi
(`UYGUNLUK_NOTU.md` §6 satırı ham künye/tarifname satırlarına bağlandı). Üç kanıtsız: bayt-bayt
geri yükleme kapalı döngü SHA-256 ile kanıtlandı (`KANIT_kirmizi_kapali_donunum.txt`), `dogrul_figurler.py`
`DOGRUL_figurler.json:289` satırında çıktı dosyasını yazıyor (metne işlendi), "ilk koşumdaki bulgu
adedi" diskte kayıt olmadığı için KANITSIZ olarak bırakıldı.

## Bu kaydın kapsamadığı (kapalı değildir)

- Patent+NPL veritabanı taraması 3. turda YAPILDI — künye `docs/patent/pct_v2/ON_SANAT.md:185`,
  ham arşiv `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/`. Yapılmayanlar: resmî CPC /
  Espacenet / PATENTSCOPE taraması, atıf ağları, X/Y/P sınıflaması (`ON_SANAT.md` §5.5).
- PCT/RO/101 talep formu, ücret, vekâletname, gönderim: YOK — `docs/patent/pct_v2/BASVURU_PAKETI.md` §3.
- Bağımsız doğrulayıcı turu metin/şekil katmanını denetler; **vekil onayının yerine geçmez**.

## 3. tur: ön-sanat defteri + doğrulayıcı kapanışı (2026-11-19 23:48)

Tam hat yeniden koştu (defterler §5 ile büyüdükten sonra): `kapilar_taze.txt:48` `6 rapor kapisi : 2`
· `:51` `SONUC: yeşil` · `:52` `CIKIS=0`. Rapor kapısının ham özeti `rapor_kapisi.txt:26`
`doğru 15 · kaynakta yok 0 · ölü adres 0 · kaynaksız 47`; dosya payları `4 + 0 + 15 + 28 = 47`
(`rapor_kapisi.txt:5,11,17,23`) — önceki denetmen turundaki "28 ile 47 uyuşmuyor" notu kısmi okumaydı,
toplam dosya paylarının toplamıdır.

Bağımsız doğrulayıcı turu 3 (salt-okunur, `ON_SANAT.md` §5–§6, 42 sorgunun ham kayıtlarına karşı):
**`27 doğru / 3 yanlış / 1 kanıtsız`**. Üç yanlışın kapanışı (ölçüm değiştirilmedi, cümle/adres düzeltildi):

| Yanlış madde | Uygulanan düzeltme | Doğrulanan kaynak |
|---|---|---|
| §6'daki derin kayıt sayısı `9` için gösterilen aralıkta (`KANIT_db_taramasi.txt:54-61`) 9 geçmiyordu | Sayı üretime bağlandı: `derin_sayi.py` → `db_taramasi/KANIT_derin_kayit_sayisi.txt:4-5` (`ham_sayfa_dosyasi 9` · `derin_kayit_basligi 8`); §6 ve `BASVURU_PAKETI.md` §6/1 bu adresi taşıyor | `db_taramasi/KANIT_derin_kayit_sayisi.txt:4-5` |
| "istem numaraları `CLAIMS_EN.md:46-50`" — 9. ve 14. istemler aralık dışındaydı | Atıf `docs/patent/pct_v2/CLAIMS_EN.md:44-64` yapıldı | `docs/patent/pct_v2/CLAIMS_EN.md:44,46,50,64` |
| Boş iddia dilimleri `DERIN_kayit.md:98` ve `:180` olarak adreslenmişti; o satırlar kayıt başlığı | Adresler boş satırlara taşındı: `:102` (WO2026046480A1) ve `:183-184` (WO2026170161A1) | `db_taramasi/DERIN_kayit.md:102,183,184` |

Turun 1 kanıtsız maddesi (doğrulayıcı `kapilar.sh` çıkış kodunu salt-okunur kuralı nedeniyle
koşmamıştı) ana oturumun taze koşumuyla kapandı: `kapilar_taze.txt:52` `CIKIS=0`. Aynı turda
`ON_SANAT.md` §5.1'deki `503` satırı genişletildi (yorum kaydı iki betikte: `db_taramasi/gp_tara.py:5-6`
ve `db_taramasi/npl_tara.py:6`) ve §5.4 x02 atfı üç terimi de kapsayacak şekilde
`db_taramasi/KANIT_isim_taramasi.txt:12-16` yapıldı.

Kapanış turu (aynı bağımsız bağlam, 9 kontrol): **`9 doğru / 0 yanlış / 0 kanıtsız`** — yedi
düzeltmenin her biri ham kaynakta doğrulandı (`ON_SANAT.md:194,234,235,264,290,301` ·
`db_taramasi/KANIT_derin_kayit_sayisi.txt:4-5` · `db_taramasi/DERIN_kayit.md:102,183-184` ·
`db_taramasi/KANIT_isim_taramasi.txt:12-16`), H kontrolünde paylar `6+0+1+8=15` ve
`4+0+15+28=47` (`rapor_kapisi.txt:3-26`), I kontrolünde `kapilar_taze.txt:48,51,52` taze.

**Kapanış hükmü:** metin + şekil + ön-sanat katmanlarında kapı bulgusu yok; kalanlar KANITSIZ
etiketli künyeler ve vekil işleridir. **Başvuru kararı için vekil onayı gerekli.**
