# KAPAK — pct_v3 vekil paketi (2026-11-20)

Bu paket, WIPO/PCT uluslararası başvuru hedefiyle hazırlanmış **taslak** başvuru dosyası ve
vekil kararını bekleyen maddelerden oluşur. Hukuki hüküm içermez; başvuru kararı, istem
kapsamı ve resmî ücretler vekile aittir.

## Paket içeriği

| Klasör | İçerik |
|---|---|
| `01_Gonderim_Nuslari/` | Gönderim nüshaları (7 dosya): EN+TR istem (15 istem / 3 bağımsız), EN+TR tarifname (98/98 paragraf, kesintisiz [00001]–[00098]), EN+TR özet (148/132 kelime), şekiller PDF (8 sayfa A4, yalnız çizim — metin katmanı yok) |
| `02_Vekil_Karari/` | `ALTERNATIF_ISTEM_KUMESI.md` — EPC Rule 43(2) bulgusuna iki yol: (a) istem 13+15 bağımlıya (taslak metinler hazır), (b) metin değiştirmeden gerekçelendirme |
| `03_Degisiklik_Defterleri/` | v2→v3 her değişikliğin gerekçeli kaydı (istem 43 iz, tarifname, şekil), 56 numaralık referans tablosu, kurulum planı |
| `04_Kayitlar_Kanitlar/` | Bağımsız denetim kayıtları: yapısal özdenetim (`dogrulama.txt`), güncel v2 künyesi (`dogrulama_v2kabul_...txt`), linter raporu (`lint_v3.txt`), otomatik denetim motoru karşılaştırması (`MOTOR_KARSI_KARSTIRMA.md`) |
| `05_Sekiller_Kaynak_SVG/` | 8 şeklin kaynak çizimi (fig1–fig8 SVG) — referans numaraları yeniden numaralandırılırsa şekillerin yeniden türetilmesi için |
| `MANIFEST_sha1.txt` | Paketteki her dosyanın sha1 bütünlük değeri |

## Vekilden beklenen kararlar (özet)

1. **Alternatif istem kümesi** — bağımsız 1/13/15 için iki yoldan biri seçilecek (`02_Vekil_Karari/`).
2. **Geri alınabilir düzeltmeler** — 43 öncül-dayanak izinin her düzeltimi tek satırda geri alınabilir
   (`DEGISIKLIK_KAYDI_ISTEM.md` §"GERİ ALINDI" mekanizması).
3. **Mutlak ifadeler** — linter 11 uyarı: "only / never / best / critical" (`lint_v3.txt:9-19`).
4. **Bilinçli açıklar** — ABD başlık bulguları (2 CRITICAL) yalnız-PCT kararıyla açık; ABD/Faz-2
   düşünülüyorsa yeniden açılır (`KURULUM_PLAN.md` §4).
5. **Referans numaraları** — taslak telif kararı; vekil yeniden numaralandırabilir
   (`REFERANS_NUMARALARI.md`).
6. **Başvuru verileri** — buluşçu/başvuru sahibi, rüçhan, ISA seçimi, vekaletname, resmî ücretler;
   30 sayfa üzeri ek ücret tutarı henüz DOĞRULANMADI (resmî tarifeden teyit gerekiyor).

## Bütünlük

- Dosyaların sha1 değerleri `MANIFEST_sha1.txt` içindedir; iletildiği hâliyle doğrulanabilir.
- Gönderim nüshaları denetim kapılarından geçmiştir: yapısal özdenetim yeşil (`dogrulama.txt`),
  paket bütünlük skoru 100, bağımsız doğrulayıcı turu 7/8 madde ✓ (tek ✗ bilinen ve kabul edilmiş
  v2 taban değişikliğidir — `dogrulama_v2kabul_2026-11-20.txt`).

**vekil onayı gerekli** — bu paketin hiçbir dosyası vekil imzası olmadan başvuru nüshası sayılmaz.
