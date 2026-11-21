# DEĞİŞİKLİK KAYDI — İSTEMLER (pct_v3)

- Kaynak izler: `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/B_us_claims_en_antecedentON.json`
  — `total_issues: 43` (`patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/B_us_claims_en_antecedentON.json:9`), hepsi `severity IMPORTANT`,
  aracın kendi `fix` alanı: "MANUAL VERIFICATION REQUIRED ... Tool has high false positive rate." §
- Taban metin: `docs/patent/pct_v2/filing/CLAIMS_FILING-COPY_EN.txt` (satır no'ları bundan; TR karşılığı
  `docs/patent/pct_v2/filing/ISTEMLER_FILING-COPY_TR.txt`). Kısaltma: **B** = yukarıdaki kayıt dosyası;
  **EN:n / TR:n** = taban metin satırları §.
- Ölçülemeyen: aracın yanlış-pozitif oranı **ölçülmedi** (kayıtta yok) §; yalnız kendi LOW güven + FP notu var.
  Ayrıca tüm izlerin `location` alanı "preamble" görünüyor oysa işaretli dizeler gövdede — segment
  ayracı sınırlaması, FP değerlendirmesinde ek kanıt olarak kullanıldı.
- Kapsam kuralı: istem kapsamını daraltan hiçbir değişiklik yapılmadı; 15 istem / 3 bağımsız (1, 13, 15) §
  korundu (§Özet; ham sayaç `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/B_us_claims_en_antecedentON_v3.json:468`).
  İstemlerde referans numarası yok (PCT Rule 13.4) §.

| # | istem | eski (alıntı) | yeni (alıntı) | gerekçe | dayanak (kayıt dosyası + alan) | durum |
|---|---|---|---|---|---|---|
| 1 | 7 | "wherein the region naming is performed by running" | "wherein the region naming of claim 1 is performed by running" | İstem 1 "region naming"i (ii) madde etiketi olarak adıyla ve artikel vermeden tanıtır (EN:19); bağımlıda "the" öncülü aynı-metin artikel girişi bulamaz → açık istem atfıyla dayanak verildi; kapsam daralmaz | B:L251 `term: "the region naming"`; EN:35; EN:19 | uygulandı |
| 2 | 8 | "wherein the region shift comprises quantifying point shares" | "wherein the region shift of claim 1 comprises quantifying point shares" | Aynı durum: "region shift" EN:21'de artikel-siz madde etiketi; bağımlı "the region shift" için atıf-dayanak | B:L272 `term: "the region shift comprises quantifying point shares..."`; EN:37; EN:21 | uygulandı |
| 2T | 7 (TR) | "olup, bölge isimlendirmenin, kümeleme" | "olup, İstem 1'de tanımlı bölge isimlendirmenin, kümeleme" | EN #1'in birebir TR karşılığı; TR terminolojisinde "claim" geçmez | #1 satırı; TR:17; TR:5 | uygulandı |
| 2T | 8 (TR) | "olup, bölge kaymasının, referans" | "olup, İstem 1'de tanımlı bölge kaymasının, referans" | EN #2'nin birebir TR karşılığı | #2 satırı; TR:19; TR:5 | uygulandı |
| 3 | 1 | "the method" | — (dokunulmadı) | "a computer-implemented machine learning method ... the method comprising" aynı giriş cümlesinde (EN:5) | B:L19 `term: "the method"` | GERİ ALINDI (yanlış-pozitif adayı) |
| 4 | 1 | "the task input" | — | "from a task input" EN:5'te hemen önce anılmış | B:L30 `term: "the task input"` | GERİ ALINDI (yanlış-pozitif adayı) |
| 5 | 1 | "the trained model" | — | "of a trained model" EN:5 girişi | B:L41 `term: "the trained model"` | GERİ ALINDI (yanlış-pozitif adayı) |
| 6 | 1 | "the context-encoded representations and processing the derived reader input" | — | Araç uzun dizi kaptı; "to produce context-encoded representations" EN:5'te hemen önce | B:L52 (aynı `term`) | GERİ ALINDI (yanlış-pozitif adayı) |
| 7 | 1 | "the trained model" (2. kayıt) | — | #5 ile aynı terimin yinelenen kaydı; öncül EN:5 | B:L63 (aynı `term`) | GERİ ALINDI (yanlış-pozitif adayı) |
| 8 | 1 | "the task input consists of one of an ordered sequence of" | — | (a) bendi; "task input" girişte tanımlı (EN:5→EN:7) | B:L74 (aynı `term`) | GERİ ALINDI (yanlış-pozitif adayı) |
| 9 | 1 | "the measured field values of" | — | "a plurality of measured field values" aynı cümlede hemen önce (EN:7) | B:L85 (aynı `term`) | GERİ ALINDI (yanlış-pozitif adayı) |
| 10 | 1 | "the point cloud of" | — | "constructing a point cloud" aynı bentte hemen önce (EN:7) | B:L96 (aynı `term`) | GERİ ALINDI (yanlış-pozitif adayı) |
| 11 | 1 | "the point features and the pair-geometry features into a single set representation as the reader input by a symmetric set aggregation" | — | Terimler (b) EN:9 ve giriş EN:5'te; araç bent-ötesi uzun dizi kaptı | B:L107 (aynı `term`) | GERİ ALINDI (yanlış-pozitif adayı) |
| 12 | 1 | "the task output from the set representation" | — | "task output" EN:5, "set representation" (c) EN:11 | B:L118 (aynı `term`) | GERİ ALINDI (yanlış-pozitif adayı) |
| 13 | 1 | "the task output" | — | "producing a task output" EN:5 girişi | B:L129 `term: "the task output"` | GERİ ALINDI (yanlış-pozitif adayı) |
| 14 | 1 | "the point cloud of the trained model itself" | — | "point cloud" (a) EN:7; "trained model" EN:5 | B:L140 (aynı `term`) | GERİ ALINDI (yanlış-pozitif adayı) |
| 15 | 1 | "the task output as contribution data of" | — | "task output" EN:5; "contribution data" aynı bentte ilk kez ve "as" ile tanıtılıyor (EN:17) | B:L151 (aynı `term`) | GERİ ALINDI (yanlış-pozitif adayı) |
| 16 | 1 | "the records" | — | "a plurality of records" aynı cümlede hemen önce (EN:19) | B:L162 `term: "the records"` | GERİ ALINDI (yanlış-pozitif adayı) |
| 17 | 3 | "the pair-geometry features" | — | Bağımlılık "The method of claim 1" ile İstem 1(b) EN:9'daki tanım aktarılır | B:L173 `term: "the pair-geometry features"`; EN:27 | GERİ ALINDI (yanlış-pozitif adayı) |
| 18 | 4 | "the symmetric set aggregation" | — | İstem 1(c) EN:11 "a symmetric set aggregation"; atıf 1 | B:L184 | GERİ ALINDI (yanlış-pozitif adayı) |
| 19 | 4 | "the point features and average-pooling over the point pairs" | — | "point features"/"point pairs" İstem 1(b) EN:9; araç uzun dizi kaptı | B:L195 | GERİ ALINDI (yanlış-pozitif adayı) |
| 20 | 4 | "the max-pooling and a result of the average-pooling" | — | "max-pooling ... average-pooling" aynı cümlede hemen önce (EN:29) | B:L206 | GERİ ALINDI (yanlış-pozitif adayı) |
| 21 | 5 | "the task input at which context" | — | "task input" İstem 1 girişi EN:5; atıf 1 | B:L217 | GERİ ALINDI (yanlış-pozitif adayı) |
| 22 | 6 | "the point-removal effect" | — | İstem 1(e)(i) EN:17 "a point-removal effect" (artikel ile) ; atıf 1 | B:L228 | GERİ ALINDI (yanlış-pozitif adayı) |
| 23 | 6 | "the reader" | — | "a reader of the trained model" EN:5; atıf 1 | B:L239 | GERİ ALINDI (yanlış-pozitif adayı) |
| 24 | 7 | "the records assigned" | — | "a plurality of records" EN:19; "records assigned to that region" aynı yapıda (EN:35) | B:L261 | GERİ ALINDI (yanlış-pozitif adayı) |
| 25 | 9 | "the reader" | — | EN:5 "a reader"; atıf 1 | B:L283 | GERİ ALINDI (yanlış-pozitif adayı) |
| 26 | 9 | "the model input identities" | — | "(ii) a quantization table mapping input values to model input identities" aynı istemde hemen önce (EN:39) | B:L294 | GERİ ALINDI (yanlış-pozitif adayı) |
| 27 | 9 | "the same package produces both the task output and a human-readable" | — | "a single self-contained package" aynı cümlede (EN:39); araç uzun dizi kaptı | B:L305 | GERİ ALINDI (yanlış-pozitif adayı) |
| 28 | 10 | "the task input" | — | EN:5 girişi; atıf 1 | B:L316 | GERİ ALINDI (yanlış-pozitif adayı) |
| 29 | 10 | "the multi-sentence set representation" | — | "to obtain a multi-sentence set representation" aynı cümlede hemen önce (EN:41) | B:L327 | GERİ ALINDI (yanlış-pozitif adayı) |
| 30 | 11 | "the set of global set statistics" | — | "computing a set of global set statistics" aynı cümlede hemen önce (EN:43) | B:L338 | GERİ ALINDI (yanlış-pozitif adayı) |
| 31 | 13 | "the instructions configures the system" | — | "instructions executable by the at least one processor, wherein execution of the instructions" EN:47 | B:L349 | GERİ ALINDI (yanlış-pozitif adayı) |
| 32 | 13 | "the instructions further configures the system" | — | Aynı "instructions" girişi EN:47 | B:L360 | GERİ ALINDI (yanlış-pozitif adayı) |
| 33 | 13 | "the context-encoded representations into d-dimensional vectors" | — | "to produce context-encoded representations" EN:47 girişte | B:L371 | GERİ ALINDI (yanlış-pozitif adayı) |
| 34 | 13 | "the task input consists of one of an ordered sequence of" | — | "processing a task input" EN:47 girişi | B:L382 | GERİ ALINDI (yanlış-pozitif adayı) |
| 35 | 13 | "the point cloud of" | — | "to construct a point cloud consisting of N points" aynı bentte (EN:49) | B:L393 | GERİ ALINDI (yanlış-pozitif adayı) |
| 36 | 13 | "the same point cloud from which the task output" | — | "a point cloud" EN:49; "task output" EN:47 | B:L404 | GERİ ALINDI (yanlış-pozitif adayı) |
| 37 | 14 | "the model input identities" | — | aynı istemde "(ii) a quantization table mapping input values to model input identities" (EN:59) | B:L415 | GERİ ALINDI (yanlış-pozitif adayı) |
| 38 | 14 | "the interpretation engine" | — | İstem 13 EN:55 "an interpretation engine"; atıf 13 | B:L426 | GERİ ALINDI (yanlış-pozitif adayı) |
| 39 | 15 | "the set of instructions causes the computer" | — | "a set of instructions executable by a computer ... execution of the set of instructions" EN:61 | B:L437 | GERİ ALINDI (yanlış-pozitif adayı) |
| 40 | 15 | "the task input consists of one of an ordered sequence of" | — | "processing a task input" EN:61 girişi | B:L448 | GERİ ALINDI (yanlış-pozitif adayı) |
| 41 | 15 | "the point cloud of" | — | "constructing a point cloud consisting of N points" aynı cümlede (EN:61) | B:L459 | GERİ ALINDI (yanlış-pozitif adayı) |
| 42 | 15 | "the point cloud and" | — | aynı cümle: "for points of the point cloud and for point pairs" (EN:61) | B:L470 | GERİ ALINDI (yanlış-pozitif adayı) |
| 43 | 15 | "the task output from the set representation" | — | "a task output" EN:61 giriş; "a set representation" aynı cümlede önce | B:L481 | GERİ ALINDI (yanlış-pozitif adayı) |

## Özet

- Kayıt izi 43 (`patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/B_us_claims_en_antecedentON.json:9`). **Uygulanan düzeltme: 2 §** (istem 7 ve 8 EN; TR'ye birebir yansıdı,
  dosya başına 2 satır değişikliği — diff sayısı §kapılar). **GERİ ALINDI (yanlış-pozitif adayı): 41.**
- Eşik/sayı değişikliği yok: istemlerdeki sayılar (3–64, N+1, INT8) pct_v2 tabanından aynen kaldı.
- Kural tabanlı denetim: `patent_lint` (öncül dayanak / belirsiz dil) pct_v3 nüshalarında ayrıca koşulur
  (KURULUM_PLAN §6 kapı 2); bu defter yalnız `B` kaydının 43 izini karşılar.
- **vekil onayı gerekli** — "GERİ ALINDI" satırlarının her biri vekil tarafından tek satırda geri açılabilir.
