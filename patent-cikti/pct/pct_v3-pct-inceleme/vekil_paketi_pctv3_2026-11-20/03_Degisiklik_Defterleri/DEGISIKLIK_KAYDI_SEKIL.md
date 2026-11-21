# DEĞİŞİKLİK KAYDI — ŞEKİLLER (pct_v2 → pct_v3)

Ham çıktılar satır numaralı günlüklerde durur; her sayı kendi satırındaki atfa bağlıdır:
`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt` · `patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt` ·
`patent-cikti/pct/pct_v3-pct-inceleme/sekil_diff_olcumlu.txt` · `patent-cikti/pct/pct_v3-pct-inceleme/olcum_kayitlari.txt`

- **Üretici:** `docs/patent/pct_v3/generate_figures_pct3.py` (kaynak: `docs/patent/pct_v2/generate_figures_pct.py`, `cp` + metin etiketi yamaları §). PDF: `docs/patent/pct_v3/build_drawings_pct3.py` (kaynak: `build_drawings.py`, yalnız künye değişti §).
- **Kapsam:** geometri, kutu konumları/boyutları, oklar ve ÖLÇÜMLÜ sayılar aynen korundu; yalnız (a) etiket metinlerine `REFERANS_NUMARALARI.md` işaretleri eklendi §, (b) iki etikete sığmayan kutularda numara ikinci satıra alındı §, (c) FIG. 2'de iki etiket çakışınca `difference vector`/`pair distance` etiketleri 1,75 → 1,45'e kaydı (`docs/patent/pct_v3/generate_figures_pct3.py:389`), (d) FIG. 7 altına tek satırlık şerit eklendi: `deployment package 904: ... 704 · 706 · 708` (`docs/patent/pct_v3/generate_figures_pct3.py:584`).
- **Ham doğrulama:** üretim exit 0 (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:20`) · `KUTU-SIGMA: 0 tasma` (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:19`) · PDF `KABUL: 8 sayfa A4` (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:56`) · PDF boyutu 979834 B (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:45`) · metin katmanı 0 karakter (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:55`).

## 1 — Şekil bazında eklenen işaretler §

| Şekil | Eklenen numaralar (etiket sonuna/uygun satıra) | Kanıt |
|---|---|---|
| FIG. 1 | 100, 102, 104, 106, 110, 112, 116, 118, 120, 122, 124, 126, 128 | 13/13 |
| FIG. 2 | 200, 202, 204, 206, 208, 210, 212, 214, 216, 218 | 10/10 |
| FIG. 3 | 300, 302, 304, 306, 308, 310, 312 | 7/7 |
| FIG. 4 | 400, 402, 404, 406, 408, 410, 412, 414 | 8/8 |
| FIG. 5 | 500, 502, 504, 506 | 4/4 |
| FIG. 6 | 600 (panel a başlığı), 602 (panel b başlığı) | 2/2 |
| FIG. 7 | 700, 702 + şerit: 904, 704, 706, 708 | 6/6 |
| FIG. 8 | 800 (her cümle kutusu), 802, 804, 806, 808, 810 | 6/6 |

Numara kapsama testinin ham satırı (yukarıdaki 13/13 … 6/6 sayaçları): `patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:23`.

## 2 — Ölçümlü şekillerde dokunulmazlık kanıtı §

- `grep -o -E` sayaçları FIG.6/7 SVG (pct_v2 = pct_v3) `0.9451` 1|1 · `0.9365` 1|1 (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:26-27`); `0.9367` 1|1 · `528.9 KB` 1|1 · `138.2 KB` 1|1 (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:28-30`).
- Sayı içeren `<text>` satırlarının `diff`i FIG. 6 (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_diff_olcumlu.txt:3-8`): yalnız iki panel başlığı eklendi; ölçüm etiketlerinde fark **yok**.
- Diff FIG. 7 (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_diff_olcumlu.txt:9-15`): iki panel başlığı + şerit satırı eklendi; ölçüm etiketlerinde fark **yok** (diff yalnız `>` satırları veriyor).
- `docs/patent/pct_v3/fig/fig_manifest.json` §: `measured` bayrakları (FIG. 6/7 `true`, diğerleri `false`) ve `sources` alanları pct_v2 ile birebir (`complaints/results/complaints_scale_stability.json`, `maintenance/results/e1_cmapss.json`, `fraud/results/edge_eval_full_scale.json`).
- Şematik şekillerde `Schematic; no measurement.` notu duruyor §; ölçüm değerleri elle yazılmadı, betik gerçek JSON'lardan çizdi §.

## 3 — Fizik/uygunluk ölçümleri

- Metin yüksekliği pct_v2 en küçük **0,2175** (FIG. 4) ve pct_v3 en küçük **0,2175** (FIG. 4) — PCT Kural 11.13 asgari 0,21 cm hedefinin üstünde (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:34`).
- FIG. 7 metin yüksekliği 0,2456 → 0,2297 (yeni şerit 7,5 pt) (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:34`).
- PDF `pdfinfo` `Pages: 8` ve `Page size: 595.304 x 841.89 pts (A4)` (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:46`).
- Her sayfada mürekkep; en düşük %1,06 sayfa 4 (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:50`), tam liste `patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:47-54`.
- Kutu-sığma denetimi (`_check_fits`, piksel ölçümü): 0 taşma (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:19`).
- En dolu kutu FIG. 2, %97 genişlik (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:14`).

## 4 — pct_v2 dokunulmazlığı

- Bu görev `docs/patent/pct_v2/` altında yalnız `cp`/`grep`/`diff` okumaları yaptı; hiçbir pct_v2 dosyasına yazmadı §. Kanıt 1 §: üretim stdout'u yalnız `docs/patent/pct_v3/fig/` dosyalarını listeliyor (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:1-12`).
- Kanıt 2 §: yazılan PDF tek dosyadır, 979834 B (`patent-cikti/pct/pct_v3-pct-inceleme/sekil_dogrulama.txt:45`).
- Kanıt 3 §: pct_v2'nin sekiz dosyasının sha1 künyesi `patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:29-36` satırlarında sabitlendi.
- **Güncelleme § (2026-11-20 akşamı, kullanıcı kararı (a)):** v2 tabanı bilerek değişti; yeni künyede
  8 sha1'den 7'si aynı kaldı (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama_v2kabul_2026-11-20.txt:15`);
  şekil hattı etkilenmedi: PDF ve manifest sha'ları yeni künyenin sha bloğunda aynen duruyor
  (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama_v2kabul_2026-11-20.txt:7-14`). Yukarıdaki künye
  tarihsel kanıt olarak geçerlidir.
- pct_v2 `fig_manifest.json` `uretic` alanı hâlâ `docs/patent/pct_v2/generate_figures_pct.py` (`docs/patent/pct_v2/fig/fig_manifest.json:2`).

**Not:** şekillerde referans işaretleri parantezsizdir; istemlerde işaret yoktur — üç basamaklı işaret sayaçları EN ve TR istem nüshalarında 0 çıktı (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama.txt:7-8`, PCT Kural 13.4).
Şekil numaralandırması ve şerit etiketleri telif kararıdır — **çizim künyesi vekil onayı ile sabitlenir.**
