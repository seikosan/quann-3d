# Patent v3 — Nihai Başvuru Paketi (son sürüm)

Bu klasör `docs/patent/final_v3/`, önceki taslaklara bağlı olmayan nihai TÜRKPATENT
başvuru paketidir. Buluş çekirdeği: **tek nokta-bulutu geometri temsilinden hem
görev çıktısı hem açıklama üretilmesi** (nokta çıkarma etkisi · bölge isimlendirme ·
bölge kayması). Yazım kuralları ve kırmızı çizgiler: `BLUS_NOTU.md`.

## 1. Tek dosyada teslim

| Dosya | Ne içerir | Sayfa |
|---|---|---|
| `basvuru_v3.pdf` | Başvuru sırası: ÖZET → TARİFNAME → İSTEMLER → ŞEKİLLER (kapaksız) | 26 |
| `inceleme_v3.pdf` | Kapak + aynı sıra + TEKİK ETKİ (vekil/inceleme nüshası) | 29 |

(Sayfa sayıları `pdfinfo basvuru_v3.pdf` / `pdfinfo inceleme_v3.pdf` çıktısıdır.)

## 2. Kaynak belgeler (düzenlenebilir)

| Dosya | Açıklama |
|---|---|
| `OZET.md` → `ozet_v3.docx/pdf` | Özet, 265 kelime; teknik etki cümleleriyle |
| `TARIFNAME.md` → `tarifname_v3.docx/pdf` | §1–§13; ölçümlü örnekler (§6–§10); §12 dürüstlük kayıtları; her sayı `dosya:satır` atıflı |
| `ISTEMLER.md` → `istemler_v3.docx/pdf` | 15 istem; bağımsız: 1 (yöntem), 13 (sistem), 15 (CRM/okunabilir ortam) |
| `TEKNIK_ETKI.md` → `teknik_etki_v3.docx/pdf` | Etki eksenleri (bellek, gecikme, eğitim yükü, ölçüm-doğruluk) + dürüst sınırlar |
| `BLUS_NOTU.md` | Buluş çekirdeği, embodiment tablosu, kırmızı çizgiler (iç çalışma belgesi) |

## 3. Şekiller (`fig/*.svg` — siyah-beyaz, A4)

Ölçümlü şekiller YALNIZ gerçek JSON alanlarından çizilir (`generate_figures_v3.py`
içinde `json.loads`); elle yazılmış ölçüm sayısı yoktur. Şematik şekillerin künyesinde
"şematiktir" yazar.

| Şekil | Dosya | Tür | Anlatım | Dayanak |
|---|---|---|---|---|
| 1 | `fig1_sistem_mimarisi.svg` | şematik | Girdi → bağlam kodlayıcı → D-boyutlu projeksiyon → nokta bulutu → (a) okuyucu→görev çıktısı, (b) aynı buluttan açıklanabilirlik üçlüsü | İstem 1 |
| 2 | `fig2_cift_geometri.svg` | şematik | Nokta çifti; bileşenler [Δp(D), ‖Δp‖(1), cosθ(1), ‖pᵢ‖(D), ‖pⱼ‖(D)] ∈ R^{3D+2} | `cloudlm/model.py:63-64`; İstem 3 |
| 3 | `fig3_simetrik_agregasyon.svg` | şematik | Nokta max-pool + çift mean-pool birleştirmesi; kesikli yol: O(N) küresel-istatistik varyantı | `cloudlm/model.py:153,177`; İstem 4, 11 |
| 4 | `fig4_nokta_cikarma.svg` | şematik | P ve P\{pₖ} için K+1 ileri geçiş; etki eₖ = s_tam − s₋ₖ | `fraud/src/explain_pilot.py:93-112`; İstem 6 |
| 5 | `fig5_bolge_isimlendirme.svg` | şematik | KMeans → bölge istatistikleri → insan-okunur bölge etiketi | `aml/src/e1_explain.py:296-342`; İstem 7 |
| 6 | `fig6_bolge_kaymasi.svg` | **ölçümlü** | (a) şikâyet korpusunda bölge nokta-payı kayması (−0.170…+0.207); (b) C-MAPSS ayar kayması alarm oranı 0.0117→0.0281, orantı 2.4× | `complaints/results/complaints_scale_stability.json:64-73`; `maintenance/results/e1_cmapss.json:190-192`; İstem 8 |
| 7 | `fig7_int8_paket.svg` | **ölçümlü** | (a) AUC FP32 0.9451 / INT8-tensor 0.9365 / INT8-kanal 0.9367 (n=4.700.000); (b) paket 528.9→138.2 KB | `fraud/results/edge_eval_full_scale.json:15-16,59-66`; İstem 9 |
| 8 | `fig8_window_bulut.svg` | şematik | Cümle-bulut → komşu pencere-bulut → ContextCloudLM hiyerarşisi | `contextcloud/model.py:20,121,196,246`; İstem 10 |

## 4. Üretim hattı (bu klasörde)

```
python3 generate_figures_v3.py    # 1. 8 SVG (ölçümlüler gerçek JSON'dan)
python3 build_v3.py               # 2. md→DOCX→PDF + kapılar:
                                  #    "20 nokta" yasağı, şekil-atıf denetimi,
                                  #    sıkı siyah-beyaz PNG denetimi
python3 build_deliverables.py     # 3. kapak + basvuru_v3.pdf + inceleme_v3.pdf
```

## 5. Denetim durumu (18 Kas 2026)

- Bağımsız reviewer turu: 71 sayı-atıf ham kaynakta ✓; kırmızı çizgi ihlali 0;
  uydurma sayı 0. Bulunan 2 atıf hatası + 5 künye boşluğu giderildi.
- Linter: `ISTEMLER.md` 0 hata. Kalan denetçi bulguları şekil/bölüm/istem NUMARASI
  kaynaklı yanlış-pozitiflerdir (ölçüm hatası değildir).
- Vekil kararları bekliyor: özet uzunluğu (265 kelime), 3 bağımsız istemin birliği,
  iki-bölümlü kalıp tercihi (`BLUS_NOTU.md` ile birlikte götürülecek).

## 6. Bu paket NE iddia etmez

Doğruluk üstünlüğü iddiası yoktur: tabular karşılaştırmada XGBoost AUC 0.9934,
dil görevinde nedensel temel çizgi PPL 2.94 aleyhte sınır olarak tarifnamede
kayıtlıdır (§6, §8, §12). Kazanım kümesi: görev çıktısı ile açıklamanın AYNI
temilden türemesi, INT8 uç paket, bölge-kayması gözlemi ve ölçümle izlenebilirlik.
