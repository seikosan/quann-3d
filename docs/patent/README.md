# Patent Paketi

Düşük-Boyutlu Nokta-Bulutu Geometrisi temelli icatlar için **doğrudan başvuruya hazır**
üç dosya paketi. Her klasör kendi metin kaynakları + `fig/` çizimlerini içerir
(self-contained). Detaylı gönderim/strateji notu: [`VEKIL_GONDERIM_NOTU.md`](VEKIL_GONDERIM_NOTU.md).

## Paketler

| Klasör | İcat | Dil | İddia | Çizim | Not |
|---|---|---|---|---|---|
| [`dil/`](dil/) | Düşük boyutlu nokta bulutu geometrisi tabanlı **dil modelleme** | TR + EN | 3 bağımsız + 16 bağımlı = 19 | 8 SVG | **T1** — Türkiye ulusal başvuru |
| [`evrensel/`](evrensel/) | Çift-geometri nokta bulutu: birleşik hesaplama + native açıklanabilirlik (dil/fraud/AML/bakım) | TR + EN | 3 bağımsız + 10 bağımlı = 13 | 3 SVG + 6 PNG | **T2** — PCT uluslararası (bağımsız) |
| [`tabular/`](tabular/) | Tabular/fraud kimlik-bucket bulut, açıklama, edge | TR + EN | alt küme | 1 PNG | çekirdeğin (evrensel) alt kümesi |

**Strateji:** `dil` = tek dikey, düşük risk → önce. `evrensel` = 4 dikeyli birleşik çekirdek,
`dil`'den **öncelik alamaz** (kapsam farklı) → bağımsız T2. `tabular` = evrenselin alt kümesi
(cross-ref; ayrı dosya olarak referans). Ayrıntı + vekile not: [`VEKIL_GONDERIM_NOTU.md`](VEKIL_GONDERIM_NOTU.md) §2.

## İçerik düzeni (her pakette)

```
<ad>/
  <metin>.md            # kaynak (TR veya EN)
  <metin>.docx          # filing-ready (build_docx.py ile üretilir)
  fig/                  # çizimler (kaynak SVG +/veya PNG)
```

- `dil/`: `TURKPATENT_basvuru_v2.3.{md,docx}`, `patent_application_v2.5.{md,docx}`
- `evrensel/`: `geometry_dual_use_core_v1.{md,docx}`, `geometry_dual_use_core_v1_EN.{md,docx}`
- `tabular/`: `patent_tabular_explain_v1.{md,docx}` plus English Markdown source

## Çizim kaynakları

- `dil/fig/fig1..8_*.svg` — 8 çizim; kaynak `docs/figures/` (repo kökü viz fig'leri).
- `evrensel/fig/fig1..3_*.svg` — 3 yeni çekirdek çizim; `fig4..7*.png` — dikey artifact'lerin
  kopyaları (kaynak: `aml/figures`, `fraud/figures`, `maintenance/figures`).
- `tabular/fig/fig1_sistem_mimarisi.svg`, `fig2_fraud_bulut.svg`, `fig3_ablation.svg`,
  `fig4_karsifaktuel.svg`, `fig5_dagitim.svg` — siyah-beyaz, referanslı çizimler;
  `fig2_fraud_bulut.svg`, `fig1_fraud_bulut.png` kaynağını etiketli olarak kullanır.

Not: `fig/` klasöründeki PNG'ler, dikey deney figlerinin **kopyasıdır** (paket self-contained
olsun diye). Tek kaynak doğru (source of truth) dikey `figures/` klasörleridir; orası değişirse
buradaki kopya yeniden üretilir.

## .docx üretim

Filing-ready `.docx` dosyaları [`build_docx.py`](build_docx.py) ile markdown kaynaktan üretilir:

```
python3 docs/patent/build_docx.py
```

Ara PNG'ler ve `build_docx.log`, `docs/patent/_render/`'a yazılır (gitignore'da;
SVG/dikey kaynaktan yeniden üretilir). Tabular paketteki çizim şemaları `tabular/fig/` altındaki
referans numaralı SVG kaynaklarıdır; fraud bulutu mevcut PNG kaynağından hazırlanır.
`.docx` dosyaları commit'lenir (teslimat ürünü).

## Vekil yazışmaları

- [`VEKIL_ANLATIM.md`](VEKIL_ANLATIM.md) — buluşun ana yapısı, jargonsuz anlatım.
- [`VEKIL_SORU_CEVAP.md`](VEKIL_SORU_CEVAP.md) — vekilden gelen sorular ve kod-kanıtlı yanıtları.

## Arşiv

Sürüm dışı eski taslaklar: [`_archive/`](_archive/README.md).
