# ÖZET / ABSTRACT (PCT Kural 8)

**İki nüsha vardır:** aşağıdaki **ÖZET (EN)** uluslararası yayın için esas nüshadır; **ÖZET (TR)**
TÜRKPATENT'e verilecek Türkçe nüshadır. Her ikisi de Kural 8.1(b) gereği 50-150 kelime aralığındadır
(kelime sayıları dosyanın sonunda ölçüm olarak kayıtlıdır). Kural 8.1(c) gereği övme/değerlik yargısı
içermez; Kural 8.1(d) gereği referans işareti kullanılmaz — şekillerde modül referans numarası yoktur
(bkz. `UYGUNLUK_NOTU.md#D3`). İç künyeler HTML yorumundadır, başvuru metnine basılmaz.

---

## ÖZET (EN) — başvuru nüshası

A computer-implemented method constructs a point cloud of N points in a D-dimensional space from a task
input, which is an ordered token sequence or the measured field values of a tabular record. Point
features and pair-geometry features derived from the positions of point pairs are combined by a set
aggregation independent of input order into one set representation, from which a reader produces a task
output. Explanation data is obtained from that same point cloud of the trained model by a point-removal
effect, region naming or region shift, without a separate explanation model. A single self-contained
deployment package carries INT8 quantized weights together with a quantization table and a resolution
table mapping model input identities to human-readable labels, so that a package of 138.2 KB delivers
both a decision and its audit on an edge device.

## ÖZET (TR) — başvuru nüshası

Görev girdisi, sıralı bir token dizisi veya bir tabular kaydın ölçülen alan değerleri ise D-boyutlu
uzayda N noktadan oluşan nokta bulutu kurulur. Nokta özellikleri ile nokta çiftlerinin konumlarından
türeyen çift-geometri özellikleri, girdi sırasından bağımsız bir küme agregasyonuyla tek bir küme
gösteriminde birleştirilir ve bir okuyucu bu gösterimden görev çıktısı üretir. Açıklama verisi, eğitilmiş
modelin aynı nokta bulutu üzerinden, nokta çıkarma etkisi, bölge isimlendirme veya bölge kayması ile
elde edilir; ayrı bir açıklama modeli gerekmez. Tek bir kendi kendine yeten dağıtım paketi, INT8 kantize
ağırlıklar ile model girdi kimliklerini insan-okunur etiketlere eşleyen iki tablo taşır; 138.2 KB'lık bir
paket uç aygıtta hem kararı hem de denetimini verir.

---

## İç künyeler (başvuru metni değildir)

| Alan | Değer | künye |
|---|---|---|
| N, D, küme agregasyonu, okuyucu, üç açıklama işlemi | yapısal tanım | docs/patent/pct_v2/CLAIMS_EN.md:istek 1; DESCRIPTION_EN.md#S9 |
| INT8 + nicelleştirme/çözümleme tabloları | yapısal tanım | fraud/results/deploy/manifest.json:21, 23, 97-98 |
| 138.2 KB | INT8 dağıtım ayak izi | fraud/results/edge_eval_full_scale.json:16 (bkz. OLCUM_KAYITLARI.md#S1) |

## Ölçüm (bu oturumda `wc -w` ile)

| Nüsha | kelime | Kural 8.1(b) 50-150 |
|---|---|---|
| ÖZET (EN) | 137 kelime | 137 → 50-150 aralığında ✓ |
| ÖZET (TR) | 106 kelime | 106 → 50-150 aralığında ✓ |

*Taslak — vekil onayı gerekli.*
