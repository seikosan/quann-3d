# REFERANS NUMARALARI — pct_v3 (koor-dinasyon künyesi, taslak)

- **Amaç:** tarifname, özet ve şekillerde **aynı** referans işaretlerinin kullanılmasını sağlamak.
  pct_v2'de tarifname ve şekillerde referans numarası **yoktu** (ölçüm:
  `grep -oE '\([0-9]{2,3}\)' docs/patent/pct_v2/filing/DESCRIPTION_FILING-COPY_EN.txt` → eşleşme yok §;
  `docs/patent/pct_v2/fig/fig_manifest.json` içinde rakam-etiketi yalnız ölçü eksenlerinde §).
- **Kullanım kuralları:**
  1. **İstemlerde referans işareti kullanılmaz** (PCT Rule 13.4 — istem işaretlerle sınırlanmış sayılmaz) §.
  2. **Tarifname** her şekil numarasını ilk anışında parantez içinde verir: "reader (116)" §.
  3. **Özet** anıştığı unsurlar için parantezli numara taşır (PCT Rule 47(4); bu turun `H` bulgusu) §.
  4. **Şekillerde** numara parantezsiz, tek satır etiketi olarak durur; üç basamağı geçmez §.
- **Durum:** numara değerleri **taslak telif kararıdır** (vekil yeniden numaralandırabilir);
  önemli olan üç dosyanın (tarifname / özet / şekil) aynı tabloyu kullanmasıdır.
- **Kaynak:** unsur adları pct_v2 şekil metinlerinden aynen alınmıştır
  (`docs/patent/pct_v2/fig/fig_manifest.json` `texts[].text` alanları).

## 1 — Sistem mimarisi (FIG. 1)

| Numara | Unsur (Şekil 1 etiketi) |
|---|---|
| 100 | task input (token sequence / tabular record) |
| 102 | context encoder |
| 104 | point projection (D-dimensional) |
| 106 | point cloud P (N points) |
| 110 | pair-geometry encoder (points and pairs) |
| 112 | aggregation module (one set representation) |
| 116 | reader |
| 118 | task output |
| 120 | interpretation engine |
| 122 | point-removal effect |
| 124 | region naming |
| 126 | region shift |
| 128 | output interface |

## 2 — Çift geometrisi (FIG. 2)

| Numara | Unsur |
|---|---|
| 200 | first point |
| 202 | second point |
| 204 | difference vector |
| 206 | pair distance (Euclidean norm) |
| 208 | cosine similarity |
| 210 | norm vector of the first point |
| 212 | norm vector of the second point |
| 214 | concatenation of the five components |
| 216 | pair-feature width (etiket Şekil 2'de "3D + 2" olarak çizilir; tarifname [0033] "beş bileşen" der) |
| 218 | average-pooling over the point pairs |

## 3 — Simetrik küme toplaması (FIG. 3)

| Numara | Unsur |
|---|---|
| 300 | point features |
| 302 | pair-geometry features |
| 304 | max-pooling branch over the points |
| 306 | average-pooling branch over the point pairs |
| 308 | concatenation layer |
| 310 | one set representation |
| 312 | global set statistics variant (maximum · minimum · mean · standard deviation pooling, cloud centre, centre deviation) |

## 4 — Nokta çıkarma etkisi (FIG. 4)

| Numara | Unsur |
|---|---|
| 400 | point cloud P (N points) |
| 402 | reader, unmasked path |
| 404 | base output |
| 406 | masked cloud variant (point k zeroed, mask closed) |
| 408 | reader, masked path |
| 410 | masked output |
| 412 | contribution (base output minus masked output) |
| 414 | batch input of the N+1 forward passes |

## 5 — Bölge adlandırma (FIG. 5)

| Numara | Unsur |
|---|---|
| 500 | point cloud states (training window) |
| 502 | centroid-based clustering |
| 504 | region statistics (centre · point share) |
| 506 | human-readable region label |

## 6/7 — Ölçümlü şekiller (FIG. 6, FIG. 7)

| Numara | Unsur |
|---|---|
| 600 | panel (a) region point-share shift |
| 602 | panel (b) setting shift → alarm rate |
| 700 | panel (a) quantization effect |
| 702 | panel (b) delivered package footprint |
| 704 | INT8 quantized weights |
| 706 | quantization table |
| 708 | resolution table (model input identities → human-readable labels) |

## 8 — Pencere bulutu (FIG. 8)

| Numara | Unsur |
|---|---|
| 800 | sentences of the text task input |
| 802 | sentence cloud states (first tier) |
| 804 | neighbouring window cloud states (second tier) |
| 806 | window-geometry encoder |
| 808 | multi-sentence set representation |
| 810 | reader → task output (third tier) |

## Ortak unsurlar (birden çok şekilde anışılır)

| Numara | Unsur |
|---|---|
| 904 | deployment package (self-contained) |

**Not:** `deployment package (904)` Şekil 7'de ayak izi paneliyle birlikte etiketlenir; §
`INT8 quantized weights (704)`, `quantization table (706)`, `resolution table (708)` aynı şekilde §
kompakt bir blok şeridi olarak gösterilir. Tablo **yalnız şekillerde fiilen çizilen** unsurları §
taşır — çizilmeyen unsurlar (ör. `distribution store`, `model input identities`) tarifnamede §
işaretsiz anılır. Şekil 6 ve 7 **ölçümlüdür** (`measured: true`) ve sayıları §
`docs/patent/pct_v2/OLCUM_KAYITLARI.md` defterindeki ham kaynaklardan gelir — yeni sayıda uydurulmaz.
