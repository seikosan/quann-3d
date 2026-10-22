# maintenance/ — Endüstriyel Makine Sağlığı

## Ürün odağı

**Binary drift/alarm modu:** pencereyi bulut temsiline çevir, normal çalışma
bölgesinden uzaklaşma veya ikili arıza sinyali görülünce bakım ekibini uyar.
Mühendis anlatısı, sınıf etiketi yerine bölge haritasındaki değişimi açıklar:
"geçen hafta batı bölgesi genişledi." Bu cümle, zamana göre bölge karşılaştırmalı
ürün deneyiminin örneğidir; mevcut PCA haritası tek başına haftalık trend kanıtı
değildir.

- CWRU record-disjoint binary değerlendirmesi: hybrid ROC-AUC **0.9995**,
  PR-AUC **0.99997**, accuracy **0.9958**; 480 segment/20 test kaydı
  (`results/cwru_expansion.json`). Testte normal kayıt sayısı azdır; segmentler
  kayıt içi bağımsız örnekler değildir.
- C-MAPSS: sağlıklı referans bulutuna uzaklıkla kontrollü çalışma ayarı kayması
  alarm oranını **2.4×** artırdı; bölge/prototip açıklama kapsamı **%81.7**
  (`results/e1_cmapss.json`). Bu, arıza türü teşhisi değil drift demosudur.
- **Çok-sınıflı arıza türü tespiti ürün iddiası değildir.** Record-disjoint
  CWRU 3-class sonucu hybrid **.678** ve GBT **.900**; hybrid inner recall
  **0.00** (`results/e5_type_design.json`).

## Edge deploy kanıtı — kriter #3

`results/edge/MANIFEST.md` ölçüm raporu; ONNX/INT8 dosyaları ve SHA-256
manifesti `results/edge/` altında. Binary C-MAPSS görevindeki kayıtlı
checkpoint'ler içinde PR-AUC/F2 en yüksek `quann_binary_backprop.pt` seçildi.
ONNX girişi `[batch, 10, 36]` boyutlu, önceden ölçeklenmiş nokta bulutudur.
Checkpoint eğitim ölçekleyicisini içermediğinden ham sensörden ONNX'e tam
ön-işleme paketi değildir.

FP32 ONNX Torch ile **5.1e-7** maksimum mutlak farka sahip; batch=1/16/64
CPU median gecikmesi **0.0408/0.2868/1.0405 ms**. INT8 dosyası FP32'den küçük
(**78,924 B** vs **120,930 B**) ama bu ARM CPU'da daha yavaş ve Torch'a göre
batch=64'te **0.0207** fark üretiyor. Bu nedenle INT8 artefaktı oluşmuştur;
alarm eşiği/kalite eşdeğerliği kanıtlanmış sayılmaz. Ayrıntı:
`results/edge/MANIFEST.md`.

## Veri ve temsil

| Kaynak | Girdi | Kullanım |
|---|---|---|
| C-MAPSS FD001 | 36 özellik, 10-adım pencere | drift, bölge haritası, binary RUL alarmı |
| CWRU rulman | 12 kHz sinyalden 32 öznitelik | record-disjoint binary normal/arızalı karşılaştırması |

Pencere → bulut: her sensör örneği bir nokta, zaman penceresi nokta kümesidir.
Model ve veri hazırlama ayrıntıları `src/` altındadır.

## Deney durumu

| Deney | Durum | Sonuç |
|---|---|---|
| E0/E0b | tamam | C-MAPSS binary/RUL ve aşama baseline'ları; `results/e0b_cmapss.json` |
| E1 | tamam | CWRU binary; C-MAPSS bölge haritası ve drift alarmı |
| E4 | tamam | retrain'siz QUANN adaptasyonu avantaj göstermedi (`results/e4_adapt.json`) |
| E5 | tamam | 3-class karşılaştırması zayıf; ürün kapsamı dışında |
| Edge | tamam | ONNX FP32 + ORT dynamic INT8, latency/parite: `results/edge/` |

## Üretim sınırları

- .9995 CWRU ROC-AUC, belirli veri/kayıt ayrımındaki binary sonuçtur; saha
  performansı garantisi değildir ve C-MAPSS edge checkpoint'inin metriği değildir.
- Drift alarmı kontrollü ayar kayması demosudur; çok-sınıflı bakım teşhisi veya
  nedensel arıza açıklaması iddiası yoktur.
- CWRU normal kayıtları az ve E4 adaptasyon üstünlüğü gösteremedi; yeni tesis ve
  zaman ayrımlı doğrulama gerekir.

## Layout

`src/` eğitim, veri yükleyicileri ve edge export; `data/` girdiler;
`results/` ölçümler/checkpoint'ler; `figures/` bulut/bölge görselleri;
`logs/` deney logları; `tests/` doğrulamalar.
