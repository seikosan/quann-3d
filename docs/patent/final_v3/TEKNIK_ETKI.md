# TEKNIK ETKI — Patent v3

Bu belge buluşun teknik etkisini, yalnız depoda ölçülmüş ve satır düzeyinde
doğrulanmış sayılarla özetler. Doğruluk üstünlüğü iddiası içermez (sınırlar
TARIFNAME.md §6, §8 ve §12'de kayıt altındadır); etki bellek, gecikme, eğitim
yükü, uç dağıtım ve ölçüm-doğruluk eksenlerindedir.

## 1. Etki eksenleri ve ölçülmüş karşılıkları

| Eksen | Ölçülen etki | Kaynak |
|---|---|---|
| Uç paket boyutu | FP32 dağıtım 528.9 KB → INT8 dağıtım 138.2 KB (k=13 bulut yapılandırması) | fraud/results/edge_eval_full_scale.json:15-16, fraud/results/edge_eval_full_scale.json:20 |
| Gecikme / verim | tek işlem gecikmesi p50 0.0159 ms, 62.845 işlem/s (k=13, n=2.000 işlem) | fraud/results/edge_eval_full_scale.json:20-21, fraud/results/edge_eval_full_scale.json:23, fraud/results/edge_eval_full_scale.json:25 |
| Kantizasyon doğruluk maliyeti | doğal oranlı n=4.700.000 ayrık testte FP32 AUC 0.9450746 → INT8-kanal 0.9367114 (fark −0.008363); düşüş ölçülüp sınır olarak kabul edilmiştir | fraud/results/edge_eval_full_scale.json:59, fraud/results/edge_eval_full_scale.json:62, fraud/results/edge_eval_full_scale.json:64, fraud/results/edge_eval_full_scale.json:66 |
| Görev-türü birliği | tek temsil hem dil modelleme (TEST yitimi 3.2318, ayrık 80/10/10 holdout) hem tabular sınıflandırma (AUC-ROC 0.9351) örneklerinde çalışır | logs/run_r5_cc_stable.log:1236; fraud/results/full_scale_hybrid.json:10 |
| Eğitim yükü | nokta konumları gradyansız, sıfırıncı mertebeden kabul/geri alma ile daraltılır; aday konumun görev yitimi değişimi eşik üzerinden kabul edilir | quann_cloud/src/qcloud_quann.py:212, quann_cloud/src/qcloud_quann.py:220 |
| Hesap karmaşıklığı varyantı | Transformer ve tam çift hesabı olmadan O(N) küresel-istatistik agregatörü (performans iddiası taşımaz, seçenek rejimidir) | purecloudlm/model.py:57 |

## 2. Açıklanabilirliğin ölçülebilirliği (aynı temsilin doğal çıktısı)

1. **Bölge kayması tek skalerle izlenir:** şikayet korpusunda drift−eğitim
   bölge nokta-payı farkları −0.170 … +0.207 arasında ölçülmüştür
   (kaynak: complaints/results/complaints_scale_stability.json:64-73, Şekil 6a).
2. **Kontrollü ortam kayması alarm üretimini artırır:** C-MAPSS ayar kayması
   deneyinde alarm oranı 0.0117'den 0.0281'e çıkmış, orantı 2.4× olarak
   kaydedilmiştir (kaynak: maintenance/results/e1_cmapss.json:190-192,
   Şekil 6b).
3. **Denetim kararlılığı tek sayıya indirgenir:** beş tohumda bölge etiketi
   kararlılığı 0.325 (kaynak: aml/results/e5_stability.json:29); bu, aynı
   temsilin yeniden eğitimler arası karşılaştırılabilir olduğunu gösterir.
4. **Karşı-olgusal nokta etkisi:** bir noktanın çıkarılması K+1 ileri geçişle
   ölçülür; model-sonrası ayrı bir araç gerekmez
   (kaynak: fraud/src/explain_pilot.py:93-112, Şekil 4).

## 3. Etkinin sınırı (dürüst kayıt)

- INT8 geçişinin AUC maliyeti −0.008363 … −0.008593 aralığında ölçülmüştür;
  yararlılık iddiası bu maliyetin kabulüne dayanır
  (kaynak: fraud/results/edge_eval_full_scale.json:62-66).
- Sayılar tek yapılandırma/korpus kombinasyonlarında ölçülmüştür (k=13,
  D=8, d_cloud=128, d_model=32: fraud/results/edge_eval_full_scale.json:3-6, fraud/results/edge_eval_full_scale.json:20);
  genelleme iddiası kurulmaz.
- Model XGBoost veya nedensel dil-modellemeden daha doğru değildir; iddia
  doğruluk değil, görev çıktısı ile açıklamanın AYNI noktadan türemesidir
  (sınırlar: TARIFNAME.md §6, §8, §12).
