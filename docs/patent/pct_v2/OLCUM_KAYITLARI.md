# ÖLÇÜM KÜNYELERİ — pct_v2 (iç denetim defteri, başvuru metni değildir)

Bu dosya `pct_v2/DESCRIPTION_EN.md` ve `DESCRIPTION_TR.md` içinde geçen **her sayının** ham
kaynak satırını tutar (künye biçimi `dosya:satır`). Her künye, sayının geçtiği ham satır
okunarak doğrulandı; doğrulama tekrar üretilebilir: ilgili satırı `sed -n 'Np' <dosya>`
komutuyla açın.

## S0 — İddia çerçevesi (sayı yok)

| Konu | künye |
|---|---|
| Doğruluk üstünlüğü iddiası yok; iddia = aynı temsilden görev çıktısı + açıklama, ayrı model olmadan | docs/patent/final_v3/TARIFNAME.md:9, 154-155, 171 |
| Uygulama alanları (uç, finans, makine sağlığı, metin) | docs/patent/final_v3/TARIFNAME.md:9 |

## S1 — Uç dağıtım / INT8 ölçümleri

| Değer | ham | künye |
|---|---|---|
| FP32 paket ayak izi | 528.9 KB | fraud/results/edge_eval_full_scale.json:15 |
| INT8 paket ayak izi | 138.2 KB | fraud/results/edge_eval_full_scale.json:16 |
| kullanılmayan dil parametresi | 8392 | fraud/results/edge_eval_full_scale.json:11 |
| gecikme p50 (işlem başına) | 0.015912843984551728 ms | fraud/results/edge_eval_full_scale.json:23 |
| gecikme p99 (işlem başına) | 0.017278228622744793 ms | fraud/results/edge_eval_full_scale.json:24 |
| verim | 62845.10641151401 işlem/s | fraud/results/edge_eval_full_scale.json:25 |
| gecikme ölçüm örneklemi | n_tx = 2000 | fraud/results/edge_eval_full_scale.json:21 |
| ONNX çalıştırma gecikmesi | 0.02578230228391476 ms | fraud/results/edge_eval_full_scale.json:50 |
| ONNX çalıştırma verimi | 39056.267050717805 işlem/s | fraud/results/edge_eval_full_scale.json:51 |
| INT8 sürüklenme örneklemi | 4096 girdi — ham kaynak niteleyicisi: "4096 random QUANN-point inputs (proxy)"; [0057] bu "vekil (proxy)" niteleyicisini taşır | fraud/logs/run_full_scale_edge_eval.log:46 |
| INT8 çıktı sürüklenmesi en büyük | 0.3432929813861847 | fraud/results/edge_eval_full_scale.json:54 |
| INT8 çıktı sürüklenmesi ortalama | 0.008917972445487976 | fraud/results/edge_eval_full_scale.json:55 |
| INT8 çıktı sürüklenmesi p99 | 0.14572913944721222 | fraud/results/edge_eval_full_scale.json:56 |
| ayrık test satır / fraud / taban oran | 4700000 / 6110 / 0.0013 | fraud/results/edge_eval_full_scale.json:59-61 |
| AUC FP32 / INT8-tensör / INT8-kanal | 0.9450745846153957 / 0.9364812440872093 / 0.9367113790742326 | fraud/results/edge_eval_full_scale.json:62-64 |
| kantizasyon AUC maliyeti | -0.008593340528186388 / -0.008363205541163099 | fraud/results/edge_eval_full_scale.json:65-66 |
| yuvarlanmış log biçimi | 0.9451 / 0.9365 / 0.9367 | fraud/logs/run_full_scale_edge_eval.log:52-54 |
| D / d_cloud / K (tabular) | 8 / 128 / 13 | fraud/results/full_scale_hybrid.json:56, 57, 61 |
| eğitim alt-küpe / fraud / taban oran | 100000 / 130 / 0.0013 | fraud/results/full_scale_hybrid.json:5-7 |
| AUC-ROC / hatırlama | 0.9351451502337654 / 0.7307692307692307 | fraud/results/full_scale_hybrid.json:10, 16 |
| eğitim süresi | 2852.3 s = 47,54 dakika | fraud/results/full_scale_hybrid.json:23 |
| karşılaştırma AUC (xgboost) | 0.9933922442255942 | fraud/results/duels.json:4 (etiket: satır 3) |
| karşılaştırma AUC (lightgbm) | 0.9364403244701196 | fraud/results/duels.json:26 (etiket: satır 25) |
| karşılaştırma AUC (ftt) | 0.9694542323758608 | fraud/results/duels.json:50 (etiket: satır 49) |
| tam ölçek test satırı | 4716078 | fraud/results/duels.json:113 |

## S2 — Çift hesabı maliyeti (k-ölçekleme)

| Değer | ham | künye |
|---|---|---|
| k=13 → çarp-topla/işlem | 2522200 | fraud/results/edge_eval_full_scale.json:29-30 |
| k=50 → çarp-topla/işlem | 34091488 | fraud/results/edge_eval_full_scale.json:41-42 |
| ölçülen MAC oranı | 13.5 (34091488 / 2522200 = 13.5166) | hesap: yukarıdaki iki satır |
| nokta oranının karesi | 14.8 — ham hesap (50/13)² = 14.7929; tarifname 14.8'e yuvarlıyor (`docs/patent/pct_v2/DESCRIPTION_EN.md:123`) | hesap: yukarıdaki iki satır |

## S3 — Dağıtım paketi (manifest)

| Değer | ham | künye |
|---|---|---|
| kantize çıkarım modeli (dağıtım paketi) | 548.4 KB | fraud/results/deploy/manifest.json:18 |
| nicelleştirme tablosu | 5.4 KB | fraud/results/deploy/manifest.json:21 |
| çözümleme tablosu | 16.5 KB | fraud/results/deploy/manifest.json:23 |
| çözümleme girdisi | 221 | fraud/results/deploy/manifest.json:24 |
| öz-test satır sayısı | 19963 | fraud/results/deploy/manifest.json:27 |
| öz-test en büyük mutlak fark | 1.2814998626708984e-06 | fraud/results/deploy/manifest.json:28 |
| öz-test p99 mutlak fark | 8.401274840252881e-07 | fraud/results/deploy/manifest.json:29 |
| öz-test verimi | 26473.303851652894 işlem/s | fraud/results/deploy/manifest.json:30 |
| eşleme tanımları (satır→kimlik, kimlik→etiket) | quantizer.json / token_decoder.json | fraud/results/deploy/manifest.json:97-98 |
| model bloğu D/d_model/d_cloud/vocab/points/K | 8 / 32 / 128 / 221 / 214 / 13 | fraud/results/deploy/manifest.json:9-14 |
| ONNX dosyası (değerlendirme ürünü) | 543.5 KB | fraud/results/edge_eval_full_scale.json:48 |
| ONNX en büyük mutlak fark | 1.0058283805847168e-07 | fraud/results/edge_eval_full_scale.json:49 |

## S4 — Açıklama kararlılığı (beş tohum)

> Not: `aml/results/e5_stability.json` 39.663 satırdır; `:34057` künyesi geçerlidir
> (kontrol `sed -n '34057p'` ile yapıldı; tek satır okuma araçları satırı 2000 karakterde kırpıyor).

| Değer | ham | künye |
|---|---|---|
| tohumlar | 20261016…20261020 (5 adet) | aml/results/e5_stability.json:8-12 |
| test alt-örneği | 400 | aml/results/e5_stability.json:14 |
| atıf aracı sürümü | TreeExplainer, kurulu 0.54.1 | aml/results/e5_stability.json:25 |
| bölge etiketi kararlılığı | 0.325 | aml/results/e5_stability.json:29 |
| kararlılık tanımı | "fraction of test nodes whose nearest name matches the reference-seed name across all retrains" | aml/results/e5_stability.json:30 |
| oturtma+kodlama süresi (tohum başına) | 0.35445423298515377 s ≈ 0.3545 s | aml/results/e5_stability.json:38 |
| atıf birinci-sıra kararlılığı | 0.1025 | aml/results/e5_stability.json:41 |
| ilk-üç küme jaccard | 0.43405 | aml/results/e5_stability.json:42 |
| atıf Spearman | 0.6510458604357529 | aml/results/e5_stability.json:43 |
| model-sonrası atıf yolu süresi (tohum başına) | 0.7402620668290183 s ≈ 0.7403 s (`mean_gbm_treeexplainer_sec`) | aml/results/e5_stability.json:34057 |

## S5 — Makine sağlığı (C-MAPSS)

| Değer | ham | künye |
|---|---|---|
| gömme boyutu | 48 | maintenance/results/e1_cmapss.json:18 |
| ayrık test n / en-yakın-prototip doğruluğu | 1064 / 0.8167293233082706 | maintenance/results/e1_cmapss.json:23-24 |
| kritik kademe n / doğruluk | 25 / 0.64 | maintenance/results/e1_cmapss.json:27-28 |
| ara kademe n / doğruluk | 114 / 0.18421052631578946 | maintenance/results/e1_cmapss.json:43-44 |
| sağlıklı kademe n / doğruluk | 853 / 0.9378663540445487 | maintenance/results/e1_cmapss.json:51-52 |
| eşik çeyreği / mesafe | 0.95 / 8.460468292236328 | maintenance/results/e1_cmapss.json:186-187 |
| ayrıktaki sağlıklı kayıt | 853 | maintenance/results/e1_cmapss.json:189 |
| alarm oranı değişmemiş / kaymış / oran | 0.011723329425556858 / 0.02813599062133646 / 2.4 | maintenance/results/e1_cmapss.json:190-192 |
| uygulanan kayma vektörü | [6.0, 6.0, -6.0] | maintenance/src/e1_real.py:293 |

## S6 — Bölge isimlendirme ve bölge kayması (şikâyet korpusu)

| Değer | ham | künye |
|---|---|---|
| kümeleme algoritması | KMeans (n_clusters, n_init=10) | complaints/src/explain_region.py:47 |
| bölge sayısı | 8 | complaints/results/region_map.json:2 |
| etiket üretiminde kullanılan sembol sayısı | ilk 8 (`ranked_tokens[:8]`) | complaints/src/explain_region.py:66 |
| ölçülen etiket (bölge 2 ve bölge 4, AYNI metin) | "Ġcredit Ġreport Ġinformation" | complaints/results/region_map.json:422, 838 (bölge anahtarları: 421, 837) |
| ölçülen etiket (bölge 7) | "Ġcredit Ġreport Ġconsumer" | complaints/results/region_map.json:1462 |
| tam değişim ölçütü tanımı | `0.5 * np.abs(shares - previous).sum()` | complaints/src/audit_drift.py:36 |
| KL ölçütü tanımı | `np.sum(p * np.log(p / q))` | complaints/src/audit_drift.py:40 |
| yeni bölge işareti tanımı | alt/üst eşik koşulu | complaints/src/audit_drift.py:44 |
| yöntem: sürükleme etiketi yok | "fixed train-window KMeans prototypes; no drift labels" | complaints/results/drift_report.json:2 |
| hafta 1 etiketi (metinde geçen tarih aralığı) | "2025-02-03/2025-02-09" | complaints/results/drift_report.json:21 |
| hafta 2 etiketi (metinde geçen tarih aralığı) | "2025-02-10/2025-02-16" | complaints/results/drift_report.json:37 |
| hafta 1 TV / KL | 0.0903703703703704 / 0.0341397185589417 | complaints/results/drift_report.json:33-34 (hafta etiketi: 21) |
| hafta 2 TV / KL | 0.02469135802469135 / 0.0018965315544816688 | complaints/results/drift_report.json:49-50 (hafta etiketi: 37) |
| alt-örnek n / eğitim / sürüklenme | 20000 / 14000 / 6000 | complaints/results/complaints_scale_stability.json:2-4 |
| ARI ortalaması | 0.9938922500989028 | complaints/results/complaints_scale_stability.json:30 |
| bölge payı sıra korelasyonu ortalaması | 1.0 | complaints/results/complaints_scale_stability.json:32 |
| pay farkı bloğu | share_delta_drift_minus_train | complaints/results/complaints_scale_stability.json:64 |
| pay farkı ucu (bölge 1) | -0.17 | complaints/results/complaints_scale_stability.json:66 |
| pay farkı ucu (bölge 7) | +0.2068 | complaints/results/complaints_scale_stability.json:72 |
| sınıf prototipi tanımı (yalnız eğitim) | eğitim ortalaması | aml/src/e1_explain.py:297 |
| grafik test n / usulsüz n | 400 / 40 | aml/results/e2_explain.json:14-15 |
| en-yakın-prototip ayrık doğruluk | 0.75 | aml/results/e2_explain.json:16 |
| sınıf bazında doğruluk | 0.7833333333333333 / 0.45 | aml/results/e2_explain.json:18-19 |
| grafik sınırı | transductive; işlem düğümü | aml/results/e2_explain.json:11 |

## S7 — Nokta çıkarma etkisi

| Değer / tanım | ham | künye |
|---|---|---|
| pilot n / fraud / normal / taban oran | 827 / 27 / 800 / 0.032648125755743655 | fraud/results/explain_pilot.json:6-9 |
| uyarı: tabakalı geliştirme örnek | "not the natural 4.7M held-out population" | fraud/results/explain_pilot.json:10 |
| katkı tanımı (logit farkı) | "base logit - logit with point j masked" | fraud/results/explain_pilot.json:21 |
| işaret anlamı | pozitif = dolandırılma tahminini destekler | fraud/results/explain_pilot.json:22 |
| toplamsal değil uyarısı | "ablation sensitivity, not SHAP" | fraud/results/explain_pilot.json:23 |
| sıralama kuralı | `mean_abs_logit_contrib * sqrt(count)` | fraud/results/explain_pilot.json:24 |
| pilot AUC-ROC | 0.8286111111111112 | fraud/results/explain_pilot.json:32 |
| uygulama sırası (bulut→taban→sıfırla→maske→yeniden→fark) | — | fraud/src/explain_pilot.py:102-111 |
| kantize yolda toplu girdi / maske / tek oturum / fark | — | fraud/pilot/infer.py:97-98, 100, 104, 107 |

## S8 — Dil görevi, boyut seçimi, çok cümleli hiyerarşi

| Değer | ham | künye |
|---|---|---|
| held-out PPL D=32 / D=64 | 61.94 / 61.19 | logs/run_r4_resume.log:67, 70 |
| fark / kazanan / protokol | 0.76 / D64 / n=2000, seed=0, rows [200000,202000), seq_len=10 | logs/run_r4_resume.log:71 |
| karşılaştırılan PPL (CloudLM / nedensel) | 6.45 / 2.94 | logs/run_compare.log:17-18 |
| karşılaştırılan parametre sayıları | 20999648 / 22724768 — ham log biçimi ayraçlı: "20,999,648 params" / "22,724,768 params" | logs/run_compare.log:13-14 |
| bölme oranı ve tohum | 0.8 / 0.1 / 0.1, seed 42 | contextcloud/task_utils.py:17 |
| bölme sayıları | 159888 / 19986 / 19986 | logs/run_r5_cc_stable.log:2 |
| sözcük dağarcığı / parametre | 10000 / 12260784 | logs/run_r5_cc_stable.log:6-7 |
| en iyi doğrulama yitimi / TEST yitimi | 3.2305 / 3.2318 (çapraz-entropi) | logs/run_r5_cc_stable.log:1235-1236 |
| cümle-bulut / pencere kodlayıcı sınıfları | SentenceBulutEncoder / WindowBulutEncoder | contextcloud/model.py:20, 121 |
| pencere boyutu / adım | 5 / 2 | contextcloud/model.py:127-128 |
| pencere-geometri kodlayıcı / LM | WindowGeometryEncoder / ContextCloudLM | contextcloud/model.py:196, 246 |

## S9 — Mimari tanım künyeleri

| Tanım | künye |
|---|---|
| sıralama yok; maskeyle padding çiftleri sıfırlanır | cloudlm/model.py:47 |
| çift girdi genişliği = 3D + 2 (diff D, dist 1, cos 1, norm D, norm D) | cloudlm/model.py:63-64 |
| mesafe = fark vektörünün normu | cloudlm/model.py:158 |
| kosinüs paydası (norm çarpımı) | cloudlm/model.py:161 |
| iki norm bileşenin genişletilmesi ve birleştirme | cloudlm/model.py:165, 168 |
| nokta maskeleme + maksimum havuzlama + maskeyle temizleme | cloudlm/model.py:151-155 |
| çift maskesi, sıfırlama, geçerli çift sayısına göre normalizasyon | cloudlm/model.py:171-174 |
| çiftler üzerinden ortalama havuzlama | cloudlm/model.py:177 |
| birleştirme katmanı (concat + fusion) | cloudlm/model.py:179 |
| ilerleyiş geçişi maskeyle tanımlı | cloudlm/model.py:305 |
| bulut durumu bulut kodlayıcısından | cloudlm/model.py:313 |
| küresel havuzlama (max, min, mean, std) O(N) | purecloudlm/model.py:57, 62 |
| bulut merkezi + varyans O(N) | purecloudlm/model.py:63 |
| çift hesabı YOK/yaklaşık; yalnız küme istatistikleri | purecloudlm/model.py:8-9 |

## S10 — Gradyanz güncelleme (nokta konumları)

| Değer / tanım | künye |
|---|---|
| aday kabul koşulu (`err < base_err - cfg.eps`) | quann_cloud/src/qcloud_quann.py:212 |
| geri alma (`p[fi] = old`) | quann_cloud/src/qcloud_quann.py:220 |
| arama aralığının daraltılması | quann_cloud/src/qcloud_quann.py:223, 226 |
| ölçek sınırı: 0. mertebe aday başına 1 ilerleyiş; 21M modelde uygulanabilir değil | quann_cloud/FINDINGS.md:22 |
| küçük ölçek mimari (SlimCloudLM, 111.248 param) | quann_cloud/FINDINGS.md:15, 17 |
| veri: 6000 eğitim + 500 değerlendirme, seq_len 8; BPE 1024 | quann_cloud/FINDINGS.md:18-19 |
| held-out PPL: eğitilmemiş / geri yayılım / 0. mertebe | 1045.03 / 563.45 / 1016.30 — quann_cloud/FINDINGS.md:28, 29, 30 |

## S11 — Dürüst sınır kayıtları

| Sınır | künye |
|---|---|
| INT8 AUC maliyeti kabul edilmiş sınırdır | fraud/results/edge_eval_full_scale.json:65-66 |
| sayılar tek yapılandırma/korpus kombinasyonunda ölçüldü | docs/patent/final_v3/TEKNIK_ETKI.md:40-42 |
| doğruluk üstünlüğü yok (tabular) | fraud/results/duels.json:4, 50 + docs/patent/final_v3/TARIFNAME.md:154-155 |
| doğruluk üstünlüğü yok (dil) | logs/run_compare.log:17-18 + docs/patent/final_v3/TARIFNAME.md:171 |
| yeni bölge olayı ölçülen sürüklenmede kaydedilmedi | docs/patent/final_v3/TARIFNAME.md:227 |
| yalnız Şekil 6 ve Şekil 7 ölçüm taşır | docs/patent/final_v3/TARIFNAME.md:54 |

## D1 — final_v3 TARIFNAME'de bulunan, DÜZELTİLMİŞ künyeler (pct_v2 doğru değeri kullanır)

Aşağıdaki beş sayı `docs/patent/final_v3/TARIFNAME.md` içinde künye ile birlikte yazılıdır; künye
edilen satırlar bu oturumda okundu ve değerler ham kaynakta **bulunamadı**. pct_v2 metinleri ham
kaynağın doğru değerini kullanır; `final_v3` kural gereği değiştirilmedi.

| final_v3 iddiası (satır) | künye edilen satırda gerçek | pct_v2'de kullanılan ham değer |
|---|---|---|
| "öz-testte … en büyük mutlak fark 1.4603e-06" (docs/patent/final_v3/TARIFNAME.md:37, 135) | fraud/results/deploy/manifest.json:28 = 1.2814998626708984e-06 | 1.2814998626708984e-06 (fraud/results/deploy/manifest.json:28) |
| "p99 mutlak fark 8.8483e-07" (docs/patent/final_v3/TARIFNAME.md:135) | fraud/results/deploy/manifest.json:29 = 8.401274840252881e-07 | 8.401274840252881e-07 (fraud/results/deploy/manifest.json:29) |
| "öz-testte … 54063 işlem/saniye" (docs/patent/final_v3/TARIFNAME.md:37, 135) | fraud/results/deploy/manifest.json:30 = 26473.303851652894 | 26473.303851652894 (fraud/results/deploy/manifest.json:30) |
| "kantize ONNX modeli 543.5 KB" dağıtım paketi olarak (docs/patent/final_v3/TARIFNAME.md:133) | fraud/results/deploy/manifest.json:18 = 548.4 KB (543.5 KB → fraud/results/edge_eval_full_scale.json:48, ayrı ürün) | 548.4 KB (fraud/results/deploy/manifest.json:18); 543.5 KB yalnız değerlendirme ürünü olarak |
| "bölge payı farkları −0.170 … +0.207" (docs/patent/final_v3/TEKNIK_ETKI.md:22) | complaints/results/complaints_scale_stability.json:66, 72 = -0.17 ve +0.2068 (yuvarlama farkı) | -0.17 / +0.2068 |

Doğrulanan ve **sorunsuz** künyeler (şüphe edilip doğrulananlar; metindeki değer yuvarlanmış,
ham değer ortada):

| Metindeki değer | Ham değer | Kaynak satırı |
|---|---|---|
| 0.7403 | 0.7402620668290183 | `aml/results/e5_stability.json:34057` |
| 0.3545 | 0.35445423298515377 | `aml/results/e5_stability.json:38` |
| 8.4605 | 8.460468292236328 | `maintenance/results/e1_cmapss.json:187` |
| 0.2068 | 0.2068 | `complaints/results/complaints_scale_stability.json:72` |
