# RESEARCH.md — Quann-3D Tek Durum Kaynağı

> **Kurallar:** Her oturum BAŞINDA okunur. Bu dosyanın dışında ağır içerik
> (logs/, checkpoints/, kod) ana oturuma gelmez — domain agent'ları (`experiment`,
> `geometry`, `paper`) okur, distile sonuç döndürür. Güncelleme protokolü:
> `AGENTS.md` §5. Boyut bütçesi: **≤150 satır** — taşarsa en eski deney "Son
> Deneyler" tablosundan çıkarılır (detay `logs/`'da kalır).
> Son güncelleme: 2026-10-17 (E2 figürleri → makale §6.5 gerçek veri; önceki 00:21: matched fraud duels LightGBM SIGSEGV teşhisi)

## Bir Cümle

Transformer'ın "sıralı dikkat + tek vektör" paradigmasının yerine: kelime → ND
uzayda nokta, cümle → nokta bulutu, anlam → bulut geometrisi; tahmin bu
konfigürasyondan. İkinci katman QUANN-Cloud: aynı mimariyi backprop'suz (0th-order)
eğiten yöntem.

## Canlı Runlar (2026-10-17 00:21)

| Run | Komut | Durum | Log |
|---|---|---|---|
| CloudLM 1.5M (ölçek) | `train_wiki.py --corpus turkish_wiki_1500k --d-dim 32 --epochs 20 --resume` | canlı, epoch 12/20; son train PPL 91.00 | `logs/run_final_pipeline.log` (epoch/PPL); `logs/run_wiki1500k_resume_nohup.log` (yalnız başlık) |
| fraud duels (kontrollü retry) | `train_duels.py --split quann --models xgboost,lightgbm,ftt --n-jobs 18` | canlı, CPU; kontrollü eş-ölçek retry | `fraud/logs/run_duels.log` |
| fraud duels MATCHED (1.02M) | `train_duels.py --split quann --models xgboost,lightgbm,ftt --n-jobs 18` | blocked; LightGBM full QUANN split'te native SIGSEGV, aktif PID yok; XGBoost tamamlandı | `fraud/logs/run_duels_matched.log` |

**Teşhis:** Epoch/PPL aslında kaydedilmiş; resume çıktısı `run_final_pipeline.log`'a yönlenmiş, nohup logunda yalnızca başlatma başlığı var. RSS artışı ~4.5 GB/saat; swap baskısı yüksek. `train_wiki.py` her epoch'ta tüm token pencerelerinden dev `examples` listesi kurup shuffle ediyor; RSS artışının muhtemel nedeni bu geçici tahsislerin allocator'da tutulması (kanıtlanmış kalıcı Python referans sızıntısı değil).

Doğrula: `ps aux | grep -E "train_wiki|train_baseline" | grep -v grep`

## Mevcut Best (ölçülmüş, ortak held-out 2000 cümle)

| Model | Config | Param | Held-out PPL | Checkpoint |
|---|---|---|---|---|
| CloudLM | D=32, 200K, 50ep | 21.0M | **6.45** | `checkpoints/wiki_checkpoints_200k/wiki_cloud_best.pt` |
| Causal baseline | d=384, 4L, 200K, 50ep | 22.7M | **2.92** | `checkpoints/baseline_200k/baseline_best.pt` |

**Kök teşhis (sensitivity_analysis, ölçülmüş):** kapasite (D boyutu) —
hassasiyet değil. Bulut (10×32=320 sayı), transformer gizli durumunun
(10×256=2560) %12.5'ini taşıyor. Detay: `quann_cloud/REVISION.md`.

## Aktif Hipotezler (R1–R4 — `quann_cloud/REVISION.md`)

| # | Hipotez | Öncelik | Durum |
|---|---|---|---|
| R1 | Kapasite: D=64 / D=128 | 1 | **sonuçlandı/değerlendirme** — 20K held-out: baseline 10.80, D=32 201.18, D=64 181.47, D=128 189.47. D=64 iyileşti, D=128 geri gitti; baseline farkı çok büyük, 200K'a ölçekleme yok. |
| R2 | Cloud attention (ilişkisel sözlük genişletme) | 3 | bekliyor |
| R3 | Attention-weighted pooling (max-pool onarımı) | 2 | bekliyor |
| R4 | QUANN: aramayı pair-kanalına yönlendir (freeze varyantı) | 4 | bekliyor |

## Son Deneyler (en yeni üstte)

| Tarih | Run | Sonuç | Ders | Log |
|---|---|---|---|---|
| 10-17 | aml/E2 Elliptic + E2′ (darknet gerçek veri; K10/B8 çift-kanal açı çifti) | aynı config: hybrid .838/.370, fraud@1%FPR .025 vs GBT .943/.758, .625; temel E2 hybrid .542/.140, .050 vs GBT .907/.795, .625; E2 explain (36D base hybrid) held-out nearest-prototype .750 (illicit .450) | gerçek-veri kapısı tamam ✓; en iyi açı varyantı temele göre kaldıraç sağladı ama GBT açığı sürüyor; 2K sampled tx-node proxy, hesap/ham tutar yok ve graf özellikleri transductive — prototip açıklaması karar kanıtı değil | `aml/results/e2p_design.json`; `aml/logs/run_e2p_smoke.log`; `aml/results/e2_explain.json` |
| 10-16 | maintenance E5 (3-sınıf arıza-tipi: nokta tasarımı gridi — açı-çifti/bucket/K, aynı kayıt-disjoint split) | en iyi hybrid (mixed_angle_B8_K1) .678/.530 (inner .000); en yüksek inner (.889) konfiginde GBT +30.5 ROC önde; açı kaldırağı +9.3 pt, B8 +7.5 pt (Elliptic E2′ ile aynı desen); hybrid–GBT en küçük gap 11.7 pt (≤10 hedefi tutmadı) | 3-sınıf tip görevi bu veride KAPANDI; açı kaldırağı evrensel — **maintenance vaka = ikili tespit + açıklanabilirlik + edge** (ana oturum kararı) | `maintenance/logs/run_e5_smoke.log` |
| 10-16 | maintenance CWRU genişletme (80 kayıt resmi kaynak, kayıt-disjoint 60/20) + E4 saha-adaptasyon | ikili: hybrid .999/1.000 vs GBT 1.000 (12/6 yakınsadı) — GERÇEK rulman verisinde tavana ≈; 3-sınıf tip: hybrid .572/.406 (inner .000) vs GBT .907/.838 — yetersiz; E4 (RPM A→B): frozen/0th/backprop hepsi 1.000 → kazanç 0, retrain'siz üstünlük KANITLANAMADI (demo dejener: frozen zaten %100) | ikili arıza tespiti (kayıt-disjoint) güçlü ✓; tip ayrımı + E4 adaptasyon iddiası açık borç | `maintenance/logs/run_cwru_exp.log` |
| 10-16 | aml/E4 (düzenleyici değişikliği 10K→5K; haftalık adaptasyon, sentetik) | frozen 5K-recall hafta 1-4: 0.00; 0th-order adaptasyon: 5K-recall 0.00 — backprop-retrain (1.00) kazanımının **%0'ı** (macro AUC .871/.879/1.000); stabilite: eski 10K 1.00'da kalıyor (normal −.065, velocity −.060); 5K kümesi haritada ayrışmıyor, en yakın prototip normal | 0th-order adaptasyon görülmemiş şekilde BAŞARISIZ (quann_cloud bulgusuyla tutarlı); AML "haftalık uyum" iddiası kanıtlanamadı → AML = yalnız açıklanabilirlik wedge'i | `aml/logs/run_e4_smoke.log` |

## Sıradaki Aksiyonlar

1. CloudLM 1.5M canlı run'ını izlemeye devam et (MPS'e dokunma); bitince held-out PPL + R1/R2 değerlendirmesi.
2. CPU fraud-duels kontrollü retry tamamlanınca AUC-PR head-to-head ve eş-ölçek kıyasını güncelle; sonra QUANN zengin nokta adımı (D 8→16/32, K↑).
3. E2 figürleri üretildi (`aml/figures/e2_elliptic_{region_map,naming_comparison}.png`) ve makale §6.5 gerçek veriye bağlandı. E3′ için daha büyük/bağımsız örnek ve zaman-disjoint doğrulama planla; AML hesap/amount iddiası yok.
4. `aml/`, `maintenance/`, `fraud/` ve `docs/` dosyaları commit'te değil; commit kararı kullanıcıda (committer).
5. İstek gelirse `paper` agent'ı §6.6 ile §7 Conclusion arasında tutarlılık geçişi yapsın.

## Subproje: quann_cloud/

Kendi durum dosyaları var — ana oturum bunları okumaz, `geometry`/`experiment`
agent'ı okur: `quann_cloud/REVISION.md` (teşhis+adaylar), `FINDINGS.md`
(dürüst bulgular), `EXPERIMENTS.md`, `PROJECT_SUMMARY.md`.
Tek cümle: QUANN 0th-order XOR'da %100, gerçek LLM'de random'a yakın → savunulabilir
iddia "küçük/yorumlanabilir modeller için backprop'suz eğitim".

## Subproje: fraud/

IBM kredi kartı fraud verisi (`~/Downloads/archive`): tam 4.7M testte standart tavan RF .973/.693, GBT .981/.598 (ROC/PR). QUANN full: hybrid .9581/.258, lift 36×; pure 0th .4502, backprop .8938. Strong-reader (b): sabit 214 yakınsamış D=8 nokta, backprop reader 64/128/256 → PR .383/.411/.462 (ROC .969–.975); reader kapasitesi kaldıraç, D=8 geometri tavanı. Edge: 191 KB fp32 / 53 KB int8, ~75k tx/s; doğal fraud oranında INT8 ΔROC −.014. Duels (a) XGB/LGBM/FT-Transformer vs QUANN AUC-PR karşılaştırması sürüyor. Detay `fraud/README.md` + `fraud/results/`.

## Market Öncelikleri

`docs/market_priorities.md`: Öncelik 3 dikeyleri (AML tipolojisi, işlem
fraud/edge, endüstriyel bakım) + 10 ikincil alan. Kazanma mantığı =
açıklanabilirlik + uyum hızı + boyut, ham doğruluk değil. Kanıt zemini:
`fraud/` (tam veride hybrid AUC-ROC .9581; 191 KB fp32 / 53 KB int8 edge).

## Subproje: aml/, maintenance/

2026-10-16 açıldı (ticari gerekçe `docs/market_priorities.md`): `aml/` =
sentetik E0/E1 sonrası Elliptic E2/E2′ gerçek-veri deneyi tamamlandı; en iyi
K10/B8 hybrid aynı-split GBT'den geride. E2 açıklanabilirlik probe'u
`aml/results/e2_explain.json`'da (held-out nearest-prototype %75, illicit %45);
ham tutar ve hesap kimliği yok, tek-hop proxy/transductive sınırlaması açık.
Gerçek-veri bölge haritası `viz` işi; E3′ bağımsız/doğrulama örneklemi bekliyor.
`maintenance/` = C-MAPSS/CWRU (pencere→bulut); alt-proje run'ları CPU (MPS: 1.5M).
Positioning: `aml/figures/typology_naming_{comparison,layering}.png` + `aml/figures/e2_elliptic_{region_map,naming_comparison}.png` (GBM/SHAP vs bölge-prototip; sentetik + gerçek) + makale `docs/paper_draft.md` §6.5 (isimlendirme boşluğu).

## Dondurulmuş Kurallar

`AGENTS.md` §3'e bak: `state_dict` ana adları, checkpoint adlandırması,
`logs/` append-only, `_archive/` yolu.
