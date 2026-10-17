# RESEARCH.md — Quann-3D Tek Durum Kaynağı

> **Kurallar:** Her oturum BAŞINDA okunur. Bu dosyanın dışında ağır içerik
> (logs/, checkpoints/, kod) ana oturuma gelmez — domain agent'ları (`experiment`,
> `geometry`, `paper`) okur, distile sonuç döndürür. Güncelleme protokolü:
> `AGENTS.md` §5. Boyut bütçesi: **≤150 satır** — taşarsa en eski deney "Son
> Deneyler" tablosundan çıkarılır (detay `logs/`'da kalır).
> Son güncelleme: 2026-10-17 09:27 (E3′ zaman-disjoint doğrulama tamamlandı; 1.5M post-train kuyruğu aktif)

## Bir Cümle

Transformer'ın "sıralı dikkat + tek vektör" paradigmasının yerine: kelime → ND
uzayda nokta, cümle → nokta bulutu, anlam → bulut geometrisi; tahmin bu
konfigürasyondan. İkinci katman QUANN-Cloud: aynı mimariyi backprop'suz (0th-order)
eğiten yöntem.

## Canlı Runlar (2026-10-17 09:18)

| Run | Komut | Durum | Log |
|---|---|---|---|
| CloudLM 1.5M (ölçek) | `train_wiki.py --corpus turkish_wiki_1500k --d-dim 32 --epochs 20 --resume` | canlı (PID 56169), epoch 16/20; son train PPL 75.78 | `logs/run_final_pipeline.log` (epoch/PPL); `logs/run_wiki1500k_resume_nohup.log` (yalnız başlık) |

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
| 10-17 | aml E3′ Elliptic (46,564 labeled, timestep-disjoint 60/10/30) | test hybrid .5865/.0972, fraud@1%FPR .018; GBT .8764/.5606, .481; nearest-prototype .617 (illicit .482) | bağımsız zaman testi E2/E2′ hybrid sonucunu doğrulamadı; GBT farkı sürüyor, prototip gerekçesi karar kanıtı değil | `aml/logs/run_e3_prime.log`; `aml/results/e3_prime.json` |
| 10-17 | fraud matched duels + standard LGBM refresh | QUANN-split matched (1.02M/100K/4.7M): XGB .8047/.9938/.4884, LGBM .7665/.9899/.4455, FTT .6866/.9831/.4897 (test PR/ROC/F2); QUANN strong-reader .462 PR. Standard LGBM .6387 PR/.9364 ROC | matched kontroller QUANN strong-reader PR'ını geçti; LGBM düzeltmesi n_jobs=1, spw=1.0 ile tamamlandı (SIGSEGV/imbalance kaynaklı hatalar giderildi) | `fraud/logs/run_duels_matched.log`; `fraud/results/duels_quann.json`; `fraud/results/duels.json` |
| 10-17 | aml/E2 Elliptic + E2′ (darknet gerçek veri; K10/B8 çift-kanal açı çifti) | aynı config: hybrid .838/.370, fraud@1%FPR .025 vs GBT .943/.758, .625; temel E2 hybrid .542/.140, .050 vs GBT .907/.795, .625; E2 explain (36D base hybrid) held-out nearest-prototype .750 (illicit .450) | gerçek-veri kapısı tamam ✓; en iyi açı varyantı temele göre kaldıraç sağladı ama GBT açığı sürüyor; 2K sampled tx-node proxy, hesap/ham tutar yok ve graf özellikleri transductive — prototip açıklaması karar kanıtı değil | `aml/results/e2p_design.json`; `aml/logs/run_e2p_smoke.log`; `aml/results/e2_explain.json` |
| 10-16 | maintenance E5 (3-sınıf arıza-tipi: nokta tasarımı gridi — açı-çifti/bucket/K, aynı kayıt-disjoint split) | en iyi hybrid (mixed_angle_B8_K1) .678/.530 (inner .000); en yüksek inner (.889) konfiginde GBT +30.5 ROC önde; açı kaldırağı +9.3 pt, B8 +7.5 pt (Elliptic E2′ ile aynı desen); hybrid–GBT en küçük gap 11.7 pt (≤10 hedefi tutmadı) | 3-sınıf tip görevi bu veride KAPANDI; açı kaldırağı evrensel — **maintenance vaka = ikili tespit + açıklanabilirlik + edge** (ana oturum kararı) | `maintenance/logs/run_e5_smoke.log` |

## Sıradaki Aksiyonlar

1. CloudLM 1.5M sonrası kuyruğu aktif (PID 56169); bitince held-out PPL + R1/R2 değerlendirmesi otomatik log'a düşer.
2. 1.5M compare PPL'ini yorumlamadan önce CloudLM 1.5M ile baseline BPE eşleşmesini doğrula; ardından held-out sonucu/R1-R2 değerlendirip QUANN fraud zengin nokta run'ını (D 16/32, K↑, orta ölçekli train) planla.
3. ✅ E3′: tüm 46,564 labeled tx-node proxy CPU'da, zaman-disjoint 60/10/30 doğrulandı; sonuç E2/E2′ hybrid'i doğrulamadı. E2 figürleri `aml/figures/e2_elliptic_{region_map,naming_comparison}.png`; AML hesap/amount iddiası yok.
4. Maintenance vaka ikili tespit + açıklanabilirlik + edge; 3-sınıf tip görevi kapalı, E4 adaptasyon iddiası kanıtlanmadı.
5. `aml/`, `maintenance/`, `fraud/` ve `docs/` dosyaları commit'te değil; commit kararı kullanıcıda (committer).
6. ✅ Makale §7 Conclusion tutarlılık: uygulama paragrafı + dürüst sınırlar + "dikey-bağımsız çekirdek" kapanışı eklendi; §6.5/§6.6 değerleri birincil kaynakla doğrulandı (CWRU F2 .996, C-MAPSS %81.7). Abstract/Intro §6.5-§6.6'ya dokunulmadı (15ep/PPL 57/350×, §4.4 ile tutarlı).

## Subproje: quann_cloud/

Kendi durum dosyaları var — ana oturum bunları okumaz, `geometry`/`experiment`
agent'ı okur: `quann_cloud/REVISION.md` (teşhis+adaylar), `FINDINGS.md`
(dürüst bulgular), `EXPERIMENTS.md`, `PROJECT_SUMMARY.md`.
Tek cümle: QUANN 0th-order XOR'da %100, gerçek LLM'de random'a yakın → savunulabilir
iddia "küçük/yorumlanabilir modeller için backprop'suz eğitim".

## Subproje: fraud/

IBM kredi kartı fraud verisi (`~/Downloads/archive`): tam 4.7M testte standart tavan RF .973/.693, GBT .981/.598 (ROC/PR). QUANN full: hybrid .9581/.258, lift 36×; pure 0th .4502, backprop .8938. Strong-reader (b): sabit 214 yakınsamış D=8 nokta, backprop reader 64/128/256 → PR .383/.411/.462 (ROC .969–.975); reader kapasitesi kaldıraç, D=8 geometri tavanı. Edge: 191 KB fp32 / 53 KB int8, ~75k tx/s; doğal fraud oranında INT8 ΔROC −.014. Detay `fraud/README.md` + `fraud/results/`.
Matched duels tamamlandı (train/val/test 1.02M/100K/4.7M; fraud 20334/2000/6181): QUANN strong-reader PR .462 karşısında XGB .8047/.9938/.4884, LGBM .7665/.9899/.4455, FTT .6866/.9831/.4897 (PR/ROC/F2). Standard LGBM refresh test PR .6387, ROC .9364. `fraud/src/train_duels.py` düzeltmesi `n_jobs=1`, `spw=1.0`; tanı kanıtı `fraud/results/duels_quann_lgbm_fixed_diag.json` (matched LGBM PR .7665/ROC .9899); duels tamam, D=8 QUANN farkını zengin nokta varyantıyla sınamak sıradaki adım.

> ⚠️ **Bütünlük (10-16, edge-eval):** Yukarıdaki .9581 (ve .9745) checkpoint'ten
> **reproduce edilemiyor** — input `FeatureQuantizer` (bin/cat/offset) asla
> checkpoint'e kaydedilmemişti; yeniden fit farklı token-id → AUC kayıyor. Tam
> 4.7M doğal testte re-fit quantizer **FP32 ≈0.77**, INT8(channel) ≈0.75. Fix
> (bundled quantizer + scale'li ad + `quann_hybrid_full.pt` pin) + repro kanıtı
> (bundled→bit-identik token/AUC; re-fit→kayma): `fraud/REVISION.md` +
> `fraud/results/MANIFEST.md`. Gerçek edge AUC'i için taze `--scale full` hybrid
> (quantizer'lı) + `edge_eval --with-auc` şart — ağır CPU, 1.5M bitince.

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
