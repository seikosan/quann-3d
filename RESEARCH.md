# RESEARCH.md — Quann-3D Tek Durum Kaynağı

> **Kurallar:** Her oturum BAŞINDA okunur. Bu dosyanın dışında ağır içerik
> (logs/, checkpoints/, kod) ana oturuma gelmez — domain agent'ları (`experiment`,
> `geometry`, `paper`) okur, distile sonuç döndürür. Güncelleme protokolü:
> `AGENTS.md` §5. Boyut bütçesi: **≤150 satır** — taşarsa en eski deney "Son
> Deneyler" tablosundan çıkarılır (detay `logs/`'da kalır).
> Son güncelleme: 2026-10-19 01:25

## Bir Cümle

Transformer'ın "sıralı dikkat + tek vektör" paradigmasının yerine: kelime → ND
uzayda nokta, cümle → nokta bulutu, anlam → bulut geometrisi; tahmin bu
konfigürasyondan. İkinci katman QUANN-Cloud: aynı mimariyi backprop'suz (0th-order)
eğiten yöntem.

## Canlı Runlar (2026-10-19 08:00)

Canlı run yok — wiki scale gate kuyruğu tamamlandı (sonuç "Son Deneyler"); MPS boş.

**Bitiş teşhisi (tarih):** 1.5M wiki run'ı epoch 17/20'de exit 137 ile OOM/kill oldu (RSS ~13.2 GB, free ~1.2 GB); epoch 18–20, bellek düzeltmesi sonrası kuyruktan resume edildi. Queue'daki step-4 eval, stdin çalıştırmasında `training/`'in sys.path'te olmaması ile bir kez FAILED (ModuleNotFoundError); script düzeltildi, eval retry ile tamamlandı.

## Mevcut Best (ölçülmüş, ortak held-out 2000 cümle)

| Model | Config | Param | Held-out PPL | Checkpoint |
|---|---|---|---|---|
| CloudLM | D=32, 200K, 50ep | 21.0M | **6.45** | `checkpoints/wiki_checkpoints_200k/wiki_cloud_best.pt` |
| Causal baseline | d=384, 4L, 200K, 50ep | 22.7M | **2.92** | `checkpoints/baseline_200k/baseline_best.pt` |

> 1.5M: 143.00 **ölçüm artefaktı** (BPE/dağıtım eşleşmemesi). Scale-gate final: tam wiki dosyasında 1.5M **62.15** @20ep vs 200K-wiki 218.84 (baskın OOD); aynı 200K kesitinde 1.5M 64.28 vs 200K **24.53** (train 32.01) → aynı dağıtımda tekrar derinliği (50ep, ~231M token) veri genişliğine (20ep, ~716M, 3.1×) üstün; 1.5M'in kazanımı tam konu-kapsamı. Best 200K 6.45 (cümle dağıtımı) korundu.

**Kök teşhis (sensitivity_analysis, ölçülmüş):** kapasite (D boyutu) —
hassasiyet değil. Bulut (10×32=320 sayı), transformer gizli durumunun
(10×256=2560) %12.5'ini taşıyor. Detay: `quann_cloud/REVISION.md`.

## Aktif Hipotezler (R1–R4 — `quann_cloud/REVISION.md`)

| # | Hipotez | Öncelik | Durum |
|---|---|---|---|
| R1 | Kapasite: D=64 / D=128 | 1 | **kapandı** — 20K: D=64 en iyi (181 vs 201); 1.5M karşılaştırması artefaktı elenince wiki scale-gate'te (aynı 200K dağıtımı) 50ep/200K 24.5 PPL, 20ep/1.5M 64.3 → bu bütçelerde kaldıraç tekrar derinliği. D×ölçek grid'i (D=64×200K-wiki) açık aday. |
| R2 | Cloud attention (ilişkisel sözlük genişletme) | 3 | bekliyor |
| R3 | Attention-weighted pooling (max-pool onarımı) | 2 | bekliyor |
| R4 | QUANN: aramayı pair-kanalına yönlendir (freeze varyantı) | 4 | bekliyor |

## Son Deneyler (en yeni üstte)

| Tarih | Run | Sonuç | Ders | Log |
|---|---|---|---|---|
| 2026-10-19 | aml E5 retraining stability + audit batch | 5-seed Elliptic pilot (2000 sampled labeled nodes, test_n=400): `geometry.name_stability` **0.325**, `mean_pca2_disp` **4.488**; SHAP: top-1 stability **0.1025**, top-3 Jaccard **0.4341**, Spearman **0.6510** | name label SHAP'tan ~3× daha stabil; 2D map/magnitudes stable değil → §6.5 name-scope | `aml/logs/run_e5_stability.log` |
| 10-18 | CloudLM wiki scale gate (1.5M resume 18–20 + 200K-wiki/50ep kontrol + eş dağıtım eval) | tam wiki dosyası: 1.5M **62.15** vs 200K 218.84; aynı 200K kesit: 1.5M 64.28 vs 200K **24.53** (train 32.01; maruziyet 716M vs 231M, 3.1×) | Aynı dağıtımda tekrar derinliği veri genişliğine üstün — D=32'de baskın kaldıraç 50 epoch tekrar; 1.5M'in kazancı tam konu-kapsamı (62 vs 219 OOD). Ölçek sorusu tek dağıtımda netleşti | `logs/run_wiki_scale_gate.log` |
| 10-18 | fraud full-scale hybrid + doğal-test edge eval (1ep; 1.02M train; 214 converged) | Dengeli 100K: PR .2673/ROC .9351/F2 .1564/lift 29.03×; doğal 4.7M: FP32 .9451, INT8-channel .9367 (Δ−.0084), tensor .9365 | Bundled quantizer ile AUC yeniden üretilebilir; eski .9581/.9745 iddiaları doğrulanmıyor ve FP32 standart RF .973/GBT .981 tavanlarının altında | `fraud/logs/run_full_scale_edge_eval.log` |
| 10-17 | CloudLM 1.5M wiki ölçek (D=32, epoch 1–17/20; exit 137 OOM) | 143.00 held-out PPL **cross-BPE/cross-distribution artefaktı** (wiki-BPE'li model, cümle corpus'u üzerinde); dağıtım-içi (wiki, kendi BPE'si): **66.26**; baseline wiki'de 191K (OOD) | 143 ≠ ölçek regresyonu: train 72.37 vs eval 66.26 → overfit yok; eğri düşüyor = undertraining; ölçek kararı aynı dağıtım karşılaştırmasıyla (resume + 200K-wiki kontrolü) | `logs/wiki1500k_wikeval.log`; `logs/run_final_pipeline.log`; `logs/monitor_wiki1500k.log` |
## Sıradaki Aksiyonlar

1. **P0 tamam:** doküman/makale bütünlüğü — paper §6.5 E5 name-scope; CWRU 3-class baseline/E5 açıklaması; ana README PPL 6.45; `aml/`, `maintenance/`, `quann_cloud/` README/REVISION senkronu.
2. **P1 kararı kullanıcıda:** AML named explanation, maintenance pilot, fraud edge veya akademik/LM deneyi.
3. **Aday (MPS boş):** **D=64 × 200K-wiki/50ep** — R1 kapasite × derinlik gate.
4. Commit kararı kullanıcıda; commit/push yapma.
5. İsteğe bağlı: fraud K/D grid; D=8 geometri tavanı hâlâ ayakta.

## Subproje: quann_cloud/

Kendi durum dosyaları var — ana oturum bunları okumaz, `geometry`/`experiment`
agent'ı okur: `quann_cloud/REVISION.md` (teşhis+adaylar), `FINDINGS.md`
(dürüst bulgular), `EXPERIMENTS.md`, `PROJECT_SUMMARY.md`.
Tek cümle: QUANN 0th-order XOR'da %100, gerçek LLM'de random'a yakın → savunulabilir
iddia "küçük/yorumlanabilir modeller için backprop'suz eğitim".

## Subproje: fraud/

IBM kredi kartı fraud verisi (`~/Downloads/archive`): standart tam-test tavanı RF .973/.693, GBT .981/.598 (ROC/PR). Taze `--scale full` hybrid dengeli 100K testte PR .2673/ROC .9351/F2 .1564/lift 29.03×; 4.7M doğal testte FP32 ROC .9451, INT8-channel .9367 (Δ−.0084). Strong-reader (b): sabit 214 yakınsamış D=8 nokta, backprop reader 64/128/256 → PR .383/.411/.462 (ROC .969–.975); reader kapasitesi kaldıraç, D=8 geometri tavanı. Sonuçlar: `fraud/results/full_scale_hybrid.json`, `fraud/results/edge_eval_full_scale.json`; checkpoint: `fraud/results/quann_hybrid_full.pt`.
Matched duels tamamlandı (train/val/test 1.02M/100K/4.7M; fraud 20334/2000/6181): QUANN strong-reader PR .462 karşısında XGB .8047/.9938/.4884, LGBM .7665/.9899/.4455, FTT .6866/.9831/.4897 (PR/ROC/F2). Standard LGBM refresh test PR .6387, ROC .9364. `fraud/src/train_duels.py` düzeltmesi `n_jobs=1`, `spw=1.0`; tanı kanıtı `fraud/results/duels_quann_lgbm_fixed_diag.json` (matched LGBM PR .7665/ROC .9899); duels tamam, D=8 QUANN farkını zengin nokta varyantıyla sınamak sıradaki adım.
Rich-points (K=20) tamamlandı: D=16 PR .1919/ROC .9770/F2 .2524/lift 50.92×; D=32 PR .4261/ROC .9792/F2 .2694/lift 55.54×. D=32 D=8 strong-reader PR .462'ye yaklaşsa da aşmadı, D=16 daha düşük; böylece D=8 geometri tavanı iddiası büyük ölçüde ayakta (matched XGB PR .8047/ROC .9938 ile fark sürüyor). 0th-order hız D=16'da 108.6, D=32'de 88.8 rows/s.

> **Bütünlük — doğrulandı:** Taze `--scale full` hybrid'in bundled quantizer'ı ile
> tam doğal testte FP32 AUC-ROC .9451 / INT8-channel .9367 (Δ−.0084); checkpoint
> ile tokenization ve AUC yeniden üretilebilir. Eski .9581/.9745 yerini bu değerlere
> bırakıyor; ikisi de standart RF .973 / GBT .981 tavanlarının altında.
> Kanıt: `fraud/results/full_scale_hybrid.json`,
> `fraud/results/edge_eval_full_scale.json`, checkpoint
> `fraud/results/quann_hybrid_full.pt`; log `fraud/logs/run_full_scale_edge_eval.log`.

## Market Öncelikleri

`docs/market_priorities.md`: Öncelik 3 dikeyleri (AML tipolojisi, işlem
fraud/edge, endüstriyel bakım) + 10 ikincil alan. Kazanma mantığı =
açıklanabilirlik + uyum hızı + boyut, ham doğruluk değil. Kanıt zemini:
`fraud/` (tam doğal testte hybrid FP32 AUC-ROC .9451 / INT8-channel .9367;
RF .973 / GBT .981 tavanlarının altında).

## Subproje: aml/, maintenance/

2026-10-16 açıldı (ticari gerekçe `docs/market_priorities.md`): `aml/` =
sentetik E0/E1 sonrası Elliptic E2/E2′ gerçek-veri deneyi tamamlandı; E5 beş-seed
stability pilotu kaba name label'ı SHAP top-1'den ~3× stabil buldu; sürekli 2D
koordinatlar ve attribution magnitudeleri retraining-stable değil (§6.5 name-scope).
En iyi
K10/B8 hybrid aynı-split GBT'den geride. E2 açıklanabilirlik probe'u
`aml/results/e2_explain.json`'da (held-out nearest-prototype %75, illicit %45);
ham tutar ve hesap kimliği yok, tek-hop proxy/transductive sınırlaması açık.
Gerçek-veri bölge haritası `viz` işi; E3′ bağımsız zaman-disjoint doğrulaması tamamlandı.
`maintenance/` = C-MAPSS/CWRU (pencere→bulut); alt-proje run'ları CPU. MPS şu an boş.
Positioning: `aml/figures/typology_naming_{comparison,layering}.png` + `aml/figures/e2_elliptic_{region_map,naming_comparison}.png` (GBM/SHAP vs bölge-prototip; sentetik + gerçek) + makale `docs/paper_draft.md` §6.5 (isimlendirme boşluğu).

## Dondurulmuş Kurallar

`AGENTS.md` §3'e bak: `state_dict` ana adları, checkpoint adlandırması,
`logs/` append-only, `_archive/` yolu.
