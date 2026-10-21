# RESEARCH.md — Quann-3D Tek Durum Kaynağı

> **Kurallar:** Her oturum BAŞINDA okunur. Bu dosyanın dışında ağır içerik
> (logs/, checkpoints/, kod) ana oturuma gelmez — domain agent'ları (`experiment`,
> `geometry`, `paper`) okur, distile sonuç döndürür. Güncelleme protokolü:
> `AGENTS.md` §5. Boyut bütçesi: **≤150 satır** — taşarsa en eski deney "Son
> Deneyler" tablosundan çıkarılır (detay `logs/`'da kalır).
> Son güncelleme: 2026-10-20

## Bir Cümle

Transformer'ın "sıralı dikkat + tek vektör" paradigmasının yerine: kelime → ND
uzayda nokta, cümle → nokta bulutu, anlam → bulut geometrisi; tahmin bu
konfigürasyondan. İkinci katman QUANN-Cloud: aynı mimariyi backprop'suz (0th-order)
eğiten yöntem.

## Canlı Runlar (2026-10-20)

| Run | Config | Durum | PID | Log |
|---|---|---|---|---|
| R1 D=64 × 200K-wiki | 50ep, batch 64, seed/eval düzeni sabit; ortak BPE | çalışıyor | queue 42316 (train 42331) | `logs/run_d64_200k.log` |

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
| R1 | Kapasite: D=64 / D=128 | 1 | 20K: D=64 en iyi (181 vs 201); wiki D=32 kontrolünde derinlik baskın çıktı. D=64×200K-wiki/50ep gate çalışıyor (`logs/run_d64_200k.log`). |
| R2 | Cloud attention (ilişkisel sözlük genişletme) | 3 | bekliyor |
| R3 | Attention-weighted pooling (max-pool onarımı) | 2 | bekliyor |
| R4 | QUANN: aramayı pair-kanalına yönlendir (freeze varyantı) | 4 | bekliyor |

## Son Deneyler (en yeni üstte)

| Tarih | Run | Sonuç | Ders | Log |
|---|---|---|---|---|
| 2026-10-20 | P1 online-cloud-separation (reader/head BCE fine-tune, 3ep; full 4.7M) | @.9969 FP32 FP 9,942→8,468, TP 1,842→1,930; P/R .156/.301→.186/.316; PR/ROC .2560/.9451→.2706/.9474. INT8-channel FP 27,858→10,856, TP 1,931→1,922; PR/ROC .2506/.9367→.2573/.9389 | Reader-space ayrım zayıf-orta (d/σ=.71 cloud, 1.00 encoded); fine-tune AUC'yi artırdı ama INT8 FP hâlâ FP32 baz çizgisinden yüksek — eşik davranışı çözülmedi | `fraud/logs/run_online_cloud_separation.log` |
| 2026-10-20 | P1 fraud false-alarm audit re-run (full 4.7M, `--dom-pairs`; eski JSON → `false_alarm_audit_20261019.json`) | FP 9,942 birebir repro; dominant çiftler (n=558/80/80): joint **0.92/0.00/0.97** ≈ max(single 0.44/0.96, 0/0, 0.96/0) → **additif** | n=1 süper-adektivite (Rome+MCC=5970) istatistiksel n'de yok — operasyonel FP popülasyonu tek generic online tokenlar (ONLINE %94, MCC=5311 %61, amount>100 %73); pair-geometri non-additivity caveat'ı pratikte önemsiz | `fraud/logs/run_false_alarm_audit.log` |
| 2026-10-19 | P1 fraud explanation deploy surface (ONNX + bundled quantizer + token decoder; 19.96K User0 row self-test) | gerçek-satır torch↔ORT parite max **1.5e-6**, ~**45.5k tx/s**; decoder 221 girdi = tam sözlük | edge_eval ONNX'i + quantizer + decoder tek paket (`fraud/results/deploy/`); `model.onnx` gitignored → sha256 manifest'te, `run_export_deploy.sh` ile repro | `fraud/logs/run_export_deploy.log` |
| 2026-10-19 | P1 fraud false-alarm audit (full 4.7M; operational thr 0.9969 → 9.942 FP + 2.000 TN; per-token slice + joint counterfactual) | FP olasılığı ~.999 doymuş; popülasyonu generic online tokenlar sürüyor (MCC=5311 %61, ONLINE %94; tek-mask unflag 94–100%) | upper-slice'in per-row adayları (Rome %5.6, MCC=5970 n=1) güçlü (+8.2) ama popülasyonda küçük pay; joint mask'ta non-additive (n=1, tanımlayıcı) | `fraud/logs/run_false_alarm_audit.log` |
## Sıradaki Aksiyonlar

1. **P1:** Online-cloud reader fine-tune TAMAM: full 4.7M FP32 FP 9,942→8,468 (+88 TP), INT8-channel FP 27,858→10,856 (−9 TP), ROC .9367→.9389; FP32 metrikleri iyileşti ama INT8 FP hâlâ 9,942 üstünde. Sonraki gate: aynı `.9969` eşiğinde quantization-aware fine-tune; geometri `fraud/results/online_cloud_separation.json`.
2. **Canlı (MPS):** R1 D=64 × 200K-wiki/50ep — `logs/run_d64_200k.log`; bitince held-out PPL'yi D=32/6.45 ile karşılaştır ve bu satırı deney geçmişine taşı.
3. Remote yok: `181539a`/`5f66aed` yerelde; push değil — gerekiyse `scripts/backup_ssd.sh`/bundle ile yedekle.
4. **Doküman satır bütçesi** kuralı kondu (AGENTS.md §3.7 + `scripts/check_docs_lines.sh`; formal istisna: `docs/patent/**`, `paper_draft`). Geçiş kuyruğu TAMAM (2026-10-20, ağ kapalı → kullanıcı onayıyla ana oturumda): 2 bayat patent `_archive/quann_cloud/`a (gerekçe `ARCHIVE.md`), 4 dosya ≤200'ye distile (fraud/README 198, FINDINGS 195, PROJECT_SUMMARY 173, DESIGN 173), checker listesi temiz → `OK`.

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
P1 explanation pilotu (`fraud/src/explain_pilot.py`, `fraud/results/quann_full_hybrid.pt`: D=8, d_cloud=128, d_model=32, 214 nokta, bundled quantizer): User0 small CSV'den seed 20261015 ile stratified 827 satır (27 fraud/800 normal); PR-AUC .2555, ROC-AUC .8286. All-sample etki tokenları çoğunlukla fraud-koruyucu (Merchant City=other, zip range, Merchant State=CA, error_type=none, Use Chip=Swipe); pozitif gerçek-fraud sürücüleri nadir Merchant Name kimliği, amount≈[30.79,46.26), saat=16. Leave-one-point-out logit farkı kullanıldı; bu SHAP değildir ve pair geometry non-additive. Çıktı `fraud/results/explain_pilot.json`; log `fraud/logs/run_explain_pilot.log`; `fraud/tests/test_fraud.py`: 14/14 geçti. Upper-slice (tam 4.7M testin 6110 fraud + 43.890 en-olasılı normali; base 12.2%): slice AUC-PR .2908 vs full .2560 (truncation artefaktı); çıktı `fraud/results/explain_upper_slice.json`, log `fraud/logs/run_explain_upper_slice.log`. False-alarm audit'ı (`fraud/src/audit_false_alarms.py`, tam 4.7M, operational thr .9969: 1.842 TP/9.942 FP/4.268 FN; P .156/R .301): FP olasılığı ~.999 doymuş ve popülasyonu generic online tokenlar sürüyor (MCC=5311 %61, amount>100 %73, ONLINE %94; tek-mask counterfactual 94–100% unflag); upper-slice'in per-row adayları Rome (%5.6) ve MCC=5970 (n=1) güçlü (+8.2) ama küçük pay. Çıktı `fraud/results/false_alarm_audit.json`, log `fraud/logs/run_false_alarm_audit.log`. Explanation deploy yüzeyi tamam (`fraud/src/export_deploy.py`, `fraud/results/deploy/`): ONNX (543KB, dynamic batch) + bundled quantizer + token decoder (221 = tam sözlük) tek paket; `model.onnx` gitignored → sha256 manifest'te, `run_export_deploy.sh` ile repro; 19.96K User0 satırda gerçek-satır torch↔ORT parite **1.5e-6**, ~**45.5k tx/s**. Audit'e `--dom-pairs` (dominant çift joint) eklendi; tam 4.7M re-run ≥12GB bellek kapısında. `fraud/tests/test_fraud.py`: 16/16.

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
`maintenance/` = C-MAPSS/CWRU (pencere→bulut); alt-proje run'ları CPU. MPS şu an R1 D=64 eğitiminde.
Positioning: `aml/figures/typology_naming_{comparison,layering}.png` + `aml/figures/e2_elliptic_{region_map,naming_comparison}.png` (GBM/SHAP vs bölge-prototip; sentetik + gerçek) + makale `docs/paper_draft.md` §6.5 (isimlendirme boşluğu).

## Dondurulmuş Kurallar

`AGENTS.md` §3'e bak: `state_dict` ana adları, checkpoint adlandırması,
`logs/` append-only, `_archive/` yolu.

## Yedek Altyapısı

`scripts/backup_ssd.sh`: günlük 03:00 (launchd `com.quann3d.backup-ssd`) →
`/Volumes/x/backups/quann-3d/`: git bundle (tam geçmiş) + tam repo tar (artifact'lar
dahil; 1 jenerasyon rotasyon `.prev`). SSD yoksa exit 2, zararsız. Restore: o README.
