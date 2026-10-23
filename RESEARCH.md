# RESEARCH.md — Quann-3D Tek Durum Kaynağı

> **Kurallar:** Her oturum BAŞINDA okunur. Bu dosyanın dışında ağır içerik
> (logs/, checkpoints/, kod) ana oturuma gelmez — domain agent'ları (`experiment`,
> `geometry`, `paper`) okur, distile sonuç döndürür. Güncelleme protokolü:
> `AGENTS.md` §5. Boyut bütçesi: **≤150 satır** — taşarsa en eski deney "Son
> Deneyler" tablosundan çıkarılır (detay `logs/`'da kalır).
> Son güncelleme: 2026-10-23

## Bir Cümle

Transformer'ın "sıralı dikkat + tek vektör" paradigmasının yerine: kelime → ND
uzayda nokta, cümle → nokta bulutu, anlam → bulut geometrisi; tahmin bu
konfigürasyondan. İkinci katman QUANN-Cloud: aynı mimariyi backprop'suz (0th-order)
eğiten yöntem.

## Canlı Runlar (2026-10-23)

R3 attention-pool 20K matched ön-test (MPS): `logs/run_r3_20k_max_v2.log`/`_attn_v2.log` (wiki korpus, R1 matched-duel protokolü; max kontrol → attn ardışık). Hüküm disjoint holdout DUEL_SUMMARY'ye: **MAX 315.99 > ATTN 334.89 (Δ18.90)** — attention-pool performansı düşürdü. Katman-2 complaints ölçümü CPU'da tamamlandı (3.77M, region stability ARI 0.994, Spearman 1.0).

**Bitiş teşhisi (tarih):** 1.5M wiki run'ı epoch 17/20'de exit 137 ile OOM/kill oldu (RSS ~13.2 GB, free ~1.2 GB); epoch 18–20, bellek düzeltmesi sonrası kuyruktan resume edildi. Queue'daki step-4 eval, stdin çalıştırmasında `training/`'in sys.path'te olmaması ile bir kez FAILED (ModuleNotFoundError); script düzeltildi, eval retry ile tamamlandı.

## Mevcut Best (legacy protokol: seed=0 ile 2000 örnek; gerçek held-out değil)

| Model | Config | Param | PPL (legacy eval) | Checkpoint |
|---|---|---|---|---|
| CloudLM | D=32, 200K, 50ep | 21.0M | **6.45** | `checkpoints/wiki_checkpoints_200k/wiki_cloud_best.pt` |
| Causal baseline | d=384, 4L, 200K, 50ep | 22.7M | **2.92** | `checkpoints/baseline_200k/baseline_best.pt` |

> 1.5M: 143.00 **ölçüm artefaktı** (BPE/dağıtım eşleşmemesi). Wiki PPL'leri eğitim korpusundan seed=0 örneklemidir: 62.15 tam 1.5M train'inden, 64.28/24.53 ilk-200K train kesitinden; D32'nin 218.84 ve D64'ün 59.32 full-1.5M örnekleri ilk-200K training slice ile kısmen örtüşebilir. D64 19.08 ilk-200K slice'tan tamamen in-sample'dır. Bu yüzden hiçbiri strict held-out değil; D32 wiki'nin 218.84/24.53 değerleri ayrıca D64 BPE'siyle eşleşmez ve aralarında kapasite kıyası yapılamaz. **10,698,546.20 / 8,041,277.58 BPE-mismatch ölçüm artefaktlarıdır.** 6.45 de `turkish_sentences` D32 eğitim kümesinden seed=0 örneklemesidir; held-out değildir. Mevcut Best satırı korunmuştur; bağımsız test gerekir.

> Eşleşmiş holdout referansı: D64 **71.09** (200K-wiki ilk-200K train, ortak BPE; satırlar [200000,202000), n=2000, seed=0); eski PPL'lerle protokol farkı nedeniyle doğrudan kıyaslanmaz.

**Kök teşhis (sensitivity_analysis, ölçülmüş):** kapasite (D boyutu) —
hassasiyet değil. Bulut (10×32=320 sayı), transformer gizli durumunun
(10×256=2560) %12.5'ini taşıyor. Detay: `quann_cloud/REVISION.md`.

## Aktif Hipotezler (R1–R4 — `quann_cloud/REVISION.md`)

| # | Hipotez | Öncelik | Durum |
|---|---|---|---|
| R1 | Kapasite: D=64 / D=128 | 1 | Matched duel tamam: D64 71.09 < D32 73.27 (Δ2.18) — kapasite yönü doğru, büyüklük sınırlı; ikisi de baseline 2.92'nin çok üstünde → darboğaz mimari/temsil, kapasite değil. |
| R2 | Cloud attention (ilişkisel sözlük genişletme) | 3 | bekliyor |
| R3 | Attention-weighted pooling (max-pool onarımı) | 2 | bekliyor |
| R4 | QUANN: aramayı pair-kanalına yönlendir (freeze varyantı) | 4 | bekliyor |

## Son Deneyler (en yeni üstte)

| Tarih | Run | Sonuç | Ders | Log |
|---|---|---|---|---|
| 2026-10-23 | R3 attention-weighted pooling 20K duel (max vs attn, disjoint holdout [200000,202000), n=2000, seed=0) | Held-out PPL MAX **315.99** < ATTN **334.89** (Δ18.90, MAX kazandı); train PPL MAX 88.19 vs ATTN 124.22. | Attention-pool 20K'da overfitting'e yol açtı (train PPL 124.22 vs 88.19). 200K'da tekrar test edilmeli — küçük korpusta attention sinyali gürültüye dönüşüyor olabilir. | `logs/run_r3_20k_max_v2.log` + `logs/run_r3_20k_attn_v2.log` + `training/eval_r3_duel.py` |
| 2026-10-22 | R1 matched duel D32 vs D64 (aynı 200K-wiki + ortak BPE; disjoint holdout [200000,202000), n=2000, seed=0) | Held-out PPL D64 **71.09** < D32 **73.27** (Δ2.18, D64 kazandı); train 21.20 vs 22.12. | D64 yönü destekleniyor ama ~%3 fark, REVISION kapasite teşhisine göre bulutun temsil kapasitesi sınırlı; ölçülen büyük train→holdout gap de bölge driftini araştırmayı gerektiriyor. | `logs/run_d32_wiki200k.log` (D32 eval) + `logs/run_d32_200k.log` (D32 train) |
| 2026-10-21 | R1 D=64 × 200K-wiki (50ep) | Wiki ilk 200K satır (189,872 uzun-sekans), referans BPE; train PPL **21.20**; full-corpus 59.32 (kısmi overlap riski), slice 19.08 (in-sample); 6.45 setinde D32=6.45, D64=2621.31; BPE-mismatch kontrolleri artefakt | 6.45 örneği D32'nin eğitim verisinden; D64 farklı corpus'ta eğitildi, wiki D32 BPE'si farklı — run kapasite etkisini ayırmıyor ve R1 sonuçsuz. | `logs/run_d64_200k.log` |
| 2026-10-21 | P1 QAT gate (online-neutralized ckpt + STE fake-quant: per-channel weight + static per-tensor act; 4ep; full 4.7M) | @.9969: FP32 FP 8,468→6,817, TP 1,930→2,413; P/R .186/.316→.261/.395; PR/ROC .2706/.9474→.3097/.9502. INT8-channel FP 10,856→**3,834**, TP 1,922→2,014; P/R .150/.315→.344/.330; PR/ROC .2573/.9389→.2869/.9424 | Parity gap sistematik kaymaydı (100K: ΔP +0.059 ortalama, %72 yukarı); QAT FP32'yi de düzeltti — INT8 FP (3,834) FP32'den (6,817) bile düşük; kriter tuttu, operasyonel eşik değişmedi | `fraud/logs/run_qat_gate.log` |
| 2026-10-20 | P1 online-cloud-separation (reader/head BCE fine-tune, 3ep; full 4.7M) | @.9969 FP32 FP 9,942→8,468, TP 1,842→1,930; P/R .156/.301→.186/.316; PR/ROC .2560/.9451→.2706/.9474. INT8-channel FP 27,858→10,856, TP 1,931→1,922; PR/ROC .2506/.9367→.2573/.9389 | Reader-space ayrım zayıf-orta (d/σ=.71 cloud, 1.00 encoded); fine-tune AUC'yi artırdı ama INT8 FP hâlâ FP32 baz çizgisinden yüksek — eşik davranışı çözülmedi | `fraud/logs/run_online_cloud_separation.log` |
## Sıradaki Aksiyonlar

1. **R3 tekrar 200K:** Attention-pool 20K'da overfitting gösterdi (train PPL 124.22 vs MAX 88.19). 200K'da tekrar test edilmeli — küçük korpusta attention sinyali gürültüye dönüşüyor olabilir.
2. **R1 hükmü:** D64'ün 2.18 PPL (~%3) avantajı kapasite yönünü destekliyor ama REVISION'daki temsil-kapasitesi teşhisine göre ana darboğazı çözmüyor; D=128'e geçmeden önce mimari/temsil tavanını ve matched sonucun anlamını değerlendir.
3. **Yeni deney adayı:** train→holdout gap kök-nedenini ölç; [0,200000) ve [200000,202000) arasında BPE token frekans KL/kosinüs benzerliğini hesapla. Drift büyükse aynı bölgeden holdout varyantıyla duel'ü teyit et.
4. **P1 QAT:** deploy kararı hâlâ kullanıcıda; mevcut paket/kanıt `fraud/results/deploy/` ve `fraud/results/qat_gate.json` altında. Remote yok: `181539a`/`5f66aed` yerelde; push değil — gerekiyse `scripts/backup_ssd.sh`/bundle ile yedekle.

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

Patent: evrensel çekirdek `docs/patent/evrensel/geometry_dual_use_core_v1.md` (dual-use geometri: aynı bulut hem hesaplama hem native açıklama; dil/fraud/aml/bakım embodiment); dil v2.3 + tabular v1 onun uygulaması.

## Subproje: complaints/

Full-corpus ölçüm (3.77M usable, product mix, 2015–2026) + region stability (ARI mean 0.994, share Spearman 1.0) tamamlandı. Pilot scale drift: region 7 +21%, region 1 −17%. Output: `complaints/results/complaints_scale_stability.json`. Katman-2 scale ceiling: reader 20K pilot eğitildi, full 3.77M (188×) region assignment CPU'da infeasible.

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
`maintenance/` = C-MAPSS/CWRU (pencere→bulut); alt-proje run'ları CPU. R1 bitti; MPS'te aktif ağır eğitim yok.
Positioning: `aml/figures/typology_naming_{comparison,layering}.png` + `aml/figures/e2_elliptic_{region_map,naming_comparison}.png` (GBM/SHAP vs bölge-prototip; sentetik + gerçek) + makale `docs/paper_draft.md` §6.5 (isimlendirme boşluğu).

## Dondurulmuş Kurallar

`AGENTS.md` §3'e bak: `state_dict` ana adları, checkpoint adlandırması,
`logs/` append-only, `_archive/` yolu.

## Yedek Altyapısı

`scripts/backup_ssd.sh`: günlük 03:00 (launchd `com.quann3d.backup-ssd`) →
`/Volumes/x/backups/quann-3d/`: git bundle (tam geçmiş) + tam repo tar (artifact'lar
dahil; 1 jenerasyon rotasyon `.prev`). SSD yoksa exit 2, zararsız. Restore: o README.
