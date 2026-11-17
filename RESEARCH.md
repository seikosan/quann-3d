# RESEARCH.md — Quann-3D Tek Durum Kaynağı

> Her oturum başında okunur. Ağır log/checkpoint/kod incelemesi alan agent'larında kalır.
> Güncelleme protokolü: `AGENTS.md` §5. Dosya bütçesi: en çok 150 satır.
> Son güncelleme: 2026-11-16 — CC global sonucu işlendi

## Bir Cümle

CloudLM, metni uzaydaki nokta bulutlarıyla temsil eder; QUANN-Cloud aynı yaklaşımı
gradyan geri yayılımı olmadan eğitmeyi araştırır.

## Canlı Runlar

| Run | Durum / PID | Başlangıç | Plan / beklenen süre | Log |
|---|---|---|---|---|

## Son Deneyler

| Tarih | Run | Tek satır sonuç | Ders | Log |
|---|---|---|---|---|
| 2026-11-17 | Görev-sabit ablation: stable mekanizma memory görevinde | TEST loss 3.5425 (`logs/run_r5_ablate_stable_on_memorytask.log:1332`); best val 3.4802 ep 26 (`logs/run_r5_ablate_stable_on_memorytask.log:1151`); 30 ep doldu. | ~0.4'lik farkın ayrımı: görev zorluğu 0.3107 (3.5425−3.2318) + mekanizma 0.0846 (3.6271−3.5425); memory mekanizması kendi görevinde pencere/bulut'tan tek seed'le 0.08 kötü — çürük değil ama üstün de değil. | `logs/run_r5_ablate_stable_on_memorytask.log` |
| 2026-11-17 | Memory stable-boyutu kontrolü (d_model 128, d_sentence 256, vocab 10000) | TEST loss 3.6271 (`logs/run_r5_cc_memory_stablesz.log:894`); best val 3.5578 ep 15 (`logs/run_r5_cc_memory_stablesz.log:668`); erken durdurma ep 20. | Model boyutu farkı önemsiz (3.6295→3.6271, Δ0.0024); stable'a kalan ~0.4 fark boyuttan değil — görev zorluğu (4-chunk hedef) + mekanizma karışımı; ayrım için görev-sabit mekanizma ablation'ı gerekir. | `logs/run_r5_cc_memory_stablesz.log` |
| 2026-11-17 | R5 memory vocab=10000 kontrollü rerun | Aynı split/seed/vocab (10000, stable ile birebir): memory TEST loss 3.6295 vs stable 3.2318; best val 3.5735, erken durdurma ep 10 (`logs/run_r5_cc_memory_v10k.log:453`). | Vocab-eşleşik kıyas: 9.56→3.63 düşüşün baskın nedeni vocab boyutu; kalan ~0.4 fark model boyutu (memory d_model 64 vs stable 128) + görev farkıyla karışık, mekanizma kusuru kanıtı değil. | `logs/run_r5_cc_memory_v10k.log` |

## Ölçülmüş Bulgular

- Eşleşmiş wiki holdout duel'inde eski kodda D64 held-out PPL 71.09 (`logs/run_d32_wiki200k.log:14`),
  D32 73.27 (`logs/run_d32_wiki200k.log:11`) — eski kodda (0df8af5 worktree) birebir
  yeniden üretildi (`logs/run_r1_duel_oldcode.log:1,2`); bu eski-kod sonucu güncel
  mimarideki R4 duel'iyle ayrıca sınandı (`logs/run_r4_resume.log:71`).
- R1 duel token farkının kökü bulundu: commit e429df6 `BPETokenizer.encode`
  sonuna `<eos>` eklemeye başladı; token 5149 (`logs/run_d32_wiki200k.log:14`) →
  5354 (`logs/run_d32_wiki200k.log:19`).
- Eski D32/D64 checkpoint'leri güncel forward'la sadık değerlendirilemez (814a468
  mimari drift; pair_mlp girdi formülü `cloudlm/model.py:61`); güncel kodla ölçülen
  holdout PPL D32 26372.36 (`logs/run_r1_duel_repro2.log:3`) /
  D64 137.78 (`logs/run_r1_duel_repro2.log:6`).
- 2026-11-16 düzeltme: memory vocab'ü 280151 — `logs/run_r4_contextcloud_memory.log:4`
  ; best val loss 9.4801 — `logs/run_r4_contextcloud_memory.log:550`
  ; ln(vocab)=12.5431 olduğundan 9.4801 kendi rastgele seviyesinin altındadır
  → memory öğreniyor; ancak bu val değerleri held-out performansını doğrulamaz.
- 2026-11-16 düzeltme: Stable vocab 10000 (`logs/run_r4_contextcloud_stable.log:3`),
  memory vocab 280151 (`logs/run_r4_contextcloud_memory.log:4`); görev/vocab farklı
  olduğundan stable-memory loss'ları doğrudan kıyaslanamaz.
- 2026-11-16: CC global yeniden koşusu durdu — son epoch 22/30
  (`logs/run_r4_contextcloud_global2.log:2190`); Early stopping tetiklendi
  (`logs/run_r4_contextcloud_global2.log:2293`); best val loss —
  `logs/run_r4_contextcloud_global2.log:2296` satırında 7.7961; vocab —
  `logs/run_r4_contextcloud_global2.log:2` satırında 280151.
  ln(vocab)=12.5431 (hesap bu oturumda) — global loss kendi rastgele-seviye
  referansının altında. R4'ün üç CC varyantı tamamlandı; vocab'ler farklı,
  loss'lar doğrudan kıyaslanamaz: stable vocab —
  `logs/run_r4_contextcloud_stable.log:3` satırında 10000; val —
  `logs/run_r4_contextcloud_stable.log:1532` satırında 3.1327; memory vocab —
  `logs/run_r4_contextcloud_memory.log:4` satırında 280151; val —
  `logs/run_r4_contextcloud_memory.log:550` satırında 9.4801; global val —
  `logs/run_r4_contextcloud_global2.log:2296` satırında 7.7961.
- Contextcloud memory/stable validation değerleri gerçek held-out değildir:
  “val” aynı train dataloader üzerinden hesaplanıyor
  (`contextcloud/train_memory_context.py:340`; `contextcloud/train_stable.py:191`).
- Kanıtsız katkı adayları: memory encoder'daki padding-pool ve paylaşılan
  `encoder_layer`; etkileri henüz ölçülmedi
  (`contextcloud/train_memory_context.py:93`; `contextcloud/train_memory_context.py:108`).
- Contextcloud global ilk denemesi MPS `aten::_nested_tensor_from_mask_left_aligned`
  eksikliğiyle sonlandı; yeniden koşu için `enable_nested_tensor=False` ayarı
  kullanıldı (`logs/run_r4_contextcloud_global.log`; `contextcloud/train_global_context.py:34`).
- R3 20K karşılaştırmasında MAX PPL 315.99 (`quann_cloud/REVISION.md:56`).
- R3 20K karşılaştırmasında ATTN PPL 334.89 (`quann_cloud/REVISION.md:56`).
- R3 200K karşılaştırmasında MAX PPL 61.91 (`quann_cloud/REVISION.md:57`).
- R3 200K karşılaştırmasında ATTN PPL 66.85 (`quann_cloud/REVISION.md:57`).
- R2 cloud-attention logunda D32 PPL 58.88 (`logs/run_r2_cloud_attn_200k_d32.log:26`).
- R2 cloud-attention logunda D64 PPL 57.04 (`logs/run_r2_cloud_attn_200k_d64.log:26`).
- R2 cloud-attention logunda D128 PPL 58.55 (`logs/run_r2_cloud_attn_200k_d128.log:26`).
- Nokta sayısı varyantı logunda PPL 37.63 (`logs/run_r2_cloud_attn_200k_d64_n20.log:26`); satır held-out olarak etiketlenmediğinden held-out sonucu sayılmamalıdır.
- Fraud doğal testinde FP32 AUC-ROC 0.9451 (`fraud/logs/run_full_scale_edge_eval.log:52`).
- Fraud doğal testinde INT8-channel AUC-ROC 0.9367 (`fraud/logs/run_full_scale_edge_eval.log:54`).

Detay: `quann_cloud/REVISION.md`.

## Hipotezler ve Durum

- Kapasite: R4 duel'i güncel mimaride D64 lehine sonuçlandı; held-out farkı
  küçüktür (`logs/run_r4_resume.log:71`).
- Cloud attention: boyut varyantlarının kaydedilen PPL'leri birbirine yakındır;
  sonuçlar held-out diye sunulmamıştır.
- Attention-weighted pooling: REVISION.md iki korpustaki karşılaştırmayı MAX
  lehine raporlar; attention-pool bu nedenle reddedilmiştir.
- QUANN arama yönünü pair-kanalına taşıma: beklemede
  (`quann_cloud/REVISION.md`).

## Sıradaki Aksiyonlar

- Tamamlandı (17 Kas): ayrık 80/10/10 holdout bölmesi (seed=42) üç CC script'ine
  eklendi ve R5 koşuları bitti; gerçek TEST loss'lar `logs/run_r5_cc_*.log` içinde.
- Tamamlandı (17 Kas): vocab=10000 eşitlenmiş kontrollü memory rerun bitti;
  memory TEST 3.6295 (`logs/run_r5_cc_memory_v10k.log:454`) ve stable TEST
  3.2318 (`logs/run_r5_cc_stable.log:1236`).
- Kalan ~0.4 farkı ayırmak için memory'yi stable boyutunda (d_model 128,
  d_sentence 256) vocab=10000 ile koş; model boyutu vs görev farkı ayrışır.
- Kullanıcı D=128'i önceliklendirirse kapasite adayı olarak ayrıca dene.
- Padding-pool ve paylaşılan `encoder_layer` katkılarını kontrollü deneyle ölç;
  etkileri henüz ölçülmedi (`contextcloud/train_memory_context.py:93`; `contextcloud/train_memory_context.py:108`).
- R4 duel hükmünü paper/patent notlarına yansıt (`logs/run_r4_resume.log:71`).
- Cloud attention varyantlarını held-out protokolüyle karşılaştır; mevcut PPL
  loglarını test PPL'i olarak yorumlama.
- QUANN pair-kanalı/freeze adayını ayrı bir deney olarak planla.
- Fraud deploy kararı kullanıcıda; deney kanıtları `fraud/results/` altındadır.

## Alt-Projeler

- `quann_cloud/`: teşhis ve deney adayları `REVISION.md`, deney kaydı `EXPERIMENTS.md`.
- `fraud/`: sonuç ve dağıtım kanıtları `results/` altında.
- `aml/`, `maintenance/`: deney sonuçları kendi `results/`, `figures/` ve README
  dosyalarında tutulur; burada sayısal özet verilmemiştir.

## Pazar ve Patent

- Dikey öncelikler ve kazanma mantığı (açıklanabilirlik + uyum hızı + boyut):
  `docs/market_priorities.md`.
- Patent: evrensel çekirdek `docs/patent/evrensel/geometry_dual_use_core_v1.md`
  (dual-use geometri); dil ve tabular embodiment'ları `docs/patent/` altında.

## Alt-Proje Kanıt Pointer'ları

- `fraud/`: tam ölçek hybrid + duels + deploy paketi `fraud/results/` altında
  (`full_scale_hybrid.json`, `edge_eval_full_scale.json`, `deploy/`,
  `explain_pilot.json`).
- `complaints/`: ölçek kararlılığı `complaints/results/complaints_scale_stability.json`.
- `aml/`: açıklanabilirlik probe'u `aml/results/e2_explain.json`; bölge haritası
  figürleri `aml/figures/` altında.

## Dondurulmuş Kurallar

`AGENTS.md` §3'e bak: state_dict ana adları, checkpoint adlandırması,
`logs/` append-only ve `_archive/` yolu.

## Yedek Altyapısı

Yedek alma/geri yükleme yönergeleri `scripts/backup_ssd.sh` ve ilgili README'dedir.
