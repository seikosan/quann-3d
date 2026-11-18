# RESEARCH.md — Quann-3D Tek Durum Kaynağı

> Her oturum başında okunur. Ağır log/checkpoint/kod incelemesi alan agent'larında kalır.
> Güncelleme protokolü: `AGENTS.md` §5. Dosya bütçesi: en çok 150 satır.
> Son güncelleme: 2026-11-18 23:12 — R7 kapandı (R2 kollar geçersiz); R8 koşuyor

## Bir Cümle

CloudLM, metni uzaydaki nokta bulutlarıyla temsil eder; QUANN-Cloud aynı yaklaşımı
gradyan geri yayılımı olmadan eğitmeyi araştırır.

## Canlı Runlar

| Run | Durum / PID | Başlangıç | Plan / beklenen süre | Log |
|---|---|---|---|---|
| R8 QUANN pair/freeze (2 kol × 10 epoch) | Koşuyor / kabuk 9727 + python 9759 (`ps -p 9727 -o pid=,etime=`, `pgrep -f quann_pair_experiment`, 18 Kas 23:11 kontrolü) | 18 Kas 23:05:40 (`logs/run_r8_quann_pair.log:16`) | 20K cümle, seed=42, split 80/10/10; kontrol + pair-freeze kolu (`training/shell/run_r8_quann_pair.sh:37,43`) | `logs/run_r8_quann_pair.log` |

## Son Deneyler

| Tarih | Run | Tek satır sonuç | Ders | Log |
|---|---|---|---|---|
| 2026-11-18 | R5 seed=43 tekrarı — mekanizma farkı 0.0846→0.0755 tekrarlandı | Memory TEST: ablation 3.5425→3.5581, memory-stablesz 3.6271→3.6336; fark 0.0846→0.0755. Stable TEST: 3.2318→3.2131 (Δ0.0187). | Yaklaşık 0.08 mekanizma farkı iki bağımsız örnekte aynı yön ve büyüklükte tekrarlandı; orijinal koşular tohumlanmamış, seed=43 ayrı örnektir, aynı-seed tekrarı değildir. | Orijinal: `logs/run_r5_ablate_stable_on_memorytask.log:1332`, `logs/run_r5_cc_memory_stablesz.log:894`, `logs/run_r5_cc_stable.log:1236`; seed=43: `logs/run_r5_seed2_ablate_s43.log:982`, `logs/run_r5_seed2_memory_stablesz_s43.log:631`, `logs/run_r5_seed2_cc_stable_s43.log:1237` |
| 2026-11-17 | Görev-sabit ablation: stable mekanizma memory görevinde | TEST loss 3.5425 (`logs/run_r5_ablate_stable_on_memorytask.log:1332`); best val 3.4802 ep 26 (`logs/run_r5_ablate_stable_on_memorytask.log:1151`); 30 ep doldu. | İlk örnekte görev zorluğu/mekanizma farkı ayrıştırıldı; ikinci bağımsız örnek sonucu yukarıda. | `logs/run_r5_ablate_stable_on_memorytask.log` |
| 2026-11-18 | R7 cloud-attn held-out (eval-only, protokol `logs/run_r7_cloud_attn_heldout.log:11`) | R4 kontrolleri heldout_PPL: D32 69.7113 (`logs/run_r7_cloud_attn_heldout.log:14`), D64 69.1251 (`logs/run_r7_cloud_attn_heldout.log:17`); R2_attn kolları 61438.8713 (`logs/run_r7_cloud_attn_heldout.log:19`), 110738.9649 (`logs/run_r7_cloud_attn_heldout.log:21`), 991073.3084 (`logs/run_r7_cloud_attn_heldout.log:23`) — üçünden de önce legacy pos_enc remap uyarısı (`logs/run_r7_cloud_attn_heldout.log:18`). | Değerlendirme hattı sağlam: R4 kontrolleri aynı protokolün orijinal duel koşusuyla aynı mertebede — D32 61.94 (`logs/run_r4_resume.log:67`), D64 61.19 (`logs/run_r4_resume.log:70`). R2 attention sayıları bu kıyas için GEÇERSİZ: legacy forward uyumsuzluğu (`cloudlm/model.py:267-285`; bkz. yukarıda 814a468 notu); sapmanın kesin nedeni doğrulanmadı. | `logs/run_r7_cloud_attn_heldout.log` |
| 2026-11-18 | R6 pool/encoder katkıları, seed=43 (queue HEAD `7e13dae`, `logs/run_r6_pool_enc.log:2`; masked_pool / unshared flags `logs/run_r6_maskpool_s43.log:5`, `logs/run_r6_unshared_s43.log:5`, `logs/run_r6_both_s43.log:5`) | Held-out TEST: maskpool 3.6431 (Δ+0.0095), unshared 3.6363 (Δ+0.0027), both 3.6762 (Δ+0.0426), kontrol 3.6336 (`logs/run_r6_maskpool_s43.log:588`; `logs/run_r6_unshared_s43.log:456`; `logs/run_r6_both_s43.log:412`; kontrol `logs/run_r5_seed2_memory_stablesz_s43.log:631`; Δ hesapları bu TEST değerlerinden). | Üç varyant da kontrolün gerisinde; tek seedde katkı negatif/yararsız, neden doğrulanmadı. | `logs/run_r6_maskpool_s43.log`; `logs/run_r6_unshared_s43.log`; `logs/run_r6_both_s43.log` |

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
- Stable train-loss sıçramaları: seed=43 Epoch 3 5.6595, Epoch 5 15.2557; orijinal Epoch 4 8.1102. Her iki koşuda best checkpoint Epoch 1'de seçildi; sıçramalar TEST'e yansımadı (best-ckpt TEST 3.2131/3.2318); sebebi doğrulanmadı (`logs/run_r5_seed2_cc_stable_s43.log:211,619,1027,1237`; `logs/run_r5_cc_stable.log:211,822,1236`).
- R5 memory mekanizma farkı iki bağımsız örneğe dayanır: orijinal tohumlanmamış, seed=43 ayrı örnek olup aynı-seed tekrarı değildir (`logs/run_r5_ablate_stable_on_memorytask.log:1332`; `logs/run_r5_seed2_ablate_s43.log:982`).

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

- Tamamlandı (17 Kas): ayrık 80/10/10 holdout bölmesi (seed=42) üç CC script'ine eklendi ve R5 koşuları bitti; gerçek TEST loss'lar `logs/run_r5_cc_*.log` içinde.
- Tamamlandı (17 Kas): vocab=10000 eşitlenmiş kontrollü memory rerun bitti;
  memory TEST 3.6295 (`logs/run_r5_cc_memory_v10k.log:454`) ve stable TEST
  3.2318 (`logs/run_r5_cc_stable.log:1236`).
- Tamamlandı (18 Kas): R5 seed=43 ablation → memory-stablesz → stable kuyruğu. Mekanizma farkı iki TEST değeri farkıdır: seed42 3,6271 (`logs/run_r5_cc_memory_stablesz.log:894`) eksi 3,5425 (`logs/run_r5_ablate_stable_on_memorytask.log:1332`); seed43 3,6336 (`logs/run_r5_seed2_memory_stablesz_s43.log:631`) eksi 3,5581 (`logs/run_r5_seed2_ablate_s43.log:982`).
- Kullanıcı D=128'i önceliklendirirse kapasite adayı olarak ayrıca dene.
- Tamamlandı (18 Kas): R6 üç varyant da kontrol 3.6336'nın gerisinde (`logs/run_r5_seed2_memory_stablesz_s43.log:631`), NEGATİF katkı (delta +0,0027…+0,0426; hesap kontrol ve varyant TEST satırlarından: `logs/run_r6_unshared_s43.log:456`, `logs/run_r6_maskpool_s43.log:588`, `logs/run_r6_both_s43.log:412`); tek seed, neden doğrulanmadı.
- Tamamlandı (18 Kas): R4 hükmü paper §6.8 ile `BLUS_NOTU.md:31`/`TARIFNAME.md:167`
  içinde zaten var; ham log doğrulandı: winner=D64, delta=0.76 (`logs/run_r4_resume.log:71`).
- Tamamlandı (18 Kas, koşu başlangıcı `logs/run_r7_cloud_attn_heldout.log:1`): R7 cloud-attn held-out bitti; R4 kontrolleri aynı mertebede: 69.7113 (`logs/run_r7_cloud_attn_heldout.log:14`), 69.1251 (`logs/run_r7_cloud_attn_heldout.log:17`). R2 attention kolları legacy pos_enc uyumsuzluğu nedeniyle GEÇERSİZ (`cloudlm/model.py:267-285`) — attention-vs-no-attn sorusu bu checkpoint'lerle yanıtlanamaz; çözüm adayları (uyumlu değerlendirici ya da güncel kodla yeniden eğitim) kullanıcı kararı adayıdır.
- Koşuyor (18 Kas 23:05:40, `logs/run_r8_quann_pair.log:16`): R8 QUANN pair/freeze; ilk deneme kontrol kolunda öldü — shell `Killed: 9` bildirdi, logda izi `R8 control FAILED` (`logs/run_r8_quann_pair.log:15`); bellek kıtlığı şüphesi, sebep doğrulanmadı (jetsam kaydı bulunamadı). İkinci deneme kullanıcı onayıyla mlx_lm kapatıldıktan sonra başladı.
- Tamamlandı (18 Kas): Fraud deploy kullanıcı onayıyla tazelendi (paket 18 Kas 17:30,
  parity max=1.27e-07); tester 5/5 ✓ (manifest↔sha, ORT sözleşme, FP32 0.9451 ham
  satırı `fraud/logs/run_full_scale_edge_eval.log:52`).

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
- Vekil taslağı kullanici onayıyla üretiliyor → `docs/patent/vekil_v1/` (18 Kas);
  final_v3 dosyaları ve `son_draft.zip` dokunulmaz.

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
