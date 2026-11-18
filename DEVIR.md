---
id: devir
baslik: DEVIR — quann-3d (makine üretimi durum + sıradaki iş)
konu: [devir, handover, durum]
durum: güncel
sahiplik: koordinator
guncellendi: 2026-11-18
kunye-kaynak: otomatik (devir_yaz.py)
---

# DEVIR — quann-3d

<!-- MAKINE:BASLA -->
## Durum (makine üretti · 2026-11-18 15:47)
Bu blok elle düzenlenmez; `devir_yaz.py --repo /Users/serkankose/Documents/quann-3d` ile yenilenir.

- HEAD: 3dc1ad6 DEVIR: push kaydi BLOCKED'ya gecti — GitHub 100MB siniri + gecmisteki 1,76GiB backup tarball (e33deb7); kullanici karari 'simdilik push yok'
- Değişen dosya: 3 → README.md, docs/patent/final_v3/Archive/, docs/patent/final_v3/son_draft.zip
- Kod kapısı: bulgu 0 · uyarı 0 · bilgi 0 · düzeltme planı: 0 (ayrıntı: python3 ~/.config/opencode/denetim/denetim_plan.py)
- Doküman kapısı: bulgu 0 · uyarı 0 · bilgi 2 · düzeltme planı: 0 (ayrıntı: python3 ~/.config/opencode/denetim/denetim_plan.py)
- ölçüm: === EĞİTİM DURUMU (kaynak: ham dosya) ===
- ölçüm: kaynak        : logs/run_resume_r2_fixed.log:212
- ölçüm: ham satır     : Epoch 300/300 | loss=2.5816 | ppl=13.22 | 62815s
- ölçüm: epoch         : 300/300  (100% tamamlandı)
- ölçüm: loss / ppl    : 2.5816 / 13.22
- ölçüm: geçen süre    : 62815 s = 17.45 saat
- Açık kartlar (pano yok): (belirtilmedi)
<!-- MAKINE:BITIR -->















## Sıradaki iş (ajan)

- **Patent v3 NİHAİ paket hazır (18 Kas, commit `1afc688`):** `docs/patent/final_v3/`
  — ISTEMLER (15 istem, 3 bağımsız) + TARIFNAME (237 satır, §12 dürüstlük kayıtları)
  + OZET + TEKNIK_ETKI + 8 SVG (Şekil 6/7 ölçümlü, gerçek JSON'dan) + build_v3.py
  kapılarıyla; 5 DOCX + 5 PDF üretildi. Bağımsız reviewer turu: 71 sayı-atıf ✓,
  kırmızı çizgi ihlali 0, bulunan 2 atıf hatası + 5 künye boşluğu giderildi.
  Eski `patent-cikti/basvuru_paketi` ve `docs/patent/dil` artık yalnız referans.
- **Vekil kararları (v3, kullanıcı tarafı):** özet uzunluğu (265 kelime), 3 bağımsız
  istemin birliği, iki-bölümlü kalıp tercihi — `BLUS_NOTU.md` kırmızı çizgileriyle
  birlikte vekile götürülecek (vekil raporu §d).
- **Push BLOCKED — kullanıcı kararı "şimdilik push yok" (18 Kas):** refactor
  commit'leri (`1f21cec`…`64555cc`) + v3 paket commit'leri `deneme_context_cloud`
  dalında yerel. Push denemesi GitHub'ın 100 MB tek-dosya sınırına takıldı: geçmişte
  `quann-3d-backup-20261028.tar.gz` = 1.891.553.280 B ≈ 1,76 GiB (ekleyen commit
  `e33deb7`; `git rev-list --objects origin/main..HEAD` taraması — gönderilecek
  1238 nesne, 10 MB üstü TEK dosya bu). Kalan ~10 MB. İki seçenek: (a) geçmişi
  yeniden yaz (dal hiç push edilmediği için güvenli; diskteki dosyaya dokunulmaz,
  `.gitignore`'a eklenir) → push saniyeler; (b) Git LFS. Kullanıcı (a)'yı reddetti,
  push ertelendi.
- **Tamamlandı (17 Kas):** CC ayrık holdout (80/10/10, seed=42) üç script'e eklendi,
  R5 koşuları bitti; vocab=10000 kontrollü memory rerun da bitti (aşağıda Kanıt).
- **Tamamlandı (17 Kas):** Model-boyutu kontrollü memory koşusu bitti; boyut
  farkı önemsiz (Δ0.0024), kalan ~0.4 fark görev+mechanizma karışımı (Kanıt).
- **Tamamlandı (17 Kas):** Görev-sabit ablation bitti; farkın ayrımı: görev 0.31 +
  mekanizma 0.08 (Kanıt). Memory mekanizması üstünlük kanıtlayamadı (tek seed).
- **Seed tekrarı (opsiyonel):** ablation + memory koşularını 2. seed'la tekrarla;
  0.08'lik mekanizma farkı tek seed'li, istatistiksel sağlamlık ölçülmedi
- **Patent paket düzeltmesi (vekil sorusuyla bulundu, 17 Kas):** (1)
  `docs/patent/dil/TURKPATENT_basvuru_v2.3.md:2` başlığı fiilen v2.5 diyor —
  gerçek v2.3 docx/`_archive`'te; vekil paket listesiyle senkronla. (2) Üç farklı
  aralık kuralı var (QUANN motor: ödülde aralık sabit+sigma×0.82
  `quann_cloud/src/qcloud_quann.py:212-228`; classic: ödülde genişlet
  `quann_cloud/src/qcloud_weight.py:115-133`; hybrid: ödülde segmente daralt
  `:172-191`) — başvuru metnine hangi varyantın yazılacağı netleştirilmeli;
  "hata azalınca aralık daralıyor" cümlesinin kaynağı v2.3 metninde bulunamadı
- **R4 duel hükmü paper'a işlendi** (0.76, 61.94/61.19); patent dokümanlarında
  duel sayısı geçen yer varsa kontrol edilmeli
- **Fraud deploy kararı:** kullanıcı onayı bekliyor; paket `fraud/results/deploy/`

## Kanıt

### Çelişki teşhisleri (denetim turu 2, 2026-11-14)

- R1 duel token farkı: commit `e429df6` encode'a `<eos>` eklemeyi getirdi
  (`git show e429df6 -- cloudlm/tokenizer.py`); token sayısı
  `logs/run_d32_wiki200k.log:14` satırında 5149; ikinci blok
  `logs/run_d32_wiki200k.log:19` satırında 5354 — elle eklenmiş
- Eski checkpoint'ler güncel forward'la sadık değil (814a468 pos_enc refactor;
  pair_mlp girdi formülü `cloudlm/model.py:61`); güncel ölçüm
  `logs/run_r1_duel_repro2.log:3` = 26372.36 (D32),
  `logs/run_r1_duel_repro2.log:6` = 137.78 (D64); eski
  `logs/run_d32_wiki200k.log:14` = 71.09 ve `logs/run_d32_wiki200k.log:11` = 73.27
  hükmü eski kodda BİREBİR yeniden üretildi (0df8af5 worktree;
  `logs/run_r1_duel_oldcode.log:1,2`) — checkpoint sağlam, drift kod tarafında
- Loader şimşi eklendi (`cloudlm/model.py` CloudLM.load_state_dict: pos_enc→pos_enc.pe
  + attn_proj zero-fill + uyarı logları); regresyon testi
  `cloudlm/tests/test_model.py::test_legacy_learned_pos_enc_checkpoint_loads_strictly`;
  tam paket `python3 -m pytest -q` → "98 passed"
- contextcloud hedef sızıntısı: hedef token model girdisinde
  (`contextcloud/train_memory_context.py:249-263`, forward `contextcloud/train_memory_context.py:162-181`);
  stable ayrıca 10K vocab (`logs/run_contextcloud_train_stable.log:3`) → loss'lar LM metriği değil
- Sızıntı DÜZELTİLDİ: `contextcloud/task_utils.py` shift_target_2d/3d hedefi girdiden
  çıkarır; eğitim script'leri (train_stable/train_memory_context/train_global_context)
  patchlendi; regresyon testleri `contextcloud/tests/test_task_utils.py` altına eklendi;
  tam paket `python3 -m pytest -q` → "102 passed"
- R4 kuyruğu BAŞLADI: kod kilidi HEAD `logs/run_r4_full.log:2` + sha256'lar
  `logs/run_r4_full.log:4-13`; D32 eğitim satırı `logs/run_r4_full.log:16`
  (Vocab=20000, Device=mps, Epochs=50); pid 85691 (başlatma çıktısı, bu oturum)
- R4 DİSK OLAYI (15 Kas): D32 50 ep tamamlandı (train ppl 20,66 —
  `logs/run_r4_full.log:79`); D64'ün son tam epoch kaydı
  15/50 — `logs/run_r4_full.log:116`; kayıt anında disk doldu
  (`logs/run_r4_full.log:140` civarı traceback: No space left on device); kullanıcı onayıyla R2
  n20_v2/n20 epoch checkpoint'leri + R4 ara epoch'leri silindi (best/bpe/json,
  R4 d32 epoch50 ve tüm R1 dizinleri korundu); boş alan 19Gi → 123Gi
  (`df -h /System/Volumes/Data`); resume `training/shell/run_r4_resume.sh`
  15 Kas 23:00'de başladı (`logs/run_r4_resume.log` Resume satırı)
- CC global BİTTİ (16 Kas, erken durdurma epoch 22/30): best val loss —
  `logs/run_r4_contextcloud_global2.log:2296` satırında 7.7961; vocab —
  `logs/run_r4_contextcloud_global2.log:2` satırında 280151
  (ln=12.5431, hesap bu oturumda) → öğreniyor. "memory rastgele üstü" iddiası
  referans vocab yanlıştı (memory vocab 280151, ln=12.5431; 9.4801 altında)
- R4 DUEL SONUCU (güncel kod): D32 heldout_PPL 61.94 — `logs/run_r4_resume.log:67`
  ; D64 heldout_PPL 61.19 — `logs/run_r4_resume.log:70`
  ; kazanan D64, delta 0.76 — `logs/run_r4_resume.log:71`
- R5 CC AYRIK HOLDOUT (17 Kas, 80/10/10 seed=42; split üç logda da
  `Split: train 159888 / val 19986 / test 19986`): TEST loss stable 3.2318 —
  `logs/run_r5_cc_stable.log:1236`
  ; memory 9.5559 — `logs/run_r5_cc_memory.log:586`
  ; global 8.6610 — `logs/run_r5_cc_global.log:516`
  ; vocab artık train-only: 10000 / 245124 / 245124; ln(245124)=12.4095 ve
  ln(10000)=9.2103 (hesap bu oturumda) → üçü de rastgele altında, öğreniyor
- R5 MEMORY VOCAB=10000 KONTROLLÜ RERUN (17 Kas): vocab stable ile birebir aynı
  (eşitlik kontrolü bu oturumda True); TEST loss 3.6295 —
  `logs/run_r5_cc_memory_v10k.log:454`
  ; best val 3.5735 — `logs/run_r5_cc_memory_v10k.log:453`
  ; son epoch 10'da erken durdurma — `logs/run_r5_cc_memory_v10k.log:448`
  ; hüküm: 9.5559→3.6295 düşüşün baskın nedeni vocab boyutu; stable'a kalan
  ~0.4 fark model boyutu (d_model 64 vs 128) + görev farkıyla karışık
- R5 MEMORY STABLE-BOYUTU KONTROLÜ (17 Kas, d_model 128 d_sentence 256,
  vocab stable ile eşitlenmiş): TEST loss 3.6271 —
  `logs/run_r5_cc_memory_stablesz.log:894`
  ; best val 3.5578 ep 15 — `logs/run_r5_cc_memory_stablesz.log:668`
  ; küçük boyutlu koşuya (3.6295) göre Δ0.0024 → boyut farkı önemsiz; stable'a
  kalan ~0.4 görev zorluğu + mekanizma karışımı, tek başına mekanizma kusuru
  iddiası YOK (ayrım için görev-sabit ablation gerekir)
- R5 GÖREV-SABİT ABLATION (17 Kas, stable pencere/bulut mekanizması memory
  görevinde, vocab 10000, reçete memory ile birebir): TEST loss 3.5425 —
  `logs/run_r5_ablate_stable_on_memorytask.log:1332`
  ; best val 3.4802 ep 26 — `logs/run_r5_ablate_stable_on_memorytask.log:1151`
  ; ayrım: görev zorluğu 0.3107 (3.5425−3.2318) + mekanizma 0.0846
  (3.6271−3.5425) (hesap bu oturumda); memory mekanizması kendi görevinde
  pencere/bulut'tan 0.08 kötü (tek seed) — çürük değil, üstün de değil
- CC sızdırmasız görev sonuçları: stable Best val loss 3.1327 —
  `logs/run_r4_contextcloud_stable.log:1532`
  ; memory Best val loss 9.4801 — `logs/run_r4_contextcloud_memory.log:550`
  ; ln(10000)=9.21 rastgele seviyesinin üstünde — neden doğrulanmadı, inceleme açık
  ; global ilk denemede MPS nested-tensor op eksikliğiyle öldü
  (`logs/run_r4_contextcloud_global.log` traceback); düzeltme
  `contextcloud/train_global_context.py:34` enable_nested_tensor=False;
  yeniden koşu CANLI — pid 33076 (başlatma çıktısı, bu oturum)
- Fraud "balanced" düzeltildi: `fraud/results/full_scale_hybrid.json:6` = 130 fraud,
  `fraud/results/full_scale_hybrid.json:7` = base_rate 0.0013; kesme
  `fraud/src/train_quann.py:85-93`; README kod hakemliğinde düzeltildi
- XOR: `quann_cloud/logs/xor_gate.json:5-12` içinde quann_acc 100.0; LLM'de
  `quann_cloud/logs/full_run.log:46` = 1016.30, random
  `quann_cloud/logs/full_run.log:73` = 1045.03 — FINDINGS atfı düzeltildi

### Uzun Metin Testi — kaynak: `contextcloud/long_text_results.txt:11,15` (örneklem sayısı aynı dosyanın satır 2'sinde)

| Model | Avg Loss | Avg PPL |
|-------|----------|---------|
| CloudLM | 12.5740 | 294969.40 |
| PureCloudLM | 12.5657 | 291931.32 |
| ContextCloudLM | 12.5508 | 285467.06 |

ContextCloudLM vs CloudLM: PPL farkı 294969.40 − 285467.06 = 9502.34 → %3,22. Bu ölçüm eğitim korpusundan olduğu için held-out değildir; PPL ~290K mutlak değeri model/eval uyumsuzluğuna işaret edebilir (sebep doğrulanmadı).

### Dosyalar

- `contextcloud/model.py` — ContextCloudLM + generate metodu
- `contextcloud/train_simple.py` — eğitim scripti (çalışmıyor)
- `contextcloud/compare_models.py`, `contextcloud/test_long_text.py` — kıyaslama araçları
- `contextcloud/long_text_results.txt` — uzun metin ölçümü (yukarıdaki tablonun kaynağı)
- `contextcloud/tests/test_model.py` — test dosyası (artık testpaths'te)

### Kod Kapısı

- `python3 ~/.config/opencode/denetim/denetim_kos.py hepsi --repo .` koş; son kayıt `~/.local/share/opencode/denetim/` altındaki JSON'da

### Patent İşleri

- İstem linter'ı: `patent_patent_lint` → `patent-cikti/basvuru_paketi/istemler_final.md`
- PDF üretimi: `python3 patent-cikti/basvuru_paketi/build_package.py`
- Sekiz siyah-beyaz şekil, A4 PDF: `pdfinfo patent-cikti/basvuru_paketi/sekiller.pdf`
