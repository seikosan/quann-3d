---
id: devir
baslik: DEVIR — quann-3d (makine üretimi durum + sıradaki iş)
konu: [devir, handover, durum]
durum: güncel
sahiplik: koordinator
guncellendi: 2026-11-16
kunye-kaynak: otomatik (devir_yaz.py)
---

# DEVIR — quann-3d

<!-- MAKINE:BASLA -->
## Durum (makine üretti · 2026-11-16 22:00)
Bu blok elle düzenlenmez; `devir_yaz.py --repo /Users/serkankose/Documents/quann-3d` ile yenilenir.

- HEAD: 3efcdd6 Patent docs: lint-clean TR+EN (locale-aware linter fixes); archive stale v2.3_EN
- Değişen dosya: 44 → .gitignore, DEVIR.md, RESEARCH.md, test_model.py -> _archive/test_model.py, cloudlm/model.py
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

- **Patent tarafı kapandı (16 Kas):** 6 güncel doküman (3 TR + 3 EN) `patent_lint`
  0 hata · 0 uyarı; commit `3efcdd6` (bu depo) + `24e1d62` (patent-hatti linter
  dil-desteği). Açık patent işi yok; push bekliyor.
- **CC global yeniden koşusunu izle:** CANLI pid 33076 (başlatma çıktısı, bu
  oturum); log `logs/run_r4_contextcloud_global2.log`; bitince val loss'u
  RESEARCH.md'ye işle
- **CC memory val loss 9.4801 = ln(10000)=9.21 rastgele seviyesinin üstünde**
  (`logs/run_r4_contextcloud_memory.log:550`) — kök neden incelemesi gerekli
  (sebep doğrulanmadı; uydurma açıklama yazılmayacak)
- **R4 duel hükmünü paper/patent notlarına yansıt:** D64 61.19 < D32 61.94
  (`logs/run_r4_resume.log:71`)
- **R4 sonrası hüküm:** D32/D64 karşılaştırması R4 eval'iyle doğrulandı
  (eski sayılar yalnız eski kodda geçerli — aşağıdaki teşhis)
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
- R4 DUEL SONUCU (güncel kod): D32 heldout_PPL 61.94 — `logs/run_r4_resume.log:67`
  ; D64 heldout_PPL 61.19 — `logs/run_r4_resume.log:70`
  ; kazanan D64, delta 0.76 — `logs/run_r4_resume.log:71`
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
