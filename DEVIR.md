---
id: devir
baslik: DEVIR — quann-3d (makine üretimi durum + sıradaki iş)
konu: [devir, handover, durum]
durum: güncel
sahiplik: koordinator
guncellendi: 2026-11-22
kunye-kaynak: otomatik (devir_yaz.py)
---

# DEVIR — quann-3d

<!-- MAKINE:BASLA -->
## Durum (makine üretti · 2026-11-22 22:44)
Bu blok elle düzenlenmez; `devir_yaz.py --repo /Users/serkankose/Documents/quann-3d` ile yenilenir.

- HEAD: 49848a3 kara_kutu: doğrulayıcı bulguları 4+6 kapandı
- Değişen dosya: 12 → DEVIR.md, docs/KARA_KUTU_PLAN.md, kara_kutu/adapters/fraud.py, kara_kutu/report/internal.py, kara_kutu/tests/test_contract.py
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

- **Kara Kutu P0+P1 commit'te; Karar A1 KAPANDI (22 Kas, ölçüm):** kutu-yerel eşik = Ekim alarm
  bütçesinin quantili 0,8566042184829712 (`kara_kutu/artifacts/esik_fraud.json`; üretici `kara_kutu/`
  `tools/esik_kalibre.py`). Final te geçişi (4.932.768 satır; --hizli smoke 193423 PLAN §10'da):
  alarm 12.368=bütçe, TP 1.265, P 0,10228, R 0,204659 (doğrulama kaynağı: `kara_kutu/artifacts/esik_fraud.json` — runs/esik_te_20261122-193451/ozet.json 2026-11-23 kazara silindi; DOĞRULANMADI) ↔ Ekim FP32 TP 1.842, FP 9.942, alarm 11.784 (`fraud/results/false_alarm_audit.json:11-13`) — TP farkının mekanizması DOĞRULANMADI; ayrıntı `docs/KARA_KUTU_PLAN.md` §10. 24 Kas: eğitim-fold kalibrasyonu yeniden koştu (`kara_kutu/runs/esik_kalibre_20261124-063247/ozet.json`; değerler birebir aynı; artefakt `kanit_kosum` canlı yola bağlandı; 2 test skip kalır).
  Web: `python3 -m kara_kutu web --port 8734`; browser turu `runs/20261121-234528_fraud` — 2026-11-23 kazara silindi; DOĞRULANMADI (web turu artefaktı repo'da yok).
- **Maintenance retrain (23 Kas, cbe6609):** gerçek checkpoint + edge paketi (test AUC 0.9908 @ rerun_2026-11-23.json:229; val 0.9896 @218; eşik 0.2379 @217; synthetic /tmp yedek); bağımsız tester turu TAMAM (2026-11-23): 1–4 & 6 ✓, 5 elle ✓ (tool vacuous)
- **P2 complaints adaptörü depoda (22 Kas, teslim + kapılar):** `kara_kutu/adapters/complaints.py`
  + `kara_kutu/tests/test_complaints.py`; issue top-1 0,3900 ↔ çoğunluk tavanı 0,32 = 64/200 (`docs/complaints_tavan_olcum_2026-11-22.txt:1-3`), dengeli 0,0701 (runs/20261122-091904_complaints/ic_rapor.md 2026-11-23 kazara silindi; DOĞRULANMADI — değer kalıntı, artefakt yok) — niş eval; SOTA iddiası yok.
  Geçici önlem: iç rapor fraud-string'leri adaptör-koşullu (`kara_kutu/report/internal.py`);
  genelleştirme kartı beklemede: `contract.Gerekce.yon`, `runner` ETIKET_KOLON/fraud_sayisi,
  external HTML "FRAUD", `__main__` özet. Sıradaki: aml; ayrıntı PLAN §10-§11.
- Borç kartı: DEVIR §Kanıt eski PCT satırlarında çıplak-dosya atıfları kalıyor (rapor_denetle
  ölü adres üretiyor) — gerçek yollar `patent-cikti/pct/pct_v3-pct-inceleme/` ve `docs/patent/pct_v2/` altında; düzelt: tam yol verme.
- **PCT v3 paketi hazır (20 Kas, commit YOK):** `docs/patent/pct_v3/filing/` 7 dosya (EN/TR istem+
  tarifname+özet + 8 sayfa A4 PDF 979834 B) + 8 SVG. Sayaçlar: 15 istem/3 bağımsız · 98 paragraf
  EN=TR · özet 148/132 kelime · kapsama 56/56 (hepsi `patent-cikti/pct/pct_v3-pct-inceleme/
  dogrulama.txt:2,3,10-12,14,17,18`). Motor: B 43→41 · C 47→17 `compliant true` coverage 100
  (`motor_2026-11-20/*_v3.json`). Atıf kapısı 8 dosya: doğru 82 · kaynakta yok 0 · ölü adres 0.
  **Bağımsız `reviewer` turu KOŞTU (lokal GGUF): 7 ✓ / 1 ✗ / 0 KANITSIZ** (`dogrulayici_tur_2026-11-20.txt`).
  Tek ✗ = v2 sha1 künyesi: başka oturum v2 tabanını akşam 84 paragraflık adayla değiştirdi → kullanıcı
  kararı (a) KABUL: yeni künye `dogrulama_v2kabul_2026-11-20.txt` + 4 defter + ölçüm defteri + DEVIR güncellendi
  (v3 sha `4ef0d32a` değişmedi). D'nin 2 CRITICAL'ı (ABD) kullanıcı kararıyla açık; `ALTERNATIF_ISTEM_KUMESI.md` + `lint_v3.txt:14-24` vekile; >30 sayfa ücreti DOĞRULANMADI.
- **PCT v2 motor denetimi (20 Kas):** 15 ham JSON + `INCELEME_RAPORU_2026-11-20.md` +
  `KAPI_OUT_2026-11-20.txt` (`patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/`);
  doğrulayıcı tur 8 ✓. Öne çıkan: US `ready_to_file false` (başlık), 47 yazılı-tanım izi, opt-in 43 LOW.
- **PCT v2 paketi (19 Kas, untracked):** 15 istem, 70 paragraf, `filing/` 6 kopya; defterler
  `UYGUNLUK_NOTU.md` + `OLCUM_KAYITLARI.md`; kapı `kapilar.sh` çıkış 0.
- **PCT v2 anomali (kaynak doğrulanmadı):** §1 matrisi 12:09'da kuruldu, 12:14 yazımında 9 satıra
döndü; hangi sürecin yazdığını kanıtlayamadık. Onarım üretime çevrildi: `yeniden_bolum1.py` (`patent-cikti/pct/pct_v2-pct-inceleme/`) her künyeyi ham kaynakta açar, tutmazsa yazmaz.
- **PCT açık işler (vekil + insan):** DB taraması 3. turda yapıldı (`docs/patent/pct_v2/ON_SANAT.md:185`,
  arşiv `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/`); resmî CPC/Espacenet/PATENTSCOPE, atıf
  ağları, X/Y/P, fiziksel Rule 11.2–11.16, PCT/RO/101, ücret/vekaletname DENETLENMEDİ. Vekil olmadan başvuru yok.
- **Kapı istisnası (kullanıcı kararı):** `patent-cikti/` (`.gitignore`'da üretilmiş
  çıktı) `scripts/check_docs_lines.sh`'den tek satırlık istisnayla çıkarıldı; kapı yeşil.
- **Patent final_v3 paketi (18 Kas, commit `1afc688`):** `docs/patent/final_v3/` — ISTEMLER (15/3) +
  TARIFNAME (237 satır) + OZET + TEKNIK_ETKI + 8 SVG + 5 DOCX/5 PDF; reviewer 71 sayı-atıf ✓.
  Eski `patent-cikti/basvuru_paketi` ve `docs/patent/dil` yalnız referans.
- **Vekil kararları (v3, kullanıcı tarafı):** özet uzunluğu (265 kelime), 3 bağımsız
  istemin birliği, iki-bölümlü kalıp tercihi — `BLUS_NOTU.md` kırmızı çizgileriyle
  birlikte vekile götürülecek (vekil raporu §d).
- **Push BLOCKED — kullanıcı kararı "şimdilik push yok" (18 Kas):** refactor (`1f21cec`…`64555cc`) +
  v3 paket commit'leri `deneme_context_cloud` dalında yerel. Push GitHub 100 MB tek-dosya sınırına
  takıldı: geçmişte `quann-3d-backup-20261028.tar.gz` = 1.891.553.280 B ≈ 1,76 GiB (commit `e33deb7`;
  `git rev-list --objects origin/main..HEAD` — 1238 nesne, 10 MB üstü TEK dosya bu; kalan ~10 MB).
  (a) geçmiş yeniden yazma (güvenli, diskteki dosyaya dokunmaz) kullanıcıca REDDEDİLDİ; (b) Git LFS.
- **Tamamlandı (17-18 Kas):** CC ayrık holdout (80/10/10, seed=42) üç script'e eklendi; model-boyutu
  kontrollü memory (Δ0.0024 → boyut önemsiz) + görev-sabit ablation (görev 0.31 + mekanizma 0.08) bitti —
  memory üstünlüğü kanıtlanamadı. seed=43 kuyruğu tekrarladı (commit `51fa24a`); ham TEST: 3.6336
  (`logs/run_r5_seed2_memory_stablesz_s43.log:631`) ↔ 3.5581 (`logs/run_r5_seed2_ablate_s43.log:982`) (Kanıt).
- **Patent paket düzeltmesi (vekil sorusuyla bulundu, 17 Kas):** (1)
  `docs/patent/dil/TURKPATENT_basvuru_v2.3.md:2` başlığı fiilen v2.5 diyor — gerçek v2.3 docx/`_archive`'te;
  vekil paket listesiyle senkronla. (2) Üç aralık kuralı: QUANN `quann_cloud/src/qcloud_quann.py:212-228`
  sabit+sigma×0.82 · classic `quann_cloud/src/qcloud_weight.py:115-133` genişlet · hybrid `:172-191` daralt —
  hangisinin yazılacağı net; "hata azalınca aralık daralıyor" cümlesinin kaynağı v2.3'te bulunamadı
- **R4 duel hükmü paper'a işlendi** (0.76, 61.94/61.19 — Kanıt); patent dokümanlarında duel sayısı
  geçen yer kontrol edilmeli. **Fraud deploy tamam (kullanıcı onayı, commit `a52ddc1`):** paket + parity doğrulandı, tester 5/5 ✓. **R8 tamam (19 Kas):** pair-freeze 5233.4086 (`logs/run_r8_pair_freeze_only.log:12`) vs kontrol 19770.644 (`logs/run_r8_quann_pair.log:30`), tester 6/6 ✓; tekrar adayı kullanıcı kararı.

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
  `logs/run_r5_cc_stable.log:1236`; memory 9.5559 — `logs/run_r5_cc_memory.log:586`; global
  8.6610 — `logs/run_r5_cc_global.log:516`
  ; vocab artık train-only: 10000 / 245124 / 245124; ln(245124)=12.4095 ve
  ln(10000)=9.2103 (hesap bu oturumda) → üçü de rastgele altında, öğreniyor
- R5 MEMORY VOCAB=10000 KONTROLLÜ RERUN (17 Kas): vocab stable ile birebir aynı (eşitlik
  kontrolü bu oturumda True); TEST loss 3.6295 — `logs/run_r5_cc_memory_v10k.log:454`; best val
  3.5735 — `logs/run_r5_cc_memory_v10k.log:453`; son epoch 10'da erken durdurma —
  `logs/run_r5_cc_memory_v10k.log:448`
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
- R5 SEED=43 TEKRARI (18 Kas, bağımsız ikinci örnek; commit `51fa24a`, kod kilidi `logs/run_r5_seed2.log:2-6`): TEST ablation 3.5581 — `logs/run_r5_seed2_ablate_s43.log:982`; memory-stablesz 3.6336 — `logs/run_r5_seed2_memory_stablesz_s43.log:631` → mekanizma farkı 0.0755 ↔ orijinal 0.0846 (yön+büyüklük korundu); stable kontrolü 3.2131 — `logs/run_r5_seed2_cc_stable_s43.log:1237` vs 3.2318 (Δ0.0187); orijinal checkpoint'ler korundu (yalnız `*_s43_best.pt` yazıldı); tekrarlanan epoch-ort. train-loss sıçraması iki koşuda da var (s43 5.66/15.26 — `logs/run_r5_seed2_cc_stable_s43.log:619,1027`; orijinal 8.11 — `logs/run_r5_cc_stable.log:822`), best ckpt ikisinde de ep 1, TEST'e yansımadı, sebep doğrulanmadı. Bağımsız tester turu: 8/8 ✓
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

### Kod Kapısı

- `python3 ~/.config/opencode/denetim/denetim_kos.py hepsi --repo .` koş; son kayıt `~/.local/share/opencode/denetim/` altındaki JSON'da

### Patent İşleri

- PCT v2 kapısı (9 kapı, 6 KARAR): `bash patent-cikti/pct/pct_v2-pct-inceleme/kapilar.sh`
  → `kapilar_taze.txt:51` `SONUC: yeşil` · `:52` `CIKIS=0` (koşum 2026-11-19 23:48); rapor kapısı
  `2` = yalnız uyarı (`rapor_kapisi.txt:26`: `kaynakta yok 0 · ölü adres 0`). Doğrulayıcı turu 3
  kapanışı `27 ✓ / 3 ✗ / 1 KANITSIZ` → üçü düzeltildi, KANITSIZ taze koşumla kapandı
  (`patent-cikti/pct/pct_v2-pct-inceleme/KAPI_KARARI_2026-11-19.md` son bölüm). Vekilde: RO/101+ücret.
