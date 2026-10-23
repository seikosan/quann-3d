# CloudLM Mimari Revizyon — Teşhis ve Adaylar (15 Ekim)

## Teşhis (ölçülmüş — sensitivity_analysis.py)

CloudLM 200K (D=32, 50ep) üzerinde:

| Ölçüm | Değer | Yorum |
|---|---|---|
| Tek nokta ΔCE (ε=0.5) | ort +0.066, \|Δ\| 0.189, p95 0.662 | Manzara DÜZ DEĞİL — model hassas |
| Δcloud_state | ortalama %7 | Belirgin ama sınırlı |
| Top-1 flip | 13.4% | Noktalar hâlâ belirleyici |
| max-pool branch Δ | 0.070 | Sessiz taraf |
| pair-pool branch Δ | 0.097 | **Baskan kanal (1.39×)** |
| Kapasite | 10×32=320 sayı vs 10×256=2560 | **Bulut, transformer'ın %12.5'i bilgi taşıyor** |

**Kök sorun:** D boyutu (kapasite), hassasiyet değil. 200K'da Transformer'ın
2.94'e inmesinin CloudLM'in 6.45'te kalmasının ana nedeni, 16/32D nokta
şişesinin 256D gizli durumun taşıyabildiğinin çok azını taşıması.

İkincil bulgular:
- QUANN hibritinde arama no-op (Δ≈0) — head, noktaları "emerek" manzarayı
  düzleştiriyor (ko-adaptasyon). Aramanın sinyali, eğitilen head'in
  robustluğu tarafından yutuluyor.
- Pair-branch baskın → ilişkinsel sözlük (mesafe/açı/norm) tek nokta
  etkisinin ana kanalı.

## Aday Revizyonlar (20K ön-test, sonra 200K)

### R1 — Kapasite: D=64 / D=128
**Hat: Makale hattı.** Ürün D=8–32'de kalır; küçük D ürün özelliğidir,
tavan ve okunurluk kapasite deneyi makale hattındadır.
- En doğrudan müdahale. Point/pair MLP'ler D ile ölçeklenir (maliyet ~2-4×).
- **Test:** 20K, 10ep, D=64 ve D=128 → held-out PPL + Transformer'a oran.
- Beklenti: 6.45 → ~4-5 (D=64), ~3-4 (D=128)? 2.94'ü geçebilir mi?
- **ARA SONUÇ (2026-10-21, D=64×200K-wiki/50ep, `logs/run_d64_200k.log`):**
  - D=64: train **21.20**; slice (ilk-200K, in-sample) **19.08**; full-1.5M pool **59.32** (kısmi overlap). Train≈slice → kendi dağıtımında overfit yok.
  - 2621.31 (`R1_MATCHED_645`) **korpus OOD artefaktı**: 6.45 referansı `turkish_sentences.txt`'de, D=64 wiki'de eğitildi. Protokolün D32 ayağı 6.45'i birebir repro'ladı (validasyon) — D64 ayağı kapasite sinyali taşımıyor.
  - Aynı-pool 19.08 vs 24.53 + 1.5M 59.32 vs 218.84 yönsel olarak D=64 lehine ama (a) in-sample, (b) eski D32 kontrolü farklı BPE → **tek başına hüküm verilmez**.
  - **FİNAL hüküm** = eşleşen duel: yeni D32 (aynı shared BPE `wiki_bpe.json`, aynı 200K-wiki) `logs/run_d32_wiki200k.log` içinde disjoint [200000,202000) holdout ile `DUEL_SUMMARY` üretecek (eğitim 6/50ep, ~30dk/ep → bitiş ~yarın akşam). Sonra 2.94 (causal baseline) kıyası ve D=128 adımı.

### R2 — Cloud Attention (ilişkisel sözlük genişletme)
**Hat: Araştırma.**
- CloudEncoderND'ye, pooling öncesi nokta-üstü self-attention (2-4 head) ekle.
- Pair MLP'nin kaba (dist, cos, norm) sözlüğü yerine tam dikkat.
- Permutation-invariance korunur (attention + pool).
- **Test:** 20K, D=32 (kontrolle) → PPL + hassasiyet analizi tekrar.

### R3 — Attention-Weighted Pooling
**Hat: Ürün + makale kesişimi.**
- max-pool (sessiz branch) yerine öğrenilmiş attention-pool.
- Düzeltme: tek nokta etkisi max-pool'da argmax'a bağlı (0/1); attention'da
  yumuşak ve sürekli → hem hassasiyet hem gradyan/arama sinyali artar.
- **Test:** 20K, D=32.
- **Durum:** 20K duel tamamlandı (2026-10-23) — MAX 315.99 > ATTN 334.89 (Δ18.90); attention-pool performansı düşürdü. 200K'da tekrar test ediliyor (`training/shell/run_r3_200k.sh`).
- **Teşhis:** Attention pooling 20K'da overfitting'e yol açtı (train PPL 124.22 vs MAX 88.19). 200K'da tekrar test ediliyor — küçük korpusta attention sinyali gürültüye dönüşüyor olabilir.
- **Kuyruk:** `bash training/shell/run_r3_200k.sh` (max → attn, MPS ardışık); checkpoint'ler `checkpoints/r3_200k/{max,attn}/`.

### R4 (QUANN özel) — Aramayı pair-kanalına yönlendir
**Hat: Araştırma.**
- Aramanın no-op olmasının nedeni head'in ko-adaptasyonu. İki seçenek:
  a) Arama sırasında head'i dondur (freeze) → manzara düz kalmaz,
     noktalar gerçekten öğrenir; sonra head'i fine-tune et.
  b) Arama yönünü pair-feature değişimine göre seç (koordinat yerine
     ilişki uzayında arama).
- **Test:** 20K, hibrit + freeze varyantı → Δcand ve held-out PPL.

## Öncelik

1. **R1 (D=64)** — en ucuz, en doğrudan; kapasite hipotezini doğrular.
2. **R3 (attention-pool)** — küçük değişiklik, max-pool kırıklığını onarır.
3. **R2 (cloud attention)** — en güçlü ama en pahalı.
4. **R4 (QUANN freeze)** — QUANN'ın gerçek katkısını ölçer.

Her varyant: 20K × 10ep (~2-4 saat) → vaat edeni 200K'a ölçekle.

## Durum

- [x] Hassasiyet analizi (bu dosya)
- [x] Ölçek kuyruğu: 1.5M OOM/kill teşhisi + `train_wiki.py` bellek düzeltmesi + resume 18–20 + 200K-wiki/50ep kontrol + eş dağıtım eval; on-wiki 62.15, aynı 200K kesitinde 200K/50ep 24.53 vs 1.5M/20ep 64.28
- [x] R1 20K ön-test: D=64 en iyi (181 vs 201); wiki scale gate'inde D=32'de tekrar derinliği baskın
- [x] R1 D=64×200K-wiki/50ep TAMAM (2026-10-21) + eşleşen D32 kontrolü; **duel (disjoint holdout, aynı BPE) tamamlandı — D64 71.09 < D32 73.27 (Δ2.18)**
- [x] R3 attention-weighted pooling 20K TAMAM (2026-10-23) — 20K duel: MAX 315.99 > ATTN 334.89 (Δ18.90); attention-pool performansı düşürdü
- [ ] R3 200K tekrar test ediliyor — `training/shell/run_r3_200k.sh` (max → attn, MPS ardışık)
- [ ] R2 cloud attention, R4 QUANN freeze ön-testleri
