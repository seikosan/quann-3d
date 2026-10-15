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
- En doğrudan müdahale. Point/pair MLP'ler D ile ölçeklenir (maliyet ~2-4×).
- **Test:** 20K, 10ep, D=64 ve D=128 → held-out PPL + Transformer'a oran.
- Beklenti: 6.45 → ~4-5 (D=64), ~3-4 (D=128)? 2.94'ü geçebilir mi?

### R2 — Cloud Attention (ilişkisel sözlük genişletme)
- CloudEncoderND'ye, pooling öncesi nokta-üstü self-attention (2-4 head) ekle.
- Pair MLP'nin kaba (dist, cos, norm) sözlüğü yerine tam dikkat.
- Permutation-invariance korunur (attention + pool).
- **Test:** 20K, D=32 (kontrolle) → PPL + hassasiyet analizi tekrar.

### R3 — Attention-Weighted Pooling
- max-pool (sessiz branch) yerine öğrenilmiş attention-pool.
- Düzeltme: tek nokta etkisi max-pool'da argmax'a bağlı (0/1); attention'da
  yumuşak ve sürekli → hem hassasiyet hem gradyan/arama sinyali artar.
- **Test:** 20K, D=32.

### R4 (QUANN özel) — Aramayı pair-kanalına yönlendir
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
- [ ] Ölçek kuyruğu: hibrit 200K → CloudLM 1.5M (çalışıyor)
- [ ] R1-R4 ön-testleri (kuyruğa eklenecek)
