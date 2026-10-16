# RESEARCH.md — Quann-3D Tek Durum Kaynağı

> **Kurallar:** Her oturum BAŞINDA okunur. Bu dosyanın dışında ağır içerik
> (logs/, checkpoints/, kod) ana oturuma gelmez — domain agent'ları (`experiment`,
> `geometry`, `paper`) okur, distile sonuç döndürür. Güncelleme protokolü:
> `AGENTS.md` §5. Boyut bütçesi: **≤150 satır** — taşarsa en eski deney "Son
> Deneyler" tablosundan çıkarılır (detay `logs/`'da kalır).
> Son güncelleme: 2026-10-16 (aml/E1, maintenance/E0b)

## Bir Cümle

Transformer'ın "sıralı dikkat + tek vektör" paradigmasının yerine: kelime → ND
uzayda nokta, cümle → nokta bulutu, anlam → bulut geometrisi; tahmin bu
konfigürasyondan. İkinci katman QUANN-Cloud: aynı mimariyi backprop'suz (0th-order)
eğiten yöntem.

## Canlı Runlar (2026-10-16 07:54)

| Run | Komut | Durum | Log |
|---|---|---|---|
| CloudLM 1.5M (ölçek) | `train_wiki.py --corpus turkish_wiki_1500k --d-dim 32 --epochs 20 --resume` | canlı, epoch 5/20; son train PPL 129.55; RSS 58,141,552 KiB, swap yalnızca ~1.3 GB boş | `logs/run_final_pipeline.log` (epoch/PPL); `logs/run_wiki1500k_resume_nohup.log` (yalnız başlık) |

**Teşhis:** Epoch/PPL aslında kaydedilmiş; resume çıktısı `run_final_pipeline.log`'a yönlenmiş, nohup logunda yalnızca başlatma başlığı var. RSS artışı ~4.5 GB/saat; swap baskısı yüksek. `train_wiki.py` her epoch'ta tüm token pencerelerinden dev `examples` listesi kurup shuffle ediyor; RSS artışının muhtemel nedeni bu geçici tahsislerin allocator'da tutulması (kanıtlanmış kalıcı Python referans sızıntısı değil).

Doğrula: `ps aux | grep -E "train_wiki|train_baseline" | grep -v grep`

## Mevcut Best (ölçülmüş, ortak held-out 2000 cümle)

| Model | Config | Param | Held-out PPL | Checkpoint |
|---|---|---|---|---|
| CloudLM | D=32, 200K, 50ep | 21.0M | **6.45** | `checkpoints/wiki_checkpoints_200k/wiki_cloud_best.pt` |
| Causal baseline | d=384, 4L, 200K, 50ep | 22.7M | **2.92** | `checkpoints/baseline_200k/baseline_best.pt` |

**Kök teşhis (sensitivity_analysis, ölçülmüş):** kapasite (D boyutu) —
hassasiyet değil. Bulut (10×32=320 sayı), transformer gizli durumunun
(10×256=2560) %12.5'ini taşıyor. Detay: `quann_cloud/REVISION.md`.

## Aktif Hipotezler (R1–R4 — `quann_cloud/REVISION.md`)

| # | Hipotez | Öncelik | Durum |
|---|---|---|---|
| R1 | Kapasite: D=64 / D=128 | 1 | **sonuçlandı/değerlendirme** — 20K held-out: baseline 10.80, D=32 201.18, D=64 181.47, D=128 189.47. D=64 iyileşti, D=128 geri gitti; baseline farkı çok büyük, 200K'a ölçekleme yok. |
| R2 | Cloud attention (ilişkisel sözlük genişletme) | 3 | bekliyor |
| R3 | Attention-weighted pooling (max-pool onarımı) | 2 | bekliyor |
| R4 | QUANN: aramayı pair-kanalına yönlendir (freeze varyantı) | 4 | bekliyor |

## Son Deneyler (en yeni üstte)

| Tarih | Run | Sonuç | Ders | Log |
|---|---|---|---|---|
| 10-16 | aml/E1 (bölge haritası + prototip-mesafe → SAR demosu, 2K pencere, CPU) | prototip-mesafe ayrımı: structuring .81/11.5, velocity .59/10.1, normal .019/.027, layering .030/.040 (normalle üst üste); en-yakın-prototip doğruluk %87.8 (400 held-out), argmax %72.8 | bölge haritası açıklanabilir: structuring/velocity net adlandırılıyor; layering prototipi normalle örtüşüyor ama gerekçe metni gözlenen zinciri (10 hop) yine adlandırıyor → E3 çift-kanal şart | `aml/logs/run_e1_smoke.log` |
| 10-16 | maintenance/E0b denemesi (gerçek C-MAPSS FD001) | 5 public kaynak erişilemedi (NASA→HTML, 3×GitHub 404, S3 403) — E0b koşmadı; sentetik E0 sonucu ayakta | gerçek veri = kapı: Kaggle token veya partner veri kopyası gerekli | `maintenance/logs/run_e0b_smoke.log` |
| 10-16 | aml/E0 (sentezik tipoloji, 20K işlem / 2K pencere, CPU) | hybrid AUC-ROC .960 / AP .872 (F2 .938; 65/616 nokta yakınsadı); standart tavan 1.000'ye ulaşmadı (kolay sentetik); sınıf recall @1% FPR: structuring .99, velocity .95, layering .04, normal .07 | structuring+velocity pencere özelliklerinde görünür; layering (çok-hop zincir) işlem-bazı özelliklerden görünmez → çift-kanal/zincir tasarımı gerekli | `aml/logs/run_e0_smoke.log` |
| 10-16 | maintenance/E0 (sentezik turbofan fallback — C-MAPSS'a ulaşılamadı; 48 motor / 2686 pencere, CPU) | ikili RUL≤30: hybrid .947/.831 (F2 .843; 18/24) vs tavan MLP .977/.944; evre: hybrid top-1 .515 vs .584 | hybrid standart tavanın altında ama pure'e (.693/.407) belirgin üstün; gerçek kapı C-MAPSS/CWRU veri erişimi | `maintenance/logs/run_e0_smoke.log` |

## Sıradaki Aksiyonlar

1. 1.5M CloudLM run'ı canlı: RSS/swap + log takibi, müdahale yok; bitince held-out PPL ölç.
2. aml/E3: E1'deki prototip örtüşmesi + gerekçe-adlandırma bulgusuyla layering için çift-kanal/zincir özellikleri tasarla.
3. maintenance: Kaggle token / partner veri kopyası kullanıcı kararı; yoksa CWRU ayna denemesi.
4. Canlı run varken yeni ağır run başlatma (MPS).

## Subproje: quann_cloud/

Kendi durum dosyaları var — ana oturum bunları okumaz, `geometry`/`experiment`
agent'ı okur: `quann_cloud/REVISION.md` (teşhis+adaylar), `FINDINGS.md`
(dürüst bulgular), `EXPERIMENTS.md`, `PROJECT_SUMMARY.md`.
Tek cümle: QUANN 0th-order XOR'da %100, gerçek LLM'de random'a yakın → savunulabilir
iddia "küçük/yorumlanabilir modeller için backprop'suz eğitim".

## Subproje: fraud/

IBM kredi kartı fraud verisi (24.4M satır, `~/Downloads/archive`) üzerinde iki
katman: (1) standart ML baselines (tam veri: LR/RF/GBT/MLP); (2) QUANN nokta-bulut
(özellik→nokta, tx→bulut, CloudEncoderND geometri) 0th-order vs backprop.
**Bulgu:** hybrid (QUANN nokta + backprop okuyucu) AUC-ROC 0.9745 ile standart
tavanını (RF .973/GBT .981) yakalıyor; pure 0th-order ≈random → QUANN küçük/
yorumlanabilir görevde iş yapar. Detay `fraud/README.md` + `fraud/results/`.

## Market Öncelikleri

`docs/market_priorities.md`: Öncelik 3 dikeyleri (AML tipolojisi, işlem
fraud/edge, endüstriyel bakım) + 10 ikincil alan. Kazanma mantığı =
açıklanabilirlik + uyum hızı + boyut, ham doğruluk değil. Kanıt zemini:
`fraud/` (hybrid AUC-ROC 0.9745).

## Subproje: aml/, maintenance/

2026-10-16 açıldı (ticari gerekçe `docs/market_priorities.md`): `aml/` =
tipoloji tespiti (sentezik structuring/layering/velocity üreteci önce, sonra
Elliptic); `maintenance/` = C-MAPSS/CWRU (pencere→bulut). Deney kuyruğu +
eşlemeler README'lerde; run'lar CPU (MPS: 1.5M).

## Dondurulmuş Kurallar

`AGENTS.md` §3'e bak: `state_dict` ana adları, checkpoint adlandırması,
`logs/` append-only, `_archive/` yolu.
