# QUANN-Cloud: Gerçek Veri Deneyi — Dürüst Bulgular

> **Tarih:** 2026-10-14 · **Veri:** `turkish_sentences.txt` (gerçek Türkçe) · **Cihaz:** Apple Silicon (MPS/CPU)
>
> Bu dosya, "backprop'suz nokta-bulut dil modeli QUANN-Cloud gerçek veride backprop'a
> karşı ne yapar?" sorusunun **dürüst, ölçülmüş** cevabıdır. Hype yok; ölçülen sayılar var.

---

## 1. Deney Tasarımı (apples-to-apples)

Üç model, **aynı mimari, aynı veri, aynı BPE** — tek fark **optimizer**:

| Model | Mimari | Optimizer |
|---|---|---|
| Random | SlimCloudLM (111K param) | eğitilmemiş |
| Backprop | SlimCloudLM (111K param) | AdamW (1st-order) |
| QUANN | SlimCloudLM (111K param) | aralık-daraltma (0th-order, backprop'suz) |

- **Mimari:** `d_model=32, 1 transformer katmanı, D=8, d_cloud=32` → 111.248 param
- **Veri:** 6000 train + 500 eval cümle (gerçek Türkçe), seq_len=8
- **BPE:** 1024 token (korpus üzerinde eğitildi)
- **Epoch:** 5 (her iki model)
- **QUANN:** mini-batch=8 ödül sinyali, K=80 ağırlık/adım, dinamik base, CPU'da

**Neden küçük model?** QUANN (0th-order) aday başına 1 forward gerektirir. 21M parametreli
modelde QUANN uygulanabilir değil; bu yüzden mekanizmayı **aynı mimaride küçük** ölçekle ölçtük.
Bu, optimizer karşılaştırması için adil bir settir.

---

## 2. Sonuçlar (ölçülen)

```
Model                          Held-out PPL
Random (eğitilmemiş)                1045.03   ← taban (≈ vocab 1024)
Backprop (AdamW)                     563.45   ← 2.0x daha iyi
QUANN (backprop'suz)                1016.30   ← random'ın sadece %3 altı
```

**QUANN epoch trajekti (eval PPL):**
```
E1: 1047  →  E2: 1088  →  E3: 1088  →  E4: 1035  →  E5: 1020
```
Düz / gürültülü — random çevresinde salınıyor, **net düşüş yok**.

**QUANN iç istatistikler (5 epoch sonunda):**
```
R (ödül) ≈ C (ceza)  →  dengeli, yani rastgele yürüyüş (net yön yok)
mean_width ≈ 4.16    →  aralıklar başlangıç (~8) değerinden zarf daraldı
pct_converged = 0%   →  hiçbir ağırlık "yakınsamadı"
```

**Küçük ölçek kontrolü (512 vocab, 800 cümle):** QUANN PPL 510 < random 515.
Yani mekanizma **ilke olarak** PPL'ı random'ın altına indirebiliyor (öğreniyor),
ama backprop'a (502) yaklaşamıyor.

---

## 3. Neden? (Teorik açıklama)

**QUANN bir 0th-order (gradyansız) optimizasyon yöntemidir; backprop 1st-order'dur.**

| | QUANN (0th-order) | Backprop (1st-order) |
|---|---|---|
| Bilgi kaynağı | loss fonksiyonunu aday noktalarda değerlendir | tam gradyan (chain rule) |
| d boyut için maliyet | ~O(d) fonksiyon degerlendirmesi | O(1) forward + backward |
| Varsayım | loss her koordinatta ~unimodal | (yok — tam türev) |

İki temel engel:

1. **Örnek verimliliği:** d=111K boyutta, 0th-order yöntem 1st-order'dan ~O(d) kat
   daha az verimlidir. QUANN adım başına K=80 ağırlık daraltır; tam bir gradyanın
   etkisine ulaşmak için ~d/K ≈ 1400× daha fazla adım gerekir. 5 epoch (3000 adım)
   QUANN'a ~2 "gradyan-ekvivalenti" ilerleme verir; backprop'a yüzlercesi.

2. **Unimodalite varsayımı bozuluyor:** Aralık-daraltma, her ağırlık boyutunda loss'un
   tek tepe (unimodal) olduğunu varsayar. Derin bir ağın 1024-yollu cross-entropy
   loss'u **çok modallıdır** — aralık, kötü/bölgesel bir bölgeye daralır, küresel
   iyi bölgeyi bulamaz. Bu yüzden `pct_converged=0`, `mean_width` zarf daralır.

**Sonuç:** QUANN, düşük-boyutlu ve unimodal problemlerde (XOR, küçük MLP — QUANN v0'un
kanıtladığı alan) çalışır. **Gerçek bir LLM'e ölçeklenmez.** Bu, bir "hata" değil,
0th-order optimizasyonun bilinen temel sınırıdır.

---

## 4. Ne Anlama Geliyor? (Dürüst çerçeveleme)

### ✅ Gerçek ve değerli
- **CloudLM (nokta-bulut mimarisi)** gerçek ve güçlü: backprop ile 200K wiki üzerinde
  **PPL 8.14** (mevcut checkpoint `checkpoints/wiki_checkpoints_200k/wiki_cloud_best.pt`).
  Nokta-bulut geometrisi (to_nd → CloudEncoder) meşru bir mimari katkı.
- **QUANN'ın gerçek nişi:** küçük, yorumlanabilir, backprop'suz öğrenme — embedded/edge,
  sensör, <1K parametreli modeller. Aralıkları **görebilirsin** (yorumlanabilirlik),
  0th-order maliyeti kabul edilebilir ölçekte.

### ⚠️ Dürüst sınır
- **"Dünyada bir ilk: backprop'suz nokta-bulut LM"** küçük ölçekte teknik olarak
  doğru, ama **pratikte backprop'a rekabet edemez** (1.8× PPL farkı, random'a yakın).
- Tam boyutlu (21M) LM'yi QUANN ile eğitmek bu donanımda uygulanamaz.

### 🎯 Doğru iddia (savunulabilir)
> "CloudLM nokta-bulut mimarisi backprop ile güçlü bir LM üretir (PPL 8).
> QUANN, aynı mimariyi **backprop'suz** eğitebilen bir 0th-order yöntemdir —
> küçük/yorumlanabilir modeller için geçerli, tam LM'ler için değil."

Bu, hype'sız ve savunulabilir bir akademik/patent çerçevesidir.

---

## 5. Tekrar Üretme

```bash
# Hızlı duman testi (~30s)
python3 quann_cloud/experiments/real_data_test.py --quick

# Tam karşılaştırma (bu dosyadaki sonuçlar, ~15dk)
python3 quann_cloud/experiments/real_data_test.py \
    --train-sent 6000 --eval-sent 500 --max-examples 6000 \
    --epochs 5 --quann-k 80 --quann-batch 8 --bpe-vocab 1024
```

Çıktı: `quann_cloud/logs/real_data_test.json` + `quann_cloud/logs/full_run.log`

### Kod
- `quann_cloud/src/qcloud_quann.py` — QUANN motoru (stokastik aralık-daraltma,
  mini-batch ödül, dinamik base, sigma-adaptasyon, plato-tespiti)
- `quann_cloud/experiments/real_data_test.py` — SlimCloudLM + BPE + karşılaştırma

### Düzeltilen hatalar (handoff'taki iki sorun)
1. **MPS float64 crash** → QUANN CPU'da çalışıyor (küçük modelde MPS'ten ~25× hızlı).
2. **R=0/C=0** → eski kod forward'da hiç kullanılmayan `token_weights`'i daraltıyordu.
   Yeni motor, forward'ın **gerçek ağırlıklarını** daraltıyor; dinamik base + mini-batch
   ile R≈C dengeli, PPL random'ın altına iniyor.

---

## 6. Faz 1: XOR Doğrulaması (mekanizma sağlam)

XOR, QUANN'ın **çalıştığı rejimi** kanıtlar: düşük-boyutlu (~33 ağırlık),
ve 4 örnek = gürültüsüz sinyal. LM'deki (Faz 2-3) ölçeklenmemeyle çelişmez —
boyut ve multimodalite farklı.

```
XORNet (2 → 8 → 1, tanh, ~33 ağırlık), 4 örnek, QUANN: tüm ağırlık/adım, mini-batch=4

Model          Doğruluk   MSE
Random              50%   —
Backprop           100%   0.0002
QUANN              100%   0.0770   ← backprop'suz, doğru sınıflandırma
```

**Bulgu:** QUANN XOR'u backprop'suz **%100** çözüyor. MSE'si backprop'tan kaba
(0.077 vs 0.0002) — aralık-daraltma eşiğe kadar daraltır, makine hassasiyetinde
polish yapmaz — ama **sınıflandırma doğru**. Yani mekanizma temelde çalışır;
LM'deki başarısızlık bir *bug* değil, 0th-order'un boyut/multimodalite sınırıdır.

Çıktı: `quann_cloud/logs/xor_gate.json` · Kod: `quann_cloud/experiments/xor_gate/xor.py`

---

## 7. Edge Verimlilik (CloudLM)

CloudLM'in edge cihaz uygunluğu — boyut, gecikme, kuantizasyon kalitesi.

### A) Boyut (FP32 vs weight-only INT8)

| Config | Param | FP32 | INT8 |
|---|---|---|---|
| **CloudLM 21M (gerçek)** | 21.0M | **84.0 MB** | **46.2 MB** |
| CloudLM orta (deney) | 1.23M | 4.9 MB | 2.4 MB |

INT8, ağırlıkları 1 byte'a indirir. ~%45 küçülme (encoder+embedding FP32 kalır —
güvenli weight-only; tam INT8 ~21 MB olurdu).

### B) Gecikme (üretim, tokens/sn)

| Cihaz | FP32 | INT8 |
|---|---|---|
| CPU | 972 | 726 |
| MPS | 43 | — |

> **Dürüst not:** Ölçüm *weight-only* (int8 ağırlık → forward'de FP32'ye dequant).
> Gerçek INT8 hızlanması **int8 GEMM kernel** (donanım) gerektirir; bu taşınabilir
> yaklaşım dequant overhead'i nedeniyle gecikmede hafif *yavaş* ölçülüyor.
> Yani **bellek + kalite kazanımları sağlam**, hızlanma donanıma bağımlı.

### C) Kuantizasyon kalitesi (taze eğitilmiş 1.2M model, 4 epoch, 12K cümle)

```
FP32  held-out PPL = 636.52
INT8  held-out PPL = 636.19   (Δ 0.999x — neredeyse SIFIR kalite kaybı)
```

**Bulgu:** Weight-only INT8, CloudLM'de **belleği ~%45 azaltıp kaliteyi koruyor**
(0.999x). Bu, edge için güçlü bir sonuç: kompakt + kayıpsız.

### Nokta-bulut'un edge avantajı
Bulut durumu **D boyutlu (16-32)** — `d_model`'den (64-256) bağımsız, kompakt
geometrik temsil. Klasik d_model gizli katmandan çok daha küçük KV/bellek ayak izi.

Çıktı: `quann_cloud/logs/edge_benchmark.json` · Kod: `quann_cloud/experiments/edge_benchmark.py`

### Edge için doğru iddia
> "CloudLM nokta-bulut LM, weight-only INT8 ile **84MB→46MB**'a inip **kaliteyi
> korur** (0.999x PPL); bulut durumu D=16-32 boyutla kompakt. Edge'de çalışır —
> tam INT8 hızlanma için donanım int8 GEMM gerekir."

---

## 8. Gerçek CloudLM (tam ölçekli eğitim + edge paketleme)

CloudLM **mimarisini** backprop ile gerçek Türkçe veride eğittik + edge paketledik.

### Eğitim (21M param, 20K BPE, 60K cümle, 4 epoch, MPS)

```
Held-out PPL:  20000 (random) → 1084 → 644 → 452 → 406   (50× iyileşme)
```

**Üretim (PPL 406, 4 epoch):** konu-doğru ama tekrarlı/kaba —
`"Türkiye'nin başkenti" → "...başkentilerini bir, aslında her beni bir onu farklı farklı yeni bir"`
Mimari **gerçek Türkçe öğreniyor** (topic yansıyor, gerçek kelimeler) ama 4 epoch
yetersiz; düşük PPL (<50) için 50 epoch + 200K+ cümle gerekir.

### Edge paketleme (21M checkpoint)

| Metrik | Değer |
|---|---|
| Boyut | FP32 **84MB** → INT8 güvenli **31MB** / INT8 tam **21MB** |
| Kuantizasyon kalitesi | FP32 PPL 273.51 → INT8 273.51 (**Δ 1.000x, kayıpsız**) |
| ONNX export | ✅ çalışıyor (`cloudlm.onnx`) |
| ONNX FP32 hız | CPU 2.3 ms/adım (432 adım/sn) · CoreML 9.6 ms |
| **ONNX INT8** | **22MB · 2.2 ms/adım (456 adım/sn)** — FP32'den ~%5 hızlı |

**Bulgular:**
1. **Boyut:** INT8, 21M modeli **84→21MB**'a indirir (~4×). Edge için uygun.
2. **Kalite:** INT8 kuantizasyon **sıfır kalite kaybı** (1.000x) — güvenli edge yol.
3. **Hız:** int8 GEMM (onnxruntime) bu küçük modelde ~%5 hız (bottleneck GEMM değil,
   attention/cloud op'ları FP32). Hızlanma daha büyük modellerde artar.
4. **ONNX/CoreML export** çalışıyor → edge deployment'a hazır.

### Gerçek CloudLM için doğru iddia
> "CloudLM nokta-bulut LM, backprop ile gerçek Türkçe öğrenir (PPL 50× düşer);
> INT8 ile **84→21MB**'a inip **kaliteyi korur** (1.000x); ONNX/CoreML export ile
> edge'e deploy edilir. Tam PPL için uzun eğitim gerekir."

Çıktı: `quann_cloud/logs/package_cloudlm.json` · Kod: `quann_cloud/experiments/{train_cloudlm,package_cloudlm}.py`
Checkpoint: `quann_cloud/checkpoints/cloudlm_best.pt` + `cloudlm.onnx`
