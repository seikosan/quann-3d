# Edge deploy — kriter #3 kanıtı

**Kriter #3:** edge-deploy kanıtı; model boyutu, CPU inference süresi ve
Torch↔ONNX paritesi ölçülmüş olmalı. Ölçümler: ONNX Runtime 1.31.0,
CPUExecutionProvider, macOS ARM64, tek thread; 10 warmup + 80 tekrar.

## Paket ve giriş/çıkış

- Checkpoint: `../quann_binary_backprop.pt` (`quann_binary_backprop.pt`),
  gerçek C-MAPSS FD001 (2026-11-23 gerçek veri yeniden eğitimi; E0b akışı birebir,
  `maintenance/src/train_binary_real.py`). Eski synthetic E0 checkpoint'i
  (`quann_binary_backprop.pt`, /tmp/quann_binary_backprop_synthetic.pt.bak altında
  yedekli) bu dosyanın yerini aldı.
- Girdi: `cloud`, float32 `[batch, 10, 36]` — önceden ölçeklenmiş 10 zaman
  adımı / nokta-bulut penceresi. Model çıktı sınıfları `RUL > 30` ve `RUL ≤ 30`.
- Çıktı: sınıf olasılıkları, softmax.
- `model.onnx`: dinamik batch FP32 ONNX; `model.int8.onnx`: ONNX Runtime
  dynamic INT8 (MatMul weights). `manifest.json` dosya boyutu, SHA-256, ölçüm
  ayarları ve batch bazlı parite/gecikme değerlerini içerir.
- Checkpoint eğitim ölçekleyicisini içermiyor; artifact ham sensörden tam giriş
  hattı değildir (scaler train fold'undan — `maintenance/results/e0b_artifacts/
  e0b_preprocessing.json`, 2026-11-23). Ek quantizer gerekmedi; this backprop
  checkpoint'inde token/point tablosu yoktur.
- Diğer regime (pure/hybrid) checkpoint'leri E0 synthetic-trained kalır;
  bu paket yalnızca backprop gerçek-veri checkpoint'inden kurulur.

## Ölçülen sonuçlar (gerçek veri checkpoint, 2026-11-23)

| Artifact | Boyut | Batch=1 median | Batch=16 median | Batch=64 median |
|---|---:|---:|---:|---:|
| FP32 ONNX | 131,894 B (128.9 KiB) | 0.0481 ms | 0.3316 ms | 1.174 ms |
| INT8 ONNX | 83,986 B (82.0 KiB) | 0.0558 ms | 0.4419 ms | 1.6294 ms |

- Torch ↔ FP32 ONNX max mutlak fark: **1.19e-7** (batch 16/64; batch 1 için
  1.30e-8) — hedef ≤1e-5, **geçti** (manifest'te `parity.passed: true`).
- FP32/INT8 p95 ms: **0.0492/0.3393/1.2046** / **0.0575/0.4627/1.6852**
  (batch 1/16/64).
- Torch ↔ INT8 ONNX max mutlak fark: **0.00062/0.00082/0.00170** (batch
  1/16/64). INT8 FP32'den ~%36 daha küçük (131,894 B → 83,986 B), ancak bu
  makinede daha yavaş; eşik/kalite eşdeğerliği veya saha performansı iddia
  edilmez.

## Model kalitesi (gerçek veri rerun — maintenance/results/e0b_cmapss_rerun_2026-11-23.json)

- Val (engine-disjoint 20 engine, 305 pencere): AUC-ROC **0.9896**, AUC-PR
  **0.9524**, F2 **0.9158**, recall **1.00** (val threshold 0.2379, :217).
- Test aralığı (official FD001 test, 1.064 pencere): AUC-ROC **0.9908**, AUC-PR
  **0.8008**, F2 **0.7143**, recall **0.8846** (:229).
- Eski kayıt (`e0b_cmapss.json`, 2026-10-16): binary/cloud/backprop val 0.9554
  / test 0.8853 (stale — güncel kod/kütüphanelerle üretilmemiş; cf. provenance
  notu rerun dosyası). Standard baselines aynı kalmıştır (lr 0.9950, gbt 0.9963).
- Karar eşiği: val eşiği 0.23788578033447244 (rerun dosyası:217).

## Sonuç

Kriter #3'ün edge artifact, boyut, gecikme ve FP32 parite kanıtı tamamlandı
(gerçek veri checkpoint, 2026-11-23). INT8 dosyası üretilip ölçüldü; INT8
parity/latency profili kalite ya da hız iyileşmesi kanıtlamadığından alarm
eşiği/kalite eşdeğerliği kanıtlanmış sayılmaz.
