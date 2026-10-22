# Edge deploy — kriter #3 kanıtı

**Kriter #3:** edge-deploy kanıtı; model boyutu, CPU inference süresi ve
Torch↔ONNX paritesi ölçülmüş olmalı. Ölçümler: ONNX Runtime 1.31.0,
CPUExecutionProvider, macOS ARM64, tek thread; 10 warmup + 80 tekrar.

## Paket ve giriş/çıkış

- Checkpoint: `../quann_binary_backprop.pt` (`quann_binary_backprop.pt`),
  C-MAPSS binary; saklanan binary checkpoint'ler içinde test PR-AUC **0.7336**
  ve F2 **0.7692** ile en yüksek.
- Girdi: `cloud`, float32 `[batch, 10, 36]` — önceden ölçeklenmiş 10 zaman
  adımı / nokta-bulut penceresi. Model çıktı sınıfları `RUL > 30` ve `RUL ≤ 30`.
- Çıktı: sınıf olasılıkları, softmax.
- `model.onnx`: dinamik batch FP32 ONNX; `model.int8.onnx`: ONNX Runtime
  dynamic INT8 (MatMul weights). `manifest.json` dosya boyutu, SHA-256, ölçüm
  ayarları ve batch bazlı parite/gecikme değerlerini içerir.
- Checkpoint eğitim ölçekleyicisini içermiyor; artifact ham sensörden tam giriş
  hattı değildir. Ek quantizer gerekmedi; bu seçilmiş backprop checkpoint'inde
  token/point tablosu yoktur.

## Ölçülen sonuçlar

| Artifact | Boyut | Batch=1 median | Batch=16 median | Batch=64 median |
|---|---:|---:|---:|---:|
| FP32 ONNX | 120,930 B (118.1 KiB) | 0.0408 ms | 0.2868 ms | 1.0405 ms |
| INT8 ONNX | 78,924 B (77.1 KiB) | 0.0496 ms | 0.3855 ms | 1.4725 ms |

- Torch ↔ FP32 ONNX max mutlak fark: **5.07e-7** (hedef ≤1e-5, geçti).
- FP32/INT8 batch 1/16/64 p95 ms: **0.0443/0.3281/1.1195** /
  **0.0581/0.4222/1.5911**.
- Torch ↔ INT8 ONNX max mutlak fark: **0.00253/0.00867/0.02068** (batch
  1/16/64). INT8 yaklaşık **%34.7** daha küçük, ancak bu makinede daha yavaş;
  eşik/kalite eşdeğerliği veya saha performansı iddia edilmez.

## Sonuç

Kriter #3'ün edge artifact, boyut, gecikme ve FP32 parite kanıtı tamamlandı.
INT8 dosyası üretilip ölçüldü; INT8 parity/latency profili kalite ya da hız
iyileşmesi kanıtlamadığından alarm kullanımına eşdeğer model olarak sunulmaz.
