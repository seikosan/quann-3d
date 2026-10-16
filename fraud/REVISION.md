# fraud/REVISION.md — teşhis + revizyon

> Tek durum kaynağı `../RESEARCH.md`; bu dosya fraud alt-projesinin teşhis/
> revizyon adayları. Edge-eval (10-16) sırasında checkpoint bütünlüğü açığı
> bulundu.

## Teşhis: kayıtlı AUC checkpoint'ten reproduce edilemiyor

**Belirti:** README/RESEARCH hybrid başlık **AUC-ROC 0.9745** (210 nokta, 10K
%5-fraud test). Diskteki `quann_hybrid.pt` (214 nokta, "full" run) aynı 4.7M
testte ölçülünce **0.77** veriyor; kendi metadata'sındaki 0.9581 bile
reproduce olmuyor.

**Kök neden (iki ayrı açık):**

1. **Checkpoint adlandırma eziliyor.** `train_quann.py` `.pt`'i
   `quann_{cfg}.pt` (scale'siz) kaydediyordu; JSON'a `--out` ile "full" girilmiş.
   Sonuç: "full" run'ın modeli `quann_hybrid.pt`'i ezdi, "large" (0.9745, 210pt)
   modelin checkpoint'i **kayboldu** — yalnızca `quann_large.json` (metrik) kaldı.
2. **Input quantizer checkpoint'e gömülmemiş.** `save_quann` model_state +
   token_weights + config yazıyordu ama **FeatureQuantizer'ı (bin edge, category
   map, offset) yazmıyordu**. Model token-id'leriyle çalışır; quantizer
   yeniden fit edilince (farklı train sample → farklı top-N/bins) token-id'ler
   kayar → gather yanlış nokta çeker → AUC 0.9581→0.77. Deploy artifact da
   eksik kalıyordu (53KB model + tokenizer yok = çalışmaz).

## Yapılan düzeltme (10-16)

* `FeatureQuantizer.to_dict()/from_dict()` (`src/quantize.py`) — fit state
  JSON-able snapshot.
* `train_quann.py`: checkpoint artık `quann_{scale}_{cfg}.pt` + **quantizer
  gömülü** (`extra["quantizer"]`) + `quann_{cfg}.pt` compat alias. `--scale` argümanı.
* `edge_eval.py`: checkpoint'teki quantizer varsa onu kullanır (reproducible);
  yoksa re-fit fallback. Default → `quann_hybrid_full.pt`.
* `train_strong_reader.py` default → `quann_hybrid_full.pt`.
* İyi model pinlendi: `results/quann_hybrid_full.pt` (full/214pt).
* Doğrulama: `fraud/tests/test_fraud.py` **10/10**; quantizer round-trip
  identical tokens; save/load quantizer taşır.

## Sonraki adımlar

1. **Yeni hybrid (210pt-ereken) yeniden eğit + `--scale full`** → quantizer'lı
   checkpoint; `edge_eval.py --with-auc` ile **reproducible** AUC + INT8
   (tensor/channel) tek run'da ölç. Bu, 0.9745/0.9581 tartışmasını kapatır.
2. Edge iddiasını kapatan set: reproducible FP32 AUC + per-channel INT8 AUC +
   hedef cihaz latency (şu an laptop CPU alt sınır).
3. README/RESEARCH: 0.9745 satırına "10K %5-enriched test; checkpoint kayıp;
   tam testte ~0.77" notu + edge-eval bulgusu (53KB, ~13µs/tx, INT8 −1.4 AUC).

## Edge-eval özeti (10-16, ölçülmüş)

| Eksen | Değer | |
|---|---|---|
| Boyut | 47K param → **53KB INT8** / 191KB FP32 | ✅ |
| Latency | **~13µs/tx** @K=13 (75K tx/s); K=50→0.25ms | ✅ laptop CPU alt sınır |
| Taşınabilirlik | ONNX temiz, ORT parite 2.5e-6 | ✅ |
| INT8 | per-channel ΔAUC **−0.018**, per-tensor **−0.059** | ✅ (geometri/LayerNorm fp32) |
| AUC | tam 4.7M testte FP32 **0.77** (0.9581 reproduce edilemez) | ⚠️ quantizer eksik (yukarı) |
