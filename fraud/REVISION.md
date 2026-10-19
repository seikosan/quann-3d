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
* Doğrulama: `fraud/tests/test_fraud.py` **14/14**; quantizer round-trip
  identical tokens; save/load quantizer taşır.

## Kapanan revizyonlar + P1 durumu

1. **Kapatıldı (10-17):** yeniden eğitilen `quann_full_hybrid.pt`
   (`D=8/d_cloud=128/d_model=32`, 214 nokta, gömülü quantizer) tam 4.7M
   testte **0.9451 FP32** ve **0.9367 INT8 channel** veriyor; 0.9745/0.9581
   tartışması `results/MANIFEST.md` içinde çözüldü.
2. **Kapatıldı (10-17):** edge set reproducible: 529KB FP32 / 138KB INT8,
   ONNX parite `1.0e-7`, ~63k tx/s native.
3. **P1 explanation pilot (10-19):** küçük User0 örneğinde (827 satır;
   27 fraud/800 normal) leave-one-point-out logit katkılarları çalıştı.
4. **P1 upper-slice (10-19):** tam 4.7M doğal test skorlandı (AUC-PR 0.2560,
   AUC-ROC 0.9451); 50K üst dilimde baskın fraud-sinyalleri `MCC=5311`,
   `Merchant State=ONLINE`, `Use Chip=Online Transaction`, düşük `zip` bini ve
   `error_type=none`. Nadir merchant isimleri fraud vakalarında tekrar ediyor.
   Bu yorum additif SHAP değil, non-additive point-ablation'dır.
5. **P1 false-alarm audit (10-19):** operational threshold (best-F2 0.9969)
   9.942 normal satırı flag'liyor (1.842 TP / 4.268 FN; P .156 / R .301).
   Popülasyon generic online tokenlarca sürülüyor (MCC=5311 %61, amount>100
   %73, ONLINE %94; tek-mask counterfactual 94–100% unflag) — model tüm
   online-transaction bulutunu fraud-benzeri okuyor. `Merchant City=Rome`
   (%5.6) ve `MCC=5970`/name-hash (n=1) per-row güçlü (+8.2) ama küçük pay.
   Joint mask non-additivity'yi ölçüyor: Rome varken (Rome+MCC=5970) çift
   mask unflag'ler (1.00), tek `MCC=5970` mask etmez (0.00) — baskın token,
   buluttaki diğer tokenlara bağlı. Çıktı `results/false_alarm_audit.json`,
   log `logs/run_false_alarm_audit.log`; testler 15/15.
6. **Sıradaki P1 adımı:** explanation yüzeyini deploy artifact'ına (ONNX +
   quantizer + token decoder) taşımak; küçük takibi: dominant token çiftleri
   için (MCC=5311 × ONLINE) joint counterfactual.

## Edge-eval özeti (10-16, ölçülmüş)

| Eksen | Değer | |
|---|---|---|
| Boyut | current full → **138KB INT8** / 529KB FP32 | ✅ |
| Latency | **~16µs/tx** @K=13 (~63k tx/s); K=50→0.23ms | ✅ laptop CPU alt sınır |
| Taşınabilirlik | ONNX 556KB, ORT parite `1.0e-7` | ✅ |
| INT8 | per-channel ΔAUC **−0.0084**, per-tensor **−0.0086** | ✅ (geometri/LayerNorm fp32) |
| AUC | tam 4.7M testte FP32 **0.9451** / INT8 channel **0.9367** | ✅ quantizer gömülü |
