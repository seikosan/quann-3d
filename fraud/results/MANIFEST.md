# fraud/results — artifact manifest (provenance)

> Why this file: the `.pt` checkpoints were originally saved as `quann_{cfg}.pt`
> (no scale tag) so runs **clobbered** each other, while metrics went to
> `quann_{scale}.json`. The checkpoint and the JSON drifted apart and recorded
> AUCs stopped reproducing. See `../REVISION.md`.
>
> All times local. `quantizer=` means the input FeatureQuantizer (bin edges,
> category maps, offsets) is bundled in the checkpoint — required for a
> reproducible AUC and a complete deploy artifact. **Old 10-16 files predate
> the fix** and may carry `quantizer=no`.

## Canonical artifacts

| Artifact | Saved | Size | Model (D/d_cloud/points) | Recorded AUC-ROC | Test n | quantizer | Status |
|---|---|---|---|---|---|---|---|
| `quann_full_hybrid.pt` | 10-17 20:26 | 637KB | 8/128/**214** | 0.9351 stored / **0.9451** edge | 4.7M | yes | **current full-scale hybrid** |
| `quann_hybrid_full.pt` | 10-17 20:26 | 637KB | 8/128/**214** | 0.9351 stored / **0.9451** edge | 4.7M | yes | pinned alias of current full model |
| `quann_hybrid.pt` | 10-17 20:26 | 637KB | 8/128/**214** | 0.9351 stored / **0.9451** edge | 4.7M | yes | mutable alias; currently = full model |
| `quann_pure.pt` | 10-16 13:40 | 294KB | 8/64/214 | 0.4502 (stored) | 4.7M | no | historical pure 0th-order run (≈ random) |
| `quann_backprop.pt` | 10-16 11:59 | 227KB | 8/64/0 | 0.8938 (stored) | 4.7M | no | historical backprop ceiling |
| `quann_rich_d16.pt` | 10-17 11:31 | 3.0MB | 16/256/242 | 0.9770 | 4.7M | yes | rich-points D=16, K=20 |
| `quann_rich_d32.pt` | 10-17 13:41 | 3.2MB | 32/256/242 | **0.9792** | 4.7M | yes | rich-points D=32, K=20 |
| `quann_full.json` | 10-16 | 4KB | — | hybrid 0.9581 / pure 0.45 / bp 0.89 | 4.7M | — | historical full-run metrics |
| `quann_large.json` | 10-16 | 4KB | — | hybrid 0.9745 / bp 0.96 | 10K (5% fraud) | — | historical enriched-test metrics; checkpoint lost |
| `duels_quann.json` | 10-17 | 4KB | — | XGB 0.9938 / LGBM 0.9899 / FTT 0.9831 | 4.7M | — | matched baselines |
| `strong_reader.json` | 10-16 | 1KB | — | 64/64→0.969, 128/128→0.975, 256/256→0.975 | 4.7M | — | frozen D=8 points + strong readers |
| `rich_points.json` | 10-17 | 3.6KB | — | D=16 0.9770 / D=32 0.9792 | 4.7M | — | rich-points metrics |
| `edge_eval.json` | 10-17 20:32 | 1.8KB | 8/128/214 | FP32 0.9451 / INT8-channel 0.9367 | 4.7M | bundled | current edge footprint/latency/ONNX/INT8 |
| `edge_eval_full_scale.json` | 10-17 20:32 | 1.8KB | 8/128/214 | FP32 0.9451 / INT8-channel 0.9367 | 4.7M | bundled | full-scale edge snapshot |
| `full_scale_hybrid.json` | 10-17 20:26 | 1.5KB | 8/128/214 | balanced 100K 0.9351 | 100K | — | current full-scale training metrics |
| `edge_deploy.onnx` | 10-17 20:26 | 556KB | 8/128/214 | — | — | baked graph | current deploy graph |
| `explain_pilot.json` | 10-19 11:06 | 75KB | 8/128/214 | sample AUC-PR 0.2555 / ROC 0.8286 | 827 User0 sample | bundled | point-ablation explanation pilot |
| `explain_upper_slice.json` | 10-19 17:08 | 172KB | 8/128/214 | full AUC-PR 0.2560 / ROC 0.9451 | 4.7M scored / 50K attributed | bundled | full-test upper-slice explanation |
| `false_alarm_audit.json` | 10-19 19:48 | 41KB | 8/128/214 | full AUC-PR 0.2560 / ROC 0.9451 | 4.7M scored / 9,942 FP + 2,000 TN attributed | bundled | operational-threshold false-alarm audit (slices, single + joint counterfactuals) |
| `deploy/` (manifest+quantizer+decoder; `model.onnx` gitignored) | 10-19 20:20 | 3+5+16+543KB | 8/128/214 | — | 19.96K User0 self-test | bundled | current deploy package: ONNX + quantizer + token decoder |

## The 0.9745 vs 0.9581 question, resolved

* `quann_large` (210 pts, 0.9745) was measured on a **10K, 5%-fraud enriched
  test**; its checkpoint was clobbered and is lost.
* The historical `quann_full` 0.9581 reading came from a no-quantizer checkpoint
  and could not be reproduced bit-for-bit.
* The current `quann_full_hybrid.pt` (10-17, `d_cloud=128`, 214 points) bundles
  its trained quantizer and reproduces **0.9451** on the full 4.7M natural-rate
  test; INT8 channel-wise is **0.9367**.

## Current explanation surface

`explain_pilot.json` uses the current full hybrid checkpoint and reports:
decoded tokens, leave-one-point-out logit contributions, class-stratified
token rankings, and human-readable cases. `explain_upper_slice.json` extends
this to the full 4.7M natural test (50K upper slice). `false_alarm_audit.json`
isolates the operational-threshold false-alarm population (9,942 normals at/
above the best-F2 threshold 0.9969), adds per-token slices (prevalence,
probability split, single-mask counterfactual) and joint pair masks, and
records single-vs-joint unflag rates to document non-additivity.

The same surface is packaged for deployment in `deploy/` (`export_deploy.py`):
the `edge_deploy.onnx` graph re-exported + the bundled `FeatureQuantizer`
(`quantizer.json`) + a full-vocabulary `token_decoder.json`, with a
`manifest.json` recording the source-checkpoint sha256, each artifact's
sha256/size, and a 19,963-row real-row torch↔ONNXRuntime parity of 1.5e-6.