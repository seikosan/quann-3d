# fraud/results — artifact manifest (provenance)

> Why this file: the `.pt` checkpoints were saved as `quann_{cfg}.pt` (no scale
> tag) so every run **clobbered** the previous one, while metrics went to
> `quann_{scale}.json`. The checkpoint and the JSON drifted apart and the
> recorded AUCs stopped reproducing. See `../REVISION.md`.
>
> All times local. `quantizer=` is whether the input FeatureQuantizer (bin edges,
> category maps, offsets) is bundled in the checkpoint — required for a
> reproducible AUC and a complete deploy artifact. **Old files predate the fix**
> and carry `quantizer=no`.

| Artifact | Saved | Size | Model (D/d_cloud/points) | Recorded AUC-ROC | Test n | quantizer | Status |
|---|---|---|---|---|---|---|---|
| `quann_hybrid_full.pt` | 10-16 19:26 | 294KB | 8/64/**214** | 0.9581 (stored) | 4.7M | no | **PINNED good model** (copy of the "full" run's model) |
| `quann_hybrid.pt` | 10-16 11:03 | 294KB | 8/64/214 | 0.9581 (stored) | 4.7M | no | **mutable alias** — `train_quann` rewrites it each run; currently = full model |
| `quann_pure.pt` | 10-16 13:40 | 294KB | 8/64/214 | 0.4502 (stored) | 4.7M | no | pure 0th-order regime (≈ random) — full run |
| `quann_backprop.pt` | 10-16 11:59 | 227KB | 8/64/0 | 0.8938 (stored) | 4.7M | no | backprop ceiling (learned emb→nd) |
| `quann_full.json` | 10-16 13:40 | 4KB | — | hybrid 0.9581 / pure 0.45 / bp 0.89 | 4.7M | — | metrics for the "full" run (matches `quann_hybrid_full.pt`) |
| `quann_large.json` | 10-16 00:45 | 3KB | — | hybrid **0.9745** / bp 0.96 | 10K (5% fraud) | — | metrics for the "large" run — **its `.pt` was clobbered/lost** |
| `quann_smoke.json` | 10-16 08:43 | 3KB | — | hybrid 0.875 | 400 | — | tiny verify run (D=6/d_cloud=32) |
| `edge_eval.json` | 10-16 | — | — | — | — | — | edge harness output (footprint/latency/onnx/int8) — newest run |
| `edge_eval_300k.json` | 10-16 | — | — | — | 300K (natural) | — | edge harness snapshot (300K natural-rate AUC) |
| `edge_deploy.onnx` | 10-16 | 206KB | 8/64/214 | — | — | n/a | exported deploy graph (gather→PointNet→head) |
| `standard_full.json` | 10-16 | — | — | RF 0.973 / GBT 0.981 | 4.72M | — | standard-ML baselines (full data) |
| `strong_reader.json` | 10-16 | — | — | 64/64→0.969, 128/128→0.975 | 4.7M | — | strong-reader (backprop reader) results |

## The two "good" models and the 0.9745 vs 0.9581 question

* **`quann_large` (210 pts, 0.9745)** — the README/RESEARCH headline. Its
  **checkpoint is lost** (overwritten by a later run). Only its metrics survive
  here. The 0.9745 was on a **10K test at 5% fraud** (enriched-ish), not the
  natural base rate.
* **`quann_full` (214 pts, 0.9581)** — the model actually on disk (pinned to
  `quann_hybrid_full.pt`). Validated on the **4.7M test at the ~0.13% natural
  rate**. This is the deployable "good" model.

The 0.9745 > 0.9581 gap is partly the **enriched 5% test** vs natural rate, and
partly QUANN 0th-order **run-to-run variance** (different points converge to
different quality). Neither checkpoint can currently reproduce its own recorded
AUC — see `../REVISION.md` (the bundled-quantizer fix makes *future* checkpoints
reproducible).
