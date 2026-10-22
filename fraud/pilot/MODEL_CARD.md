# Model Card — QUANN Fraud Edge Pilot

## Intended use

Small-footprint, CPU/edge shadow scoring of IBM-schema card transactions, with a
human-readable local explanation. The intended wedge is constrained edge
deployment and reviewability, not best-in-class fraud discrimination. Scores
are prioritization signals; they are not authorization policy or a guarantee.

## Architecture

```text
IBM transaction CSV
        │  bundled FeatureQuantizer (13 fields → 13 tokens)
        ▼
 ONNX ids (B,13) + mask ──► QUANN table (214 learned points, D=8)
                                │
                       cloud reader (d=128)
                                │
                  backprop-trained head (d=32)
                                ▼
                     P(fraud) + sigmoid
                                │
     per-token leave-one-point-out logit deltas + bundled decoder
```

The deploy graph is the hybrid QUANN strong-reader plus backprop reader. The
QUANN geometry supplies the token point table; the cloud reader/head produce
the score. The 543 KB ONNX file contains the scoring graph; the quantizer and
token decoder are required alongside it.

## Reported results

Full natural-prevalence test where indicated; PR means average precision / PR-AUC.

| Model / setting | PR-AUC | ROC-AUC | Interpretation |
|---|---:|---:|---|
| Hybrid FP32, full natural test | 0.2560 | 0.9451 | Floating-point reference |
| Hybrid INT8-channel QAT, full natural test | 0.2573 | 0.9367 | Packaged edge-model result; small PR gain, ROC decrease |
| Strong-reader, D=8 | 0.462 | — | Research variant, not the packaged model score |
| Rich-points, D=32 | 0.4261 | 0.9792 | Research variant, not the packaged model score |
| Matched XGBoost | 0.8047 | 0.9938 | Same-protocol benchmark; material gap remains |
| Random Forest reference | 0.693 | 0.973 | Separate standard benchmark ceiling |
| Gradient-boosted-tree reference | 0.598 | 0.981 | Separate standard benchmark ceiling |

These rows come from different model variants and, for the RF/GBT references,
different benchmark protocol. Do not interpret every row as a single matched
ranking. The strongest matched XGBoost comparison shows the model is not more
accurate; this pilot does not depend on closing that gap.

## Package and operating points

| Property | Value |
|---|---:|
| ONNX graph | 543.5 KB (dynamic batch) |
| Quantizer / decoder | 5.4 KB / 16.5 KB; 221 decoder entries |
| Architecture | D=8, 214 QUANN points, d_cloud=128, d_model=32, K=13 |
| Real-row Torch ↔ ONNX maximum absolute probability difference | 1.5e-6 |
| Apple-Silicon CPU throughput, batched score-only | about 45–54k transactions/s |
| Reference decision threshold | P(fraud) ≥ 0.9969 |

Throughput is a batched score-only measurement on one laptop, not an edge-device
SLA. The client also performs 13 masked scoring passes per row for explanations;
measure full end-to-end latency on the target hardware. Threshold 0.9969 is the
current operational reference, not a customer-calibrated threshold. Lowering it
will generally flag more transactions; raising it will generally flag fewer.
Choose it against customer-specific false-decline and missed-fraud costs.

## Limitations and known caveats

- There is a large matched XGBoost PR-AUC gap (0.2560/0.2573 hybrid vs 0.8047
  XGBoost). Do not sell this as a more accurate model or a replacement for a
  stronger incumbent without customer validation.
- At the very high 0.9969 threshold, a probability just below the cutoff is
  NORMAL under this client; a score is not a calibrated authorization decision.
- The explanation is a local leave-one-point-out logit delta, not SHAP or an
  additive causal account. Feature interactions mean deltas need not sum to the
  prediction. The packaged graph returns sigmoid probabilities; the client
  reconstructs logits after clipping probabilities to [1e-7, 1-1e-7], so effects
  on saturated scores are approximate and may be compressed.
- Rare-token audit anecdotes include a candidate with n=1 (for example, a
  particular MCC/city observation); they are unstable and not general rules.
- Training/evaluation use IBM-derived data. Customer distribution shift, card
  product, fraud prevalence, and policy costs can change both ranking and
  threshold behavior.
- The bundled quantizer maps unseen categories to `other`; changed column
  semantics, units, currencies, or time zones invalidate the input contract.
- Evaluation includes a single-user small-data explanation pilot; its metrics
  are not population-level evidence. Primary reported hybrid metrics use the
  full natural test set as stated above.

## Use / do not use

**Consider** a time-bounded shadow pilot when compact CPU/edge deployment and
transaction-level review cues are valuable, the incumbent remains authoritative,
and compliance reviewers can assess explanation usefulness.

**Do not use** as a standalone decline/block authority, as evidence of superiority
to XGBoost/RF/GBT, on non-IBM schemas without retraining/validation, or where a
missed fraud / false decline cost has not been quantified. Do not treat the
explanation as causal or as a complete regulatory rationale.
