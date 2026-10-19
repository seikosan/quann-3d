# fraud/ — Credit Card Fraud Detection (IBM) + QUANN point-cloud

Two-layer study on the **IBM Credit Card Fraud** dataset, built as a sub-project
of the QUANN-Cloud research repo.

* **Standard layer** — applied ML baselines on the full 24.4M-row dataset
  (logistic, random forest, GBT, MLP): *does a good fraud model work?*
* **QUANN layer** — the research's **point-cloud** architecture (feature → point,
  transaction → cloud, geometry → prediction) trained **without backprop**
  (0th-order QUANN): *can the no-backprop method work on a real tabular task?*

## Data

Single source of truth: `~/Downloads/archive` (override with `FRAUD_DATA_DIR`).

| File | Rows | Role |
|---|---|---|
| `credit_card_transactions-ibm_v2.csv` | 24.4M | full dataset (referenced in place) |
| `User0_credit_card_transactions.csv` | 20K | bundled small subset (dev/QUANN) |
| `sd254_cards.csv`, `sd254_users.csv` | — | card/user reference (unused yet) |

Schema: `User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name,
Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?` — ~0.12% fraud.

## Layout

```
fraud/
  src/
    config.py        paths + sys.path bootstrap
    features.py      numeric FeatureTransformer (fit=train, leak-safe)
    quantize.py      FeatureQuantizer -> (feature,bucket) tokens (cloud input)
    datasets.py      load / GroupKFold-by-User / fraud-enriched sampling
    metrics.py       AUC-PR, AUC-ROC, P/R/F1/F2@threshold, lift
    qcloud_fraud.py  QCloudFraudClassifier (reuses WordToND+CloudEncoderND+QCloudWeight)
    train_standard.py, train_quann.py, train_duels.py
    train_strong_reader.py, train_rich_points.py, make_figures.py
    explain_pilot.py     point-ablation attribution + human-readable explanations
    audit_false_alarms.py  false-alarm population audit (slices + counterfactuals)
    export_deploy.py     deploy package: ONNX + bundled quantizer + token decoder
  data/              bundled small CSVs
  results/           metrics JSON + model checkpoints
  figures/           plots
  logs/              run logs (append-only)
  tests/test_fraud.py
```

## Run

```bash
cd <repo root>

# standard baselines (full data)
python3 fraud/src/train_standard.py                      # lr rf gb mlp on 24.4M
python3 fraud/src/train_standard.py --n-rows 300000 --models lr gb   # quick

# QUANN point-cloud, 3 regimes (pure / hybrid / backprop)
python3 fraud/src/train_quann.py --smoke                 # tiny, verify
python3 fraud/src/train_quann.py                         # large (50K, 3 epochs)

# full-data baseline duels (XGBoost / LightGBM / FT-Transformer)
bash fraud/scripts/run_duels.sh

# reader-capacity diagnostic; rich-points D=16/32 results are complete
python3 fraud/src/train_strong_reader.py
python3 -u fraud/src/train_rich_points.py --configs 16,32 --normal-cap 450000 \
  --reader-epochs 3 --out fraud/results/rich_points.json

# point-ablation explanation pilot (small bundled data, CPU)
python3 -u fraud/src/explain_pilot.py --n-fraud 27 --n-normal 800 \
  --cases 5 --top-k 5 --out fraud/results/explain_pilot.json

# full 4.7M natural test: score all rows, explain a bounded 50K upper slice
bash fraud/scripts/run_explain_upper_slice.sh

# false-alarm audit: operational-threshold FP population, per-token slices,
# single-mask + joint counterfactuals (CPU, ~2 min)
bash fraud/scripts/run_false_alarm_audit.sh

# deploy package: ONNX + bundled quantizer + token decoder (CPU, ~1 min)
bash fraud/scripts/run_export_deploy.sh

# regenerate the four project figures
python3 fraud/src/make_figures.py

# tests
python3 fraud/tests/test_fraud.py
```

Runs on **CPU** (one heavy CloudLM/MPS run is reserved at a time).

## The point-cloud mapping (QUANN layer)

QUANN's points are **identity-based** (one D-dim point per token identity). A
tabular feature has many *values*, so each feature is **quantized** into buckets
and the token identity = `(feature, value-bucket)`:

```
transaction ──K features──> K points (one per feature, each D-dim)
                          ──> point CLOUD (set, order-free)
                          ──> CloudEncoderND (PointNet: point-MLP + max-pool,
                                pair-geometry [diff,dist,cos-angle] + mean-pool)
                          ──> class_head ──> P(fraud)
```

Three regimes, same model + same input, differ only in how the representation is
learned:

| regime | points | network | backprop? |
|---|---|---|---|
| `pure`     | QUANN 0th-order narrows | frozen (random) | no |
| `hybrid`   | QUANN 0th-order narrows | backprop (Adam) | reader only |
| `backprop` | learned `emb→to_nd`    | backprop (Adam) | yes (ceiling) |

## Results (measured, held-out, group-disjoint by User)

**Standard layer** — 24.39M rows, 27 features; split = 14.61M / 5.06M / 4.72M
(train / validation / test). Test has 6,480 fraud (base 0.137%):

| model | AUC-PR | AUC-ROC | @best-thr F2 | lift |
|---|---|---|---|---|
| Logistic | 0.013 | 0.858 | 0.073 | 16.8x |
| RandomForest | **0.693** | 0.973 | 0.637 | **277x** |
| HistGBT | 0.598 | **0.981** | 0.607 | 272x |
| MLP (64-64) | 0.252 | 0.956 | 0.294 | 134x |

**Full-data duels** on the same standard split:

| model | AUC-PR | AUC-ROC | lift |
|---|---:|---:|---:|
| XGBoost | **0.779** | **0.993** | **268x** |
| LightGBM | 0.639 | 0.936 | 255x |
| FT-Transformer (183K params) | 0.687 | 0.969 | 242x |

**QUANN point-cloud layer** — matched split = 1.02M / 100K / 4.7M
(train / validation / test), 6,181 test fraud (base 0.132%):

| model | AUC-PR | AUC-ROC | lift / note |
|---|---:|---:|---|
| hybrid (full scale; balanced 100K test) | **0.2673** | **0.9351** | F2 0.1564; lift 29.0x; 214 converged points |
| backprop | 0.080 | 0.894 | — |
| pure 0th-order | 0.001 | 0.450 | near-random; predicts almost all fraud |
| strong reader 64/64, D=8 points frozen | 0.383 | 0.969 | 188x |
| strong reader 128/128, D=8 points frozen | 0.411 | 0.975 | 197x |
| strong reader 256/256, D=8 points frozen | **0.462** | **0.975** | 207x |
| matched XGBoost | 0.805 | 0.994 | same 1.02M / 4.7M split |
| matched LightGBM | 0.766 | 0.990 | same 1.02M / 4.7M split |
| matched FT-Transformer (183K params) | 0.687 | 0.983 | same 1.02M / 4.7M split |

**Rich points (K=20, matched split; normal-cap 450,000; 3 reader epochs):**

| model | AUC-PR | AUC-ROC | F2 | lift | 0th-order speed | checkpoint |
|---|---:|---:|---:|---:|---:|---|
| rich points D=16 | 0.1919 | 0.9770 | 0.2524 | 50.92× | 108.6 rows/s | `results/quann_rich_d16.pt` |
| rich points D=32 | 0.4261 | 0.9792 | 0.2694 | 55.54× | 88.8 rows/s | `results/quann_rich_d32.pt` |

Rich-points runs are **complete**. D=32 approaches, but does not exceed, the
D=8 strong-reader AUC-PR of 0.462; D=16 regresses. Increasing point dimension
from D=8 to D=16/32 therefore did **not** break the D=8 geometry ceiling. The
results point to reader capacity—not point dimension—as the leverage. The
matched XGBoost gap remains substantial (0.8047 vs 0.4261 AUC-PR for D=32).
Exact metrics and run details: `logs/run_rich_points.log` and
`results/rich_points.json`.

Matched-duel test metrics on this split (AUC-PR / AUC-ROC / F2): XGBoost
0.8047 / 0.9938 / 0.4884; LightGBM 0.7665 / 0.9899 / 0.4455;
FT-Transformer 0.6866 / 0.9831 / 0.4897. The D=8 strong-reader reference is
0.462 AUC-PR (ROC 0.969–0.975 across the 64/128/256 readers; 214 fixed points).

## Honest positioning

This is **not SOTA**. The defensible results are narrower:

1. Point-cloud readers reach near-top tabular AUC-ROC on real data: 0.975–0.977
   versus RF 0.973 and GBT 0.981.
2. On the full 4.7M natural test, hybrid FP32 ROC is 0.9451 (backprop: 0.8938);
   it remains below the standard RF/GBT ceiling, so there is no parity claim.
3. There is an edge/explainability wedge: 529 KB FP32, 138 KB INT8, about 63k
   tx/s, and identity-based points that can be inspected.

AUC-PR is the meaningful metric under this class imbalance. The best QUANN
strong-reader result (0.462) remains about 0.34 below matched XGBoost (0.805).
More data does not explain the gap: tree/transformer baselines are broadly flat
between the 1.02M and 14.6M training runs. Rich-point ablations are complete:
D=16/32 did not surpass the D=8 strong-reader result—D=32 reached 0.426 AUC-PR,
while D=16 regressed to 0.192. This supports a D=8 geometry ceiling for these
experiments and indicates reader capacity, rather than point dimension, as the
more useful lever. The matched XGBoost gap remains (0.8047 vs D=32's 0.4261
AUC-PR).

**Full-scale hybrid and edge evaluation (2026-10-18; bundled quantizer):** on
the full 4.7M natural-rate test, AUC-ROC is **0.9451 FP32**, **0.9367 INT8
channel-wise** (Δ FP32→INT8 −0.0084), and **0.9365 INT8 tensor-wise**. On the
balanced 100K test, hybrid scores AUC-PR 0.2673, AUC-ROC 0.9351, F2 0.1564,
and lift 29.0×, with 214 converged points. RF 0.973 / GBT 0.981 ROC on the
same full test remain above hybrid; there is no parity claim. The checkpoint
bundles its trained quantizer, enabling bit-level AUC reproduction; the old
0.9581 / 0.9745 readings came from checkpoints without a saved quantizer and
are not reproducible. The full-scale `D=8/d_cloud=128/d_model=32` artifact is
**529 KB FP32 / 138 KB INT8**, with ONNX max error `1.0e-7`, about **63k tx/s**
native (ORT about 39k tx/s). Sources: `results/edge_eval_full_scale.json`,
`results/full_scale_hybrid.json`, `results/MANIFEST.md`, and `REVISION.md`.

**Explanation pilot (2026-10-19; point ablation):** `explain_pilot.py` loads
`quann_full_hybrid.pt`, decodes its bundled quantizer, and computes
leave-one-point-out logit differences for a stratified User0 sample
(**827 rows: 27 fraud / 800 normal**). The highest-impact tokens in the whole
sample are mostly fraud-protecting — `Merchant City=other`, a `zip` range,
`Merchant State=CA`, `error_type=none`, `Use Chip=Swipe` — reflecting the
normal-majority sample. Among true-fraud rows, positive drivers include rare
merchant identities, an `amount~[30.79,46.26)` bin, and `hour=16`. This is an
ablation sensitivity, not SHAP: pair geometry makes contributions non-additive.
Output: `results/explain_pilot.json` and `logs/run_explain_pilot.log`.

**Upper-slice explanation (2026-10-19; full 4.7M natural test):** the full
test reproduces the canonical full-scale reading — **AUC-PR 0.2560**,
**AUC-ROC 0.9451** on 4.7M rows / 6,110 fraud — and a bounded **50K upper
slice** (all fraud + highest-probability normals; 12.2% base rate by
construction) receives point-ablation attribution. Its dominant fraud-supporting
tokens are `MCC=5311`, `Merchant State=ONLINE`, `Use Chip=Online Transaction`,
a low `zip` bin, and `error_type=none`; rare merchant names and `MCC=5942`
recur in true-fraud cases. The high-prob normal population is driven largely by
`Merchant City=Rome` and `MCC=5970`, while `Merchant Name=other` is the largest
fraud-protecting token. The slice’s within-slice AUC is a truncation artifact,
not an operating metric. Output: `results/explain_upper_slice.json` and
`logs/run_explain_upper_slice.log`.

**False-alarm audit (2026-10-19; full 4.7M natural test):** `audit_false_alarms.py`
scores the full test and isolates the operational-threshold false-alarm
population — **9,942** normal rows at/above the best-F2 threshold 0.9969
(1,842 TP / 4,268 FN; precision 0.156, recall 0.301) — plus 2,000 top true
negatives for contrast. Its probabilities are saturated (median 0.999) and the
population is driven by **generic online-transaction tokens**: `MCC=5311`
(61% of FPs, signed +3.80), `amount>100` (73%), `Merchant State=ONLINE` /
`Use Chip=Online` (94%). The single-mask counterfactual (row unflagged if the
token were absent) fires for 94–100% of each token's rows: the model treats
the whole online-transaction cloud as fraud-like, not a few rare merchants.
The upper-slice's per-row candidates — `Merchant City=Rome` (5.6% of FPs,
signed +8.19), `MCC=5970` and rare merchant-name hashes (1 row each) — push
hard per row but cover a small share of the FP population. Joint pair masks
measure the non-additivity directly (n=1, descriptive): with `Merchant
City=Rome` present, masking the pair (Rome + `MCC=5970`) unflags the row
(1.00) even though masking `MCC=5970` alone does not (0.00) — which token is
"decisive" depends on the rest of the cloud; the `MCC=5970` + name-hash pair
row is not unflagged by any of these masks. Output:
`results/false_alarm_audit.json` and `logs/run_false_alarm_audit.log`.

**Deploy package (2026-10-19; `export_deploy.py`):** turns the explanation
surface into a single self-contained artifact at `results/deploy/` —
`model.onnx` (543 KB, dynamic batch; the `edge_deploy.onnx` graph re-exported),
`quantizer.json` (the bundled `FeatureQuantizer`: raw row → token ids) and
`token_decoder.json` (221 ids → human-readable labels, full vocabulary). A
`manifest.json` records the source checkpoint sha256, each artifact's sha256/
size, the model config, and the self-test. On the 19,963-row User0 CSV the
real-row torch↔ONNXRuntime parity is **max |Δp| 1.5e-6** (p99 8.3e-7) at about
**45.5k tx/s**; decoded cases render the top tokens in plain text. `model.onnx`
is gitignored (`*.onnx`) and is regenerated by `run_export_deploy.sh` from the
checkpoint; the two JSON artifacts + manifest are committed. Input contract:
`ids (B,K) int64` + `mask (B,K) bool` → `p (B) = P(fraud)`.

## Figures and result files

Figures in `figures/` (regenerate with `python3 fraud/src/make_figures.py`):

| PNG | Contents |
|---|---|
| `standard_models.png` | Standard-model metrics |
| `quann_regimes.png` | QUANN regime comparison |
| `quann_pr.png` | Precision-recall curves, including duels and strong-reader results |
| `quann_cloud_3d.png` | Fraud/normal point-cloud view |

| JSON | Contents / status |
|---|---|
| `results/standard_full.json` | Full-data standard baselines |
| `results/quann_full.json` | Full-scale QUANN regimes on matched split |
| `results/quann_large.json` | Historical 10K / 5%-fraud run; not comparable to natural-rate results |
| `results/duels.json` | Full standard-split XGBoost, LightGBM, FT-Transformer |
| `results/duels_quann.json` | Matched 1.02M / 100K / 4.7M baselines |
| `results/strong_reader.json` | Frozen D=8 points with 64/128/256 readers |
| `results/rich_points.json` | Completed D=16/32 rich-points metrics (K=20, matched split) |
| `results/edge_eval.json` | Footprint, latency, ONNX parity, and INT8 evaluation |
| `results/edge_eval_full_scale.json` | Full 4.7M natural-test FP32 / INT8 AUC with bundled quantizer |
| `results/full_scale_hybrid.json` | Reproducible full-scale hybrid metrics (bundled quantizer) |
| `results/explain_pilot.json` | Point-ablation explanation pilot (827 stratified User0 rows) |
| `results/explain_upper_slice.json` | Full 4.7M natural-test score + 50K upper-slice point-ablation attribution |
| `results/deploy/` | Deploy package: `model.onnx` (gitignored) + `quantizer.json` + `token_decoder.json` + `manifest.json` |

Other artifacts and checkpoint provenance: `results/MANIFEST.md`.
