# fraud/ — Credit Card Fraud Detection (IBM) + QUANN point-cloud

Two-layer study on the **IBM Credit Card Fraud** dataset, built as a sub-project of the QUANN-Cloud research repo.

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

Schema: `User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name, Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?` — ~0.12% fraud.

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
     train_standard.py, train_quann.py, train_duels.py, train_strong_reader.py, train_rich_points.py, make_figures.py
     explain_pilot.py     point-ablation attribution + human-readable explanations
     audit_false_alarms.py, export_deploy.py   FP population audit (slices + counterfactuals); ONNX + quantizer + decoder
  data/              bundled small CSVs
  results/           metrics JSON + model checkpoints
  figures/           plots
  logs/              run logs (append-only)
  tests/test_fraud.py
```

## Run

```bash
cd <repo root>

# standard baselines (full 24.4M) / quick
python3 fraud/src/train_standard.py
python3 fraud/src/train_standard.py --n-rows 300000 --models lr gb
# QUANN point-cloud, 3 regimes (pure / hybrid / backprop)
python3 fraud/src/train_quann.py --smoke          # tiny, verify
python3 fraud/src/train_quann.py                  # large (50K, 3 epochs)
# full-data baseline duels (XGBoost / LightGBM / FT-Transformer)
bash fraud/scripts/run_duels.sh
# reader-capacity diagnostic; rich-points D=16/32
python3 fraud/src/train_strong_reader.py
python3 -u fraud/src/train_rich_points.py --configs 16,32 --normal-cap 450000 \
  --reader-epochs 3 --out fraud/results/rich_points.json
# explanation + audit + deploy (CPU)
python3 -u fraud/src/explain_pilot.py --n-fraud 27 --n-normal 800 \
  --cases 5 --top-k 5 --out fraud/results/explain_pilot.json
bash fraud/scripts/run_explain_upper_slice.sh     # full 4.7M + 50K upper slice
bash fraud/scripts/run_false_alarm_audit.sh       # FP population audit (~2 min)
bash fraud/scripts/run_export_deploy.sh           # ONNX + quantizer + decoder (~1 min)

# figures + tests
python3 fraud/src/make_figures.py
python3 fraud/tests/test_fraud.py
```

Runs on **CPU** (one heavy CloudLM/MPS run is reserved at a time).

## The point-cloud mapping (QUANN layer)

QUANN's points are **identity-based** (one D-dim point per token identity); a tabular feature is
**quantized** into buckets so token identity = `(feature, value-bucket)`:

```
transaction ──K features──> K points (one per feature, each D-dim)
                           ──> point CLOUD (set, order-free)
                           ──> CloudEncoderND (PointNet: point-MLP + max-pool,
                                 pair-geometry [diff,dist,cos-angle] + mean-pool)
                           ──> class_head ──> P(fraud)
```

Three regimes, same model + same input, differ only in how the representation is learned:

| regime | points | network | backprop? |
|---|---|---|---|
| `pure`     | QUANN 0th-order narrows | frozen (random) | no |
| `hybrid`   | QUANN 0th-order narrows | backprop (Adam) | reader only |
| `backprop` | learned `emb→to_nd`    | backprop (Adam) | yes (ceiling) |

## Results (measured, held-out, group-disjoint by User)

**Standard layer** — 24.39M rows, 27 features; split = 14.61M / 5.06M / 4.72M (train / validation / test). Test has 6,480 fraud (base 0.137%):

| model | AUC-PR | AUC-ROC | @best-thr F2 | lift |
|---|---|---|---|---|
| Logistic | 0.013 | 0.858 | 0.073 | 16.8x |
| RandomForest | **0.693** | 0.973 | 0.637 | **277x** |
| HistGBT | 0.598 | **0.981** | 0.607 | 272x |
| MLP (64-64) | 0.252 | 0.956 | 0.294 | 134x |

Full-data duels on this standard split — XGBoost **0.779**/0.993/268x, LightGBM 0.639/0.936/255x,
FT-Transformer (183K) 0.687/0.969/242x (AUC-PR/AUC-ROC/lift; `results/duels.json`).

**QUANN point-cloud layer** — matched split = 1.02M / 100K / 4.7M (train / validation / test), 6,181 test fraud (base 0.132%):

| model | AUC-PR | AUC-ROC | F2 / lift / note |
|---|---:|---:|---|
| hybrid (full scale; balanced 100K test) | **0.2673** | **0.9351** | 0.1564; 29.0x; 214 converged points |
| backprop | 0.080 | 0.894 | — |
| pure 0th-order | 0.001 | 0.450 | near-random; predicts almost all fraud |
| strong reader 64/64, D=8 points frozen | 0.383 | 0.969 | 188x |
| strong reader 128/128, D=8 points frozen | 0.411 | 0.975 | 197x |
| strong reader 256/256, D=8 points frozen | **0.462** | **0.975** | 207x |
| matched XGBoost | 0.8047 | 0.9938 | F2 0.4884; same 1.02M / 4.7M split |
| matched LightGBM | 0.7665 | 0.9899 | F2 0.4455; same split |
| matched FT-Transformer (183K) | 0.6866 | 0.9831 | F2 0.4897; same split |

**Rich points (K=20, matched split; normal-cap 450,000; 3 reader epochs; complete):**
D=16 → AUC-PR 0.1919 / ROC 0.9770 / F2 0.2524 / lift 50.92× / 108.6 rows/s (`results/quann_rich_d16.pt`);
D=32 → 0.4261 / 0.9792 / 0.2694 / 55.54× / 88.8 rows/s (`results/quann_rich_d32.pt`; details
`results/rich_points.json`). D=32 approaches but does not exceed the D=8 strong-reader 0.462; D=16 regresses.

## Honest positioning

This is **not SOTA**. The defensible results are narrower:

1. Point-cloud readers reach near-top tabular AUC-ROC on real data: 0.975–0.977 versus RF 0.973 and GBT 0.981.
2. On the full 4.7M natural test, hybrid FP32 ROC is 0.9451 (backprop: 0.8938); below the standard RF/GBT ceiling — no parity claim.
3. Edge/explainability wedge: 529 KB FP32, 138 KB INT8, about 63k tx/s, and identity-based points that can be inspected.

AUC-PR is the meaningful metric under this class imbalance. The best QUANN strong-reader (0.462) stays
about 0.34 below matched XGBoost (0.8047); more data does not explain the gap (tree/transformer baselines
broadly flat between the 1.02M and 14.6M training runs); reader capacity, not point dimension, is the lever.

**Full-scale hybrid + edge (2026-10-18; bundled quantizer):** full 4.7M natural test: AUC-ROC **0.9451 FP32**, **0.9367 INT8 channel-wise** (Δ −0.0084), **0.9365
INT8 tensor-wise**; RF 0.973 / GBT 0.981 remain above hybrid. Balanced 100K test:
AUC-PR 0.2673, AUC-ROC 0.9351, F2 0.1564, lift 29.0×, 214 converged points. The
checkpoint bundles its trained quantizer (bit-level AUC reproducible); the old
0.9581/0.9745 readings (no saved quantizer) are not reproducible. The `D=8/d_cloud=128/d_model=32` artifact is **529 KB FP32 / 138 KB INT8**, ONNX max error `1.0e-7`, about **63k tx/s** native (ORT ~39k). Sources: `results/edge_eval_full_scale.json`, `results/full_scale_hybrid.json`, `results/MANIFEST.md`, `REVISION.md`.

**Explanation pilot (2026-10-19; point ablation, `quann_full_hybrid.pt`):** leave-one-point-out logit
differences on a stratified User0 sample (**827 rows: 27 fraud / 800 normal**).
Highest-impact tokens are mostly fraud-protecting (`Merchant City=other`, `zip`
range, `Merchant State=CA`, `error_type=none`, `Use Chip=Swipe`); among true-fraud
rows, positive drivers: rare merchant identities, `amount~[30.79,46.26)`, `hour=16`.
Ablation sensitivity, not SHAP (pair geometry is non-additive). Output:
`results/explain_pilot.json`, `logs/run_explain_pilot.log`.

**Upper-slice explanation (2026-10-19; full 4.7M):** full test reproduces **AUC-PR 0.2560**,
**AUC-ROC 0.9451** (6,110 fraud); a bounded **50K upper slice** (all fraud +
highest-probability normals; 12.2% base by construction) receives attribution.
Dominant fraud-supporting tokens: `MCC=5311`, `Merchant State=ONLINE`, `Use Chip=Online
Transaction`, low `zip`, `error_type=none`; rare merchant names and `MCC=5942` recur
in true fraud. High-prob normals driven by `Merchant City=Rome` and `MCC=5970`;
`Merchant Name=other` is the largest fraud-protecting token. Within-slice AUC is a
truncation artifact, not an operating metric. Output: `results/explain_upper_slice.json`,
`logs/run_explain_upper_slice.log`.

**False-alarm audit (2026-10-19; full 4.7M):** isolates the operational-threshold FP
population — **9,942** normal rows at/above best-F2 threshold 0.9969 (1,842 TP /
4,268 FN; precision 0.156, recall 0.301) + 2,000 top true negatives. Probabilities
saturated (median 0.999); driven by **generic online-transaction tokens**: `MCC=5311`
(61% of FPs, +3.80), `amount>100` (73%), `Merchant State=ONLINE`/`Use Chip=Online`
(94%). Single-mask counterfactual fires for 94–100% of each token's rows: the model
treats the whole online-transaction cloud as fraud-like, not a few rare merchants.
Upper-slice candidates (`Rome` 5.6% of FPs, +8.19; `MCC=5970` and name hashes, 1
row each) push hard per row but cover a small share. Joint pair masks (n=1,
descriptive): with Rome present, masking (Rome + `MCC=5970`) unflags the row (1.00)
though `MCC=5970` alone does not (0.00). Output: `results/false_alarm_audit.json`, `logs/run_false_alarm_audit.log`.

**Deploy package (2026-10-19; `export_deploy.py`):** self-contained `results/deploy/` —
`model.onnx` (543 KB, dynamic batch), `quantizer.json` (raw row → token ids),
`token_decoder.json` (221 ids → labels, full vocabulary), `manifest.json` (checkpoint
sha256, artifact sha256/size, config, self-test). On the 19,963-row User0 CSV:
torch↔ONNXRuntime parity **max |Δp| 1.5e-6** (p99 8.3e-7) at about **45.5k tx/s**.
`model.onnx` is gitignored, regenerated by `run_export_deploy.sh`. Input: `ids (B,K) int64` + `mask (B,K) bool` → `p (B) = P(fraud)`.

## Figures and result files

Figures in `figures/` (regenerate: `python3 fraud/src/make_figures.py`): `standard_models.png`, `quann_regimes.png`, `quann_pr.png`, `quann_cloud_3d.png`.

| JSON | Contents |
|---|---|
| `results/standard_full.json`, `quann_full.json`, `quann_large.json` | Standard baselines (full data); QUANN regimes (full scale; 10K run not comparable) |
| `results/duels.json`, `duels_quann.json` | Baseline duels (standard split; matched 1.02M/4.7M) |
| `results/strong_reader.json`, `rich_points.json` | D=8 reader capacity; D=16/32 rich points (K=20) |
| `results/edge_eval.json`, `edge_eval_full_scale.json` | Footprint/latency/parity; full 4.7M FP32/INT8 AUC |
| `results/full_scale_hybrid.json` | Reproducible full-scale hybrid metrics |
| `results/explain_pilot.json`, `explain_upper_slice.json`, `false_alarm_audit.json` | Explanation + audit outputs |
| `results/deploy/` | `model.onnx` (gitignored) + `quantizer.json` + `token_decoder.json` + `manifest.json` |

Other artifacts and checkpoint provenance: `results/MANIFEST.md`.
