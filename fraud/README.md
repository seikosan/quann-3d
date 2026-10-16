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
    train_standard.py, train_quann.py
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

# tests
python3 fraud/tests/test_fraud.py
```

Runs on **CPU** (MPS is held free for the live CloudLM 1.5M run in this repo).

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

**Standard layer** — full 24.4M rows; test = 4.72M (6,480 fraud, base 0.14%):

| model | AUC-PR | AUC-ROC | @best-thr F2 | lift |
|---|---|---|---|---|
| Logistic | 0.013 | 0.858 | 0.073 | 16.8x |
| RandomForest | **0.693** | 0.973 | 0.637 | **277x** |
| HistGBT | 0.598 | **0.981** | 0.607 | 272x |
| MLP (64-64) | 0.252 | 0.956 | 0.294 | 134x |

**QUANN point-cloud layer** — test = 10K (500 fraud, base 5%):

| regime | AUC-PR | AUC-ROC | @best-thr F2 | points converged |
|---|---|---|---|---|
| pure (0th-order only) | 0.061 | 0.573 | 0.210 | 0 (drifts to all-fraud) |
| **hybrid (QUANN pts + backprop net)** | **0.829** | **0.9745** | 0.679 | 210/210 |
| backprop (ceiling) | 0.810 | 0.9647 | 0.639 | — |

**Headline:** the point-cloud representation is expressive enough that the
**hybrid** regime (QUANN 0th-order narrows the points; a small backprop reader
decodes the cloud) matches the strongest standard baselines
(AUC-ROC 0.9745 ≈ RF 0.973 / GBT 0.981). **Pure** 0th-order (no backprop
anywhere) fails (~random) — consistent with the repo's finding that QUANN
scales to small/interpretable models, not complex ones. The cloud geometry
separates fraud from normal in 3D (centroid distance 1.95; see
`figures/quann_cloud_3d.png`).

Artifacts: `results/standard_full.json`, `results/quann_large.json`,
`results/quann_{pure,hybrid,backprop}.pt`, `figures/*.png`.
