# Customer Data Spec — Validation and retraining path

## Transaction input

One CSV row per transaction. Current deploy quantizer expects the IBM raw
feature semantics below (column names are case- and punctuation-sensitive):

| Column | Meaning / expected representation |
|---|---|
| `Year`, `Month`, `Day` | Transaction date components as numeric values |
| `Time` | Local transaction time in `HH:MM` form |
| `Amount` | Amount string or numeric text; `$`, commas, and negative values accepted |
| `Use Chip` | Transaction channel/category (for example chip, swipe, online) |
| `Merchant Name` | Merchant identifier/category, consistently represented |
| `Merchant City`, `Merchant State` | Merchant location; missing state maps to online sentinel |
| `Zip` | Numeric postal code; missing values are supported |
| `MCC` | Merchant category code |
| `Errors?` | Required column; use an empty value when there is no error description |

The full IBM source also has `User`, `Card`, and `Is Fraud?`. `User` and `Card`
are not scoring inputs; a label is not required at inference and must never be
passed as an input feature. For training/evaluation, provide `Is Fraud?` with
`Yes`/`No` labels (or an explicitly mapped equivalent) and stable user/card
group identifiers to prevent leakage. Retain a transaction timestamp and
stable event ID outside the model fields for temporal splits and audit joins.

The deployed model derives amount buckets, hour, day-of-week, weekend, zip
missingness, and a primary error type. Keep units, currency convention, time
zone, missing-value policy, and category meaning fixed between train and serve.
This is a schema contract, not a claim that arbitrary bank schemas are
drop-in compatible.

## Tokenization and explanation

There is **no BPE or text tokenizer**. Each transaction becomes K=13 discrete
feature tokens. Numeric amount/year/zip use fitted quantile boundaries; hour,
month, day-of-week and weekend use bounded integer buckets; channel/error
categories use categorical maps; MCC, state, city and merchant name use the
top categories learned at fit time plus an `other` bucket. Token IDs and edges
are specific to the training quantizer. Never refit a quantizer and use it with
an old checkpoint/ONNX graph. Keep the new quantizer and decoder paired with
the new model. The current pilot explanation uses a leave-one-token-out logit
delta and the paired decoder labels; it is not SHAP or a causal attribution.

## Minimum data and split plan

Planning floor (pilot recommendation, not an experimentally proven guarantee):
at least **100,000 labeled transactions and 1,000 fraud examples**, with at least
200 fraud examples reserved in each of train, validation, and untouched test.
Prefer ≥1,000,000 transactions and ≥5,000 frauds when available. If the floor,
label quality, or subgroup coverage is not met, do not claim a stable customer
model; extend collection or keep the pilot shadow-only.

Use a time-forward, user/card-group-disjoint train/validation/test design where
possible. Fit quantile edges/category maps on training rows only. Train on the
training period, select architecture/threshold on validation, freeze both, and
report exactly once on a later untouched test period at the customer's natural
fraud prevalence. Avoid balancing the validation/test sample; if training is
resampled, document it and recalibrate probabilities on natural-prevalence
validation data. Report count and rate of fraud in every split, time coverage,
and any customer-approved exclusions.

## Retraining workflow

1. Map customer columns to the IBM names/semantics above; validate missingness,
   units, currency, time zone, and labels. Keep raw ID/label columns out of
   model features. Create leakage-safe group/time splits before fitting.
2. Point the fraud training loader to the approved customer CSV (the existing
   project supports `FRAUD_DATA_DIR` for the IBM-named source); adapt/map the
   schema inside the customer-controlled environment if its file/columns differ.
3. Fit a fresh `FeatureQuantizer` using training rows only and train the hybrid
   model; preserve its bundled quantizer. Validate at natural prevalence and
   compare against the customer's rule/XGB incumbent on identical event IDs.
4. Select threshold using documented false-decline and missed-fraud costs on
   validation. Export a matched ONNX + quantizer + decoder package and verify
   ONNX-vs-training-runtime parity on real customer rows before deployment.
5. Monitor feature/category drift, unknown-to-`other` rates, score distribution,
   matured fraud labels, false declines, and explanation review outcomes. Retrain
   only with a new versioned model/quantizer/decoder bundle and repeat the gates.

## Validation targets

Report PR-AUC (with fraud prevalence), ROC-AUC, precision, recall, FPR, fraud
loss captured, false-decline count/cost at the chosen operating threshold,
latency, package size, and score parity. Use prevalence as the PR baseline and
compare against the incumbent on the same later-period rows. Existing IBM
references (hybrid INT8 PR 0.2573 / ROC 0.9367; matched XGB PR 0.8047) are
context, not guaranteed customer targets. A customer's target is preregistered
from its incumbent, cost budget, and compliance use case; do not claim success
from ROC-AUC alone or from a balanced test sample.
