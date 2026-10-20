# PATENT APPLICATION — Tabular Identity-Cloud Interpretation (v1, New Invention)

> **Scope note:** This application is a **separate** embodiment of the language
> modeling invention (`patent_application_v2.3.md`, sentence → point-cloud → next
> token). It claims the **same point-cloud (CloudEncoder) core** applied to
> **tabular records** (e.g. credit-card transactions), combined with
> **identity-based feature points + a native set-ablation plain-text explanation
> + a counterfactual false-alarm audit + self-contained edge deployment**. The
> cloud-encoder + 0th-order-narrowing core is cited as an **adjacent (known)**
> invention; what is new here is the **tabular identity-bucket projection, the
> decoded ablation explanation, the counterfactual audit, and the self-contained
> package**.

---

## 1. TITLE

**A system and method for a tabular-record identity-based point cloud with a
native set-ablation explanation, a counterfactual false-alarm audit, and
self-contained edge deployment**

---

## 2. DESCRIPTION

### 2.1 Technical Field

This invention relates to the field of machine learning; in particular to
**anomaly / fraud detection on tabular data** (rows of numeric + categorical
features) and to **explainable machine learning (explainable ML)**. It
specifically provides a system and method in which a decision is (i) explained at
the **record level as plain text**, (ii) audited **at the population level** as a
false-alarm analysis, and (iii) distributed as a **single self-contained,
edge-deployable artifact**.

### 2.2 Prior Art

#### 2.2.1 The Dominant Paradigm: Gradient-Boosted Ensembles + SHAP

Standard production fraud / anomaly detectors are gradient-boosted ensembles
(Gradient Boosting, XGBoost, LightGBM), isolation forests, autoencoders, or
recurrent networks. They output a risk score and, at best, a **feature-level**
attribution (SHAP contributions for a GBM): *which features* made the record
risky.

Limitations:

1. **Feature level, not record level:** SHAP assumes a *linear sum of
   contributions* (the SHAP additivity axiom). In a coupled (non-additive)
   representation such as point-cloud pair-geometry, single-feature ablations do
   not reconstruct the total score; the "which feature" question is misleading.
2. **Not plain text:** SHAP output is a feature-name + numeric weight; it is not a
   *sentence* an inspector (analyst/auditor) can read directly.
3. **No population audit:** SHAP is per-record attribution; it does not answer
   *which generic token drives the false-alarm population* (e.g. is every online
   purchase read as fraud-like?).
4. **Not self-contained, not edge-deployable:** A GBM + SHAP is a large,
   environment-dependent bundle (model + feature dictionary + attribution code);
   it is not a single, bit-exact, edge artifact that produces plain text.

#### 2.2.2 Related Work

- **Point-cloud / set-based tabular representations** (FT-Transformer,
  TabTransformer, the set-encoding family): process tabular rows as
  feature-vectors and pool the unordered set into one vector. The points'
  **spatial positions are not present as input** — geometry is not preserved, so a
  point-level ablation explanation cannot be defined.
- **The QUANN point-cloud core** (adjacent invention, `patent_application_v2.3.md`):
  the core that projects tokens into low-dimensional points, decides from cloud
  geometry, and trains via 0th-order (backprop-free) narrowing. This core is
  referenced here as **known**; what is new is its application to a **tabular
  record** (identity-bucket) and the **explanation / audit / deployment**
  components.
- **0th-order (backprop-free) optimization:** methods that update weights/points
  from function values (no gradients); optionally used here for point narrowing
  (the reader may still be trained by backprop — the hybrid regime).

#### 2.2.3 The Gap in the Prior Art

No existing method combines, in **a single self-contained** system, the four
components below: (a) projecting a tabular record's features to **identity-based
`(feature, value-bucket)` points** forming a point cloud; (b) producing an
explanation via **native leave-one-point-out logit-ablation** decoded to **plain
text** by a **full-vocabulary token decoder**; (c) performing a
**population-level false-alarm audit** via single- and pair-mask
**counterfactuals**; and (d) distributing all of it as **one artifact** with
bit-exactness verification and provenance (sha256) records.

### 2.3 Technical Problem

The invention aims to solve the following technical problems:

1. Explaining a record's score, in place of a **feature-weight list**, as a
   **record-level, directly readable plain-text** statement (e.g. "amount>100;
   MCC=5311; Merchant State=ONLINE");
2. Handling, in a coupled (non-additive) representation, the problem that
   single-feature ablations do not reconstruct the total score, by means of
   **counterfactual** tests (is the record flagged if the token were absent?);
3. Automatically identifying the **driver tokens** of the false-alarm
   **population** (generic vs rare);
4. Distributing the model as a **sub-megabyte, self-contained, edge-deployable**
   artifact that produces plain text.

### 2.4 The Core of the Invention (Solution)

The core: each feature of a tabular record is converted, according to its
measured value-bucket, into an **identity token** `(feature, value-bucket)`; each
identity token is projected into a **point** in low-dimensional (3 ≤ D ≤ 64)
Euclidean space; the record is the **permutation-invariant cloud** of these
points. The cloud is reduced by the known CloudEncoder (point MLP + max-pool;
pair geometry [difference, distance, cosine-angle] + mean-pool) into a cloud
state, and a small reader produces P(anomaly).

Three new components:

- **Explanation module (260):** the record's score is measured as the change in
  the logit when a **single point is removed** from the cloud (leave-one-point-out
  ablation); the highest-contribution points are decoded to **plain text** by a
  **full-vocabulary token decoder (270)**. This differs from SHAP: the pair-
  geometry term makes the contributions **non-additive**, so it is reported as a
  *counterfactual sensitivity*, not a linear sum of contributions.
- **Counterfactual audit module (280):** a token (single-mask) or a pair of tokens
  (pair-mask) is **removed** and the question asked is whether the record remains
  above the threshold. This answers "does that token decide it alone / only
  together with which other point?"; it measures the split of the false-alarm
  population into generic vs rare tokens.
- **Deployment packager (290):** the model becomes a single self-contained
  package — `model.onnx` + `quantizer.json` (raw row → token ids) +
  `token_decoder.json` (id → plain-text label) + `manifest.json` (source
  checkpoint sha256, each artifact's sha256/size, configuration, self-test);
  torch↔ONNX bit-exactness (max |Δp|) and edge latency are verified on raw rows.

### 2.5 Technical Effects

1. **Record-level plain-text explanation (primary effect):** an anomalous
   record's score is decoded into feature-sentences an inspector can read directly
   (e.g. "amount>100; MCC=5311; Merchant State=ONLINE"); a *readable statement* is
   obtained in place of SHAP's numeric weight list.
2. **Honest non-additive attribution:** the contribution is given as the
   leave-one-point-out logit-change + a counterfactual mask; single-feature values
   that do not reconstruct the total score are honestly reported as "the decision
   arose from this cloud configuration."
3. **Population audit:** single/pair-mask counterfactuals automatically separate
   the **generic** tokens (large population share) from the **rare** tokens
   (large per-row contribution) that drive the false-alarm population — measuring
   whether the model answers "is this an online purchase?" in a fraud-like way.
4. **Self-contained edge deployment:** the whole system (reader + input
   quantizer + decoder + manifest) is one package; a sub-megabyte artifact,
   bit-exact on raw rows and producing plain text, ships to the edge device.
5. **Optionally 0th-order geometry:** point narrowing can be done 0th-order
   (backprop-free); the reader is trained by backprop, giving the hybrid regime
   (pure 0th-order stays near-random on real tasks — the invention claims the
   hybrid regime).

### 2.6 Brief Description of the Drawings

- **Figure 1:** System architecture — 200: input record, 210: feature
  quantization, 220: identity-point projection, 230: cloud encoder (231: point
  MLP, 232: pair MLP, 233: fusion), 240: 0th-order point narrowing, 250: reader,
  260: ablation explanation, 270: full-vocabulary decoder, 280: counterfactual
  audit, 290: deployment packager.
- **Figure 2:** Identity-bucket projection — the points of a record's
  `(feature, bucket)` tokens in the cloud.
- **Figure 3:** Leave-one-point-out ablation — the logit change on removing one
  point.
- **Figure 4:** Counterfactual mask (single vs pair token) — above/below
  threshold.
- **Figure 5:** The self-contained deployment package (onnx + quantizer + decoder
  + manifest) and the raw-row bit-exactness check.

### 2.7 Detailed Description of the Invention

#### 2.7.1 Input Record (200)

A record is one row of N features (e.g. a credit-card transaction: Amount, Use
Chip, Merchant Name/City/State, Zip, MCC, Errors, Is Fraud). Numeric and
categorical features, plus the label, are separated.

#### 2.7.2 Feature-Quantization Module (210)

Each numeric feature is split into buckets determined on the training set
(leak-safe); each categorical feature is bucketed by its observed values. The
result is, per feature, an **identity token** `(feature, value-bucket)`. A record
yields K identity tokens for its K features. The quantizer is `fit` on the
training set and applied to the test set; it is bundled into the checkpoint (for
bit-exact AUC).

#### 2.7.3 Identity-Point Projection (220)

Each identity token is projected to a point in 3 ≤ D ≤ 64 dimensional space
(optionally by 0th-order narrowing, Claim 11). The D=8 strong-reader regime uses
d_cloud=128, d_model=32, 214 converged points.

#### 2.7.4 Cloud Encoder (230)

The known CloudEncoder: (231) a shared point MLP per point + max-pool; (232) a
pair MLP per point-pair with input `[pᵢ−pⱼ, ‖pᵢ−pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖]` +
mean-pool; (233) a fusion of the two into a permutation-invariant cloud state. The
pair-geometry term is the source of the **non-additive** contributions (Claim 13).

#### 2.7.5 0th-Order Point Narrowing (240) — optional

Point positions may be learned by 0th-order (gradient-free) QUANN narrowing; the
reader (250) is trained by backprop. The pure 0th-order regime stays near-random on
real tasks; the claimed **hybrid** regime is 0th-order point narrowing + backprop
reader.

#### 2.7.6 Reader (250)

A small reader (MLP) produces P(anomaly) from the cloud state; it is optionally
quantized to INT8 (529 KB FP32 → 138 KB INT8).

#### 2.7.7 Explanation Module (260) + Full-Vocabulary Decoder (270)

For each point, the logit change on removing the point from the cloud
(leave-one-point-out) is computed. The largest-|change| points are decoded to
plain text by the **full-vocabulary** token decoder (270) — 221 ids = the complete
feature dictionary; every bucket/value maps to a name, and no "other" bucket is
dropped. The output is a list of feature-sentences (e.g. "amount>100; MCC=5311;
Merchant State=ONLINE").

#### 2.7.8 Counterfactual Audit Module (280)

- **Single-mask:** if token T is removed, does the record remain above the
  threshold? (in what fraction → does T decide it alone).
- **Pair-mask:** if (T, U) are removed together? If U is ineffective alone but
  effective together, the decision depends on the cloud configuration
  (non-additivity).
This measures the split of the false-alarm population into **generic** (large
share) vs **rare** (strong per-row) tokens.

#### 2.7.9 Deployment Packager (290)

Package = `model.onnx` (dynamic batch) + `quantizer.json` + `token_decoder.json` +
`manifest.json`. The manifest carries: source checkpoint sha256, each artifact's
sha256/size, the model configuration, and the self-test result. On raw rows,
torch↔ONNXRuntime bit-exactness (max |Δp| 1.5e-6) and edge latency (~45.5k
records/s) are verified. Input contract: `ids (B,K) int64` + `mask (B,K) bool` →
`p (B) = P(anomaly)`.

---

## 3. CLAIMS

### Independent Claims

**Claim 1**

A system comprising a machine-learning detector known in the prior art that
processes the features of a tabular record to produce a risk score,

**characterized in that it comprises**

(a) a feature-quantization module (210) that maps each numeric and/or categorical
feature of the record to a value-bucket, producing an `(feature, value-bucket)`
identity token;

(b) an identity-point module (220) that projects each said identity token to a
point in 3 ≤ D ≤ 64 dimensional Euclidean space and combines said points into a
permutation-invariant point cloud;

(c) a reader (250) that produces a cloud state from said cloud and produces a score
from said cloud state;

(d) an explanation module (260) that measures the logit-change of the score on
**removing a single point** from said cloud and decodes said points to **plain-text
labels** through a token decoder (270);

(e) a counterfactual-audit module (280) that, on a subset of said records, tests
whether the score remains above a threshold when one or more tokens are **removed**;
and

(f) a module (290) that assembles said reader, feature-quantization module, token
decoder, and a provenance/integrity record of each artifact into **a single
self-contained deployment package**,

wherein said system presents, together, a record's score as a **readable
feature-statement**, a **population-level counterfactual** measurement, and a
**single edge-deployable artifact**.

**Claim 2**

A method, known in the prior art, of producing a risk score from tabular records,

**characterized in that it comprises the steps of**

(a) converting each record's features into `(feature, value-bucket)` identity
tokens;

(b) projecting each identity token to a point in D-dimensional space to form a
cloud;

(c) producing a permutation-invariant cloud state from the cloud and predicting a
score;

(d) measuring the logit-change of the score by removing a single point from the
cloud and decoding the highest-contribution points to plain text through a
decoder; and

(e) testing whether the score remains above a threshold when a token or a pair of
tokens is removed,

wherein said steps (d) and (e), by reporting the non-additivity of a coupled
point-geometry representation **counterfactually**, provide a record-level,
readable explanation in place of a feature-weight list.

**Claim 3**

A computer program comprising program instructions which, when executed by a
processor, perform the steps of the method of Claim 2, wherein said program
instructions produce a tabular-record explanation as plain text and as a
self-contained artifact.

### Dependent Claims

**Claim 4:** Dependent on Claim 1, wherein the pair-geometry input vector is, for
each point pair, of the form `[pᵢ − pⱼ, ‖pᵢ − pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖]` and is
rotation-invariant.

**Claim 5:** Dependent on Claim 1, wherein the reader (250) is a small MLP,
optionally quantized to INT8.

**Claim 6:** Dependent on Claim 1, wherein the token decoder (270) is
**full-vocabulary**: every `(feature, value-bucket)` identity maps to a
human-readable label, and no bucket is dropped as "other."

**Claim 7:** Dependent on Claim 1, wherein the explanation module (260) reports
the leave-one-point-out logit-change, in the case that single-feature values do not
reconstruct the total score (non-additive), as a *counterfactual sensitivity*.

**Claim 8:** Dependent on Claim 1, wherein the counterfactual-audit module (280)
runs both single-mask and pair-mask (two tokens together) tests.

**Claim 9:** Dependent on Claim 8, wherein cases in which a token has no effect
alone but has an effect together with a second token are identified, proving that
the decision depends on the cloud configuration (non-additivity).

**Claim 10:** Dependent on Claim 1, wherein the counterfactual-audit module (280)
separates the above-threshold false-alarm population into **generic** (large
population-share) and **rare** (large per-row contribution) tokens.

**Claim 11:** Dependent on Claim 1, wherein the identity-point module (220)
produces point positions by a 0th-order (gradient-free) narrowing, the reader (250)
being trained by backprop, forming a hybrid regime.

**Claim 12:** Dependent on Claim 11, wherein D = 8, d_cloud = 128, d_model = 32,
and 214 converged points are used.

**Claim 13:** Dependent on Claim 1, wherein the cloud encoder takes max-pool over
point features and mean-pool over pair features and combines the two.

**Claim 14:** Dependent on Claim 1, wherein the deployment package (290) comprises
`model.onnx`, `quantizer.json`, `token_decoder.json`, and a `manifest.json`; the
manifest carries the source checkpoint's and each artifact's identity (sha256) and
size record, the model configuration, and a self-test result.

**Claim 15:** Dependent on Claim 14, wherein the `model.onnx` output on raw rows is
verified **bit-exact** against a reference (torch) output (below a max |Δp|
threshold).

**Claim 16:** Dependent on Claim 1, wherein the input contract is `ids (B,K)
int64` + `mask (B,K) bool` → `p (B)` and carries a dynamic batch.

**Claim 17:** Dependent on Claim 3, wherein the program instructions are stored on
a non-transitory computer-readable recording medium.

**Claim 18:** Dependent on Claim 1, wherein the values of D, d_cloud and d_model
are selected from a predefined set based on the measured complexity of the task
(region count, cloud spread).

**Claim 19:** Dependent on Claim 1, wherein the system is applicable to credit-card
transactions, banking transactions, or any `(numeric, categorical)` tabular-record
domain.

---

## 4. ABSTRACT

Each feature of a tabular record is converted, according to its measured
value-bucket, into an `(feature, value-bucket)` identity token and a low-
dimensional (3–64) point; the record is the permutation-invariant cloud of these
points. A known point-cloud encoder produces a cloud state and a small reader a
risk score. The score is measured as the logit-change on **removing a single
point** from the cloud, and the highest-contribution points are decoded to **plain
text** by a **full-vocabulary** decoder — yielding a readable record-explanation
in place of a feature-weight list. The pair-geometry makes the contributions
non-additive; this is reported **counterfactually** via single- and pair-token
masks, and the false-alarm population is split into generic vs rare tokens. The
model is packaged as `onnx` + `quantizer` + `decoder` + `manifest` (sha256,
config, self-test) — a **self-contained**, sub-megabyte artifact that is bit-exact
on raw rows and produces plain text on the edge.

---

## 5. DRAWINGS (Separate Pages)

| Page | Figure | Content | Source |
|---|---|---|---|
| 1 | Figure 1 | System architecture (reference numbers 200–290) | New drawing (block diagram) |
| 2 | Figure 2 | Identity-bucket projection — record points in the cloud | `fraud/figures/quann_cloud_3d.png` |
| 3 | Figure 3 | Leave-one-point-out ablation (logit change) | New drawing (schematic) |
| 4 | Figure 4 | Single- and pair-mask counterfactual | New drawing (schematic) |
| 5 | Figure 5 | Self-contained package + raw-row bit-exactness | `fraud/results/deploy/` |

> **Attorney note:** Figures must be prepared as separate pages, black-and-white
> line art with reference numbers, in PDF drawing format. This document is a **new
> invention (v1)**, filed separately from the language-modeling patent
> (`patent_application_v2.3.md`); the overlap of the shared cloud-encoder core
> between the two must be assessed by the attorney for cross-reference / priority.

---

## 6. STRATEGY NOTE (not part of the filing)

### 6.1 Why a separate application?

The language-modeling invention (`patent_application_v2.3.md`) is **sentence →
word-token → cloud → next token**. This invention moves the same core to a
**tabular-record** target, and carries its real novelty in the **explanation
surface** (identity-bucket + decoded ablation + counterfactual audit +
self-contained deployment). A separate file: (a) narrows the claim scope away from
the language domain into **tabular / compliance**, (b) sets up a cleaner gap claim
against the SHAP/GBM prior art, and (c) reduces rejection risk from over-broad
claims (language + tabular in one application).

### 6.2 Novelty risk

- **Strong:** the *combination* of (d) decoded ablation + (e) counterfactual +
  (f) self-contained package as one artifact — this quartet is not known together
  in the SHAP/GBM and set-tabular literature.
- **Caution:** the cloud-encoder + 0th-order narrowing is within the adjacent
  patent (v2.3); independent Claim 1 references that core as **known**. The
  attorney should assess whether paragraphs (b)(c) of Claim 1 overlap v2.3 and,
  if so, move (b)(c) to the dependents and narrow the independent claim to the
  explanation / audit / deployment trio.
- **Data:** IBM credit-card (24.4M), matched split 1.02M/100K/4.7M; AUC-PR 0.256
  vs XGB 0.80 (honest gap); deploy parity 1.5e-6, 45.5k records/s; audit
  1842/9942/4268. The patent makes **no performance-superiority** claim; the claim
  is the **explanation + audit + deployment** combination.

### 6.3 Next steps

- This EN translation mirrors the v2.3_EN section scheme.
- Figures 1, 3, 4 require new drawings; 2 and 5 rest on the existing
  figure / artifact.
- To the attorney: the v2.3 overlap of independent Claim 1's (b)(c) paragraphs +
  the priority-date strategy.
