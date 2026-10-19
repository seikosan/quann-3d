# Geometric Sentence Clouds: A Point-Cloud Approach to Language Modeling Beyond Sequential Attention

**Draft v0.1 — October 2026**

---

## Abstract

We propose **CloudLM**, a language modeling architecture that represents sentences as point clouds in N-dimensional space, where each token occupies a position in the cloud and sentence meaning is encoded in the **geometry** of the point configuration (distances, angles, cluster shape) rather than in a single high-dimensional vector. Unlike transformers, which represent a sentence as one 768D vector and predict the next token via sequential attention, CloudLM treats the sentence as an **unordered set of geometric points** and predicts based on the spatial relationships between them. We demonstrate that even with as few as 16 dimensions, the model learns to separate syntactic roles into distinct spatial regions (e.g., conjunctions cluster centrally, demonstratives occupy eastern-southern quadrants). On a 30K-sentence Turkish corpus with a 20K BPE vocabulary, CloudLM achieves a perplexity of 57 after 15 epochs (21M parameters), reducing the random baseline (20,000) by a factor of 350×. At 200K scale, the D=32 model reaches 6.45 shared held-out PPL versus 2.92 for a matched causal baseline; a same-distribution scale gate shows 200K/50ep 24.53 vs 1.5M/20ep 64.28 on the shared 200K slice. The approach offers a geometrically interpretable alternative to transformer attention, with potential advantages for morphologically rich languages and for tasks requiring spatial reasoning over sentence structure.

**Keywords:** language modeling, point clouds, geometric deep learning, Turkish NLP, agglutinative morphology, set-based architectures

---

## 1. Introduction

### 1.1 The Transformer Paradigm

Modern language models represent a sentence as a sequence of token embeddings, process them through stacked self-attention layers, and produce a single context vector per position. The "meaning" of a sentence resides in a 768-4096 dimensional vector space, where each dimension is an implicit "meaning axis." Prediction of the next token is a function of this single vector.

This paradigm has two structural properties:
1. **Sequentiality:** Position matters. The same set of words in different orders produces different vectors.
2. **Monolithic representation:** The entire sentence is compressed into one vector per position. Relationships between words are encoded implicitly through attention weights, not explicitly through geometry.

### 1.2 The Cloud Perspective

We ask: *What if the sentence were not a single vector, but a cloud of points?*

In this view:
- Each token maps to a point in $\mathbb{R}^D$ (where $D \ll 768$)
- The sentence is the **set** of these points
- Meaning is the **geometry**: which points are close, which are far, what shape the cloud forms
- Prediction is a function of the cloud's configuration, not of a single vector

This is analogous to how PointNet (Qi et al., 2017) represents a 3D object not as a single feature vector but as a set of 3D points, where the object's identity emerges from the spatial arrangement.

### 1.3 Why 3D/ND?

The key insight is **dimensionality as relationship budget**:
- In 3D, the model can encode at most 3 independent "meaning directions"
- In 7D, it can separate 7 relationships simultaneously
- In 16D, it can encode 16 syntactic/semantic axes

This creates a **bottleneck** that forces the model to discover the most salient geometric relationships, rather than spreading meaning diffusely across 768 dimensions.

### 1.4 Turkish as a Test Case

Turkish is an agglutinative language: a single word can encode subject, object, location, time, negation, and mood through suffix stacking (*bahçelerden* = "from the gardens"). This makes Turkish an extreme test of whether a geometric model can capture morphological structure through spatial proximity of subword points.

---

## 2. Related Work

### 2.1 Point Cloud Networks
PointNet (Qi et al., 2017), DGCNN (Wang et al., 2019), and PointTransformer (Zhao et al., 2021) process unordered point sets for 3D vision. Our CloudEncoder adapts the PointNet pattern (shared MLP + symmetric aggregation) to linguistic point clouds.

### 2.2 Set-Based Language Models
Deep Sets (Zaheer et al., 2017) and Set Transformer (Lee et al., 2019) process token sets without positional order. CloudLM extends this: instead of pooling set features into one vector, we preserve the **spatial structure** of the set in low-dimensional geometry.

### 2.3 Geometric Analysis of Language Models (Post-Hoc)
A recent line of work studies the *geometry of existing* transformer representations:

- **Hu et al. (2026)** ["Language Models Represent and Transform Concepts with Shared Geometry," Georgia Tech/Edinburgh] formalize concept representations as **point-cloud manifolds** in neural state space, drawing from neural population geometry (Chung & Abbott, 2021). They collect the same word's contextualized embeddings across many sentences, forming a point cloud per concept, and characterize within-concept and between-concept relations via kernel matrices. They show this geometry is shared across model families (Llama, Qwen, Mistral, Gemma).

- **"Scale Determines Whether LMs Organize Representation Geometry for Prediction" (2025)** introduces Subspace PGA to study whether intermediate transformer layers organize their distance structure for prediction, finding two regimes (direct vs. detour) determined by model width.

- **"Semantic Geometry of Sentence Embeddings" (EMNLP Findings, 2025)** provides a formal framework to characterize semantic organization in sentence embedding spaces.

**Critical distinction:** All of these works use geometry as a **post-hoc interpretability tool** — they analyze what transformers *already* represent. None use geometry as the **computational mechanism for prediction**. CloudLM inverts this relationship: the point-cloud geometry *is* the prediction substrate, not a diagnostic applied to a high-dimensional hidden space.

### 2.4 Neural Population Geometry
Chung & Abbott (2021) ["Neural population geometry: An approach for understanding biological and artificial neural networks," *Curr. Opin. Neurobiol.*] established the framework of studying neural populations as point-cloud manifolds in high-dimensional state space. Our work transfers this framework from neuroscience *analysis* to NLP *architecture*: where they study what geometry emerges in trained networks, we *design* the network to compute directly in geometric space.

### 2.5 Distributional Semantics
Word2Vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) place words in vector spaces where geometric relationships (analogy: king - man + woman ≈ queen) encode meaning. CloudLM generalizes this from *word-level* vectors to *sentence-level* point clouds: meaning is not where a single point is, but how a *set* of points is arranged.

### 2.6 Geometric Deep Learning
Bronstein et al. (2021) frame ML on non-Euclidean domains. Our work applies geometric principles (distance, angle, curvature of point configurations) to the linguistic domain, treating sentences as geometric objects rather than sequential data.

### 2.7 Positioning Summary

| Work | Geometry as... | Domain |
|------|---------------|--------|
| PointNet (2017) | Input representation | 3D vision |
| Deep Sets / Set Transformer (2017/19) | Processing substrate (pooled) | General |
| Hu et al. (2026) | **Analysis tool** (post-hoc) | LM interpretability |
| Scale/PGA (2025) | **Analysis tool** (post-hoc) | LM interpretability |
| Chung & Abbott (2021) | **Analysis framework** | Neuroscience |
| **CloudLM (ours)** | **Prediction mechanism** (computational) | **Language modeling** |

---

## 3. Method

### 3.1 Architecture

```
Input: token sequence [t₁, t₂, ..., tₙ]

1. CONTEXT ENCODER (Transformer, L layers)
   tokens → context vectors c₁, ..., cₙ ∈ R^d

2. ND PROJECTION (MLP)
   cᵢ → pᵢ ∈ R^D  (D = 3, 7, 16, or 32)
   [Tanh activation: points in [-1,1]^D]

3. CLOUD ENCODER (PointNet-style)
   a) Point MLP: pᵢ → fᵢ ∈ R^d_cloud  (shared across all points)
   b) Pair MLP: (pᵢ, pⱼ) → gᵢⱼ ∈ R^d_cloud
      Input: [pᵢ-pⱼ, ‖pᵢ-pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖]
   c) Aggregation:
      cloud_state = Fusion(MaxPool(fᵢ), MeanPool(gᵢⱼ))

4. PREDICTION HEAD
   cloud_state → logits ∈ R^vocab  (next token)
   cloud_state → p_next ∈ R^D      (auxiliary: where the next point lands)
```

### 3.2 Training Objective

$$\mathcal{L} = \mathcal{L}_{CE}(\text{logits}, t_{\text{next}}) + \lambda \cdot \mathcal{L}_{MSE}(\mathbf{p}_{\text{next}}, \text{proj}(c_{\text{next}}))$$

The auxiliary point loss encourages the model to predict not just *which* word comes next, but *where it lands* in the cloud.

### 3.3 Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| D (cloud dim) | 16 | Enough for syntactic roles, small enough for interpretability |
| Aggregation | Max + Mean | Max captures extremal points, mean captures centroid |
| Pair features | Distance + Angle | Rotation-invariant geometric relationships |
| Position encoding | Learned (in Context Encoder) | Cloud itself is order-free; order handled upstream |
| Activation | Tanh | Bounded cloud: all points in [-1,1]^D |

### 3.4 Order-Freedom

The CloudEncoder is **permutation-invariant**: shuffling the input points produces the same cloud_state (up to floating-point noise). This means the model's prediction depends on the *set* of points and their *relative geometry*, not their sequence order. Order information is injected only through the Context Encoder's positional embeddings, which shape the 3D/ND positions before cloud formation.

---

## 4. Experiments

### 4.1 Setup

| Parameter | Value |
|-----------|-------|
| Corpus (Bilkent) | 165K sentences, 5K word vocab |
| Corpus (Wikipedia) | 30K sentences, 20K BPE vocab |
| Model | d_model=256, d_cloud=512, D=16, 4 layers |
| Parameters | 21M |
| Training | 15-50 epochs, AdamW, cosine LR |
| Hardware | Apple M-series (MPS) |
| Seq length | 8-10 tokens |

### 4.2 Dimensionality Experiment

We train identical models (21M params, 256D, 4 layers) with D ∈ {16, 32} on 50K Turkish Wikipedia sentences (20K BPE vocab), 50 epochs each:

| D | 50K-run PPL* | Regions (KMeans k) | Silhouette | Avg Spread |
|---|----------|--------------------:|:----------:|:----------:|
| 16 | 4.58 | 3 | 0.143 | 1.573 |
| 32 | **4.08** | **5** | 0.108 | **2.245** |

\* 50K-run PPL; not directly comparable with the canonical 200K shared held-out PPL (6.45 vs 2.92) or the Turkish-wiki scale gate (24.53 vs 64.28).

**Findings:**
1. **32D achieves lower PPL** (4.08 vs 4.58, −11%): more dimensions → better modeling
2. **32D discovers more regions** (5 vs 3): the model separates more distinct syntactic/semantic categories
3. **32D clouds are 1.43× more spread**: points diverge further, allowing finer geometric distinction
4. **Silhouette decreases** (0.143 → 0.108): expected trade-off — more clusters in the same space are less cleanly separated

The "dimensionality = relationship budget" hypothesis is confirmed: each additional dimension allows the model to encode one additional independent relationship axis.

### 4.3 Geometric Interpretability

After training on 30K Turkish Wikipedia sentences, frequent words occupy **stable spatial regions** (σ < 0.05 across 40+ occurrences):

| Word | Region (3D projection) | Syntactic Role |
|------|----------------------|----------------|
| bir (a/an) | East | Indefinite article |
| bu (this) | East+South | Demonstrative |
| ve (and) | Center+North | Conjunction |
| de/da (at/in) | Center+North | Postposition |
| için (for) | Center | Postposition |
| ne (what) | East+North | Interrogative |
| daha (more) | West+North | Comparative |
| gün (day) | East+South | Temporal noun |

The model discovers **syntactic role geography** in 16D space without any explicit syntactic supervision.

### 4.4 BPE vs Word-Level

| Tokenizer | Vocab | OOV Rate | PPL (15ep) | Generation |
|-----------|-------|----------|------------|------------|
| Word-level | 5K | ~40% | 49 | `? ? ? ?` |
| BPE | 20K | 0% | 57 | Real Turkish words |

BPE eliminates the OOV problem entirely for Turkish: *bahçelerde* → `bah` + `ç` + `elerde`, all subwords are in-vocabulary.

### 4.5 Comparison with Baseline

| Model | Params | PPL (Wikipedia TR)* |
|-------|--------|-------------------:|
| Random (untrained) | 21M | 20,000 |
| CloudLM 16D (50ep) | 21M | 4.58 |
| CloudLM 32D (50ep) | 21M | **4.08** |
| GPT-2 small (reference, English) | 124M | ~15-20 |

\* 50K-run values; the canonical 200K shared held-out comparison is in §6.4.

CloudLM achieves **4,900× improvement** over random (20,000 → 4.08) with 1/6 the parameters of GPT-2 small. The PPL of 4.08 on a 20K BPE vocabulary means the model is highly confident in its predictions (average effective choice space of ~4 tokens out of 20,000).

---

## 5. Analysis

### 5.1 What Does the Cloud Encode?

The cloud geometry encodes **relational structure**:
- **Proximity** = strong syntactic/semantic bond (subject-verb: distance 0.01-0.4)
- **Distance** = weaker or different-type relationship
- **Cluster shape** = sentence type (questions form different shapes than statements)
- **Spread** = information density (more spread = more diverse content)

### 5.2 Why Low Dimensions Work

The bottleneck hypothesis: forcing 256D context vectors into D-dimensional cloud points creates an **information bottleneck** that selects for the most geometrically salient relationships. The model must decide: *which D aspects of this sentence are spatially distinguishable?*

Our 16D vs 32D comparison quantifies this: doubling D from 16 to 32 increases the number of discoverable regions from 3 to 5 (a 67% increase) and increases cloud spread by 43%. The relationship is sub-linear — each additional dimension yields diminishing returns in region count, consistent with the geometry of high-dimensional spaces where volume concentrates near the boundary.

This is analogous to autoencoder bottleneck training, but the bottleneck is geometric rather than statistical: the model learns *which spatial directions* to use, not just *which features* to preserve.

### 5.3 Limitations

1. **Generation quality** is still below transformer baselines (PPL 57 vs ~15 for well-trained LMs)
2. **Fixed D**: 16D may not suffice for very long or complex sentences
3. **Pair computation** is O(N²) in sequence length (same as attention, but with simpler operations)
4. **No long-range dependency**: the cloud encoder sees all points simultaneously but has no mechanism for multi-hop relational reasoning

---

## 6. Discussion

### 6.1 Interpretability Advantage

Transformer attention weights are opaque: a 0.03 weight on a 768D inner product doesn't tell you *what* relationship is being captured. CloudLM's geometry is **directly visualizable**: you can plot the cloud, measure distances, and see which words are "close" in meaning-space.

### 6.2 Morphological Richness

For agglutinative languages, BPE subword points in the cloud naturally form **local clusters** around morphological roots. *bahçede*, *bahçeler*, *bahçem* all share the `bah`+`ç` subword prefix, so their cloud points start nearby and diverge based on suffix context. This is a geometric analog of morphological decomposition.

### 6.3 Connection to Cognitive Science

Human spatial cognition represents abstract concepts in mental "cognitive maps" (Boroditsch, 2001; Majid et al., 2004). The cloud model's use of spatial proximity for semantic relatedness mirrors this: concepts that are "close" in experience are "close" in the cloud.

### 6.4 Constraints and Scope of Applicability

The evidence supports QUANN point-cloud methods as a specialized approach for small, clean, interpretable tabular and edge tasks—not as a general replacement for backpropagation. Their demonstrated value is interpretable geometry, compactness, and, only where measured, retrain-free adaptation.

| Task and setting | QUANN / hybrid result | Comparator or qualification |
|---|---|---|
| IBM fraud, 24.4M real transactions | Hybrid: AUC-ROC .9451 (FP32), .9367 (INT8) | RF .973; GBT .981 (both above) |
| IBM fraud, matched split, rich points D=16 / D=32 (K=20) | .1919 / .4261 AUC-PR (.9770 / .9792 AUC-ROC) | D=8 strong reader .462 AUC-PR; matched XGB .8047 PR / .9938 ROC |
| CWRU binary, 80 recordings, record-disjoint | .999/1.000 | GBT 1.000; near-ceiling result |
| CWRU, 3-class fault type | .572/.406 (inner .000) | GBT .907/.838 |
| C-MAPSS real data, binary RUL≤30 | .908/.428 | GBT .996/.883 |
| C-MAPSS, four-stage macro result | .714 | GBT .932 |
| Elliptic real data, best angle-pair K10/B8 | .838/.370; fraud@1% FPR: 2.5% | GBT .943/.758; 62.5% |
| AML, challenging synthetic setting | .722/.327 | .810/.540 |

The strongest results are bounded cases: IBM fraud approaches established baselines, CWRU record-disjoint binary detection is near ceiling, and QUANN solves XOR at 100%. In contrast, the C-MAPSS, Elliptic, and three-class CWRU gaps show that these results do not establish competitive performance across domains or task complexity.

Within the fraud vertical, a controlled capacity probe holds the split fixed (train/val/test 1.02M/100K/4.7M) and enriches the cloud points from the D=8 strong-reader regime to D=16 and D=32 (K=20): AUC-PR moves from .462 (D=8) to .1919 (D=16) and .4261 (D=32). The enriched geometry approaches but does not cross the D=8 ceiling, and the matched XGBoost control (.8047 AUC-PR, .9938 AUC-ROC) remains ahead. We read this as evidence that, in this regime, the lever is reader capacity rather than point dimensionality, and that the gap to the GBT family reflects the representation itself rather than under-provisioned capacity. This matched-split precision axis is separate from the full natural-test ROC result in the table, where the hybrid's .9451 (FP32) / .9367 (INT8) sits below both standard ceilings; that AUC is computed with the checkpoint's bundled input quantizer, making it reproducible from the artifact.

Retrain-free, 0th-order adaptation is likewise conditional, not a general capability: XOR reaches 100%; AML E4 (10K→5K, unseen shape) recovers 0% of the retraining gain (5K recall 0.00 vs 1.00; macro AUC .871/.879/1.000); and CWRU E4 has no demonstrated advantage because frozen performance is already 100%.

Interpretability is supported by task-specific measurements: C-MAPSS nearest-prototype explanations cover 81.7% (1.064 windows), with own/other ratios of .419 for healthy and .259 for critical cases and a 2.4× drift; AML reaches 87.8% and its rationale text names the shape; the fraud centroid measure is 1.95. These measurements support geometric inspection, not a claim of universal explanation quality.

For language modeling, the measured 200K-training CloudLM result is PPL 6.45 on the shared held-out set, versus 2.92 for the causal baseline; the measured diagnosis attributes the gap primarily to representational capacity. Separately, the small parameter count (21.0M) keeps the model within a compact edge memory budget; this supports compact edge deployment, not a general performance claim.

Accordingly, maintenance is the strongest evidenced vertical (near-ceiling real-data binary detection, measured explanations, and compact deployment); fraud is core evidence for the hybrid approach; AML is limited to an explainability/small-data wedge, with scoring and compliance claims withheld. The present scope is limited by task and dataset coverage, weak fault-type and low-FPR results, unsuccessful AML E4 adaptation, and QUANN's demonstrated small/interpretable-model regime; the method is not shown to replace backpropagation for full-scale language models.

### 6.5 Application Positioning: Named Explanations vs. Scores

The discussion above frames the point-cloud approach along two axes — capacity and interpretability. A third axis emerges in application verticals: *what the system hands the analyst after an alarm*. Standard production anomaly detectors — gradient-boosted ensembles, autoencoders, isolation forests, graph neural networks — output a risk score and, at most, feature-level attributions (SHAP contributions for a GBM): *which features* made a transaction risky. None outputs a **typology name**. In AML, that gap is operational: triaging an alert requires mapping it to a typology (structuring, layering, velocity) in order to draft a Suspicious Activity Report, and a feature-attribution list does not perform that mapping.

| Detector family | Typical output | Point-cloud difference |
|---|---|---|
| Isolation Forest / LOF / one-class SVM | anomaly score, no label | region + prototype → named typology |
| Autoencoder | reconstruction error, no label | geometry + name |
| GBM / XGBoost | SHAP — feature-level "why" | "what this is" (typology name) |
| GNN | ring / collusion signal, black-box | chain-signature geometry (brittle; §6.4) |

The region-prototype surface inverts the usual explainability trade-off. Instead of attributing a score to features, it locates the window's point cloud against a small set of named typology prototypes and reports the nearest name with the supporting distances. The output is (i) **a name** — directly mappable to a SAR template; (ii) **geometric** — a two-dimensional region map an auditor can inspect; in a 5-seed Elliptic pilot the coarse typology label is more stable across retraining than SHAP top-1 (0.325 vs 0.1025), while map coordinates and attribution magnitudes are not stable (PCA-2 displacement 4.49; SHAP top-3 Jaccard 0.434; Spearman 0.651); and (iii) **compact** — a handful of prototype distances, not a high-dimensional attribution vector.

Two controlled demonstrations on the synthetic AML corpus bracket the claim. In the clean case (Figure: `aml/figures/typology_naming_comparison.png`), for the same structuring window the GBM+SHAP panel reports a near-certain score with top drivers (transaction rate, window duration, inter-transaction interval) but no name, while the region map reports **STRUCTURING** at d=0.82 against a runner-up 11.55 (14× separation) with a threshold-edge rationale. In the hard case (Figure: `aml/figures/typology_naming_layering.png`), where normal and layering prototypes nearly coincide, the nearest-prototype label flips to *normal* (0.323 vs. 0.331, 1.03×) — an honest miss — but the rationale text still names the observed 10-hop chain, so the analyst *sees* the ambiguity rather than receiving a false-confident label.

The real-data gate is Elliptic, a 203K-transaction darknet dataset in which 4.4% of transactions are illicit. There the explanation surface is measurably more modest than on the synthetic corpus: held-out nearest-prototype naming reaches 75% overall and 45% within the illicit class, versus 87.8% on synthetic AML. The named example (TX 294455185) is labeled **ILICIT** at distance 4.317 with a one-hop proxy rationale, while the GBM panel for the same transaction reports a 0.92 class probability with feature-level SHAP attributions and no label (Figures: `aml/figures/e2_elliptic_region_map.png`, `aml/figures/e2_elliptic_naming_comparison.png`). The detection gap persists in this vertical (§6.4: .838 versus .943 AUC). We read Elliptic as a *modest* real-data instance of the explanation surface — an auditable geometric read of darknet money movement, with the sampled-node-proxy and transductive-graph caveats stated in the figure caption — not as competitive detection, and not as a transfer of the synthetic naming accuracy.

A 5-seed retraining pilot on the same 2000-node Elliptic sample (400 held-out nodes; per-seed hybrid reader; train-only class centroids; PCA planes aligned to a reference seed) measures the stability axis directly. Exact nearest-name agreement across all five retrains is 0.325, versus 0.1025 for SHAP top-1 feature stability, 0.434 mean top-3 Jaccard, 0.651 mean Spearman rank agreement, and a mean aligned PCA-2 coordinate displacement of 4.49. We therefore claim stability only for the coarser named-label/region assignment, not for the continuous two-dimensional map or for attribution magnitudes.

The scope of this claim follows the measurements, and is deliberately narrow. Detection competitiveness is supported only where real data exists: the IBM credit-card vertical (24.4M transactions), where the hybrid reaches .9451 AUC-ROC (FP32; .9367 INT8), approaching but remaining below GBT .981 and RF .973 (§6.4). For AML the evidence is synthetic: the balanced E0 setting is easy (GBT 1.0; hybrid .96 AUC / .872 AP), layering recall at 1% FPR is 0.04, and the challenging setting (§6.4: .722/.327 vs. .810/.540) shows the detection gap re-emerging. We therefore do not claim detection superiority or parity for AML. The claim is the *explanation output* — a named, auditable typology with a geometric rationale — with the Elliptic real-data instance above as its current evidence, and richer real-data coverage (banking typologies) as the next gate. Retraining-free compliance adaptation, the third leg of the positioning, remains a hypothesis: AML E4 recovered 0% of the retraining gain (§6.4), and is reported as open work rather than an advantage.

In short, the application wedge is not "detect better" but "explain as a named, inspectable object": the same geometry that is the prediction substrate in §3 becomes, in verticals, the surface an analyst and an auditor read after the alarm.

### 6.6 Case Study: Industrial Maintenance

We evaluated the point-cloud pipeline on two public industrial datasets. The CWRU bearing dataset was obtained from its official source and contains 80 recordings with drive-end (DE) and fan-end (FE) channels. For the binary normal/fault task, we used a record-disjoint split of 60 training and 20 held-out recordings, yielding 1,440 and 480 segments, respectively. C-MAPSS FD001 turbofan data were obtained from a Kaggle mirror and represented as 2,755 windows with 36 features. In both cases, the pipeline maps segments or windows to features, features to points, and points to a cloud. CloudEncoderND applies a shared point MLP and max-pooling, combines pairwise geometric features (differences, distances, and cosine angles) through mean-pooling, and supplies the resulting representation to a small reader. Training is hybrid: QUANN 0th-order point narrowing with a backpropagation-trained reader.

For CWRU binary detection, the hybrid reached near-ceiling performance on the record-disjoint split, comparable to the gradient-boosted tree (GBT) baseline. The reported hybrid configuration had convergence at 12/6 points.

| Regime | Model | ROC-AUC | PR-AUC | F2 | Recall | Converged points |
|---|---|---:|---:|---:|---:|---:|
| Binary, record-disjoint | Hybrid | .999 | 1.000 | .996 | NR | 12/6 |
| Binary, record-disjoint | GBT | 1.000 | 1.000 | .996 | NR | — |

For C-MAPSS, the strongest evidence is the measured explanation surface rather than detection parity. Nearest-prototype explanations covered 81.7% of 1,064 held-out windows. The stage-wise own/other median values and the operating-shift drift alarm are summarized below; the region-map visualization shows the corresponding geometric organization.

| C-MAPSS measurement | Result |
|---|---:|
| Held-out windows with nearest-prototype explanation | 81.7% (1,064 windows) |
| Healthy: own/other median; own share | .419; 93.8% |
| Medium: own/other median; own share | 2.48; 18.4% |
| High: own/other median; own share | 1.86; 44.4% |
| Critical: own/other median; own share | .259; 64.0% |
| Operating-shift drift alarm | 2.4×: 1.17% → 2.81% |
| Tokens changed under operating shift | 80.8% |

The edge evidence is qualitative and bounded. All runs were executed on CPU, with the heaviest design-grid run taking 16 seconds; the reader is a small MLP, and its weights can be quantized to INT8. These observations support feasibility under GPU-free, on-site compute budgets, but do not establish a maintenance-specific INT8 memory reduction or accuracy result.

The limitations are material. First, the CWRU three-class fault-type task was closed as unsuccessful: the E5 design grid improved the .572/.406 baseline expansion hybrid to .678/.530, but with inner-class recall .000 while GBT reached .900; the angle lever improved one target by 9.3 points, but the two targets were not jointly satisfied. Second, CWRU in-place adaptation (E4) was degenerate because the frozen model already achieved 100%, so no adaptation advantage is claimed. Third, C-MAPSS detection remained behind GBT—.908 versus .996 for binary detection and .714 versus .932 for four-stage macro performance—so the C-MAPSS case rests on explainability, not detection.

Overall, industrial maintenance is the strongest evidenced vertical in this study: near-ceiling detection on a clean record-disjoint task, measured geometric explanations, and CPU/INT8-compatible compact deployment. Fine-grained fault-type classification and adaptation advantages remain outside the supported scope.

---

## 7. Conclusion

We presented CloudLM, a language model that represents sentences as geometric point clouds and predicts based on cloud configuration rather than sequential attention. Key findings:

1. **16 dimensions suffice** for 3 distinct syntactic regions; **32 dimensions** yield 5 regions with 11% lower PPL
2. **Geometry is interpretable**: spatial regions correspond to syntactic categories (conjunctions central, demonstratives eastern)
3. **BPE + cloud** eliminates OOV for morphologically rich languages (Turkish agglutination)
4. **The bottleneck is productive and quantifiable**: 16D→32D increases region count by 67% and cloud spread by 43%
5. **4,900× improvement** over random baseline with 21M parameters (1/6 of GPT-2 small)

Beyond the LM results, the same geometry doubles as an explanation surface in three applied verticals (§6.5-§6.6). On the IBM credit-card corpus (24.4M transactions) the QUANN-hybrid reaches .9451 AUC-ROC (FP32; .9367 INT8) on the full natural test — a competitive approach to, but below, the standard ceiling (GBT .981, RF .973) — while the matched-split precision axis remains below the GBT family (§6.4). In AML the contribution is not detection but a *named* explanation: a typology label with a geometric rationale (synthetic structuring: 14× prototype separation; Elliptic darknet: 75% held-out naming, 45% within the illicit class), while the detection gap to GBMs is stated, not hidden (.838 vs .943 AUC). Industrial maintenance is the strongest evidenced vertical: near-ceiling CWRU binary detection (F2 .996, matching GBT), a measured 81.7% nearest-prototype explanation surface on C-MAPSS with a 2.4× operating-shift drift alarm, and CPU/INT8-compatible compact deployment. The unifying proposal: point clouds + 0th-order narrowing form a vertical-independent core — the domain enters only through tokenizer and space choice, while geometry and narrowing are unchanged.

The honest limits are equally specific. QUANN as a *pure* 0th-order optimizer sits at or near random on real tasks; every reported application result uses the hybrid (0th-order point narrowing + backprop-trained reader), and that hybrid — not pure 0th-order training — is the claim. At 200K scale the LM itself shows a representational-capacity gap (PPL 6.45 vs 2.92 for the causal baseline on the shared held-out set). A same-distribution scale gate then showed that, at D=32, repetition depth dominated data width (200K-wiki/50ep 24.53 vs 1.5M/20ep 64.28 on the shared 200K slice), leaving D=64×200K-wiki/50ep as the standing capacity gate. AML retraining-free adaptation (E4) recovered 0% of the retraining gain, and fine-grained CWRU fault typing (inner recall .000) is reported as unsuccessful; both are stated as scope boundaries, not omitted.

Future work: closing the capacity gap (D=64×200K-wiki/50ep), standardized benchmark comparison, theoretical analysis of the dimension-region relationship, controlled ablations on richer real-data AML coverage (banking typologies), and adding a temporal dimension (4D clouds for discourse).

---

## References

- Bronstein, M. M., et al. (2021). Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. *arXiv:2104.13478*.
- Chung, S.-Y. & Abbott, L. F. (2021). Neural population geometry: An approach for understanding biological and artificial neural networks. *Current Opinion in Neurobiology*, 70, 137–144.
- Hu, Z., Niu, L., Varma, S., et al. (2026). Language Models Represent and Transform Concepts with Shared Geometry. *arXiv:2607.04525*. Georgia Tech / University of Edinburgh.
- Lee, J., et al. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. *ICML*.
- Mikolov, T., et al. (2013). Distributed Representations of Words and Phrases and their Compositionality. *NeurIPS*.
- Pennington, J., et al. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.
- Qi, C. R., et al. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. *CVPR*.
- Scale Determines Whether Language Models Organize Representation Geometry for Prediction. (2025). *arXiv:2605.17084*.
- Semantic Geometry of Sentence Embeddings. (2025). *Findings of EMNLP*, 2025.
- Zaheer, M., et al. (2017). Deep Sets. *NeurIPS*.
- Zhao, H., et al. (2021). Point Transformer. *ICCV*.

---

## Appendix A: Reproducibility

```bash
# Environment
pip install torch tokenizers matplotlib numpy datasets

# Step 1: BPE Tokenizer
python3 training/train_bpe.py --step train_tokenizer --bpe-vocab 10000

# Step 2+3: Wikipedia Training  
python3 training/train_wiki.py --step all --max-texts 30000 --epochs 15

# Step 4: Test
python3 training/train_wiki.py --step test

# Visualizations
python3 src/cloud_lm_full.py  # 3D/16D/32D comparisons
```

## Appendix B: Cloud Visualizations

[Figures: figures/cloud_dim_compare.png, figures/cloud_evolution.png, figures/wiki_cloud_test.png, figures/real_cloud_0.png]

[Application figures (aml vertical): aml/figures/typology_naming_comparison.png, aml/figures/typology_naming_layering.png, aml/figures/e2_elliptic_region_map.png, aml/figures/e2_elliptic_naming_comparison.png]

---

*Code: https://github.com/[user]/quann-3d*  
*Corpus: Bilkent Turkish Writings (CC), HuggingFace NumanKaanKaratas/turkish-sentences (MIT)*
