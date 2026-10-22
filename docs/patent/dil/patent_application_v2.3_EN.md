# PATENT APPLICATION — v2.3 (English Version)

**Application document compliant with Industrial Property Law No. 556**

| | |
|---|---|
| **Application No.:** | [to be assigned by TÜRKPATENT] |
| **Filing Date:** | [filing date = priority date] |
| **Inventor:** | [Inventor Full Name, National ID No., address] |
| **Applicant:** | [Applicant Name, address] |
| **Agent:** | [Patent attorney / firm] |
| **Title of the Invention:** | Low-Dimensional Point Cloud Geometry-Based Language Modeling System, Method and Computer Program |
| **Technical Field (IPC suggestion):** | G06F 18/00; G06N 3/04; G06N 5/02 |

> **CONFIDENTIALITY / PRIORITY WARNING:** This document is prepared for a patent application. Until the filing date (right of priority) is obtained, no publication (arXiv, conference, journal, blog, publicly accessible GitHub, podcast, presentation) relating to this invention may be made. Publication would have a devastating effect on the novelty requirement (self-depublication).

**Document contents:** 1. Title — 2. Description — 3. Claims — 4. Abstract — 5. Drawings (8 figures, separate pages).

**v2.3 change summary (October 16, 2026):**
1. The PPL table has been reconciled with verified run logs: the **training-side** PPL of the 50K runs (4.08/4.58) and the **shared held-out** PPL of the 200K run (6.45; causal reference 2.92) are given in separate tables; the "≈4.9×" statement is tied to the training-side number.
2. The final sentence of Claim 1 has been rewritten: the context encoder (100) may include attention; the characterizing feature is that the prediction derives from the D-dimensional point cloud geometry rather than from the d_model-dimensional context vectors.
3. The unmeasured "computational savings / total cost" promise in Claims 3 and 19 has been narrowed to a structural expression (D+4 ≪ d_model).
4. Embodiment parameters verified against run logs (d_cloud = 512; 50,000 sentences); 2.7.8 adds fraud / AML / maintenance application examples; 2.7.4 adds the D=64/128 pre-test finding and the capacity limitation.
5. Claims 11–12 merged; the "has been trained" outcome claim narrowed to a "capable of representing" capability statement; Claims 10/13/14 retained.

---

## 1. TITLE

**Low-Dimensional Point Cloud Geometry-Based Language Modeling System, Method and Computer Program**

---

## 2. DESCRIPTION

### 2.1 Technical Field

The present invention relates to the field of natural language processing (NLP), and in particular to language modeling and text generation. More specifically, the invention provides a system, a method and a computer program for predicting a next token from the **geometric configuration of a point cloud** (inter-point distances, angles, cluster spread) formed by projecting each token of a sentence to a point in low-dimensional (3 ≤ D ≤ 64) Euclidean space.

The invention is designed in particular for application in language modeling of morphologically rich (agglutinative) languages — Turkish, Finnish, Hungarian, Japanese — and in systems operating with low latency and low memory requirements.

### 2.2 Prior Art

#### 2.2.1 The Dominant Paradigm: The Transformer

Today, the dominant paradigm in language modeling is the transformer architecture (Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I., 2017, *NeurIPS 30*). Transformers:

- Process a sentence as an ordered sequence of tokens;
- Perform computation at O(N²) sequence complexity via the self-attention mechanism;
- Represent each token in a 768–4096 dimensional hidden space;
- Predict the next token from the last hidden state vector — that is, from **a single vector**;
- Inject ordering information via positional encoding.

The known limitations of this approach are as follows:

1. **High-dimensional hidden space:** The majority of 768–4096 dimensional vectors is functionally unused; as dimension increases, computational and memory costs grow linearly.
2. **Attention opacity:** Attention weights (high-dimensional inner products) do not permit interpretation of which semantic relation is captured; the decision mechanism is embedded in an uninterpretable weight matrix.
3. **KV cache:** In long sequences, memory consumption grows as O(N × d_model); this is a decisive limitation for edge devices and low-latency applications.
4. **Inefficiency in morphologically rich languages:** In agglutinative languages such as Turkish, a single word carries 10+ morphological units ("bahçelerden" = bahçe + ler + den). Transformers process these units as independent subword tokens; the morphological structure is not represented spatially and is only implicitly encoded in the attention weights.

#### 2.2.2 Related Work

- **PointNet** (Qi, C., Yi, L., Su, H., & Guibas, L. J., 2017, *IEEE/CVF CVPR*, 77–80): Processes 3D objects as point clouds; produces an order-invariant set feature via a shared MLP + symmetric (max-pooling) set operator. However, it is designed for 3D vision input; point positions enter as **raw sensor data** and have not been used in language modeling.
- **Deep Sets** (Zaheer, M., Kottur, M., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., & Smola, A. J., 2017, *NeurIPS 30*) / **Set Transformer** (Lee, J., Lee, J., Kim, M., Kosiorek, A. R., Choi, S., & Teh, Y. W., 2019, *ICML*, 3714–3723): Process token sets without ordering. However, they reduce set features to **a single vector via pooling**; because the **spatial position information of the points is not present as input**, the geometric structure is not preserved.
- **Word2Vec** (Mikolov, T., Chen, K., Corrado, G., & Dean, J., 2013, *arXiv:1301.3781*) / **GloVe** (Pennington, J., Socher, R., & Manning, C. D., 2014, *EMNLP*, 1532–1543): Position words in a vector space. However, they operate at the **single word → single point** level and do not use a multi-point cloud geometry at the sentence level.
- **Hu et al. (2026)** (Hu, Z., Niu, L., & Varma, S., 2026, *arXiv:2607.04525* [cs.CL]; DOI 10.48550/arXiv.2607.04525): Formulates concept representations inside transformers/LLMs as **point-cloud manifolds** within the neural population geometry framework and **analyzes** contextual transformations as vector fields (post-hoc interpretability). It does not use the geometry as the mechanism for predicting the next token and does not propose a new architecture.
- **Chung & Abbott (2021)** (Chung, S., & Abbott, L. F., 2021, *Nature* 592(7853), 117–122): Develops the neural population geometry framework for neuroscience. It is not used as an NLP architecture.
- **Xu (2026) — "Scale/PGA"** (Xu, W., 2026, *arXiv:2605.17084* [cs.LG]; DOI 10.48550/arXiv.2605.17084): Using the *Subspace PGA* metric, **investigates** to what extent the **distance structure** of a layer is aligned with the readout subspace of its unembedding matrix; demonstrates scale dependence. It does not propose a new architecture.

#### 2.2.3 The Gap in the Prior Art

No existing work uses point cloud geometry (distances, angles, cluster shape) **as the mechanism for predicting the next token**. Existing work uses geometry either (a) as 3D vision input (PointNet), (b) as a pooled set feature that does not include spatial position (Deep Sets, Set Transformer), or (c) as a post-hoc analysis of an existing model (Hu et al., 2026; Xu, 2026).

The closest works in the prior art — Hu et al. (2026) formulate concept representations as point-cloud manifolds; Xu (2026) examines layer distance structure with the *Subspace PGA* metric — use geometry only for **analysis of an existing model**; neither uses geometry as the **prediction mechanism** for the next token, and neither proposes a new architecture.

The fundamental gap in the prior art is as follows: **the geometry itself serving as the computational substrate** — that is, the prediction of the next token being made directly from the spatial relations (distances, angles) of learned low-dimensional point positions — is unknown in the state of the art.

### 2.3 Technical Problem

The present invention aims to solve the following technical problems:

1. Reducing the unnecessary computational and memory cost of high-dimensional hidden space in language modeling;
2. Making the model's prediction decisions **geometrically measurable and interpretable** (distances, angles, regions can be computed directly);
3. Providing a spatial representation of morphological structure in morphologically rich languages (Turkish, Finnish, Hungarian);
4. Ensuring that the prediction of the next token is made from low-dimensional, geometrically interpretable quantities (distances, angles, regions) rather than from high-dimensional, opaque attention weights; keeping the representational dimension of the pair-geometry computation (D+4) lower than the dimension of the attention weights (d_model), thereby structurally reducing the computational load.

### 2.4 The Core of the Invention (Solution)

The core of the invention is the projection of each token of a sentence to a **point in low-dimensional (3 ≤ D ≤ 64) Euclidean space** and the coding of the meaning of the sentence in the **geometry of this point cloud** (inter-point distances, angles, cluster spread). The prediction of the next token is made from the current geometric configuration of the cloud — via a permutation-invariant set-state vector — and not from a single vector of the last hidden state.

This approach is inspired by the principle of "neural population geometry" in neuroscience (Chung & Abbott, 2021): in response to a stimulus, a population of neurons forms not a single point but a point cloud, and the information lies in the shape and position of this cloud. The invention is the first system that applies this principle to language modeling **as a computational mechanism**.

### 2.5 Technical Effects

The invention provides the following technical effects over the prior art:

1. **Interpretability (primary effect):** Distance, angle, and region (cluster) statistics can be directly computed, measured, and visualized; the geometric relation (which point pair, which region) from which the model's next-token decision originates can be inspected. Unlike attention weights (high-dimensional inner products), geometric features are low-dimensional, human-interpretable quantities.
2. **Memory efficiency (representation level):** The D-dimensional point representation (N × D) is more compact than the d_model-dimensional context representation (N × d_model); for D = 16, d_model = 256, the point layer requires 1/16 of the memory of the context layer. (The total memory of the system is determined by the context encoder; the savings lie in the geometric representation itself.)
3. **Morphological representation:** In agglutinative languages, subword tokens derived from the same root tend to cluster spatially in the cloud; affixes tend to separate in directions different from the root (observed in region maps; see Fig. 4, 8).
4. **Ordering flexibility:** The aggregation of the cloud encoder is permutation-invariant; however, because point positions are order-sensitive via the context encoder, the cloud geometry is indirectly dependent on token order (see 2.7.1).
5. **Dynamic scalability:** D can be adjusted according to the sentence length or the measured cloud complexity.
6. **Computational efficiency (structural; alternative embodiment):** When the context encoder is chosen attention-free (e.g., linear / Mamba-like, O(N)), the system carries O(N) context encoding + O(N²)×(D+4) pair-geometry computation; a self-attention-based system carries O(N²)×d_model attention computation. Since D+4 ≪ d_model (e.g., 36 ≪ 256), the per-point-pair computational load is structurally lower. The total latency depends on the chosen context encoder and hardware; this application does not carry a measured total-cost claim (see 2.7.6.4, Claim 19).

### 2.6 Brief Description of the Drawings

**Fig. 1:** The system architecture of the invention — 100: context encoder, 110: ND projection module, 120: cloud encoder (121: point MLP, 122: pair MLP, 123: set fusion), 130: prediction module (131: auxiliary point output), 140: tokenization module, 150: interpretation module.

**Fig. 2:** Point cloud example — 3D projection of the sentence "kedi bahçede uyudu" ("the cat slept in the garden"); 200: token points, 210: pair distance lines.

**Fig. 3:** Cloud evolution during generation — the incremental construction of the sentence (1 → 2 → 3 → 4 points); 300: successive cloud configurations.

**Fig. 4:** 16D and 32D region maps — PCA 2D projection, KMeans clusters; 400: 16D regions (3 regions), 410: 32D regions (5 regions).

**Fig. 5:** Word region geography — average 3D positions of frequent words (prepositions in the center, case markers in the east-south).

**Fig. 6:** Pair geometry — components of the input vector between two points: 600: difference vector, 610: distance, 620: cosine angle, 630: norms.

**Fig. 7:** Training curve — 50 epochs; 700: PPL curve (training side; ~20,000 → 4.08).

**Fig. 8:** Morphological clustering — positions in the cloud of subword tokens of the root "bahçe"; 800: root sub-cluster, 810: affix sub-clusters.

### 2.7 Detailed Description of the Invention

#### 2.7.1 System Architecture

As shown in Fig. 1, the system comprises a **context encoder** denoted 100, an **ND projection module** denoted 110, a **cloud encoder** denoted 120, a **prediction module** denoted 130, a **tokenization module** denoted 140, and preferably an **interpretation module** denoted 150.

**(1) Tokenization Module (140)**

Module 140 uses byte-level Byte Pair Encoding (BPE). Special tokens `<pad>`, `<eos>`, `<unk>` are defined; the vocabulary size is in the range of 5,000–50,000. Thanks to the byte-level pre-tokenizer, any writing system can be represented and **no OOV (out-of-vocabulary) problem arises**.

In agglutinative languages this module performs a critical function: the word "bahçelerden" is segmented into the subword tokens `bah` + `ç` + `eler` + `den`. These subwords belong to the same morphological root in the cloud described below and tend to cluster spatially (Fig. 8).

**(2) Context Encoder (100)**

Module 100 is an L-layer (preferably L = 2–8) transformer encoder using a d_model-dimensional (preferably 128–512) hidden space. Its input is the token sequence [t₁, …, tₙ], and its output is contextual vectors c₁, …, cₙ ∈ ℝ^d_model. Its role is to transform raw token embeddings into context-sensitive vectors.

This module may be any context encoder known in the prior art (transformer, RNN, linear attention-free architectures); the characterizing features of the invention lie beyond this module. The context encoder may include a self-attention mechanism; the characterizing feature is that the next token is derived **not from the high-dimensional output of this module but from the geometry of the D-dimensional point cloud via steps (3)–(5) below**.

**(3) ND Projection Module (110)**

Module 110 transforms each contextual vector cᵢ ∈ ℝ^d_model into a point pᵢ ∈ [-1,1]^D in D-dimensional (3 ≤ D ≤ 64) Euclidean space. Preferably, module 110 uses a two-layer MLP of dimension d_model → d_model → D and a Tanh activation.

**Critical technical feature:** The Tanh activation confines all points to the [-1,1]^D hypercube. This (a) keeps the cloud bounded and comparable across sentences, (b) keeps the pair-geometry features (distance, angle) within bounded ranges and numerically stable, and (c) ensures that the clouds of different sentences are compared in the same reference frame.

**(4) Cloud Encoder (120)**

Module 120 produces a set-state vector cloud_state ∈ ℝ^d_cloud (preferably d_cloud = 256–512) from the point cloud {p₁, …, pₙ} ⊂ ℝ^D. It comprises three sub-components:

**(a) Point MLP (121):** Each point pᵢ is transformed into a point feature fᵢ ∈ ℝ^d_cloud by a shared-weight MLP. All points use the same weights.

**(b) Pair MLP (122):** For each point pair (pᵢ, pⱼ) an input vector is constructed:

```
x_ij = [ p_i − p_j ,  ‖p_i − p_j‖ ,  cos(θ_ij) ,  ‖p_i‖ ,  ‖p_j‖ ]  ∈  ℝ^(D+4)
```

- `p_i − p_j` (D dimensions): direction and magnitude;
- `‖p_i − p_j‖` (1 dimension): Euclidean distance;
- `cos(θ_ij)` (1 dimension): cosine of the angle between the vectors drawn from the origin of the space to the two points;
- `‖p_i‖`, `‖p_j‖` (2 dimensions): distances of the points from the origin.

For D = 16, the total input is 20 dimensional. Note that the pair MLP operates on every point pair and thus has O(N²) complexity; this is the same order of magnitude as the asymptotic complexity of the self-attention mechanism. The difference between the two lies in the **representation dimension** processed per pair: the pair-geometry input is (D + 4)-dimensional and carries a lower-dimensional, geometrically interpretable input compared to the high-dimensional (d_model) inner products of the attention weights. When the context encoder is chosen attention-free (O(N)) (see 2.7.6.4, Claim 19), since the representational dimension of the pair computation (D+4) is lower than the dimension of the attention weights (d_model), the computational load is structurally reduced; the total-latency measurement depends on the chosen architecture and hardware.

**Rotation invariance:** Distance and cosine angle are invariant to rotations of the coordinate system. Therefore, the information is not which direction the cloud "looks" in space, but the **positions of the points relative to each other**; this increases the robustness of the representation to rotational noise.

**(c) Set Fusion (123):**

```
cloud_state = Fusion( MaxPool(f_i) , MeanPool(g_ij) )  ∈  ℝ^d_cloud
```

Max-pooling captures the "most extreme" feature of the cloud; mean-pooling captures the "average relation" of the cloud. The fusion is performed, as a preference, by concatenation + MLP.

**Permutation invariance and order information:** Because the point MLP is shared and symmetric (max/mean) aggregation is used, the cloud_state is the same even if the order within the point **set** is changed; that is, the aggregation step itself is permutation-invariant. However, since the point positions pᵢ = proj(cᵢ) are produced via the context encoder (100) and the context encoder carries positional encoding, the point positions are **indirectly dependent on token order**: two sentences written with the same words in different orders have different point positions and therefore different cloud geometries. Thus the meaning of the sentence is encoded in the **geometric configuration of the cloud**; the aggregation step itself is order-insensitive.

**(5) Prediction Module (130)**

Module 130 produces the next token from the cloud_state vector:

- **Output 1:** word_logits ∈ ℝ^vocab — the vocabulary distribution of the next token;
- **Output 2 (131, preferably):** p_next ∈ ℝ^D — the predicted point position of the next token in the cloud.

The prediction of the next token is made from the set-state vector derived from the geometry of the D-dimensional point cloud (distances, angles, cluster statistics), and not from the d_model-dimensional contextual vectors themselves or from the attention weight matrix. The context encoder (100) may include self-attention — this is within the scope of the invention; the characterizing step is the prediction layer that operates through the geometry.

The modules of the system (100–150) may also be implemented as program instructions stored in a non-transitory, computer-readable recording medium for execution by a processor.

#### 2.7.2 Training Method

The total loss function:

```
L = L_CE(logits, t_next) + λ · L_MSE(p_next, proj(c_next))
```

- **CE term:** correct prediction of the next token;
- **MSE term (preferably λ = 0.1):** that the point of the next token lies at the correct projection produced by the context encoder (proj(c_next); here proj(·) is the projection module 110).

The second term ensures that the model learns not only "which word comes next" but also "where this word will fall in the cloud". This is a regularizing task that forces the cloud geometry to remain semantically consistent — that is, that the cloud does not disperse during generation.

#### 2.7.3 Inference Method

1. Take the tokens of the current sentence;
2. Context encoder (100) → contextual vectors c₁…cₙ;
3. ND projection (110) → point cloud {p₁…pₙ};
4. Cloud encoder (120) → cloud_state;
5. Prediction module (130) → next token + p_next;
6. The point p_next of the new token is added to the cloud; cloud_state is recomputed;
7. Steps 4–6 are iterated until the `<eos>` token is produced (Fig. 3).

At each step the cloud geometry is updated; the next prediction is made from the **configuration of the updated cloud**.

#### 2.7.4 Why Low Dimension? — The Relation Budget

The choice of D is not arbitrary. Our experiments show the following pattern:

| D | Independent relations the model can encode | Observed regions (KMeans) |
|---|---|---|
| 3 | 1 dominant relation (location-verb OR subject-verb) | 1–2 |
| 7 | 2–3 relations | 2–3 |
| 16 | 3–5 syntactic regions (prepositions, case markers, adjectives, time, comparison) | 3 |
| 32 | 5+ regions | 5 (67% increase over 16D) |

Each dimension is the number of **independent relation axes** the model can encode simultaneously. When D is increased, the number of regions and the mean spread grow **sub-linearly** (16D → 32D: spread 1.573 → 2.245, 43%). This is experimental evidence that D functions as a "relation budget" and forms the basis for selecting D according to sentence complexity (see Claim 10).

This principle, measured at D = 3, 7, 16, 32, supports the endpoints of the claimed 3 ≤ D ≤ 64 range; the other integer values in the range are within the design scope of the invention even if not individually measured.

**Capacity finding (20K sentences, 10-epoch pre-test):** D = 32, 64 and 128 variants were compared on the same corpus and vocabulary: causal reference 10.80, D = 32 201.18, D = 64 181.47, D = 128 189.47 (held-out PPL). D = 64 improved over D = 32, while D = 128 regressed. Together with the sensitivity analysis at the 200K scale, this shows that the main source of the held-out PPL gap is representational capacity: a 10-token cloud carries 10×32 = 320 numbers, whereas the hidden state of a transformer of the same length carries 10×256 = 2,560 numbers (~12.5%). The PPL gap is reported as a design variable to be closed by increasing D; full-scale (200K) training of D = 64 is ongoing.

#### 2.7.5 Example Embodiment (Specific Embodiment)

The invention has been realized with the following specific parameters:

- Context encoder: L = 4 layers, d_model = 256, 8 heads;
- Projection: 256 → 256 → D, Tanh; D = 16 or 32;
- Cloud encoder: d_cloud = 512; point MLP D → 512, pair MLP (D+4) → 512;
- Prediction module: 512 → 20,000 (BPE vocabulary) + 512 → D (auxiliary point);
- Total: ~21M parameters (D = 32 run: 20,999,648);
- Corpus: 200,000 Turkish sentences (Wikipedia); 20,000 BPE vocabulary;
- **Run A:** 50,000 sentences (49,833 sentences ≥9 tokens), 50 epochs, λ = 0.1;
- **Run B:** 200,000 sentences, 50 epochs, λ = 0.1 (for the shared held-out comparison).

Results:

**A) Training-side PPL (Run A; in-run report, over the training data):**

| Configuration | PPL (best epoch) | Regions | Spread |
|---|---|---|---|
| Untrained (21M) | ≈20,000 (vocabulary size; upper bound) | — | — |
| D = 16, 50 epochs | 4.58 | 3 | 1.573 |
| D = 32, 50 epochs | **4.08** | **5** | **2.245** |

The PPL of the untrained model equals the uniform distribution over the 20,000 vocabulary; 4.08 is reported as an improvement of ~4.9× over this lower bound. PPL 4.08 indicates that the effective choice space of the model over the 20,000 vocabulary is ~4 tokens — that is, that it carries high confidence in its predictions.

**B) Shared held-out comparison (Run B; same 2,000-sentence held-out set, same corpus, same 20,000 BPE vocabulary):**

| Model | Parameters | Held-out PPL |
|---|---|---|
| CloudLM, D = 32 (the present invention) | 21.0M | **6.45** |
| Causal transformer, d = 384, 4 layers (reference) | 22.7M | **2.92** |

**Honest limitation:** The advantage of the invention is not defined in raw PPL: a causal reference trained with the same parameter budget and the same corpus achieves a lower held-out PPL. The main source of the gap is representational capacity — the point cloud (320 numbers) carries ~12.5% of the representational capacity of the transformer hidden state (2,560 numbers) (see 2.7.4). The technical effects of the invention (low representational dimension, memory, interpretability, auditable decisions via region/prototype analysis) are reported together with this measurement.

#### 2.7.6 Alternative Embodiments

The scope of the invention also covers the following alternatives:

1. **Alternative aggregation:** Instead of max/mean pooling, a light attention (pair-attention) may be used over the pair features; permutation invariance is preserved by a symmetric normalization.
2. **Alternative pair features:** A cross product (D(D−1)/2 dimensions) or a Jacobi distance may be added to the input vector; the "at least distance and angle" expression in the claims covers this extension.
3. **Alternative activation:** Instead of Tanh, normalizing functions that project points onto a bounded sphere (‖p‖ ≤ 1) may be used.
4. **Alternative (attention-free) context encoder:** Instead of a transformer, an RNN, linear (Mamba-like), or attention-free architecture may be used; the characterizing features of the invention are defined from the projection module onwards. When the context encoder is chosen attention-free (O(N) complexity), the total system — O(N) context encoding + O(N²) low-dimensional (D+4) pair-geometry computation — carries a lower-dimensional (D+4 < d_model) computational load per point pair compared to a self-attention-based system (O(N²)×d_model); the total-latency measurement depends on the chosen architecture and hardware (see 2.5/6, Claim 19).
5. **Multi-sentence cloud:** At the paragraph level, sentence clouds may form an upper cloud (hierarchical cloud).
6. **Temporal dimension:** In a D+1 dimensional space, the generation order of the tokens may be encoded as an extra dimension (4D cloud).
7. **Dynamic D:** D may be selected according to the token count of the sentence or the region count/spread values measured during training (see Claim 10).

#### 2.7.7 Fields of Application

1. **Language modeling:** next-token prediction;
2. **Sentence classification:** sentence type from the cloud geometry (shape, spread, compactness);
3. **Semantic similarity:** distance between the set-state vectors of two sentences;
4. **Morphological analysis:** root/affix separation from subword clustering;
5. **Language comparison:** comparison of the cloud geometry of different languages;
6. **Error detection:** anomalous cloud geometry (outlier point, anomalous spread) → possible spelling/grammar error.

The invention is industrially applicable including language-modeling APIs, translation systems, text generation, language analytics, and edge devices.

#### 2.7.8 Non-Language Application Examples (Alternative Embodiment)

The geometric principle of the invention — a unit (token, feature vector, sensor reading) → low-dimensional point; a window/sentence → point cloud; a decision → cloud configuration — can be generalized to sequential data types other than language. The claims of this application are within the scope of language modeling; the following are measured example application areas of the principle:

1. **Transaction fraud detection (real data):** On a 24.4M-row credit-card transaction dataset, the 50-dimensional feature vector of each transaction is coded as a point and the transaction window as a point cloud. The geometry layer + backprop reader (hybrid) achieves AUC-ROC 0.9745; the standard ML ceiling (random forest 0.973) is matched, slightly below the gradient-boosting ceiling (0.981). The point-cloud representation competes at the standard accuracy ceiling while carrying auditable decisions via region analysis.
2. **AML typology detection (synthetic data):** On a synthetic structuring/layering/velocity generator, window → cloud coding: hybrid reader AUC-ROC 0.960 / AP 0.872. Region map + prototype-distance analysis: the structuring (separation ratio 0.81) and velocity (0.59) typologies form regions clearly separated from the normal distribution; nearest-prototype classification accuracy is 87.8% (400 held-out windows) versus 72.8% for argmax. The layering typology (multi-hop chains) overlaps with the normal distribution; chain/dual-channel features are required for this typology (ongoing work).
3. **Industrial maintenance (synthetic data):** On synthetic turbofan data (48 engines / 2,686 windows), remaining-useful-life (RUL) classification: hybrid reader AUC-ROC 0.947 / AP 0.831 (standard MLP ceiling 0.977/0.944; pure 0th-order 0.693/0.407). It will be validated once access to real C-MAPSS data is obtained.

The common finding in these examples: point-cloud geometry + hybrid training (geometry layer 0th-order or backprop; reader backprop) carries near-standard-ML-ceiling performance together with auditable decisions via region/prototype analysis.

---

## 3. CLAIMS

### Independent Claims

**Claim 1**

A language modeling system, known from the prior art, comprising a context encoder that produces a context vector for each token of a token sequence,

**characterized in that** it comprises:

(a) a projection module (110) that projects each of the recited context vectors to a point in D-dimensional (3 ≤ D ≤ 64) Euclidean space and keeps all points within a bounded hypercube by a saturating activation function;

(b) a permutation-invariant cloud encoder (120), the encoder comprising:
    - a point MLP (121) with shared weights for each point,
    - a pair MLP (122) that produces a pair feature from an input vector comprising at least the geometric properties of the pair, namely the distance between the two points of the recited pair and the cosine of the angle between the origin vectors of the two recited points, for each point pair,
    - a fusion module (123) that combines the recited point and pair features by a symmetric aggregation to produce a set-state vector;

(c) a prediction module (130) that predicts a next token from the recited set-state vector;

wherein the prediction of the next token is made via the set-state vector derived from the geometric configuration of the recited points; the context encoder (100) may or may not comprise a self-attention mechanism for producing the contextual vectors, but the prediction module (130) predicts the next token not from the d_model-dimensional contextual vectors themselves but from the set state derived from the geometry of the D-dimensional point cloud (distances, angles, cluster statistics). By virtue of the recited D-dimensional point representation being lower-dimensional than the d_model-dimensional contextual vectors (D < d_model), the memory cost of the geometric representation is reduced relative to the context representation, and by virtue of the recited distances, angles and cluster statistics being directly computable, the model decision is rendered geometrically interpretable and auditable.

**Claim 2**

A language modeling method, known from the prior art, comprising the step of transforming the tokens of a sentence into contextual vectors,

**characterized in that** it comprises the steps of:

(a) projecting each contextual vector to a point in D-dimensional (3 ≤ D ≤ 64) Euclidean space that lies within a bounded hypercube;

(b) transforming each of the recited points into a point feature by a function with shared weights;

(c) computing, for each point pair, at least the geometric properties of the pair, namely the distance between the two points of the recited pair and the cosine of the angle between the origin vectors of the two recited points;

(d) combining the recited point features and pair features by symmetric aggregation to form a permutation-invariant set-state vector;

(e) predicting a next token from the recited set-state vector;

wherein the symmetric aggregation in the step (d) is permutation-invariant, and by virtue of the point positions being indirectly dependent on token order via the contextual vectors, the meaning of the sentence is encoded in the geometric configuration of the point cloud.

**Claim 3**

A computer program comprising program instructions which, when executed by a processor, implement the steps of the method of Claim 2, the recited program instructions providing that a next token is predicted from the point cloud geometry and that memory savings are provided by the low-dimensional representation in the cloud layer.

### Dependent Claims

**Claim 4:** Dependent on Claim 1, wherein the pair MLP (122) uses, for each point pair (pᵢ, pⱼ), a D + 4 dimensional input vector of the form [pᵢ − pⱼ, ‖pᵢ − pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖], the distance and cosine components of this input being invariant to rotation of the coordinate system.

**Claim 5:** Dependent on Claim 1, wherein the projection module (110) uses a two-layer MLP of dimension d_model → d_model → D and a Tanh activation, the points being bounded in the [-1,1]^D hypercube.

**Claim 6:** Dependent on Claim 1, wherein the fusion module (123) takes a max-pool over the point features and a mean-pool over the pair features, and combines the two features.

**Claim 7:** Dependent on Claim 1, wherein the prediction module (130) additionally produces the predicted point position (p_next) of the next token in D-dimensional space, and the system combines, during training, an auxiliary loss term minimizing the mean squared error to the correct point position of the next token with the cross-entropy loss term with coefficient λ (0 < λ ≤ 1).

**Claim 8:** Dependent on Claim 7, wherein 0.01 ≤ λ ≤ 1, preferably λ = 0.1.

**Claim 9:** Dependent on Claim 1, wherein, during generation, at each step, the predicted point of the newly generated token is added to the cloud, the set-state vector is updated, and the prediction of the next token is made from the geometric configuration of the updated cloud.

**Claim 10:** Dependent on Claim 1, wherein the system further comprises a dimension selection module that selects the D value from a predefined set of values based on the token count of the sentence and/or the measured cloud complexity — region count, mean spread.

**Claim 11:** Dependent on Claim 1, wherein the system further comprises a byte-level BPE-based tokenization module (140) that produces morphological units as subword tokens.

**Claim 12:** Dependent on Claim 11, wherein the projection module (110) uses a D-dimensional space in which subword tokens derived from the same morphological root can be represented at close positions in the cloud and morphological affixes can be represented in directions different from the root sub-cluster, this structure being applicable for an agglutinative language — Turkish, Finnish or Hungarian.

**Claim 13:** Dependent on Claim 1, wherein the system further comprises an error detection module that, when predefined threshold values are exceeded by the cloud geometry, detects its outliers — abnormal distance from the origin at the point level or abnormal spread at the cluster level — and produces a grammar or spelling-error indicator.

**Claim 14:** Dependent on Claim 1, wherein the system further comprises a similarity module that computes the distance between the set-state vectors produced for two sentences and produces a semantic similarity measure.

**Claim 15:** Dependent on Claim 1, wherein L is 2–8, d_model 128–512, D is {16, 32}, and d_cloud 256–512 in range/value.

**Claim 16:** Dependent on Claim 1, wherein the system further comprises an interpretation module (150) that processes point positions, the pair distance matrix, and cluster statistics — spread, centroid, compactness — and presents them to the user.

**Claim 17:** Dependent on Claim 3, wherein the program instructions are stored in a non-transitory, computer-readable recording medium.

**Claim 18:** Dependent on Claim 2, wherein, after the step (e), the predicted token is transformed into a context vector, projected to a point by the step (a), the recited point is added to the cloud, and the steps (b)–(e) are iterated until an end token is produced.

**Claim 19:** Dependent on Claim 1, wherein the context encoder (100) is an O(N) complexity architecture that does not comprise a self-attention mechanism — attention-free, linear attention, or recurrent — such that the total system is composed of O(N) context encoding and O(N²) low-dimensional pair-geometry computation, and, compared to a d_model-dimensional O(N²) self-attention-based system, it carries a lower-dimensional (D+4 < d_model) computational load per point pair.

---

## 4. ABSTRACT

Each token of a sentence is projected to a point within a bounded hypercube in 3–64 dimensional Euclidean space. The meaning of the sentence is coded in the geometry of this point cloud — inter-point distances, angles, and cluster spread. A symmetric (permutation-invariant) PointNet-like encoder produces a set-state vector from the point and pair geometry; the next token is predicted from this vector. Point positions are order-sensitive via the context encoder; the aggregation itself is permutation-invariant. The system provides a low-dimensional (D ≪ d_model) and directly computable — distances, angles, regions — interpretable representation. In agglutinative languages, the spatial arrangement of subword tokens tends to reflect morphology. In the 21M-parameter example, the 32D configuration: training-side PPL 4.08, 6.45 on a shared held-out set; an equally sized causal reference achieves 2.92 on the same set — the advantage of the invention is defined not in raw PPL but in representational dimension, memory, and interpretability. For D = 16 → 32, the region count grows 3 → 5 and the spread by 43%.

---

## 5. DRAWINGS (Separate Pages)

TÜRKPATENT requires each drawing on a separate page, in black-and-white line art, with reference numerals and a figure title. The following 8 drawings have been prepared as SVG in the `fig/` folder; each is added to the formal file as a separate page.

| Page | Figure | Title | File |
|---|---|---|---|
| 1 | Fig. 1 | System architecture (100–150) | `fig/fig1_mimarisi.svg` |
| 2 | Fig. 2 | "kedi bahçede uyudu" 3D point cloud (200, 210) | `fig/fig2_bulut.svg` |
| 3 | Fig. 3 | Cloud evolution 1→4 points (300) | `fig/fig3_evrim.svg` |
| 4 | Fig. 4 | 16D and 32D region maps (400, 410) | `fig/fig4_bolge.svg` |
| 5 | Fig. 5 | Word region geography (500) | `fig/fig5_kelime.svg` |
| 6 | Fig. 6 | Pair geometry vector components (600–630) | `fig/fig6_cift.svg` |
| 7 | Fig. 7 | Training curve PPL (700) | `fig/fig7_egri.svg` |
| 8 | Fig. 8 | Morphological clustering (800, 810) | `fig/fig8_morfoloji.svg` |

> **Note:** Drawings must be compiled as separate pages in PDF drawing format (black-and-white, with reference numerals) before filing. The SVG files are kept as source. The PPL axis range of Fig. 7 must be consistent with the Run A curve (~20,000 → 4.08, training side).

---

*This document is the technical text for a TÜRKPATENT patent application (English version).*
*Prepared by: [Inventor Name]*
*Revision: v2.3 (EN) — measured PPL reconciliation (training-side vs shared held-out), causal transformer baseline 2.92, Claim 1 attention tension, structural computation expression (Claim 3/19), 2.7.8 non-language application examples, Claim 11–12 narrowing*
*Date: October 16, 2026*
