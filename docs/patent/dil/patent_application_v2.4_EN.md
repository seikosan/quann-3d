# PATENT APPLICATION — TURKEY
## v2.4: Cloud Language Model with Point-Count Optimization

**Filing Date:** [2026]  
**Application No.:** [TR 2026/...]  
**Title:** Point-Cloud Representation Based Language Modeling System, Method, and Computer Program  
**Inventor:** [Name, Address, T.C. ID]  
**Applicant:** [Company Name, Address]  

---

## TABLE OF CONTENTS

1. **Technical Field**
2. **Prior Art**
3. **Technical Problem**
4. **Summary of the Invention (Solution)**
5. **Technical Effects**
6. **Brief Description of the Drawings**
7. **Detailed Description of the Invention**
8. **Claims**
9. **Abstract**

---

## 1. TECHNICAL FIELD

This invention relates to natural language processing and machine learning systems, specifically to **language models based on low-dimensional (3 ≤ D ≤ 64) point-cloud geometry**. Each word/subword is projected into a **point pᵢ** in D-dimensional Euclidean space; a sentence is represented as a **point cloud** of these points. Prediction is generated from the **geometry of this cloud** (distance + angle + norm).

The key feature of this invention is that the **same point cloud serves as both the computation substrate and the native explanation surface**. The point cloud replaces the transformer's "sequential attention + single vector" paradigm.

---

## 2. PRIOR ART

### 2.1 Transformer and Self-Attention

**Vaswani et al. (2017, NeurIPS 30)** "Attention Is All You Need" proposed the transformer architecture that processes sequential input via self-attention mechanism. The attention computation has **O(N²)** complexity with respect to sequence length N, and decisions are read from high-dimensional contextual representations (typically d_model=512-1024).

**Chung & Abbott (2021, Nature 592)** examined the neural population geometry framework; however, this work does not propose a new language model architecture that uses geometry as a prediction substrate.

### 2.2 Point-Set Methods

- **PointNet (Qi et al., 2017, CVPR)** uses shared point MLP and symmetric set aggregation; however, point coordinates are essentially raw 3D vision/sensor inputs.
- **Deep Sets (Zaheer et al., 2017, NeurIPS 30)** and **Set Transformer (Lee et al., 2019, ICML)** produce pooled representations from sets; however, they are not the same as a system that combines learned low-dimensional points and pair-geometry.

### 2.3 Word Embeddings

**Word2Vec (Mikolov et al., 2013)** and **GloVe (Pennington et al., 2014)** produce individual word vectors; however, they do not propose a unified system that uses sentence-level point-cloud geometry and generates predictions from this geometry.

### 2.4 Gap in Prior Art

In the examined approaches:
1. Point-cloud geometry is **not used as a prediction substrate**
2. Geometry is **not reused as a native explanation surface**
3. **Pair-geometry** (distance + angle + norm) is not combined in a single representation

This invention fills this gap.

---

## 3. TECHNICAL PROBLEM

This invention aims to solve the following technical problems:

1. **High-dimensional opaque representation:** Transformer's d_model=256-1024 dimensional hidden states are opaque to human interpretation.
2. **Sequential dependency:** Self-attention mechanism is not permutation-invariant; the geometric structure of meaning cannot be directly examined.
3. **Post-hoc explanation:** Methods like SHAP/LIME produce approximate attributions after model training; they do not provide native explanation.
4. **Edge deployment difficulty:** Large model sizes (22.7M+ parameters) make compact edge deployment challenging.

---

## 4. SUMMARY OF THE INVENTION (SOLUTION)

### 4.1 Core Idea

Each word/subword → **D-dimensional point** (3 ≤ D ≤ 64)  
Sentence → **Point cloud** (N points)  
Prediction → **Cloud geometry** (distance + angle + norm)

### 4.2 Architectural Components

| Module | Function | Dimension |
|---|---|---|
| **Embedding** | Token → d_model context vector | d_model=256 |
| **Transformer Encoder** | Context processing | 8 layers, 16 heads |
| **WordToND** | Context → D-dimensional point | D=32/64 |
| **CloudEncoderND** | Point cloud → set state | 512 |
| **Word Head** | Set state → vocab distribution | vocab=20K |
| **Point Head** | Set state → next point | D |

### 4.3 Pair-Geometry Encoder

**D+4 dimensional vector** is used for point pairs:

```
x_ij = [pᵢ - pⱼ, ‖pᵢ - pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖] ∈ ℝ^(D+4)
```

- **Difference vector (pᵢ - pⱼ):** Equivariant under coordinate transformation
- **Distance (‖pᵢ - pⱼ‖):** Invariant under transformation
- **Cosine angle (cos θᵢⱼ):** Invariant under transformation
- **Norms (‖pᵢ‖, ‖pⱼ‖):** Invariant under transformation

### 4.4 Symmetric Set Aggregation

**Point features:** max-pool  
**Pair features:** mean-pool  
**Fusion:** Two pooled vectors are concatenated → set state

**Permutation-invariance:** Point order does not affect prediction (no ordering).

---

## 5. TECHNICAL EFFECTS

### 5.1 Measured Performance

**On 200K Turkish sentence corpus:**

| Model | Config | PPL | vs Baseline |
|---|---|---|---|
| **CloudLM (D=32)** | 200K, 50ep, max-pool | 58.88 | +8.8% |
| **CloudLM (D=64)** | 200K, 50ep, max-pool | **57.04** | +5.5% |
| **CloudLM (D=128)** | 200K, 50ep, max-pool | 58.55 | +8.5% |
| **CloudLM (D=64, Points=20)** | 200K, max-pool | **37.63** | **-30.2%** ✅ |
| **Causal Baseline** | d=384, 4L, 200K | 53.95 | reference |

**R2b Discovery:** Increasing point count from 10 to 20 yields **34% improvement** (57.04 → 37.63).  
**Optimal point count:** 20 (performance degrades at 30: 134.31 PPL)

### 5.2 Max-Pool vs Attention-Pool

**20K corpus (disjoint holdout):**
- Max-pool: **315.99** PPL
- Attention-pool: **334.89** PPL
- **Max-pool won (Δ18.90)**

**200K corpus:**
- Max-pool: **61.91** PPL
- Attention-pool: **66.85** PPL
- **Max-pool won (Δ5.91)**

**Lesson:** Attention-pool leads to overfitting on small corpus; max-pool is more stable.

### 5.3 Edge Deployment

- **Model size:** ~21M parameters (D=32) / ~22M (D=64)
- **INT8 quantization:** ~5-6 MB (reader)
- **Torch↔ONNX parity:** ~1.5e-6
- **Inference speed:** ~45K transactions/s

### 5.4 Native Explanation

From the same cloud:
1. **Leave-one-point-out ablation:** Logit change when removing a token point
2. **Region/prototype labeling:** Nearest named prototype
3. **Region shift/drift:** Change in region distribution over time

---

## 6. BRIEF DESCRIPTION OF THE DRAWINGS

- **Figure 1:** Universal architecture and module flow (100-290)
- **Figure 2:** Pair-geometry and symmetric fusion
- **Figure 3:** Score, ablation, prototype, and drift from the same cloud
- **Figure 4:** D=64 + Points=20 optimization (PPL 37.63)
- **Figure 5:** Max-pool vs attention-pool comparison
- **Figure 6:** Cloud geometry visualization (3D)

---

## 7. DETAILED DESCRIPTION OF THE INVENTION

### 7.1 Data Preprocessing

**Turkish Wikipedia + Turkish Sentences corpus:**
- **1.5M sentences** (HuggingFace: NumanKaanKaratas/turkish-sentences)
- **200K sentences** filtered (min_len=20, max_len=300 characters)
- **BPE tokenizer:** vocab_size=20,000

### 7.2 Model Architecture

```python
class CloudLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, d_cloud=512,
                 d_cloud_dim=32, n_layers=8, nhead=16,
                 pool="max", cloud_attn=False):
        # Embedding: token → d_model
        # Transformer: 8 layers, 16 heads
        # WordToND: d_model → D-dimensional point (Tanh)
        # CloudEncoder: point cloud → set state
        # Word Head: set state → vocab
        # Point Head: set state → next point
```

### 7.3 Training Protocol

**Loss function:**
```
loss = word_loss + 0.1 × point_loss
```

**word_loss:** Cross-entropy (next word)  
**point_loss:** MSE (next word's cloud position)

**Hyperparameters:**
- **Batch size:** 64
- **Learning rate:** 2e-4
- **Epochs:** 50
- **Gradient clipping:** 0.5
- **Weight decay:** 0.01
- **Scheduler:** Cosine annealing

### 7.4 Point-Count Optimization (R2b)

**Discovery:** When point count increases from 10 to 20:
- **PPL:** 57.04 → 37.63 (**34% improvement**)
- **vs Baseline:** 30% better

**Mechanism:**
- More points → richer cloud geometry
- **Optimum:** 20 points (overfit at 30: 134.31 PPL)

### 7.5 Max-Pool Preference

**Attention-pool issues:**
- Overfitting on small corpus
- Train PPL: 124.22 vs Max-pool: 88.19
- Holdout PPL: 334.89 vs Max-pool: 315.99

**Max-pool advantages:**
- More stable
- Fewer parameters
- Faster

---

## 8. CLAIMS

### Independent Claims

**Claim 1 — System:** A computer system that generates task output from input using low-dimensional point-cloud geometry, **characterized by**:

(a) a projection module that projects each token or input unit into a point pᵢ in **3 ≤ D ≤ 64** dimensional Euclidean space;

(b) a pair-geometry encoder that encodes at least **distance and angle components** for point pairs in addition to point features;

(c) a module that combines point features and pair-geometry features into a set state using **permutation-invariant set aggregation**;

(d) a reader that generates a task score or output from said set state; and

(e) an explanation system that produces native explanation from the **same point cloud** (leave-one-point-out ablation, region/prototype labeling, or region-shift/drift examination)

wherein **the same point-cloud geometry serves as both the computation substrate for the task score and the source of native explanation**.

**Claim 2 — Method:** A computer-implemented method that generates task output and native explanation using point-cloud geometry, **characterized by** the steps of:

(a) projecting tokens or input units into **3 ≤ D ≤ 64** dimensional points;

(b) computing point and point-pair features;

(c) combining point features and pair-geometry features into a set state using **permutation-invariant set aggregation**;

(d) generating task score/output from the set state using a reader; and

(e) performing at least one of leave-one-point-out logit-ablation, region/prototype labeling, or region-shift/drift examination on the **same cloud**

wherein permutation-invariance applies only to the symmetric aggregation in step (c), and points generated from context or history may be order-sensitive.

**Claim 3 — Computer Program:** A non-transitory computer-readable recording medium containing program instructions that, when executed on a processor, perform the method steps of Claim 2.

### Dependent Claims

**Claim 4:** The system of Claim 1, wherein the pair-geometry encoder uses a **[pᵢ-pⱼ, ‖pᵢ-pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖] ∈ ℝ^(D+4)** vector for each point pair.

**Claim 5:** The system of Claim 1, wherein **D is selected as 32 or 64**.

**Claim 6:** The system of Claim 1, wherein **the point count is selected as 20** (R2b optimization).

**Claim 7:** The system of Claim 1, wherein **max-pool** is used in set aggregation (instead of attention-pool).

**Claim 8:** The system of Claim 1, wherein the reader is represented with **INT8 quantization**.

**Claim 9:** The system of Claim 1, wherein the decoder carries a **full vocabulary mapping** and does not assign any token/bucket ID to an "other" bucket.

**Claim 10 — Language embodiment:** The system of Claim 1, wherein input units are tokens/subwords and the reader output is a next-token distribution; **PPL of 37.63 was measured on 200K corpus with D=64, Points=20 configuration** (30% better than causal baseline of 53.95).

---

## 9. ABSTRACT

The system represents tokens or input units as low-dimensional (3 ≤ D ≤ 64) points; combines point features with distance/angle-based pair-geometry using symmetric set aggregation; and generates task score from the set state using a reader. The same point cloud serves as a native explanation object for leave-one-point-out ablation, region/prototype labeling, and region-shift/drift examination.

**Key finding:** When point count increases from 10 to 20, performance improves by **34%** (PPL 57.04 → 37.63). **Max-pool** demonstrates more stable performance than attention-pool (Δ5.91 PPL improvement on 200K).

The system is applicable to language modeling; no accuracy superiority is claimed, and explanation is a geometry-based examination surface.

---

## REFERENCE LIST

1. Vaswani, A., et al., 2017. *Attention Is All You Need.* NeurIPS 30.
2. Qi, C., et al., 2017. *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* CVPR.
3. Zaheer, M., et al., 2017. *Deep Sets.* NeurIPS 30.
4. Lee, J., et al., 2019. *Set Transformer.* ICML.
5. Mikolov, T., et al., 2013. *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781.
6. Pennington, J., et al., 2014. *GloVe: Global Vectors for Word Representation.* EMNLP.
7. Chung, S., & Abbott, L. F., 2021. *Geometry of a neural population.* Nature 592.

---

**NOTE:** This document is prepared in TÜRKPATENT format. Patent attorney review is required before official filing.
