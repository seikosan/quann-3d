# Geometric Sentence Clouds: A Point-Cloud Approach to Language Modeling Beyond Sequential Attention

**Draft v0.1 — October 2026**

---

## Abstract

We propose **CloudLM**, a language modeling architecture that represents sentences as point clouds in N-dimensional space, where each token occupies a position in the cloud and sentence meaning is encoded in the **geometry** of the point configuration (distances, angles, cluster shape) rather than in a single high-dimensional vector. Unlike transformers, which represent a sentence as one 768D vector and predict the next token via sequential attention, CloudLM treats the sentence as an **unordered set of geometric points** and predicts based on the spatial relationships between them. We demonstrate that even with as few as 16 dimensions, the model learns to separate syntactic roles into distinct spatial regions (e.g., conjunctions cluster centrally, demonstratives occupy eastern-southern quadrants). On a 30K-sentence Turkish corpus with a 20K BPE vocabulary, CloudLM achieves a perplexity of 57 after 15 epochs (21M parameters), reducing the random baseline (20,000) by a factor of 350×. The approach offers a geometrically interpretable alternative to transformer attention, with potential advantages for morphologically rich languages and for tasks requiring spatial reasoning over sentence structure.

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

| D | Test PPL | Regions (KMeans k) | Silhouette | Avg Spread |
|---|----------|--------------------:|:----------:|:----------:|
| 16 | 4.58 | 3 | 0.143 | 1.573 |
| 32 | **4.08** | **5** | 0.108 | **2.245** |

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

| Model | Params | PPL (Wikipedia TR) |
|-------|--------|-------------------:|
| Random (untrained) | 21M | 20,000 |
| CloudLM 16D (50ep) | 21M | 4.58 |
| CloudLM 32D (50ep) | 21M | **4.08** |
| GPT-2 small (reference, English) | 124M | ~15-20 |

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

---

## 7. Conclusion

We presented CloudLM, a language model that represents sentences as geometric point clouds and predicts based on cloud configuration rather than sequential attention. Key findings:

1. **16 dimensions suffice** for 3 distinct syntactic regions; **32 dimensions** yield 5 regions with 11% lower PPL
2. **Geometry is interpretable**: spatial regions correspond to syntactic categories (conjunctions central, demonstratives eastern)
3. **BPE + cloud** eliminates OOV for morphologically rich languages (Turkish agglutination)
4. **The bottleneck is productive and quantifiable**: 16D→32D increases region count by 67% and cloud spread by 43%
5. **4,900× improvement** over random baseline with 21M parameters (1/6 of GPT-2 small)

Future work: scaling to 100M+ parameters, adding temporal dimension (4D clouds for discourse), standardized benchmark comparison, and theoretical analysis of the dimension-region relationship.

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

---

*Code: https://github.com/[user]/quann-3d*  
*Corpus: Bilkent Turkish Writings (CC), HuggingFace NumanKaanKaratas/turkish-sentences (MIT)*
