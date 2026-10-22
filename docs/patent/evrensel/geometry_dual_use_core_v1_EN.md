# PCT INTERNATIONAL APPLICATION — Universal Dual-Use Geometry Core (v1)

**International application under the Patent Cooperation Treaty (PCT)**
**PCT/RO/101 request form + technical document compliant with Industrial Property Law No. 556**

## 0. PCT REQUEST (PCT/RO/101 — basic data)

| Field | Data |
|---|---|
| **Receiving Office (RO)** | TÜRKPATENT (TR) — national/receiving office |
| **Type of application** | Patent — International (PCT) application |
| **Application No.** | [to be assigned by the International Bureau: TR 2026/… ] |
| **Filing Date** | [international filing date] |
| **Inventor(s)** | [Full Name, address, nationality, National ID] |
| **Applicant** | [Name, address, nationality] |
| **Agent** | [Patent attorney / firm, PCT-authorized] |
| **Title of the Invention** | Universal System, Method and Computer Program for Joint Computation and Native Explainability Managed by Dual-Geometry Point-Cloud Representation |
| **Technical Field (IPC)** | G06F 18/00; G06N 3/04; G06N 5/02; G06V 10/77 (suggestion) |
| **Claim of Priority** | [To be left blank — see 7.2: this universal scope is not disclosed in the TR national (language) filing, so no priority can be claimed from it; independent application] |
| **Designation of Contracting States** | [The agent will designate all PCT states or the target countries] |
| **Procedure requested** | International search (IIS: TÜRKPATENT/EPO) + international publication (18 months) + (optional) supplemental international search (22 months) |

> **CONFIDENTIALITY / PRIORITY WARNING:** This document is prepared prior to a patent application. Until the international filing date (right of priority) is obtained, no publication (arXiv, conference, journal, blog, publicly accessible GitHub, podcast, presentation) relating to this invention may be made. Publication would have a devastating effect on the novelty requirement (self-depublication). No public disclosure should be made even prior to PCT publication (18 months).

## TABLE OF CONTENTS

0. **PCT Request** — basic data
1. **Title**
2. **Description** — 2.1 Technical Field · 2.2 Prior Art · 2.3 Problem · 2.4 Essence · 2.5 Effects · 2.6 Figures · 2.7 Detailed Description (2.7.1–2.7.6)
3. **Claims** — 3 independent (System, Method, Computer Program) + 10 dependent (6 mechanism + 4 vertical embodiments) = 13 claims
4. **Abstract**
5. **Drawings** — 7 figures, separate pages
6. **Bibliography**
7. **Agent Assessment** — unity, priority, non-obviousness, corrections (not part of the filing)

> **TR/PCT relationship (critical):** This **PCT (universal)** application has a **different, broader** scope than the **language** embodiment in our TR national filing `TURKPATENT_basvuru_v2.3.md` (dual-use geometry core + language/fraud/AML/maintenance). Because the universal scope is not disclosed in the TR language filing, this PCT document **cannot claim priority** from the TR filing — as an independent application it has its own priority date (T2). A textual link between the two documents does not, by itself, create an earlier date.

---

## 1. TITLE

**Universal System, Method and Computer Program for Joint Computation and Native Explainability Managed by Dual-Geometry Point-Cloud Representation**

(Turkish: “Çift-Geometri Nokta-Bulutu Temsili ile Birleşik Hesaplama ve Native Açıklanabilirlik Yöneten Evrensel Sistem, Yöntemi ve Bilgisayar Programı”)

---

## 2. DESCRIPTION

### 2.1 Technical Field

The present invention relates to machine-learning systems based on low-dimensional (3 ≤ D ≤ 64) point-cloud geometry; in particular to a system, a method and a computer program in which the same learned geometric representation is used both to compute the prediction/decision score and as the native explanation surface. The units may be, depending on the input type, a token, a feature-value identity, a transaction, a window or another observation; each unit is projected to a Euclidean point.

The dual-geometry representation jointly encodes the individual point features and the relations belonging to point pairs. The set state obtained from the point geometry is supplied to a reader. The same point cloud and the reader tied to that cloud are the source of native-explanation operations such as leave-one-point-out ablation, region/prototype naming and region-shift/drift analysis. The system may be applied to the domains of language modeling, transaction fraud, anti-money-laundering (AML) typology and industrial maintenance.

### 2.2 Prior Art

#### 2.2.1 Single-decision representations and feature attributions

Transformer architectures (Vaswani et al., 2017, *NeurIPS 30*) process sequential input with self-attention; the attention computation is O(N²) in the sequence length N and the decision is read mostly from high-dimensional contextual representations. The distinction advanced here is not the absence of any attention use; it is that the prediction/decision reader uses the set state produced from the geometry and that the same geometry is the source of the explanation.

Decision trees and SHAP present a list mostly of feature names with numerical attribution values. The SHAP additivity axiom means that the additivity of attributions does not guarantee that the native leave-one-point-out sensitivity in a dual-geometry, interacting point-cloud representation reproduces the total score. Record-level attribution also does not, by itself, define population-level false-alarm drivers or region/prototype drift.

#### 2.2.2 Point-set methods and geometric analysis

- **Deep Sets** (Zaheer et al., 2017, *NeurIPS 30*) and **Set Transformer** (Lee et al., 2019, *ICML*, 3714–3723) produce a pooled representation from sets. The specific mechanism here is that, while carrying the learned coordinates and the geometry of point pairs into the same decision representation, the same cloud is preserved for the native-explanation operations.
- **PointNet** (Qi et al., 2017, *IEEE/CVF CVPR*, 77–80) uses a shared point MLP and symmetric set aggregation; the point coordinates are essentially raw 3D vision/sensor input. This is not the same as the system that combines learned low-dimensional points from different data types, the dual-geometry and the native ablation/prototype/drift explanation from the same cloud; PointNet does not establish this combination for language or tabular records.
- **Word2Vec** (Mikolov et al., 2013, arXiv:1301.3781), **GloVe** (Pennington et al., 2014, *EMNLP*, 1532–1543) address single-word vectors; **Chung & Abbott** (2021, *Nature* 592(7853), 117–122) address the neural population geometry framework, but do not define the combined decision-and-explanation system described here.
- **Hu et al.** (2026, arXiv:2607.04525; DOI 10.48550/arXiv.2607.04525) analyze in-model concepts post-hoc as a point-cloud manifold and vector fields. **Xu** (2026, arXiv:2605.17084; DOI 10.48550/arXiv.2605.17084) examines the layer distance structure with Subspace PGA. The cited works offer geometric analysis; they do not propose the new system architecture that makes the geometry both the score-computation substrate and the native explanation surface produced from the same object.

In the surveyed sources (Transformer, GBM+SHAP, DeepSets, Set Transformer, PointNet, Hu-2026 and Xu-2026) this specific combination — in which the geometry is both the computation and the native-explanation substrate — is not shown. This is not a global absence claim; its scope is limited to the sources examined here and requires a broader prior-art search.

#### 2.2.3 Technical gap in the prior art

The surveyed approaches do not combine, in a single technical core, the representation that uses low-dimensional learned point-cloud geometry in decision production with the reuse of this same representation as the native-explanation object. As a result, in most systems the object in which the decision is formed is separated from the explanation object inspected after the decision; single attribution, population audit and deployment evidence are also distributed across different tools.

### 2.3 Technical Problem

The present invention aims to solve the following technical problems:

1. A prediction/decision being inspected through a single opaque high-dimensional vector or attribution list; the geometric sub-components of the decision not being directly auditable.
2. Post-hoc feature attributions on an interactive, non-additive representation not reproducing the native effect of the decision, and the single explanation not showing the population drivers.
3. Because separate representation and tool chains are used for decision and explanation, edge deployment not being established in a compact, self-contained and bit-exact verifiable manner.
4. A common geometry core not being applicable to different input types such as text, tabular transaction/record, AML transaction window and sensor/complaint window.

### 2.4 Essence of the Invention (Solution)

The essence of the invention is that a single low-dimensional point cloud is **simultaneously the computation substrate and the native explanation surface**. Instead of computing the geometry and then approximately interpreting it with a separate explanation model, the score and the explanation are derived from the same points, the same point-pair relations and the same region/prototype structure.

| Column | Module and function | Tie to the same cloud |
|---|---|---|
| Representation | Input units are projected to learned points (110); the source module may be a token, an identity-bucket or a window. | The points are the shared object of all subsequent steps. |
| Dual-geometry | The point MLP and the pair MLP encode the point features and the distance/angle and optional difference/norm components (120). | Point and pair relations are combined into the set state by symmetric aggregation (123). |
| Reader | The reader produces the task score/output from the set state (130). | The score is computed from the same cloud geometry. |
| Native explanation and deployment | Interpretation (150), ablation/decoder (260/270), counterfactual audit (280), packager (290). | Ablation, prototype/region and drift are read from the same cloud; the reader and required components are delivered in a self-contained package. |

Permutation-invariance holds only for the symmetric set aggregation applied after the shared point/pair functions. Point coordinates produced from context or history may be order-sensitive; it is not claimed that all stages of the system are order-free. The difference vector is covariant under a transformation; the distance, the cosine angle and the norms are components invariant to a coordinate transformation.

The deployment principle is that the reader, the required quantizer/decoder and the manifest are combined in a compact, self-contained and bit-exactly verifiable package. The package size and numerical parity depend on the application, the input contract and the target hardware; no single universal size or parity value is asserted.

### 2.5 Technical Effects

1. **Native, readable explanation:** the leave-one-point-out ablation from the same geometry is decoded to text; the region/prototype is named; the region shift between times or operating conditions is measured as drift.
2. **Honest presentation of non-additive attribution:** the logit change upon point removal is reported as a counterfactual sensitivity; it is not assumed to be a SHAP-type additive contribution or that the sum of individual effects equals the score.
3. **Population audit:** single and pair mask counterfactuals allow examining common and rare drivers in the alarm population; the inter-example region/prototype distribution allows monitoring the shift.
4. **Compact edge deployment:** the reader, quantizer, decoder and manifest can be carried as a single self-contained package; bit-exact parity and source integrity can be verified on the relevant target.
5. **Transferable core across verticals:** while the representation source and region names vary by vertical, the geometric decision and native-inspection mechanism can be used in language, fraud, AML and maintenance applications. This is not a proof of accuracy or generalization superiority across verticals.

**Boundary of the results:** The present invention makes no accuracy-superiority or SOTA claim. In the fraud measurement the matched PR-AUC is 0.462 whereas XGBoost is 0.8047. QUANN is only a 0th-order point-reduction option for small and limited models; it has not been shown to scale to large-model training. A region map is not causality, not a complete explanation, not regulatory approval and not guaranteed decision accuracy. In AML, coarse label stability does not mean the map coordinates are stable; the maintenance edge package takes the scaled cloud input, not raw sensor frames.

### 2.6 Brief Description of the Figures

- **Figure 1:** Combined architecture — 100 input/context module, 110 point projection, 120 dual-geometry encoder, 123 symmetric fusion, 130 score reader, 150 interpretation, 260 ablation, 270 decoder, 280 counterfactual audit, 290 deployment.
- **Figure 2:** Point and dual-geometry — D-dimensional points, the [pᵢ−pⱼ, ‖pᵢ−pⱼ‖, cosθ, ‖pᵢ‖, ‖pⱼ‖] vector and symmetric aggregation.
- **Figure 3:** Native explanation from the same cloud — the logit difference of the removed point, the text decoder, the region/prototype name and the drift across times.
- **Figure 4:** AML region/prototype naming examples — `fig/fig4a_aml_tipoloji.png`, `fig/fig4b_aml_tipoloji.png`.
- **Figure 5:** Fraud identity-bucket cloud — `fig/fig5_fraud_bulut.png`.
- **Figure 6:** Maintenance sensor-window cloud and region shift — `fig/fig6a_bakim_bolge.png`, `fig/fig6b_bakim_drift.png`.
- **Figure 7:** Self-contained deployment and verification — fraud ONNX/manifest and maintenance INT8 edge examples.

### 2.7 Detailed Description of the Invention

#### 2.7.1 Common architecture (100–150)

The input module (100) splits the observations into units. These units may be a token, a (feature, value-bucket) identity or the measurement/summary components in a time window. The projection module (110) converts each unit to a point pᵢ ∈ ℝᴰ; 3 ≤ D ≤ 64. The pᵢ coordinates produced using context or history may be order-sensitive.

The dual-geometry encoder (120) produces, for each point, point features via a shared point MLP; for each (pᵢ,pⱼ), geometric relations via a pair MLP. The preferred pair vector is:

```text
x_ij = [p_i − p_j, ‖p_i − p_j‖, cos(θ_ij), ‖p_i‖, ‖p_j‖] ∈ ℝ^(D+4)
```

Here pᵢ−pⱼ is covariant under a coordinate transformation; the distance ‖pᵢ−pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖ and ‖pⱼ‖ are invariant to rotation. The fusion module (123), for example applying max-pool to the point features and mean-pool to the pair features and combining them, produces cloud_state. Permutation-invariance is a property of this symmetric aggregation step; it is not generalized to the context/history ordering that prepares the input.

The reader (130) produces from cloud_state a classification score, risk, next-token distribution or another task-appropriate output. The interpretation module (150) inspects the point, pair and region/prototype information of the same cloud. This structure does not turn the geometry into a post-hoc drawing outside the decision: the input of the reader and the object of the explanation are the same cloud.

#### 2.7.2 Native explanation, counterfactual and package (260–290)

The ablation module (260) computes the change in the reader logit when a single pᵢ point is removed. Because of the interactive dual-geometry, this value is termed a **leave-one-point-out counterfactual sensitivity**; it is not claimed that the contributions are additive or that the sum of the absolute changes equals the initial logit. The full-vocabulary decoder (270) can convert the points/identities in the explanation to human-readable units; for example, in the fraud embodiment all 221 identities are resolved and no bucket is left under “other”.

The counterfactual audit module (280) tests whether the score remains above the threshold after masking/removing a point or a point pair. At the population level these results can separate common drivers from rare drivers. The region/prototype naming gives the position of the observation relative to the nearest named prototype and the supporting geometric measures. The region distribution under different times or operating conditions can be compared to measure drift; this is not a guarantee of causal explanation or automatic fault diagnosis.

The packager (290) collects, according to the task, the reader, quantizer, decoder and manifest into a single self-contained package. The manifest may carry the source identity/hash, the configuration and the self-test/parity results. The numerical results are application-specific measurements; the general claim has no fixed package size or parity threshold.

#### 2.7.3 Language embodiment

In the text input each token (including sub-words) is projected to a point by 110; the sentence cloud is encoded by 120/123 and the reader (130) produces the next-token distribution. The leave-one-point-out logit difference of the token point can be inspected via 150/260; the vocabulary decoder can convert this to text. In agglutinative languages the cloud regions of sub-words from the same root can be studied; this does not mean that the whole ordering is permutation-invariant.

In the 200K configuration of the v2.3 file the reported D32 PPL is 6.45 and the causal reference is 2.92 (about a 2.2× difference; not an accuracy superiority in favor of CloudLM). **Measurement limitation:** the current `RESEARCH.md` records that this 6.45 value is a seed=0 sample from the training set and is not a strict held-out; accordingly the value is carried here as a comparison indicator and not presented as a verified held-out result. From D=16 to D=32 the region count rose from 3 to 5 and the spread from 1.573 to 2.245 (v2.3 §2.7.5). These results are a vertical-embodiment proof, not a general performance guarantee.

#### 2.7.4 Fraud embodiment

In the tabular transaction input each feature/value-bucket pair becomes an identity unit (210); the identity-point module (220) projects these to the cloud. The reader (130) produces the risk logit, the ablation (260) the leave-one-point-out difference, the full-vocabulary decoder (270) the plain-text label, the counterfactual audit (280) the single/pair-mask alarm effect and the packager (290) the ONNX/quantizer/decoder/manifest package.

In the measured examples the single-masked audit left 94–100% of the alarm population unflagged. The matched PR-AUC is 0.462 and the XGBoost PR-AUC is 0.8047; superiority is not claimed. The fraud deployment example measured 543 KB ONNX, Torch↔ORT parity 1.5e-6 and about 45.5k transactions/s; these are only the conditions of this embodiment, not a universal package bound.

#### 2.7.5 AML embodiment

In AML each transaction or transaction window is projected to a point cloud; the reader gives an output toward typology/risk; the prototype-naming module compares the cloud with named typology prototypes. In the Elliptic held-out example the prototype naming is 75% overall and 45% in the illicit class. In the five-seed pilot the coarse named-typology stability is 0.325 and the SHAP top-1 stability is 0.1025 (about 3.2×); **this does not mean the map coordinates are stable**: the coordinates and the attribution magnitudes have not been shown to be retraining-stable. The claim is not a guarantee of accuracy or of a SAR rationale, but a named geometric-inspection surface.

#### 2.7.6 Maintenance embodiment

In maintenance the measurement window or its features (100/110) are converted to sensor-points; 120/123 give the cloud state to the reader (130); the maintenance engineer inspects the prototype and the region shift with 150. In the CWRU record-disjoint binary task ROC-AUC 0.999 and PR-AUC 1.000 were measured; the GBT counterpart is 1.000/1.000. In C-MAPSS the nearest-prototype explanation covered 81.7% over 1,064 windows; the operating-shift drift alarm was 2.4× (1.17%→2.81%). The INT8 reader edge example is 79 KB and the Torch↔ONNX parity is 5.07e-7. This artifact takes the scaled cloud input, not the raw sensor frame; the CWRU metric is not asserted for this C-MAPSS package.

---

## 3. CLAIMS

### Independent Claims

**Claim 1 — System:** A computer system that produces a task output from an input unit via low-dimensional point-cloud geometry and provides a geometric explanation, **characterized in that** it comprises:

(a) a projection module (110) that projects each token or input unit to a pᵢ point in 3 ≤ D ≤ 64-dimensional Euclidean space, and a module (123) that combines the point/pair features into a set state by **permutation-invariant set aggregation**;

(b) a dual-geometry encoder (120) that, in addition to the point features, encodes for point pairs at least the distance and angle components;

(c) a reader (130) that produces a task score or output from the set state of the cloud referred to above;

(d) an interpretation system that produces a native explanation from **the same point cloud**; this system comprising, in order to perform at least one of (i) converting the leave-one-point-out logit-ablation to text via the decoder (270), (ii) region/prototype naming and (iii) region-shift/drift analysis, the ablation module (260), the decoder (270) and/or the interpretation module (150) together with the counterfactual module (280); and

(e) a packager (290) that combines the reader, the required quantizer, the decoder and the manifest in a self-contained compact deployment package

and in that **the same point-cloud geometry is both the computation substrate of the task score and the source of the native explanation**.

**Claim 2 — Method:** A method, implemented by a computer, that produces a task output and a native explanation via point-cloud geometry, **characterized in that** it comprises the steps of:

(a) projecting tokens or input units to points of 3 ≤ D ≤ 64 dimensions;
(b) computing the point and point-pair features;
(c) combining the point features and the dual-geometry features into a set state by permutation-invariant set aggregation;
(d) producing a task score/output from the set state via the reader;
(e) performing at least one of, on **the same cloud**, a leave-one-point-out logit-ablation, text via the decoder, region/prototype naming or region-shift/drift analysis; and
(f) delivering the reader together with the required quantizer, decoder and manifest in a compact, self-contained, verifiable deployment package

and in that permutation-invariance holds only for the symmetric aggregation in (c) and the points produced from context/history may be order-sensitive.

**Claim 3 — Computer program:** A non-transitory computer-readable storage medium storing program instructions that, when executed by a processor, perform the method steps of Claim 2.

### Dependent Claims

**Claim 4:** The system of Claim 1, wherein the dual-geometry encoder uses, for each point pair, the `[pᵢ−pⱼ, ‖pᵢ−pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖] ∈ ℝ^(D+4)` vector; the distance and cosine components are invariant to a coordinate transformation and the difference vector is covariant under the coordinate transformation.

**Claim 5:** The system of Claim 1, wherein the D value is selected from the set `{8, 16, 32, 64}` or is selected dynamically according to the cloud complexity such as the measured region count and/or spread.

**Claim 6:** The system of Claim 1, wherein the reader (130) is represented by INT8 quantization and its parity is verified in the target deployment package.

**Claim 7:** The system of Claim 1, wherein the decoder (270) carries a full vocabulary mapping and does not assign any token/bucket identity to an “other” bin.

**Claim 8:** The system of Claim 1, wherein the counterfactual module (280) runs single-point and pair-point masking/removal experiments and reports their outcome as a counterfactual sensitivity rather than an additive contribution.

**Claim 9:** The system of Claim 1, wherein a hybrid training regime is used, applying preferably QUANN 0th-order reduction for the point positions and backpropagation training for the reader; the 0th-order step is preferably applied with small and limited models.

**Claim 10 — Language embodiment:** The system of Claim 1, wherein the input units are tokens/sub-words, the reader output is the next-token distribution and the interpretation is a cloud/region analysis aimed at agglutinative morphology; in the D32 example PPL 6.45 and the causal reference 2.92 are reported. Since the 6.45 value is marked in the current status source as a training sample, this measurement does not constitute a strict held-out performance claim.

**Claim 11 — Fraud embodiment:** The system of Claim 1, wherein the input units are identity-bucket transaction features and the native explanation comprises leave-one-point-out text, counterfactual false-alarm audit and a self-contained edge package; in the example measurement the single-masked audit is 94–100% unflagged, 543 KB ONNX, 1.5e-6 parity and about 45.5k transactions/s are reported; the matched PR-AUC is 0.462 whereas XGBoost is 0.8047.

**Claim 12 — AML embodiment:** The system of Claim 1, wherein the interpretation module gives named typology prototypes and the geometric rationale; in the Elliptic held-out example the naming is 75% overall and 45% in the illicit class, the five-seed coarse-label stability is 0.325 and the SHAP top-1 stability is 0.1025 (about 3.2×); this stability does not mean the map coordinates are stable.

**Claim 13 — Maintenance embodiment:** The system of Claim 1, wherein the input units are the sensor window/measurements, the explanation is region/prototype and region-shift/drift; the CWRU record-disjoint binary ROC/PR is 0.999/1.000 (GBT 1.000/1.000), the C-MAPSS prototype-explanation coverage is 81.7% over 1,064 windows, the drift is 2.4× (1.17%→2.81%) and the INT8 edge reader is 79 KB / 5.07e-7 parity; the edge artifact takes the scaled cloud input, not the raw sensor input.

---

## 4. ABSTRACT

The system represents input units such as a token, a feature identity or a window as low-dimensional points; it combines the point features and the distance/angle-based dual-geometry by symmetric set aggregation and produces a task score from the set state via the reader. The same point cloud is the native-explanation object for decoding the leave-one-point-out ablation to text, for region/prototype naming and for region-shift/drift analysis. The reader, the required quantizer, decoder and manifest can be deployed in a compact, self-contained and verifiable package. Thus the same geometry is both the computation and the explanation substrate. The language, fraud, AML and maintenance embodiments apply the common core with different units/namings. No accuracy-superiority is claimed; the explanation is a geometry-based audit surface and does not imply causality or a guarantee.

---

## 5. DRAWINGS (To Be Submitted on Separate Pages)

The drawings must be prepared, on separate pages from the description text, as black-and-white line art with reference numerals.

| Page | Figure | Content | Source / preparation note |
|---|---|---|---|
| 1 | Figure 1 | Universal architecture and module flow (100–290) | `fig/fig1_mimari.svg` |
| 2 | Figure 2 | Dual-geometry and symmetric fusion | `fig/fig2_ciftgeometri.svg` |
| 3 | Figure 3 | Score, ablation, prototype and drift from the same cloud | `fig/fig3_aciklama.svg` |
| 4 | Figure 4 | AML typology naming examples | `fig/fig4a_aml_tipoloji.png`; `fig/fig4b_aml_tipoloji.png` |
| 5 | Figure 5 | Fraud identity-bucket cloud | `fig/fig5_fraud_bulut.png` |
| 6 | Figure 6 | Maintenance cloud and operating-condition drift | `fig/fig6a_bakim_bolge.png`; `fig/fig6b_bakim_drift.png` |
| 7 | Figure 7 | Self-contained deployment and parity/manifest audit | Fraud deploy artifact and maintenance edge example |

> **Note:** Figures 1–3 are new line drawings to be produced (black-and-white, with reference numerals, each on its own page); Figures 4–7 derive from the per-vertical artifact figures cited above and must be reproduced as black-and-white line/paper drawings on separate pages.

---

## 6. BIBLIOGRAPHY

The prior-art sources cited in section 2.2:

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I., 2017. *Attention Is All You Need.* Advances in Neural Information Processing Systems 30 (NeurIPS 30).
2. Qi, C., Yi, L., Su, H., & Guibas, L. J., 2017. *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* IEEE/CVF CVPR, 77–80.
3. Zaheer, M., Kottur, M., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., & Smola, A. J., 2017. *Deep Sets.* NeurIPS 30.
4. Lee, J., Lee, J., Kim, M., Kosiorek, A. R., Choi, S., & Teh, Y. W., 2019. *Set Transformer.* ICML, 3714–3723.
5. Mikolov, T., Chen, K., Corrado, G., & Dean, J., 2013. *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781.
6. Pennington, J., Socher, R., & Manning, C. D., 2014. *GloVe: Global Vectors for Word Representation.* EMNLP, 1532–1543.
7. Chung, S., & Abbott, L. F., 2021. *Geometry of a neural population.* Nature 592(7853), 117–122.
8. Hu, Z., Niu, L., & Varma, S., 2026. *Neural population geometry / point-cloud manifold analysis.* arXiv:2607.04525 [cs.CL]; DOI 10.48550/arXiv.2607.04525.
9. Xu, W., 2026. *Scale/PGA: Subspace Procrustes Geometry alignment analysis.* arXiv:2605.17084 [cs.LG]; DOI 10.48550/arXiv.2605.17084.

> **Citation-scope note:** As expressly stated in 2.2, the statement that “the specific combination is not shown in the surveyed sources” is limited to the sources in this list; it is not a global absence claim. A formal prior-art search (Espacenet, Google Patents, TÜRKPATENT, WIPO) must be completed by the agent.

---

## 7. AGENT ASSESSMENT (**NOT ADDED TO THE FILING — strategy note**)

### 7.1 Unity and divisional application

Presenting language, fraud, AML and maintenance together in a single application raises a “multiple inventions”/unity objection risk. The defense is that the single inventive idea is the **dual-use geometry mechanism** of Claim 1: the same learned low-dimensional cloud gives the reader a score and is the source of the native explanation (ablation, prototype/region, drift); the verticals are applications of this common mechanism. The independent system claim is vertical-independent and the four verticals are in separate dependent claims. If the examiner makes a unity objection, the option of separating the vertical embodiments into divisional applications can be considered.

### 7.2 Relationship to the TR national (language) filing / priority

This document expresses the universal dual-use core and is the superset of two specific documents: `../dil/patent_application_v2.3.md` (the language/next-token application) and `../tabular/patent_tabular_explain_v1.md` (the tabular/fraud identity-bucket, explanation and edge application).

**PCT–TR priority relationship (critical):** Our TR national filing `TURKPATENT_basvuru_v2.3.md` is **language-only** in scope. Since the scope of this PCT document (dual-use core + language/fraud/AML/maintenance) is **not disclosed** in the TR language filing, this PCT application **cannot claim priority** from the TR national (language) filing — it must be treated as two independent inventions/scopes. The recommended strategy: **TR national = language** (priority T1) and **PCT = universal + 4 verticals** (its own priority T2); a textual link between the documents does not, by itself, create an earlier date. A priority date arises only from a duly filed application; the confidentiality warning must be maintained until no document is published.

### 7.3 Non-obviousness and technical effect (Art. 69)

- **PointNet distinction:** in PointNet the points are raw 3D sensor/image input; here the point is a learned representation from different domain data.
- **Dual-geometry distinction:** the preferred D+4 vector — difference, distance, cosine and two norms — is specified here; the general symmetric pooling of PointNet/Deep Sets does not by itself show this combination.
- **Native-explanation distinction:** ablation, text/prototype and drift analysis are performed from the same decision cloud; this is not presented as the same as a SHAP feature list.
- **Technical effect:** the claims are tied to a readable native audit surface and to a self-contained, compact, verifiable deployment. With respect to Art. 69, the technical effect in the description is a computer system operating with the reader and the verifiable package, not an abstract mathematical method.

### 7.4 Scope and correctness corrections

- **C1 — Permutation:** Claim 1(a), Claim 2(c) and 2.7.1 limit the symmetric set aggregation; they state that points produced from context/history may be order-sensitive.
- **C2 — Transformation:** Claim 4 and 2.7.1 make the difference vector covariant and the distance/cosine/norm components invariant.
- **C3 — Package:** Claim 1(e), 2.4, 2.5 and 2.7.2 set the principle of a compact, self-contained, bit-exact verifiable deployment in place of a fixed KB/parity claim; the fraud and maintenance values are only embodiment examples.
- **C4 — Prior art:** 2.2.3 states that the specific combination is not shown in the surveyed sources; it makes no global “nowhere has it” claim and records the need for a broader search.

**Evidence-consistency note:** The 6.45/2.92 comparison for language is labelled in v2.3 as a shared held-out, but the current `RESEARCH.md` records that 6.45 is a sample from the training set and is not a strict held-out. This protocol difference must be resolved before filing; the present text does not advance 6.45 as a held-out proof.

---

*This document is the technical text of the PCT (international) patent application for the universal dual-use geometry core.*
*Prepared by: [Inventor Name]*
*Revision: v1 (PCT format: request form + EN version)*
*Date: 22 October 2026*
