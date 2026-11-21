# DESCRIPTION
<!-- ADAY kopyası — DESCRIPTION_EN.md (sha change yok) + AÇILIM_PLANI.md uygulaması: 14 paragraf eklendi, [00NN] topluca yeniden numaralandırıldı. TASLAK, vekil onayı gerekli; orijinal dosya değişmedi. -->

**Title of the invention (PCT Rule 4.3):** Dual-use point-cloud representation with embedded explainability

*Working copy — English description of a PCT international application. Paragraphs are numbered in fours
([0001], [0002], …). Section headings follow the order and names recommended in PCT Rule 5.1(a) and in
Section 204 of the PCT International Search and Preliminary Examination Guidelines. HTML comments carry
internal measurement provenance for the attorney and are stripped mechanically: the text to be filed is
generated into `filing/` by `make_filing.py`. Draft — attorney review required.*

## TECHNICAL FIELD

[0001] The present disclosure relates to the production of a task output by a machine learning model and to the production of data explaining that output. More particularly, the disclosure relates to computer-implemented methods in which a point cloud representation produced by a context encoder serves as the single source of both the task output and the explanation data, to computers configured to carry out such methods, and to non-transitory computer-readable recording media carrying instructions executable thereon.

[0002] The disclosure finds application in edge computing, financial transaction classification, machine health monitoring and text processing, that is, in systems in which the explanation of an output decision is subject to an auditable cost or obligation.

[0003] The disclosure carries no assertion that the accuracy of one class of model is higher than that of another class of model. The subject matter defined by the appended claims concerns the generation of task output data and explanation data from the same trained representation, without a separate explanation model and within a transportable package.

<!-- kaynak: OLCUM_KAYITLARI.md#S0 (iddia çerçevesi: docs/patent/final_v3/TARIFNAME.md:9, 154-155, 171) -->

## BACKGROUND ART

[0004] A prevailing approach in explainable machine learning is to generate the explanation by a post-hoc tool: alongside a trained classifier, a separate attribution tool operating on the input features is run, so that the task output and the explanation originate from different systems. Three technical gaps are observed in working prototypes built on that approach.

[0005] First, the stability of a post-hoc attribution is limited by retraining itself. On the same architecture retrained with five separate training seeds on a fixed test subsample of 400 records, a tree-based post-hoc attribution tool (installed version 0.54.1) produced a first-rank stability of 0.1025 for the attribution ranking, a Jaccard similarity of 0.43405 for the top-three set, and a Spearman correlation of 0.65105 for the attribution ranking.

[0006] On the same data and the same five seeds, the stability of a nearest-region label derived from the geometric representation of the model itself, defined as the fraction of test records whose label matches the reference-seed label in all retrains, was 0.325. The two sets of figures are not the same quantity: the first set describes an attribution ranking, the second describes the persistence of a label across retrains. No ratio and no superiority is asserted between them. The gap is that a post-hoc attribution yields a quantity that is less stable than the model itself and whose audit requires an additional step.

<!-- kaynak: OLCUM_KAYITLARI.md#S4 (aml/results/e5_stability.json:8-12, 14, 25, 29-30, 41-43) -->

[0007] Second, the explanation requires a separate software stack on the edge device: running a post-hoc attribution tool adds a further tool and a further computation step beside the deployed model. In one recorded prototype, over the same five training seeds, the fit-and-encode path of the geometric representation, which also produces the explanation data, averaged 0.35445423298515377 seconds per seed, and the path that retrains the tree model and runs the post-hoc attribution averaged 0.7402620668290183 seconds per seed. These two figures are durations of two different work sets; no general complexity ratio is derived from them. The recorded fact is that a post-hoc explanation carries an additional fitting and attribution step.

<!-- kaynak: OLCUM_KAYITLARI.md#S4 (aml/results/e5_stability.json:38) -->

[0008] Third, a drift audit also requires a separate instrument: monitoring the shift of region shares over time is derived from externally defined feature windows rather than from the structure of the representation, so a second monitoring pipeline has to be installed in order to measure the shift.

[0009] The common cause of these three gaps is that the explanation data is not present in the representation itself but is attached to the representation from outside as a second system.

## DISCLOSURE OF THE INVENTION

[0010] According to one aspect, a computer-implemented machine learning method processes a task input with a context encoder to produce context-encoded representations, wherein the task input consists of one of an ordered sequence of tokens and a plurality of measured field values of a tabular record, and represents each context-encoded representation as a point in a D-dimensional space, thereby constructing a point cloud of N points; computes point features for points and pair-geometry features, derived from geometric quantities dependent on the positions of the two points of a pair, for point pairs; combines both feature sets into a single set representation by a symmetric set aggregation independent of the input order of the points and of the pairs; and obtains a task output from a reader operating on a reader input, wherein the set representation derived from the context-encoded representations constitutes that reader input.

[0011] Explanation data is produced by applying at least one of three explanation operations to the point cloud of the trained model that produced the task output itself, namely a point-removal effect, region naming and region shift, without a separate explanation model, a separate explanation training or a separate explanation input. The task output and the explanation data are presented jointly, via an output interface, as outputs derived from the same set representation of the same trained model.

[0012] In further aspects, D is selected as an integer between 3 and 64; the pair-geometry features comprise a difference vector, a pair distance equal to its Euclidean norm, an angular similarity between the two point vectors and the norms of both points; the symmetric aggregation applies max-pooling over the points and average-pooling over the pairs and combines the results with a concatenation layer; N follows the context length of the task input and unfilled context positions and the pairs formed by them are excluded by a mask signal.

[0013] In further aspects, the point-removal effect masks a selected point, re-runs the reader and records the resulting logit difference as contribution data, with the masked clouds supplied jointly as one batch input so that the contributions of N points are obtained in N+1 forward passes; region naming partitions set representations or point cloud states by a centroid-based clustering operation and assigns each region a human-readable label built from at least one symbol of a frequency ranking of the input symbols of the records assigned to that region; region shift compares region point shares between a reference data group and a comparison data group and quantifies the difference with a probability distribution divergence measure, marking as an emerging region a region whose share is below a lower threshold in the reference distribution and above an upper threshold in the comparison distribution.

[0014] In further aspects, at least one parametric layer of the reader is represented in INT8 integer quantization and a single self-contained deployment package carries the quantized weights, a quantization table mapping input values to model input identities and a resolution table mapping the model input identities to human-readable input labels, so that both the task output and a human-readable form of the explanation data are produced on an edge device; the task input may be a text of several sentences, encoded as sentence cloud states and slid into neighbouring window cloud states by a fixed window size and stride and processed by a window-geometry encoder; and the pair computation may be replaced by global set statistics comprising maximum, minimum, mean and standard deviation pooling together with a cloud centre and a centre deviation, so that the computational cost is proportional to the number of points.

[0015] In further aspects, a training step updates parameters of the reader by gradient-based backpropagation and updates point positions of the point cloud without computing gradients, by zeroth-order accept and revert operations based on the measured change of a task loss caused by a candidate point position.

[0016] According to further aspects, a computer system comprising at least one processor and a memory holds the corresponding modules, namely a context encoder module, a point projection module, a pair-geometry encoder, an aggregation module, a reader, an interpretation engine operating on the same point cloud, and an output interface; and a non-transitory computer-readable recording medium stores instructions causing a computer to carry out the corresponding steps, including the joint recording of the task output and the explanation data.

[0017] The technical effect is expressed on measured axes, with the measurement conditions stated. The footprint of the package delivered to the edge device for the tabular task fell from 528.9 KB in 32-bit floating point format to 138.2 KB in INT8 format. Latency of the INT8 reader was 0.015912843984551728 ms per transaction at the 50th percentile and 0.017278228622744793 ms at the 99th percentile over 2000 transactions, at 62845.10641151401 transactions per second. The explanation cost is bounded because the point-removal effect requires N+1 forward passes and the pair computation grows with the square of the number of points: the multiply-accumulate count per transaction was 2522200 for 13 active points and 34091488 for 50 active points.

<!-- kaynak: OLCUM_KAYITLARI.md#S1, #S2 -->

[0018] The accuracy cost of quantization is measured and accepted as a boundary rather than omitted: on a naturally distributed held-out test set of 4700000 rows containing 6110 fraudulent records, at a base rate of 0.0013, the area under the receiver operating characteristic curve was 0.9450745846153957 in 32-bit floating point format, 0.9364812440872093 with per-tensor INT8 quantization and 0.9367113790742326 with per-channel INT8 quantization, the differences being -0.008593340528186388 and -0.008363205541163099 respectively. Because the explanation data is derived from the region of the representation itself, the stability of the explanation across five retraining seeds is auditable as a single figure, namely the region-label stability 0.325.

<!-- kaynak: OLCUM_KAYITLARI.md#S1, #S4 -->

[0019] The point positions of the cloud are updated without gradient computation: a candidate position is generated for a point, the task loss is measured with the candidate on the same mini-batch, the candidate is accepted when its loss lies below the base loss by a threshold, otherwise the position is reverted to its previous value and the search interval is narrowed on the candidate side.

<!-- kaynak: OLCUM_KAYITLARI.md#S10 (quann_cloud/src/qcloud_quann.py:212, 220, 223, 226) -->

## BRIEF DESCRIPTION OF THE DRAWINGS

[0020] FIG. 1 is a schematic block diagram of a system architecture, showing the branch in which a task output and a trio of explainability operations are derived from one and the same point cloud.

[0021] FIG. 2 is a schematic diagram of a point pair and of the components of the pair-geometry feature: a difference vector, a pair distance as a Euclidean norm, a cosine similarity, a norm vector of a first point and a norm vector of a second point.

[0022] FIG. 3 is a schematic diagram of the symmetric set aggregation combining max-pooling over the points and average-pooling over the pairs; a dashed path denotes the global-statistics variant whose cost is linear in the number of points.

[0023] FIG. 4 is a schematic diagram of the point-removal operation, the N+1 forward passes and the contribution definition e_k = s_full - s_-k, that is, the output in the unmasked state minus the output with point k masked.

[0024] FIG. 5 is a schematic diagram of the production of a human-readable region label from region statistics obtained by centroid-based clustering.

[0025] FIG. 6 is a measured diagram: panel (a) shows region point-share shifts between a training window and a drifted window of a complaint corpus; panel (b) shows the alarm rate of a machine-health data set before and after an applied setting shift.

[0026] FIG. 7 is a measured diagram: panel (a) shows areas under the receiver operating characteristic curve for 32-bit floating point, per-tensor INT8 and per-channel INT8 quantization on 4.7 million test rows; panel (b) shows the delivered package footprint of the same model.

[0027] FIG. 8 is a schematic diagram of a multi-sentence hierarchy built from sentence cloud states and neighbouring window cloud states.

[0028] FIGS. 1 to 5 and FIG. 8 define structure and contain no measurement; measurements are carried by FIG. 6 and FIG. 7.

## BEST MODE FOR CARRYING OUT THE INVENTION

[0029] The following describes the best mode currently contemplated for carrying out the invention; the variants described in paragraph [0079] are alternatives within the same teaching. A chain schematized in FIG. 1 comprises, in order, a task input; a context encoder processing the task input; a point projection mapping each context-encoded representation into a D-dimensional space; a pair-geometry encoder computing features for points and for point pairs; an aggregation module combining those features into one set representation; a reader producing the task output from the set representation; an interpretation engine producing explanation data from the same point cloud; and an output interface presenting the task output and the explanation data jointly.

[0030] Two features of this architecture are structural. The representation is processed as a set: the input ordering is not carried into the aggregation, and points corresponding to context-unfilled positions, together with the pairs formed by such points, are disabled by a mask signal. The interpretation engine is not a separate model; it re-runs the forward pass of the reader that produces the task output with the points of that same cloud modified. The explanation is therefore a second use of the representation rather than a second system attached to it.

<!-- kaynak: OLCUM_KAYITLARI.md#S9 (cloudlm/model.py:47) -->

[0031] The task input arrives in two forms. For an ordered token sequence, the representation produced by the context encoder at each position is projected into a D-dimensional vector and the number N of points equals the context length. For a tabular record, the measured field values of the record are themselves the points of the cloud of that record, so the number of fields determines N and thereby the number of pairs.

[0032] The task input is defined by the structure it presents to the context encoder. An ordered token sequence is composed of sub-word units selected from a fixed vocabulary, each position of the sequence carrying the identity of one such unit; the vocabulary of 10000 units recorded in paragraph [0074] is one recorded example of the language body. A tabular record is a set of measured field values of one entity, each field carrying its own measured value and its own field identity; the eight field values of the credit-card transaction record of paragraph [0068] are one recorded example, a transaction amount or a sensor reading being a measured field value of the kind referred to in paragraph [0002]. No further form of task input is required: the ordered sequence and the tabular record are the two forms from which the point cloud of paragraph [0010] is constructed.

<!-- kaynak: ACILIM_METINLERI_EN.md G1-P1 (kaynaklar: OLC:156 vocab; tarifname [0010] [0054] [0060] [0002]); AÇILIM_PLANI.md §2 kartı -->

[0033] Three terms carry a precise meaning in this description. The input order of the points is the numbering of the positions at which the input units are placed: position one, position two and so on along the sequence, or position one through position F over the fields of a record. That the symmetric set aggregation is independent of this order means that any permutation of the numbering yields the same set representation, so the aggregation output is a function of the set of points and not of their arrangement. A point cloud is a collection of N points in which no position within the collection carries information to the aggregation; in the tabular body the points are the fields of the record, and in the language body the points are the context positions of the sequence. A point in a D-dimensional space is the position vector of D coordinates assigned to one input unit by the point projection; the recorded values of D are those of paragraph [0035].

<!-- kaynak: ACILIM_METINLERI_EN.md G1-P2 (tarifname [0010] [0030] [0032]; aralık uydurulmadı); AÇILIM_PLANI.md §2 kartı -->

[0034] A context-encoded representation is the vector that the context encoder attributes to one input unit of the task input: for an ordered token sequence, one representation per token position; for a tabular record, one representation per field identity. The representation is context-dependent in that the value attributed to a unit is produced jointly with the representations of the other units of the same input, and not from the unit alone. The point projection then maps each such representation onto a point of the D-dimensional space, the pair input width of 3D + 2 defined in paragraph [0036] being consistent with this projection. The only encoder alternative recorded for the language body is the three-tier encoder of paragraph [0046]. As a recorded example of scale, the language chain of paragraph [0073] holds 20999648 parameters; this figure describes the chain as a whole and is not attributed to the context encoder alone.

<!-- kaynak: ACILIM_METINLERI_EN.md G2-P1 (OLC:153 ölçek; tarifname [0033] [0037] [0059]); AÇILIM_PLANI.md §2 kartı -->

[0035] In the recorded implementation examples the dimension D was chosen per task domain: D = 8 with a cloud state size of 128 for a tabular fraud task; D = 32 and D = 64 for two compared configurations of a language task; and an embedding size of 48 for a machine health task. The selected values of D lie between 3 and 64. In the tabular embodiment the number of model input identities was 221, the number of narrowed point positions 214 and the number of active points K 13.

<!-- kaynak: OLCUM_KAYITLARI.md#S1, #S3, #S4, #S5, #S8 -->

[0036] As schematized in FIG. 2, the pair-geometry feature of a pair (p_i, p_j) is the concatenation of five components: the difference vector between the two point vectors; the pair distance, that is the Euclidean norm of the difference vector; the cosine similarity between the two point vectors; the norm vector of the first point; and the norm vector of the second point. The difference vector and the distance carry the relative position of the pair, the cosine similarity carries its angular relation, and the two norm components carry the absolute position of each point with respect to the cloud centre. The resulting pair-feature width is the algebraic sum 3D + 2, which is the pair input width defined in the implementation.

[0037] The Euclidean norm of the difference vector is computed as the square root of the sum of the squares of the coordinate differences of the two point positions; it is the same quantity as the pair distance named above, stated here as an arithmetic operation rather than as a name. Of the five components, the difference vector and this distance express where the two points lie relative to one another, the cosine similarity expresses the angular relation between their position vectors, and the two norm components express where each point lies with respect to the cloud centre. The cost of carrying these pair features scales as recorded in paragraph [0049], which states the measured multiply-accumulate figures.

<!-- kaynak: ACILIM_METINLERI_EN.md G4-P1 (tarifname [0021] [0033] [0039]; alternatif norm kayıtlı değil); AÇILIM_PLANI.md §2 kartı -->

<!-- kaynak: OLCUM_KAYITLARI.md#S9 (cloudlm/model.py:63-64, 158, 161, 165, 168) -->

[0038] The aggregation schematized in FIG. 3 has two paths. In the point path the point features are reduced by max-pooling over the points, yielding a result of the max-pooling; in the pair path the pair-geometry features are reduced by average-pooling over the point pairs, yielding a result of the average-pooling; the result of the max-pooling and the result of the average-pooling are collected in one concatenation layer into a single set representation.

[0039] The result of the max-pooling is the vector obtained by selecting, in each point-feature dimension, the largest value among the points of the cloud; its width is fixed by the point-feature width and is therefore independent of the number N of points. The result of the average-pooling is the vector obtained by taking, in each pair-feature dimension, the mean over the valid pairs alone, consistent with the normalization by the number of valid pairs recorded in paragraph [0040]. Both are operations over a set: permuting the points leaves each pooled result unchanged, which makes the order-independence of paragraph [0030] visible on the pooling results themselves. The concatenation of the two results is supplied to the reader as the reader input of paragraph [0010]. In the global-statistics variant, the cloud centre is the mean position of the points of the cloud, in the sense of paragraph [0048]; no numerical range for it is recorded.

<!-- kaynak: ACILIM_METINLERI_EN.md G3-P1 (OLC:174 yalnız varlık künyesi; tarifname [0030] [0035] [0038]); AÇILIM_PLANI.md §2 kartı -->

[0040] Masking applies in both paths. In the point path, points corresponding to context-unfilled positions are set to an invalid value before the pooling and the pooled result is cleared by the mask afterwards. In the pair path a pair counts as valid only if both of its ends are valid; the features of invalid pairs are zeroed and the average is normalized by the number of valid pairs alone, so that unfilled positions contribute no numerical value to the set representation. Since both poolings are operations over a set, the set representation is independent of the numbering order of the points, which keeps the representation defined for a tabular record whose fields carry no ordering.

[0041] The context length of the task input is the number of input units the task arrangement admits: positions in an ordered sequence, fields in a tabular record. The number N of points equals that number when every position is filled, and a context-unfilled position, as masked in paragraph [0040], is a position within the context length that carries no input unit. The recorded examples tie this bound to concrete figures already described: a sequence length of 10 in the dimension comparison of paragraph [0072], a sequence length of 8 in the smaller-scale training of paragraph [0080], and K = 13 active points with 221 model input identities and 214 narrowed point positions in paragraph [0035]. No numerical mask value or threshold is recorded.

<!-- kaynak: ACILIM_METINLERI_EN.md G5-P1 (yalnız tarifname içi mevcut örnekler: [0032] [0035] [0058] [0066]); AÇILIM_PLANI.md §2 kartı -->

<!-- kaynak: OLCUM_KAYITLARI.md#S9 (cloudlm/model.py:151-155, 171-179) -->

[0042] The set representation is supplied to the reader. In a language task the reader produces a probability distribution of a next token, with the forward pass defined together with an attention mask, and the cloud state is taken from the cloud encoder. In a tabular task the reader produces a class assignment score, and the explanation step is built on the logit of the same reader. The quantity on which the explanation rests is therefore the output of the same reader that produces the task output.

[0043] The reader input is read by the reader in the following steps. The single set representation produced by the concatenation layer of paragraph [0038] enters the reader as its input state. In the language task the forward pass through the parametric layers of the reader is defined together with the attention mask, so that masked positions take no part in the pass, and the cloud state on which the explanation operations act is taken from the cloud encoder. The reader then maps that input state onto the task output. The term reader denotes one and the same structural element in both bodies: the module that turns the set representation into the class assignment score in the tabular task, and the module that turns it into the distribution of a next token in the language task, is a single component of the chain of paragraph [0029]. No further layer count or layer width of the reader is recorded beyond the implementation values of paragraph [0035].

<!-- kaynak: ACILIM_METINLERI_EN.md G7-P1 (OLC:174-176 bileşen künyeleri; tarifname [0029] [0034] [0036]); AÇILIM_PLANI.md §2 kartı -->

[0044] The two task outputs carry the following meanings. The probability distribution of a next token is a distribution over the identities of the vocabulary of the language task, the recorded vocabulary of 10000 units of paragraph [0074] being one example of the identity set over which it is defined; the distribution assigns to each vocabulary identity the probability that the identity follows the context seen so far. The class assignment score is a scalar on which the decision boundary between classes of the record is placed; the fraud score of the eight-field transaction record of paragraph [0068] is a recorded example. In the tabular task the explanation operation is built on the logit that carries this score, so that the reading recorded in paragraph [0042] and the explanation operate on one and the same output quantity of one and the same reader.

<!-- kaynak: ACILIM_METINLERI_EN.md G7-P2 (OLC:156 vocab; tarifname [0036] [0054] [0060]); AÇILIM_PLANI.md §2 kartı -->

[0045] As a recorded example of the reader in deployment, the self-test of the package of paragraph [0061] reports, over 19963 rows, a largest absolute difference of 1.2814998626708984e-06 against the 32-bit floating point reference and a throughput of 26473.303851652894 transactions per second, and the INT8 reader latency was measured at 0.015912843984551728 ms per transaction at the 50th percentile over 2000 transactions. The first figures are recorded in paragraph [0063] and the latency in paragraph [0017]; they are gathered here once, under the heading of the reader, to state that the quantized reader which produces the task output also produces the quantities on which the explanation operations rest, within the recorded difference.

<!-- kaynak: ACILIM_METINLERI_EN.md G7-P3 (OLC:22,25 edge_eval_full_scale.json:23,21; OLC:62,63,65 manifest.json:27,28,30; tarifname [0017] [0051]); AÇILIM_PLANI.md §2 kartı -->

<!-- kaynak: OLCUM_KAYITLARI.md#S9, #S7 (cloudlm/model.py:305, 313; fraud/src/explain_pilot.py:103) -->

[0046] In the variant schematized in FIG. 8 the task input is a text of several sentences and the context encoder operates in three tiers. In the first tier each sentence is encoded as a sentence cloud state. In the second tier consecutive sentence cloud states are slid by a fixed window size and a fixed stride to derive neighbouring window cloud states; a window size of 5 and a stride of 2 are defined in the implementation. In the third tier geometric quantities between windows are collected by a window-geometry encoder and the resulting multi-sentence set representation is supplied to the reader. The fixed window size is a second design knot bounding the growth of the number of pairs: the number of pairs within a window scales with the square of the window size and the number of windows scales with the sentence count divided by the stride.

[0047] The hierarchy of paragraph [0046] operates as follows. The text of the task input is divided into sentences; this division is performed as an ordinary pre-processing step of the language body, and no particular segmentation algorithm is required. Each sentence is encoded as a sentence cloud state. A window covering a consecutive run of sentence cloud states is then advanced along the text by the fixed stride, and each position of the window produces one window cloud state. Because the stride is smaller than the window size, consecutive windows share sentences, so a sentence that lies near a change of subject contributes to the window cloud states of both neighbours; this overlap is what the fixed stride produces. The window-geometry encoder collects the geometric quantities between the windows of the window cloud states and emits the multi-sentence set representation, which in this variant is the reader input. The recorded window size and stride, 5 and 2, are those of paragraph [0046]; the number of sentences in a text and the number of windows thereby produced are not bounded by any recorded value, and the recorded split counts are those of paragraph [0074].

<!-- kaynak: ACILIM_METINLERI_EN.md G9-P1 (OLC:159,160 contextcloud künyeleri; tarifname [0037] [0060]); AÇILIM_PLANI.md §2 kartı -->

<!-- kaynak: OLCUM_KAYITLARI.md#S8 (contextcloud/model.py:20, 121, 127-128, 196, 246) -->

[0048] The dashed path of FIG. 3 is a variant replacing the pair computation. Instead of pair-geometry features and average-pooling over pairs, global set statistics of the point cloud are computed, comprising maximum, minimum, mean and standard deviation pooling over the point dimensions together with the cloud centre and the centre deviation. Because no pair computation is performed, or it is performed approximately, the cost is proportional to the number of points.

<!-- kaynak: OLCUM_KAYITLARI.md#S8 (purecloudlm/model.py:8-9, 57, 62-63) -->

[0049] The cost bound that motivates this variant is visible in the pair path: raising the number of active points from 13 to 50 raised the multiply-accumulate count per transaction from 2522200 to 34091488. The measured ratio of the multiply-accumulate counts is 13.5 and the square of the ratio of the point counts, (50/13)^2, is 14.8, which is consistent with a quadratic growth of the number of pairs. N is therefore bounded by the context length of the task input, and on an edge target the global-statistics path may be selected instead of the pair path.

<!-- kaynak: OLCUM_KAYITLARI.md#S2 -->

[0050] In the first explanation operation schematized in FIG. 4, the point cloud from which the task output was produced is preserved and a selected point is temporarily disabled. A point cloud is produced from the input units; the reader is run in the unmasked state to obtain a base output; the point concerned is set to a zero value in a copy of the cloud and its mask is closed; the reader is re-run with the masked cloud; and the contribution of that point is recorded as the base output minus the masked output. The definition is recorded as a logit difference, namely the base logit minus the logit with the point concerned masked.

[0051] The same operation runs on the quantized deployment: masked cloud variants are prepared as a single batch input, each variant mask being built by closing the position concerned, and the INT8 inference session is invoked once; the contribution is produced as a logit difference. The N+1 forward passes for N points are thereby reduced to one batched call.

<!-- kaynak: OLCUM_KAYITLARI.md#S7 (fraud/src/explain_pilot.py:102-111; fraud/pilot/infer.py:97-107) -->

[0052] The interpretation of this operation is bounded as recorded in the prototype: the quantity produced is a removal sensitivity, not a single-additive attribution, and because of the pair computation the contributions are not additive. A contribution is therefore presented as the measured change of the decision when that point is removed, and not as the share of one input feature. In one pilot the input units are ranked by the product of the mean absolute logit contribution and the square root of the observation count.

<!-- kaynak: OLCUM_KAYITLARI.md#S7 (fraud/results/explain_pilot.json:21, 23, 24) -->

[0053] In the second explanation operation schematized in FIG. 5, the set representations or point cloud states of a plurality of records are partitioned into geometric regions by a centroid-based clustering operation; KMeans is used in the prototype, with eight regions in the complaint embodiment. In a classification embodiment the regions may alternatively be defined as prototypes taken as the means of the class representations of the training set, a definition established on training data only.

[0054] Each region receives a human-readable label by combining at least one symbol selected from a frequency ranking of the input symbols of the records assigned to that region; the first eight ranked symbols are used in the prototype. Measured labels, recorded with sub-word boundary markers, are "credit report information" for two distinct regions and "credit report consumer" for a further region.

[0055] A verifiable boundary of region naming is recorded: two geometrically distinct regions may carry the same human-readable label, as in the measured example in which two separate regions share the label "credit report information". A region label is therefore not an identity by itself; the audit is performed jointly with the region index.

<!-- kaynak: OLCUM_KAYITLARI.md#S6 (complaints/src/explain_region.py:47, 66; complaints/results/region_map.json:2, 422, 838, 1462; aml/src/e1_explain.py:297) -->

[0056] In the third explanation operation, whose measurements are shown in FIG. 6, the point shares of the same geometric regions are compared between a reference data group and a comparison data group, and the difference between the share distributions is quantified by a probability distribution divergence measure. Two measures are used: a total variation measure defined as half the sum of the absolute differences of the region shares, and a Kullback-Leibler divergence. Regions whose share lies below a lower threshold in the reference distribution and above an upper threshold in the comparison distribution are marked as emerging regions. No drift label is used by definition; the reference is the set of region centres fixed in the training window.

[0057] The mark of an emerging region is carried in the audit record as a triple: the region index, the human-readable region label of paragraph [0054], and the pair of point shares measured for that region in the reference distribution and in the comparison distribution; consistent with paragraph [0055], the label accompanies the index and does not replace it. The lower and upper thresholds are parameters of the task arrangement; no numerical threshold values are recorded. The mark is produced by the comparison step alone: it states that the share of the region lay below the lower threshold in the reference distribution and above the upper threshold in the comparison distribution, and, as recorded in paragraph [0081], it is claimed as a criterion so defined and not as an event observed in the measured drift.

<!-- kaynak: ACILIM_METINLERI_EN.md G6-P1 (OLC:201 sınır notu; tarifname [0044] [0045] [0046] [0067]); AÇILIM_PLANI.md §2 kartı -->

<!-- kaynak: OLCUM_KAYITLARI.md#S6 (complaints/src/audit_drift.py:36, 40, 44; complaints/results/drift_report.json:2) -->

[0058] Measured weekly values, reported against the preceding week, were 0.0903703703703704 total variation and 0.0341397185589417 Kullback-Leibler divergence for the week 2025-02-03/2025-02-09, and 0.02469135802469135 total variation and 0.0018965315544816688 Kullback-Leibler divergence for the following week 2025-02-10/2025-02-16.

[0059] The quantity plotted in panel (a) of FIG. 6 is the difference of region point shares between the training window and the drifted window, measured on a subsample of 20000 records consisting of 14000 training and 6000 drifted records. The measured extremes of the per-region share difference are -0.17 for region 1 and +0.2068 for region 7. The stability of the region assignment under subsampling was also measured: a mean adjusted Rand index of 0.9938922500989028 and a mean rank correlation of region shares of 1.0.

<!-- kaynak: OLCUM_KAYITLARI.md#S6 (complaints/results/complaints_scale_stability.json:2-4, 30, 32, 64-66, 72) -->

[0060] Panel (b) of FIG. 6 measures the effect of a setting shift on the alarm rate. On held-out healthy records that were not disturbed, the alarm rate was 0.011723329425556858; after a constant shift vector [6.0, 6.0, -6.0] was applied to three measured fields, the alarm rate was 0.02813599062133646, recorded as a ratio of 2.4. The alarm threshold is the distance value computed from the 0.95 quantile of the training distribution, namely 8.460468292236328, and the number of held-out healthy records is 853.

<!-- kaynak: OLCUM_KAYITLARI.md#S5 (maintenance/results/e1_cmapss.json:186-192; maintenance/src/e1_real.py:293) -->

[0061] In the deployment schematized in panel (b) of FIG. 7, the parametric layers of the reader are quantized to INT8 integer format and the reader carries a distribution store holding, in one self-contained package, three things: the quantized model weights, a quantization table mapping input values to model input identities, and a resolution table mapping the model input identities to human-readable input labels. The recorded components of the deployment package are a quantized inference model of 548.4 KB, a quantization table of 5.4 KB, a resolution table of 16.5 KB and 221 resolution entries. The mappings are recorded as follows: raw rows become identities through the quantization table, and identities become human-readable labels through the resolution table.

[0062] The distribution store is a physical store of one kind: the three structures named above are held together as a single file or a single set of files, and they are read into memory in the order of use, the quantization table mapping a raw input row onto a model input identity and the resolution table then mapping that identity onto its human-readable label. The computer system of paragraph [0016] holds these structures in its memory, and its at least one processor executes the stored instructions, which configure the system to operate as the modules of the chain of paragraph [0029]: the context encoder module, the point projection module, the pair-geometry encoder, the aggregation module, the reader, the interpretation engine and the output interface are hardware blocks configured by that instruction set, not free-standing machines. The same set of weights then produces the task output and the three explanation operations, the interpretation engine being a second use of the representation and not a second system, as stated in paragraph [0030]. No hardware designation and no range of processor counts is recorded.

<!-- kaynak: ACILIM_METINLERI_EN.md G11-P1 (OLC:66 manifest.json:97-98; tarifname [0011] [0016] [0029] [0030] [0050]); AÇILIM_PLANI.md §2 kartı -->

[0063] The task output and the human-readable explanation thus leave the same package on the edge device: the point-removal operation is executed on the INT8 session and the label of the masked input identity is read from the resolution table. In the self-test of the package, over 19963 rows, the largest absolute difference against the 32-bit floating point reference was 1.2814998626708984e-06, the 99th percentile absolute difference was 8.401274840252881e-07, and the throughput was 26473.303851652894 transactions per second.

<!-- kaynak: OLCUM_KAYITLARI.md#S3 (fraud/results/deploy/manifest.json:18, 21, 23, 24, 27-30, 97-98; fraud/pilot/infer.py:104) -->

[0064] The footprint budget is also tracked by counting unused parameters: in the tabular deployment the number of parameters belonging to the language model and unused in the package was recorded as 8392. This record shows in an auditable manner which part of one and the same architecture is carried for which task; it contains no measurement of multi-task performance.

<!-- kaynak: OLCUM_KAYITLARI.md#S1 (fraud/results/edge_eval_full_scale.json:11) -->

[0065] Training has two parts. The weights of the reader are updated by gradient-based backpropagation. The positions of the point cloud are updated without computing a gradient: a candidate position is produced for a point position, the task loss is measured with the candidate on the same mini-batch, the candidate is accepted if its loss falls below the base loss by a threshold, and otherwise the position is reverted to its previous value while the search interval is narrowed on the candidate side. This zeroth-order acceptance and reversion requires no backpropagation for updating point positions.

[0066] Gradient-based backpropagation denotes the update of parameters by evaluating the gradient of the task loss with respect to those parameters and moving the parameters along that gradient. That the point positions are updated without computing gradients means that no gradient of the task loss with respect to a point position is evaluated at any step. The update proceeds as a loop. For a selected point, a candidate position is generated in the neighbourhood of its current position. The task loss is measured with the candidate in place of the current position, on the same mini-batch that produced the base loss, so that candidate and base are compared under identical data conditions. If the loss measured with the candidate falls below the base loss by at least the threshold, the candidate is accepted as the new position of the point; otherwise the previous position is restored and the search interval on the side of the candidate is narrowed before the next candidate is generated. The numerical width of the neighbourhood, the value of the threshold and the interval parameters are not recorded.

<!-- kaynak: ACILIM_METINLERI_EN.md G10-P1 (OLC:185-187 qcloud_quann.py adımları; tarifname [0019] [0053]); AÇILIM_PLANI.md §2 kartı -->

[0067] A further working example of the third variant of paragraph [0079] is recorded on a language corpus. In a run of a training arrangement providing two point-update channels, the pair channel was selected and the recorded selection covered 86080 parameters; the held-out perplexity measured for that run was 5233.4086. A control run under a protocol recorded as identical at the level of the command arguments measured a held-out perplexity of 19770.644. The recorded boundary is stated with the figures: the comparison rests on a single seed, no paired repeat run exists, and repetition is contemplated; accordingly no improvement or reduction is asserted here. The example shows that the gradient-free update of the third variant was exercised with a selected pair channel at the recorded parameter selection.

<!-- kaynak: ACILIM_METINLERI_EN.md G10-P2 (RESEARCH:23,107 → run_r8_pair_freeze_only.log:11,12 = 5233.4086, 86080; run_r8_quann_pair.log:30 = 19770.644; protokol run_r8_quann_pair.sh:37,43; tek seed/tekrar adayı sınırı aynen yazıldı); AÇILIM_PLANI.md §2 kartı -->

<!-- kaynak: OLCUM_KAYITLARI.md#S10 (quann_cloud/src/qcloud_quann.py:212, 220, 223, 226) -->

[0068] In a first measured example, a tabular classification task produced a fraud score from the eight field values of a credit-card transaction record, with D = 8, a cloud state size of 128 and K = 13. Training on a 100000-record sub-slice containing 130 fraudulent records, at a base rate of 0.0013, completed in 2852.3 seconds and produced, in this configuration, an area under the receiver operating characteristic curve of 0.9351451502337654 and a recall of 0.7307692307692307.

[0069] The boundary of this example is recorded. In a full-scale comparison on the same data family the test set holds 4716078 rows, and two further comparison models produced higher areas under the curve: 0.9933922442255942 for a gradient-boosted tree ensemble, 0.9694542323758608 for a transformer-based tabular model, while a second gradient-boosted tree ensemble produced 0.9364403244701196. No superiority in accuracy is asserted. What this example supports is that a representation which is deliverable to an edge device and which produces its own explanation and region audit yields the same task output beside a high-accuracy tree model.

<!-- kaynak: OLCUM_KAYITLARI.md#S1, #S1 (fraud/results/full_scale_hybrid.json:5-7, 10, 16, 23, 56-57, 61; fraud/results/duels.json:4, 26, 50, 113) -->

[0070] In a second measured example the areas under the curve of FIG. 7, panel (a), were taken on a naturally distributed held-out test set whose base rate is preserved: 4700000 rows, 6110 fraudulent records, base rate 0.0013, giving 0.9450745846153957 in 32-bit floating point format, 0.9364812440872093 with per-tensor INT8 quantization and 0.9367113790742326 with per-channel INT8 quantization, with quantization costs of -0.008593340528186388 and -0.008363205541163099. The same measurement is written in rounded form in the run log as 0.9451, 0.9365 and 0.9367.

[0071] The output drift of the quantized model was also measured over 4096 randomly selected QUANN-point proxy inputs: the largest probability difference was 0.3432929813861847, the mean difference 0.008917972445487976 and the 99th percentile difference 0.14572913944721222. This is the measure of an output difference that quantization accepts, per individual decision, for the sake of deliverability. The footprint of panel (b) is the size of the package delivered to the edge device: 528.9 KB in 32-bit floating point format against 138.2 KB in INT8 format, and the quantized inference file of the evaluation artefact is 543.5 KB with a largest absolute difference of 1.0058283805847168e-07 against the floating point reference.

<!-- kaynak: OLCUM_KAYITLARI.md#S1, #S2, #S3 -->

[0072] In a third measured example, two configurations of one training arrangement were compared on a separate held-out set for the choice of dimension: a held-out perplexity of 61.94 for D = 32 and 61.19 for D = 64, a difference of 0.76, with D = 64 recorded as the winner. The protocol is recorded on the same line: 2000 samples, seed 0, row interval [200000, 202000), sequence length 10. These values are the measurement of one dimension choice and are not compared with perplexity values measured under other protocols.

[0073] The boundary is recorded here too. On a common held-out set of 2000 sentences the perplexity was 6.45 for the point-cloud language model and 2.94 for a causal language model baseline, the baseline measuring lower; the compared models hold 20999648 and 22724768 parameters respectively. No superiority in accuracy is asserted for the language task. What the language task supports is that the task output and the three explanation operations are producible from one and the same representation.

<!-- kaynak: OLCUM_KAYITLARI.md#S8 (logs/run_r4_resume.log:67, 70-71; logs/run_compare.log:13-14, 16-18) -->

[0074] In the multi-sentence hierarchy of FIG. 8 the data was divided in the ratio 80/10/10 with seed 42, and the split was recorded as 159888 training, 19986 validation and 19986 test sentences; the vocabulary was 10000 and the model 12260784 parameters. Training reached a lowest validation loss of 3.2305 and a separate held-out test loss of 3.2318. These are cross-entropy losses, not perplexities, and they are not presented as the same metric as the values of the preceding paragraphs.

<!-- kaynak: OLCUM_KAYITLARI.md#S8 (contextcloud/task_utils.py:17-19; logs/run_r5_cc_stable.log:2, 6-7, 1235-1236) -->

[0075] In a fourth measured example the explanation outputs were produced without any separate explanation model. The contribution pilot is a stratified development sample of 827 records containing 27 fraudulent records at a base rate of 0.032648125755743655, and the file records expressly that this sample is not the natural 4.7 million held-out population; the area under the curve measured on this sample was 0.8286111111111112. What this pilot supports is not production accuracy but that contribution data can be produced without a separate explanation model: the contribution is a logit difference, a positive value means that the input unit supports the fraud prediction, and the recorded caveat states that the contributions are removal sensitivities and are not additive.

[0076] For region and prototype labelling, the held-out accuracy of a nearest-prototype assignment against prototypes fitted in training was 0.75 on a graph transaction data family whose test set holds 400 nodes of which 40 are illicit, with per-class accuracies of 0.7833333333333333 and 0.45. The prototypes are defined as the means of the training nodes only. The recorded boundary states that one-hop graph features are transductive and that the source data carries transaction nodes rather than account identities.

[0077] The state-dependent limit of region purity was likewise measured. In a machine health task the nearest-prototype accuracy on a held-out test set of 1064 records was 0.8167293233082706, and it fell with the degradation stage: 0.9378663540445487 on 853 healthy records, 0.18421052631578946 on 114 intermediate records and 0.64 on 25 records of the critical stage. This is the measured statement that a region label cannot serve alone as audit evidence for degraded states.

<!-- kaynak: OLCUM_KAYITLARI.md#S7, #S4, #S5 -->

[0078] In a fifth measured example the stability of the explanation was audited. On five separate training seeds and a fixed test subsample of 400 records, the region-label stability was 0.325, defined as the fraction of test records whose nearest label matches the reference-seed label across all retrains. The post-hoc attribution path measured under the same protocols yielded three separate quantities: a top-1 rank stability of 0.1025, a top-3 set Jaccard similarity of 0.43405 and an attribution-ranking Spearman correlation of 0.6510458604357529, with a tree attribution tool of installed version 0.54.1. Since these quantities are not the same quantity as 0.325, no ratio and no superiority is drawn between them; the result used by the disclosure is that the stability of the explanation becomes traceable by a single figure without a second instrument independent of the training.

<!-- kaynak: OLCUM_KAYITLARI.md#S4 -->

[0079] Further variants are contemplated. In a first variant the pair computation is replaced by global set statistics, as described above with reference to the dashed path of FIG. 3, and this path is selected when the number of points has to grow on an edge device. In a second variant the sentence-cloud and window-cloud hierarchy of FIG. 8 is used, in which the task input is a text consisting of a plurality of sentences and the context encoder encodes each sentence of the plurality of sentences as a sentence cloud state. In a third variant the point positions are updated by the acceptance and reversion described above while the reader is trained by backpropagation; the cost limit of this third variant was also measured, since a zeroth-order update requires one forward pass per candidate and was therefore not found practicable on a larger model, so the measurement was made at a smaller scale of the same architecture.

[0080] In that smaller-scale measurement the held-out perplexity was 1045.03 for an untrained baseline, 563.45 for training by backpropagation and 1016.30 for the zeroth-order update, on 6000 training and 500 evaluation sentences, a sequence length of 8 and a vocabulary of 1024 pieces. The zeroth-order variant produced a higher perplexity than backpropagation training; it is sought for its provision of a point update that requires no backpropagation and not for an accuracy assertion.

<!-- kaynak: OLCUM_KAYITLARI.md#S10 -->

[0081] The following boundaries apply to the whole of the above description. The figures were measured in single configuration and corpus combinations, for example D = 8, a cloud state size of 128 and K = 13 for the tabular task; no generalization beyond those combinations is asserted. The accuracy cost of the INT8 transition was measured and its acceptance is the basis of the deliverability statement. The explanation data is not an additive attribution. A region label is not an identity. An emerging-region mark is defined as a criterion; no emerging-region event was recorded in the measured complaint drift, and the mark is therefore claimed as a criterion and not as an observed occurrence. No area of this description asserts superiority of accuracy over tree ensembles, transformer-based tabular models or causal language models.

<!-- kaynak: OLCUM_KAYITLARI.md#S11, #S6, #S4, #S1 -->

## INDUSTRIAL APPLICABILITY

[0082] The present disclosure is directly applicable in systems that run on an edge device and whose decisions are subject to an audit requirement. In financial transaction classification the INT8 package is delivered with a footprint of 138.2 KB, produces a latency of 0.015912843984551728 ms per transaction and a throughput of 62845.10641151401 transactions per second, and the input identities on which a declined transaction rests are read from the resolution table as human-readable labels.

[0083] In machine health monitoring the region shift shows, without any label, that a setting change moved the alarm rate from 0.011723329425556858 to 0.02813599062133646. In text processing the multi-sentence hierarchy is trained on a separate 80/10/10 split with seed 42, and the region labels are produced from ranked human-readable symbols of the complaint subjects of the records forming each region.

[0084] The components required for production are standard: a processor and a memory, a reader whose weights are stored in INT8 format, two mapping tables, and an interpretation step that re-runs the forward pass with masked inputs. Since no separate explanation training, no separate explanation model and no separate explanation input are required, the explanation capability is versioned and audited together with the model, which is the property relied upon in regulated edge deployments such as payment decisioning, predictive maintenance and document triage.

<!-- kaynak: OLCUM_KAYITLARI.md#S1, #S3, #S5, #S6, #S8 -->
