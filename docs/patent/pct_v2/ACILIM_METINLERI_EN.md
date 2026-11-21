# AÇILIM METINLERİ (EN) — 14 taslak paragraf, AÇILIM_PLANI.md kartlarına göre

*Durum: TASLAK — vekil onayı olmadan başvuru metnine dönüştürülmez. Bu dosya hiçbir hukuki
hüküm kurmaz; amaç, `review_specification` bulgularının işaretlediği öğeler için destek
görünürlüğünü ölçülebilir biçimde artıracak paragraf metnini üretmektir.*

- **Plan:** `patent-cikti/uspto/pct_v2-uspto-inceleme/ACILIM_PLANI.md` (grup kartları §2).
- **Hedef dosya:** `docs/patent/pct_v2/DESCRIPTION_EN.md` (orijinal, değiştirilmedi) → aday:
  `docs/patent/pct_v2/DESCRIPTION_EN_ADAY.md` (ekleme + yeniden numaralandırma, `scratchpad/insert_and_renumber.py`).
- **Sayı kuralı:** aşağıdaki her paragrafta geçen sayı ya tarifnamede ZATEN YAZILIDIR (atıf
  cümlesiyle bağlanır) ya AÇILIM_PLANI'nın künyeli değeridir (OLC / RESEARCH:23,107).
  Plan "kaydedilmemiş" dediği yerde metin aralık vermez, nitel anlatır.
- Aşağıdaki `[00NN]` atıfları ORİJİNAL numaralandırmaya göredir; ADAY dosyada script
  topluca yeniden numaralandırır.
- Her paragrafın altındaki tek satır künye: hedef konum · grup/istemler · dayanak kaynaklar.

---

## G1 — Görev girdisi + nokta-bulut inşası (P1 · 20 bulgu · istem 1, 10, 11, 12, 13, 15)

### G1-P1

The task input is defined by the structure it presents to the context encoder. An ordered token sequence is composed of sub-word units selected from a fixed vocabulary, each position of the sequence carrying the identity of one such unit; the vocabulary of 10000 units recorded in paragraph [0060] is one recorded example of the language body. A tabular record is a set of measured field values of one entity, each field carrying its own measured value and its own field identity; the eight field values of the credit-card transaction record of paragraph [0054] are one recorded example, a transaction amount or a sensor reading being a measured field value of the kind referred to in paragraph [0002]. No further form of task input is required: the ordered sequence and the tabular record are the two forms from which the point cloud of paragraph [0010] is constructed.

> Künye: [0031] sonrasına, 1. paragraf · G1 (task input ×4, tabular record ×3, plurality of sentences ×1; istem 1,10,13,15) · dayanak: [0010],[0054],[0060],[0002]; vocab künyesi OLC:156 → logs/run_r5_cc_stable.log:6.

### G1-P2

Three terms carry a precise meaning in this description. The input order of the points is the numbering of the positions at which the input units are placed: position one, position two and so on along the sequence, or position one through position F over the fields of a record. That the symmetric set aggregation is independent of this order means that any permutation of the numbering yields the same set representation, so the aggregation output is a function of the set of points and not of their arrangement. A point cloud is a collection of N points in which no position within the collection carries information to the aggregation; in the tabular body the points are the fields of the record, and in the language body the points are the context positions of the sequence. A point in a D-dimensional space is the position vector of D coordinates assigned to one input unit by the point projection; the recorded values of D are those of paragraph [0032].

> Künye: [0031] sonrasına, 2. paragraf · G1 (point cloud ×5, input order ×3, ordered sequence ×3, point in D-space ×1; istem 1,11,12,13,15) · dayanak: [0010],[0030],[0032]; aralık uydurulmadı (D aralığı [0012]'de yazılı).

## G2 — Bağlam kodlayıcı çıktısı (P2 · 3 bulgu · istem 1, 13, 15)

### G2-P1

A context-encoded representation is the vector that the context encoder attributes to one input unit of the task input: for an ordered token sequence, one representation per token position; for a tabular record, one representation per field identity. The representation is context-dependent in that the value attributed to a unit is produced jointly with the representations of the other units of the same input, and not from the unit alone. The point projection then maps each such representation onto a point of the D-dimensional space, the pair input width of 3D + 2 defined in paragraph [0033] being consistent with this projection. The only encoder alternative recorded for the language body is the three-tier encoder of paragraph [0037]. As a recorded example of scale, the language chain of paragraph [0059] holds 20999648 parameters; this figure describes the chain as a whole and is not attributed to the context encoder alone.

> Künye: G1 bloğundan hemen sonra ([0031] sonrası bölge, 3. paragraf) · G2 (context-encoded representations ×3; istem 1,13,15) · dayanak: [0010],[0031],[0033],[0037],[0059]; ölçek künyesi OLC:153 → logs/run_compare.log:13-14 (tarifnamede [0059]'de yazılı); encoder-özel aralık kaydedilmemiş — yazılmadı.

## G3 — Poolama sonuçları + bulut merkezi (P2 · 3 bulgu · istem 4, 11)

### G3-P1

The result of the max-pooling is the vector obtained by selecting, in each point-feature dimension, the largest value among the points of the cloud; its width is fixed by the point-feature width and is therefore independent of the number N of points. The result of the average-pooling is the vector obtained by taking, in each pair-feature dimension, the mean over the valid pairs alone, consistent with the normalization by the number of valid pairs recorded in paragraph [0035]. Both are operations over a set: permuting the points leaves each pooled result unchanged, which makes the order-independence of paragraph [0030] visible on the pooling results themselves. The concatenation of the two results is supplied to the reader as the reader input of paragraph [0010]. In the global-statistics variant, the cloud centre is the mean position of the points of the cloud, in the sense of paragraph [0038]; no numerical range for it is recorded.

> Künye: [0034] sonrasına · G3 (result of the max-pooling · result of the average-pooling · cloud centre; istem 4,11) · dayanak: [0030],[0034],[0035],[0038],[0010]; birleşik temsil genişliği kayıtlı değil (OLC:174 → cloudlm/model.py:179 varlık künyesi) — sayı yazılmadı.

## G4 — Öklid normu (P3 · 1 bulgu · istem 3)

### G4-P1

The Euclidean norm of the difference vector is computed as the square root of the sum of the squares of the coordinate differences of the two point positions; it is the same quantity as the pair distance named above, stated here as an arithmetic operation rather than as a name. Of the five components, the difference vector and this distance express where the two points lie relative to one another, the cosine similarity expresses the angular relation between their position vectors, and the two norm components express where each point lies with respect to the cloud centre. The cost of carrying these pair features scales as recorded in paragraph [0039], which states the measured multiply-accumulate figures.

> Künye: [0033] sonrasına · G4 (Euclidean norm of the difference vector; istem 3) · dayanak: [0021],[0033],[0039]; alternatif norm (Manhattan vb.) kaynaklarda kayıtlı değil — alternatif dil kurulmadı.

## G5 — Bağlam uzunluğu (P3 · 1 bulgu · istem 5)

### G5-P1

The context length of the task input is the number of input units the task arrangement admits: positions in an ordered sequence, fields in a tabular record. The number N of points equals that number when every position is filled, and a context-unfilled position, as masked in paragraph [0035], is a position within the context length that carries no input unit. The recorded examples tie this bound to concrete figures already described: a sequence length of 10 in the dimension comparison of paragraph [0058], a sequence length of 8 in the smaller-scale training of paragraph [0066], and K = 13 active points with 221 model input identities and 214 narrowed point positions in paragraph [0032]. No numerical mask value or threshold is recorded.

> Künye: [0035] sonrasına · G5 (context length of the task input; istem 5) · dayanak: [0012],[0031],[0032],[0035],[0039],[0058],[0066]; yeni sayı yok — yalnız mevcut örneklerin sınır kuralına bağlanması.

## G6 — emerging region işareti (P3 · 1 bulgu · istem 8)

### G6-P1

The mark of an emerging region is carried in the audit record as a triple: the region index, the human-readable region label of paragraph [0044], and the pair of point shares measured for that region in the reference distribution and in the comparison distribution; consistent with paragraph [0045], the label accompanies the index and does not replace it. The lower and upper thresholds are parameters of the task arrangement; no numerical threshold values are recorded. The mark is produced by the comparison step alone: it states that the share of the region lay below the lower threshold in the reference distribution and above the upper threshold in the comparison distribution, and, as recorded in paragraph [0067], it is claimed as a criterion so defined and not as an event observed in the measured drift.

> Künye: [0046] sonrasına · G6 (emerging region; istem 8) · dayanak: [0013],[0044],[0045],[0046],[0067]; eşik değerleri kaydedilmemiş; sınır notu OLC:201 → docs/patent/final_v3/TARIFNAME.md:227.

## G7 — Reader başlığı + çıktı türleri (P1 · 8 bulgu · istem 1, 6, 9, 10, 14, 15)

### G7-P1

The reader input is read by the reader in the following steps. The single set representation produced by the concatenation layer of paragraph [0034] enters the reader as its input state. In the language task the forward pass through the parametric layers of the reader is defined together with the attention mask, so that masked positions take no part in the pass, and the cloud state on which the explanation operations act is taken from the cloud encoder. The reader then maps that input state onto the task output. The term reader denotes one and the same structural element in both bodies: the module that turns the set representation into the class assignment score in the tabular task, and the module that turns it into the distribution of a next token in the language task, is a single component of the chain of paragraph [0029]. No further layer count or layer width of the reader is recorded beyond the implementation values of paragraph [0032].

> Künye: [0036] sonrasına, 1. paragraf · G7 (reader ×6; istem 1,6,9,10,14,15) · dayanak: [0029],[0032],[0034],[0036]; bileşen künyeleri OLC:174-176 → cloudlm/model.py:179,305,313; katman aralığı kaydedilmemiş (OLC:67).

### G7-P2

The two task outputs carry the following meanings. The probability distribution of a next token is a distribution over the identities of the vocabulary of the language task, the recorded vocabulary of 10000 units of paragraph [0060] being one example of the identity set over which it is defined; the distribution assigns to each vocabulary identity the probability that the identity follows the context seen so far. The class assignment score is a scalar on which the decision boundary between classes of the record is placed; the fraud score of the eight-field transaction record of paragraph [0054] is a recorded example. In the tabular task the explanation operation is built on the logit that carries this score, so that the reading recorded in paragraph [0036] and the explanation operate on one and the same output quantity of one and the same reader.

> Künye: [0036] sonrasına, 2. paragraf · G7 (probability distribution of a next · class assignment score; istem 1) · dayanak: [0036],[0054],[0060]; vocab künyesi OLC:156 → logs/run_r5_cc_stable.log:6 ([0060]'da yazılı).

### G7-P3

As a recorded example of the reader in deployment, the self-test of the package of paragraph [0050] reports, over 19963 rows, a largest absolute difference of 1.2814998626708984e-06 against the 32-bit floating point reference and a throughput of 26473.303851652894 transactions per second, and the INT8 reader latency was measured at 0.015912843984551728 ms per transaction at the 50th percentile over 2000 transactions. The first figures are recorded in paragraph [0051] and the latency in paragraph [0017]; they are gathered here once, under the heading of the reader, to state that the quantized reader which produces the task output also produces the quantities on which the explanation operations rest, within the recorded difference.

> Künye: [0036] sonrasına, 3. paragraf · G7 (reader — istem 9,14 INT8 bağlamı) · dayanak: [0017],[0050],[0051]; künyeler OLC:22,25 → fraud/results/edge_eval_full_scale.json:23,21; OLC:62,63,65 → fraud/results/deploy/manifest.json:27,28,30. Sayılar tarifnamede mevcut, TEK künyeyle bağlandı; tablo çoğaltılmadı.

## G9 — Cümle/pencere hiyerarşisi (P1 · 3 bulgu · istem 10)

### G9-P1

The hierarchy of paragraph [0037] operates as follows. The text of the task input is divided into sentences; this division is performed as an ordinary pre-processing step of the language body, and no particular segmentation algorithm is required. Each sentence is encoded as a sentence cloud state. A window covering a consecutive run of sentence cloud states is then advanced along the text by the fixed stride, and each position of the window produces one window cloud state. Because the stride is smaller than the window size, consecutive windows share sentences, so a sentence that lies near a change of subject contributes to the window cloud states of both neighbours; this overlap is what the fixed stride produces. The window-geometry encoder collects the geometric quantities between the windows of the window cloud states and emits the multi-sentence set representation, which in this variant is the reader input. The recorded window size and stride, 5 and 2, are those of paragraph [0037]; the number of sentences in a text and the number of windows thereby produced are not bounded by any recorded value, and the recorded split counts are those of paragraph [0060].

> Künye: [0037] sonrasına · G9 (sentence cloud state · fixed stride · multi-sentence set representation; istem 10) · dayanak: [0027],[0037],[0060]; pencere/adım künyesi OLC:159 → contextcloud/model.py:127-128; sınıf künyeleri OLC:160 → model.py:196,246; metin/pencere sayısı aralıkları kaydedilmemiş.

## G10 — Gradyansız eğitim + aday konum (P2 · 2 bulgu · istem 12)

### G10-P1

Gradient-based backpropagation denotes the update of parameters by evaluating the gradient of the task loss with respect to those parameters and moving the parameters along that gradient. That the point positions are updated without computing gradients means that no gradient of the task loss with respect to a point position is evaluated at any step. The update proceeds as a loop. For a selected point, a candidate position is generated in the neighbourhood of its current position. The task loss is measured with the candidate in place of the current position, on the same mini-batch that produced the base loss, so that candidate and base are compared under identical data conditions. If the loss measured with the candidate falls below the base loss by at least the threshold, the candidate is accepted as the new position of the point; otherwise the previous position is restored and the search interval on the side of the candidate is narrowed before the next candidate is generated. The numerical width of the neighbourhood, the value of the threshold and the interval parameters are not recorded.

> Künye: [0053] sonrasına, 1. paragraf · G10 (reader by gradient-based backpropagation döngü bağlamı; istem 12) · dayanak: [0015],[0019],[0053]; adım künyeleri OLC:185-187 → quann_cloud/src/qcloud_quann.py:212,220,223,226; eps/aralık kaydedilmemiş.

### G10-P2

A further working example of the third variant of paragraph [0065] is recorded on a language corpus. In a run of a training arrangement providing two point-update channels, the pair channel was selected and the recorded selection covered 86080 parameters; the held-out perplexity measured for that run was 5233.4086. A control run under a protocol recorded as identical at the level of the command arguments measured a held-out perplexity of 19770.644. The recorded boundary is stated with the figures: the comparison rests on a single seed, no paired repeat run exists, and repetition is contemplated; accordingly no improvement or reduction is asserted here. The example shows that the gradient-free update of the third variant was exercised with a selected pair channel at the recorded parameter selection.

> Künye: [0053] sonrasına, 2. paragraf · G10 (candidate point position / gradyan-sız eğitim çalışma örneği; istem 12) · dayanak: AÇILIM_PLANI G10-P2 künyeleri — RESEARCH:23 ve RESEARCH:107 → logs/run_r8_pair_freeze_only.log:11,12 (5233.4086; 86080) ve logs/run_r8_quann_pair.log:30 (19770.644); protokol eşliği training/shell/run_r8_quann_pair.sh:37,43. SINIR (tek seed, çift koşum yok, tekrar adayı) paragrafa AYNEN işlendi; "iyileştirir" hükmü yok.

## G11 — Sistem çerçevesi + dağıtım paketi (P2 · 5 bulgu · istem 1, 13, 14)

### G11-P1

The distribution store is a physical store of one kind: the three structures named above are held together as a single file or a single set of files, and they are read into memory in the order of use, the quantization table mapping a raw input row onto a model input identity and the resolution table then mapping that identity onto its human-readable label. The computer system of paragraph [0016] holds these structures in its memory, and its at least one processor executes the stored instructions, which configure the system to operate as the modules of the chain of paragraph [0029]: the context encoder module, the point projection module, the pair-geometry encoder, the aggregation module, the reader, the interpretation engine and the output interface are hardware blocks configured by that instruction set, not free-standing machines. The same set of weights then produces the task output and the three explanation operations, the interpretation engine being a second use of the representation and not a second system, as stated in paragraph [0030]. No hardware designation and no range of processor counts is recorded.

> Künye: [0050] ile [0051] arasına · G11 (computer system · at least one processor · instructions configures · distribution store holding · same trained model; istem 1,13,14) · dayanak: [0011],[0016],[0029],[0030],[0050],[0051]; eşleme künyesi OLC:66 → fraud/results/deploy/manifest.json:97-98. Paket boyutları [0050]'de mevcut — sayı çoğaltılmadı.

---

*Toplam: 14 paragraf (G1:2 · G2:1 · G3:1 · G4:1 · G5:1 · G6:1 · G7:3 · G9:1 · G10:2 · G11:1).
ADAY dosyada yeni numaralar: G1/G2 → [0032]-[0034]; G4 → [0037]; G3 → [0039]; G5 → [0041];
G7 → [0043]-[0045]; G9 → [0047]; G6 → [0057]; G11 → [0062]; G10 → [0066]-[0067]
(script çıktısı esastır; bu satır kontrol amaçlıdır).*
