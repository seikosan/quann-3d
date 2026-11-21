#!/usr/bin/env python3
"""Deterministik ekleme + [00NN] yeniden numaralandırma (PCT v2 aday tarifname).

Girdi : docs/patent/pct_v2/DESCRIPTION_EN.md  (SALT-OKUR — yazılmaz)
Çıktı : docs/patent/pct_v2/DESCRIPTION_EN_ADAY.md
Kural : Yeni paragraflar "<<NEW>>" sentinel'i ile hedef orijinal paragrafın
        sonrasına eklenir; tek geçişte satır-başı etiketler (sentinel dahil)
        0001'den başlayarak artan sayılır; metin-içi [NNNN] atıfları
        eski->yeni haritasıyla topluca kaydırılır (örn. [0029]->[0065] atfı).
Başlıklar (## ...) ve mevcut <!-- kaynak --> yorumları aynen korunur.
"""
import re
from pathlib import Path

ROOT = Path("/Users/serkankose/Documents/quann-3d")
SRC = ROOT / "docs/patent/pct_v2/DESCRIPTION_EN.md"
DST = ROOT / "docs/patent/pct_v2/DESCRIPTION_EN_ADAY.md"

P = {
"G1P1": "The task input is defined by the structure it presents to the context encoder. An ordered token sequence is composed of sub-word units selected from a fixed vocabulary, each position of the sequence carrying the identity of one such unit; the vocabulary of 10000 units recorded in paragraph [0060] is one recorded example of the language body. A tabular record is a set of measured field values of one entity, each field carrying its own measured value and its own field identity; the eight field values of the credit-card transaction record of paragraph [0054] are one recorded example, a transaction amount or a sensor reading being a measured field value of the kind referred to in paragraph [0002]. No further form of task input is required: the ordered sequence and the tabular record are the two forms from which the point cloud of paragraph [0010] is constructed.",
"G1P2": "Three terms carry a precise meaning in this description. The input order of the points is the numbering of the positions at which the input units are placed: position one, position two and so on along the sequence, or position one through position F over the fields of a record. That the symmetric set aggregation is independent of this order means that any permutation of the numbering yields the same set representation, so the aggregation output is a function of the set of points and not of their arrangement. A point cloud is a collection of N points in which no position within the collection carries information to the aggregation; in the tabular body the points are the fields of the record, and in the language body the points are the context positions of the sequence. A point in a D-dimensional space is the position vector of D coordinates assigned to one input unit by the point projection; the recorded values of D are those of paragraph [0032].",
"G2P1": "A context-encoded representation is the vector that the context encoder attributes to one input unit of the task input: for an ordered token sequence, one representation per token position; for a tabular record, one representation per field identity. The representation is context-dependent in that the value attributed to a unit is produced jointly with the representations of the other units of the same input, and not from the unit alone. The point projection then maps each such representation onto a point of the D-dimensional space, the pair input width of 3D + 2 defined in paragraph [0033] being consistent with this projection. The only encoder alternative recorded for the language body is the three-tier encoder of paragraph [0037]. As a recorded example of scale, the language chain of paragraph [0059] holds 20999648 parameters; this figure describes the chain as a whole and is not attributed to the context encoder alone.",
"G3P1": "The result of the max-pooling is the vector obtained by selecting, in each point-feature dimension, the largest value among the points of the cloud; its width is fixed by the point-feature width and is therefore independent of the number N of points. The result of the average-pooling is the vector obtained by taking, in each pair-feature dimension, the mean over the valid pairs alone, consistent with the normalization by the number of valid pairs recorded in paragraph [0035]. Both are operations over a set: permuting the points leaves each pooled result unchanged, which makes the order-independence of paragraph [0030] visible on the pooling results themselves. The concatenation of the two results is supplied to the reader as the reader input of paragraph [0010]. In the global-statistics variant, the cloud centre is the mean position of the points of the cloud, in the sense of paragraph [0038]; no numerical range for it is recorded.",
"G4P1": "The Euclidean norm of the difference vector is computed as the square root of the sum of the squares of the coordinate differences of the two point positions; it is the same quantity as the pair distance named above, stated here as an arithmetic operation rather than as a name. Of the five components, the difference vector and this distance express where the two points lie relative to one another, the cosine similarity expresses the angular relation between their position vectors, and the two norm components express where each point lies with respect to the cloud centre. The cost of carrying these pair features scales as recorded in paragraph [0039], which states the measured multiply-accumulate figures.",
"G5P1": "The context length of the task input is the number of input units the task arrangement admits: positions in an ordered sequence, fields in a tabular record. The number N of points equals that number when every position is filled, and a context-unfilled position, as masked in paragraph [0035], is a position within the context length that carries no input unit. The recorded examples tie this bound to concrete figures already described: a sequence length of 10 in the dimension comparison of paragraph [0058], a sequence length of 8 in the smaller-scale training of paragraph [0066], and K = 13 active points with 221 model input identities and 214 narrowed point positions in paragraph [0032]. No numerical mask value or threshold is recorded.",
"G6P1": "The mark of an emerging region is carried in the audit record as a triple: the region index, the human-readable region label of paragraph [0044], and the pair of point shares measured for that region in the reference distribution and in the comparison distribution; consistent with paragraph [0045], the label accompanies the index and does not replace it. The lower and upper thresholds are parameters of the task arrangement; no numerical threshold values are recorded. The mark is produced by the comparison step alone: it states that the share of the region lay below the lower threshold in the reference distribution and above the upper threshold in the comparison distribution, and, as recorded in paragraph [0067], it is claimed as a criterion so defined and not as an event observed in the measured drift.",
"G7P1": "The reader input is read by the reader in the following steps. The single set representation produced by the concatenation layer of paragraph [0034] enters the reader as its input state. In the language task the forward pass through the parametric layers of the reader is defined together with the attention mask, so that masked positions take no part in the pass, and the cloud state on which the explanation operations act is taken from the cloud encoder. The reader then maps that input state onto the task output. The term reader denotes one and the same structural element in both bodies: the module that turns the set representation into the class assignment score in the tabular task, and the module that turns it into the distribution of a next token in the language task, is a single component of the chain of paragraph [0029]. No further layer count or layer width of the reader is recorded beyond the implementation values of paragraph [0032].",
"G7P2": "The two task outputs carry the following meanings. The probability distribution of a next token is a distribution over the identities of the vocabulary of the language task, the recorded vocabulary of 10000 units of paragraph [0060] being one example of the identity set over which it is defined; the distribution assigns to each vocabulary identity the probability that the identity follows the context seen so far. The class assignment score is a scalar on which the decision boundary between classes of the record is placed; the fraud score of the eight-field transaction record of paragraph [0054] is a recorded example. In the tabular task the explanation operation is built on the logit that carries this score, so that the reading recorded in paragraph [0036] and the explanation operate on one and the same output quantity of one and the same reader.",
"G7P3": "As a recorded example of the reader in deployment, the self-test of the package of paragraph [0050] reports, over 19963 rows, a largest absolute difference of 1.2814998626708984e-06 against the 32-bit floating point reference and a throughput of 26473.303851652894 transactions per second, and the INT8 reader latency was measured at 0.015912843984551728 ms per transaction at the 50th percentile over 2000 transactions. The first figures are recorded in paragraph [0051] and the latency in paragraph [0017]; they are gathered here once, under the heading of the reader, to state that the quantized reader which produces the task output also produces the quantities on which the explanation operations rest, within the recorded difference.",
"G9P1": "The hierarchy of paragraph [0037] operates as follows. The text of the task input is divided into sentences; this division is performed as an ordinary pre-processing step of the language body, and no particular segmentation algorithm is required. Each sentence is encoded as a sentence cloud state. A window covering a consecutive run of sentence cloud states is then advanced along the text by the fixed stride, and each position of the window produces one window cloud state. Because the stride is smaller than the window size, consecutive windows share sentences, so a sentence that lies near a change of subject contributes to the window cloud states of both neighbours; this overlap is what the fixed stride produces. The window-geometry encoder collects the geometric quantities between the windows of the window cloud states and emits the multi-sentence set representation, which in this variant is the reader input. The recorded window size and stride, 5 and 2, are those of paragraph [0037]; the number of sentences in a text and the number of windows thereby produced are not bounded by any recorded value, and the recorded split counts are those of paragraph [0060].",
"G10P1": "Gradient-based backpropagation denotes the update of parameters by evaluating the gradient of the task loss with respect to those parameters and moving the parameters along that gradient. That the point positions are updated without computing gradients means that no gradient of the task loss with respect to a point position is evaluated at any step. The update proceeds as a loop. For a selected point, a candidate position is generated in the neighbourhood of its current position. The task loss is measured with the candidate in place of the current position, on the same mini-batch that produced the base loss, so that candidate and base are compared under identical data conditions. If the loss measured with the candidate falls below the base loss by at least the threshold, the candidate is accepted as the new position of the point; otherwise the previous position is restored and the search interval on the side of the candidate is narrowed before the next candidate is generated. The numerical width of the neighbourhood, the value of the threshold and the interval parameters are not recorded.",
"G10P2": "A further working example of the third variant of paragraph [0065] is recorded on a language corpus. In a run of a training arrangement providing two point-update channels, the pair channel was selected and the recorded selection covered 86080 parameters; the held-out perplexity measured for that run was 5233.4086. A control run under a protocol recorded as identical at the level of the command arguments measured a held-out perplexity of 19770.644. The recorded boundary is stated with the figures: the comparison rests on a single seed, no paired repeat run exists, and repetition is contemplated; accordingly no improvement or reduction is asserted here. The example shows that the gradient-free update of the third variant was exercised with a selected pair channel at the recorded parameter selection.",
"G11P1": "The distribution store is a physical store of one kind: the three structures named above are held together as a single file or a single set of files, and they are read into memory in the order of use, the quantization table mapping a raw input row onto a model input identity and the resolution table then mapping that identity onto its human-readable label. The computer system of paragraph [0016] holds these structures in its memory, and its at least one processor executes the stored instructions, which configure the system to operate as the modules of the chain of paragraph [0029]: the context encoder module, the point projection module, the pair-geometry encoder, the aggregation module, the reader, the interpretation engine and the output interface are hardware blocks configured by that instruction set, not free-standing machines. The same set of weights then produces the task output and the three explanation operations, the interpretation engine being a second use of the representation and not a second system, as stated in paragraph [0030]. No hardware designation and no range of processor counts is recorded.",
}

PROV = {
"G1P1": "ACILIM_METINLERI_EN.md G1-P1 (kaynaklar: OLC:156 vocab; tarifname [0010] [0054] [0060] [0002])",
"G1P2": "ACILIM_METINLERI_EN.md G1-P2 (tarifname [0010] [0030] [0032]; aralık uydurulmadı)",
"G2P1": "ACILIM_METINLERI_EN.md G2-P1 (OLC:153 ölçek; tarifname [0033] [0037] [0059])",
"G3P1": "ACILIM_METINLERI_EN.md G3-P1 (OLC:174 yalnız varlık künyesi; tarifname [0030] [0035] [0038])",
"G4P1": "ACILIM_METINLERI_EN.md G4-P1 (tarifname [0021] [0033] [0039]; alternatif norm kayıtlı değil)",
"G5P1": "ACILIM_METINLERI_EN.md G5-P1 (yalnız tarifname içi mevcut örnekler: [0032] [0035] [0058] [0066])",
"G6P1": "ACILIM_METINLERI_EN.md G6-P1 (OLC:201 sınır notu; tarifname [0044] [0045] [0046] [0067])",
"G7P1": "ACILIM_METINLERI_EN.md G7-P1 (OLC:174-176 bileşen künyeleri; tarifname [0029] [0034] [0036])",
"G7P2": "ACILIM_METINLERI_EN.md G7-P2 (OLC:156 vocab; tarifname [0036] [0054] [0060])",
"G7P3": "ACILIM_METINLERI_EN.md G7-P3 (OLC:22,25 edge_eval_full_scale.json:23,21; OLC:62,63,65 manifest.json:27,28,30; tarifname [0017] [0051])",
"G9P1": "ACILIM_METINLERI_EN.md G9-P1 (OLC:159,160 contextcloud künyeleri; tarifname [0037] [0060])",
"G10P1": "ACILIM_METINLERI_EN.md G10-P1 (OLC:185-187 qcloud_quann.py adımları; tarifname [0019] [0053])",
"G10P2": "ACILIM_METINLERI_EN.md G10-P2 (RESEARCH:23,107 → run_r8_pair_freeze_only.log:11,12 = 5233.4086, 86080; run_r8_quann_pair.log:30 = 19770.644; protokol run_r8_quann_pair.sh:37,43; tek seed/tekrar adayı sınırı aynen yazıldı)",
"G11P1": "ACILIM_METINLERI_EN.md G11-P1 (OLC:66 manifest.json:97-98; tarifname [0011] [0016] [0029] [0030] [0050])",
}

INSERT_AFTER = {
    31: ["G1P1", "G1P2", "G2P1"],
    33: ["G4P1"],
    34: ["G3P1"],
    35: ["G5P1"],
    36: ["G7P1", "G7P2", "G7P3"],
    37: ["G9P1"],
    46: ["G6P1"],
    50: ["G11P1"],
    53: ["G10P1", "G10P2"],
}

assert SRC.read_text(encoding="utf-8").count("\n") > 0  # salt-okur giriş

lines = SRC.read_text(encoding="utf-8").splitlines()
out = []
inserted = []
for line in lines:
    out.append(line)
    m = re.match(r"^\[(\d{4})\] ", line)
    if m:
        n = int(m.group(1))
        for name in INSERT_AFTER.get(n, []):
            out.append("")
            out.append("<<NEW>> " + P[name])
            out.append("")
            out.append(f"<!-- kaynak: {PROV[name]}; AÇILIM_PLANI.md §2 kartı -->")
            inserted.append((n, name))

# --- yeniden numaralandırma: satır-başı etiketler 0001'den artan; eski->yeni haritası ---
old_to_new, new_to_old, counter = {}, {}, 0
tagged = []
for line in out:
    if line.startswith("<<NEW>>"):
        counter += 1
        new_to_old[counter] = None  # yeni paragraf
        tagged.append(f"[{counter:04d}]{line[len('<<NEW>>'):]}")
    else:
        m = re.match(r"^\[(\d{4})\] (.*)$", line)
        if m:
            counter += 1
            old_to_new[int(m.group(1))] = counter
            new_to_old[counter] = int(m.group(1))
            tagged.append(f"[{counter:04d}] {m.group(2)}")
        else:
            tagged.append(line)

# --- metin-içi [NNNN] atıflarını eski->yeni haritasıyla kaydır (paragraf gövdeleri) ---
ref = re.compile(r"\[(\d{4})\]")
def shift(body: str) -> str:
    def sub(mm):
        old = int(mm.group(1))
        return f"[{old_to_new[old]:04d}]" if old in old_to_new else mm.group(0)
    return ref.sub(sub, body)

final = []
for line in tagged:
    m = re.match(r"^\[(\d{4})\] (.*)$", line)
    if m:
        final.append(f"[{m.group(1)}] {shift(m.group(2))}")
    else:
        final.append(line)

# ADAY künye yorumu: ilk satırdan sonra tek HTML yorum (başlıklar aynen kalır)
final.insert(1, "<!-- ADAY kopyası — DESCRIPTION_EN.md (sha change yok) + AÇILIM_PLANI.md "
               "uygulaması: 14 paragraf eklendi, [00NN] topluca yeniden numaralandırıldı. "
               "TASLAK, vekil onayı gerekli; orijinal dosya değişmedi. -->")

DST.write_text("\n".join(final) + "\n", encoding="utf-8")
print(f"DST: {DST}")
print(f"toplam paragraf: {counter}")
print(f"eklenen blok sayısı: {len(inserted)}")
print("ekleme noktaları (eski->[yeni ilk paragraf]):",
      {f"[{k:04d}]": f"[{k + 1:04d}]" for k, _ in INSERT_AFTER.items()})
print("kaydırma kontrolleri: [0029]->[%04d]  [0065]->[%04d]  [0031]->[%04d]  [0050]->[%04d]  [0070]->[%04d]"
      % (old_to_new[29], old_to_new[65], old_to_new[31], old_to_new[50], old_to_new[70]))
