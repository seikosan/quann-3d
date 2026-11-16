# Prior-Art Tarama Stratejisi ve IPC Sınıflandırma — Quann-3D Patentleri

> **id:** PRIOR_ART_TARAMA_v1 · **konu:** üç patent dosyası için prior-art genişletme, tarama sorguları, IPC/CPC önerisi, çakışma riski
> **durum:** strateji taslağı — resmî tarama ve IPC ataması vekilin işidir (`geometry_dual_use_core_v1.md:260`)
> **sahiplik:** paper · **güncellendi:** 2026-11-03
>
> **Kanıt disiplini:** Dosya atıfları `dosya:satır` biçimindedir. Dış literatürde arXiv kimliği
> arXiv API ile 2026-11-03'te doğrulanan 8 kaynak "✓arXiv" ile işaretlidir; diğer dış atıfların
> DOI/arXiv değerleri başvuru öncesi vekilce link-teyidinden geçirilmelidir. Doğrulanamayan
> kalemler `DOĞRULANMADI` etiketi taşır.

---

## 0. Mevcut Durum (envanter özeti)

| Dosya | Anılan prior-art | Atıf listesi | Boşluk cümlesi |
|---|---|---|---|
| Evrensel (`geometry_dual_use_core_v1.md`) | 9 kaynak (2.2.1–2.2.3, satır 60–69) | 9 kalemli (satır 250–258) | satır 71, 288 |
| Dil (`patent_application_v2.5.md`) | 7 kaynak (2.1–2.3, satır 40–51) | 7 kalemli (satır 282–288) | satır 55–60, 322 |
| Tabular (`patent_tabular_explain_v1.md`) | nitel anım, kaynak listesi YOK (2.2.1–2.2.2, satır 41–77) | YOK | satır 81–89 |

**Bulgu:** Tabular dosyasının atıf listesi hiç yok — GBM/XGBoost/isolation-forest/aotoekod (satır
41–45), FT-Transformer/TabTransformer (satır 66) ve 0. sıra optimizasyon (satır 75–77) ismen
anılıp künyesiz. Başvuru öncesi §1'deki tablo tabular dosyasına eklenmelidir.

---

## 1. Genişletilmiş Prior-Art Listesi

### 1.1 Dosyalarda zaten anılan kaynaklar (değişiklik yok)

| # | Kaynak | Anıldığı yer |
|---|---|---|
| A1 | Vaswani ve ark. 2017, *Attention Is All You Need*, NeurIPS 30 | `geometry_dual_use_core_v1.md:250`, `patent_application_v2.5.md:282` |
| A2 | Qi ve ark. 2017, *PointNet*, CVPR 77–80 | `geometry_dual_use_core_v1.md:251`, `patent_application_v2.5.md:283` |
| A3 | Zaheer ve ark. 2017, *Deep Sets*, NeurIPS 30 | `geometry_dual_use_core_v1.md:252`, `patent_application_v2.5.md:284` |
| A4 | Lee ve ark. 2019, *Set Transformer*, ICML 3714–3723 | `geometry_dual_use_core_v1.md:253`, `patent_application_v2.5.md:285` |
| A5 | Mikolov ve ark. 2013, *Word2Vec*, arXiv:1301.3781 | `geometry_dual_use_core_v1.md:254`, `patent_application_v2.5.md:286` |
| A6 | Pennington ve ark. 2014, *GloVe*, EMNLP 1532–1543 | `geometry_dual_use_core_v1.md:255`, `patent_application_v2.5.md:287` |
| A7 | Chung & Abbott 2021, *Geometry of a neural population*, Nature 592(7853) 117–122 | `geometry_dual_use_core_v1.md:256`, `patent_application_v2.5.md:288` |
| A8 | Hu, Niu & Varma 2026, arXiv:2607.04525 | `geometry_dual_use_core_v1.md:257` (yalnız evrensel; dış teyit yapılmadı) |
| A9 | Xu 2026, arXiv:2605.17084 | `geometry_dual_use_core_v1.md:258` (yalnız evrensel; dış teyit yapılmadı) |

### 1.2 Eklenecek kaynaklar — YENİ

#### (a) Tabular derin öğrenme (tabular dosyasının künyesiz animları için)

| # | Kaynak | Kimlik | Risk nedeni |
|---|---|---|---|
| B1 | Huang ve ark. 2020, *TabTransformer: Tabular Data Modeling Using Contextual Embeddings*, KDD | arXiv:2012.06678 ✓arXiv | Kimlik/bağlam embedding'li tabular temsil — (özellik, bucket) projeksiyonuyla komşu |
| B2 | Gorishniy ve ark. 2021, *Revisiting Deep Learning Models for Tabular Data* (FT-Transformer), NeurIPS | arXiv:2106.11959 ✓arXiv | Token benzeri özellik işleme; tabular İddia 1'in (b)'sine karşı en yakın alan |
| B3 | Popov, Morozov & Babenko 2020, *Neural Oblivious Decision Ensembles (NODE)*, ICLR | arXiv:1909.06312 ✓arXiv | GBM+nöral hibrit — tabular 2.2.1'in "dönüşümlü sinir ağları" animını künyeler |
| B4 | Arik & Pfister 2021, *TabNet: Attentive Interpretable Tabular Learning*, AAAI | arXiv:1908.07442 (teyit edilmeli) | "Interpretable tabular" başlığı — yerel anlaşılırlık iddialarıyla alan çakışması |
| B5 | Gorishniy ve ark. 2022, *On Embeddings for Numerical Features in Tabular Deep Learning*, NeurIPS | arXiv:2203.05556 ✓arXiv | **Sayısal özelliğin bucket (piecewise) embedding'i** — kimlik-bucket token üretiminin doğrudan öncülü; YÜKSEK öncelikli tarama |

#### (b) Nokta-bulut çift-geometri (İddia 4 / D+4 vektör riski)

| # | Kaynak | Kimlik | Risk nedeni |
|---|---|---|---|
| B6 | Qi, Yi, Su & Guibas 2017, *PointNet++*, NeurIPS 30 | arXiv:1706.02413 (teyit edilmeli) | Hiyerarşik küme agregasyonu — permütasyon-değişmezlik animının künyesi |
| B7 | Wang ve ark. 2019, *Dynamic Graph CNN for Learning on Point Clouds (DGCNN/EdgeConv)*, ACM TOG 38(5) | DOI 10.1145/3326362 (teyit edilmeli) | **EdgeConv çift özelliği [pᵢ−pⱼ] temelli** — İddia 4'ün çift-vektörüne en yakın üçüncü-taraf teknik; YÜKSEK öncelikli tarama |
| B8 | Zhao ve ark. 2021, *Point Transformer*, ICCV | arXiv:2110.08184 (teyit edilmeli) | Nokta-çifti ilişki kodlama (relational position encoding) |

#### (c) Açıklanabilir ML (XAI) — native ablation/karşı-faktürel animları için

| # | Kaynak | Kimlik | Risk nedeni |
|---|---|---|---|
| B9 | Lundberg & Lee 2017, *SHAP*, NeurIPS 30 | arXiv:1705.07874 ✓arXiv | Additivity aksiyomu ayrımının karşıtı; künye zaten metinlerde var, listeye eklenmeli |
| B10 | Ribeiro, Singh & Guestrin 2016, *LIME*, KDD | arXiv:1602.04938 ✓arXiv | Dil dosyası 3. sorun maddesinde ismen anılıyor (`patent_application_v2.5.md:70`), künye yok |
| B11 | Sundararajan, Taly & Yan 2017, *Integrated Gradients*, ICML | arXiv:1703.01365 (teyit edilmeli) | Aksiyomatik atıf alternatifi — boşluk cümlesinin güvencesi için kapsanmalı |
| B12 | Shrikumar, Greenside & Kundaje 2017, *DeepLIFT*, ICML | arXiv:1605.01713 (teyit edilmeli) | DeepSHAP'ın temeli; kompozisyon-tabanlı atıf |
| B13 | Koh & Liang 2017, *Influence Functions*, ICML | arXiv:1703.04730 (teyit edilmeli) | "Leave-one-out" mantığının eğitim-verisi düzeyi öncülü — leave-one-point-out ayrımını netleştirmek için anılmalı |
| B14 | Zeiler & Fergus 2014, *Visualizing and Understanding CNNs* (occlusion), ECCV | arXiv:1311.2901 (teyit edilmeli) | Maske/çıkar-ile-açıkla — leave-one-point-out'un genel kalıbı |
| B15 | Fong & Vedaldi 2017, *Meaningful Perturbation*, ICCV | arXiv:1704.03296 (teyit edilmeli) | Karşı-faktürel maskelemenin görsel öncülü — tabular 280 modülüyle komşu |
| B16 | Jain & Wallace 2019, *Attention is not Explanation*, AAAI | arXiv:1902.10186 (teyit edilmeli) | Dil dosyası 1(a) ayrım cümlesini destekleyen alan bilgisi |
| B17 | Jethani ve ark. 2022, *FastSHAP*, ICLR | arXiv:2107.07436 ✓arXiv | Öğrenilmiş atıf üreteci — "SetSHAP" aramasının gerçek karşıtı (bkz. not) |

> **"SetSHAP" notu:** Görev tanımında anılan "SetSHAP" adında kanonik bir makaleye
> rastlanmadı — bu ad **arama terimi** olarak kullanılmalı, künye olarak listelenmemelidir
> (`DOĞRULANMADI`). Gerçek karşıtlar: B17 (FastSHAP) + B12 (DeepSHAP) + küme-uyarlamaları
> (B9'un set girdi varyantları).

#### (d) 0. sıra (gradyansız) optimizasyon — İddia 9 / tabular İddia 11-12

| # | Kaynak | Kimlik | Risk nedeni |
|---|---|---|---|
| B18 | Spall 1992, *SPSA*, IEEE Trans. Autom. Control 37(3) 332–341 | DOI 10.1109/9.119632 (teyit edilmeli) | İki-nokta stokastik gradyansız güncelleme — 0. sıra daraltmanın genel kalıbı |
| B19 | Nesterov & Spokoiny 2017, *Random Gradient-Free Minimization*, Found. Comput. Math. 17 | DOI 10.1007/s10208-015-9296-2 (teyit edilmeli) | Rastgele-yönlü sıfırıncı mertebe — QUANN daraltma animını künyeler |
| B20 | Salimans ve ark. 2017, *Evolution Strategies*, | arXiv:1703.03864 (teyit edilmeli) | Popülasyon-tabanlı gradyansız eğitim |
| B21 | Jones, Schonlau & Welch 1998, *Efficient Global Optimization (Bayesian)*, J. Global Optim. 13 | DOI 10.1023/A:1008307431021 (teyit edilmeli) | Değer-değerlendirme temelli siyah-kutu optimizasyon öncülü |

> **QUANN:** Kendi motorumuzdur (`quann_cloud/src/`, yayınlanmamış) — dış prior-art DEĞİL,
> iç teknik çerçevedir; ancak "QUANN" adının marka/arXiv/patent isim-çakışması §3.4'te taranmalıdır.
> Başvuru dosyaları QUANN'ı zaten "bitişik (bilinen) icat" diye referanslıyor
> (`patent_tabular_explain_v1.md:70-74`).

#### (e) NLP / kelime vektörü tamamlayıcıları

| # | Kaynak | Kimlik | Risk nedeni |
|---|---|---|---|
| B22 | Bojanowski ve ark. 2017, *fastText (subword)*, TACL 5 135–146 | DOI 10.1162/tacl_a_00051 (teyit edilmeli) | Agglütinatif dil / alt-sözcük animına künye (dil embodiment 2.7.3) |
| B23 | Devlin ve ark. 2019, *BERT*, NAACL | arXiv:1810.04805 (teyit edilmeli) | Alan temeli — karşılaştırma çerçevesi (üstünlük iddiası yok, bkz. `geometry_dual_use_core_v1.md:109`) |

#### (f) Fraud / AML / anomali tespiti

| # | Kaynak | Kimlik | Risk nedeni |
|---|---|---|---|
| B24 | Chen & Guestrin 2016, *XGBoost*, KDD | arXiv:1603.02754 (teyit edilmeli) | Tabular 2.2.1'in baskın paradigma animının künyesi (0.8047 PR-AUC baseline'ı, `geometry_dual_use_core_v1.md:109`) |
| B25 | Ke ve ark. 2017, *LightGBM*, NeurIPS 30 | (venue; teyit edilmeli) | Aynı nedenle |
| B26 | Liu, Ting & Zhou 2008, *Isolation Forest*, ICDM | DOI 10.1109/ICDM.2008.17 (teyit edilmeli) | "Ayrık orman" animının künyesi (`patent_tabular_explain_v1.md:43`) |
| B27 | Weber ve ark. 2019, *Anti-Money Laundering in Bitcoin (Elliptic)*, KDD | arXiv:1908.02591 ✓arXiv | AML embodiment'inin veri kümesi — İddia 12'deki ölçümlerin kaynağı (`geometry_dual_use_core_v1.md:161`) |

#### (g) Popülasyon geometrisi (evrensel 2.2.2 genişletmesi)

| # | Kaynak | Kimlik | Risk nedeni |
|---|---|---|---|
| B28 | Bernardi ve ark. 2020, *The Geometry of Abstraction in the Hippocampus and Prefrontal Cortex*, Cell 183(4) | DOI 10.1016/j.cell.2020.09.031 (teyit edilmeli) | Soyutlama-uzayları nokta-bulutu benzeri geometri — A7'nin yanına kapsama eklenmeli |

#### (h) Veri kümesi atıfları (iddialarda geçen ölçümlerin kaynağı)

| # | Veri kümesi | Anıldığı yer |
|---|---|---|
| B29 | CWRU Bearing Dataset (Loparo, Case Western Reserve Univ.) | `geometry_dual_use_core_v1.md:165` (İddia 13) |
| B30 | NASA C-MAPSS turbofan (Saxena & Goebel 2008) | `geometry_dual_use_core_v1.md:165` (İddia 13) |
| B31 | IBM TabFormer kredi-kartı işlem veri kümesi | `patent_tabular_explain_v1.md:432` ("IBM kredi kartı 24.4M") — künyesi eklenmeli |

---

## 2. Tarama Sorguları

> Sözdizimi: Espacenet — alan-önekli CQL (`ti`, `ab`, `cpc`); PATENTSCOPE — `EN_TI:`, `EN_AB:`,
> `CPC:`; Google Patents — serbest ifade + CPC filtresi. Her sorgu EN (ve gerekirse TR) dilinde
> çalıştırılmalı; A-X (yenilik-kıran) belge bulunan tüm sonuçlar §5 risk tablosuna işlenmelidir.

### 2.1 Dil patent ortak çekirdek sorguları

| # | Sorgu | DB notu |
|---|---|---|
| S1 | `ti any "point cloud" AND (ti any "language model" OR ti any "token" OR ti any "text")` | Espacenet; PATENTSCOPE karşılığı `EN_TI:("point cloud" AND ("language model" OR token))` |
| S2 | `ab any "token" AND ab any "low-dimensional" AND ab any "embedding" AND ab any "point"` | Espacenet |
| S3 | `ab any "permutation" AND (ab any "next token" OR ab any "language modeling" OR ab any "permutation-invariant")` | Espacenet; Google Patents: `q="permutation invariant" "next token"` |
| S4 | `CPC:("G06N3/04") AND (EN_AB:("point cloud") AND EN_AB:("language" OR "token"))` | PATENTSCOPE CPC+kelime kesişimi |
| S5 | `q=("nokta bulutu" AND ("dil modeli" OR "token"))` | Google Patents TR dilinde — TR ulusal aşaması için |

### 2.2 Evrensel patent sorguları

| # | Sorgu | DB notu |
|---|---|---|
| S6 | `(ti any "point cloud" OR ab any "point cloud") AND (ab any "dual" OR ab any "pairwise") AND (ab any "explanation" OR ab any "explainab" OR ab any "interpretab")` | Espacenet — *explainab* kökü interpretable/explainability'yı kapsar |
| S7 | `ab any "leave-one-out" AND ab any (ablation OR "occlusion" OR masking) AND ab any (prediction OR logit OR score)` | Espacenet; Google: `q=("leave one out" ablation explanation neural)` |
| S8 | `q=("native explainability" OR "intrinsic explainability" OR "inherently interpretable" OR "native explanation")` | Google Patents — "native/intrinsic XAI" ifadeli başvurular |
| S9 | `(ab any "prototype" OR ab any "region") AND ab any "drift" AND ab any ("point cloud" OR embedding OR representation)` | Espacenet — bölge-prototip + drift incelemesi |
| S10 | `(ab any "cosine" OR ab any "invariant") AND ab any ("point pair" OR "pairwise features" OR "edge features") AND ab any ("point cloud" OR "set")` | Espacenet — D+4 çift-vektör kalıbı; B7 (EdgeConv) buradan yakalanmalı |
| S11 | `CPC:("G06N3/045" OR "G06N3/04") AND CPC:("G06V10/77" OR "G06T19/00") AND EN_AB:("explanation")` | PATENTSCOPE CPC-çifti + kelime |
| S12 | `(ab any "zeroth-order" OR ab any "derivative-free" OR ab any "gradient-free") AND ab any ("training" OR "optimization") AND ab any (embedding OR representation OR "point")` | Espacenet — hibrit eğitim rejimi (İddia 9) |

### 2.3 Tabular patent sorguları

| # | Sorgu | DB notu |
|---|---|---|
| S13 | `(ab any "fraud" OR ab any "anomaly") AND (ab any "identity" OR ab any "bucket" OR ab any "binning") AND (ab any "explanation" OR ab any "interpretab")` | Espacenet — kimlik-bucket + açıklama |
| S14 | `q=("counterfactual" AND (fraud OR "tabular") AND (explanation OR attribution OR audit))` | Google Patents |
| S15 | `ab any "plain-text" AND ab any (explanation OR justification OR rationale) AND ab any (fraud OR transaction OR record)` | Espacenet — düz-metin açıklama iddiası |
| S16 | `(ab any "edge" OR ab any "on-device" OR ab any "self-contained") AND (ab any "ONNX" OR ab any "quantiz" OR ab any "INT8") AND (ab any "manifest" OR ab any "sha256" OR ab any "provenance")` | Espacenet — self-contained paket (290) |
| S17 | `CPC:("G06Q20/38" OR "G06Q20/40" OR "G06Q40/02") AND EN_AB:("explainable" OR "interpretable" OR "counterfactual")` | PATENTSCOPE — ödeme/banka + XAI |
| S18 | `ab any ("feature" AND "value") AND ab any "token" AND ab any ("tabular" OR "record") AND ab any ("point cloud" OR "set encoding")` | Espacenet — (özellik, değer-bucket) token kalıbı |

### 2.4 Kurgusal olmayan literatür (NPL) taraması

- arXiv tam-metin: `"point cloud" "language model"`, `"leave-one-point-out"`,
  `"identity bucket" embedding tabular`, `"dual geometry" representation`,
  `"zeroth-order" "point cloud" training`, `FastSHAP set`.
- Google Scholar / Semantic Scholar / DBLP: B1–B31'nin "Cited by" ağları (2. derece tarama).
- Öncelik sırası: **B5, B7, B14, B15** (İddia 4 + native ablation animlarına en yakın) →
  bunların atıf ağları ilk turda taranmalıdır.

---

## 3. IPC/CPC Sınıflandırma Önerileri

> Evrensel dosyanın mevcut önerisi `G06F 18/00; G06N 3/04; G06N 5/02; G06V 10/77`
> (`geometry_dual_use_core_v1.md:18`); dil dosyalarında `G06F 18/00; G06N 3/04; G06N 5/02`
> (`TURKPATENT_basvuru_v2.3.md:17`). Aşağıda genişletme + alt-sınıf gerekçesi; **CPC alt
> sınıf tamamlarının resmî CPC şemasından teyidi vekilindir** (`DOĞRULANMADI` işaretliler).

### 3.1 Ortak çekirdek (üç dosyada da)

| Kod | Neden uygun |
|---|---|
| **G06N 3/04** (NN mimarisi) | Projeksiyon MLP'si + çift MLP + küme agregasyonu bir nöral mimaridir (`geometry_dual_use_core_v1.md:127-133`) |
| **G06N 3/08** (NN öğrenme yöntemi) | Hibrit rejim: 0. sıra nokta daraltma + backprop okuyucu (`geometry_dual_use_core_v1.md:212`, `patent_tabular_explain_v1.md:201-206`) |
| **G06N 20/00** (makine öğrenmesi) | Genel ML kapsamı — okuyucu/skor çıktısı her dosyada ML sistemidir |
| **G06F 18/00** (özet örüntü tanıma) | Dosyalarda zaten önerili (satır 18); CPC'de alt gruplar (ör. 18/21 tasarım/doğrulama, 18/24 sınıflama) vekilce seçilmeli `DOĞRULANMADI` |

### 3.2 Evrensel patent (ek)

| Kod | Neden uygun |
|---|---|
| **G06V 10/77** | Dosyada zaten var (satır 18) — nokta-bulut uzamsal özellik işleme; Şekil 2/3'ün görsel-geometrik doğası |
| **G05B 23/02** | Bakım embodiment'i: makine/duyarga izleme + arıza tahmini (İddia 13, `geometry_dual_use_core_v1.md:163-165,220`) |
| **G06N 5/02** | Dosyada var (satır 18): tam-sözlük decoder + prototip isimlendirme "bilgi temsili" tarafına oturur. **Not:** EPO pratiğinde XAI çoğunlukla G06N 20/00 + G06F 18 altında sınıflanır; 5/02'nin kalması tartışmalıdır — vekil kararı `DOĞRULANMADI` |

### 3.3 Dil patent (ek)

| Kod | Neden uygun |
|---|---|
| **G06F 40/30** (NLP — anlamsal çözümleme) | Girdi token/alt-sözcük → bir-sonraki-token üretimi (`patent_application_v2.5.md:30,264`); TR ulusal başvuru için NLP sınıfı eklenmeli |
| **G06N 3/04 + G06N 3/08** | 3.1'de ortak; dil dosyasında ayrıca BPE önişleme tanımı var (`patent_application_v2.5.md:161`) |

### 3.4 Tabular patent (ek)

| Kod | Neden uygun |
|---|---|
| **G06Q 20/38 / G06Q 20/40** | Ödeme işlemleri güvenlik/dolandırıcılık bağlamı (fraud embodiment; `patent_tabular_explain_v1.md:368-370`) — alt-grup seçimi vekilce `DOĞRULANMADI` |
| **G06Q 40/02** | Bankacılık / AML tipolojisi (evrensel İddia 12'nin tabular uzantısı) |
| **G06F 18/24** (sınıflama/anomali) | Aykırı-değer dedektörü özü (`patent_tabular_explain_v1.md:29-35`) |
| **G06N 20/20** (CPC, ensamble) | İddiaların karşılaştırıldığı GBM öncül tekniklerine karşılık gelen inceleme-kolayı kod (baseline sınıfı) |

---

## 4. Çakışma Riski Analizi

### 4.1 İç çakışmalar (kendi dosyalarımız arası)

| Risk | Seviye | Kanıt | Savunma stratejisi |
|---|---|---|---|
| **Dil ↔ Evrensel:** İddia 1'in (a)–(d) paragrafları neredeyse aynı (`patent_application_v2.5.md:220-232` ↔ `geometry_dual_use_core_v1.md:173-185`) | YÜKSEK | Aynı çekirdek iki dosyada bağımsız iddia ediliyor | Evrensel bağımsız iddiaları **(d) native açıklama + (e) paketleyici** birleşimine bağlanmış durumda (İddia 1(d)(e)); dil dosyası TR ulusal aşamasında (a)–(d) çekirdeğini *token-embodiment'e daraltmalı*. TR ulusal evresinde PCT yayınlanmadan önce "özdeş icat" itirazı gelirse: dil iddialarını İddia 10 embodiment + İddia 11-12 (nokta-sayısı/agregasyon varyantı, `patent_application_v2.5.md:266-268`) kadar daraltmak |
| **Tabular ↔ Evrensel:** Evrensel İddia 11 fraud embodiment'ini tabular İddia 1 ile kopyalıyor (`geometry_dual_use_core_v1.md:216` ↔ `patent_tabular_explain_v1.md:244-276`) | YÜKSEK | Tabular dosya kendini evrensel çekirdeğin tabular uygulaması diye tanımlıyor (`patent_tabular_explain_v1.md:13`) | Tabular dosyada zaten önerilen seçenek: bağımsız iddiayı salt **(d) açıklama + (e) karşı-faktürel denetim + (f) self-contained paket** üçlüsüne çekmek, (b)(c) çekirdeği bağımlıya almak (`patent_tabular_explain_v1.md:429-431`). Alternatif: evrenselin fraud embodiment'i bölünmüş (divisional) dosyaya taşınır |
| **Öncelik zinciri:** Evrensel TR'dan öncelik ALAMAZ; bağımsız T2 tarihi taşır | YÜKSEK | `geometry_dual_use_core_v1.md:19,272-274` | T1 (TR dil, v2.3) ile T2 (PCT) arasında kalan yayınlara karşı evrensel dosya T2 günlü prior-art'a maruz — PCT öncesi ve 18 ay yayın öncesi **mutlak gizlilik** korunmalı (`geometry_dual_use_core_v1.md:23`) |
| **v2.3 → v2.5 yeni madde:** nokta-sayısı aralığı (İddia 11) ve agregasyon varyantı (İddia 12) T1'de açıklanmamış olabilir | ORTA | `patent_application_v2.5.md:337` ("yeni özellikler için T2 geçerli"), 357-364 | Yeni özelliklerin v2.3'te zaten açıklandığının kanıtı vekilce satır-bazlı eşlenmeli; eşlenemeyen özellik T2 dayanaklı yazılmalı |
| **Ölçüm tutarlılığı:** fraud paket boyutu evrenselde "543 KB ONNX" (`geometry_dual_use_core_v1.md:157`), tabular'da "529 KB FP32 → 138 KB INT8" (`patent_tabular_explain_v1.md:211`) | DÜŞÜK | Farklı artefaktlar (ONNX dosyası ↔ FP32 model) olabilir | Vekil, ölçüm protokolü tek satırlık açıklama eklemeli; sayılar kaynak log'larla eşlenmeli (sebep uydurulmadan) |

### 4.2 Üçüncü-taraf çakışma riskleri (tarama bulgularına bağlı)

| Kaynak (§1) | Risk | Yıkıcı senaryo | Savunma |
|---|---|---|---|
| **B7 DGCNN/EdgeConv** | YÜKSEK | EdgeConv çift özelliği `[pᵢ−pⱼ]` temelli → İddia 4'ün çift-vektörü barizlik/ifşa riski | Ayrım: **cos θ + norm bileşenleri + dönüşüm-eşdeğişkenlik karması** (`patent_application_v2.5.md:102-105`) VE aynı bulutun native açıklama yüzeyi olması (`geometry_dual_use_core_v1.md:280`). Sadece çift-özellik iddiası tek başına asla korunamaz |
| **B5 Gorishniy 2022 bucket embedding** | YÜKSEK | Sayısal özelliğin bucket embedding'i → tabular (a) kimlik-bucket projeksiyonu | Ayrım: bucket embedding'in *noktaya projekte edilip çift-geometriyle kümelenmesi* + decoder + paket (tabular 2.2.3 dört bileşen birleşimi, `patent_tabular_explain_v1.md:81-89`) |
| **B14 Zeiler-Fergus occlusion / B15 meaningful perturbation** | ORTA | Maske-çıkar-ile-açıkla kalıbı → leave-one-point-out + karşı-faktürel modüller | Ayrım: ablation, kararın **üretildiği aynı** bulut üzerinde native yapılıyor (dual-use çekirdek, `geometry_dual_use_core_v1.md:88`); post-hoc modelden değil |
| **B9 SHAP** | ORTA | Özellik-atfı iddialarının eski hâli | Metinlerde ayrım zaten kurulmuş (additivity aksiyomu, `geometry_dual_use_core_v1.md:62`, `patent_tabular_explain_v1.md:49-52`) — korunmalı |
| **B6 PointNet++ / A4 Set Transformer** | ORTA | Simetrik küme agregasyonu tek başına bilinir | Ayrım: öğrenilmiş düşük-D projeksiyon + çift-geometri + native açıklama birleşimi (`geometry_dual_use_core_v1.md:67`); metin ayrımı mevcut, tarama ile pekiştirilmeli |
| **A8/A9 Hu-2026/Xu-2026** | ORTA | Geometri analiz edebiyatı iddia kapsamını daraltabilir | Ayrım cümleleri mevcut (`geometry_dual_use_core_v1.md:69`): onlar post-hoc analiz, bu buluş hesaplama+açıklama birleşik mimari |
| **B18–B21 0. sıra optimizasyon** | ORTA | 0. sıra güncelleme kalıbı bilinen yöntem | Ayrım: iddia 0. sıra yöntem değil, **hibrit rejim** (0. sıra nokta + backprop okuyucu; `patent_tabular_explain_v1.md:201-206,152-154`) |
| **B4 TabNet "interpretable tabular"** | DÜŞÜK | Başlık düzeyi kesişim | TabNet'te açıklama seçim-maskeleri düzeyinde; nokta-bulut/native yüzey yok |
| **"QUANN" isim çakışması** | DÜŞÜK-ORTA | Aynı/özdeş ticari adlı başka patent/marka | TÜRKPATENT marka + Espacenet'te `ti any QUANN` taraması; gerekirse başvurularda yöntem adı kullanılmadan tanım yazılır |

### 4.3 Özet hüküm

- İç dosyalar arası **iki YÜKSEK** risk var (Dil↔Evrensel çekirdek kopyası; Tabular↔Evrensel
  fraud embodiment kopyası). Her ikisinin de dosyalarda yazılı savunma hattı var; vekil
  iddia-çekme (claim withdrawal) kararı başvuru öncesi verilmelidir.
- Dış tarafta **iki YÜKSEK** tarama önceliği var (B7 DGCNN, B5 bucket-embedding) — bu ikisinin
  atıf ağları taranmadan yenilik cümleleri (2.2.3 / 2.4 boşluk paragrafları) kesinleştirilmemelidir.
- Küresel yokluk iddiası hiçbir dosyada yok (bilinçli olarak, `geometry_dual_use_core_v1.md:71,288`;
  `patent_application_v2.5.md:322`) — bu tutum tarama sırasında da korunmalı; boşluk cümlesi
  "incelenen kaynaklarda" kalırsa yenilik savunması çökmez.

---

## 5. Vekil Eylem Listesi (bu belgeden çıkan)

- [ ] §1.2 tablosunu tabular dosyasına ATIF LİSTESİ olarak ekle (tabular'da liste yok — §0 bulgu)
- [ ] B7, B5, B14, B15 atıf ağları: 1. tur kapsamlı tarama (yenilik-kıran adaylar)
- [ ] §2 sorgularını Espacenet + PATENTSCOPE + Google Patents'ta koş; A-X belgeleri §4.2 tablosuna işle
- [ ] CPC alt-grup seçimlerini resmî CPC şemasından teyit et (3.1–3.4'te `DOĞRULANMADI` işaretliler)
- [ ] Dil↔Evrensel ve Tabular↔Evrensel için iddia-çekme/divisional kararını yaz
- [ ] "QUANN" isim/marka taraması (Espacenet `ti any QUANN` + TÜRKPATENT marka)
- [ ] Ölçüm protokolü tek satırları: 543 KB ↔ 529/138 KB tutarlılık notu (§4.1)
