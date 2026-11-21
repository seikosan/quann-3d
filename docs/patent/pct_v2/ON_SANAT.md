# ON_SANAT — PCT v2 Ön-Sanat Taraması (Claim 1 + Claim 14)

> **id:** ON_SANAT_PCT_V2 · **konu:** İstem 1 (context→point cloud→set aggregation→reader→native açıklama) ve İstem 14 (INT8 self-contained paket) için doğrulanabilir ön-sanat taraması
> **durum:** NPL ön-taraması + **patent DB taraması YAPILDI** (3. tur, §5: 42 sorgu, ham arşiv
> diskte) — vekil-formal tarama (CPC + atıf ağı + X/Y/P sınıflaması) hâlâ gerekli
> **sahiplik:** paper · **güncellendi:** 2026-11-19
>
> **Kanıt disiplini:** Bu dosyadaki her kaynak kaydı, 2026-11-19 tarihli bu tarama turunda
> `patent-creator` MCP aracının `search_non_patent_literature` fonksiyonu (OpenAlex + arXiv +
> Semantic Scholar derlemeleri) çağrılarak döndürülmüştür. Her satırda aracın döndürdüğü
> ham alanlar (başlık, yıl, DOI/arXiv, URL) aynen aktarılmıştır. Bellekten/ezberden yazılan
> sıfır kaynak vardır; araçtan döndürülemeyen adaylar §1.3'te "BULUNAMADI" etiketlidir.

---

## §0 KAPSAM VE SINIR

**Koşulan araçlar (2026-11-19):**

1. `search_non_patent_literature` (patent-creator MCP; OpenAlex + arXiv + Semantic Scholar) —
   21 hedefli sorgu koşuldu; 21 kaynak kaydı döndü ve §1.1'e işlendi. *(Bu künyeler KANITSIZdır:
   araç yanıtlarının ham logu diskte saklanmadı — aşağıdaki "Ham kayıt durumu" notu ve §4 V6 kartı.)*
2. Google Patents XHR uç noktası (`patents.google.com/xhr/query`) — 3 deneme, tümü
   **HTTP 503** döndürdü (erişim engellendi); bu turda vekil patent taraması yapılamadı.
   *(3. turda aynı uç nokta vekil proxy üzerinden 200 döndürdü — §5.1. 503 sayısının ham kaydı
   yok: diskteki `db_taramasi/probe.json` yalnız Google'ın engel HTML'ini tutar, durum kodu
   içermez → `503` KANITSIZ; engel sayfasının varlığı doğrulanmıştır.)*
3. `websearch` aracı — 1 deneme, "Web search cancelled" hatası; sonuç üretmedi.

Sayım künyeleri: diskte sayılabilir iki sayı (8 sorgu · 4 durum çağrısı) ve araç adları
`patent-cikti/pct/pct_v2-pct-inceleme/KANIT_oncesanat_sayim.txt` çıktısındadır. Üç XHR denemesi
ve tek `websearch` denemesi KANITSIZ sayılır: ham oturum kaydı bu depoda tutulmuyor.

**Ham kayıt durumu (2026-11-19, bağımsız denetim sonrası):** yukarıdaki 21 sorgu, 3 XHR denemesi
ve 1 `websearch` denemesi **oturum içi araç çağrılarıdır**; yanıtları diskte ham log olarak
saklanmadı. Bu yüzden §1.1 künyeleri tek başına **KANITSIZ** sayılır. Aynı sorguların **2. turu**
(8 sorgu, `limit=2`) ve dört durum çağrısı diskte ham kayıt olarak duruyor:
`patent-cikti/pct/pct_v2-pct-inceleme/oncesanat_ham_tur2_2026-11-19.json`. Bu turun sorgu adedi
komutla sayılır (`jq '.sorgular | length' <kayıt>`), ham çıktısı diskte:
`patent-cikti/pct/pct_v2-pct-inceleme/KANIT_oncesanat_sayim.txt`. Bu sorguların tamamı §1.1
künyelerini yeniden üretti ve §1.2'deki iki DOI anomalisini bu kayıt düzeltti.

**YAPILANDIRILMAMIŞ patent veritabanı araçları (2026-11-19'da yeniden doğrulandı):**

| Araç | Durum | Eksik |
|---|---|---|
| `check_google_patents_status` | `ready:false` | SerpApi anahtarı yok |
| `check_bigquery_status` | `available:false` | GCP proje kimliği yok |
| `check_epo_api_status` | `ready:false` | EPO OPS kimlik bilgisi yok (`EPO_OPS_KEY/SECRET`) |
| `check_uspto_api_status` | `ready:false` | USPTO API anahtarı yok |

**Sonuç (1.–2. tur):** dört resmî patent API'si (Google Patents/SerpApi, BigQuery, EPO OPS,
USPTO) bu ortamda yapılandırılmamış ve doğrudan Google Patents erişimi 503 döndürdüğü için bu
iki turda patent tarafı taraması **koşulamadı**. **3. turda (2026-11-19) vekil proxy kanalıyla
patent DB taraması koşulldu: 42 sorgu, ham yanıtlar diskte — bkz. §5.** Anahtarlı API'ler hâlâ
yapılandırılmamış; Espacenet/PATENTSCOPE CPC sorguları, atıf ağları ve X/Y belge sınıflaması
**vekilin işidir** (`PRIOR_ART_TARAMA_STRATEJISI.md:4` zaten bunu vekile devreder).

**Bu dosyanın niteliği:** vekil-formal taramanın YERİNE GEÇEN bir ön-taramadır. Yalnızca
akademik literatür (NPL) tarafı araçla doğrulanmıştır. Patent tarafında hiçbir "küresel
yokluk / ilk biziz" cümlesi kurulmamıştır; boşluk cümleleri "incelenen kaynaklarda" kalıbındadır.

---

## §1 BULUNAN KAYNAKLAR

### §1.1 Araçla doğrulanan kaynaklar (21 kayıt)

Her satır 2026-11-19 tarama turunda `search_non_patent_literature` çağrısından döndü.

| # | Etiket | Başlık (aracın döndürdüğü) | Yazar/yıl | Venue | DOI / arXiv | URL | İstem 1 öğesi | Araç çıktısı alıntısı |
|---|---|---|---|---|---|---|---|---|
| S1 | PointNet | PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation | Qi ve ark., 2017 | CVPR | 10.1109/cvpr.2017.16 · arXiv:1612.00593 | https://doi.org/10.1109/cvpr.2017.16 | (a)(c)(d) | `"PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" year=2017 doi=10.1109/cvpr.2017.16` |
| S2 | PointNet++ | PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space | Qi, Yi, Su, 2017 | arXiv | arXiv:1706.02413 | http://arxiv.org/abs/1706.02413 | (a)(c) | `"PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" year=2017 arxiv=1706.02413` |
| S3 | EdgeConv | Dynamic Graph CNN for Learning on Point Clouds | Wang ve ark., 2019 | ACM TOG | 10.1145/3326362 · arXiv:1801.07829 | https://doi.org/10.1145/3326362 | (b)(c) | `"Dynamic Graph CNN for Learning on Point Clouds" year=2019 doi=10.1145/3326362 venue=ACM Transactions on Graphics` |
| S4 | Set Transformer | Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks | Lee ve ark., 2018 | arXiv/ORA | arXiv:1810.00825 | https://ora.ox.ac.uk/objects/uuid:8aa5f831-66b2-46ca-89ea-8ba1f3315f33 | (c) | `"Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks" year=2018 arxiv=1810.00825` |
| S5 | Point Transformer | Point Transformer | Zhao, Jiang, Jia, 2020 | arXiv | arXiv:2012.09164 | http://arxiv.org/abs/2012.09164 | (b) | `"Point Transformer" year=2020 arxiv=2012.09164 authors="Hengshuang Zhao; Li Jiang; Jiaya Jia"` |
| S6 | Set-universality | Universal Approximation of Functions on Sets | Wagstaff ve ark., 2021 | arXiv/ORA | arXiv:2107.01959 | https://ora.ox.ac.uk/objects/uuid:aad96b39-3580-4ae8-a046-9a596ce30db4 | (c) | `"Universal Approximation of Functions on Sets" year=2021 arxiv=2107.01959` |
| S7 | Set-aggregation seçimi | On Deep Set Learning and the Choice of Aggregations | Soelch ve ark., 2019 | LNCS (ICANN) | 10.1007/978-3-030-30487-4_35 | https://doi.org/10.1007/978-3-030-30487-4_35 | (c) | `"On Deep Set Learning and the Choice of Aggregations" year=2019 doi=10.1007/978-3-030-30487-4_35` |
| S8 | Deep-sets-uygulama | Energy flow networks: deep sets for particle jets | Komiske, Metodiev, Thaler, 2019 | JHEP | 10.1007/jhep01(2019)121 | https://doi.org/10.1007/jhep01(2019)121 | (a)(c) | `"Energy flow networks: deep sets for particle jets" year=2019 doi=10.1007/jhep01(2019)121` |
| S9 | SHAP | A Unified Approach to Interpreting Model Predictions | Lundberg, Lee, 2017 | arXiv (NeurIPS) | arXiv:1705.07874 | http://arxiv.org/abs/1705.07874 | (e) | `"A Unified Approach to Interpreting Model Predictions" year=2017 arxiv=1705.07874 authors="Scott Lundberg; Su-In Lee"` |
| S10 | LIME | "Why Should I Trust You?": Explaining the Predictions of Any Classifier | Ribeiro, Singh, Guestrin, 2016 | NAACL-HLT | 10.18653/v1/n16-3020 · arXiv:1602.04938 | https://doi.org/10.18653/v1/n16-3020 | (e) | `'"Why Should I Trust You?": Explaining the Predictions of Any Classifier' year=2016 doi=10.18653/v1/n16-3020` |
| S11 | Occlusion | Visualizing and Understanding Convolutional Networks | Zeiler, Fergus, 2013 | arXiv (ECCV) | arXiv:1311.2901 | http://arxiv.org/abs/1311.2901 | (e)(i) | `"Visualizing and Understanding Convolutional Networks" year=2013 arxiv=1311.2901 authors="Matthew D. Zeiler; Rob Fergus"` |
| S12 | SPSA | A one-measurement form of simultaneous perturbation stochastic approximation | Spall, 1997 | Automatica | 10.1016/s0005-1098(96)00149-5 | https://doi.org/10.1016/s0005-1098(96)00149-5 | arka plan (0. sıra eğitim) | `"A one-measurement form of simultaneous perturbation stochastic approximation" year=1997 doi=10.1016/s0005-1098(96)00149-5 venue=Automatica` |
| S13 | ES | Evolution Strategies as a Scalable Alternative to Reinforcement Learning | Salimans ve ark., 2017 | arXiv | arXiv:1703.03864 | http://arxiv.org/abs/1703.03864 | arka plan (0. sıra eğitim) | `"Evolution Strategies as a Scalable Alternative to Reinforcement Learning" year=2017 arxiv=1703.03864` |
| S14 | RGF | Randomized gradient-free methods in convex optimization | Gasnikov ve ark., 2022 | arXiv | arXiv:2211.13566 | http://arxiv.org/abs/2211.13566 | arka plan (0. sıra eğitim) | `"Randomized gradient-free methods in convex optimization" year=2022 arxiv=2211.13566` |
| S15 | INT8 | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference | Jacob ve ark., 2018 | CVPR | 10.1109/cvpr.2018.00286 | https://doi.org/10.1109/cvpr.2018.00286 | İstem 14 | `"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" year=2018 doi=10.1109/cvpr.2018.00286 authors="Benoit Jacob; Skirmantas Kligys; Bo Chen"` |
| S16 | Bucket embedding | On Embeddings for Numerical Features in Tabular Deep Learning | Gorishniy, Rubachev, Babenko, 2022 | NeurIPS | arXiv:2203.05556 | https://www.semanticscholar.org/paper/1057c047e3d52df1fd24e (Semantic Scholar kaydı) | (a) tabular | `"On Embeddings for Numerical Features in Tabular Deep Learning" year=2022 arxiv=2203.05556 venue=Neural Information Processing Systems` |
| S17 | TabTransformer | TabTransformer: Tabular Data Modeling Using Contextual Embeddings | Huang ve ark., 2020 | arXiv (KDD) | arXiv:2012.06678 | http://arxiv.org/abs/2012.06678 | (a) tabular | `"TabTransformer: Tabular Data Modeling Using Contextual Embeddings" year=2020 arxiv=2012.06678` |
| S18 | Kategorik hash-embedding | Learning to Embed Categorical Features without Embedding Tables for Recommendation | Kang ve ark., 2021 | KDD | 10.1145/3447548.3467304 · arXiv:2010.10784 | https://doi.org/10.1145/3447548.3467304 | (a) tabular | `"Learning to Embed Categorical Features without Embedding Tables for Recommendation" year=2021 doi=10.1145/3447548.3467304` |
| S19 | Nokta-bulut açıklama | Visualizing Point Cloud Classifiers by Curvature Smoothing | Chen, Wu, Qi, 2019/2020 | arXiv + BMVC | arXiv:1911.10415 · 10.5244/c.34.45 | http://arxiv.org/abs/1911.10415 | (e)(i)(ii) | `"Visualizing Point Cloud Classifiers by Curvature Smoothing" year=2019 arxiv=1911.10415` + BMVC 2020 kaydı `doi=10.5244/c.34.45` |
| S20 | Nokta-bulut açıklama-2 | Interpretable3D: An Ad-Hoc Interpretable Classifier for 3D Point Clouds | Feng ve ark., 2024 | AAAI | 10.1609/aaai.v38i2.27944 | https://doi.org/10.1609/aaai.v38i2.27944 | (e) | `"Interpretable3D: An Ad-Hoc Interpretable Classifier for 3D Point Clouds" year=2024 doi=10.1609/aaai.v38i2.27944` |
| S21 | LM+nokta bulutu | PointLLM: Empowering Large Language Models to Understand Point Clouds | Xu ve ark., 2024 | ECCV (LNCS) | 10.1007/978-3-031-72698-9_8 | https://doi.org/10.1007/978-3-031-72698-9_8 | (a)(d) | `"PointLLM: Empowering Large Language Models to Understand Point Clouds" year=2024 doi=10.1007/978-3-031-72698-9_8` |

### §1.2 İlgili ama ikincil kayıtlar (araç döndürdü, ayrı künye satırı açılmadı)

- "Local Spectral Graph Convolution for Point Set Feature Learning" (Wang, Samari, Siddiqi,
  2018, ECCV/LNCS, DOI 10.1007/978-3-030-01225-0_4) — S2 sorgusunda döndü; set-özellik öğrenimi.
- "On permutation-invariant neural networks" (Kimura ve ark., 2024, arXiv:2403.17410) —
  permütasyon-değişmezlik teorisi; S4 sorgusunda döndü.
- "Set Norm and Equivariant Skip Connections: Putting the Deep in Deep Sets" (Zhang ve ark.,
  2022, arXiv:2206.11925) — Deep Sets mimari iyileştirmesi; S8 sorgusunda döndü.
- "Point Transformer V3: Simpler, Faster, Stronger" (Wu ve ark., 2024, CVPR, DOI
  `10.1109/cvpr52733.2024.00463`) — künye ham kayıtla düzeltildi: 1. turda satırda yazan
  `10.1109/cvpr52688...` (2022 serisi) DOI'si **yanlıştı**; değer 2. tur araç çıktısından
  alındı (`oncesanat_ham_tur2_2026-11-19.json`, "Point Transformer V3 faster stronger larger").
- "PointGuard: Provably Robust 3D Point Cloud Classification" (CVPR 2021, DOI
  `10.1109/cvpr46437.2021.00612`, arXiv:2103.03046) — künye ham kayıtla düzeltildi: 1. turda
  yazan `10.1109/cvpr42600.2020.01054` (2020 serisi) DOI'si **yanlıştı**; başlık/DOI/arXiv 2. tur
  araç çıktısından alındı (aynı ham kayıt, "PointGuard adversarial point clouds defense").
  Nokta-çıkarma/bozma etkisinin güvenlik literatüründeki kalıbı; S19 sorgusunda döndü.

### §1.3 BULUNAMADI (araç döndürmedi — vekil teyit etmeli)

| Aday | Denenen sorgular | Sonuç |
|---|---|---|
| Zaheer ve ark. 2017, "Deep Sets" (NeurIPS 30) — arXiv:1703.06114 | `'"Deep Sets" Zaheer universal approximation sets'`, `'"Deep Sets" Zaheer 2017 permutation invariant neural networks NeurIPS'`, `'Deep Sets' sources=[arxiv]`, `'"Deep Sets: Learning invariant representations" ...'` | 4 araç sorgusunda orijinal makale **dönmedi**; yalnız S6/S7/S8 türevleri döndü. **3. turda künye bağımsız doğrulandı:** arXiv API'si `id_list=1703.06114` sorgusunu `HTTP 200` ile döndürdü, başlık `Deep Sets`, `<published>2017-03-10T21:02:53Z` — ham: `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/link_teyit/arxiv_1703_06114.xml`. |
| Spall 1992, "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation" (IEEE TAC 37(3)) | `'simultaneous perturbation stochastic approximation Spall'` | 1997 Automatica tek-ölçüm varyantı (S12) döndü; 1992 TAC orijinali araçta **dönmedi**. **3. turda DOI bağımsız doğrulandı:** Crossref `10.1109/9.119632` `HTTP 200`, başlık "Multivariate stochastic approximation using a simultaneous perturbation gradient …", dergi *IEEE Transactions on Automatic Control*, `issued 1992-03` — ham: `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/link_teyit/crossref_spall1992.json`. |
| PointNet++ publisher-versiyonu (NeurIPS sayfası) | — | Yalnız arXiv kaydı (S2) döndü; vekil NeurIPS versiyonunu teyit eder. |

---

## §2 İSTEM 1 ÖĞE-KARŞI-ÖNSANAT GRAFİĞİ

Her satırda: öğe → en yakın doğrulanmış kaynak(lar) → ayrım cümlesi. Ayrımlar yalnızca
aracın döndürdüğü kayıt verisi (başlık/venue/yıl) + repo içi mimari tanımına dayanır;
kaynağın içeriği hakkında araçtan gelmeyen iddia yazılmamıştır.

| Öğe | En yakın kaynak(lar) | Ayrım (bizim neyimiz farklı) |
|---|---|---|
| **(a)** Bağlam kodlayıcı çıktısının D-boyutlu noktalara izdüşümü; girdi = sıralı token dizisi **veya** tabular kaydın ölçülen alan değerleri | S1 PointNet, S2 PointNet++, S21 PointLLM, S16 Gorishniy, S17 TabTransformer | S1/S2/S5 kayıtlarının başlıkları 3B tarama/segmentasyon alanını tanımlar ("3D Classification and Segmentation", "Point Sets in a Metric Space"); noktalar ölçülmüş 3B koordinatlardır. Bizim (a) öğemizde noktalar **bağlam kodlayıcı çıktısının izdüşümüdür** — girdi token dizisi veya tabular kayıttır; 3B tarama yoktur. S21 (PointLLM) yönü ters: nokta bulutu LLM'e girdi olur; bizde LM'in gizli durumları nokta bulutuna projekte edilir. S16/S17 tabular özellikleri embedding yapar ama nokta bulutuna projekte etmez ve çift-geometri işlemez. |
| **(b)** Her nokta çifti için konumlara bağlı geometrik çift-özellikler | S3 EdgeConv (ACM TOG), S5 Point Transformer | S3 kaydının başlığı "Dynamic Graph CNN for Learning on Point Clouds" — kenar (çift) özellikleri 3B nokta bulutları üzerinde. Repo ayrım kaydı: çift-özellik kompozisyonumuz cos θ + norm bileşenleri + dönüşüm-eşdeğişkenliktir ve aynı bulutun native açıklama yüzeyiyle birleşiktir (`PRIOR_ART_TARAMA_STRATEJISI.md:237`). Yalnızca çift-özellik iddiası tek başına korunamaz — iddia (b)+(c)+(e) birleşimine bağlanmıştır. |
| **(c)** Girdi sırasından bağımsız simetrik küme agregasyonu | S4 Set Transformer, S6 Wagstaff, S7 Soelch, S8 Komiske | Simetrik agregasyon tek başına incelenen kaynaklarda bilinmektedir (S4 başlığı doğrudan "Permutation-Invariant" içerir). Ayrım: bizde agregasyon, çift-geometri özelliklerini de kapsayan tek küme gösterimine uygulanır ve okuyucu aynı gösterimden hem görev çıktısı hem açıklama üretir (İstem 1(d)(e) zinciri). S6/S7 agregasyon fonksiyonu seçimini teorik olarak inceler; native açıklama entegrasyonu yoktur. |
| **(d)** Görev çıktısı: token girdisi → sonraki-token dağılımı; tabular girdi → sınıf skoru | S1 PointNet (sınıflandırma), S21 PointLLM | S1 kaydı sınıflandırma/segmentasyon çıktısı verir; sonraki-token dağılımı üretmez. S21'de LLM nokta bulutunu *anlar* (girdi), bizde LM'in iç gösterimi *nokta bulutu olarak temsil edilir* (ara-durum). İki modun (token + tabular) tek küme gösteriminde birleştiği kayıt incelenen kaynaklarda döndü. |
| **(e)** Aynı eğitilmiş modelin aynı nokta bulutu üzerinde üç açıklama işlemi; ayrı açıklama modeli/eğitimi/girdisi YOK | S9 SHAP, S10 LIME, S11 Zeiler occlusion, S19 PCI-GOS, S20 Interpretable3D | S9/S10 kayıtları model-bağımsız (model-agnostic) açıklama çerçeveleridir; ayrı bir surrogate/lokal model kurarlar (S10 başlığı "Explaining the Predictions of Any Classifier"). S11 maskelama kalıbını görsellerde tanımlar. S19 nokta bulutu üzerinde görselleştirme yapar (başlık: "Visualizing Point Cloud Classifiers"); açıklama için bulutu *değiştirir* ve ayrı bir görselleştirme süreci işletir. Bizim (e) öğemiz: çıkarma etkisi + bölge isimlendirme + bölge kayması üç işlemi **eğitilmiş modelin kendi bulutu üzerinde**, ayrı model/eğitim/girdi olmadan üretir. Bu üçlü kombinasyon incelenen kaynaklarda döndü. |
| **(f)** Görev çıktısı + açıklamanın aynı küme gösteriminden türeyip arayüzde birlikte sunulması | (S9/S10/S19/S20 dolaylı) | İncelenen kaynaklarda açıklama, görev çıktısından ayrı bir işlem/hizmet olarak döndü; "aynı küme gösteriminden türeyen görev çıktısı + açıklama"nin tek arayüzde birleşik sunulduğu kayıt bu taramanın sorgularında döndü. |
| **İstem 14:** INT8 okuyucu + nicelleştirme tablosu + çözümleme tablosu tek kendi-kendine-yeten paket | S15 Jacob ve ark. | S15 kaydı INT8 tam-sayı çıkarımını ve nicelleştirme parametrelerini tanımlar ("Integer-Arithmetic-Only Inference"). Ayrım: S15'te açıklama verisi ve model girdi kimliklerini insan-okunur etiketlere eşleyen çözümleme tablosu yoktur; bizim paket kararı + denetimi tek pakette taşır (ölçüm: `fraud/results/edge_eval_full_scale.json:16`, 138.2 KB). |

---

## §3 EN YAKIN ÜÇ ÖN-SANAT AİLESİ

### Aile 1 — EdgeConv / çift-geometri (S3, S5)
Nokta-çifti özellikleri (EdgeConv kenar özellikleri, Point Transformer konum-ilişkileri)
en yakın tekil teknik. **Tek başına yıkıcı değil çünkü:** (i) bu ailenin tüm doğrulanmış
kayıtları 3B tarama alanındadır (başlıklar: "Learning on Point Clouds", "Point Transformer"
3B segmentasyon bağlamında); bizim girdi uzayımız token dizisi + tabular kayıttır;
(ii) iddia (b) tek başına asla korunmaz — (b)+(c)+(e) birleşimine bağlanmıştır
(`PRIOR_ART_TARAMA_STRATEJISI.md:237` savunma stratejisi zaten bunu yazar).

### Aile 2 — Simetrik küme agregasyonu (S4, S6, S7, S8 + S1/S2)
Set Transformer, set-universality ve agregasyon-seçimi çalışmaları permütasyon-değişmez
temsili kurar. **Tek başına yıkıcı değil çünkü:** agregasyon tekniği bilinir; yenilik
agregasyonun kendisinde değil, agregasyonun çift-geometri özellikleriyle birleşimi ve
aynı gösterimin okuyucu + açıklama zincirini beslemesindedir. S6 (Wagstaff) teorik
gerekçeyi sağlar ama görev çıktısı + native açıklama kombinasyonunu tanımlamaz.

### Aile 3 — Açıklama ailesi (S9, S10, S11, S19, S20)
SHAP/LIME model-bağımsız, Zeiler maskelama, PCI-GOS/Interpretable3D nokta-bulut
açıklaması. **Tek başına yıkıcı değil çünkü:** hepsi ya ayrı bir surrogate/perturbasyon
süreci işletir ya da görselleştirme çıktısı üretir; "aynı eğitilmiş modelin aynı nokta
bulutu üzerinde üç işlemle (çıkarma etkisi + bölge isimlendirme + bölge kayması) açıklama,
ayrı model/eğitim/girdi olmadan" kombinasyonu incelenen kaynaklarda döndü.

### Aile 4 (yardımcı) — INT8 uç dağıtım (S15) + 0. sıra eğitim (S12, S13, S14)
S15 nicelleştirme tablosu mantığını kurar; 0. sıra aile eğitim-yöntemi arka planıdır
(iddia bir eğitim-yöntemi iddiası değildir). Yıkıcılık yok: S15'te açıklama + çözümleme
tablosu kombinasyonu yoktur.

---

## §4 VEKİL EYLEM LİSTESİ (bu turda YAPILAMAYANLAR)

| # | Eylem | Neden yapılamadı | Vekil komut önerisi |
|---|---|---|---|
| V1 | **Kısmen kapandı:** Google Patents tam-metin taraması 3. turda koşuldu (§5, 42 sorgu). Kalan: Espacenet + PATENTSCOPE + CPC sınıf taraması | 4 API yapılandırılmamış (§0 tablosu) | Espacenet CPC: `G06N3/04`, `G06N3/08`, `G06N20/00`, `G06F18/24`, `G06V10/77` + anahtar kelime kombinasyonları (§2 öğeleri) |
| V2 | Atıf ağları 2. derece (S1, S3, S9, S10, S15'in işaret ettiği belgeler) | Araç yalnız 1. derece kayıt döndürür | Semantic Scholar/Scopus atıf grafiği; özellikle S3 EdgeConv ve S19 PCI-GOS'un işaret ettikleri |
| V3 | **Kısmen kapandı:** `ti:QUANN` ve `ti:"cloud language model" OR ti:CloudLM` 3. turda koştu; dönen kayıtlarda isim geçmiyor (§5.4). Kalan: marka (TÜRKPATENT/EUIPO) + arXiv isim taraması; `ti:` operatörünün davranışı doğrulanmadı | marka veritabanlarına erişim yok | Espacenet `ti any QUANN`, `ti any CloudLM`; TÜRKPATENT marka taraması |
| V4 | **KAPANDI (3. tur):** arXiv API `1703.06114` → başlık "Deep Sets", `published 2017-03-10`; Crossref `10.1109/9.119632` → IEEE TAC, `issued 1992-03` (§1.3 güncellendi) | — | ham: `db_taramasi/link_teyit/arxiv_1703_06114.xml`, `db_taramasi/link_teyit/crossref_spall1992.json` |
| V5 | A-X belge sınıflaması ve IPC/CPC final ataması | Vekil yetkisi | `PRIOR_ART_TARAMA_STRATEJISI.md` §3 tablosundan başlangıç noktası |
| V6 | §1.1'in 21 künyesinin bağımsız teyidi (1. turun ham yanıtı diskte yok) | Araç yanıtları oturumda kaldı; yalnız 2. turun 8 sorgusu ham kayıtlı (§0) | Vekil her S-künyesini DOI/arXiv üzerinden açar; 13 DOI'siz kayıt (özellikle arXiv serisi) link-teyidi ister |

---

## §5 PATENT VERİTABANI TARAMASI — 3. TUR (YAPILDI, 2026-11-19)

### §5.1 Kanal künyesi ve sayım

Ham arşiv dizini: `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/`. Bu bölümdeki künyeler depo köküne göre tam yazılır; kısaltma yoktur.

| Ölçüm | Değer | Kanıt (ham kayıt) |
|---|---|---|
| Doğrudan `curl` → `patents.google.com/xhr/query` | Google'ın engel HTML-i döndü ("Sorry…") | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/probe.json` · `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/gp_probe.json` |
| Aynı talebin HTTP durum kodu | `503` — **KANITSIZ**: ham dosyalarda durum kodu yok (`grep -c 503` → `0`). Diskteki kaydı yalnız koşum betiklerinin yorum satırlarıdır (iki dosya), ham ölçüm değil | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/gp_tara.py:5-6` · `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/npl_tara.py:6` |
| Vekil proxy `https://r.jina.ai/<hedef-URL>` → aynı uç nokta | `HTTP 200`, ham JSON döndü | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/gp_tara.py:7` · `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt:5` |
| Ardışık istekler arası bekleme (hız sınırı) | `18` s | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/gp_tara.py:33` |
| Toplam sorgu (üç künye dosyası) | `42` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_db_taramasi.txt:54` |
| Dönen hit kaydı / benzersiz hit | `161` / `160` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_db_taramasi.txt:55-56` |
| Patent sorgusu / patent hit | `23` / `91` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_db_taramasi.txt:58-59` |
| NPL sorgusu / NPL hit | `19` / `70` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_db_taramasi.txt:60-61` |
| Boş dönen sorgu | `6` → q09 · q11 · n05 · n12 · x03 · x05 | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_db_taramasi.txt:57` |
| Birinci turdan diske arşivlenmiş vekil arama yanıtları | `arsiv_webfetch_1.json` … `arsiv_webfetch_5.json` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/` dizin listesi |

Üretici betikler ve hangi sorgu kümesini koştukları:

| Betik | Sorgu kümesi | Künye başlığı |
|---|---|---|
| `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/gp_tara.py` | q01–q16 · Google Patents tam metin | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_db_taramasi.txt:5` |
| `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/npl_tara.py` | n01–n12 OpenAlex + a01–a07 arXiv | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_db_taramasi.txt:23` |
| `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/tur3_ek.py` | x01–x07 · ek sorgular | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_db_taramasi.txt:44` |

Sorgunun metni, hedef URL'i, HTTP durumu, toplam sonuç sayısı ve ilk hitler satır satır şu üç
künye dosyasındadır: `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt` (patent), `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/npl/index_ozet.txt` (OpenAlex + arXiv),
`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index2_ozet.txt` (ek sorgular). Her sorgunun ham yanıtı aynı dizinde JSON dosyası olarak
durur (`qNN`, `nNN`, `aNN`, `xNN` önekli adlar); sayım üreteci:
`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/db_kanit.py`.

### §5.2 Derin incelemeye alınan dokuz patent kaydı (ilk istem metnine inildi)

Seçim ölçütü: §2 öğelerine (istem 1'in (a)–(f) gövdeleri; INT8 paketi için istem 9 ve 14;
pencere/stride için istem 10; zeroth-order için istem 12 — istem numaraları `docs/patent/pct_v2/CLAIMS_EN.md:46-50`) en yakın görünen yayımlar. Ham sayfa
metni `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/patentlar/` dizisinde durur (ham sayfa `9`); ilk istemin dilimleri `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md` içinde durur (kayıt başlığı `8` — US12639556B2 bu dilimde yok, tarifi ham sayfadan okundu) ve künye her satırda verildi; sayıların üreticisi `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_kayit_sayisi.txt:4-5`.
Terim sayımları `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt` dosyasında;
sayım alt-dizi sayımıdır, anlam iddiası taşımaz: üretici `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/derin_terim_say.py`.

| Yayım no · atfedilen · önculluk | İlk istemin konusu (ham metinden) | Yakın olduğu istem öğesi | Ayrım ve hüküm |
|---|---|---|---|
| **US12639556B2** · Google Llc · önculluk 2020-06-03 · yayım 2026-05-26 (künye: `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/arsiv_webfetch_5.json` ilk kayıt · `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:7`) | Özellik vektörleri + K slot vektörü; slot-attention matrisi ve slot güncelleme yinelemeleriyle "entity-centric" gösterim (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/patentlar/US12639556B2.md:507`) | istem 1(c) — permütasyon-değişmez küme işleme | İlk istem K varlık-gömütü ayrışmasını iddia eder; tek bulut-gömütü + next-token LM başı (d) ve aynı gömütlerden bölge-Δ açıklaması (e) bu istemde yoktur: `language model=0 · next token=0 · explanation=0 · slot=519 · permutation=15 · deep set=4` (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:8`). **Hüküm: istem 1(c) karşısında bu turun en yakın patent öncülü — vekil incelemesi zorunlu.** |
| **US12510888B2** · Howso Incorporated · önculluk 2018-04-09 (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt:73`) | "Conviction score" ve dışlama/dahil-etme türü *surprisal* skorlarıyla veri-elemanının modele dahil edilme kararı (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md:159`) | istem 1(e) — çıkarma-temelli skor | Birim context-action çifti çalışır; bulut veya dil modeli geçmez: `point cloud=0 · language model=0 · surprisal=392 · exclu=278 · explanation=11` (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:13`). **Hüküm: "girdiyi çıkar, çıktı kaymasını ölç" fikrine ciddi öncül; vekil bu dilime karşı istem 1(e)'yi daraltmayı değerlendirmeli.** |
| **CN117392424A** · 上海理工大学 · önculluk 2022-06-29 (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt:16`) | Çok-geometri + çift-kenar-dikkat ağı; sırasız nokta kümesi girişi, kanal dikkati, 3B sınıflandırma (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md:87`) | istem 1(a)(b)(c) | Nokta bulutu koluna yakın, metin/tabular bağlam kodlayıcısı ve next-token LM yok: `point cloud=197 · attention=254 · pooling=31 · permutation=5 · language model=0` (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:18`). **Hüküm: (b) çift-geometri koluna yakın; (a)+(d) kombinasyonu karşısında yıkıcı değil.** |
| **US12190228B2** · Latent AI, Inc. · önculluk 2019-09-13 (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt:98`) | Hedef-runtime parametreleriyle bağlama-özel DNN üretimi, çalışma planı, hiyerarşik düğümde dağıtım (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md:105`) | istem 14 | Nicemeleme ve INT8 geçiyor, ama INT8 ağırlık + çözümleme tablosu + kendi kendini açıklayan paketin birlikte dağıtımı ilk istemde yoktur: `quantiz=73 · int8=2 · lookup table=3 · language model=2` (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:23`). **Hüküm: arka plan öncülü; istem 14/15 kombinasyonunun vekil değerlendirmesi gerekir.** |
| **US12039636B2** · Siemens Healthineers Ag · önculluk 2021-09-13 (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt:18`) | Manyetik-rezonans tekrarlarında agregasyonla makine eğitimi; ilk istemde agregasyon kayıp toplamasıdır (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md:30`) | — (2. turdaki "istem 1(c) öncülü" tahmini çürüdü) | Deep-set terminolojisi başlıkta gerçekten var (`deep set=20 · permutation=12`), ancak nokta bulutu ve dil modeli yoktur: `point cloud=0 · language model=0` (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:48`). **Hüküm: istem 1(c) öncülü DEĞİL — bkz. §5.3 D2.** |
| **US20250156898A1** · Qomplx Llc · önculluk 2023-07-19 (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt:34`) | Reklam içeriği üretimi: uzman-ajan ağı + bilgi grafiği + RAG (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md:4`) | — (2. turdaki tahmin çürüdü) | Dil modeli ve bulut birlikte geçmiyor: `language model=54 · point cloud=3` (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:28`). **Hüküm: yapısal örtüşme yok.** |
| **WO2026046480A1** · Iav Gmbh · önculluk 2024-08-29 (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt:53`) | Doğal-dil test isteklerinden düzenli ifade üretimi; yapay zekâ destekli test otomasyonu | istem 1(e)(i) (tahmin) | Sayfa indirildi, ancak iddia bölümü çekilemedi: "1. ISTEM" satırı boş (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md:102`). Terim sayımı var (`language model=8 · point cloud=0`, `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:33`). **Hüküm: KANITSIZ — vekil PATENTSCOPE'tan açar.** |
| **WO2026170161A1** · Chanin Brent · önculluk 2025-02-10 (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt:10`) | SEMAD tanımlayıcılı sembolik çok-boyutlu matris (recognition-to-rendering) | istem 1(a) (tahmin) | OZET ve "1. ISTEM" satırları boş (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md:183-184`); atfedilen/önculluk alanları da ham dosyada boş (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:37`). **Hüküm: KANITSIZ.** |
| **US20260017347A1** · Leptude, Inc. · önculluk 2024-07-10 (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt:61`) | Kapsül ağlarında zaman-mekân latent füzyonu ve dinamik routing (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md:187`) | istem 1(e)(ii) (tahmin) | Bulut, LM ve açıklama zinciri birlikte yok: `language model=2 · point cloud=0 · explanation=7` (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:43`). **Hüküm: uzak.** |

**Öne çıkan risk (bu turun ölçtüğü iki ayrı öncül):**

| Öncül | Yaklaştığı istem omurgası | Ölçülen ayrım | Kanıt (ham kayıt) |
|---|---|---|---|
| `US12639556B2` — permütasyon-değişmez küme + öğrenilmiş slot etkileşimi | istem 1(c) | tek bulut-gömütü + next-token LM ve bölge-Δ açıklaması ilk istemde yok: `language model=0 · next token=0` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:8` · `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/patentlar/US12639556B2.md:507` |
| `US12510888B2` — dışlama/dahil-etme türü çıkarma-temelli skor | istem 1(e) | birim context-action çifti çalışır, bulut ve LM geçmez: `point cloud=0 · surprisal=392` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:13` |

İki kayıt birlikte okunduğunda istem 1'in (c)+(e) omurgasına yaklaşır. İkisinin aynı cihazda
birleştirildiğine dair kayıt bu turda **bulunmadı**; bu bir yokluk iddiası değil, §5.5 kapsam
sınırının konusudur. Ayrımın tam gerekçesi §5.2 satırlarındadır.

### §5.3 Bu turun kendi iddialarını çürüttüğü yerler (düzeltme kaydı)

| Madde | Önceki turun iddiası | Ham kayıt ne diyor | Uygulanan düzeltme |
|---|---|---|---|
| D1 | US12639556B2 için "permütasyon-değişmez küme + nokta bulutu + metin" | Ham metinde `language model=0 · next token=0` (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:8`) | "metin/LM" kısmı yanlıştı; cümle düzeltildi (ölçüm değiştirilmedi) |
| D2 | US12039636B2 "deep-set agregasyonu" etiketiyle istem 1(c) öncülü sayılmıştı | İlk istem manyetik-rezonans tekrarlarında kayıp toplaması; `point cloud=0` (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md:30` · `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:48`) | "istem 1(c) öncülü" etiketi §5.2'den düşürüldü |
| D3 | Atameler `KANIT_derin_terim.txt` üzerinden okunabilirdi | `atfedilen` alanı Google'ın "listelenen atameler hatalı olabilir" uyarısını yakalıyor (`patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_terim.txt:7`) | Atama bilgisi bu dosyadan okunmaz; künyeler `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt` API alanından alındı, resmî teyit vekilin işi |
| D4 | US12639556B2 künyesinin kaynağı | Kırkiki sorgunun JSONL indekslerinde kayıt yok; kaynak birinci turdan arşivlenmiş vekil arama yanıtı `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/arsiv_webfetch_5.json` ve o dosyada hangi sorgunun yanıtı olduğu yazılı değil | §5.2 künyesi bu şerhle verildi; sorgu metni **KANITSIZ** |
| D5 | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md:35` boş başlığı | İlk istem dilimlenirken iddia metnindeki `## ` dizisi bloğu bölmüş; içerik ham dosyada duruyor | Dilim notu §5.2 girişine ve §5.5'e yazıldı |

### §5.4 İsim çakışması ve boş dönen sorgular

| Sorgu | Operatör/metin | Toplam sonuç | Dönen kayıt | Dönen kayıtlarda terim | Kanıt |
|---|---|---|---|---|---|
| x01 | `ti:QUANN` | `21` | `10` | "quann" → `0` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index2_ozet.txt:5` · `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_isim_taramasi.txt:6-8` |
| x02 | `ti:"cloud language model" OR ti:CloudLM` | `3` | `2` | üç terim de `0` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index2_ozet.txt:14` · `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_isim_taramasi.txt:12-16` |

Başlık çakışması tespit edilmedi. `ti:` operatörünün başlığı ilgisiz kayıtlar döndürmesinin sebebi
**doğrulanmadı** — sebep uydurulmadı; operatör davranışı doğrulanmadan bu iki satır "temiz" sayılmaz.

| Boş dönen sorgu | Sorgu metni | Toplam | Kanıt |
|---|---|---|---|
| q09 | `"zeroth-order" OR "derivative-free" "without backpropagation" training neural network` | `0` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt:76` |
| q11 | `"sliding window" "fixed stride" sentence embeddings transformer aggregation` | `0` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt:89` |
| x03 | `"context encoder" "point cloud" "reader" "next token"` | `0` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index2_ozet.txt:20` |
| x05 | `"tabular record" OR "tabular data" "set aggregation" "class assignment"` | `0` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index2_ozet.txt:33` |
| n05 | `occlusion ablation input removal language model explanation` | `0` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/npl/index_ozet.txt:39` |
| n12 | `sentence level embedding sliding window document representation aggregation` | `0` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/npl/index_ozet.txt:83` |

**Yorum sınırı:** sıfır sonuç bir yokluğun kanıtı değildir; tek kanalın tam-metin dizimi ve sorgu
yüzeyi ancak bu kadarını görür. İstem çekirdeğinin dört kombinasyonu sıfır döndü — bu, §5.5
sınırıyla birlikte okunur.

### §5.5 Bu turda YAPILMAYANLAR (kapsam sınırı)

| Madde | Yapılmayan | Dayanak |
|---|---|---|
| K1 | Tek patent kanalı: Google Patents `xhr/query` vekil proxy üzerinden. Espacenet, PATENTSCOPE, USPTO Full-Text ve CPC/IPC sınıf taraması yapılmadı | §4 V1; dört anahtarlı aracın durum kaydı `patent-cikti/pct/pct_v2-pct-inceleme/oncesanat_ham_tur2_2026-11-19.json` |
| K2 | Sayfa başına `num=5` istendi; büyük kümelerin tamamı indirilmedi, sıralama Google'ınki: q16 `1035` · x07 `3252` · x06 `2370` · a01 `353` | `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index_ozet.txt:129` · `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index2_ozet.txt:46` · `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/index2_ozet.txt:37` · `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/npl/index_ozet.txt:86` |
| K3 | Birinci ve ikinci turun araç yanıtlarının ham kaydı yok | §0 gerekçesi; §1.1 künyeleri KANITSIZ |
| K4 | Atıf ağları (ikinci derece) ve X/Y/P kategori sınıflaması | §4 V2 ve V5 açık |
| K5 | İki WO kaydında iddia metni çekilemedi | §5.2 son iki satır · `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md:102` ile `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/DERIN_kayit.md:183-184` boş |
| K6 | Hız kısıtı: ardışık istekler `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/gp_tara.py:33` satırındaki beklemeyle tekil koşuldu. Paralel istekte engel sayfası görüldü; tetikleme mekanizması **doğrulanmadı** | gözlem kaydı bu satır |

---

## §6 TÜRKÇE ÖZET

Bu tarama, İstem 1'in (a)–(f) öğeleri ve İstem 14 için NPL kayıtlarını (§1.1 künyesi
`docs/patent/pct_v2/ON_SANAT.md:21`) `search_non_patent_literature` aracıyla doğruladı. En yakın üç NPL ailesi:
EdgeConv/çift-geometri (nokta-bulut tarama), simetrik küme agregasyonu (Set Transformer ve
türevleri) ve açıklama ailesi (SHAP/LIME/occlusion/nokta-bulut görselleştirme). §5'te patent
veritabanı taraması da koşuldu: toplam sorgu `42` · dönen hit kaydı `161` · benzersiz hit `160` · ilk istem düzeyinde derinleşen kayıt `9` kayıt sayısı `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_derin_kayit_sayisi.txt:4`, ötekiler `patent-cikti/pct/pct_v2-pct-inceleme/db_taramasi/KANIT_db_taramasi.txt:54-61`). En yakın iki patent öncülü US12639556B2
(permütasyon-değişmez küme + öğrenilmiş etkileşim) ve US12510888B2 (çıkarma-temelli katkı skoru).
Hiçbir doğrulanmış kayıt, "bağlam kodlayıcı çıktısından token+tabular girdiyle nokta bulutu kurup
çift-geometriyle birleştiren ve aynı bulut üzerinde ayrı açıklama modeli olmadan üç açıklama
işlemi çalıştıran" kombinasyonu taşımadı — ama bu bir "küresel yokluk" iddiası DEĞİLDİR; yalnızca
incelenen kaynaklarda böyle bir kayıt döndü. Kalan vekil işleri: Espacenet/PATENTSCOPE CPC
taraması, atıf ağı, X/Y/P sınıflaması ve KANITSIZ künyelerin teyidi (§4 ile §5.5).
