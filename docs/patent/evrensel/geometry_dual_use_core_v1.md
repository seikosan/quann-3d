# PCT ULUSLARARASI BAŞVURU — Evrensel Çift-Kullanımlı Geometri Çekirdeği (v1)

**PCT (Patent Cooperation Treaty) kapsamında uluslararası başvuru**
**PCT/RO/101 formu + 556 sayılı Sınai Mülkiyet Kanunu'na uygun teknik metin**

## 0. PCT REQUEST (PCT/RO/101 — temel veriler)

| Alan | Bilgi |
|---|---|
| **Receiving Office (RO)** | TÜRKPATENT (TR) — ulusal alan ofisi |
| **Başvuru Türü** | Patent — Uluslararası (PCT) başvuru |
| **Başvuru No** | [PCT tarafından verilecektir: TR 2026/… ] |
| **Başvuru Tarihi** | [uluslararası başvuru tarihi] |
| **Buluşçu(lar)** | [Ad Soyad, adres, uyrukluk, T.C. Kimlik] |
| **Başvuru Sahibi** | [Unvan, adres, uyrukluk] |
| **Vekil** | [Patent vekili / ofis, PCT yetkili] |
| **Buluş Başlığı** | Çift-Geometri Nokta-Bulutu Temsili ile Birleşik Hesaplama ve Native Açıklanabilirlik Yöneten Evrensel Sistem, Yöntemi ve Bilgisayar Programı |
| **Teknik Alan (IPC)** | G06F 18/00; G06N 3/04; G06N 5/02; G06V 10/77 (öneri) |
| **Öncelik Beyanı** | [Boş bırakılacaktır — bu evrensel kapsam (dual-use çekirdek + dil/fraud/AML/bakım), TR ulusal (dil) başvuruda (`TURKPATENT_basvuru_v2.3.md`) açıklanmadığından TÜRKPATENT m.31 uyarınca öncelik talep edemez; bağımsız başvuru, kendi öncelik tarihi T2] |
| **Atanan Sözleşmeci Devletler (Designation)** | [Vekil tarafından PCT'nin tümü veya hedef ülkeler seçilecektir] |
| **Aranan Usul** | Uluslararası araştırma (IIS: TÜRKPATENT/EPÖ) + 18. ay uluslararası yayın (PCT Rule 48) + (isteğe bağlı) 22. ay uluslararası ön inceleme |

> **GİZLİLİK / ÖNCELİK UYARISI:** Bu dosya patent başvurusu öncesi hazırlanmıştır. Uluslararası başvuru tarihi (öncelik hakkı) alınana kadar bu buluşa ilişkin hiçbir yayımlama (arXiv, konferans, dergi, blog, genel erişimli GitHub, podcast, sunum) yapılmamalıdır. Yayımlama, yenilik koşulu açısından yıkıcı etki (self-depublication) yaratabilir. PCT yayınımı (18 ay) öncesinde dahi hiçbir kamuya açıklama yapılmamalıdır.

## İÇİNDEKİLER

0. **PCT Request** — temel veriler
1. **Başlık**
2. **Açıklama** — 2.1 Teknik Alan · 2.2 Ön Teknik · 2.3 Sorun · 2.4 Özü · 2.5 Etkiler · 2.6 Şekiller · 2.7 Detay (2.7.1–2.7.6)
3. **İstemler** — 3 bağımsız (Sistem, Yöntem, Bilgisayar Programı) + 10 bağımlı (6 mekanizma + 4 dikey embodiment) = 13 istem
4. **Özet (Abstract)**
5. **Şekiller** — 7 şekil, ayrı sayfalar
6. **Atıf Listesi**
7. **Vekil Değerlendirmesi** — unity, öncelik, barizlik, düzeltmeler (başvuruya girmez)

> **TR/PCT ilişki (kritik):** Bu **PCT (evrensel)** başvuru, TR ulusal başvurumuz `TURKPATENT_basvuru_v2.3.md`'deki **dil** uygulamasından **farklı, daha geniş** bir kapsamdır (dual-use geometri çekirdeği + dil/fraud/AML/bakım). Evrensel kapsam TR dil başvurda açıklanmadığından bu PCT dosyası TR'dan **öncelik talep edemez** — bağımsız bir başvuru olarak kendi öncelik tarihine (T2) sahiptir. İki dosyanın metin bağlantısı kendiliğinden önceki tarih yaratmaz.

---

## 1. BAŞLIK

**Çift-Geometri Nokta-Bulutu Temsili ile Birleşik Hesaplama ve Native Açıklanabilirlik Yöneten Evrensel Sistem, Yöntemi ve Bilgisayar Programı**

(İngilizce: “Universal System, Method and Computer Program for Joint Computation and Native Explainability Managed by Dual-Geometry Point-Cloud Representation”)

---

## 2. AÇIKLAMA

### 2.1 Teknik Alan

Bu buluş, düşük boyutlu (3 ≤ D ≤ 64) nokta-bulutu geometrisine dayanan makine öğrenmesi sistemlerine; özellikle aynı öğrenilmiş geometrik temsilin hem tahmin/karar skorunu hesapladığı hem de native açıklama yüzeyi olarak kullanıldığı sistem, yöntem ve bilgisayar programlarına ilişkindir. Birimler, girdinin türüne göre token, özellik-değer kimliği, işlem, pencere veya başka bir gözlem olabilir; her birim bir Öklid noktasına projekte edilir.

Çift-geometri temsili, tekil nokta özelliklerini ve nokta çiftlerine ait ilişkileri birlikte kodlar. Nokta geometrisinden elde edilen küme durumu bir okuyucuya verilir. Aynı nokta bulutu ve bu buluta bağlı okuyucu, leave-one-point-out ablation, bölge/prototip isimlendirme ve bölge-kayması/drift incelemesi gibi native açıklama işlemlerinin kaynağıdır. Sistem dil modelleme, işlem dolandırıcılığı (fraud), kara para aklama (AML) tipolojisi ve endüstriyel bakım alanlarına uygulanabilir.

### 2.2 Önceki Teknik (Prior Art)

#### 2.2.1 Tekil karar temsilleri ve özellik atıfları

Transformer mimarileri (Vaswani ve ark., 2017, *NeurIPS 30*) sıralı girdiyi self-attention ile işler; attention hesabı sekans uzunluğu N bakımından O(N²)'dir ve karar çoğunlukla yüksek boyutlu bağlamsal gösterimlerden okunur. Bu çalışmada ileri sürülen ayrım, her türlü attention kullanımının yokluğu değil; tahmin/karar okuyucusunun geometriden üretilen küme durumunu kullanması ve aynı geometrinin açıklama kaynağı olmasıdır.

Dereceli ağaçlar ve SHAP, çoğunlukla özellik adı ile sayısal atıf değerlerinden oluşan liste sunar. SHAP additivity aksiyomu uyarınca atıfların toplanabilirliği, çift-geometrili ve etkileşimli bir nokta-bulutu temsilinde native leave-one-point-out duyarlılığın toplam skoru yeniden ürettiğini garanti etmez. Kayıt düzeyindeki atıf ayrıca tek başına popülasyon düzeyindeki hatalı-alarm sürücülerini veya bölge/prototip driftini tanımlamaz.

#### 2.2.2 Nokta-kümesi yöntemleri ve geometrik analiz

- **Deep Sets** (Zaheer ve ark., 2017, *NeurIPS 30*) ve **Set Transformer** (Lee ve ark., 2019, *ICML*, 3714–3723) kümelerden havuzlanmış temsil üretir. Buradaki özgül mekanizma, öğrenilmiş koordinatları ve nokta çiftlerinin geometrisini aynı karar temsiline taşırken aynı bulutu native açıklama işlemlerinde korumaktır.
- **PointNet** (Qi ve ark., 2017, *IEEE/CVF CVPR*, 77–80), paylaşılan nokta MLP'si ve simetrik küme agregasyonu kullanır; nokta koordinatları esasen ham 3B vizyon/sensör girdisidir. Bu, farklı veri türlerinden öğrenilmiş düşük boyutlu noktaları, çift-geometriyi ve aynı buluttan native ablation/prototip/drift açıklamasını birleştiren sistemle aynı değildir; PointNet dil veya tabular kayıtlar için bu birleşimi ortaya koymaz.
- **Word2Vec** (Mikolov ve ark., 2013, arXiv:1301.3781), **GloVe** (Pennington ve ark., 2014, *EMNLP*, 1532–1543) tekil kelime vektörlerini; **Chung & Abbott** (2021, *Nature* 592(7853), 117–122) nöral popülasyon geometrisi çerçevesini ele alır, ancak burada tarif edilen birleşik karar ve açıklama sistemini tanımlamaz.
- **Hu ve ark.** (2026, arXiv:2607.04525; DOI 10.48550/arXiv.2607.04525) model içi kavramları point-cloud manifold ve vektör alanlarıyla post-hoc analiz eder. **Xu** (2026, arXiv:2605.17084; DOI 10.48550/arXiv.2605.17084) Subspace PGA ile katman mesafe yapısını inceler. Anılan çalışmalar geometrik analiz sunar; burada tarif edilen geometriyi hem skor hesaplama substratı hem aynı nesneden üretilen native açıklama yüzeyi yapan yeni sistem mimarisini önermez.

İncelenen kaynaklarda (Transformer, GBM+SHAP, DeepSets, Set Transformer, PointNet, Hu-2026 ve Xu-2026) geometrinin hem hesaplama hem native açıklama substratı olduğu bu özgül birleşim gösterilmemektedir. Bu, küresel bir yokluk iddiası değildir; kapsamı burada incelenen kaynaklarla sınırlıdır ve daha geniş prior-art taraması yapılmasını gerektirir.

#### 2.2.3 Önceki teknikteki teknik boşluk

İncelenen yaklaşımlar, düşük boyutlu öğrenilmiş nokta-bulutu geometrisini karar üretiminde kullanan temsil ile bu aynı temsilin native açıklama nesnesi olarak yeniden kullanılmasını tek bir teknik çekirdekte birleştirmez. Bu nedenle, kararın oluştuğu nesne ile karar sonrasında incelenen açıklama nesnesi çoğu sistemde ayrışır; tekil atıf, popülasyon denetimi ve dağıtım kanıtı da farklı araçlara dağılır.

### 2.3 Teknik Sorun

Bu buluş aşağıdaki teknik sorunlara çözüm sağlamayı amaçlar:

1. Tahmin/kararın tek bir opak yüksek boyutlu vektör veya atıf listesi üzerinden incelenmesi; kararın geometrik alt-bileşenlerinin doğrudan denetlenememesi.
2. Etkileşimli, toplamsal olmayan temsil üzerinde post-hoc özellik atıflarının kararın native etkisini yeniden üretmemesi ve tekil açıklamanın popülasyon sürücülerini göstermemesi.
3. Karar ve açıklama için ayrı temsil ve araç zincirleri kullanılması nedeniyle edge dağıtımının kompakt, self-contained ve bit-kesin doğrulanabilir biçimde kurulamaması.
4. Metin, tabular işlem/kayıt, AML işlem penceresi ve sensör/şikâyet penceresi gibi farklı girdi türlerine ortak bir geometri çekirdeğinin uygulanamaması.

### 2.4 Buluşun Kısa Açıklaması (Çözüm)

İcadın özü, tek bir düşük boyutlu nokta bulutunun **aynı anda hesaplama substratı ve native açıklama yüzeyi** olmasıdır. Geometriyi hesaplayıp sonra ayrı bir açıklama modeliyle dolaylı olarak yorumlamak yerine, skor ve açıklama aynı noktalardan, aynı nokta-çifti ilişkilerinden ve aynı bölge/prototip yapısından türetilir.

| Sütun | Modül ve işlev | Aynı bulutla bağı |
|---|---|---|
| Temsil | Girdi birimleri öğrenilmiş noktalara projekte edilir (110); kaynak modül token, kimlik-bucket veya pencere olabilir. | Noktalar, sonraki tüm adımların ortak nesnesidir. |
| Çift-geometri | Nokta MLP'si ve çift MLP'si, nokta özellikleri ile mesafe/açı ve isteğe bağlı fark/norm bileşenlerini kodlar (120). | Nokta ve çift ilişkileri simetrik agregasyonla küme durumuna birleştirilir (123). |
| Okuyucu | Okuyucu küme durumundan görev skorunu/çıktısını üretir (130). | Skor aynı bulut geometrisinden hesaplanır. |
| Native açıklama ve dağıtım | Yorumlama (150), ablation/decoder (260/270), karşı-faktürel denetim (280), paketleyici (290). | Ablation, prototip/bölge ve drift aynı buluttan okunur; okuyucu ve gerekli bileşenler self-contained pakette sunulur. |

Permütasyon-değişmezlik, paylaşımlı nokta/çift fonksiyonlarının ardından uygulanan simetrik küme agregasyonu için geçerlidir. Bağlamdan veya tarihten üretilen nokta koordinatları sıra-duyarlı olabilir; sistemin tüm aşamalarının sıra-dışı olduğu ileri sürülmez. Fark vektörü dönüşüm altında eşdeğişkendir; mesafe, kosinüs açısı ve normlar koordinat dönüşümüne karşı değişmez bileşenlerdir.

Dağıtım ilkesi, okuyucu, gerekli kantizör/decoder ve manifestin kompakt, self-contained ve bit-kesin doğrulanabilir bir pakette birleştirilmesidir. Paket boyutu ve sayısal parite, uygulamaya, girdi sözleşmesine ve hedef donanıma bağlıdır; tek bir evrensel boyut veya parite değeri ileri sürülmez.

### 2.5 Teknik Etkiler

1. **Native, okunabilir açıklama:** Aynı geometriden leave-one-point-out ablationın metne çözülmesi; bölge/prototipin isimlendirilmesi; zaman veya çalışma koşulları arasındaki bölge kaymasının drift olarak ölçülmesi sağlanır.
2. **Toplamsal olmayan atfın dürüst sunumu:** Nokta-çıkarma logit değişimi karşı-faktürel duyarlılık olarak raporlanır; bunun SHAP türü toplamsal katkı olduğu veya tekil etkilerin toplamının skoru verdiği varsayılmaz.
3. **Popülasyon denetimi:** Tekil ve ikili maske karşı-faktüelleri, alarm popülasyonunda yaygın ve nadir sürücüleri incelemeye; örnekler arası bölge/prototip dağılımı ise kaymayı izlemeye imkân verir.
4. **Kompakt edge dağıtımı:** Okuyucu, kantizör, decoder ve manifest tek bir self-contained paket olarak taşınabilir; ilgili hedefte bit-kesin parite ve kaynak bütünlüğü doğrulanabilir.
5. **Dikeyler arası aktarılabilir çekirdek:** Temsil kaynağı ve bölge isimleri dikeye göre değişirken geometrik karar ve native inceleme mekanizması dil, fraud, AML ve bakım uygulamalarında kullanılabilir. Bu, dikeyler arasında doğruluk veya genelleme üstünlüğü kanıtı değildir.

**Sonuç sınırları:** Bu buluş doğruluk-üstünlüğü veya SOTA iddiası taşımaz. Fraud ölçümünde matched PR-AUC 0.462 iken XGBoost 0.8047'dir. QUANN, küçük ve sınırlı modellerde 0th-order nokta daraltma seçeneğidir; büyük model eğitimi için ölçeklenebilir olduğu gösterilmemiştir. Bir bölge haritası nedensellik, eksiksiz açıklama, düzenleyici onay veya garanti edilen karar doğruluğu değildir. AML'de kaba etiket kararlılığı harita koordinatlarının kararlı olduğu anlamına gelmez; bakım edge paketi ham sensör çerçevelerini değil ölçeklenmiş bulut girdisini alır.

### 2.6 Şekillerin Kısa Açıklaması

- **Şekil 1:** Birleşik mimari — 100 girdi/bağlam modülü, 110 nokta projeksiyonu, 120 çift-geometri kodlayıcı, 123 simetrik birleştirme, 130 skor okuyucu, 150 yorumlama, 260 ablation, 270 decoder, 280 karşı-faktürel denetim, 290 dağıtım.
- **Şekil 2:** Nokta ve çift-geometri — D-boyutlu noktalar, [pᵢ−pⱼ, ‖pᵢ−pⱼ‖, cosθ, ‖pᵢ‖, ‖pⱼ‖] vektörü ve simetrik agregasyon.
- **Şekil 3:** Aynı buluttan native açıklama — çıkarılan noktanın logit farkı, metin decoder'ı, bölge/prototip adı ve zamanlar arası drift.
- **Şekil 4:** AML bölge/prototip isimlendirme örnekleri — `fig/fig4a_aml_tipoloji.png`, `fig/fig4b_aml_tipoloji.png`.
- **Şekil 5:** Fraud kimlik-bucket bulutu — `fig/fig5_fraud_bulut.png`.
- **Şekil 6:** Bakım sensör penceresi bulutu ve bölge kayması — `fig/fig6a_bakim_bolge.png`, `fig/fig6b_bakim_drift.png`.
- **Şekil 7:** Self-contained dağıtım ve doğrulama — fraud ONNX/manifest ve bakım INT8 edge örnekleri.

### 2.7 Buluşun Detaylı Açıklaması

#### 2.7.1 Ortak mimari (100–150)

Girdi modülü (100), gözlemleri birimlere ayırır. Bu birimler token, (özellik, değer-bucket) kimliği veya zaman penceresindeki ölçüm/özet bileşenleri olabilir. Projeksiyon modülü (110), her birimi pᵢ ∈ ℝᴰ noktasına dönüştürür; 3 ≤ D ≤ 64'tür. Bağlam veya tarihçe kullanılarak üretilen pᵢ koordinatları sıra-duyarlı olabilir.

Çift-geometri kodlayıcı (120), her nokta için paylaşılan nokta MLP'si ile nokta özellikleri; her (pᵢ,pⱼ) için çift MLP'si ile geometrik ilişkiler üretir. Tercih edilen çift vektörü şöyledir:

```text
x_ij = [p_i − p_j, ‖p_i − p_j‖, cos(θ_ij), ‖p_i‖, ‖p_j‖] ∈ ℝ^(D+4)
```

Burada pᵢ−pⱼ bir koordinat dönüşümü altında eşdeğişkendir; ‖pᵢ−pⱼ‖ mesafesi, cos(θᵢⱼ), ‖pᵢ‖ ve ‖pⱼ‖ dönmeye karşı değişmezdir. Birleştirme modülü (123), örneğin nokta özelliklerine max-pool ve çift özelliklerine mean-pool uygulayıp bunları birleştirerek cloud_state üretir. Permütasyon-değişmezlik, bu simetrik agregasyon işleminin özelliğidir; girdiyi hazırlayan bağlam/tarihçe sıralamasına genellenmez.

Okuyucu (130), cloud_state'ten sınıflandırma skoru, risk, sonraki-token dağılımı veya göreve uygun başka çıktı üretir. Yorumlama modülü (150), aynı bulutun nokta, çift ve bölge/prototip bilgisini inceler. Bu yapı, geometriyi kararın dışındaki post-hoc bir çizime dönüştürmez: okuyucunun girdisi ile açıklamanın nesnesi aynı buluttur.

#### 2.7.2 Native açıklama, karşı-faktürel ve paket (260–290)

Ablation modülü (260), tek bir pᵢ noktası çıkarıldığında okuyucu logitindeki değişimi hesaplar. Bu değer, etkileşimli çift-geometri nedeniyle **leave-one-point-out karşı-faktürel duyarlılık** olarak adlandırılır; katkıların additif olduğu veya mutlak değişimlerin toplamının başlangıç logitini verdiği iddia edilmez. Tam-sözlük decoder (270), açıklamadaki nokta/kimlikleri insan-okunur birimlere dönüştürebilir; örneğin fraud embodiment'inde 221 kimliğin tamamı çözümlenir ve hiçbir bucket “other” altında bırakılmaz.

Karşı-faktürel denetim modülü (280), bir noktanın veya nokta çiftinin maskelenmesi/çıkarılması sonrasında skorun eşiğin üstünde kalıp kalmadığını sınar. Popülasyon düzeyinde bu sonuçlar yaygın sürücülerle nadir sürücüleri ayırabilir. Bölge/prototip isimlendirmesi gözlemin en yakın isimli prototipe göre konumunu ve destekleyici geometrik ölçüleri verir. Farklı zaman veya işletim koşullarındaki bölge dağılımı karşılaştırılarak drift ölçülebilir; bu, nedensel açıklama ya da otomatik arıza teşhisi garantisi değildir.

Paketleyici (290), göreve göre okuyucu, kantizör, decoder ve manifesti tek self-contained pakette toplar. Manifest kaynak kimliği/hash'i, yapılandırmayı ve self-test/parite sonuçlarını taşıyabilir. Sayısal sonuçlar uygulamaya özel ölçümlerdir; genel iddianın sabit paket boyutu veya parite eşiği yoktur.

#### 2.7.3 Dil embodiment'i

Metin girdisinde her token (alt-sözcük dahil) 110 ile bir noktaya projekte edilir; cümle bulutu 120/123 tarafından kodlanır ve okuyucu (130) bir sonraki token dağılımını üretir. 150/260 üzerinden token noktasının leave-one-point-out logit farkı incelenebilir; sözlük decoder'ı bunu metne çevirebilir. Agglütinatif dillerde aynı kökten gelen alt-sözcüklerin buluttaki bölgeleri araştırılabilir; bu, tüm sıralamanın permütasyon-değişmez olduğu anlamına gelmez.

**v2.4 güncellemesi (2026-10-27):** geniş bulut boyutu ve 20 nokta konfigürasyonunda 200K korpus üzerinde ölçülmüştür.

- PPL 37.63 ölçüldü (`logs/run_r2_cloud_attn_200k_d64_n20.log:26`).
- Causal baseline PPL 53.95 ölçtü (`logs/run_baseline_200k_v2.log:26`).
- Fark Δ16.32 PPL; iki run pencere açısından eşleşmez, fark tek başına mimariye atfedilmez.

**R2b gözlemi:** kontrollü ablation değil, gözlemdir.

- Dar pencere koşusu PPL 57.04 ölçtü (`logs/run_r2_cloud_attn_200k_d64.log:26`).
- Geniş pencere koşusu PPL 37.63 ölçtü (`logs/run_r2_cloud_attn_200k_d64_n20.log:26`).
- En geniş pencere koşusu PPL 134.31 ölçtü (`logs/run_r2_cloud_attn_200k_d64_n30.log:26`).
- Eğitim cümlesi koşullar arasında farklıdır (`logs/run_r2_cloud_attn_200k_d64.log:9`).

**Max-pool**, attention-pool'dan daha stabil ölçülmüştür; iki run aynı korpus, bulut boyutu, katman sayısı ve epoch
bütçesini paylaşır, havuzlama yöntemi farklıdır.

- Max-pool koşusu PPL 49.84 ölçtü (`logs/run_r3_200k_max.log:27`).
- Attention-pool koşusu PPL 55.75 ölçtü (`logs/run_r3_200k_attn.log:27`).

Bu sonuçlar bir dikey embodiment kanıtıdır, genel performans garantisi değildir.

#### 2.7.4 Fraud embodiment'i

Tabular işlem girdisinde her özellik/değer-bucket çifti kimlik birimi olur (210); kimlik noktası modülü (220) bunları buluta projekte eder. Okuyucu (130) risk logitini, ablation (260) leave-one-point-out farkını, tam-sözlük decoder (270) düz-metin etiketi, karşı-faktürel denetim (280) tek/çift-maske alarm etkisini ve paketleyici (290) ONNX/quantizer/decoder/manifest paketini üretir.

Ölçülen örneklerde tek-maskeli audit'te alarm popülasyonunun %94–100'ü unflag olmuştur. Matched PR-AUC 0.462, XGBoost PR-AUC 0.8047'dir; üstünlük iddia edilmez. Fraud dağıtım örneği 543 KB ONNX, Torch↔ORT paritesi 1.5e-6 ve 45.5k işlem/s ölçmüştür; bunlar bu embodiment'in koşullarıdır, evrensel paket sınırı değildir.

#### 2.7.5 AML embodiment'i

AML'de her işlem veya işlem penceresi nokta-bulutuna projekte edilir; okuyucu tipoloji/riske yönelik çıktı verir; prototip isimlendirme modülü bulutu isimli tipoloji prototipleriyle karşılaştırır. Elliptic held-out örneğinde prototip isimlendirme genel olarak %75, illicit sınıfında %45'tir. Beş-seed pilotta kaba isimli-tipoloji kararlılığı 0.325, SHAP top-1 kararlılığı 0.1025'tir (3.2×); **bu, harita koordinatlarının kararlı olduğu anlamına gelmez**: koordinatlar ve attribution büyüklükleri retraining-stable olarak gösterilmemiştir. İddia, doğruluk veya SAR gerekçesi garantisi değil, isimli geometrik inceleme yüzeyidir.

#### 2.7.6 Bakım embodiment'i

Bakımda ölçüm penceresi veya özellikleri (100/110) sensör-noktalarına dönüştürülür; 120/123 okuyucuya (130) bulut durumu verir; bakım mühendisi 150 ile prototip ve bölge kaymasını inceler. CWRU record-disjoint ikili görevde ROC-AUC 0.999 ve PR-AUC 1.000 ölçülmüştür; GBT karşılığı 1.000/1.000'dir. C-MAPSS'de nearest-prototype açıklaması 1,064 pencerede %81.7 kapsamış; çalışma kayması drift alarmı 2.4× (1.17%→2.81%) olmuştur. INT8 okuyucu edge örneği 79 KB ve Torch↔ONNX paritesi 5.07e-7 ölçmüştür. Bu artifact ham sensör çerçevesini değil, ölçeklenmiş bulut girdisini alır; CWRU metriği bu C-MAPSS paketi için ileri sürülmez.

---

## 3. İSTEMLER

### Bağımsız İstemler

**İstem 1 — Sistem:** Düşük boyutlu nokta-bulutu geometrisiyle bir girdi biriminden görev çıktısı üreten ve geometrik açıklama sağlayan bir bilgisayar sistemi olup, **karakterize edici özellik olarak**:

(a) her bir token'ı veya girdi birimini 3 ≤ D ≤ 64 boyutlu Öklid uzayda bir pᵢ noktasına projekte eden bir projeksiyon modülü (110) ve nokta/çift özelliklerini **permütasyon-değişmez küme agregasyonu** ile bir küme durumuna birleştiren bir modül (123);

(b) nokta özelliklerine ek olarak nokta çiftleri için en az mesafe ve açı bileşenlerini kodlayan bir çift-geometri kodlayıcısı (120);

(c) bu küme durumundan bir görev skoru veya çıktı üreten bir okuyucu (130);

(d) **aynı nokta bulutundan** native açıklama üreten bir yorumlama sistemi; bu sistemin (i) leave-one-point-out logit-ablationı decoder (270) ile metne çevirmesi, (ii) bölge/prototip isimlendirmesi ve (iii) bölge-kayması/drift incelemesinden en az birini yürütmesi için ablation modülü (260), decoder (270) ve/veya yorumlama modülü (150) ile karşı-faktürel modül (280) içermesi; ve

(e) okuyucuyu, gerekli kantizörü, decoder'ı ve manifesti self-contained kompakt bir dağıtım paketinde birleştiren paketleyici (290)

içermekte olup, **aynı nokta-bulutu geometrisi hem görev skorunun hesaplama substratı hem de native açıklamanın kaynağıdır**.

**İstem 2 — Yöntem:** Nokta-bulutu geometrisiyle görev çıktısı ve native açıklama üreten bilgisayarla uygulanan bir yöntem olup, **karakterize edici özellik olarak**:

(a) token veya girdi birimlerini 3 ≤ D ≤ 64 boyutlu noktalar olarak projekte etme;
(b) nokta ve nokta-çifti özelliklerini hesaplama;
(c) nokta özellikleri ve çift-geometri özelliklerini permütasyon-değişmez küme agregasyonuyla bir küme durumuna birleştirme;
(d) küme durumundan okuyucu ile görev skoru/çıktısı üretme;
(e) **aynı bulut** üzerinde leave-one-point-out logit-ablation, decoder ile metin, bölge/prototip isimlendirme veya bölge-kayması/drift incelemesinden en az birini yürütme; ve
(f) okuyucu ile gerekli kantizör, decoder ve manifesti kompakt, self-contained, doğrulanabilir dağıtım paketinde sunma

adımlarını içermektedir; permütasyon-değişmezlik (c)'deki simetrik agregasyon için geçerlidir ve bağlam/tarihçeden üretilen noktalar sıra-duyarlı olabilir.

**İstem 3 — Bilgisayar programı:** Bir işlemci üzerinde çalıştırıldığında İstem 2'deki yöntem adımlarını gerçekleştiren program talimatlarını içeren, geçici olmayan, bilgisayarca okunabilir bir kayıt ortamı.

### Bağımlı İstemler

**İstem 4:** İstem 1'deki sistem olup, çift-geometri kodlayıcısı her nokta çifti için `[pᵢ−pⱼ, ‖pᵢ−pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖] ∈ ℝ^(D+4)` vektörünü kullanır; mesafe ve kosinüs bileşenleri koordinat dönüşümüne karşı değişmez, fark vektörü ise koordinat dönüşümüyle eşdeğişkendir.

**İstem 5:** İstem 1'deki sistem olup, D değeri `{8, 16, 32, 64}` kümesinden seçilir veya ölçülen bölge sayısı ve/veya yayılımla belirlenen bulut karmaşıklığına göre dinamik seçilir.

**İstem 6:** İstem 1'deki sistem olup, okuyucu (130) INT8 kantizasyonla temsil edilir ve paritesi hedef dağıtım paketinde doğrulanır.

**İstem 7:** İstem 1'deki sistem olup, decoder (270) tam sözlük eşlemesi taşır ve hiçbir token/bucket kimliğini “other” kovasına atmaz.

**İstem 8:** İstem 1'deki sistem olup, karşı-faktürel modül (280) tek-nokta ve çift-nokta maskeleme/çıkarma deneylerini çalıştırır ve bunların sonucunu toplamsal katkı değil karşı-faktürel duyarlılık olarak raporlar.

**İstem 9:** İstem 1'deki sistem olup, nokta konumları için QUANN 0th-order daraltma ve okuyucu için backpropagation eğitimi uygulayarak hibrit bir eğitim rejimi kullanır; 0th-order adımın küçük ve sınırlı modellerle uygulanması tercih edilir.

**İstem 10 — Dil embodiment'i:** İstem 1'deki sistem olup, girdi birimleri token/alt-sözcük, okuyucu çıktısı bir-sonraki-token dağılımı ve yorumlama agglütinatif morfolojiye yönelik bulut/bölge incelemesidir; dil embodiment'inde ölçülen PPL değeri Bölüm 2.7.3'te kayıtlıdır.

Bu sonuçlar bir dikey embodiment kanıtıdır, genel performans garantisi değildir.

**İstem 11 — Fraud embodiment'i:** İstem 1'deki sistem olup, girdi birimleri kimlik-bucket işlem özellikleridir ve native açıklama leave-one-point-out metin, karşı-faktürel hatalı-uyarı denetimi ve self-contained edge paketini içerir; örnek ölçümde tek-maskeli audit %94–100 unflag, 543 KB ONNX, 1.5e-6 parite ve 45.5k işlem/s raporlanmış; matched PR-AUC 0.462 iken XGBoost 0.8047'dir.

**İstem 12 — AML embodiment'i:** İstem 1'deki sistem olup, yorumlama modülü isimli tipoloji prototiplerini ve geometrik gerekçeyi verir; Elliptic held-out örneğinde isimlendirme genel %75 ve illicit sınıfında %45, beş-seed kaba-etiket kararlılığı 0.325 ve SHAP top-1 kararlılığı 0.1025 (3.2×) olarak raporlanmıştır; bu kararlılık harita koordinatlarının kararlılığı anlamına gelmez.

**İstem 13 — Bakım embodiment'i:** İstem 1'deki sistem olup, girdi birimleri sensör penceresi/ölçümleri, açıklama bölge/prototip ve bölge-kayması/drift'tir; CWRU record-disjoint ikili ROC/PR 0.999/1.000 (GBT 1.000/1.000), C-MAPSS prototip açıklama kapsamı 1,064 pencerede %81.7, drift 2.4× (1.17%→2.81%) ve INT8 edge okuyucu 79 KB/5.07e-7 parite olarak raporlanmıştır; edge artifact ölçeklenmiş bulut girdisini alır, ham sensör girdisi değildir.

---

## 4. ÖZET (ABSTRACT)

Sistem, token, özellik kimliği veya pencere türü girdi birimlerini düşük boyutlu noktalar olarak temsil eder; nokta özellikleri ile mesafe/açı temelli çift-geometriyi simetrik küme agregasyonuyla birleştirir ve küme durumundan okuyucuyla görev skoru üretir. Aynı nokta bulutu, leave-one-point-out ablationın metne çözülmesi, bölge/prototip isimlendirmesi ve bölge-kayması/drift incelemesi için native açıklama nesnesidir. Okuyucu, gerekli kantizör, decoder ve manifest kompakt, self-contained ve doğrulanabilir pakette dağıtılabilir. Böylece aynı geometri hem hesaplama hem açıklama substratıdır. Dil, fraud, AML ve bakım embodiment'leri ortak çekirdeği farklı birim/isimlendirmelerle uygular. Doğruluk-üstünlüğü iddia edilmez; açıklama, geometriye dayalı denetim yüzeyidir ve nedensellik veya garanti anlamına gelmez.

---

## 5. ŞEKİLLER (Ayrı Sayfalar Olarak Sunulmalıdır)

TÜRKPATENT çizimleri, açıklama metninden ayrı sayfalar halinde, siyah-beyaz çizgi ve referans numaralarıyla hazırlanmalıdır.

| Sayfa | Şekil | İçerik | Kaynak / hazırlanma notu |
|---|---|---|---|
| 1 | Şekil 1 | Evrensel mimari ve modül akışı (100–290) | `fig/fig1_mimari.svg` |
| 2 | Şekil 2 | Çift-geometri ve simetrik birleştirme | `fig/fig2_ciftgeometri.svg` |
| 3 | Şekil 3 | Aynı buluttan skor, ablation, prototip ve drift | `fig/fig3_aciklama.svg` |
| 4 | Şekil 4 | AML tipoloji isimlendirme örnekleri | `fig/fig4a_aml_tipoloji.png`; `fig/fig4b_aml_tipoloji.png` |
| 5 | Şekil 5 | Fraud kimlik-bucket bulutu | `fig/fig5_fraud_bulut.png` |
| 6 | Şekil 6 | Bakım bulutu ve çalışma koşulu drift'i | `fig/fig6a_bakim_bolge.png`; `fig/fig6b_bakim_drift.png` |
| 7 | Şekil 7 | Self-contained dağıtım ve parite/manifest denetimi | Fraud deploy artifact'i ve maintenance edge örneği |

---

## 6. ATIF LİSTESİ (BİBLİYOGRAFİ)

Açıklamanın 2.2 bölümünde atıf verilen öncü teknik kaynaklar:

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I., 2017. *Attention Is All You Need.* Advances in Neural Information Processing Systems 30 (NeurIPS 30).
2. Qi, C., Yi, L., Su, H., & Guibas, L. J., 2017. *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* IEEE/CVF CVPR, 77–80.
3. Zaheer, M., Kottur, M., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., & Smola, A. J., 2017. *Deep Sets.* NeurIPS 30.
4. Lee, J., Lee, J., Kim, M., Kosiorek, A. R., Choi, S., & Teh, Y. W., 2019. *Set Transformer.* ICML, 3714–3723.
5. Mikolov, T., Chen, K., Corrado, G., & Dean, J., 2013. *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781.
6. Pennington, J., Socher, R., & Manning, C. D., 2014. *GloVe: Global Vectors for Word Representation.* EMNLP, 1532–1543.
7. Chung, S., & Abbott, L. F., 2021. *Geometry of a neural population.* Nature 592(7853), 117–122.
8. Hu, Z., Niu, L., & Varma, S., 2026. *Neural population geometry / point-cloud manifold analysis.* arXiv:2607.04525 [cs.CL]; DOI 10.48550/arXiv.2607.04525.
9. Xu, W., 2026. *Scale/PGA: Subspace Procrustes Geometry alignment analysis.* arXiv:2605.17084 [cs.LG]; DOI 10.48550/arXiv.2605.17084.

> **Atıf kapsamı notu:** 2.2'de açıkça belirtildiği gibi, "özgül birleşimin incelenen kaynaklarda gösterilmediği" ifadesi bu listedeki kaynaklarla sınırlıdır; küresel yokluk iddiası değildir. Resmi prior-art taraması (Espacenet, Google Patents, TÜRKPATENT, WIPO) vekil tarafından tamamlanmalıdır.

---

## 7. VEKİL DEĞERLENDİRMESİ (**DOSYAYA EKLENMEZ — strateji notu**)

### 7.1 Birlik (unity) ve bölünmüş başvuru

Dil, fraud, AML ve bakımın tek başvuruda birlikte sunulması “çok buluş”/birlik (unity) itirazı riski doğurur. Savunma, tek buluş fikrinin İstem 1'deki **dual-use geometri mekanizması** olmasıdır: aynı öğrenilmiş düşük boyutlu bulut okuyucuya skor verir ve native açıklamanın (ablation, prototip/bölge, drift) kaynağı olur; dikeyler bu ortak mekanizmanın uygulamalarıdır. Bağımsız sistem istemi dikeyden bağımsız, dört dikey ise ayrı bağımlı istemlerdedir. İncelemeci birlik itirazı yaparsa dikey embodiment'ları bölünmüş (divisional) başvurulara ayırma seçeneği değerlendirilebilir.

### 7.2 TR ulusal (dil) başvur ile ilişki / öncelik

Bu dosya evrensel dual-use çekirdeğini ifade eder ve iki özel dosyanın üst kümesidir: `../dil/patent_application_v2.3.md` (dil/sonraki-token uygulaması) ve `../tabular/patent_tabular_explain_v1.md` (tabular/fraud kimlik-bucket, açıklama ve edge uygulaması).

**PCT–TR öncelik ilişkisi (kritik):** TR ulusal başvurumuz `TURKPATENT_basvuru_v2.3.md` **dil** kapsamındadır. Bu PCT dosyasının kapsamı (dual-use çekirdek + dil/fraud/AML/bakım) TR dil başvurda **açıklanmamış** olduğundan, bu PCT başvurusu TR ulusal (dil) başvurudan **öncelik talep edemez** — iki bağımsız buluş/kapsam olarak değerlendirilmelidir. Önerilen strateji: **TR ulusal = dil** (öncelik T1) ve **PCT = evrensel + 4 dikey** (kendi önceliği T2); dosyalar arası metin bağlantısı kendiliğinden önceki tarih yaratmaz. Öncelik tarihi ancak usulüne uygun bir başvuruyla oluşur; hiçbir dosya yayımlanana kadar gizlilik uyarısı korunmalıdır.

### 7.3 Barizlik ve teknik etki (m.69)

- **PointNet ayrımı:** PointNet'te noktalar ham 3B sensör/görüntü girdisidir; burada nokta, farklı alan verilerinden öğrenilmiş temsildir.
- **Çift-geometri ayrımı:** Tercih edilen D+4 vektör — fark, mesafe, kosinüs ve iki norm — burada belirginleştirilmiştir; PointNet/Deep Sets'in genel simetrik havuzlaması bu birleşimi tek başına göstermemektedir.
- **Native açıklama ayrımı:** Aynı karar bulutundan ablation, metin/prototip ve drift incelemesi yapılır; bu, SHAP özellik listesinin aynısı olarak sunulmaz.
- **Teknik etki:** İstemler okunabilir native denetim yüzeyi ve self-contained, kompakt, doğrulanabilir dağıtım özelliklerine bağlanmıştır. m.69 bakımından açıklamadaki teknik etki, soyut matematiksel yöntem değil, okuyucu ve doğrulanabilir paketle çalışan bilgisayar sistemidir.

### 7.4 Kapsam ve doğruluk düzeltmeleri

- **C1 — Permütasyon:** İstem 1(a), İstem 2(c) ve 2.7.1 simetrik küme agregasyonunu sınırlar; bağlam/tarihçeden üretilen noktaların sıra-duyarlı olabileceğini açıklar.
- **C2 — Dönüşüm:** İstem 4 ve 2.7.1 fark vektörünü eşdeğişken; mesafe/kosinüs/norm bileşenlerini değişmez olarak ayırır.
- **C3 — Paket:** İstem 1(e), 2.4, 2.5 ve 2.7.2 sabit KB/parite iddiası yerine kompakt, self-contained, bit-kesin doğrulanabilir dağıtım ilkesini koyar; fraud ve bakım değerleri yalnız embodiment örnekleridir.
- **C4 — Prior art:** 2.2.3, incelenen kaynaklarda özgül birleşimin gösterilmediğini söyler; küresel “hiçbirinde yok” iddiası yapmaz ve daha geniş tarama gereğini kaydeder.

**Kanıt tutarlılığı notu:** Dil için 6.45/2.92 karşılaştırması v2.3'te ortak held-out diye etiketlense de, güncel `RESEARCH.md` 6.45'in eğitim kümesi örneklemi olduğunu ve strict held-out olmadığını bildirir. Başvuru öncesi bu protokol farklılığı çözülmeli; mevcut metin 6.45'i held-out kanıtı olarak ileri sürmemektedir.
