# AÇILIM PLANI — PCT v2 Tarifname: 112(a) Yazılı-Tanım Bulguları (47) için Paragraf Açılım Planı

- **Tarih:** görev 2026-11-19'da açıldı, plan 2026-11-20'de tamamlandı (oturum tarihi geçişi; her iki günkü okumalar bu dosyada künyelidir).
- **Bu bir PLANDIR.** `docs/patent/pct_v2/DESCRIPTION_EN.md` bu oturumda DEĞİŞTİRİLMEDİ; hiçbir paragraf metni yazılmadı — yalnız nereye hangi içeriğin eklenebileceği planlandı.
- **Girdiler:**
  - 47 bulgu listesi: `patent-cikti/uspto/pct_v2-uspto-inceleme/INCELEME_RAPORU.md` §3.2 (satır 55–67; `review_specification` çıktısı, kalıp: "Claim element X has limited description (only Para. N)", MPEP 2163, MEDIUM güven)
  - Tarifname: `docs/patent/pct_v2/DESCRIPTION_EN.md` (70 paragraf, [0001]–[0070])
  - İstemler: `docs/patent/pct_v2/CLAIMS_EN.md` (15 istem: 1–12 yöntem, 13–14 sistem, 15 CRM)
  - **Ölçüm kaynakları (yalnız ikisi):** `docs/patent/pct_v2/OLCUM_KAYITLARI.md` (OLC) ve `RESEARCH.md` satır 23 ile satır 107'de alıntılanmış log satırları. Başka kaynaklardan sayı alınmamıştır.
- **Dil kuralı:** Bu plan hiçbir yerde "bu açılım 112(a) itirazını giderir / destekler / kurtarır" demez; yalnız **"destek görünürlüğünü artırır"** der. Hukuki hüküm içermez.
- **Sayı kuralı:** Plan içi her ölçüm değeri `OLC:satır` veya `RESEARCH:satır → log:satır` atfını taşır; ham+türetilmiş birlikte yazılır. Tarifnamede zaten yazılı olmayan parametre aralıkları "kaydedilmemiş" olarak işaretlenir; uydurma aralık yoktur. "Kelime tahmini" değerleri plan yazarının tahminidir, ölçüm değildir.

## 0. Künye gösterimi

- `OLC:22` → `docs/patent/pct_v2/OLCUM_KAYITLARI.md:22` (satır, ham kaynak künyesini taşır; örn. `fraud/results/edge_eval_full_scale.json:23`).
- `RESEARCH:107 → logs/run_r8_pair_freeze_only.log:12` → değer RESEARCH.md satır 107'de bu log künyesiyle alıntılanmıştır; her iki satır (23 ve 107) bu oturumda `sed -n` ile yeniden okundu.
- INCELEME_RAPORU §3.2 bulguları `İstem → öğe ([00NN])` biçimiyle aynen aktarılmıştır.

## 1. Grup tablosu — 47 bulgu, 10 konu grubu

Bulgu sayıları INCELEME_RAPORU §3.2 tablosundan elle yeniden sayıldı: istem bazında 11+1+2+1+1+1+6+3+1+2+6+1+7 = **47** ✓; grup bazında aşağıdaki toplam da 47'yi verir (her bulgu tam bir grupta, çift sayım yok).

Öğe bazlı yeniden sayım (INCELEME_RAPORU §3.2 satır 55-67'den, her öğe-x-istem tekrarı bir bulgu): task input ×4 · point cloud ×5 · ordered sequence ×3 · tabular record ×3 · input order ×3 · point-in-D-space ×1 (hepsi istem 1) · plurality of sentences ×1 → inşâ kümesi 20 · reader ×6 · probability distribution ×1 · class assignment score ×1 → reader kümesi 8 · context-encoded representations ×3 · max/average-pooling sonucu ×2 · cloud centre ×1 → birleştirme kümesi 3 · Euclidean norm ×1 · context length ×1 · emerging region ×1 · sentence cloud state + fixed stride + multi-sentence representation ×3 · backprop + candidate position ×2 · computer system + processor + instructions ×3 + distribution store ×1 + same trained model ×1 → sistem/paket kümesi 5. Toplam (10 grup: G1 20 + G7 8 + G2 3 + G3 3 + G4 1 + G5 1 + G6 1 + G9 3 + G10 2 + G11 5) = 20+8+3+3+1+1+1+3+2+5 = **47 ✓**. (2026-11-20 düzeltme: formülde G3'ün 3'ü eksik yazılmıştı, grup kartlarıyla tamamlanarak düzeltildi; toplanan değer değişmedi.)

| Öncelik | Grup | Bulgu | Kapsadığı öğeler (istem) | Ek paragraf | Kelime tah. |
|---|---|---|---|---|---|
| P1 | G1 Görev girdisi + nokta-bulut inşası | 20 | task input (1,10,13,15) · point cloud (1,11,12,13,15) · ordered sequence (1,13,15) · tabular record (1,13,15) · input order (1,13,15) · point in a D-dimensional space (1) · plurality of sentences (10) | 2 | ~280 |
| P1 | G7 Reader başlığı + çıktı türleri | 8 | reader (1,6,9,10,14,15) · probability distribution of a next (1) · class assignment score (1) | 3 | ~320 |
| P1 | G9 Cümle/pencere hiyerarşisi | 3 | sentence cloud state (10) · fixed stride (10) · multi-sentence set representation (10) | 1 | ~150 |
| P2 | G2 Bağlam kodlayıcı + çıktısı | 3 | context-encoded representations (1,13,15) | 1 | ~130 |
| P2 | G3 Birleştirme / poolama + bulut merkezi | 3 | result of the max-pooling (4) · result of the average-pooling (4) · cloud centre (11) | 1 | ~110 |
| P2 | G10 Gradyansız eğitim + aday konum | 2 | reader by gradient-based backpropagation (12) · candidate point position (12) | 2 | ~260 |
| P2 | G11 Sistem çerçevesi + dağıtım paketi | 5 | computer system / at least one processor / instructions configures (13) · distribution store holding (14) · same trained model (1) | 1 | ~110 |
| P3 | G4 Çift-geometri özellikleri | 1 | Euclidean norm of the difference vector (3) | 1 | ~90 |
| P3 | G5 Maskeleme + bağlam uzunluğu | 1 | context length of the task input (5) | 1 | ~90 |
| P3 | G6 Bölgesel kayma / emerging region | 1 | emerging region (8) | 1 | ~90 |
| **Toplam** | | **47** | | **14** | **~1630** |

**Tek açılımın çok istemli kapsama notu (özet, bkz. §3):** sistem (13, 14) ve CRM (15) istemlerindeki tekrarlar, 10 numarasındaki bağımlı istemin cümle/pencere katmanları ve bağımsız yöntem istemi 1'in öğeleriyle aynı tanım kümeleridir; G1/G2/G7/G9/G11 blokları (5 blok) tek yerden yazılıp 3 istem tipini (yöntem/sistem/CRM) + istem 10'u birlikte besler. Tarifnameye mükerrer paragraf önerilmez.

## 2. AÇILIM PARAGRAFI PLANI — grup kartları

Not: Paragraf numaraları 4'er artar; ekleme sonrası numaralar ve çapraz referanslar ([0029]'un [0065]'e atfı gibi) topluca kayar. Yeniden numaralandırma `make_filing.py` ile mekanik üretilir (DESCRIPTION_EN.md:7-9 künye notu); elle numara yazılmaz. Bir alternatif: yeni numaralandırma yerine mevcut paragrafların genişletilmesi — numaralar kaymaz, ama araç [00NN] başına tek kaynak sezgisiyle aynı bulguyu yeniden üretebilir.

### G1 — Görev girdisi + nokta-bulut inşası (20 bulgu · P1 · istem 1,10,11,12,13,15)

- **Öğeler (bulgu sayısı):** task input ×4 ([0065]) · point cloud ×5 ([0020]) · ordered sequence of ×3 ([0010]) · tabular record ×3 ([0010]) · input order of the points ×3 ([0010]) · point in a D-dimensional space ×1 ([0010]) · plurality of sentences ×1 ([0065]) — araç hepsini TEK paragraf kaynağıyla işaretlemiş.
- **Mevcut dayanaklar:** [0010] (DESCRIPTION_EN.md:41) — girdi iki biçimden biri, her bağlam-temsilinin D-uzayı noktası, N noktalı bulut, sıradan bağımsız toplama. [0020] (:69) — aslında Şekil 1 çizim tarifi; araç "point cloud"u buradan türettebiliyor. [0031] (:95) — iki girdi biçimi fiilen burada anlatılıyor ama araç "task input"u [0065]'e indirgemiş: destek izi paragrafa yayılmamış. [0032] (:97) — kayıt değerleri (D=8/32/64/48; 221 girdi kimliği; K=13).
- **Önerilen ek konum:** [0031] sonrasına 2 paragraf (~140+~140 kelime).
- **Anlatılacaklar (madde):** (1) "task input" tanımı: sıralı token dizisi neyle kurulur — alt-sözcük birimleri/vocab (vocab=10000 [0060]'da yazılı; kayıt: OLC:156 → logs/run_r5_cc_stable.log:6); tabular kaydın ölçülen alan değerleri (8 alanlı işlem örneği [0054]'te yazılı). (2) "input order" teriminin tanımı: pozisyon numaralandırması; toplamanın bundan bağımsızlığının ne anlama geldiği (permütasyon sonucu değiştirmez). (3) "point cloud" tanımı: sırası önemsiz N noktalı küme; tabular gövdede noktalar=alanlar; dil gövdesinde noktalar=bağlam konumları. (4) "tabular record" örnek künyesi: ölçülen alan değeri (işlem tutarı, sensör okuması gibi — [0002]'deki uygulama alanlarıyla tutarlı; yeni alan tipi icat edilmez). (5) "point in a D-dimensional space": noktanın konum-vektörü olarak tanımı ve D'nin [0032]'deki kayıtlı değerlerine tek atıf cümlesi.
- **Parametre aralıkları:** D aralığı 3–64 [0012]'de YAZILI; kayıt değerleri [0032]'de YAZILI; token başına bulut dışı aralıklar KAYDEDİLMEMİŞ — paragraf "kaydedilmemiş" diliyle yazılacak, aralık uydurulmayacak.
- **Ölçülmüş teknik etki:** gerekmez; sayı eklemeyi gerektiren bir öğe değil (inşâ tanımı).

### G2 — Bağlam kodlayıcı + context-encoded representations (3 bulgu · P2 · istem 1,13,15)

- **Öğeler:** context-encoded representations ×3 ([0010]).
- **Mevcut dayanaklar:** [0010] (:41) — kodlayıcı görev çıktısı üretir; [0031] (:95) — "her konumdaki temsil D-uzayına izdüşürülür"; [0037] (:115) — cümle/pencere katmanlı kodlayıcı varyantı; [0036] (:111) — "bulut durumu bulut kodlayıcısından alınır".
- **Önerilen ek konum:** [0031] sonrasına 1 paragraf (~130 kelime) — G1 bloğuyla aynı bölgeye, arka arkaya (G1 paragraflarından sonra).
- **Anlatılacaklar:** (1) "context-encoded representation"ın tanımı: kodlayıcının girdi birimine ATFI (token→birim, alan→kimlik), temsilin baglam-bilgili olduğunu belirten işleyiş cümlesi. (2) İzdiüşüm adımı: temsil → D-boyutlu nokta ([0033]'ün 3D+2 çift genişliğiyle tutarlı; OLC:167 → cloudlm/model.py:63-64). (3) Alternatif embodiment dili: [0037]'deki üç katmanlı (cümle/pencere) kodlayıcıya atıf — KAYNAKTA OLAN tek alternatif budur; başka kodlayıcı tipi (mimari adı vb.) kaynaklarda kayıtlı olmadığı için YAZILMAZ. (4) Kayıtlı model ölçeği cümlesi: dil gövdesi zincirinin parametresi 20,999,648 (OLC:153 → logs/run_compare.log:13-14) — "kayıtlı örnek" olarak; encoder'a tek başına atfedilmez.
- **Parametre aralıkları:** encoder-özel aralıklar KAYDEDİLMEMİŞ.
- **Ölçülmüş teknik etki:** sayı eklemek zorunlu değil; ölçek künyesi yeter.

### G3 — Simetrik birleştirme / poolama + bulut merkezi (3 bulgu · P2 · istem 4, 11)

- **Öğeler:** result of the max-pooling ([0034]) · result of the average-pooling ([0034]) · cloud centre ([0038]'in küresel-istatistik varyantı; araç istem 11'de tek paragraf olarak [0038]'i gösteriyor mu — INCELEME_RAPORU satır 63'te `cloud centre ([0033])` yazılı: [0033]'teki "cloud centre" ifadesiyle eşleşmiş; plan her iki pasajı da dayanak sayar).
- **Mevcut dayanaklar:** [0034] (:105) — iki yol: nokta yolunda max-pooling, çift yolunda ortalama-pooling, sonuçlar birleştirme (concatenation) katmanında toplanır. [0035] (:107) — maskeleme her iki yolda. [0030] (:91) — sıralamanın toplamaya taşınmadığı. [0038] (:119) — cloud centre + centre deviation küresel istatistik olarak tanımlı (OLC:178 → purecloudlm/model.py:63). Şekil 3 tarifi [0022] (:73).
- **Önerilen ek konum:** [0034] sonrasına 1 paragraf (~110 kelime).
- **Anlatılacaklar:** (1) "result of the max-pooling"in tanımı: nokta boyutları üzerinden her boyutta en büyük değerin seçilmesi, sonuç vektörün uzunluğunun nokta sayısından bağımsız olması; "result of the average-pooling"in tanımı: geçerli çiftler üzerinden boyut-vari ortalama ([0035]'teki geçerli-çifte-göre-normalizasyonla tutarlı). (2) Sırasallık sonucu cümlesi: iki toplama da küme işlemi olduğundan permütasyon değişmezliği — [0030]/[0035]'teki ifadenin POOLAMA SONUÇLARINA görünür bağlanması (araç izi burada tek paragrafta kalıyor). (3) Birleştirme sonrası temsilin reader'a reader-girdisi olarak geçtiğinin [0010]'a atıfla yeniden söylenmesi (kısa atıf cümlesi, mükerrer tanım değil). (4) "cloud centre" (istem 11): [0038]'in küresel-istatistik varyantındaki tanımıyla birlikte tek cümle — bulut noktalarının ortalaması olarak merkezin hesap tanımı; centre deviation ile ilişkisi [0038]'de yazılı; sayısal aralık KAYDEDİLMEMİŞ.
- **Parametre aralıkları:** birleşik temsil genişliği (concat+fusion sonrası) OLC kaynaklarında KAYITLI DEĞİL (OLC:174 → cloudlm/model.py:179 yalnız varlık künyesi taşır) — paragraf genişlik sayısı UYDURMADAN yazılacak.
- **Ölçülmüş teknik etki:** gerekmez.

### G4 — Çift-geometri özellikleri / Euclidean norm (1 bulgu · P3 · istem 3)

- **Öğeler:** Euclidean norm of the difference vector ([0033]).
- **Mevcut dayanak:** [0033] (:101) — beş bileşenli çift-geometri özelliği; mesafe = fark vektörünün ÖKLİD NORMU; 3D+2 genişliği. Şekil 2 tarifi [0021] (:71). [0012] (:45) — kısa tekrar.
- **Önerilen ek konum:** [0033] sonrasına 1 paragraf (~90 kelime).
- **Anlatılacaklar:** (1) Norm tanımının açılması: iki nokta konumunun koordinat farklarının kareleri toplamının karekökü; "pair distance" ile aynı nicelik olduğunun açık cümleyle söylenmesi. (2) Beş bileşenin her birinin hangi geometrik bilgiyi taşıdığının tek-şekli cümlelerle [0033]'teki ayrımı tekrarlamadan pekiştirilmesi (göreceli konum / açısal bağıntı / merkeze göre mutlak konum). (3) Farklı norm seçenekleri (örn. Manhattan) KAYNAKLARDA KAYDEDİLMEMİŞ — alternatif embodiment dili kurulmayacak.
- **Ölçülmüş teknik etki:** gerekmez; [0039] (MAC ölçeklemesi) çift yolunun maliyetini zaten ölçüyle anlatıyor — paragraf oraya atıf cümlesi ekleyebilir (sayı yeniden yazılmaz).

### G5 — Maskeleme + bağlam uzunluğu / N sınırı (1 bulgu · P3 · istem 5)

- **Öğeler:** context length of the task input ([0039]).
- **Mevcut dayanak:** [0039] (:123) — "N is therefore bounded by the context length of the task input" + MAC ölçekleme ölçümü. [0031] (:95) — N = bağlam uzunluğu (dil) / alan sayısı (tabular). [0035] (:107) — dolu olmayan konumların maskelenmesi. [0012] (:45) — maske sinyali.
- **Önerilen ek konum:** [0035] sonrasına 1 paragraf (~90 kelime).
- **Anlatılacaklar:** (1) "context length" tanımı: girdinin birim sayısı (dizide token, kayıtta alan) ve görev yapilandırmasınca belirlenen en büyük dolupozmeme sınırı; dolu-olmayan konum kavramının [0035]'le sözlük bağı. (2) Kayıtlı örnekler TARİFNAMEDEN aktarımla: seq_len 10 ([0058], :185), seq_len 8 ([0066], :209), K=13 ve 221 kimlik/214 konum ([0032], :97) — yeni sayı yok, yalnız bu örneklerin "N bağlam uzunluğuyla sınırlıdır" kuralına bağlandığının görünür cümlesi. (3) Eşik/maske sayısal değeri KAYDEDİLMEMİŞ.
- **Ölçülmüş teknik etki:** [0039]'taki 2522200 → 34091488 MAC ölçeği (OLC:49-50) ZATEN tarifnamede; yeni paragraf yalnız atıf yapar, sayı çoğaltılmaz.

### G6 — Bölgesel kayma / emerging region (1 bulgu · P3 · istem 8)

- **Öğeler:** emerging region ([0013]).
- **Mevcut dayanak:** [0013] (:47) — kısa özet (özet bölümündeki tek tekrar). [0046] (:145) — fiili tanım: alt/üst eşik koşulu + TV/KL ölçütleri. [0067] (:213) — "işaret ölçülmüş sürüklenmede olay olarak kaydedilmedi, ölçüt olarak iddia ediliyor" sınır notu (OLC:201 → docs/patent/final_v3/TARIFNAME.md:227).
- **Önerilen ek konum:** [0046] sonrasına 1 paragraf (~90 kelime).
- **Anlatılacaklar:** (1) emerging-region işaretinin denetim raporunda NASIL taşındığı: bölge dizini + bölge etiketi + referes/karşılaştırma payları çifti ([0045]'in "etiket tek başına kimlik değildir" kuralıyla birlikte). (2) Eşiklerin görev-yapilandırma parametresi olduğu; sayısal eşik değerleri KAYDEDİLMEMİŞ — uydurulmayacak. (3) Ölçülen haftalık TV/KL değerleri ([0047], :149) ve uç pay farkları ([0048], :151) zaten tarifnamede: paragraf yalnız işaret-üretimi işleyişini anlatır, sayı tekrar etmez. (4) [0067]'deki "ölçüt olarak iddia" sınırının işleyiş paragrafına bağlanması (vekilin 112(b) netlik notuyla uyumlu görünür iz).

### G7 — Reader başlığı + çıktı türleri (8 bulgu · P1 · istem 1,6,9,10,14,15)

- **Öğeler:** reader ×6 — istem 1, 6, 9, 10, 14, 15'te · probability distribution of a next ×1 (istem 1) · class assignment score ×1 (istem 1). Araç hepsini [0036]'ya indirgemiş; INCELEME_RAPORU satır 69'un aktarım notu: [0029]-[0031], [0053] gibi pasajların aracın terim-eşlemesince görülmüyor olabileceği vekil incelemesi ister.
- **Mevcut dayanaklar:** [0036] (:111) — dil görevinde bir-sonraki-token olasılık dağılımı, maskla tanımlı ilerleyiş geçişi, bulut durumu; tabular görevde sınıf atama skoru, açıklama adımı aynı reader'ın logitinde. [0029] (:89) — zincirde reader konumu. [0041] (:129) — INT8 oturumunda toplu yeniden-koşum. [0050] (:159) — reader'ın INT8 katmanları.
- **Önerilen ek konum:** [0036] sonrasına 3 paragraf (~120+~110+~90 kelime). TEK YAZIM; bağımlı istemler 6/9 ve sistem/CRM istemleri 13/14/15 ile istem 10'un reader'larını da bu blok besler (§3).
- **Anlatılacaklar — p1 (yapı/işleyiş):** set-temsilinin reader-girdisi olarak okunuşu adım adım; kayıtta tanımlı bileşen künyeleri atıfla (concat+fusion OLC:174 → cloudlm/model.py:179; maskla tanımlı ilerleyiş OLC:175 → :305; bulut durumu OLC:176 → :313); "reader" teriminin tabular ve dil gövdelerinde AYNI yapısal öğe olduğunun açık cümlesi.
- **Anlatılacaklar — p2 (iki görev başlığı):** olasılık dağılımı çıktısının anlamı (vocab kimlikleri üzerinden dağılım; vocab=10000 kayıtlı örnek [0060]'da yazılı, OLC:156); sınıf atama skorunun anlamı (skor→sınıf karar sınırı; fraud skoru örneği [0054]'te yazılı); açıklamanın aynı logit üzerinden kurulduğunun [0036]-sonrası görünür tekrani (mükerrer tanım değil — işleyiş adımı diliyle).
- **Anlatılacaklar — p3 (kayıtlı örnek künyeleri + ölçülmüş etki):** INT8 eşdeğerliliği ve hız künyeleri — paket öz-testi: 19963 satırda FP32'ye karşı en büyük mutlak fark 1.2814998626708984e-06 (OLC:63 → fraud/results/deploy/manifest.json:28) ve öz-test verimi 26473.303851652894 işlem/s ≈ 26,5 bin işlem/s (OLC:65 → manifest.json:30); INT8 reader gecikmesi p50 0.015912843984551728 ms ≈ 15,9 µs/işlem, n_tx=2000 (OLC:22,25 → fraud/results/edge_eval_full_scale.json:23,21). Bu değerlerin bir kısmı [0051]/[0017]'de zaten var; paragraf reader-başlığı bağlamında TEK künyeyle bağlar, tablo çoğaltılmaz.
- **Parametre aralıkları:** reader katman sayısı/genişliği KAYDEDİLMEMİŞ (OLC:67 → manifest.json:9-14 yalnız D/d_model/d_cloud/vocab/points/K taşır — paragrafa en fazla bu altı künye yazılabilir).

### G9 — Cümle/pencere hiyerarşisi (3 bulgu · P1 · istem 10)

- **Öğeler:** sentence cloud state ([0065]) · fixed stride ([0037]) · multi-sentence set representation ([0037]).
- **Mevcut dayanaklar:** [0037] (:115) — üç katman; pencere boyutu 5 ve adım 2 kayıtta tanımlı; [0065] (:207) — ikinci varyant olarak kısa tekrar; [0060] (:191) — ölçülmüş eğitim örneği (80/10/10, seed 42; 159888/19986/19986; test yitimi 3.2318); Şekil 8 tarifi [0027] (:83).
- **Önerilen ek konum:** [0037] sonrasına 1 paragraf (~150 kelime). İstem 10'un ALTIS bulgusu (plurality of sentences G1'de) bu blok + G1 ile kapanır; [0065]'e ikinci tarif yazılmaz.
- **Anlatılacaklar:** (1) Cümle sınırının belirlenişi (noktalama-tabanlı bölmeleme — KAYNAKTA TANIMLI DEĞİLSE "metni cümlelere bölümek" düzeyinde genel dil; algoritma icat edilmez). (2) pencere–adım kayması işleyişinin adım adımı anlatımı ve komşu pencerelerin örtüşmesinin ne ürettiği; kayıt değerleri [0037]'de YAZILI (5/2; OLC:159 → contextcloud/model.py:127-128). (3) "multi-sentence set representation"ın tanım cümlesi: pencere-geometri kodlayıcısının küme-temsil çıktısı → reader (OLC:160 → model.py:196,246 sınıf künyeleri). (4) metindeki cümle sayısı / pencere sayısı aralıkları KAYDEDİLMEMİŞ; kayıtlı bölme sayıları [0060]'ta zaten var, tekrar yazılmaz.
- **Ölçülmüş teknik etki:** eğitim örneği [0060]'ta mevcut (OLC:155-157). RESEARCH:107'deki R8 ölçümü (aşağı G10) hiyerarşi değil eğitim mekaniği ölçümüdür — G9'a değil G10'a yazılır.

### G10 — Gradyansız eğitim + aday konum / çift-donanım (2 bulgu · P2 · istem 12)

- **Öğeler:** reader by gradient-based backpropagation ([0015]) · candidate point position ([0015]).
- **Mevcut dayanaklar:** [0015] (:51) — özet düzeyinde; [0053] (:169) — fiili iki bölümlü eğitim: reader ağırlıkları gradyan-geri-yayılımla, nokta konumları 0. mertebe kabul/geri-al ile; aday→aynı mini-baçta yitim ölçümü→eşik→geri alma→aralık daraltma. [0019] (:63) — aynı işleyişin DISCLOSURE tekrarı. [0065]/[0066] (:207,:209) — üçüncü varyant + küçük ölçek ölçümü (1045.03 / 563.45 / 1016.30 PPL; OLC:191).
- **Önerilen ek konum:** [0053] sonrasına 2 paragraf (~140+~120 kelime).
- **Anlatılacaklar — p1 (tanım + döngü):** "gradient-based backpropagation" ve "gradientsiz" terimlerinin 1-şekli tanımları; 0. mertebe güncelleme DÖNGÜSÜNÜN adım adım işleyişi (aday konum üretimi → adayla yitim ölçümü → eşik karşılaştırması → kabul ya da geri alma → arama aralığının daraltılması) — adımların her biri [0053]'te var; paragraf onları GÖRÜNÜR işleyiş numarasına dönüştürür (OLC:185-187 → quann_cloud/src/qcloud_quann.py:212,220,223,226). Aday dağılımı/eps değerleri KAYDEDİLMEMİŞ.
- **Anlatılacaklar — p2 (çift-donanım varyantı + ölçülmüş etki):** araç "pair/freeze" kelimesini BULGULARDA kullanmıyor; plan, [0065]'in üçüncü varyantına EK bir çalışma örneği olarak R8 ölçümünü önerir. Yalnız künyeli alıntı: çift-kanalı eğitimli varyantta ayri-koşumda çift-kanalı 86080 parametre seçili ve heldout_PPL 5233.4086 (RESEARCH:23 ve RESEARCH:107 → logs/run_r8_pair_freeze_only.log:11,12); eşdeğer protokollü kontrol kolunda heldout_PPL 19770.644 (RESEARCH:23 → logs/run_r8_quann_pair.log:30); protokoller argüman düzeyinde birebir (RESEARCH:23 → training/shell/run_r8_quann_pair.sh:37,43). SINIR kaydı paragrafa AYNEN işlenmeli: tek seed, çift koşum yok, tekrar adayı (RESEARCH:107). "İyileştirir/azaltır" hükmü yazılmaz — iki künyeli değer ve sınır cümlesi yan yana durur.
- **Parametre aralıkları:** eşik/eps, aday yarıçapı, öğrenme oranı KAYDEDİLMEMİŞ.

### G11 — Sistem çerçevesi + dağıtım paketi (5 bulgu · P2 · istem 1, 13, 14)

- **Öğeler:** computer system ([0016]) · at least one processor ([0016]) · instructions configures the system ([0016]) · distribution store holding ([0050]) · same trained model ([0011]).
- **Mevcut dayanaklar:** [0016] (:53) — sistem modülleri (kodlayıcı, izdüşürme, çift-kodlayıcı, toplama, reader, yorumlama motoru, arayüz) + CRM adımları tek paragrafta; istem 13'ün üç çerçeve öğesi YALNIZ buraya izlenmiş. [0050] (:159) — paket üçlüsü: kantize ağırlıklar + nicelleştirme tablosu + çözümleme tablosu; boyut künyeleri yazılı (548.4/5.4/16.5 KB; 221 girdi). [0051] (:161) — paket uçta hem görev çıktısını hem insan-okur açıklama üretir. [0011] (:43) — "same set representation of the same trained model" ifadesinin DISCLOSURE'daki tek yeri.
- **Önerilen ek konum:** [0016] sonrasına yarım-Paragraf AÇILIMI önerilmez — bunun yerine BEST MODE tarafında [0050] ve [0051] ARASINA 1 paragraf (~110 kelime); çerçeve öğeleri için ayrıca [0016]'nın BEST MODE karşılığı [0029]'a tek cümle atıf: işlemci/bellek/talimat eşlemesi.
- **Anlatılacaklar:** (1) "distribution store"ın fiziksel karşılığı: tek dosya/küme halinde saklanan üç yapı; bellekte okunuş sırası (ham satır → kimlik → etiket eşlemesi [0050]'de yazılı; OLC:66 → manifest.json:97-98 tanım künyesi). (2) "same trained model" ifadesinin BEST MODE'a görünür inmesi: aynı ağırlık kümesinin hem görev çıktısını hem üç açıklama işlemini ürettiğinin işleyiş cümlesi — [0030]'un "açıklama ikinci sistemi değildir" ifadesine ATIFLA (mükerrer tanım yazılmaz). (3) "computer system / at least one processor / instructions configures the system": [0029]'un zincir tarifiyle eşleme cümlesi — her modülün (kodlayıcı, izdüşürme, toplama, reader, yorumlama, arayüz) talimat-seti tarafından yapılandırılan DONANIM bloğu olduğunun tek cümlelik görünür ifadesi; donanım-marka/saya aralıkları KAYDEDİLMEMİŞ. (4) Kayıtlı paket boyutları [0050]'de zaten var; paragraf sayı ÇOĞALTMAZ, en fazla öz-test künyelerine atıf yapar (OLC:62-64 → manifest.json:27-28).

## 3. Çakışan tekrarlar — mükerrer paragraf ÖNERİLMESİN

- **İstem 10/13/15 toplami:** INCELEME_RAPORU §3.2 tablosundan yeniden sayım (satır 62, 65, 67): istem 10 = 6 · istem 13 = 9 · istem 15 = 7 → **22 bulgu**. (Görev özeti "23 tekrar" demişti; kaynak tablodaki sayım 22 veriyor — farkın sebebi doğrulanmadı, kaynak sayım esas alındı.) Bu 22'nin dağılımı: G1 = 12 (istem 10: task input + plurality of sentences; istem 13/15'er: task input, ordered sequence, tabular record, point cloud, input order) · G2 = 2 (context-encoded, istem 13/15) · G7 = 2 (reader, istem 10/15) · G9 = 3 (istem 10'un hiyerarşi öğeleri) · G11 = 3 (istem 13'ün çerçeve öğeleri). Toplam 12+2+2+3+3 = 22 ✓. Sonuç: **5 tanım bloğu (G1, G2, G7, G9, G11) tek yerden yazıldığında yöntem (1,10), sistem (13,14) ve CRM (15) istemlerinin aynı-öğe bulguları birlikte ele alınır.** BEST MODE'na istem-tipi başlığıyla ("system embodiment", "CRM embodiment" diye) ikinci kopya paragraf EKLEMEYİN — PCT/Aşama-II tekrarı hem hacim hem tutarlılık riski taşır; yeni paragraflara kısa karşılıklı atıf cümlesi yeter.
- **Araç kapsam notu (INCELEME_RAPORU satır 69'dan aktarım):** ör. `reader` için yalnız [0036] işaretlenmişken [0029]-[0031], [0053] gibi pasajlar da konuyu anlatıyor; aracın terim-eşlemesinin kaçırıp kaçırmadığı vekil incelemesi ister. Bu plan o gözlemi "mükerrer içerik var" kabulüyle değil, "destek İZİNİ aynı paragraflar içinde görünür kılma" ilkesiyle ele alır: ek paragraflar mevcut pasajlara terim-atıf cümleleri bağlar, aynı tanımı yeniden yazmaz.
- **point cloud ×5 bulgusu** ([0020]) tek açılımla 5 öğeyi (istem 1,11,12,13,15) karşılar; [0020] bir ÇİZİM tarifidir — asıl tanım [0031]-sonrası bloğa taşınır, çizim tarifi değişmez.

## 4. Öncelik sırası

| Sıra | Grup | Gerekçe |
|---|---|---|
| P1 | G1, G7, G9 | G1: üç bağımsız istemin (1,13,15) OMURGASAL öğeleri + istem 10 (20 bulgu, en yüksek tekrar); G7: `reader` üç bağımsız istemde + 6/9/14'te geçen tek terim + istem 1'in iki çıktı türü (8 bulgu); G9: bağımsız yöntem istemi 10'un TAMAMI tek öğe grubuna dayanıyor |
| P2 | G2, G3, G10, G11 | G2/G3: istem 1 zincirinin bağımlı öğeleri (G2 aynı anda 13/15'i de besler); G10: istem 12'nin bağımsız-mekanizma öğeleri (R8 ölçüm adayı burada); G11: istem 13'ün üç çerçeve öğesi + istem 14 + istem 1'in dual-use ifadesi |
| P3 | G4, G5, G6 | Yalnız tekil bağımlı istemlere dokunur (3, 5, 8) |

## 5. Bulgu → grup kapsama haritası (47/47 kontrol)

| Bulgu kümesi (istem · sayı) | Grup karşılığı |
|---|---|
| 1 · 11 | G1 (6: task input, ordered seq., tabular record, point cloud, input order, point-in-D-space) · G2 (1: context-encoded repr.) · G7 (3: reader, probability distribution, class assignment score) · G11 (1: same trained model) |
| 3 · 1 | G4 (1) |
| 4 · 2 | G3 (2) |
| 5 · 1 | G5 (1) |
| 6 · 1 | G7 (1) |
| 8 · 1 | G6 (1) |
| 9 · 1 | G7 (1) |
| 10 · 6 | G1 (2: task input, plurality of sentences) · G9 (3: sentence cloud state, fixed stride, multi-sentence representation) · G7 (1: reader) |
| 11 · 2 | G1 (1: point cloud) · G3 (1: cloud centre) |
| 12 · 3 | G10 (2: backprop, candidate position) · G1 (1: point cloud) |
| 13 · 9 | G1 (5: task input, ordered, tabular, point cloud, input order) · G2 (1: context-encoded) · G11 (3: computer system, processor, instructions) |
| 14 · 2 | G11 (1: distribution store) · G7 (1: reader) |
| 15 · 7 | G1 (5) · G2 (1: context-encoded) · G7 (1: reader) |

Sütun toplamları: G1 6+2+1+1+5+5 = 20 ✓ · G2 1+1+1 = 3 ✓ · G3 2+1 = 3 ✓ · G4 1 ✓ · G5 1 ✓ · G6 1 ✓ · G7 3+1+1+1+1+1 = 8 ✓ · G9 3 ✓ · G10 2 ✓ · G11 1+3+1 = 5 ✓. Satır toplamları 11+1+2+1+1+1+1+6+2+3+9+2+7 = **47 ✓**; hiçbir bulgu iki grupta sayılmadı.

## 6. Toplam bütçe (yalnız bu planın kendi satırlarından toplanmıştır)

- **Önerilen yeni paragraf sayısı:** 14 (G1:2 · G2:1 · G3:1 · G4:1 · G5:1 · G6:1 · G7:3 · G9:1 · G10:2 · G11:1) — §1 tablosu satır toplamiyla aynı.
- **Tahmini ek kelime:** ~1630 (280+130+110+90+90+90+320+150+260+110 — §1 tablosundan; tahmindir, ölçüm değil).
- **Tarifnameye etkisi:** 70 paragraf → 84; ~40.000 karakterlik metne ~%10-12 hacim artışı beklentisi (kelime tahmininden kaba türetme; doğrulanmadı).
- Bu plan biç bulgularını (INCELEME_RAPORU §3.3: eksik SUMMARY / DETAILED DESCRIPTION başlıkları) KAPSAMAZ — başlık eşlemesi ayrı iş, vekil kararı.

## 7. Notlar ve uyarılar

1. Ölçüm değerlerinin hiçbiri bu planda uydurma değildir; her biri OLC satırı veya RESEARCH satır 23/107 alıntısıdır. RESEARCH:23/107'deki "tek seed / tekrar adayı / sebep doğrulanmadı" sınır kayıtları, ilgili değerler tarifnameye taşınırsa AYNEN taşınmalıdır (örüntü: [0059]'in sınır cümlesi).
2. "Açılım destek görünürlüğünü artırır" — 112(a) yazılı-tanım değerlendirmesi ve enablement yorumu denetçi/vekil işidir; bu plan hiçbir bulgu için "itiraz gider" hükmü kurmaz.
3. Uygulama sırası önerisi: P1 kartları vekile tek tek onaylat → tek koşuda tüm eklemeler + `make_filing.py` ile yeniden numaralandırma → `review_specification` yeniden koşusu (araç bazlı görünmezlik kontrolü; hukuki sonuç değil).
4. **Vekil onayı:** Plan, paragraf metinleri ve öncelik sırası patent vekilinin onayı olmadan başvuru metnine dönüştürülmemelidir.

---
*Kanıt — dosya listesi (tüm yazım/düzeltme turları bittikten sonra, 2026-11-20 00:12 yerel):*

```
total 96
drwxr-xr-x  4 serkankose  staff    128 Nov 20 00:11 .
drwxr-xr-x  6 serkankose  staff    192 Nov 19 23:14 ..
-rw-r--r--  1 serkankose  staff  30404 Nov 20 00:11 ACILIM_PLANI.md
-rw-r--r--  1 serkankose  staff  12327 Nov 19 23:22 INCELEME_RAPORU.md
```

*Kaynak dosyaların dokunulmadığı kanıtı (`ls -la`, mtime'lar bu oturumdan ÖNCEKİ):*

```
-rw-r--r--  1 serkankose  staff  11426 Nov 19 11:46 docs/patent/pct_v2/CLAIMS_EN.md
-rw-r--r--  1 serkankose  staff  39987 Nov 19 11:46 docs/patent/pct_v2/DESCRIPTION_EN.md
-rw-r--r--  1 serkankose  staff  17268 Nov 19 15:09 docs/patent/pct_v2/OLCUM_KAYITLARI.md
```
