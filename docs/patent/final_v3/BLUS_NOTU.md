# BLUS NOTU — Patent v3 (nihai, bağımsız hazırlık) — 2026-11-18

Amaç: önceki v2.x metinlerine bağlı OLMAYAN, yalnız ölçülmüş ve kodla doğrulanmış
kanıta dayanan yeni başvuru paketi. Bu klasör nihai çalışma alanıdır:
`docs/patent/final_v3/` (istemler, tarifname, özet, figürler, build scripti).

## 1. Buluşun çekirdeği (en iyi nokta)

**Çift-kullanımlı (dual-use) nokta-bulutu geometri çekirdeği ve onunla aynı
temsil üzerinden çalışan gömülü açıklanabilirlik üçlüsü.**
Görev çıktısı ile açıklama AYNI nokta bulutundan üretilir; ayrıca model tek
INT8-kantize edilmiş self-contained pakete sıkıştırılabilir. Bu üçlü önceki
dosyalarda vardı ama dağınıktı; burada bağımsız istemin OMURGASI olur:

1. **Nokta çıkarma etkisi** — bir nokta buluttan çıkarılır, okuyucu yeniden
   koşar, görev çıktısındaki değişim ölçülür (`fraud/src/explain_pilot.py:93-112`).
2. **Bölge/prototip isimlendirme** — bulut kümesine KMeans, bölgelere insan
   okunur etiket (`aml/src/e1_explain.py:296-342`; `complaints/src/explain_region.py:47-71`).
3. **Bölge kayması (drift)** — referans vs karşılaştırma veri grubunda bölge
   nokta paylarının TV/KL karşılaştırması (`complaints/src/audit_drift.py:36-45`;
   `maintenance/src/e1_real.py:288-303`).

Önceki teknik farkı: SHAP/lobby-type açıklama model-SONRASI ayrı bir araçtır;
burada açıklama, modelin kendi geometrik temsilinin (D+4 çift-geometri +
simetrik agregasyon) doğal çıktısıdır — ayrı model, ayrı girdi gerektirmez.

## 2. Embodiment'lar (yalnız ölçülmüş, etiketi doğru modeller)

| Embodiment | Model/çalışma | Kanıt (doğrulanmış) |
|---|---|---|
| Dil | cloudlm (CloudLM, D64) | HELD-OUT PPL 61.19 (D32 61.94, R4 duel: sızdırmaz satırlar [200000,202000), ortak BPE, seq_len=10) — `logs/run_r4_resume.log:67,70,71`. Dürüstlük notu: başka protokolde causal baseline 2.94 vs CloudLM 6.45 (`logs/run_compare.log:16-18`) — tarifnamede "doğruluk üstünlüğü" iddiası YOK; iddia temsil+birlikte-açıklama. |
| Çoklu-cümle bağlam (YENİ) | contextcloud stable | TEST loss 3.2318, ayrık 80/10/10 holdout seed=42 — `logs/run_r5_cc_stable.log:1236`; split `contextcloud/task_utils.py:16-28`; window-bulut mekanizması `contextcloud/model.py:121` (WindowBulutEncoder), `:196`, `:246` |
| Tabular/fraud | hibrit okuyucu + CloudEncoderND | AUC-ROC 0.9351 (`fraud/results/full_scale_hybrid.json:10`); doğal-oran held-out FP32 0.9451 / INT8-channel 0.9367 (`fraud/logs/run_full_scale_edge_eval.log:52,54`); XGB 0.9934 üstünlük iddiası YASAK, sınır dürüstçe yazılacak (`fraud/results/duels.json:4`) |
| Tabular açıklama | AML/bakım/şikayet | nearest-prototype acc 0.75 (`aml/results/e2_explain.json:16,19`); drift/kararlılık sayıları `maintenance/` ve `complaints/results/` JSON'larından VEKIL tarafından doğrulanarak alınacak |
| Uç/INT8 paket | fraud deploy | self-contained `fraud/results/deploy/`; INT8 parity `fraud/pilot/infer.py:107`; footprint/latency `fraud/results/edge_eval_full_scale.json` |
| Verim varyantı | purecloudlm O(N) | GlobalGeometryEncoder: Transformer'sız, O(N) küme istatistikleri (`purecloudlm/model.py:1-12,57`) — PERFORMANS İDDİASI YOK (geçerli held-out başarısı yok), yalnız bağımlı istem varyantı: "çift-geometri yerine küresel geometri istatistikleri kullanan alternatif agregatör" |
| Hibrit eğitim | QUANN 0th-order | Yalnız bağımlı ve sınırlı: "küçük/sınırlı modellerde nokta konumları için 0th-order daraltma, okuyucu için backprop" (`quann_cloud/src/qcloud_quann.py:212-228`). Dürüst sınır: LM'de QUANN held-out PPL 1016.30 vs random 1045.03 (`quann_cloud/FINDINGS.md:28-30`) — buluş iddiası değil, seçenek rejimi. |

## 3. İstem mimarisi (öneri)

- **İstem 1 (bağımsız yöntem):** bağlam kodlayıcı → D-boyutlu nokta bulutu →
  D+4 çift-geometri (`cloudlm/model.py:157-168`: fark, Öklid, cos, iki norm) →
  simetrik agregasyon (nokta max-pool `model.py:153` + çift mean-pool `:177`) →
  okuyucu → görev çıktısı; VE aynı nokta bulutundan ÜÇ açıklama işleminin en az
  biri (nokta-çıkarma etkisi / bölge isimlendirme / bölge kayması) ile açıklama
  verisi üretme. Görev türünden bağımsız formülasyon: dil modelleme VEYA
  sınıflandırma (tabular satır = birincil değerlerin nokta bulutu).
- **İstem ~8 (bağımsız sistem)** ve **son bağımsız istem (CRM):** aynı iş akışının
  sistem ve kayıt ortamı karşılıkları (三-kategorili klasik dizilim).
- Bağımlılar: D 3–64; D+4 bileşenleri; max+mean simetrik agregasyon;
  O(N) küresel-istatistik agregatörü (varyant, performans iddiasız);
  çoklu-cümle window-bulut hiyerarşisi (cümle-bulut + komşu pencere-bulut,
  `contextcloud/model.py:20,121`); INT8 + self-contained manifest paketi;
  bölge isimlendirmenin kümelemeye (KMeans) dayandırılması; drift için TV/KL;
  hibrit QUANN-eğitim sınırlı rejim; ikili-use (aynı eğitimli model hem dil hem
  tabular görevde — D-boyutu sabit kalarak görev-head değişimi).

## 4. KIRMIZI ÇİZGİLER (ihlal = RED sebebi)

1. "20 nokta" sabiti YASAK (kodda yok; `VEKIL_SORU_CEVAP.md:91-94` DOĞRULANMADI
   notu). D aralığı 3–64 kullanılır.
2. Her sayı tarifnamede Yalnız kaynaklı; kaynak dosya repo'da mevcut olmalı.
   Şekillere elle sayı yazılamaz — ölçümlü figür gerçek JSON/log'dan çizilir,
   şematik figür künyesinde "şematiktir" yazar.
3. Doğruluk üstünlüğü iddiası yok (XGB 0.9934, causal 2.94 örnekleri açık
   sınır olarak yazılır). Kazanım kümesi: birlikte-temsil, açıklanabilirlik,
   uyum hızı, INT8 uç paket, drift gözlemi.
4. purecloudlm ve QUANN "başarılı model" gibi sunulamaz (bkz. tablo notları).
5. Held-out olmayan sayılar held-out diye geçmez (R2/R3 etiketsiz PPL'ler
   kullanılmaz; R3 ham-log sayısı zaten yok).
6. Eski v2.3/v2.5 metinlerinden cümle kopyalanmaz; yeniden yazılır.

## 5. Dış biçim (TÜRKPATENT)

- `ISTEMLER.md` — numara sırası: yöntem istemleri, sistem istemleri, CRM.
- `TARIFNAME.md` — teknik alan, önceki tekniğin açığı, buluşun özeti, şekillerin
  kısa açıklaması (Fig.1–8), ayrıntılı açıklama, örnekler (ölçümlü tablolarla),
  endüstriyel uygulanabilirlik.
- `OZET.md` — ≤300 kelime + teknik etki.
- Figürler: `fig/fig1..fig8` siyah-beyaz SVG (A4, Times yazı). Ölçümlü olanlar
  gerçek veriden, diğerleri şematik.
- Build: `build_v3.py` (md→docx→pdf; konvansiyon: `docs/patent/dil` paketi).
