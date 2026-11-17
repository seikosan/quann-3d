# CFPB Complaint Cloud Pilot

Bu alt-proje, CFPB tüketici şikâyet anlatılarını pozisyonsuz nokta bulutlarına
dönüştürür. Aynı bulut **issue skorunu** üretir ve KMeans prototipleri ile
**native bölge açıklaması** sağlar; haftalık bölge payları süpervizesiz drift
ve yeni-bölge sinyali için izlenir. Amaç XGBoost doğruluğuyla yarışmak değil,
dual-use geometrinin metin şikâyetlerinde uygulanabilirliğini pilotlamaktır.

## Veri ve sınırlar

- Kaynak: `data/complaints/cfpb_narrative_snapshot/` (67 Parquet parçası).
- Kaynak snapshot yaklaşık 16.5M kayıt; narrative, tarih ve ürün alanı dolu
  kullanılabilir altküme yaklaşık 3.77M. Pilot bunların tamamını belleğe almaz;
  deterministik reservoir sample varsayılan `n_sample=100_000` ile sınırlıdır.
- Tarihsiz, anlatısı boş veya ürünü boş satırlar dışlanır. Örneklem kronolojik
  olarak eski %70 train-window ve yeni %30 drift-window'a ayrılır.
- CFPB yalnızca finansal-hizmet şikâyetlerini temsil eder; örneklem tüm CFPB
  nüfusunun temsili garanti edilmez. Snapshot 2026-07'de biter. Bu, kapsamlı
  marka-izleme veya tüm sektörlere genelleme kanıtı değildir.
- `issue` cross-entropy hedefi yalnızca konu temsili öğrenmek için bir proxy'dir.
  Drift ölçümü issue etiketlerini kullanmaz ve operasyonel uyarı için ayrıca
  kalibrasyon/insan incelemesi gerekir.

## Çalıştırma

Repo kökünden, gereken Python paketleri (`torch`, `tokenizers`, `pyarrow`,
`pandas`, `scikit-learn`, `matplotlib`, `onnx`, `onnxruntime`) kurulu olmalı:

```bash
python3 complaints/src/train_reader.py --n-sample 20000 --epochs 2 \
  2>&1 | tee -a complaints/logs/run_pilot.log
python3 complaints/src/explain_region.py
python3 complaints/src/audit_drift.py
python3 complaints/src/export_edge.py
```

Eğitim BPE'yi train-window anlatılarında öğrenir, küçük issue reader'ını
eğitir ve `complaints/results/reader.pt` ile tokenizer'ı yazar. Eğitimde her
epoch issue accuracy/balanced-accuracy raporlanır; bu değerler drift başarısı
olarak yorumlanmamalıdır. Bölge haritası ve drift JSON'u `results/`, PCA ve
haftalık pay grafiği `figures/`, ONNX paketiyse `results/edge/` altına çıkar.
Pilot MPS varsa onu, yoksa CPU'yu kullanır; örnek sayısı CLI ile küçültülebilir.

Testler alt-proje dizininden çalışır:

```bash
cd complaints
python3 -m pytest tests/ -q
```
