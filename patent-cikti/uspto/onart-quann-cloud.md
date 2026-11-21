# Ön Sanat Araması — QUANN-Cloud / CloudLM

- Tarih: 2026-11-19 · Araç: `patent-creator.search_non_patent_literature` (OpenAlex) · 3 sorgu × 15 sonuç = 45 aday
- Buluş tanımı (kaynak `RESEARCH.md:14-16`): CloudLM metni uzaydaki nokta bulutlarıyla
  temsil eder; QUANN-Cloud aynı yaklaşımı gradyan geri yayılımı olmadan eğitir.
- **Patent veritabanı araması YAPILMADI**: SerpApi/Google Patents, EPO OPS, BigQuery ve
  USPTO API dördü de yapılandırılmamış (bu oturumda `check_google_patents_status`,
  `check_epo_api_status`, `check_bigquery_status`, `check_uspto_api_status` koşuldu;
  dördü de anahtar istiyor). Boş patent tarafı "önsanat yok" kanıtı DEĞİLDİR.
- Bu liste yalnız makale/preprint taramasıdır; **denetçi/vekil görüşü gerekli**.

## Yenilik riski sırası (en yakın üstte)

| # | Kaynak | Yayın | Başlık | Ne öğretiyor | Bizden ayrışması MUHTEMEL nokta |
|---|--------|-------|--------|--------------|--------------------------------|
| 1 | makale | arXiv 2607.16612 (2026) | Backpropagation-Free Trunk Training via the Split Forward Gradients | Dil modelinde (GPT-2 tarzı, WikiText-103) geri-yayılımsız eğitim; donuk-trunk kontrolü ve trunk'ı kısmen eğitme | Trunk/head ayrımı yapıyor; bizim ikili kanal (pair) + freeze kombinasyonu ve nokta-bulut temsilimiz yok |
| 2 | preprint | arXiv 2410.09734 (2024) | Gradient-Free Training of Quantized Neural Networks | Kuantalize ağları gradyansız eğitme (QNN+gradient-free bileşimi) | Kuantalizasyon gradyansızlığa temsil ekseni olarak eşlik etmiyor; mimarimiz LM + geometrik gömme |
| 3 | preprint | arXiv 2601.21561 (2026) | SAL: Selective Adaptive Learning for Backpropagation-Free Training with Sparsification | Backprop'suz eğitimde seyreltme/seçici parametre güncelleme | Seçiciliğin parametre-kanal seçimi (R8: 86080 seçili) ile geometrik temsille birleşimi yok |
| 4 | preprint | arXiv 2210.14593 (2022) | Scaling Laws Beyond Backpropagation | Backprop dışı eğitimde ölümleme yasaları (LM) | Ölçekleme dili nokta-bulut gömmeli LM'ye özgü değil |
| 5 | makale | AAAI 2023, doi:10.1609/aaai.v37i8.26118 | Backpropagation-Free Deep Learning with Recursive Local Representation Alignment | Geri-yayılımsız derin öğrenme için türetilmiş local hedef mekanizması | Mekanizma ailesi biliniyor; iddiamız mekanizma + temsilin birleşimi |
| 6 | makale | arXiv 2603.24790 (2026) | Local learning for stable backpropagation-free NN training towards physical learning | Backprop'suz yerel öğrenimde karlılık koşulları | Karlılık analizi fiziksel ağlara; LM/nokta-bulut tarafı yok |
| 7 | makale | Science 2023, doi:10.1126/science.adi8474 | Backpropagation-free training of deep physical neural networks (PhyLL) | Fiziksel katmanlar için detay model gerektirmeyen geri-yayılımsız eğitim | Alan fiziksel donanım; natively LM değil |
| 8 | preprint | arXiv 2412.05667 (2024) | Training neural networks without backpropagation using particles | Parçacık yöntemleriyle geri-yayılımsız eğitim | Optimizatör-ailesi öncül; temsil ekseni yok |
| 9 | bildiri | ICASSP 2026, doi:10.1109/icassp55912.2026.11464818 | Hebbian Learning with Global Direction | Hebbian yerel kurala küresel yön bilgisi ekleme | Hebbian ailesi; LM + geometrik gömme birleşimi yok |
| 10 | makale | arXiv 2210.14593 ailesi / TechRxiv 2026 doi:10.36227/techrxiv.176740426.63642005 | Beyond-Backpropagation Training: Methods, Applications, Perspectives | Backprop-dışı yöntemlerin kapsamlı incelemesi | Anket — bileşik iddiamızın sınırlarını çizmede referans |
| 11 | makale | ECCV/LNCS 2024, doi:10.1007/978-3-031-72698-9_8 | PointLLM: Empowering LLMs to Understand Point Clouds | LLM'e nokta bulutu ANLAMA yeteneği kazandırma (modal girdi) | Yön tersi: onlar 3D'yi LLM'e besler; biz METNİ nokta bulutu geometrisiyle temsil ederiz |
| 12 | makale | NeurIPS 2023, doi:10.52202/075280-0900 | 3D-LLM: Injecting the 3D World into Large Language Models | 3D sahne bilgisini LLM'e enjekte etme | Aynı ayrışma: temsil aracı olarak nokta-bulut ≠ metnin kendisinin bulutu |
| 13 | preprint | arXiv 2309.00615 (2023) | Point-Bind & Point-LLM | Nokta bulutu–dil çoklu-modal hizalama | Hizalama (alignment) yapıyor; gömme geometrisi olarak bulut değil |
| 14 | preprint | arXiv 2504.04540 (2025) | The Point, the Vision and the Text: Does Point Cloud Boost Spatial Reasoning of LLMs? | Nokta bulutunun LLM uzamsal muhakemesine katkısı (yanlılık kontrollü) | Değerlendirme çalışması; eğitim yöntemi + temsil iddiası yok |
| 15 | makale | arXiv 1609.07061 (2016) | Quantized Neural Networks (QNN) | Düşük hassasiyetli ağırlık/aktiviteyle eğitim | "QUANN" isim benzerliği; backprop-KULLANIR, ayırt edici unsur geri-yayılımsızlık |

## Gözlem (yorum, hüküm değil)

- **Mekanizma ekseni** (backprop'suz LM eğitimi) 2022-2026 arasında yoğun çalışılmış:
  Split-FG, SAL, RLA, particle methods, Hebbian varyantları — saf "backprop'suz eğitim"
  geniş bir alan olarak iddia edilemez; dar iddialar (bizim pair/freeze kanal seçimi gibi
  ölçülmüş mekanik detaylar) üzerinden kurulmalı.
- **Temsil ekseni** (metnin kendisini nokta bulutu geometrisi olarak temsil) nokta-bulut+
  LLM literatüründe ters yönde (3D→LLM) çalışılmış; her iki eksenin BİLEŞİMİNİ tek
  referansta gören aday bulunamadı — bu, ayrışma argümanı adayıdır; yenilik hükmü için
  patent veritabanı taraması + vekil görüşü şart.

Durum: denetçi görüşü gerekli. En kritik 2 referans: **arXiv 2607.16612** (backprop'suz LM +
freeze bulgusu) ve **PointLLM/3D-LLM ailesi** (nokta bulutu-LLM hizalama tarafı).
