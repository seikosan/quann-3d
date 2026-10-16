# Market Öncelikleri — Ticari Alan Haritası

> Tarih: 2026-10-16. Dört farklılaştırıcı: (1) küme/temsil, (2) çift kanal
> (mesafe + açı), (3) açıklanabilir geometri (bölge haritaları), (4) QUANN
> (backprop'suz eğitim).
> Repo kanıtı: `fraud/README.md` + `fraud/results/` (IBM kart verisi, hybrid
> AUC-ROC 0.9745 ≈ standart tavan), `quann_cloud/` (0th-order bulgu).

## Öncelik 3 — Dört farklılaştırıcının hepsinin aynı anda değer olduğu, bütçesi ve düzenleyicisi hazır alanlar

### 1. AML / para aklama tipoloji tespiti

- Alıcı: bankalar, ödeme kuruluşları, uyum ekipleri
- Neden kazanır: Tipolojiler şekil kalıbıdır (sınıra yığılan depozito = structuring); yeni şekil = yeni tipoloji; bölge haritası denetçiye gösterilebilir; QUANN ile haftalık uyum, retrain'siz
- Örnek: "Bu hesap bulutu bilinen *layering* prototipine mesafe 0.3; SAR taslağı + gerekçe: eşik-kenarı kümesi"

### 2. İşlem fraud / kart otorizasyonu (edge)

- Alıcı: kart şebekeleri, acquirer, e-ticaret, neobank
- Neden kazanır: Doğrudan P&L (kayıp + yanlış reddetme geliri kaybı); 21MB INT8 → otorizasyon yolunda ms bütçesi içinde; hız/korelasyon sinyali çift kanalda
- Örnek: "5 dk'da 4 farklı cihazdan küçük çekimler" = çift kanalda dar açı imzası → kart testi; sessiz hesaptan yeni şehir+merchant kümesi = ATO

### 3. Endüstriyel bakım / makine sağlığı

- Alıcı: üretim, enerji, maden, F&B tesisleri
- Neden kazanır: Tesisler veri merkezi değil — edge şartı gerçek; mühendis "bölge haritası"nı SHAP'a göre çok daha kolay okur; arıza maliyeti ROI'yi net ölçer
- Örnek: 10 sensör penceresi → bulut; rulman aşınması = şekil bozulması; "geçen hafta batı bölgesi genişledi" → planlı bakım

## İkincil 10

| # | Alan | Örnek |
|---|---|---|
| 1 | Kredi kararı açıklanabilirliği | "Başvuru X bölgesinde (revolving + yeni limit)" → adverse-action gerekçesi; portföy drifti = bulut kayması alarmı |
| 2 | Piyasa manipülasyonu | Emir defteri zaten bulut; spoofing = geçici (fiyat,boy) şekil bozulması |
| 3 | Mule / halka ağları | Paylaşılan cihaz/IP → çift kanal açılarında halka geometrisi |
| 4 | Sigorta hasar fraud | Müşteri hasar geçmişi bulutu + yeni hasar noktası; yeni bölge = yeni dolandırıcılık kalıbı |
| 5 | Kopya ürün / içerik | Aynı ürün 5 satıcıda farklı cümle; bulut-bulut mesafesi → sahte ilan elenir |
| 6 | Şikayet / marka izleme | Haftalık şikayet bulutu; "iade süresi" kümesi yeni belirirse trend olmadan uyarı |
| 7 | Molekül / ilaç etkileşimi | Atom = nokta, bağ = mesafe+açı (çift kanal bire bir); küçük model, hat içi |
| 8 | İmalat kalite kontrol | Parça 3D taramı vs nominal bulut; sapma = şekil farkı |
| 9 | Cihatta kişiselleştirme (QUANN) | Telefon öneri modeli, beğenilerle gradient'siz uyum; veri çıkmaz → gizlilik |
| 10 | Gradient'siz model ayarı | Türev yok, sadece ödül (insan oyu, iş KPI) olan donmuş modelin yerel ayarı |

## Dürüst not

3'lüsü "dönüşüm" seviyesinde ticari vaka; ikinciller giriş alanı olarak iyi
ama her birinde kurulu rakip (XGBoost/GNN/sequence) var — kazanma gerekçesi
daima *açıklanabilirlik + uyum hızı + boyut* üçlüsü, ham doğruluk değil.
