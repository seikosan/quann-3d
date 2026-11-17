# Vekile Sade Anlatım — Ana Yapı

**Amaç:** Başvuru metnini yazarken teknik yapıyı doğru kurmanız için
jargonsuz, kısa cümlelerle özet. Sorularınızın ayrıntılı kod-kanıtlı
yanıtları ayrıca iletildi.

## 1. Buluş tek cümleyle

Bir girdiyi (cümle, işlem kaydı, sensör okuması) düşük boyutlu uzayda
noktalardan oluşan bir buluta çeviriyor; kararı da açıklamayı da aynı
bulutun geometrisinden üretiyoruz.

## 2. Sistem nasıl çalışıyor — beş adım

1. **Noktaya çevirme:** Her girdi birimi (kelime, işlem özelliği, sensör
   değeri) küçük bir sinir ağıyla 3–64 boyutlu uzayda bir noktaya
   dönüştürülür. Girdi birimi sayısı kadar nokta olur; havuzlama yoktur.
2. **İlişki ölçme:** Noktalar arasındaki mesafeler ve açılar hesaplanır.
   Anlam tek tek noktaların değil, noktaların birlikte nasıl durduğunun
   sonucudur.
3. **Kümeye damıtma:** Tüm nokta ve ilişki bilgileri, sıradan bağımsız
   (permütasyon-değişmez) bir özet vektöründe toplanır. Aynı bulutun
   noktaları farklı sırada girilse de aynı özet çıkar.
4. **Karar üretme:** Bu özetten basit bir ağ görev çıktısını üretir:
   dilde bir-sonraki-kelime, fraud'da risk skoru.
5. **Açıklama üretme (buluşun kalbi):** Ayrı bir açıklama modülü yoktur.
   Buluttan bir nokta çıkarılır, kararın ne kadar değiştiği ölçülür;
   kararın hangi geometrik bölgeden geldiği ve bölgenin zamanla kayıp
   kaymadığı raporlanır. Yani aynı bulut hem kararın hem cevabın
   kaynağıdır.

## 3. Transformer'dan farkı — üç cümle

- Transformer anlamı tek dev vektöre ve sıralı dikkat mekanizmasına
  sıkıştırır; biz anlamı ölçülebilir geometriye (mesafe, açı, küme şekli)
  taşırız.
- Transformer'da "neden bu karar?" sorusu SHAP/LIME gibi sonradan eklenen
  dış araçlarla cevaplanır; bizde cevap modelin kendi temsilinden çıkar.
- Sonuç küçük modellerdir: en iyi dil modelimiz 21M parametre, INT8
  paketi 21MB (`docs/quann-3d-fayda-ozeti.md:20-21`).

## 4. İki eğitim yolu — karıştırılmaması gereken ayrım

- **Yol A (şu anki en iyi sonuçlar):** standart geri-yayılım (backprop)
  ile eğitim. Başvurunun performans örnekleri bu yoldandır.
- **Yol B (bağımsız ikinci buluş katmanı, QUANN-Cloud):** geri-yayılım
  OLMADAN eğitim. Model deneme adayları üretir; hata azalırsa aday kabul,
  artınca geri alınır. Aralığın (arama sınırının) ne yaptığı **varyanta göre
  değişir**, bu yüzden metinde hangi varyantın kastedildiği açıkça
  yazılmalıdır: kabulü aralığı daraltan varyant (hybrid), kabulde aralığı
  genişleten varyant (classic), aralığı sabit tutup yalnız adım ölçeğini
  küçülten varyant (motor). Ayrıntı `VEKIL_SORU_CEVAP.md` içinde.

## 5. Nerede yeni olduğu (yenilik vuruş noktaları)

1. Düşük boyut (3 ≤ D ≤ 64) nokta-bulutu temsilinin dil/sınıflandırma
   görevlerinde çift-geometri (mesafe + açı) ile kullanılması.
2. Aynı bulutun hem skor substratı hem native açıklama kaynağı olması
   (nokta-çıkarma, bölge/prototip, drift — üçü de aynı geometriden).
3. Sıra bilgisinin taşınma biçimi: uygulanmış dil modelinde sıralı konum
   bilgisi **koşulsuz** kullanılıyor; "sırasız temsil" ancak ayrıca
   kodlanıp ölçülürse iddia edilebilir, şu hâliyle "zorunlu değil" diye
   yazılmaz.

## 6. Ne iddia ETMİYORUZ (metne de yazıldı)

- Ham doğrulukta Transformer/XGBoost üstünlüğü iddia etmiyoruz.
  Fraud'da eşleştirilmiş PR-AUC 0.462 vs XGBoost 0.8047 — üstünlük yok
  (`docs/patent/VEKIL_GONDERIM_NOTU.md:69`).
- Kazanım iddiamız üç şeydir: küçük boyut, native açıklanabilirlik,
  backprop'suz eğitim seçeneği.

## 7. İki başvurunun ilişkisi

- **Dosya A (TR ulusal):** yalnız dil uygulaması; erken öncelik tarihi.
- **Dosya B (PCT):** evrensel çekirdek + dört uygulama örneği (dil,
  fraud, AML, bakım); kendi öncelik tarihi, A'dan öncelik talep edemez.

## 8. Küçük sözlük

| Terim | Sade anlamı |
|---|---|
| Nokta bulutu | Birkaç–yirmi noktadan oluşan sayısal küme |
| Permütasyon-değişmez | Noktaların sayım sırası sonucu değiştirmez |
| Native açıklanabilirlik | Açıklamanın ayrı araç gerektirmeden modelin kendi temsilinden çıkması |
| 0th-order / backprop'suz | Türev hesaplamadan, deneme-kabul ile eğitim |
| Prototip | Bir sınıfın ortalama bulut şekli |
| Drift | Bulut bölgelerinin zamanla yer değiştirmesi |
