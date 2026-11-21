# PCT v2 Denetim Özeti (RO TÜRKPATENT · EN asıl + TR eş kopya) — 2026-11-19
Denetlenen: filing/*.txt (asıl) + OZET/OLCUM defteri. Repo dosyaları DEĞİŞTİRİLMEDİ.

## (a) GERÇEK KUSURLAR (önem sıralı)
1. CRITICAL (defter) — [0057] "4096 girdi" sayısının OLCUM_KAYITLARI.md'de satırı YOK (defter başlığı
   "her sayı" der, S1'de yok). Sayı uydurma değil: ham kaynak fraud/logs/run_full_scale_edge_eval.log:46
   "4096 random QUANN-point inputs (proxy)" — doğrulandı. Ama (i) defter girdisi eksik, (ii) log'daki
   "(proxy)" niteleyicisi TARIFNAME_EN.txt:127 / TARIFNAME_TR.txt:127'de yok. Düzelt: deftere S1 satırı
   ekle; [0057]'e "proxy-input" nitelemesi ekle veya vekile sor.
2. MED (TR kopya, sistematik) — ISTEMLER_TR istem 2–12 ve 14 sondan "karakterize edilir" ile bitiyor
   (örn. ISTEMLER_FILING-COPY_TR.txt:7,11,31); EN karşılıkları bağımlı istemlerde yalnız "wherein"
   (CLAIMS_FILING-COPY_EN.txt:25,29). İki-parçalı biçim bağımsız istem içindir (PCT Rule 13.2; EPO
   uygulaması bağımlı istemde "characterized in that" istemez). Düzelt: TR bağımlı istemleri
   "...olup, ... olmasıyla tanımlanan" kalıbına çevir; EN'deki "wherein"e bağla.
3. MED (EN kopya) — İstem 1 "reference data set" (CLAIMS_FILING-COPY_EN.txt:21) ile istem 8 "reference
   data group" (satır 37) terminoloji kayması; tarifname [0013]/[0046] "group" diyor; TR kopya tutarlı
   ("veri grubu"). Düzelt: EN istem 1'i "group"a çevir.
4. MED (kopyalar arası sapma) — TR istem 12 "geri alma işlemleriyle DARALTILARAK güncellenmesini"
   diyor (ISTEMLER_FILING-COPY_TR.txt:27); EN istem 12'de (CLAIMS_FILING-COPY_EN.txt:45) bu kısıt yok
   (interval daraltma yalnız [0053]'te). TR, EN'de olmayan bir kısıt ekliyor. Düzelt: ya TR'den çıkar
   ya EN istem 12'ye ekle — vekil kararına kadar kopyalar eşitlenmeli.
5. LOW-MED (TR kopya) — TR istem 13 önsözde "bir okuyucu" + karakterize listesinde ikinci "bir
   okuyucu" (ISTEMLER_FILING-COPY_TR.txt:29); EN "wherein the reader" ile bağlıyor
   (CLAIMS_FILING-COPY_EN.txt:53). TR'de liste içinde "söz konusu okuyucu" yaz.
6. LOW (defter) — Hafta tarih dizileri ([0047]: 2025-02-03/09, 2025-02-10/16) defterde değer olarak
   künyelenmemiş; ham kaynakta doğrulandı (complaints/results/drift_report.json:21,37). Düzelt: deftere
   tarih satırı ekle.

## (b) TARZ / VEKİL KARARI
- "distribution store" (istem 9/14 + [0050] birebir aynı — destek sorunu yok) İngilizcede "dağıtık veri
  deposu" okunabilir; milli aşama clarity riski → "deployment package/store" değerlendirilebilir.
- [0005] 0.65105 (yuvarlanmış) ↔ [0064] 0.6510458604357529: aynı ölçüm, iki hassasiyet; yuvarlama
  doğru (defter S4:84). Tek yerde "yuvarlanmış biçim" künyesi yeter ([0056]'deki gibi).
- "BEST MODE FOR CARRYING OUT THE INVENTION" EN'de mevcut (satır 69) — Rule 5.1 alt başlık listesinde
  öneriliyor (araç çıktısı: pct_regulations.pdf p.32, "Best Mode(s)..."); lint'in "best=subjective"
  uyarısı araç gürültüsü.

## UYUMLU ÇIKANLAR (ölçüldü)
- Rule 5.1 altı başlık EN'de 6/6 (grep); 70 benzersiz paragraf [0001]-[0070]; TR aynı sıralama.
- Rule 6.2(a): EN+TR istemlerde tarifname/şekil atfı 0 (grep FIG./Şekil/[00xx]).
- Rule 8.1(b): ABSTRACT_EN 137 kelime, OZET_TR 106 kelime (wc -w filing txt üzerinde) → 50–150 önerilen
  aralıkta ✓ (kural metni: pct_regulations.pdf p.32). Övme/değerlik yargısı yok (8.1(c)).
- Rule 4.3 başlık: EN 6, TR 7 kelime → "preferably two to seven words" önerilen aralıkta ✓
  (kural metni: p.24); başlık tüm kopyalarda aynı metin.
- Rule 13.2 benzeri iki parça: EN bağımsız 1/13/15 "characterized in that" ✓; TR 1/13/15 "karakterize
  edilir" ✓; gövde (context encoder+reader = yaygın önceki teknik) / karakterize (nokta bulutu + üç
  açıklama işlemi) ayrımı mantıksal ✓.
- Destek: istem 4 ↔ [0034] birebir; istem 9/14 "distribution store ... single self-contained package"
  ↔ [0050] birebir; istem 10 "text consisting of a plurality of sentences" ↔ [0065] birebir (+[0037]).
- Rule 13.1 birliği (bulgu): yöntem/sistem/kayıt ortamı üçü de aynı özel tekniği taşıyan özelliği
  paylaşıyor (aynı nokta bulutundan çıktı + ayrı açıklama modeli olmadan açıklama). Birlik bulgusu
  olumlu; hukuki hüküm vekilin/onun önünde ISA'nın.
- Sayılar: ~20 ham-kaynak spot kontrolü (edge_eval_full_scale.json, manifest.json, duels.json:4/26/50/113,
  drift_report.json, e5_stability.json:29, FINDINGS.md:28-30, run_compare/run_r4/run_r5 logları) —
  TÜMÜ defterle ve birbiryle tutarlı; defterde karşılığı olmayan tek sayı (a)1'deki 4096.

## (c) DENETLENEMEDİ / YAPILMADI
- ÖN SANAT ARAMASI YAPILMADI (görev kapsamında değil; Google Patents/EPO API durumuna bakılmadı).
- PCT/RO/101 talep formu, şekillerin kendisi (fig/*.svg içerik denetimi), vekaletname, ücret: kapsam dışı.
- Tüm ~90 sayının ham kaynağa tam yeniden-doenetimi yapılmadı; defter künyeleri + 20 spot okuma esas
  alındı. "Yenidir/patentlenebilir" hükmü kurulmaz — ISA/vekil görüşü gerekli.
- patent_lint çıktısı (15 "antecedent" + bölüm uyarıları) büyük ölçüde araç gürültüsü: ABD-regex'i
  bölum adları ve kalıp yanıltmaları; elle okumada gerçek antecedent sorunu görülmedi.
