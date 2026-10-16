# maintenance/ — Endüstriyel Bakım / Makine Sağlığı + QUANN Nokta-Bulut

QUANN-3D repo'sunun alt projesi. Ticari gerekçe: `docs/market_priorities.md`
(Öncelik 3, dikey 3).

* **Tez:** Makine aşınması şekil bozulmasıdır. Rulman aşınması = bulut
  şekil bozulması; "geçen hafta batı bölgesi genişledi" = planlı bakım
  alarmı. Tesisler veri merkezi değil — edge + retrain'siz (QUANN) gerçek şart.
* **Alıcı:** üretim, enerji, maden, F&B tesisleri.
* **Farklılaştırıcılar:** 4'ü de aynı anda aktif; mühendis bölge haritasını
  SHAP'a göre çok daha kolay okur.

## Veri (plan, tamamı public)

| Kaynak | Şekil | Rol |
|---|---|---|
| C-MAPSS (NASA turbofan, FD001–FD004) | 36 sensör özelliği, tabular, RUL etiketi | E0: pencere→bulut doğrudan eşleme (`fraud/` tabular şekliyle izomorf) |
| CWRU rulman | ham sinyal → öznitelik çıkarımı (RMS, crest, fabrika frekansı enerjisi) | aşınma = şekil bozulması |
| (aday) PU / Skf123 ivmeölçer | 3 kanal | kanal-lerarası açı (çift kanal) |

Kaynak URL'leri ve indirilebilirlik: `experiment` agent'ı doğrulayacak.

## Nokta-Bulut Eşlemesi

```
sensör penceresi (t..t+W) ──> her örnek ──c kanal──> c-boyutlu nokta
                         ──> W nokta bulutu ──> CloudEncoderND ──> aşınma evresi / RUL
çift kanal: sensör-lerarası açı (korelasyon deseni) → aşınma imzası
```

C-MAPSS: 36 özellik → 36-boyutlu nokta (D=32 indirgemesi de denenebilir);
W=10 ardışık örnek → 10 nokta bulut — cümle bulutu (10 kelime × 32) ile
aynı geometri, makale ile izomorf.

## Deney Kuyruğu

| # | Deney | Durum |
|---|---|---|
| E0 | C-MAPSS FD001: pencere→bulut; LR/RF/GBT/MLP + hybrid; aşınma evresi sınıflaması | bekliyor |
| E1 | CWRU rulman (öznitelik çıkarımlı): normal vs aşınmış; 3D bulut figürü | bekliyor |
| E2 | çift kanal: sensör-lerarası açı → aşınma imzası | bekliyor |
| E3 | bölge haritası + drift alarmı demosu ("bölge genişledi") | bekliyor |
| E4 | QUANN saha adaptasyonu: yeni işletme koşulu → 0th-order, retrain yok | bekliyor |

**Koşum notu:** CPU (MPS, canlı 1.5M CloudLM run'ı için açık tutulur).

## Başarı Kriterleri

1. aşınma evresi sınıflamasında hybrid ≥ standart baselines (`fraud/` örneği)
2. bölge haritası + drift alarmı figürü (mühendis-okunur açıklanabilirlik)
3. INT8 küçük model, edge ms bütçesi
4. QUANN: işletme drift'ine retrain'siz adaptasyon

## Dürüst Not

RUL regresyonunda LSTM/Transformer tavanları var; iddia *ham doğruluk* değil —
**açıklanabilirlik (bölge haritası) + saha (retrain'siz) + boyut (INT8)**.

## Layout

`aml/` ile aynı: `src/` (planlandı: data loader + fraud/src yeniden kullanımı),
`data/` (git-ignore), `results/`, `figures/`, `logs/`, `tests/`.
