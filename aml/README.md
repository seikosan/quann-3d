# aml/ — AML / Para Aklama Tipoloji Tespiti + QUANN Nokta-Bulut

QUANN-3D repo'sunun alt projesi. Ticari gerekçe: `docs/market_priorities.md`
(Öncelik 3, dikey 1).

* **Tez:** Tipolojiler şekil kalıbıdır. Structuring = eşiğe yığılan depozito;
  layering = geçici zincir şekli; velocity = çift kanalda dar açı imzası.
  Yeni şekil = yeni tipoloji; bölge haritası denetçiye gösterilebilir;
  QUANN ile haftalık uyum, retrain'siz.
* **Alıcı:** bankalar, ödeme kuruluşları, uyum ekipleri.
* **Farklılaştırıcılar:** 4'ü de aynı anda aktif (küme temsili, çift kanal,
  açıklanabilir geometri, backprop'suz).

## Veri (plan)

| Kaynak | Tür | Rol |
|---|---|---|
| **Sentezik tipoloji üreteci** (`src/generator.py`, kurulacak) | sentetik | structuring / layering / velocity / normal üreteçleri — ilk kanıt katmanı |
| Elliptic (darknet işlemler, public) | gerçek | mixing/mule akışı; pencere→bulut; erişilebilirlik `experiment` agent'ıyla doğrulanacak |
| IEEE-CIS (public) | gerçek | kart işlemleri; `fraud/` şeması birebir yeniden kullanılabilir |

**Gerçek kapı veri erişimi:** bankacılık tarafı structuring/layering verisi
public değil → sentetik wedge ile giriyoruz; gerçek veri kanıtı partner
(anonimleştirilmiş işlem verisi) gerektirir.

## Nokta-Bulut Eşlemesi

`fraud/src` pipeline'ı yeniden kullanılır (features → quantize →
`QCloudFraudClassifier` → `CloudEncoderND`):

```
hesap penceresi (K işlem) ──> işlem ──K özellik──> K nokta
                            ──> bulut ──> CloudEncoderND ──> tipoloji skoru
çift kanal: (zaman, tutar, muhatap) → mesafe + açı (velocity/zincir imzası)
```

Regimler `fraud/` ile aynı: `pure` / `hybrid` / `backprop` (tavan).
Tipoloji prototip bulutu = şablon; **prototipe mesafe = alarm + gerekçe**
("bu hesap bulutu bilinen *layering* prototipine mesafe 0.3; SAR taslağı +
gerekçe: eşik-kenarı kümesi").

### Sentezik üreteçler (E0)

| Tipoloji | Şekil imzası |
|---|---|
| structuring | eşik altı (ör. 10K) tutarlarda depozito kümesi; tutar dağılımı eşik-kenarında yığılma |
| layering | çok-hop transfer zinciri; kısa ömürlü ara hesaplar; zincir açısı/dalga boyu |
| velocity | kısa zamandaki sık-küçük işlemler → çift kanalda dar açı |
| normal | taban aktivite bulutu |

## Deney Kuyruğu

| # | Deney | Durum |
|---|---|---|
| E0 | sentetik tipolojiler: standart (LR/RF/GBT) + hybrid; AUC + tipoloji-bazlı recall @ sabit FPR | bekliyor |
| E1 | bölge haritası figürü + "prototipe mesafe → SAR taslağı gerekçesi" demosu | bekliyor |
| E2 | Elliptic (erişilebilirse): pencere→bulut, mixing etiketi | bekliyor |
| E3 | çift kanal velocity deseni (sentezik + gerçek) | bekliyor |
| E4 | QUANN haftalık adaptasyon: tipoloji drift'i → 0th-order güncelleme, retrain yok | bekliyor |

**Koşum notu:** CPU (MPS, canlı 1.5M CloudLM run'ı için açık tutulur) —
`fraud/` ile aynı konvansiyon.

## Başarı Kriterleri

1. hybrid AUC ≥ standart tavan (`fraud/` örneği: 0.9745 ≈ RF/GBT)
2. tipoloji-bazlı recall @ sabit FPR + bölge haritası figürü (açıklanabilirlik demosu)
3. INT8 küçük modelde ms bütçesi içinde infer
4. QUANN: yeni tipolojiyi retrain'siz yakalamak

## Dürüst Not

Public structuring/layering dataset yok; E0 kanıtı sentetik. Ticari iddia
*ham doğruluk* değil: **açıklanabilirlik + uyum hızı + boyut** üçlüsü.

## Layout

```
aml/
  src/          (planlandı: generator.py + fraud/src yeniden kullanımı)
  data/         (git-ignore; küçük dosyalar gerekirse -f ile eklenir)
  results/      metrics JSON + checkpoint
  figures/      bölge haritası, 3D bulut
  logs/         append-only
  tests/
```
