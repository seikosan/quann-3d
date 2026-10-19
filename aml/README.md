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
| E0 | sentetik tipolojiler: standart (LR/RF/GBT) + hybrid; AUC + tipoloji-bazlı recall @ sabit FPR | tamam — clean setting GBT 1.0; hybrid .96 AUC / .872 AP; challenging setting detection gap (`results/e0_smoke.json`) |
| E1 | bölge haritası figürü + "prototipe mesafe → SAR taslağı gerekçesi" demosu | tamam — held-out nearest-prototype %87.8; structuring 14× separation; SAR taslağı demosu (`results/e1_explain.json`) |
| E2 | Elliptic (erişilebilirse): pencere→bulut, mixing etiketi | tamam — 2000 labeled-node proxy: hybrid .838/.370 vs GBT .943/.758; explanation %75 overall / %45 illicit; 1-hop/transductive sınırlaması açık (`results/e2_elliptic.json`, `results/e2_explain.json`) |
| E3 | çift kanal velocity deseni (sentezik + gerçek) | tamam — chain-aware çift kanal + E3′ zaman-disjoint 46.5K Elliptic doğrulaması; detection gap sürüyor (`results/e3_chain.json`, `results/e3_prime.json`) |
| E4 | QUANN haftalık adaptasyon: tipoloji drift'i → 0th-order güncelleme, retrain yok | tamam ama başarısız — retraining gain'inin 0%'ını geri aldı; iddia open work (`results/e4_adapt.json`) |
| E5 | retraining stability + audit batch (yeni) | tamam — 5-seed Elliptic pilot: name stability 0.325 vs SHAP top-1 0.1025; top-3 Jaccard 0.434, Spearman 0.651, PCA-2 displacement 4.49; kaba label daha stabil, sürekli 2D map/magnitudes stabil değil (`results/e5_stability.json`) |

**Koşum notu:** CPU (MPS şu an boş; D=64×200K-wiki adayı için ayrılabilir) —
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
