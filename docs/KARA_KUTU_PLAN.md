# KARA KUTU — Gerçek Hayat Verisi Kanıt Arayüzü (Tasarım · v1)

> Tarih: 2026-11-21. Durum: TASLAK — refactor (REFACTOR_PLAN_2026-11-21.md) bitince uygulanacak.
> Amaç: dışarıdan gelen gerçek hayat verisini tek kapıdan geçirip **tahmin + gerekçe +
> kanıt raporu** üreten arayüz. Model/teknoloji iddiası bu kutunun çıktısıyla savunulacak.
> Kullanıcı kararları (2026-11-21): çoklu dikey çerçeve · web arayüzü · dış + iç kanıt birlikte.

## 1. Kara kutu nedir / nedir

- **Olmaz:** eğitim yapmaz (eval-only), checkpoint değiştirmez, `logs/` append-only kuralını bozmaz.
- **Olur:** veri girer → kayıt başına skor + gerekçe (bölge haritası/mesafe-açı) →
  iki rapor: **dış** (müşteri/vekil/yatırımcı) ve **iç** (RESEARCH formatı, denetlenebilir).
- **Kanıt disiplinini kod zorlar:** rapordaki her sayı ya bu koşunun ölçümü ya da
  `dosya:satır` atıflı olmalı (AGENTS §13). Elle sayı yazılan alan yok.

## 2. Depodaki mevcut motorlar (uydurma değil — dosya kanıtı)

| Dikey | Motor/çıktı | Kanıt pointer |
|---|---|---|
| Fraud | ONNX deploy paketi | `fraud/results/deploy/` (model.onnx, manifest.json, quantizer.json, token_decoder.json); FP32 AUC-ROC 0.9451 `fraud/logs/run_full_scale_edge_eval.log:52`; parity max=1.27e-07 (DEVIR:109) |
| AML | açıklanabilirlik probe + bölge haritaları | `aml/results/e2_explain.json`, `aml/figures/` |
| Complaints | ölçek kararlılığı | `complaints/results/complaints_scale_stability.json` |
| Maintenance | deney sonuçları | `maintenance/results/` |
| Metin/LM | eval hatları | `contextcloud/eval_trained_model.py`, `training/eval_model.py` (B1 yenı adları); heldout PPL R4 D64 61.19 `logs/run_r4_resume.log:70` |

Eksik olan tek şey: bunları tek sözleşmeden geçiren kapı + arayüz + rapor üretimi.

## 3. Mimari — `kara_kutu/` paketi (adapter/ plugin çerçevesi)

```
kara_kutu/
  core/contract.py    # VeriSeti, Kayit, Verdict, KanitBloğu dataclass'ları (tek sözleşme)
  core/registry.py    # adapter keşfi: KARA_KUTU_ADAPTERS env / klasör tarama
  adapters/fraud.py   # ONNX paketini sarar; manifest sha'sını kanıt bloğuna yazar
  adapters/aml.py     # tabular tipoloji skorlayıcı + bölge haritası üretimi
  adapters/maintenance.py
  adapters/complaints.py
  adapters/text_lm.py # CloudLM/ContextCloud PPL + bulut geometrisi çıktısı
  report/external.py  # HTML/PDF: skor + güven + gerekçe görseli + baseline kıyas kartı
  report/internal.py  # markdown: held-out metrikler, seed, kaynak imzaları, DOĞRULANMADI etiketi
  report/provenance.py# her koşu → koşulan komut, git HEAD, model sha256, dataset sha256
  web/app.py          # FastAPI: /api/run, /api/report/<id>, statik tek sayfa UI
  web/ui/             # tek HTML: veri yükle → tablo + gerekçe haritası → rapor indir
  tests/              # parity + sözleşme testleri (aşağıda P0 kabul kriteri)
```

**Adapter sözleşmesi** (her dikey bunu uygular):

```python
class Adapter(Protocol):
    name: str
    def input_spec(self) -> InputSpec: ...            # beklenen kolonlar/tipler, örnek CSV
    def predict(self, ds: VeriSeti) -> list[Verdict]: ...  # skor + gerekçe verisi
    def baseline(self, ds: VeriSeti) -> Baseline: ...      # kıyas (fraud: RF .973/GBT .981,
                                                           #   docs/market_priorities.md:6-8)
    def model_meta(self) -> ModelMeta: ...            # checkpoint/paket sha256, eğitim tarihi, log atıfı
```

Yeni dikey = tek dosya adapter + registry kaydı; core/web/report değişmez.

## 4. İki rapor — içerik sözleşmesi

**Dış rapor** (müşteri/vekil/yatırımcı tonu):
1. Tek cümlelik sonuç + skor dağılım grafiği
2. Kayıt bazlı: "bu kayıt neden yükseldi" → mesafe/açı + bölge haritası görseli
3. Baseline kartı: bizim model vs referans (sayı + hangi veriyle ölçüldüğü)
4. Künye: model sha, dataset sha, koşu tarihi, "bu ölçüm X tarihinde bu veride yapıldı"

**İç rapor** (RESEARCH.md damarı):
1. Metrik adı zorunlu: AUC/PPL/loss ayrı ayrı, held-out mu training korpusu mu işaretli
2. Ham kaynak imzası: her sayının yanı `dosya:satır` ya "bu koşunun ölçümü"
3. Seed/tekrar durumu; tekrar yoksa "tek seed" damgası; sebep doğrulanmadı serbest
4. `DOĞRULANMADI` etiketi renderer'da mekanik olarak üretilir (elle yazılmaz)

## 5. Fazlar ve kabul kapıları

| Faz | Ne | Kabul kapısı (mekanik) |
|---|---|---|
| **P0** | contract + `fraud` adapter + CLI (`python3 -m kara_kutu run --adapter fraud --veri x.csv --rapor html,md`) + iki rapor | parity testi: manifest'teki girdi → bilinen skor (eş tolerance fraud deploy parity'si gibi ≤1e-6); `pytest kara_kutu/tests/` yeşil; dış raporda atıfsız sayı 0 (`rapor_denetle` sıfır bulgu) |
| **P1** | FastAPI + tek sayfa UI; koşular `kara_kutu/runs/<id>/` altına yazılır | browser turu: CSV yükle → tablo → rapor indir; `denetim_kos.py hepsi` exit 0 |
| **P2** | `aml` + `maintenance` + `complaints` adapter'ları (tabular şema ortak) | her adapter için P0 benzeri smoke + en az bir gerçek-dışı-dışı veriyle ilk koşu raporu |
| **P3** | `text_lm` adapter: Türkçe metin gir → PPL + bulut geometrisi görseli; `neural-atlas` görsel katmanıyla bağ (C2 kararı sonrası) | eval hattıyla bit sayım/PPL eşleşmesi; uzun metin ölçümü held-out ibaresiyle doğru etiketlenmiş (DEVIR:180 dersi: eğitim korpusu held-out diye satılmaz) |

Sıra kararı: P0 fraud'la başlar çünkü deploy paketi + parity zaten var — "kara kutu çalışıyor"
iddiası en ucuz oradan kanıtlanır; P2-P3 çerçeveye takılır.

## 6. Gerçek hayat verisi — açık darboğaz

Kutunun asıl iddiası **eğitimde kullanılmamış dış veriyle** koşunca anlamlı. Bu depo
IBM kart verisi vb. üzerinde eğitildi; dış rapor "başka veriyle de çalışıyor" diyebilmeli.

- Fraud: Kaggle dışı taze kaynak gerekir (ör. Vixibaba/PaySim simülasyonu zayıf; gerçek
  anonim örnek kuralı: Public/Private) → kullanıcı iş bağlantısı adayı
- AML: FINCEN/FATF tipoloji metinleri + sentetik tipoloji üreteci (mevcut `aml/` generator'ı)
- Metin: bugün çıkmış haber/köşe yazısı (korpus tarihinde kesilmiş) → günlük taze çekim
- Kural: dış veri `data/external/` altına girer, sha256 + kaynak URL + lisans notu ZORUNLU

## 7. Riskler / karar noktaları

1. **MPS tekliği:** P3 LM koşuları ağır; web servisi eval-only kalsa da kuyruk kuralı
   (AGENTS §1.2) web'den tetiklenen ağır isteğe de uygulanır → P3'te queue basitliği.
2. **Web stack:** FastAPI + düz HTML önerisi (bağımlılık miniması); `neural-atlas` hazır
   bundle'ı (C2 bekliyor) yeniden kullanım adayı ama P1'in kilidi değil.
3. **Refactor bağımlılığı:** B1 yeniden adlandırmaları (`eval_*`) adapter import'larını
   etkiler → P0 ancak refactor dalı merge/kapanış noktası netleşince başlar.
4. **Sunum güveni:** dış rapor "mühür" tonu taşır; içindeki tek yanlış sayı tüm kutuyu
   itibarsızlaştırır → P0 kabul kapısına `rapor_denetle` sıfır-bulgu şartı kondu (§5).
5. **Patent:** arayüz kullanım senaryosu `docs/patent/` anlatısına (açıklanabilir geometri)
   girdi üretir; P3 görselleri vekil dosyasına aday — patent dokümanlarına dokunmaz.

## 8. İlk iş listesi (refactor bitiş sinyalinden sonra)

1. `kara_kutu/` iskeleti + contract + registry (bu dokümanın §3 şeması birebir)
2. `adapters/fraud.py` — `fraud/results/deploy/manifest.json` sözleşmesini okur
3. `report/internal.py` + provenance (HEAD + sha256'lar; `devir_yaz` ruh kardeşi)
4. P0 kabul kapısı koşumu + bağımsız `tester` turu → sonra P1 web
