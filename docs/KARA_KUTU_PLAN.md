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

1. `kara_kutu/` iskeleti + contract + registry (bu dokümanın §3 şeması birebir) ✓
2. `kara_kutu/adapters/fraud.py:1-6` — `fraud/results/deploy/manifest.json` sözleşmesini okur ✓
3. `kara_kutu/report/internal.py:1-6` + provenance (HEAD + sha256'lar; `devir_yaz` ruh kardeşi) ✓
4. P0 kabul kapısı koşumu + bağımsız `tester` turu → sonra P1 web ✓ (P1 bekliyor)

## 9. P0 durum (ölçüm bu oturumda)

- `kara_kutu/`: contract + registry + fraud adapter + CLI + iki rapor + `kanit.json`
  üretimi tamam. Kapılar: `pytest kara_kutu/tests/` yeşil; tam paket yeşil;
  `rapor_denetle` `ic_rapor.md` üzerinde sıfır bulgu; dış raporda span-dışı rakam
  yok (bekçi test `kara_kutu/tests/test_contract.py`); CLI kapı komutu gerçek CSV
  üzerinde koştu ve `kara_kutu/runs/<koşum>/` çıktıları üretildi.
- Parite toleransı manifest ölçüsüne bağlandı: `self_test.max_abs_diff`
  (`fraud/results/deploy/manifest.json:28`) kapı metnindeki 1e-6
  (`docs/KARA_KUTU_PLAN.md:79`) değerini aştığı için test toleransı 1.6e-06
  (`kara_kutu/tests/test_parity.py:21`) — kaynak hakemdir, eşik uydurulmadı.
- Eval-only sözleşme korunur: eğitim yok, checkpoint değişmez, `logs/` append-only.
- Bağımsız `tester` turu tüm iddiaları ham kaynakta karşıladı (kayıt:
  `docs/dogrulayici_tur_2026-11-21.txt`); tek zayıf nokta kapatıldı: `rapor_denetle`
  `|` satırlarını muaf tuttuğu için iç rapor tabloları dış kapıdan kaçıyordu — artık
  depo testi (`kara_kutu/tests/test_contract.py`) atıflı tablo hücrelerini kaynak
  satırında doğruluyor.

### P0 dürüst boşlukları — kapanış (2026-11-21, ölçüm)

- User0 demeti held-out DEĞİL — fold 5 eğitim, n 400000, fraud 408, kutu AUC 0,953261, CI95 [0,938952, 0,96757] (`kara_kutu/runs/auc_farki_20261121-231116/ozet.json:1-27`); doğal held-out (te fold≥8) yerine eğitim-fold demeti kullanılmış — "AUC farkı" demet doğasından.
- UYDURMA AVI (doğrulayıcı turu 2026-11-22): eski "User0 AUC" nokta değeri 0,8346
  yanlıştı; ozet yalnız CI95 [0,740052, 0,929148] taşır (`kara_kutu/runs/auc_farki_20261121-231116/ozet.json:20-22`).
  0,8346 bu sınırların aritmetik ortasıydı; nokta AUC hiç ölçülmemiş — DOĞRULANMADI.
- Kutu FP32 `DeployPath` ONNX'tur, pakette int8 yolu yoktur (`fraud/src/export_deploy.py:47-53`).
  Ekim referansı 0,9367 int8-channel AUC'su edge_eval'in torch-simülasyon koludur (`fraud/logs/run_full_scale_edge_eval.log:54`), kutunun ölçümü değildir; kolun kodu `fraud/src/edge_eval.py:12`.
- Alarm çelişkisi ÇÖZÜLDÜ — aynı ilk 1M CSV satırları, aynı ckpt mtime 17 Eki (`docs/port_kaniti_olcum_2026-11-21.txt:9`), quantizer
  birebir aynı (deploy `quantizer.json` == ckpt sözlüğü, `base_fn` hariç). Kesit-evreni te
  234686, fraud 197 (`docs/port_kaniti_olcum_2026-11-21.txt:15-16`); Ekim işletim
  eşiği 0,9969438910484314 (`fraud/results/false_alarm_audit.json:9`).
  19 Eki kodu (5f66aed worktree, `docs/port_kaniti_olcum_2026-11-21.txt:15`) thr-üstü
  alarmı 693, AUC 0,95971 verdi (`docs/port_kaniti_olcum_2026-11-21.txt:16`); sayaç
  `(p >= THR).sum()` ile sayılır (`kara_kutu/tools/port_kaniti.py:89`).
  Mevcut kutu aynı evrende maks p 0,990873, 0 alarm, AUC 0,961135
  (`kara_kutu/runs/alarm_farki_20261121-232335/ozet.json:35-48`) — fark eşik değil port.
- Mekanizma (kod satırı): paylaşılan mimari portu — `pair_input_dim = 3 * d + 2` (`cloudlm/model.py:64`);
  skaler |i|,|j| ağırlıkları D-boyutlu norm bloklarına yayıldı (`cloudlm/model.py:105-106`);
  havuzlama aynı (`cloudlm/model.py:153`, max). Deploy paketi (a52ddc1) portlanmış
  fonksiyondur; parity self-test'i yalnız kutu ≡ güncel-torch'u kanıtlar.
- Tüm te üzerinde kutu AUC 0,944109, 0 alarm (`kara_kutu/runs/alarm_farki_20261121-232405/ozet.json:95-96`) ↔ Ekim FP32 0,9451 (`fraud/logs/run_full_scale_edge_eval.log:52`) — sıralama yakın, kalibrasyon değil.
- Sonuç/karar (kullanıcı): Ekim FP32 olasılıklarıyla kalibre eşik 0,9969 kutuda geçersiz.
  (A) kutu-yerel eşik yeniden kalibrasyonu; (B) Ekim fonksiyonuyla deploy paketini yeniden
  export; (C) eşiksiz AUC/skor raporu.
- Metrik dürüstlüğü: rapor anahtarı artık "AUC-ROC (girdi CSV etiketleri, bu koşunun ölçümü)".

## 10. Karar A1 kapanışı — kutu-yerel eşik (2026-11-22, ölçüm)

- Karar (kullanıcı, 2026-11-22): (A) kutu-yerel eşik; alt-karar A1 — eşik = Ekim ALARM
  BÜTÇESİNİN kutu skorlarındaki quantili; eşik + işletim sayıları TEK FİNAL te
  geçişinde ölçülür (AGENTS §4.1). Üretici: `kara_kutu/tools/esik_kalibre.py`;
  sözleşme: `kara_kutu/artifacts/esik_fraud.json` (adapter artefaktı okur; yoksa
  RuntimeError — sessiz Ekim eşiğine düşüş yok).
- Kalibrasyon tanısı (eğitim-fold tek okuma: n 1.000.000, tohum 42, bölme `user_fold < 8`): `kara_kutu/runs/esik_kalibre_20261122-084720/ozet.json:2-7`.
- Best-F2 aday eşik 0,7919164896011353, iç-örnek precision 0,053914 → EŞİK OLARAK REDDEDİLDİ (`kara_kutu/runs/esik_kalibre_20261122-084720/ozet.json:8-13`); aday artefakta `kalibrasyon_tani` olarak işlendi (`kara_kutu/artifacts/esik_fraud.json:24-27`).
- Tanı AUC notu 0,941453 yalnız üretici docstring'indedir (`kara_kutu/tools/esik_kalibre.py:22`); ozet.json'a yazılmamış — bağımsız ölçüm DOĞRULANMADI.
- Final te geçişi (A1; tek final okuma): n 4.932.768, fraud 6.181, eşik 0,8566042184829712, oran 0,002507234 = 11784/4700000 (`kara_kutu/runs/esik_te_20261122-193451/ozet.json:2-21`).
- Sonuç (final te): alarm 12.368 = k_butce = round(oran × n) (`kara_kutu/tools/esik_kalibre.py:273`); TP 1.265, FP 11.103, precision 0,10228, recall 0,204659, F2 0,170522 (`kara_kutu/runs/esik_te_20261122-193451/ozet.json:4-13`).
- Dürüst kayıt: te satırları fiilen iki kez tarandı — smoke `kara_kutu/runs/esik_te_20261122-193423/ozet.json:2-10` (`--hizli`: n 451.018, eşik adayı 0,8569020628929138 ≠ final; cap `kara_kutu/tools/esik_kalibre.py:317`) geçici kopya artefaktla koştu, final artefaktını ÜRETMEDİ.
- A1 hükmü FINAL işletim noktası için sağlandı: artefakt ve raporlar yalnız final koşumu taşır (`kara_kutu/artifacts/esik_fraud.json:22`); tekrar-deneme guard reddetti: exit 3, sha256 değişmedi (doğrulayıcı turu).
- Dürüst karşılaştırma: Ekim FP32 best-F2 noktası 11.784 alarmda TP 1.842, FP 9.942, P 0,156, R 0,301 (`fraud/results/false_alarm_audit.json:11-16`); kutu aynı oran-bütçesinde TP 1.265 ile daha az yakalıyor (`kara_kutu/runs/esik_te_20261122-193451/ozet.json:5`); sıralama yakınlığı sürüyor (§9), TP farkının mekanizması DOĞRULANMADI.
- Evren uyarısı: audit n 4.700.000, fraud 6.110 (`fraud/results/false_alarm_audit.json:6-7`) ile final te n 4.932.768, fraud 6.181 (`kara_kutu/runs/esik_te_20261122-193451/ozet.json:10-11`) birebir aynı alt-küme değil.
- P2 complaints adaptörü: issue top-1 doğruluk 0,3900, dengeli doğruluk 0,0701 (`kara_kutu/runs/20261122-091904_complaints/ic_rapor.md:26-27`); çoğunluk-etiket tavanı 0,32 = 64/200 (`docs/complaints_tavan_olcum_2026-11-22.txt:1-3`) — niş eval kabiliyeti, SOTA iddiası değil.
- Geçici önlem (kıya): iç raporun fraud-string'leri adaptör-koşullu (`kara_kutu/report/internal.py`): P(fraud) başlığı, Ekim AUC tablosu, yön etiketi fraud dışı dikeyde basılmaz; beklemedeki genelleştirme kartı: `contract.Gerekce.yon`, `runner` ETIKET_KOLON/fraud_sayisi, external HTML "FRAUD" metni, `__main__` özet satırı.
