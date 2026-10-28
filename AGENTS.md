# AGENTS.md — Quann-3D İş Kuralları

Bu repo bir araştırma projesi: CloudLM (nokta-bulut LM) + QUANN-Cloud
(backprop'suz eğitim) + makale/patent. Context sağlığı bu dosyadaki
kurallarla korunur — her oturum bu dosya + `RESEARCH.md` ile başlar.

## 1. Oturum Protokolü

1. **Başlangıç:** `RESEARCH.md`'yi oku (≤150 satır). README.md gerekirse.
   Başka hiçbir dosyayı "keşif için" okuma — kapsamı agent'a ver.
2. **Canlı run var mı?** Ağır iş planlıyorsan önce
   `ps aux | grep -E "train_wiki|train_baseline" | grep -v grep`.
   MPS tek: aynı anda en fazla 1 ağır eğitim.
3. **Ağır okuma ana oturuma gelmez.** `logs/`, `checkpoints/`, model kodu,
   `figures/` → ilgili subagent'a devret; ana oturum sadece distile
   dönüşü (≤30 satır) görür.

## 2. Agent yönlendirme

| İhtiyaç | Agent | Not |
|---|---|---|
| Eğitim başlat/resume, kuyruk, metrik, RESEARCH.md güncelle | `experiment` | tek RESEARCH.md yazıcısı (analiz hariç) |
| Geometri/boyut analizi, bölge haritaları, hassasiyet | `geometry` | edit yok — bulgu döndürür |
| Figür üretimi (bulut evrimi, bölge haritası, eğriler → `figures/`) | `viz` | sadece `figures/**` + `training/**` yazar |
| Makale/patent bölümleri | `paper` | sadece `docs/**` + RESEARCH.md yazar |
| Kod değişikliği (model, pipeline) | `build` (global) | |
| Bağımsız doğrulama / denetim | `tester` / `reviewer` (global) | |
| Bayat içerik → `_archive/` | `archiver` (global) | gerekçeli tek satır yazar |

## 3. Dondurulmuş Kurallar (dokunma)

1. **`state_dict` ana adları sabit** (`cloudlm/model.py` başındaki NOTE):
   checkpoint uyumluluğu için anahtar isimleri değiştirilmez — yalnızca
   yeni anahtar EKLENEbilir.
2. **Checkpoint dizin adlandırma:** `checkpoints/<model>_<corpus>[_<config>]/`
   (ör. `wiki_checkpoints_200k/`, `rev1_20k/d64/`). İçinde: `*_best.pt` +
   `*_epochN.pt` + `<run>_bpe.json`.
3. **`logs/` append-only:** log dosyalarına silme/rewrite yapılmaz; yeni
   run → yeni log dosyası (`logs/run_<ad>.log`).
4. **Kuyruk scriptleri** `training/shell/run_<ad>.sh`: repo köküne
   `cd`, `python3 -u`, log `tee -a`, adım başlıkları `=== [date] N/M: ... ===`.
5. **`_archive/`:** bayat/elenen içerik yalnızca `archiver` agent'ı taşır,
   tek satır gerekçeyle.
6. **`quann_cloud/src/` motoru** (qcloud_quann vb.) değişikliği test gerektirir:
   `python3 -m pytest quann_cloud/tests/ -v` koşmadan commit etme.
7. **Doküman satır bütçesi:** repo'daki tüm `*.md` (dışı: `_archive/`) ≤200
   satır. İstisna (formal teslimat, yapısal uzunluk): `docs/patent/**`,
   `docs/paper_draft.md`. `RESEARCH.md` ≤150 (§5). Checker:
   `scripts/check_docs_lines.sh` — her commit öncesi koş (committer/ana
   oturum).

## 4. Veri

- `data/turkish_sentences.txt` (200K, ana corpus), `turkish_wiki_1500k.txt`,
  `bilkent_texts.csv` (word-tokenizer run'ları).
- BPE: `checkpoints/wiki_checkpoints_200k/wiki_bpe.json` ortak referans —
  varyant run'lar aynı BPE'yi `cp` ile kendi dizinine kopyalar
  (bkz. `run_rev1_20k.sh`).

## 5. RESEARCH.md Güncelleme Protokolü

Ne zaman: her biten deney, her yeni hipotez, her canlı run değişimi.
Kim: `experiment` agent'ı (metrikli) veya ana oturum (karar/hipotez).
Nasıl:
- "Canlı Runlar" tablosu = sadece şu an çalışanlar (biten silinir).
- "Son Deneyler" tablosu ≤4 satır; en eski düşer, detay log'da kalır.
- "Sıradaki Aksiyonlar" her seferde yeniden yazılır.
- Dosya 150 satırı geçemez; "Son güncelleme" damgası vurulur.
- Teşhis/revizyon adayları `quann_cloud/REVISION.md`'de yaşar — RESEARCH.md
  onlara pointer verir, kopyalamaz.

## 6. Sonuç Raporlama Şablonu (kullanıcı tercihi, 2026-10-19)

Bir deneysel iş bitince / sonuç-durum gösterilince ana oturum şu 4 bölümlü
biçimde yazar:

1. **Ne yaptık** — 1-2 cümle, jargonsuz, insan dili (neyi, hangi yöntemle).
2. **Sonuçlar** — basit tablo: `Ne | Değer | Basit anlam` (sayılar + günlük dil).
3. **Teknik yanıtlar** — hassas değerler: mimari, metrikler, basınç/baselines,
   audit, deploy, tavan/kısıt. Jargon burada serbest.
4. **HÜKMÜ** — en sonda, teknik bölümün altında: *iyi yolda mıyız? sonuç iyi mi?
   neden iyi / neden kötü?* Dürüst iki-yanlı; iddianın çerçevesini (niş vs SOTA)
   açıkça belirt.

Kısa bilgi sorularında kullanılmaz; sonuç/durum raporlarında zorunludur.

## 7. Kalıcı Hafıza Kuralları (2026-10-27)

1. **Gerçekçilik ilkesi:** Beni memnun etmek için hiçbir şeyi uydurma. Asla
   kullanıcıyı memnun etmek için veri/kod uydurma. Tüm veriler gerçek,
   tüm kodlar hatasız olmalı.

2. **Çift-üçlü kontrol:** Kod yazımlarında 2-3 kez tekrarlı kontrol sonrası
   yazım tamamlanır.

3. **Agent kullanımı:** İhtiyaç halinde subagent'lar serbestçe kullanılabilir.

4. **Dürüst değerlendirme:**
   - Sonuç iyi ise → açıkça "iyi" de, nedenini belirt
   - Sonuç kötü ise → açıkça "kötü" de, nerede kötü olduğunu göster
   - Her zaman açık ve şeffaf ol

5. **Kod kalitesi:** Parametrik ve SOTA kod yazım kurallarına uygun kod yaz.
   İleride sorun olabilecek kodlar yazma. Structured ve refactor edilmeye
   ihtiyaç duymayacak şekilde kodlar yaz.

## 8. Hata Yönetimi ve Kurtarma

1. **Deney başarısız:** Log'da neden + 3 dakika içinde çözüm adayı üretilir.
2. **MPS OOM:** `batch_size` yarılanır, checkpoint'tan resume edilir.
3. **BPE uyumsuzluğu:** Ana checkpoint'tan BPE'yi `cp` ile al, devam et.
4. **Veri okuma hatası:** Raw log'a düş, ana oturumu bloklama.

## 9. Kod Kalitesi Kontrolü

Commit öncesi kontrol listesi:
- [ ] Type hints eksiksiz
- [ ] Docstring (varsa)
- [ ] Edge case test edildi
- [ ] Logging eklendi
- [ ] Memory leak yok (`torch.no_grad`, context manager)
- [ ] `quann_cloud/tests/` geçiyor (motor değişikliği varsa)

## 10. Güvenlik ve Secrets

1. **API key, token, credential:** `.env` dosyasında, asla repo'ya commit edilmez.
2. **`.env`:** `.gitignore`'da mutlaka olmalı.
3. **Public paylaşım:** Veri/sonuç paylaşımı → anonimleştir, sensitive bilgi yok.
4. **Dependency:** `pip freeze` → `requirements.txt`, bilinen CVE'leri kontrol et.

## 11. Deney Takibi Standardı

1. **Log başlığı:** `[YYYY-MM-DD HH:MM] <model>_<corpus>_<config>`
2. **Metrik kaydı:** `logs/metrics_<run>.json` (loss, accuracy, time, epoch)
3. **Checkpoint:** Her epoch → `checkpoints/<run>/epochN.pt`, en iyisi → `*_best.pt`
4. **Resume:** `checkpoint_epochN.pt` + `metrics_<run>.json` ile devam

## 12. Dokümantasyon Bakımı

1. **README.md:** Proje özeti, kurulum, kullanım (≤50 satır).
2. **Yeni feature:** İlgili section'a 2-3 satır ekle.
3. **_archive/:** Ayda 1 kez gözden geçir, gereksizi sil.
4. **Çapraz referans:** Değişiklik → ilgili doc güncelle (README, ARCHITECTURE).
