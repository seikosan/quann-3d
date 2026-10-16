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
