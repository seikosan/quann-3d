# Quann-3D: Cümle Bulutu Dil Modeli

**Cloud geometry**, dil, işlem, sensör ve hesap verisini evrensel bir geometrik
temsil olarak ele alır; ticari değeri denetlenebilir tahmin, edge boyutu ve
retrain'siz uyumdur — ham doğrulukta SOTA iddiası yoktur.

- **AML:** Tipoloji haritalarını uyum ekipleriyle doğrula — [AML projesi](aml/README.md).
- **İşlem fraud / edge:** Küçük model ve operasyonel dengeyi otorizasyon akışında ölç — [fraud projesi](fraud/README.md).
- **Endüstriyel bakım:** Sensör bulutlarındaki değişimi mühendislerle doğrula — [bakım projesi](maintenance/README.md).

Ürün konumlandırması, pilotlar ve açık iddia sınırları: [`docs/positioning.md`](docs/positioning.md).

Transformer'ın "sıralı dikkat + tek vektör" paradigması yerine:
- Her kelime → ND uzayda bir **nokta**
- Cümle → **nokta bulutu**
- Anlam → bulut **geometrisi** (mesafe, açı, küme formu)
- Tahmin → bulut konfigürasyonundan bir sonraki kelime

İkinci katman: **QUANN-Cloud** — aynı mimariyi **backprop'suz** (aralık-daraltma + ödül/ceza) eğiten 0th-order yöntem.

## Depo Yapısı

```
quann-3d/
├── cloudlm/                    # ORTAK MİMARİ — tek kaynak (single source of truth)
│   ├── model.py                #   WordToND + CloudEncoderND + CloudLM (parametreli)
│   └── tokenizer.py            #   Tok (kelime) + BPETokenizer
│                               #   NOTE: state_dict ana adları checkpoint uyumluluğu
│                               #   için sabittir — değiştirilmez!
├── purecloudlm/                # PureCloudLM paketi (geometri-odaklı varyant)
│   ├── model.py                #   state_dict anahtarları donmuş (contextcloud bağımlı)
│   └── train.py                #   canonical eğitmen (21 tek-atış script → _archive)
├── contextcloud/               # ContextCloudLM — pencere/bulut + memory/global görevleri
│   ├── model.py                #   ContextCloudLM (+ generate)
│   ├── dataset.py              #   WordVocab/ChunkedVocab dataset + collate (ortak)
│   ├── task_utils.py           #   80/10/10 seed=42 split + sızıntısız hedef kaydırma
│   └── train_{stable,memory_context,global_context}.py   # R5 aktif akış
├── training/                   # DENEMELER / GELİŞTİRMELER — eğitim + karşılaştırma pipeline'ları
│   ├── train_wiki.py           #   BPE + CloudLM wiki eğitimi (ana pipeline, --resume)
│   ├── train_baseline.py       #   Causal/Bidir baseline (param-eşleşen) + karşılaştırma (--resume)
│   ├── train_bpe.py            #   BPE pipeline (tokenizers kütüphanesi)
│   ├── compare_llm.py          #   Kapsamlı LLM karşılaştırma + edge benchmark
│   ├── compare_16_32.py        #   16D vs 32D geometri (bölge haritaları, KMeans)
│   ├── test_model.py           #   Checkpoint testi + etkileşimli üretim aracı
│   ├── download_wiki.py        #   Wikipedia corpus indirici
│   └── shell/                  #   run_r4_*/run_r5_cc.sh (güncel kuyruklar)
├── data/                       # Corpus'lar
│   ├── turkish_sentences.txt   #   200K cümle (ana eğitim corpusu)
│   ├── turkish_wiki_1500k.txt  #   1.5M cümle (ölçek testi)
│   └── turkish_wiki_sentences.txt
├── checkpoints/                # Tüm checkpoint'ler
│   ├── cloudlm/                #   (eski "checkpoints/") word-tokenizer run
│   ├── wiki_checkpoints/       #   16D wiki run
│   ├── wiki_checkpoints_200k/  #   200K run (shared held-out PPL 6.45)
│   ├── wiki_checkpoints_32d/   #   32D run
│   ├── wiki_checkpoints_wiki1500k/
│   ├── baseline_200k/          #   Causal baseline 200K
│   ├── baseline_wiki/          #   Causal baseline wiki
│   ├── bpe_checkpoints/  cloudlm_1500k/  demo_training/
├── figures/                    # Tüm çıktı şekilleri (PNG)
│   └── comparison/             #   16D/32D karşılaştırma görselleri
├── docs/
│   ├── paper_draft.md          #   Makale taslağı (v0.1)
│   ├── quann_evolution_analysis.md
│   ├── figures/                #   Makale/patent SVG şekilleri (sekil_1..8)
│   └── patent/                 #   Patent taslakları (3 sürüm)
├── logs/                       # Tüm eğitim/çalıştırma logları
├── quann_cloud/                # QUANN-Cloud paketi (backprop'suz eğitim) — src/, tests/,
│                               #   experiments/, data/, logs/, checkpoints/, docs/
│                               #   (QCloudNetwork, cloudlm/'in WordToND/CloudEncoderND
│                               #    alt-modüllerini yeniden kullanır)
└── _archive/                   # Bayat/elenmiş içerik (arch/, refactor_2026_11/, src_legacy/)
```

## Mimari

```
Token → [ContextEncoder (Transformer)] → [WordToND (MLP→D)]
                                                     ↓
                                     [CloudEncoder (PointNet)]
                                        ├─ nokta MLP (paylaşılan)
                                        ├─ çift MLP (mesafe + açı)
│   └─ max/mean pool → cloud_state
                                                     ↓
                                     [PredictionHead]
                                        ├─ word_logits (vocab)
                                        └─ next_point (D)
```

## Çalıştırma

```bash
# Canonical PureCloudLM eğitimi (stable reçetesi)
python3 -u purecloudlm/train.py --epochs 15

# ContextCloudLM R5 görevleri (stable/memory/global kuyruğu)
bash training/shell/run_r5_cc.sh

# Ana eğitim pipeline (BPE + CloudLM)
python3 training/train_wiki.py --step all --corpus data/turkish_sentences.txt

# Baseline (causal/bidir) + karşılaştırma
python3 training/train_baseline.py --step all --arch causal

# Kapsamlı LLM karşılaştırma + edge benchmark
python3 training/compare_llm.py --step all \
    --cloud-ckpt checkpoints/wiki_checkpoints/wiki_cloud_best.pt \
    --base-ckpt checkpoints/baseline_200k/baseline_best.pt --arch causal

# 16D vs 32D geometri analizi
python3 training/compare_16_32.py

# Eğitilmiş checkpoint'i test et / etkileşimli üretim
python3 training/test_model.py --checkpoint checkpoints/cloudlm/cloud_lm_final.pt

# QUANN-Cloud (backprop'suz) — paket içi testler
python3 -m pytest quann_cloud/tests/ -v
python3 quann_cloud/experiments/real_data_test.py --quick
```

## 3B Çalışma Görünümü (web)

Etkileşimli 3B arayüz (Mimari şeması · gerçek CloudLM bulutu 32D→3B · QUANN
0th-order araması animasyonu) `_archive/src_legacy/visualizer/` altındadır —
koddan çıkarım yapılıyor ama aktif akışta değil; yeniden canlandırmak isteyen
`export_data.py` + `index.html`yi oradan alabilir.

Python scriptleri `ROOT`'u kendi konumundan çözer — herhangi bir dizinden çalıştırılabilir.
`.sh` kuyruk scriptleri de repo köküne `cd` eder: `bash training/shell/run_r5_cc.sh`.

## Bağımlılıklar

```
torch>=2.0
numpy
matplotlib
tokenizers
datasets        # yalnız download_wiki.py için
```

## Temel Fark (Transformer vs CloudLM)

| | Transformer | CloudLM |
|---|---|---|
| Temsil | Tek dev vektör (768D) | Nokta bulutu (N×D) |
| Anlam | Vektörün konumu | Noktaların birlikte duruşu |
| Sıralama | Kritik (positional encoding) | Yok — küme olarak |
| Dikkat | Q·K·V matris | Mesafe + açı (geometri) |
| Boyut | 768D = 768 anlam ekseni | D = D anlam ekseni |

## Bulgular

- 3D: Model 1 baskın ilişkiyi kodlar (yer-fiil veya özne-fiil)
- 7D: Hem özne-fiil hem yer-fiil ayrışır
- 16D: Sözdizimsel roller uzayda bölgeleşir (edatlar merkez, belirsizlik batı)
- Gerçek corpus'ta (3K vocab): PPL 2794→62 (44x iyileşme)
- Legacy 200K karşılaştırması: CloudLM **6.45** vs causal baseline **2.92**; bu
  değer strict held-out değildir ve SOTA iddiası olarak sunulmaz. Revizyonlar:
  `quann_cloud/REVISION.md`.
- QUANN-Cloud: XOR'u backprop'suz %100 çözer; gerçek LM'de random'a yakın (0th-order sınırı) —
  dürüst analiz: `quann_cloud/FINDINGS.md`
- Edge: 21M modelin INT8 paketi 84→21MB; fraud tam doğal testinde FP32 ROC-AUC
  .9451, INT8-channel .9367 (ölçülen fark −.0084) — `RESEARCH.md`, deploy artefaktı.

## Corpus

Bilkent Turkish Writings dataset (word-tokenizer runları için):
```bash
curl -L -o data/bilkent_texts.csv \
  "https://raw.githubusercontent.com/selimfirat/bilkent-turkish-writings-dataset/master/versions/v2/texts.csv"
```

Wikipedia (alternatif):
```bash
python3 training/download_wiki.py
```
