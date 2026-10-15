# QUANN-Cloud: Backprop'suz Nokta Bulut Modeli

> **Sekans yok + Geometri var + Backprop yok + Arama-Tabanlı Öğrenme**

## 🎯 Hedef

**Dünyada bir ilk:** Backpropagation olmadan, nokta bulut tabanlı dil modeli.

- **Mimari:** CloudLM (nokta bulut) — sekans yok, geometri var
- **Eğitim:** QUANN'ın aralık daraltma + sigma adaptasyonu — backprop yok
- **Dust (dış referans):** Backprop'suz eğitim fikrinden ilham alır, sizin değil

## 📐 Temel Felsefe

```
Token'lar → Nokta bulutu (geometri) → Ödül/ceza → Aralık daraltma → Yakınsama
```

Her token'ın buluttaki noktası bir **QCloudWeight**'tir:
- `low/high`: aralık sınırları (başlangıç: [-1, 1])
- `value`: anlık konum (rastgele başlatılır)
- `reward`: hata düştü → aynı yönde devam, sigma küçülür
- `censure`: hata arttı → **yön ters çevrilir** + sigma büyür

**Öğrenme = Yakınsama:** Aralık bir eşik altına düştüğünde token "öğrenmiş" sayılır.

## 🏗️ Proje Yapısı

```
quann_cloud/
├── README.md                  # Bu dosya
├── DESIGN.md                  # Detaylı mimari tasarım
├── EXPERIMENTS.md             # Deney planı
├── AGENTS.md                  # Agent sistemi (varsa)
│
├── src/
│   ├── __init__.py
│   ├── qcloud_weight.py       # QCloudWeight (her token'ın noktası)
│   ├── qcloud_network.py      # QCloudNetwork (CloudLM + QUANN eğitim)
│   ├── qcloud_trainer.py      # Backprop'suz eğitim pipeline'ı
│   └── utils.py               # Yardımcı fonksiyonlar
│
├── experiments/
│   ├── xor_gate.py            # XOR doğrulama (v0 benzeri)
│   ├── tiny_corpus.py         # Küçük corpus (20K cümle)
│   └── full_corpus.py         # Büyük corpus (200K cümle)
│
├── tests/
│   ├── test_weight.py
│   └── test_network.py
│
├── data/                      # Corpus dosyaları
│   └── turkish_sentences.txt  # Mevcut corpus
│
└── logs/                      # Eğitim logları
```

## 🚀 Kurulum

```bash
# Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Bağımlılıklar
pip install torch numpy matplotlib tokenizers

# Testler
python -m pytest tests/ -v
```

## 📋 Temel Kullanım

```python
from src.qcloud_weight import QCloudWeight
from src.qcloud_network import QCloudNetwork

# Token'ı bulut noktasına göm
token_weight = QCloudWeight(dimension=16)  # D=16

# Backprop'suz eğitim
trainer = QCloudNetwork(model, tokenizer, corpus, epochs=10)
trainer.train()  # Backprop YOK, reward/ceza ile öğrenir
```

## 📊 Literatür Karşılaştırması

| Yaklaşım | Backprop | Gradient | Ödül/CE | Aralık Daralt | Geometri |
|----------|----------|----------|---------|---------------|----------|
| **QUANN-Cloud** | ❌ | ❌ | ✅ | ✅ | ✅ |
| QUANN V0 (QNetwork) | ❌ | ❌ | ✅ | ✅ | ❌ |
| QUANN V0 (Bisect) | ❌ | ❌ | ✅ | ❌ (deterministik) | ❌ |
| QUANN-NLP | ❌ | ❌ | ✅ | ❌ | ✅ (3B küme) |
| CloudLM (standart) | ✅ | ✅ | ❌ | ❌ | ✅ (nokta bulut) |
| Dust (dış referans) | ❌ | ❌ | ✅ | ❌ | ❌ |
| Standart BP | ✅ | ✅ | ❌ | ❌ | ❌ |

## 📜 Lisans

Bu proje araştırma amaçlıdır.
